#!/usr/bin/env python3
"""
bridge/openbci_bridge.py -- reads an OpenBCI Cyton / Cyton+Daisy (through BrainFlow) and streams EEG to the focus engine server.

WHY A BRIDGE: a web page cannot talk to the OpenBCI USB dongle through BrainFlow, so a small program on the computer that holds the dongle does
it and joins the same session as the website (role "device"). The website (role "app") sends camera + click data; the engine scores them together.
If the server runs on the SAME laptop, raw EEG never leaves the laptop.

  OpenBCI Cyton(+Daisy) --RF--> USB dongle --> this bridge --WebSocket--> server/app.py <--WebSocket-- website (camera, clicks, UI)

USAGE
  pip install -r requirements-bridge.txt                      # brainflow, websockets (and setuptools<81 if BrainFlow asks for pkg_resources)
  python server/... (start the server first: see server/app.py)
  python bridge/openbci_bridge.py --board cyton_daisy --serial-port COM3 --user-id u1 --map "1=Fp1,2=Fp2,3=AF7,4=AF8"
  python bridge/openbci_bridge.py --board synthetic --user-id u1 --duration 30        # no hardware: BrainFlow's synthetic board

FACTS (BrainFlow / OpenBCI documentation)
  * BoardIds.CYTON_DAISY_BOARD (= 2), serial_port = the dongle's port (Windows "COM3"; macOS "/dev/cu.usbserial-..."; Linux "/dev/ttyUSB0").
  * Cyton + Daisy = 16 EEG channels, sampled at 125 Hz per channel (OpenBCI shop page); a plain Cyton = 8 channels at 250 Hz. fs is read from BrainFlow.
  * BrainFlow returns EEG in microvolts and timestamps in epoch seconds; get_board_data() returns everything new and removes it from its buffer.
  * Channel numbers in --map are the numbers printed on the board: Cyton 1-8, Daisy 9-16.
SAFETY (OpenBCI guidance; please confirm in their current docs): power the Cyton from its BATTERY ONLY while electrodes are on a person, and use the
wireless USB dongle; never connect the board itself to a computer by a USB cable while it is attached to someone.

CLOCKS: the bridge measures its offset to the SERVER clock with an NTP-style handshake (median of the lowest-round-trip samples) and stamps the first EEG
sample of each chunk as bridge_time + offset. The website does the same handshake, so EEG, camera frames and clicks share one timeline even on
different machines. EEG time inside the engine advances by SAMPLE COUNT; t_start is sent only on the first chunk, after dropped packets, or if the
host clock and the sample clock drift apart by more than 0.2 s (a few ms per minute of drift is normal and harmless for 30-s windows and 3-min rounds).

ELECTRODES (our plan): 4 frontal dry electrodes on Cyton channels 1-4 (Fp1, Fp2, AF7, AF8), ear-clip on the board's reference (SRB) and BIAS pins.
The software ignores unmapped channels. The 125 Hz rate limits the usable EEG band to < 62 Hz (we use 1-45 Hz) and puts 60 Hz mains hum just below
Nyquist; the electrode-quality monitor still detects it.
"""
import argparse
import base64
import json
import os
import statistics
import sys
import time

import numpy as np

BOARDS = {"cyton_daisy": "CYTON_DAISY_BOARD", "cyton": "CYTON_BOARD", "synthetic": "SYNTHETIC_BOARD"}
DEFAULT_MAP = "1=Fp1,2=Fp2,3=AF7,4=AF8"


def parse_map(text):
    """'1=Fp1,2=Fp2' -> [(1, 'Fp1'), (2, 'Fp2')] (1-based channel numbers as printed on the board)."""
    out = []
    for part in text.split(","):
        k, v = part.split("=")
        out.append((int(k), v.strip()))
    if len({n for _, n in out}) != len(out) or len({k for k, _ in out}) != len(out):
        raise ValueError("--map has a duplicate channel number or name")
    return out


def encode_eeg(chunk):
    """(C, n) microvolts -> base64 float32 little-endian, channel-major (the server's wire format)."""
    return base64.b64encode(np.ascontiguousarray(chunk, dtype="<f4").tobytes()).decode()


class Bridge:
    def __init__(self, a):
        self.a = a
        self.mapping = parse_map(a.map)
        self.names = [n for _, n in self.mapping]
        self.board = self.ws = self._cm = None
        self.offset = 0.0
        self.anchor_t = self.anchor_n = None
        self.n_sent = self.dropped = self.chunks = 0
        self.last_pkg = None
        self.force_t = True
        self.last_server_msg = ""

    # ------------------------------------------------------------------------------------------------ board
    def open_board(self):
        try:
            from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        except ImportError as e:
            raise SystemExit(f"BrainFlow is not usable ({e}). Install it with: pip install brainflow \"setuptools<81\"")
        BoardShim.disable_board_logger()
        bid = BoardIds[BOARDS[self.a.board]].value
        p = BrainFlowInputParams()
        if self.a.serial_port:
            p.serial_port = self.a.serial_port
        self.board = BoardShim(bid, p)
        self.fs = float(BoardShim.get_sampling_rate(bid))
        rows = BoardShim.get_eeg_channels(bid)
        for k, _ in self.mapping:
            if not 1 <= k <= len(rows):
                raise SystemExit(f"--map channel {k} does not exist: this board has {len(rows)} EEG channels")
        self.rows = [rows[k - 1] for k, _ in self.mapping]
        self.ts_row, self.pkg_row = BoardShim.get_timestamp_channel(bid), BoardShim.get_package_num_channel(bid)
        if self.a.board != "synthetic":
            print("SAFETY: battery power only while electrodes are on a person; use the USB dongle, never a cable to the board.")
        self.board.prepare_session()
        self.board.start_stream()
        print(f"board {self.a.board}: {len(rows)} EEG channels at {self.fs:g} Hz; streaming {len(self.names)} of them as {self.names}")

    def close_board(self):
        if self.board is not None:
            try:
                self.board.stop_stream()
                self.board.release_session()
            except Exception as e:  # noqa: BLE001
                print("board shutdown:", e)

    # ------------------------------------------------------------------------------------------------ server
    def connect(self):
        from websockets.sync.client import connect
        self._cm = connect(self.a.url, max_size=None, open_timeout=10)
        self.ws = self._cm.__enter__()                       # context-manager form (the form current websockets versions expect)
        self.ws.send(json.dumps(dict(type="hello", role="device", user_id=self.a.user_id, session_id=self.a.session_id or self.a.user_id,
                                     fs=self.fs, channels=self.names, token=self.a.token or None)))
        while True:
            m = json.loads(self.ws.recv(timeout=10))
            if m["type"] == "ready":
                print(f"joined session {m['session_id']} (engine ready: {m['engine_ready']}, baseline: {m['has_baseline']})")
                break
            if m["type"] == "error":
                raise SystemExit(f"server refused the connection: {m['text']}")
        self.sync_clock()
        self.force_t = True

    def sync_clock(self, n=9):
        """NTP-style offset to the server clock; keeps the lowest-round-trip half of the samples."""
        samples = []
        for _ in range(n):
            t0 = time.time()
            self.ws.send(json.dumps(dict(type="clock", t=t0)))
            while True:
                m = json.loads(self.ws.recv(timeout=10))
                if m["type"] == "clock" and m["t"] == t0:
                    break
                self.handle(m)
            t1 = time.time()
            samples.append((t1 - t0, m["server_t"] - (t0 + t1) / 2))
        best = sorted(samples)[: max(3, n // 2)]
        self.offset = statistics.median(o for _, o in best)
        print(f"clock offset to server: {self.offset * 1000:+.1f} ms (best round trip {best[0][0] * 1000:.1f} ms)")

    def handle(self, m):
        t = m.get("type")
        if t == "quality":
            for p in m.get("prompts", []):
                print(f"  [{p['code']}] {p['text']}")
            for r in m.get("resolved", []):
                print(f"  {r}")
        elif t in ("warning", "error"):
            print(f"  {t}: {m.get('text')}")
        elif t == "score":
            self.last_server_msg = f"score {m['score']}"
        elif t in ("device_connected", "device_disconnected"):
            print(f"  server: {t}")

    def drain(self):
        while True:
            try:
                self.handle(json.loads(self.ws.recv(timeout=0)))
            except TimeoutError:
                return

    # ------------------------------------------------------------------------------------------------ main loop
    def step(self):
        data = self.board.get_board_data()
        if data.shape[1] == 0:
            return 0
        eeg = data[self.rows, :]
        ts = data[self.ts_row]
        host_start = float(ts[0]) + self.offset
        pkg = data[self.pkg_row]
        if self.a.board != "synthetic" and len(pkg):                      # Cyton packet counter wraps at 256
            seq = np.r_[self.last_pkg if self.last_pkg is not None else pkg[0] - 1, pkg]
            miss = (np.diff(seq) % 256) - 1
            lost = int(np.sum(miss[(miss > 0) & (miss < 128)]))
            self.dropped += lost
            if lost:
                self.force_t = True
            self.last_pkg = float(pkg[-1])
        clock_now = self.anchor_t + (self.n_sent - self.anchor_n) / self.fs if self.anchor_t is not None else None
        send_t = self.force_t or self.anchor_t is None or abs(host_start - clock_now) > 0.2
        msg = dict(type="eeg", samples=int(eeg.shape[1]), data=encode_eeg(eeg))
        if send_t:
            msg["t_start"] = host_start
            self.anchor_t, self.anchor_n, self.force_t = host_start, self.n_sent, False
        self.ws.send(json.dumps(msg))
        self.n_sent += eeg.shape[1]
        self.chunks += 1
        return eeg.shape[1]

    def run(self):
        self.open_board()
        backoff, t_end = 1.0, (time.time() + self.a.duration if self.a.duration else None)
        last_print = time.time()
        try:
            self.connect()
            while t_end is None or time.time() < t_end:
                time.sleep(self.a.chunk_ms / 1000.0)
                try:
                    self.step()
                    self.drain()
                    backoff = 1.0
                except Exception as e:  # noqa: BLE001 -- connection problems: keep the board streaming and retry
                    print(f"connection problem ({type(e).__name__}: {e}); retrying in {backoff:.0f} s")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 10.0)
                    try:
                        self.connect()
                    except Exception as e2:  # noqa: BLE001
                        print("reconnect failed:", e2)
                if time.time() - last_print >= 5:
                    last_print = time.time()
                    print(f"streaming: {self.n_sent} samples in {self.chunks} chunks, {self.dropped} dropped packets, offset {self.offset * 1000:+.1f} ms"
                          + (f", {self.last_server_msg}" if self.last_server_msg else ""))
        except KeyboardInterrupt:
            print("stopping")
        finally:
            self.close_board()
            if self.ws is not None:
                try:
                    self._cm.__exit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass
        print(f"done: {self.n_sent} samples sent, {self.dropped} dropped packets")
        return dict(samples=self.n_sent, dropped=self.dropped, offset=self.offset)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", choices=list(BOARDS), default="cyton_daisy")
    ap.add_argument("--serial-port", help="dongle port, e.g. COM3, /dev/ttyUSB0, /dev/cu.usbserial-XXXX (not needed for --board synthetic)")
    ap.add_argument("--map", default=DEFAULT_MAP, help=f"board channel number = 10-20 name, comma separated (default {DEFAULT_MAP})")
    ap.add_argument("--url", default=os.environ.get("FOCUS_WS_URL", "ws://127.0.0.1:8000/ws"))
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--session-id", help="defaults to the user id; the website must join the same session")
    ap.add_argument("--token", default=os.environ.get("FOCUS_API_TOKEN"))
    ap.add_argument("--chunk-ms", type=int, default=250)
    ap.add_argument("--duration", type=float, help="stop after this many seconds (default: run until Ctrl-C)")
    a = ap.parse_args(argv)
    Bridge(a).run()


if __name__ == "__main__":
    main()
