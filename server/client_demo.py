#!/usr/bin/env python3
"""
server/client_demo.py -- a protocol client that plays the role of the app: it streams SYNTHETIC EEG, camera and click data to the service and
prints what comes back. Runs in-process (no network) by default, or against a live server with --url.

  python server/client_demo.py                              # in-process, accelerated (a few seconds)
  python server/client_demo.py --url ws://localhost:8000/ws # a running `uvicorn server.app:app`  (needs: pip install websockets)

Flow: resting calibration (2 min) -> passive monitoring (a sensor failure and a drift in attention) -> one training round added to the baseline.
Use it as the reference for your real client (the JavaScript equivalent is in docs/REALTIME.md).
"""
import base64
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from focus_core import simulate as sim  # noqa: E402


class Conn:
    """Tiny synchronous wrapper: send() a message, get the list of replies (reads until the server's ack)."""

    def __init__(self, url=None):
        self.url = url
        if url:
            from websockets.sync.client import connect
            self._ws_cm = connect(url, max_size=None)
            self.ws = self._ws_cm.__enter__()
            self._ctx = None
        else:
            os.environ.setdefault("FOCUS_BASELINE_DIR", tempfile.mkdtemp())
            import server.app as srv
            srv.BASELINE_DIR = os.environ["FOCUS_BASELINE_DIR"]
            from fastapi.testclient import TestClient
            self._ctx = TestClient(srv.app).websocket_connect("/ws")
            self.ws = self._ctx.__enter__()

    def _send(self, m):
        self.ws.send_text(json.dumps(m)) if not self.url else self.ws.send(json.dumps(m))

    def _recv(self):
        return json.loads(self.ws.receive_text() if not self.url else self.ws.recv())

    def send(self, m):
        self._send(dict(m, ack=True))
        out = []
        while True:
            r = self._recv()
            if r["type"] == "ack":
                return out
            out.append(r)

    def close(self):
        if self._ctx:
            self._ctx.__exit__(None, None, None)
        else:
            self._ws_cm.__exit__(None, None, None)


def eeg_msg(t_start, chunk):
    return dict(type="eeg", t_start=t_start, samples=chunk.shape[1], data=base64.b64encode(chunk.astype("<f4").tobytes()).decode())


def stream(conn, x, fs, eye, events=(), t0=0.0, verbose=True, tag=""):
    replies = []
    for item in sim.iter_stream((x, fs, None), eye, events, t0=t0):
        if item[0] == "eeg":
            r = conn.send(eeg_msg(item[1], item[2]))
        elif item[0] == "eye":
            r = conn.send(dict(type="eye", rows=item[1]))
        else:
            r = conn.send(dict(type="event", kind=item[1], t=item[2]))
        for m in r:
            replies.append(m)
            if not verbose:
                continue
            if m["type"] == "score":
                print(f"  {tag}t={m['t']:6.1f}s score {m['score']:5.1f}  {m['components']}" + (f"  PAUSED: {m['paused']['eeg']}" if m["paused"] else ""))
            elif m["type"] == "quality" and m["prompts"]:
                for p in m["prompts"]:
                    print(f"  {tag}t={m['t']:6.1f}s PROMPT  {p['text'][:95]}")
            elif m["type"] == "quality" and m["resolved"]:
                print(f"  {tag}t={m['t']:6.1f}s RESOLVED {m['resolved'][0]}")
            elif m["type"] in ("suggest_training", "warning", "error", "score_unavailable"):
                print(f"  {tag}t={m.get('t', 0):6.1f}s {m['type'].upper()}: {m.get('text') or m.get('reason')}")
    return replies


def main(url=None):
    names, fs = ["Fp1", "Fp2", "F3", "F4", "O1", "O2"], 250
    conn = Conn(url)
    print("connect + hello"); r = conn.send(dict(type="hello", user_id="demo_user", fs=fs, channels=names, acks=True,
                                                 profile=dict(age=29, sex="F", hours_since={"caffeine": 2})))
    print("  ", r[0]["type"], "has_baseline =", r[0]["has_baseline"])

    print("\n1) resting calibration (2 min, eyes open)")
    conn.send(dict(type="rest", action="start", t=0.0))
    x, _, _ = sim.synth_eeg("rest", seed=1, dur=120)
    rep = stream(conn, x, fs, sim.synth_eye("rest", seed=1, dur=120, off=0.0), verbose=False)
    r = conn.send(dict(type="rest", action="finish"))
    print("  ", [m["type"] for m in r], *[n for m in r for n in m.get("notes", [])], sep="\n   ")

    print("\n2) passive monitoring: focused, then a sensor failure, then drifting attention")
    t0 = 130.0
    x = np.concatenate([sim.synth_eeg("focused", seed=11, dur=70)[0], sim.synth_eeg("wandering", seed=12, dur=50)[0]], axis=1)
    x[0, 40 * fs:70 * fs] = 0.05 * np.random.default_rng(0).normal(size=30 * fs)
    eye = pd.concat([sim.synth_eye("focused", seed=11, dur=70), sim.synth_eye("wandering", seed=12, dur=50)])
    eye["t"] = np.arange(len(eye)) / 30.0
    stream(conn, x, fs, eye, t0=t0, tag="")

    print("\n3) training round (orbiting ball + clicks), added to the baseline")
    s = sim.simulate_session("focused", seed=21, duration=120, jitter=0.2)
    t1 = 270.0
    conn.send(dict(type="training", action="start", t=t1))
    events = [("change", c) for c in s["changes"]] + [("click", c) for c in s["clicks"]]
    stream(conn, s["eeg"][0], fs, s["eye"], events, t0=t1, verbose=False)
    r = conn.send(dict(type="training", action="finish", add_to_baseline=True))
    summ = r[0]
    res = summ["result"]
    print(f"   training summary: score {res['simple_score']}  components {res['components']}  weights {res['weights_used']}")
    print("   baseline updated:", summ["baseline_updated"], "|", *summ.get("baseline_summary", [])[-1:])
    conn.close()


if __name__ == "__main__":
    url = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else None
    main(url)
