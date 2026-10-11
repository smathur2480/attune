"""Tests for the OpenBCI bridge, shared sessions on the server, 125 Hz operation (Cyton+Daisy) and AF7/AF8 channels."""
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bridge"))

from focus_core import baseline_ops, simulate as sim, signal_quality as sq  # noqa: E402
from realtime import FocusEngine  # noqa: E402

NAMES4 = ["Fp1", "Fp2", "AF7", "AF8"]


class TestBridgeHelpers(unittest.TestCase):
    def test_channel_map_and_wire_format(self):
        import openbci_bridge as ob
        self.assertEqual(ob.parse_map("1=Fp1, 2=Fp2,9=O1"), [(1, "Fp1"), (2, "Fp2"), (9, "O1")])
        for bad in ("1=Fp1,1=Fp2", "1=Fp1,2=Fp1", "x"):
            with self.assertRaises((ValueError,)):
                ob.parse_map(bad)
        x = np.random.default_rng(0).normal(0, 20, (4, 31))
        import server.app as srv
        y = srv._decode_eeg(dict(data=ob.encode_eeg(x), samples=31), 4)             # the bridge's encoding is what the server decodes
        np.testing.assert_allclose(y, x.astype("float32"), atol=1e-6)


class TestOpenBCIRates(unittest.TestCase):
    """Cyton+Daisy samples at 125 Hz per channel: everything downstream must still work."""

    def test_quality_monitor_at_125_hz(self):
        fs = 125
        flagged = n = 0
        for seed in range(4):
            x, _, _ = sq.make_eeg(NAMES4, fs, 60, seed=seed)
            for s in range(0, x.shape[1] - 4 * fs, 2 * fs):
                a = sq.assess(x[:, s:s + 4 * fs], fs, NAMES4)
                n += len(NAMES4); flagged += sum(bool(a["channels"][c]["issues"]) for c in NAMES4)
        self.assertLess(flagged / n, 0.01)                                           # no false alarms on clean 125 Hz data
        x, _, _ = sq.make_eeg(NAMES4, fs, 4, seed=3)
        t = np.arange(x.shape[1]) / fs
        x[2] += 30 * np.sin(2 * np.pi * 60 * t)                                      # 60 Hz hum just below Nyquist is still detected
        self.assertIn("line", sq.assess(x, fs, NAMES4)["channels"]["AF7"]["issues"])
        self.assertEqual(sq.friendly("AF7"), "left outer forehead (AF7)")

    def test_engine_scores_at_125_hz_with_af7_af8(self):
        fs = 125
        x0 = sim.synth_eeg("rest", fs=fs, seed=1, dur=120, names=NAMES4)
        b, notes = baseline_ops.add_rest_session(None, (x0[0], float(fs), NAMES4), sim.synth_eye("rest", seed=1, dur=90, off=0.0))
        self.assertEqual(sorted(b["eeg"]["front"]), sorted(NAMES4))                  # AF7 and AF8 are recognised as frontal sites
        scores = {}
        for st in ("focused", "wandering"):
            eng = FocusEngine(fs, NAMES4, baseline=b)
            xs = sim.synth_eeg(st, fs=fs, seed=11, dur=60, names=NAMES4)
            msgs = []
            for item in sim.iter_stream((xs[0], float(fs), NAMES4), sim.synth_eye(st, seed=11, dur=60)):
                msgs += eng.push_eeg(item[2], item[1]) if item[0] == "eeg" else eng.push_eye(item[1])
            sc = [m for m in msgs if m["type"] == "score"]
            self.assertTrue(sc and "eeg" in sc[-1]["components"])
            scores[st] = sc[-1]["score"]
        self.assertGreater(scores["focused"], scores["wandering"] + 10)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestSessions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("pip install -r requirements-server.txt")
        import server.app as srv
        cls.srv = srv

    def setUp(self):
        self.srv.BASELINE_DIR = tempfile.mkdtemp()
        self.srv.API_TOKEN = ""
        self.srv.SESSIONS.clear()

    @staticmethod
    def send(ws, m):
        ws.send_text(json.dumps(dict(m, ack=True)))
        out = []
        while True:
            r = json.loads(ws.receive_text())
            if r["type"] == "ack":
                return out
            out.append(r)

    def hello(self, ws, role, **kw):
        base = dict(type="hello", user_id="u1", role=role)
        if role == "device":
            base.update(fs=125, channels=NAMES4)
        return self.send(ws, {**base, **kw})

    @staticmethod
    def eeg_msg(x, t_start=None):
        m = dict(type="eeg", samples=x.shape[1], data=base64.b64encode(x.astype("<f4").tobytes()).decode())
        if t_start is not None:
            m["t_start"] = t_start
        return m

    def test_device_and_app_share_one_engine(self):
        from fastapi.testclient import TestClient
        x = sim.synth_eeg("rest", fs=125, seed=1, dur=8, names=NAMES4)[0]
        with TestClient(self.srv.app) as client, client.websocket_connect("/ws") as dev, client.websocket_connect("/ws") as app_:
            r = self.hello(dev, "device")
            self.assertEqual((r[0]["type"], r[0]["engine_ready"], r[0]["role"]), ("ready", True, "device"))
            r = self.hello(app_, "app")                                              # the website joins without knowing fs or channels
            self.assertTrue(r[0]["engine_ready"]); self.assertEqual(r[0]["config"], dict(fs=125.0, channels=NAMES4))
            for k in range(0, x.shape[1] - 31, 31):
                self.send(dev, self.eeg_msg(x[:, k:k + 31], t_start=1000.0 if k == 0 else None))
            pending = self.send(app_, dict(type="eye", rows=[dict(t=1000.0 + i / 30, ear=0.3, face_ok=1) for i in range(30)]))   # also drains queued broadcasts
            got = pending + self.send(app_, dict(type="status"))
            st = [m for m in got if m["type"] == "status"][0]
            self.assertAlmostEqual(st["eeg_buffer_s"], 7.9, delta=0.5)                # the DEVICE's EEG sits in the shared engine ...
            self.assertEqual(st["eye_frames"], 30)                                   # ... together with the APP's camera rows
            self.assertEqual(st["connections"], {"app": 1, "device": 1})
            self.assertTrue([m for m in got if m["type"] == "quality"])               # quality messages were broadcast to the app too

    def test_app_first_then_device(self):
        from fastapi.testclient import TestClient
        with TestClient(self.srv.app) as client, client.websocket_connect("/ws") as app_:
            r = self.hello(app_, "app")
            self.assertFalse(r[0]["engine_ready"])
            self.assertEqual(self.send(app_, dict(type="eye", rows=[dict(t=1.0, ear=0.3)]))[0]["type"], "waiting")
            with client.websocket_connect("/ws") as dev:
                self.hello(dev, "device")
                got = self.send(app_, dict(type="ping"))
                dc = [m for m in got if m["type"] == "device_connected"]
                self.assertEqual(dc[0]["channels"], NAMES4)                           # the app is told when the headset arrives
                after = self.send(app_, dict(type="eye", rows=[dict(t=2.0, ear=0.3)]))
                self.assertNotIn("waiting", [m["type"] for m in after])                 # the engine accepts the app's data now

    def test_roles_mismatches_clock_and_cleanup(self):
        from fastapi.testclient import TestClient
        with TestClient(self.srv.app) as client, client.websocket_connect("/ws") as dev, client.websocket_connect("/ws") as app_, \
                client.websocket_connect("/ws") as other:
            self.hello(dev, "device"); self.hello(app_, "app")
            self.assertEqual(self.send(dev, dict(type="training", action="start"))[0]["type"], "error")      # a headset may only stream
            self.assertEqual(self.send(other, dict(type="hello", user_id="u2", session_id="u1", role="app"))[0]["type"], "error")   # not another user's session
            self.assertEqual(self.send(other, dict(type="hello", user_id="u1", role="device", fs=250, channels=NAMES4))[0]["type"], "error")   # config mismatch
            self.assertEqual(self.send(other, dict(type="hello", user_id="u1", role="boss"))[0]["type"], "error")
            t0 = time.time()
            c = self.send(app_, dict(type="clock", t=t0))[0]
            self.assertEqual(c["type"], "clock"); self.assertEqual(c["t"], t0); self.assertLess(abs(c["server_t"] - time.time()), 2.0)
            self.assertEqual(self.send(app_, dict(type="profile", profile=dict(age=30)))[0:0], [])
        deadline = time.time() + 3
        while self.srv.SESSIONS and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.srv.SESSIONS, {})                                       # sessions are removed when everyone leaves

    def test_device_disconnect_is_announced(self):
        from fastapi.testclient import TestClient
        with TestClient(self.srv.app) as client, client.websocket_connect("/ws") as app_:
            self.hello(app_, "app")
            with client.websocket_connect("/ws") as dev:
                self.hello(dev, "device")
            seen, deadline = [], time.time() + 3
            while time.time() < deadline and "device_disconnected" not in [m["type"] for m in seen]:
                seen += self.send(app_, dict(type="ping"))
                time.sleep(0.05)
            self.assertIn("device_disconnected", [m["type"] for m in seen])


class TestBridgeLive(unittest.TestCase):
    """The real bridge process, BrainFlow's SYNTHETIC board, a real uvicorn server on a real socket, and the website side as a second client."""

    @classmethod
    def setUpClass(cls):
        try:
            import brainflow, uvicorn, websockets  # noqa: F401
            import pkg_resources  # noqa: F401  (BrainFlow imports it)
        except ImportError:
            raise unittest.SkipTest("needs brainflow, uvicorn, websockets and setuptools<81 (requirements-bridge.txt)")

    def test_bridge_streams_into_the_session_the_website_joined(self):
        import uvicorn
        from websockets.sync.client import connect
        import server.app as srv
        srv.BASELINE_DIR, srv.API_TOKEN = tempfile.mkdtemp(), ""
        srv.SESSIONS.clear()
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(srv.app, host="127.0.0.1", port=port, log_level="error"))
        th = threading.Thread(target=server.run, daemon=True)
        th.start()
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(server.started)
        proc = None
        try:
            url = f"ws://127.0.0.1:{port}/ws"
            with connect(url, max_size=None) as app_:
                app_.send(json.dumps(dict(type="hello", role="app", user_id="live1")))
                self.assertFalse(json.loads(app_.recv(timeout=5))["engine_ready"])      # the website is first; no headset yet
                proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "openbci_bridge.py"), "--board", "synthetic", "--user-id", "live1",
                                         "--url", url, "--duration", "9", "--map", "1=Fp1,2=Fp2,3=AF7,4=AF8"],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                types, buf, end = set(), 0.0, time.time() + 25
                while time.time() < end and buf < 6:
                    app_.send(json.dumps(dict(type="status")))
                    while True:
                        m = json.loads(app_.recv(timeout=10))
                        if m["type"] == "status":
                            break
                        types.add(m["type"])
                    buf = m.get("eeg_buffer_s") or 0.0
                    time.sleep(0.5)
                self.assertGreaterEqual(buf, 6.0, "the bridge's EEG never reached the shared engine")
                self.assertIn("device_connected", types)                                  # the website was told when the headset joined
                self.assertEqual(m["connections"], {"app": 1, "device": 1})
            out, _ = proc.communicate(timeout=30)
            self.assertIn("joined session live1", out)
            self.assertIn("clock offset to server", out)
            off = float(out.split("clock offset to server:")[1].split("ms")[0])
            self.assertLess(abs(off), 50.0)                                              # same machine: offset is ~0
            self.assertRegex(out, r"done: \d+ samples sent, 0 dropped packets")
            n = int(out.split("done: ")[1].split(" samples")[0])
            self.assertTrue(1800 <= n <= 2500, n)                                        # ~250 Hz x 9 s from the synthetic board
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()
            server.should_exit = True
            th.join(timeout=10)


if __name__ == "__main__":
    unittest.main()
