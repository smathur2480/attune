"""Runs the JavaScript client (web-client/attune-client.js) under Node >= 22 against a real server: the website side of the protocol."""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def node_major():
    node = shutil.which("node")
    if not node:
        return 0
    try:
        return int(subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip().lstrip("v").split(".")[0])
    except ValueError:
        return 0


class TestJsClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import uvicorn, websockets  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("pip install -r requirements-server.txt")
        if node_major() < 22:
            raise unittest.SkipTest("needs Node.js >= 22 (built-in WebSocket)")

    def test_client_joins_session_syncs_clock_and_streams(self):
        import uvicorn
        import server.app as srv
        srv.BASELINE_DIR, srv.API_TOKEN = tempfile.mkdtemp(), ""
        srv.SESSIONS.clear()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(srv.app, host="127.0.0.1", port=port, log_level="error"))
        th = threading.Thread(target=server.run, daemon=True)
        th.start()
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        try:
            p = subprocess.run(["node", os.path.join(ROOT, "web-client", "test_client.mjs"), f"ws://127.0.0.1:{port}/ws"], capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stdout[-500:] + p.stderr[-800:])
            res = json.loads(p.stdout.split("RESULT ")[1].splitlines()[0])
        finally:
            server.should_exit = True
            th.join(timeout=10)
        self.assertFalse(res["appReadyFirst"]); self.assertTrue(res["devReady"]); self.assertTrue(res["deviceConnectedSeen"])   # headset joins after the website
        self.assertLess(abs(res["appOffsetMs"]), 50)                                   # same machine: clock offset ~0
        self.assertAlmostEqual(res["eegBufferS"], 6.0, delta=0.6)                      # JS-encoded EEG decoded correctly by the server
        self.assertEqual(res["eyeFrames"], 40)                                         # camera rows sent through the client's batching
        self.assertEqual(res["connections"], {"app": 1, "device": 1}); self.assertTrue(res["qualitySeen"])
        self.assertEqual(res["modeAfterStart"], "training"); self.assertTrue(res["trainingSummarySeen"])
        self.assertTrue(res["nowIsServerTime"])


if __name__ == "__main__":
    unittest.main()
