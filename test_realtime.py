"""Tests for the real-time engine (realtime.py) and the WebSocket service (server/app.py)."""
import base64
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import simple_score  # noqa: E402
from focus_core import baseline_ops, scoring, simulate as sim, trigger  # noqa: E402
from realtime import FocusEngine  # noqa: E402

FS, NAMES = 250, ["Fp1", "Fp2", "F3", "F4", "O1", "O2"]


def run_stream(eng, x, eye, events=(), t0=0.0, chunk_s=0.25):
    msgs = []
    for item in sim.iter_stream((x, FS, NAMES), eye, events, chunk_s=chunk_s, t0=t0):
        if item[0] == "eeg":
            msgs += eng.push_eeg(item[2], item[1])
        elif item[0] == "eye":
            msgs += eng.push_eye(item[1])
        else:
            msgs += eng.push_event(item[1], item[2])
    return msgs


def of(msgs, typ):
    return [m for m in msgs if m["type"] == typ]


class TestTrigger(unittest.TestCase):
    def test_trigger_rule(self):
        t = np.arange(0, 300, 5.0)
        low = np.where((t >= 100) & (t <= 190), 20.0, 70.0)
        hit, thr, when = trigger.should_suggest_training(t, low)
        self.assertTrue(hit); self.assertEqual(thr, 35.0); self.assertAlmostEqual(when, 145.0)
        self.assertFalse(trigger.should_suggest_training(t, np.where((t >= 100) & (t <= 120), 20.0, 70.0))[0])     # a 20-s dip is not enough
        mine = np.random.default_rng(0).normal(60, 8, 200)
        self.assertAlmostEqual(trigger.should_suggest_training(t, low, personal_scores=mine)[1], np.percentile(mine, 15))


class TestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x0, _, _ = sim.synth_eeg("rest", seed=1, dur=120)
        cls.baseline, _ = baseline_ops.add_rest_session(None, (x0, float(FS), NAMES), sim.synth_eye("rest", seed=1, dur=90, off=0.0))
        cls.profile = dict(age=29, sex="F", hours_since={"caffeine": 2, "alcohol": 20}, asrs_items=[2] * 18)

    def engine(self, **kw):
        kw.setdefault("baseline", self.baseline)
        return FocusEngine(FS, NAMES, **kw)

    def capture(self, eng):
        """Record every (eeg, eye, task) the engine hands to the scorer and the raw strength it gets back."""
        calls = []
        orig = eng._score

        def wrapper(eeg, eye, task, timeline):
            res = orig(eeg, eye, task, timeline)
            calls.append(dict(eeg=eeg, eye=eye, task=task, res=res))
            return res
        eng._score = wrapper
        return calls

    def test_warmup_cadence_and_correct_window(self):
        x, _, _ = sim.synth_eeg("focused", seed=11, dur=70)
        eye = sim.synth_eye("focused", seed=11, dur=70)
        eng = self.engine()
        calls = self.capture(eng)
        msgs = run_stream(eng, x, eye)
        sc = of(msgs, "score")
        self.assertTrue(15.0 <= sc[0]["t"] < 16.5, sc[0]["t"])                       # nothing before the 15-s warm-up
        gaps = np.diff([m["t"] for m in sc])
        self.assertTrue(((gaps > 4.8) & (gaps < 5.6)).all(), gaps)                    # one update every ~5 s
        last, c = sc[-1], calls[-1]
        n = c["eeg"][0].shape[1]
        self.assertTrue(abs(n - 30 * FS) <= 3, c["eeg"][0].shape)                      # a 30-s window ...
        j = int(np.where(np.isclose(x[0], c["eeg"][0][0, -1]))[0][0]) + 1               # ... that is exactly the LATEST 30 s of the stream
        np.testing.assert_allclose(c["eeg"][0], x[:, j - n:j], atol=1e-9)
        self.assertLessEqual(abs(j / FS - last["t"]), 0.3)                              # and ends at the newest EEG sample
        end = c["eye"]["t"].iloc[-1]
        expect = eye[(eye.t >= end - 30 - 1e-9) & (eye.t <= end + 1e-9)]
        self.assertEqual(len(c["eye"]), len(expect))
        batch = simple_score.score_simple(self.baseline, c["eeg"], c["eye"], None, timeline=False)       # same function, same inputs
        self.assertEqual(last["simple_score"], batch["simple_score"])

    def test_smoothing_is_ema_in_strength_space(self):
        x, _, _ = sim.synth_eeg("focused", seed=11, dur=45)
        eng = self.engine(smoothing=0.5)
        calls = self.capture(eng)
        sc = of(run_stream(eng, x, sim.synth_eye("focused", seed=11, dur=45)), "score")
        m = [c["res"]["mean_strength_sd"] for c in calls]
        ema = [m[0]]
        for v in m[1:]:
            ema.append(0.5 * v + 0.5 * ema[-1])
        for s, e in zip(sc, ema):
            self.assertAlmostEqual(s["score"], round(scoring.to_score(e), 1), delta=0.06)

    def test_focused_vs_drifting_attention_and_lag(self):
        x = np.concatenate([sim.synth_eeg("focused", seed=11, dur=60)[0], sim.synth_eeg("wandering", seed=12, dur=60)[0]], axis=1)
        eye = pd.concat([sim.synth_eye("focused", seed=11, dur=60), sim.synth_eye("wandering", seed=12, dur=60)])
        eye["t"] = np.arange(len(eye)) / 30.0
        sc = of(run_stream(self.engine(), x, eye), "score")
        early = np.mean([m["score"] for m in sc if 30 <= m["t"] <= 60]); late = np.mean([m["score"] for m in sc if m["t"] >= 105])
        self.assertGreater(early, late + 15)
        just_after = [m["score"] for m in sc if 60 < m["t"] <= 65][0]
        self.assertGreater(just_after, late + 8)                                      # a 30-s window lags: the score has not yet fallen fully

    def test_bad_electrode_pauses_eeg_prompts_and_resumes(self):
        x, _, _ = sim.synth_eeg("focused", seed=11, dur=100)
        x[0, 40 * FS:70 * FS] = 0.05 * np.random.default_rng(0).normal(size=30 * FS)
        msgs = run_stream(self.engine(), x, sim.synth_eye("focused", seed=11, dur=100))
        sc = of(msgs, "score")
        paused = [m for m in sc if m["paused"].get("eeg")]
        self.assertTrue(paused and all("eeg" not in m["components"] and "left forehead" in m["paused"]["eeg"] for m in paused))
        self.assertTrue(all("eye" in m["components"] for m in paused))                # the score continues from the camera alone
        prompts = [p for q in of(msgs, "quality") for p in q["prompts"]]
        self.assertTrue(any(p["code"] == "flat" and p["channels"] == ["Fp1"] for p in prompts))
        self.assertTrue(any("looks good now" in r for q in of(msgs, "quality") for r in q["resolved"]))
        self.assertIn("eeg", sc[-1]["components"]); self.assertEqual(sc[-1]["paused"], {})       # resumed after the fix
        qt = [q["t"] for q in of(msgs, "quality") if q["prompts"]]
        self.assertTrue(all(40 < t < 75 for t in qt), qt)                              # prompt times are on the APP clock

    def test_stream_gaps_and_stalls_are_never_scored_as_fresh(self):
        x, _, _ = sim.synth_eeg("focused", seed=11, dur=60)
        eye = sim.synth_eye("focused", seed=11, dur=60)
        eng = self.engine()
        msgs = []
        for item in sim.iter_stream((x, FS, NAMES), eye):                            # EEG chunks stop between 30 s and 45 s
            if item[0] == "eeg":
                if 30 <= item[1] < 45:
                    continue
                msgs += eng.push_eeg(item[2], item[1])
            else:
                msgs += eng.push_eye(item[1])
        w = of(msgs, "warning")
        self.assertTrue(any(m["code"] == "eeg_gap" for m in w))
        stalled = [m for m in of(msgs, "score") if 33 < m["t"] < 45]
        self.assertTrue(stalled and all("stalled" in m["paused"].get("eeg", "") and "eeg" not in m["components"] for m in stalled))
        eng2 = self.engine()
        eye2 = eye.copy(); eye2 = eye2[(eye2.t < 20) | (eye2.t >= 35)]
        w2 = of(run_stream(eng2, x, eye2), "warning")
        self.assertTrue(any(m["code"] == "eye_gap" for m in w2))
        with self.assertRaises(ValueError):
            eng.push_eeg(np.zeros((3, 10)))                                           # wrong channel count is rejected loudly

    def test_training_session_matches_batch_scoring(self):
        s = sim.simulate_session("lapsing", seed=3, duration=120)
        eng = self.engine()
        calls = self.capture(eng)
        eng.start_training(0.0)
        events = [("change", c) for c in s["changes"]] + [("click", c) for c in s["clicks"]]
        msgs = run_stream(eng, s["eeg"][0], s["eye"], events)
        ticks = [m for m in of(msgs, "score") if m["mode"] == "training"]
        self.assertTrue(20 <= ticks[0]["t"] < 21.5); self.assertIn("click", ticks[-1]["components"])
        last = calls[-1]["task"]                                                    # targets whose response window is still open are not scored
        self.assertTrue(all(ch <= 120 - 1.5 for ch in last["changes"]))
        fin = eng.finish_training()[0]
        delivered = (s["eeg"][0].shape[1] // 62) * 62                                  # a live client never delivers the final partial chunk
        eeg_d = (s["eeg"][0][:, :delivered], s["eeg"][1], s["eeg"][2])
        batch = simple_score.score_simple(self.baseline, eeg_d, s["eye"], dict(changes=s["changes"], clicks=s["clicks"], ball=None,
                                                                              duration_s=fin["result"]["metrics"]["click"] and 119.97))
        self.assertEqual(fin["type"], "training_summary")
        self.assertAlmostEqual(fin["result"]["simple_score"], batch["simple_score"], delta=0.15)   # live session == offline scoring of the same data
        for k in batch["components"]:
            self.assertAlmostEqual(fin["result"]["components"][k], batch["components"][k], delta=0.15)
        self.assertGreaterEqual(len(fin["result"]["timeline"]), 3)
        self.assertEqual(eng.mode, "passive")

    def test_training_updates_baseline_only_for_valid_sessions(self):
        eng = self.engine(baseline=json.loads(json.dumps(self.baseline)))
        s = sim.simulate_session("focused", seed=5, duration=120)
        eng.start_training(0.0)
        run_stream(eng, s["eeg"][0], s["eye"], [("change", c) for c in s["changes"]] + [("click", c) for c in s["clicks"]])
        fin = eng.finish_training(add_to_baseline=True)[0]
        self.assertTrue(fin["baseline_updated"]); self.assertEqual(fin["baseline_sessions"], 1)
        self.assertEqual(len(eng.baseline["training"]["sessions"]), 1)
        eng.start_training(130.0)
        short = sim.simulate_session("focused", seed=6, duration=40)
        run_stream(eng, short["eeg"][0], short["eye"], [("change", c) for c in short["changes"]] + [("click", c) for c in short["clicks"]], t0=130.0)
        fin2 = eng.finish_training(add_to_baseline=True)[0]
        self.assertFalse(fin2["baseline_updated"]); self.assertIn("colour changes", fin2["baseline_error"])
        self.assertEqual(len(eng.baseline["training"]["sessions"]), 1)               # unchanged
        self.assertEqual(eng.push_event("click", 5.0)[0]["code"], "event_outside_training")

    def test_rest_calibration_workflow(self):
        eng = FocusEngine(FS, NAMES)
        self.assertEqual(eng.start_rest(0.0)[0]["type"], "rest_started")
        x, _, _ = sim.synth_eeg("rest", seed=1, dur=100)
        msgs = run_stream(eng, x, sim.synth_eye("rest", seed=1, dur=100, off=0.0))
        self.assertEqual(of(msgs, "score"), [])                                      # no scores while calibrating
        done = eng.finish_rest()[0]
        self.assertEqual(done["type"], "rest_calibrated"); self.assertTrue(done["baseline"]["eeg"] and done["baseline"]["eye_rest"])
        x2, _, _ = sim.synth_eeg("rest", seed=9, dur=45)
        sc = of(run_stream(eng, x2, sim.synth_eye("rest", seed=9, dur=45), t0=110.0), "score")
        self.assertTrue(sc and sc[-1]["baseline_basis"]["eeg"] == "own baseline")
        self.assertTrue(30 <= sc[-1]["score"] <= 70)                                  # resting data scored against its own baseline ~ 50
        eng2 = FocusEngine(FS, NAMES); eng2.start_rest(0.0)
        eng2.push_eeg(sim.synth_eeg("rest", seed=2, dur=10)[0], 0.0)
        self.assertEqual(eng2.finish_rest()[0]["type"], "rest_failed")                # too little data is reported, not crashed on

    def test_training_suggestion_fires_once(self):
        x, _, _ = sim.synth_eeg("tired", seed=12, dur=150)
        msgs = run_stream(self.engine(), x, sim.synth_eye("tired", seed=12, dur=150))
        sug = of(msgs, "suggest_training")
        self.assertEqual(len(sug), 1)                                                 # cooldown prevents nagging
        x, _, _ = sim.synth_eeg("focused", seed=11, dur=150)
        self.assertEqual(of(run_stream(self.engine(), x, sim.synth_eye("focused", seed=11, dur=150)), "suggest_training"), [])

    def test_complex_mode_in_engine(self):
        x, _, _ = sim.synth_eeg("focused", seed=11, dur=40)
        eng = self.engine(profile=self.profile)
        sc = of(run_stream(eng, x, sim.synth_eye("focused", seed=11, dur=40)), "score")
        self.assertEqual(sc[-1]["kind"], "complex"); self.assertIn("complex_score", sc[-1]); self.assertIn("shifts_sd", sc[-1])
        self.assertNotEqual(sc[-1]["complex_score"], sc[-1]["simple_score"])
        eng.update_profile(None)
        self.assertEqual(eng.use_complex, False)
        with self.assertRaises(ValueError):
            FocusEngine(FS, ["Fp1", "Fp2"], baseline=self.baseline)                    # baseline needs channels the stream lacks


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("pip install fastapi httpx for the server tests")
        import server.app as srv
        from server.client_demo import Conn, eeg_msg
        cls.srv, cls.Conn, cls.eeg_msg = srv, Conn, staticmethod(eeg_msg)

    def setUp(self):
        self.srv.BASELINE_DIR = tempfile.mkdtemp()
        self.srv.API_TOKEN = ""
        os.environ["FOCUS_BASELINE_DIR"] = self.srv.BASELINE_DIR

    def hello(self, conn, uid="u1", **kw):
        return conn.send(dict(type="hello", user_id=uid, fs=FS, channels=NAMES, acks=True, **kw))

    def test_protocol_roundtrip_calibrate_persist_and_reconnect(self):
        conn = self.Conn()
        r = self.hello(conn)
        self.assertEqual((r[0]["type"], r[0]["has_baseline"]), ("ready", False))
        conn.send(dict(type="rest", action="start", t=0.0))
        x, _, _ = sim.synth_eeg("rest", seed=1, dur=100)
        for item in sim.iter_stream((x, FS, NAMES), sim.synth_eye("rest", seed=1, dur=100, off=0.0)):
            conn.send(self.eeg_msg(item[1], item[2]) if item[0] == "eeg" else dict(type="eye", rows=item[1]))
        r = conn.send(dict(type="rest", action="finish"))
        self.assertEqual(r[0]["type"], "rest_calibrated"); self.assertNotIn("baseline", r[0]); self.assertIn("baseline_summary", r[0])   # personal data stays server-side
        self.assertTrue(os.path.exists(os.path.join(self.srv.BASELINE_DIR, "u1.json")))
        # plain-JSON EEG samples are accepted too, and scores flow back
        x2, _, _ = sim.synth_eeg("focused", seed=11, dur=30)
        got = []
        for item in sim.iter_stream((x2, FS, NAMES), sim.synth_eye("focused", seed=11, dur=30), t0=110.0):
            got += conn.send(dict(type="eeg", t_start=item[1], samples=item[2].tolist()) if item[0] == "eeg" else dict(type="eye", rows=item[1]))
        self.assertTrue([m for m in got if m["type"] == "score"])
        conn.close()
        conn2 = self.Conn(); r = self.hello(conn2)                                    # a new session picks the stored baseline up
        self.assertTrue(r[0]["has_baseline"]); conn2.close()

    def test_bad_input_never_kills_the_session_and_ids_are_validated(self):
        conn = self.Conn()
        self.assertEqual(self.hello(conn, "../etc/passwd")[0]["type"], "error")      # user ids become file names
        self.assertEqual(conn.send(dict(type="ping"))[0]["type"], "error")           # hello required first
        self.hello(conn)
        self.assertEqual(conn.send(dict(type="eeg", t_start=0, samples=5, data=base64.b64encode(b"abc").decode()))[0]["type"], "error")
        self.assertEqual(conn.send(dict(type="nonsense"))[0]["type"], "error")
        self.assertEqual(conn.send(dict(type="event", kind="click", t=1.0))[0]["code"], "event_outside_training")
        self.assertEqual(conn.send(dict(type="ping"))[0]["type"], "pong")             # still alive
        conn.close()

    def test_token_authentication(self):
        self.srv.API_TOKEN = "secret"
        conn = self.Conn()
        conn._send(dict(type="hello", user_id="u1", fs=FS, channels=NAMES, token="wrong"))
        self.assertEqual(conn._recv()["type"], "error")
        conn2 = self.Conn()
        r = conn2.send(dict(type="hello", user_id="u1", fs=FS, channels=NAMES, token="secret", acks=True))
        self.assertEqual(r[0]["type"], "ready"); conn2.close()


if __name__ == "__main__":
    unittest.main()
