"""Tests for the relaxation / meditation score (relax_score.py), the planned heart module and the eyes-closed baseline."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import relax_score as R  # noqa: E402
from focus_core import baseline_ops, constants as K, eeg as eegm, heart, scoring, simulate as sim, trigger  # noqa: E402

NAMES = ["Fp1", "Fp2", "F3", "F4", "O1", "O2"]


def run(script, *args):
    p = subprocess.run([sys.executable, os.path.join(ROOT, script), *args], capture_output=True, text=True, cwd=ROOT)
    return p.returncode, p.stdout, p.stderr


class TestHeartModule(unittest.TestCase):
    def test_hrv_definitions(self):
        rr = np.array([800, 820, 790, 810, 830, 800, 790, 815, 805, 795, 820, 810, 800, 790, 805, 815, 800, 810, 795, 805, 800, 790, 810, 820, 805, 800, 795, 810, 805, 800, 815, 790, 800, 810], float)
        rr = np.tile(rr, 2)                                                           # ~54 s: above the 30-s minimum
        t = np.cumsum(rr) / 1000.0
        f, q = heart.hrv_features(t, rr)
        self.assertAlmostEqual(f["hr_mean"], 60000.0 / rr.mean(), places=6)
        self.assertAlmostEqual(f["ln_rmssd"], np.log(np.sqrt(np.mean(np.diff(rr) ** 2))), places=6)
        self.assertEqual(q["valid_frac"], 1.0)
        self.assertNotIn("resp_cpm", f)                                              # < 120 s: no frequency-domain features
        f2, q2 = heart.hrv_features(t[:20], rr[:20])                                  # ~16 s: too short
        self.assertEqual(f2, {}); self.assertIn("reason", q2); self.assertEqual(heart.quality_heart(q2), 0.0)

    def test_ecg_strip_to_beats(self):
        tb, rr = sim.synth_rr("rest", 120, seed=3)
        ecg = sim.synth_ecg(tb, 120, 250, seed=3)
        t_det, rr_det = heart.rr_from_ecg(ecg, 250)
        self.assertLessEqual(abs(len(t_det) - len(tb)), 1)
        j = np.argmin(np.abs(t_det[:, None] - tb[None, :]), axis=1)
        self.assertLess(np.median(np.abs(t_det - tb[j])), 0.02)                       # beats located within 20 ms
        self.assertLess(abs(60000 / rr_det.mean() - 60000 / rr.mean()), 1.0)         # mean heart rate within 1 bpm
        t_end, rr2, src = heart.from_arrays(ecg=ecg, fs=250)
        self.assertEqual(src, "ecg"); self.assertEqual(len(t_end), len(rr2))

    def test_artefact_cleaning_keeps_rmssd_honest(self):
        tb, rr = sim.synth_rr("rest", 180, seed=4)
        clean, _ = heart.hrv_features(tb, rr)
        tb2, rr2 = sim.synth_rr("rest", 180, seed=4, ectopic=6)
        dirty, q = heart.hrv_features(tb2, rr2)
        naive = np.log(np.sqrt(np.mean(np.diff(rr2) ** 2)))
        self.assertLess(q["valid_frac"], 1.0)
        self.assertLess(abs(dirty["ln_rmssd"] - clean["ln_rmssd"]), 0.2)               # cleaned estimate stays near the truth
        self.assertGreater(naive - clean["ln_rmssd"], 0.3)                            # an uncleaned RMSSD would be badly inflated

    def test_breathing_rate_and_hf_rule(self):
        fast, q = heart.hrv_features(*sim.synth_rr("relaxed", 180, seed=5))            # breathing at 0.25 Hz = 15/min
        self.assertAlmostEqual(fast["resp_cpm"], 15.0, delta=1.5); self.assertIn("ln_hf", fast)
        slow, q = heart.hrv_features(*sim.synth_rr("relaxed_paced", 180, seed=5))      # paced breathing at 0.1 Hz = 6/min
        self.assertAlmostEqual(slow["resp_cpm"], 6.0, delta=1.5)
        self.assertNotIn("ln_hf", slow)                                              # HF is not a vagal index below ~9/min (it moves to LF)
        self.assertIn("ln_rmssd", slow)

    def test_quality_and_ppg_discount(self):
        tb, rr = sim.synth_rr("rest", 90, seed=6)
        _, qe = heart.hrv_features(tb, rr, "ecg")
        _, qp = heart.hrv_features(tb, rr, "ppg")
        self.assertAlmostEqual(heart.quality_heart(qp), heart.quality_heart(qe) * K.PPG_QUALITY_FACTOR, places=6)
        _, q30 = heart.hrv_features(*sim.synth_rr("rest", 35, seed=6))
        self.assertLess(heart.quality_heart(q30), heart.quality_heart(qe))            # shorter segment, lower weight

    def test_baseline_needs_three_windows(self):
        with self.assertRaises(ValueError):
            heart.session_stats(*sim.synth_rr("rest", 80, seed=7))
        s = heart.session_stats(*sim.synth_rr("rest", 210, seed=7))
        self.assertGreaterEqual(s["quality"]["n_windows"], 3); self.assertIn("ln_rmssd", s["features"])


class TestRelaxBaselines(unittest.TestCase):
    def test_eyes_closed_block_and_rules(self):
        eo = (sim.synth_eeg("rest", seed=1, dur=100)[0], 250.0, NAMES)
        ec = (sim.synth_eeg("rest_closed", seed=2, dur=100)[0], 250.0, NAMES)
        b, notes = baseline_ops.add_rest_session(None, eo, sim.synth_eye("rest", seed=1, dur=90, off=0.0))
        b, notes = baseline_ops.add_rest_session(b, ec, condition="eyes_closed")
        self.assertIn("eeg", b); self.assertIn("eeg_closed", b)
        self.assertGreater(b["eeg_closed"]["features"]["alpha_rel"]["mean"], b["eeg"]["features"]["alpha_rel"]["mean"] + 0.1)   # Berger effect
        self.assertIn("theta_ex", b["eeg_closed"]["features"])
        with self.assertRaises(ValueError):
            baseline_ops.add_rest_session(b, ec, sim.synth_eye("rest", seed=1, dur=90), condition="eyes_closed")        # camera clip is eyes-open only
        with self.assertRaises(ValueError):
            baseline_ops.add_rest_session(b, ec, condition="eyes_shut")
        b2, notes = baseline_ops.add_rest_session(b, heart=dict(zip(("t_end", "rr_ms"), sim.synth_rr("rest", 210, seed=3)), source="ecg"))
        self.assertIn("heart_rest", b2); self.assertTrue(any(n.startswith("heart") for n in notes))
        with self.assertRaises(ValueError):
            baseline_ops.add_rest_session(b, heart=dict(zip(("t_end", "rr_ms"), sim.synth_rr("rest", 60, seed=3)), source="ecg"))


class TestRelaxScore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b, _ = baseline_ops.add_rest_session(None, (sim.synth_eeg("rest", seed=1, dur=120)[0], 250.0, NAMES), sim.synth_eye("rest", seed=1, dur=90, off=0.0))
        b, _ = baseline_ops.add_rest_session(b, (sim.synth_eeg("rest_closed", seed=2, dur=120)[0], 250.0, NAMES), condition="eyes_closed")
        tb, rr = sim.synth_rr("rest", 210, seed=3)
        cls.base, _ = baseline_ops.add_rest_session(b, heart=dict(t_end=tb, rr_ms=rr, source="ecg"))
        cls.closed = sim.synth_eye_closed(60, seed=4)
        cls.open_eye = sim.synth_eye("rest", seed=5, dur=60, off=0.0)

    def eeg(self, state, seed=7, dur=75):
        return (sim.synth_eeg(state, seed=seed, dur=dur)[0], 250.0, NAMES)

    def heart_in(self, state, seed=7, dur=150, source="ecg"):
        tb, rr = sim.synth_rr(state, dur, seed=seed)
        return dict(t_end=tb, rr_ms=rr, source=source)

    def test_condition_detection_and_baseline_matching(self):
        self.assertEqual(R.eyes_condition(self.closed, self.base)[0], "eyes_closed")
        self.assertEqual(R.eyes_condition(self.open_eye, self.base)[0], "eyes_open")
        mixed = sim.synth_eye_closed(60, seed=4, closed_frac=0.5)
        c, info = R.eyes_condition(mixed, self.base)
        self.assertEqual(c, "eyes_open"); self.assertTrue(info["mixed"])
        self.assertEqual(R.eyes_condition(None, self.base)[0], None)
        self.assertEqual(R.eyes_condition(None, self.base, "eyes_closed")[0], "eyes_closed")
        r = R.score_relax(self.base, self.eeg("relaxed_closed"), None)               # eyes state unknown -> EEG not scored (no guessing)
        self.assertIn("error", r); self.assertIn("eyes-open/closed unknown", " ".join(r["notes"]))

    def test_berger_effect_cannot_fake_relaxation(self):
        ec_rest = self.eeg("rest_closed")
        wrong, _q, _c = eegm.eeg_strengths(*ec_rest, baseline=self.base["eeg"], table=K.RELAX_EEG_FEATURES)      # eyes-closed vs eyes-OPEN baseline
        right = R.score_relax(self.base, ec_rest, self.closed)
        self.assertGreater(wrong["alpha_rel"], 1.5)                                  # the confound: eyes closed alone would look "very relaxed"
        self.assertTrue(40 <= right["relax_score"] <= 62, right["relax_score"])      # the matching baseline removes it (ordinary rest ~ 50)
        no_ec = {k: v for k, v in self.base.items() if k != "eeg_closed"}
        r = R.score_relax(no_ec, ec_rest, self.closed)
        self.assertIn("error", r); self.assertIn("eyes closed baseline", " ".join(r["notes"]))                    # refuses to score without it

    def test_states_order_and_drowsiness_guard(self):
        agg = {}
        for st in ("rest_closed", "relaxed_closed", "aroused_closed", "drowsy_closed"):
            rs = [R.score_relax(self.base, self.eeg(st, seed=s, dur=60), self.closed) for s in range(20, 26)]
            agg[st] = (np.mean([r["relax_score"] for r in rs]), np.mean([r["drowsy"] for r in rs]))
        self.assertGreater(agg["relaxed_closed"][0], agg["rest_closed"][0] + 12)
        self.assertGreater(agg["rest_closed"][0], agg["aroused_closed"][0] + 12)
        self.assertEqual((agg["rest_closed"][1], agg["relaxed_closed"][1], agg["aroused_closed"][1]), (0.0, 0.0, 0.0))     # no false drowsiness alarms
        self.assertEqual(agg["drowsy_closed"][1], 1.0)                                # drowsiness detected every time
        self.assertLess(agg["drowsy_closed"][0], agg["rest_closed"][0])               # and never rewarded as relaxation
        r = R.score_relax(self.base, self.eeg("drowsy_closed"), self.closed)
        self.assertEqual(r["state"], "drowsy"); self.assertIn("alpha", r["drowsy_reasons"][0]); self.assertTrue(r["suggestions"])

    def test_alpha_fall_alone_does_not_look_like_theta_rise(self):
        a = R.score_relax(self.base, self.eeg("aroused_closed"), self.closed)
        self.assertLess(a["strengths"]["eeg"]["alpha_rel"], -1.0)
        self.assertLess(a["strengths"]["eeg"]["theta_ex"], 1.0)                        # relative theta would have jumped; theta_ex does not
        self.assertFalse(a["drowsy"]); self.assertEqual(a["state"], "tense")

    def test_eyes_open_guard_uses_the_camera(self):
        ok = R.score_relax(self.base, self.eeg("relaxed_open"), self.open_eye)
        sleepy = R.score_relax(self.base, self.eeg("relaxed_open"), sim.synth_eye_sleepy_open(60, seed=5))
        self.assertEqual(ok["condition"], "eyes_open"); self.assertFalse(ok["drowsy"]); self.assertGreater(ok["relax_score"], 60)
        self.assertTrue(sleepy["drowsy"]); self.assertLess(sleepy["relax_score"], ok["relax_score"] - 20)
        self.assertTrue(any("perclos" in w or "blink" in w for w in sleepy["drowsy_reasons"]))

    def test_heart_modality_weights_and_formula(self):
        both = R.score_relax(self.base, self.eeg("relaxed_closed"), self.closed, self.heart_in("relaxed"))
        self.assertEqual(both["weights_used"], {"eeg": 0.5, "heart": 0.5})
        m = sum(both["weights_used"][k] * both["component_strength_sd"][k] for k in both["weights_used"])
        self.assertAlmostEqual(scoring.to_score(m - (K.DROWSY_PENALTY_SD if both["drowsy"] else 0)), both["relax_score"], delta=0.2)   # the printed formula
        self.assertGreater(both["relax_score"], R.score_relax(self.base, self.eeg("relaxed_closed"), self.closed)["relax_score"])
        hr_only = R.score_relax(self.base, None, None, self.heart_in("relaxed"))
        self.assertEqual(hr_only["weights_used"], {"heart": 1.0}); self.assertEqual(hr_only["condition"], None)
        ppg = R.score_relax(self.base, self.eeg("relaxed_closed"), self.closed, self.heart_in("relaxed", source="ppg"))
        self.assertAlmostEqual(ppg["weights_used"]["heart"], 0.35 / 0.85, places=2)     # PPG discounted (0.5*0.7 vs 0.5)
        clash = R.score_relax(self.base, self.eeg("relaxed_closed"), self.closed, self.heart_in("aroused"))
        self.assertIn("disagree", " ".join(clash["notes"])); self.assertLess(clash["relax_score"], both["relax_score"] - 40)
        eeg_only = R.score_relax(self.base, self.eeg("relaxed_closed"), self.closed)
        self.assertIn("planned feature", " ".join(eeg_only["notes"])); self.assertEqual(eeg_only["weights_used"], {"eeg": 1.0})

    def test_heart_needs_baseline_and_enough_data(self):
        nob = {k: v for k, v in self.base.items() if k != "heart_rest"}
        r = R.score_relax(nob, self.eeg("relaxed_closed"), self.closed, self.heart_in("relaxed"))
        self.assertNotIn("heart", r["weights_used"]); self.assertIn("no heart baseline", " ".join(r["notes"]))
        short = R.score_relax(self.base, self.eeg("relaxed_closed"), self.closed, self.heart_in("relaxed", dur=20))
        self.assertNotIn("heart", short["weights_used"]); self.assertIn("heart not scored", " ".join(short["notes"]))
        tb, rr = sim.synth_rr("relaxed", 150, seed=8)                                  # raw ECG strip works too
        via_ecg = R.score_relax(self.base, None, None, dict(ecg=sim.synth_ecg(tb, 150, 250, seed=8), fs=250.0), "eyes_closed")
        self.assertIn("heart", via_ecg["weights_used"]); self.assertGreater(via_ecg["relax_score"], 60)

    def test_flow_trace_and_edge_cases(self):
        r = R.score_relax(self.base, self.eeg("relaxed_closed"), self.closed, self.heart_in("relaxed"))
        self.assertEqual([s["step"] for s in r["flow"]], [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual([s["name"] for s in r["flow"]], ["input_gate", "condition", "quality", "patterns", "drowsiness_guard", "state", "suggestion"])
        self.assertEqual(r["flow"][0]["result"], "pass"); self.assertEqual(r["schema"], K.SCHEMA_RELAX)
        self.assertIn("relax_score", r["formula"])
        self.assertIn("error", R.score_relax(self.base)); self.assertEqual(R.score_relax(self.base)["flow"], [])
        none_ok = R.score_relax({}, self.eeg("relaxed_closed"), self.closed)
        self.assertIn("error", none_ok); self.assertEqual(none_ok["flow"][0]["result"], "fail")

    def test_deep_state_needs_sustained_evidence(self):
        r = R.score_relax(self.base, self.eeg("deep_closed"), self.closed, self.heart_in("relaxed"))
        self.assertTrue(r["deep_candidate"]); self.assertEqual(r["state"], "relaxed")  # a single update never claims "deeply relaxed"
        t = np.arange(0, 200, 5.0)
        self.assertFalse(trigger.sustained(t[:20], [True] * 20, 120))                  # 95 s so far
        self.assertTrue(trigger.sustained(t, [True] * len(t), 120))
        self.assertFalse(trigger.sustained(t, [True] * 30 + [False] * 10, 120))        # ended -> not currently sustained
        self.assertFalse(trigger.sustained([], [], 120))

    def test_cli_end_to_end(self):
        d = tempfile.mkdtemp()
        sim.save_eeg_csv((sim.synth_eeg("rest_closed", seed=2, dur=100)[0], 250.0, NAMES), os.path.join(d, "rest_closed.csv"))
        sim.save_eeg_csv((sim.synth_eeg("rest", seed=1, dur=100)[0], 250.0, NAMES), os.path.join(d, "rest_open.csv"))
        sim.synth_eye("rest", seed=1, dur=90, off=0.0).to_csv(os.path.join(d, "rest_eye.csv"), index=False)
        tb, rr = sim.synth_rr("rest", 210, seed=3)
        pd.DataFrame(dict(t=tb, rr_ms=rr)).to_csv(os.path.join(d, "rr_rest.csv"), index=False)
        b = os.path.join(d, "b.json")
        for args in (("--eeg", os.path.join(d, "rest_open.csv"), "--fs", "250", "--eye", os.path.join(d, "rest_eye.csv")),
                     ("--eeg", os.path.join(d, "rest_closed.csv"), "--fs", "250", "--condition", "eyes_closed"),
                     ("--heart", os.path.join(d, "rr_rest.csv"))):
            code, out, err = run("baseline.py", "rest", "--baseline", b, *args)
            self.assertEqual(code, 0, err[-600:])
        code, out, _ = run("baseline.py", "show", "--baseline", b)
        self.assertIn("EEG closed: 1 session", out); self.assertIn("heart rest: 1 session", out)
        code, out, err = run("baseline.py", "rest", "--baseline", b, "--eeg", os.path.join(d, "rest_closed.csv"), "--fs", "250",
                             "--eye", os.path.join(d, "rest_eye.csv"), "--condition", "eyes_closed")
        self.assertNotEqual(code, 0); self.assertIn("only used for eyes-open rest", err)
        sim.save_eeg_csv((sim.synth_eeg("relaxed_closed", seed=7, dur=75)[0], 250.0, NAMES), os.path.join(d, "s.csv"))
        self.closed.to_csv(os.path.join(d, "s_eye.csv"), index=False)
        tb, rr = sim.synth_rr("relaxed", 150, seed=7)
        pd.DataFrame(dict(t=tb, rr_ms=rr)).to_csv(os.path.join(d, "s_rr.csv"), index=False)
        code, out, err = run("relax_score.py", "--baseline", b, "--eeg", os.path.join(d, "s.csv"), "--fs", "250", "--eye", os.path.join(d, "s_eye.csv"),
                             "--heart", os.path.join(d, "s_rr.csv"))
        self.assertEqual(code, 0, err[-600:])
        res = json.loads(out)
        self.assertEqual((res["condition"], res["kind"]), ("eyes_closed", "relax")); self.assertGreater(res["relax_score"], 60)
        self.assertEqual(set(res["weights_used"]), {"eeg", "heart"})

    def test_references_for_relax_tables_exist(self):
        keys = {r for tab in (K.RELAX_EEG_FEATURES, K.HEART_FEATURES) for spec in tab.values() for r in spec["refs"]}
        with open(os.path.join(ROOT, "REFERENCES.md")) as fh:
            text = fh.read()
        for k in keys | {"Chokroverty", "Magosso2007", "Strauss2022", "Patent5813993", "WatchECG2024", "Taoum2022", "PRVMeta", "JMIR2020", "SmartwatchHRV2019"}:
            self.assertIn(f"[{k}]", text, k)
        for tab in (K.RELAX_EEG_FEATURES, K.HEART_FEATURES):
            for name, spec in tab.items():
                self.assertTrue(spec["direction"] in (-1, 1) and spec["weight"] > 0 and spec["floor"] > 0 and spec["refs"], name)
        self.assertAlmostEqual(sum(K.RELAX_BASE_WEIGHTS.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
