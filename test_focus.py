"""
Tests for the focus repo. Run from the repo root:   python -m unittest discover -s tests -v
They check (1) the formula numerically, (2) that the three scripts share it, (3) the baseline rules, (4) signal quality, camera.
"""
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

import baseline as baseline_script  # noqa: E402
import complex_score  # noqa: E402
import simple_score  # noqa: E402
from focus_core import baselines, clicks, constants as K, context, eeg, pursuit, scoring, simulate as sim, task  # noqa: E402
from focus_core import signal_quality as sq  # noqa: E402


def run(script, *args):
    p = subprocess.run([sys.executable, os.path.join(ROOT, script), *args], capture_output=True, text=True, cwd=ROOT)
    return p.returncode, p.stdout, p.stderr


def cli_json(script, *args):
    code, out, err = run(script, *args)
    assert code == 0, (script, args, err[-800:])
    return json.loads(out)


class TestFormula(unittest.TestCase):
    def test_strength_clip_direction_floor(self):
        spec = K.CLICK_FEATURES["lapse_rate"]                                   # direction -1, mean .05, sd .08, floor .02
        self.assertEqual(scoring.strength(0.333, spec), -3.0)                   # (0.333-0.05)/0.08 = 3.5 -> clipped
        self.assertAlmostEqual(scoring.strength(0.05, spec), 0.0)
        self.assertGreater(scoring.strength(0.0, spec), 0.0)                    # lower lapse rate = better
        hr = K.CLICK_FEATURES["hit_rate"]
        self.assertAlmostEqual(scoring.strength(0.909, hr), (0.909 - 0.95) / 0.06, places=6)
        entry = dict(mean=0.5, sd=0.0001)                                        # tiny personal SD -> floor protects
        self.assertAlmostEqual(scoring.strength(0.53, hr, entry), (0.53 - 0.5) / hr["floor"], places=6)
        self.assertIsNone(scoring.strength(float("nan"), hr)); self.assertIsNone(scoring.strength(1.0, K.EEG_FEATURES["tbr"]))   # no reference

    def test_modality_mean_is_weighted_average(self):
        s = {"hit_rate": -0.68, "median_rt_ms": -0.65, "rt_sd_ms": -0.54, "lapse_rate": -3.0, "false_alarm_per_min": 0.62}
        w = {k: K.CLICK_FEATURES[k]["weight"] for k in s}
        expect = sum(w[k] * s[k] for k in s) / sum(w.values())
        self.assertAlmostEqual(scoring.modality_mean(s, K.CLICK_FEATURES), expect)
        self.assertAlmostEqual(expect, -1.03, places=2)                           # the worked example in the documentation
        self.assertIsNone(scoring.modality_mean({}, K.CLICK_FEATURES))

    def test_combine_and_score_mapping(self):
        mods = {"click": -1.03, "eye": 0.25, "eeg": -1.65}
        m, w = scoring.combine(mods, {"click": 1, "eye": 1, "eeg": 1})
        self.assertEqual(w, {"click": 0.45, "eye": 0.35, "eeg": 0.2})
        self.assertAlmostEqual(m, 0.45 * -1.03 + 0.35 * 0.25 + 0.2 * -1.65)
        self.assertAlmostEqual(scoring.to_score(m), 50 + 25 * m)
        m2, w2 = scoring.combine(mods, {"click": scoring.quality_click(6), "eye": 1, "eeg": 1})   # only 6 targets -> click quality 0.5
        self.assertAlmostEqual(w2["click"], 0.225 / 0.775, places=3); self.assertAlmostEqual(w2["eye"], 0.35 / 0.775, places=3)
        m3, w3 = scoring.combine({"click": -1.0, "eye": 0.5, "eeg": None}, {"click": 1, "eye": 1, "eeg": 1})   # EEG missing -> renormalise
        self.assertAlmostEqual(w3["click"], 0.45 / 0.80, places=3); self.assertNotIn("eeg", w3)
        self.assertEqual(scoring.combine({"click": 1.0}, {"click": 0.0}), (None, {}))
        self.assertEqual((scoring.to_score(0), scoring.to_score(1), scoring.to_score(-5), scoring.to_score(9)), (50.0, 75.0, 1.0, 100.0))

    def test_quality_ramps(self):
        self.assertEqual((scoring.quality_click(2), scoring.quality_click(6), scoring.quality_click(10), scoring.quality_click(30)), (0.0, 0.5, 1.0, 1.0))
        self.assertEqual((scoring.quality_eye(0.4), scoring.quality_eye(0.9)), (0.5, 1.0))
        self.assertAlmostEqual(scoring.quality_eeg(7, 10), 1.0); self.assertAlmostEqual(scoring.quality_eeg(3.5, 10), 0.5)

    def test_single_source_of_truth(self):
        self.assertIs(simple_score.K, K); self.assertIs(complex_score.K, K); self.assertIs(complex_score.score_simple, simple_score.score_simple)
        self.assertIs(complex_score.scoring, scoring); self.assertAlmostEqual(sum(K.BASE_WEIGHTS.values()), 1.0)
        for tab in (K.EEG_FEATURES, K.EYE_FEATURES, K.CLICK_FEATURES):
            for name, spec in tab.items():
                self.assertTrue(spec["direction"] in (-1, 1) and spec["weight"] > 0 and spec["floor"] > 0 and spec["refs"], name)
        for name, spec in {**K.EYE_FEATURES, **K.CLICK_FEATURES}.items():
            self.assertTrue(spec["mean"] is not None and spec["sd"] > 0, name)         # placeholders exist where no EEG reference is needed

    def test_references_exist(self):
        keys = {r for tab in (K.EEG_FEATURES, K.EYE_FEATURES, K.CLICK_FEATURES) for spec in tab.values() for r in spec["refs"]}
        with open(os.path.join(ROOT, "REFERENCES.md")) as fh:
            text = fh.read()
        for k in keys:
            self.assertIn(f"[{k}]", text, f"{k} is cited in constants.py but missing from REFERENCES.md")


class TestClicksPursuit(unittest.TestCase):
    def test_click_matching_rules(self):
        r = clicks.analyze_clicks([10, 20, 30, 40, 50], [10.35, 20.8, 40.1, 40.5, 45.0, 50.2], duration_s=60)
        self.assertEqual((r["n_targets"], r["n_hits"], r["false_alarms"]), (5, 4, 2))      # 40.1 anticipation, 45.0 stray
        self.assertAlmostEqual(r["median_rt_ms"], 425, delta=1)
        self.assertAlmostEqual(r["lapse_rate"], (1 + 2) / 5)                                   # one miss + RT 800 + RT 500 (>= 500 ms)
        late = clicks.analyze_clicks([10], [12.0], duration_s=30)
        self.assertEqual((late["n_hits"], late["false_alarms"]), (0, 1))

    def test_pursuit_and_lapse_types(self):
        f, t = sim.simulate_session("focused", seed=3, duration=120), sim.simulate_session("tired", seed=3, duration=120)
        pf, _ = pursuit.analyze_pursuit(f["eye"], f["ball"]); pt, _ = pursuit.analyze_pursuit(t["eye"], t["ball"])
        self.assertAlmostEqual(pf["lag_ms"], 150, delta=70); self.assertGreater(pf["pursuit_corr"], 0.85)
        self.assertGreaterEqual(pt["n_microsleeps"], 1); self.assertGreaterEqual(pt["n_attention_lapses"], 2)
        self.assertGreater(pt["lapse_time_frac"], pf["lapse_time_frac"] + 0.02)


class TestBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        sim.write_demo(cls.d)
        cls.p = lambda self, f: os.path.join(self.d, f)

    def fresh(self):
        return os.path.join(tempfile.mkdtemp(), "b.json")

    def test_pooled_eeg_is_exact(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(1, 0.5, 40), rng.normal(1.6, 0.8, 25)
        stat = lambda v: dict(features={"f": dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), n=len(v))})   # noqa: E731
        p = baselines.pooled_eeg([stat(a), stat(b)])["f"]
        both = np.r_[a, b]
        self.assertAlmostEqual(p["mean"], both.mean(), places=10); self.assertAlmostEqual(p["sd"], both.std(ddof=1), places=10)

    def test_metric_entries_sd_rule(self):
        tab = K.CLICK_FEATURES
        two = baselines.metric_entries([dict(values=dict(hit_rate=0.9)), dict(values=dict(hit_rate=1.0))], tab)["hit_rate"]
        self.assertEqual(two["sd_source"], "placeholder"); self.assertEqual(two["sd"], tab["hit_rate"]["sd"])
        three = baselines.metric_entries([dict(values=dict(hit_rate=v)) for v in (0.9, 1.0, 0.95)], tab)["hit_rate"]
        self.assertEqual(three["sd_source"], "between-session"); self.assertAlmostEqual(three["sd"], 0.05)

    def test_rest_then_training_workflow(self):
        b = self.fresh()
        code, out, err = run("baseline.py", "rest", "--baseline", b, "--eeg", self.p("rest_eeg.csv"), "--fs", "250", "--eye", self.p("rest_eye.csv"))
        self.assertEqual(code, 0, err[-500:]); self.assertIn("clean windows", out)
        data = json.load(open(b))
        self.assertEqual(data["schema"], K.SCHEMA_BASELINE)
        self.assertEqual(sorted(data["eeg"]["front"]), ["F3", "F4", "Fp1", "Fp2"]); self.assertEqual(sorted(data["eeg"]["post"]), ["O1", "O2"])
        self.assertTrue(all(v["sd_source"] == "placeholder" for v in data["eye_rest"]["features"].values()))
        for i in (1, 2):
            code, out, err = run("baseline.py", "training", "--baseline", b, "--events", self.p(f"train_{i}_events.csv"), "--eye", self.p(f"train_{i}_eye.csv"))
            self.assertEqual(code, 0, err[-500:])
        self.assertIn("not yet", out)
        code, out, err = run("baseline.py", "training", "--baseline", b, "--events", self.p("train_3_events.csv"), "--eye", self.p("train_3_eye.csv"))
        self.assertIn("active", out)
        data = json.load(open(b))
        self.assertEqual(len(data["training"]["sessions"]), 3)
        self.assertTrue(all(v["sd_source"] == "between-session" for v in data["training"]["features"].values()))
        code, out, _ = run("baseline.py", "show", "--baseline", b)
        self.assertEqual(code, 0); self.assertIn("training  : 3 session(s)", out)

    def test_bad_channel_is_screened_out_and_channel_change_refused(self):
        x, fs, names = sim.synth_eeg("rest", seed=2, dur=100)
        x[0] = 0.05 * np.random.default_rng(1).normal(size=x.shape[1])                    # Fp1 flat
        f = os.path.join(tempfile.mkdtemp(), "flat.csv"); sim.save_eeg_csv((x, fs, names), f)
        b = self.fresh()
        code, out, err = run("baseline.py", "rest", "--baseline", b, "--eeg", f, "--fs", "250")
        self.assertEqual(code, 0, err[-500:]); data = json.load(open(b))
        self.assertIn("Fp1", data["eeg"]["excluded"]); self.assertNotIn("Fp1", data["eeg"]["front"])
        code, out, err = run("baseline.py", "rest", "--baseline", b, "--eeg", self.p("rest_eeg.csv"), "--fs", "250")   # Fp1 now fine: channel set changed
        self.assertNotEqual(code, 0); self.assertIn("channel set changed", err)
        code, out, err = run("baseline.py", "rest", "--baseline", b, "--eeg", self.p("rest_eeg.csv"), "--fs", "250", "--reset")
        self.assertEqual(code, 0, err[-500:])

    def test_short_training_session_is_refused(self):
        s = sim.simulate_session("focused", seed=5, duration=40)
        pre = os.path.join(tempfile.mkdtemp(), "short"); sim.save_session(s, pre)
        b = self.fresh()
        code, out, err = run("baseline.py", "training", "--baseline", b, "--events", pre + "_events.csv", "--eye", pre + "_eye.csv")
        self.assertNotEqual(code, 0); self.assertIn("colour changes", err)
        self.assertEqual(run("baseline.py", "training", "--baseline", b, "--events", pre + "_events.csv", "--eye", pre + "_eye.csv", "--force")[0], 0)

    def test_reference_fit_uses_covariates_only_when_they_help(self):
        d = tempfile.mkdtemp()
        rng = np.random.default_rng(3)
        n = 200
        age = rng.uniform(20, 60, n)
        coh = pd.DataFrame(dict(TDBRAIN_ID=[f"s{i}" for i in range(n)], sessID=1, group="HEALTHY", age=age, sex_F=rng.integers(0, 2, n),
                                cigarette_99=1, cigarette_miss=0, coffee_99=1, coffee_miss=0, beer_99=1, beer_miss=0))
        z = (age - 40) / 10
        feat = pd.DataFrame(dict(TDBRAIN_ID=coh.TDBRAIN_ID, sessID=1, status="ok", frontal_alpha=-0.6 - 0.3 * z + rng.normal(0, 0.15, n),
                                 frontal_tbr=rng.normal(0, 0.4, n), frontal_beta=rng.normal(-0.8, 0.2, n)))
        coh.to_csv(os.path.join(d, "cohort.csv"), index=False); feat.to_csv(os.path.join(d, "f.csv"), index=False)
        code, out, err = run("baseline.py", "reference", "--cohort", os.path.join(d, "cohort.csv"), "--features", os.path.join(d, "f.csv"), "--out", os.path.join(d, "ref.json"))
        self.assertEqual(code, 0, err[-500:])
        ref = json.load(open(os.path.join(d, "ref.json")))
        self.assertTrue(ref["features"]["alpha_rel"]["covariates_used"]); self.assertFalse(ref["features"]["tbr"]["covariates_used"])


class TestScores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        sim.write_demo(cls.d)
        cls.b = os.path.join(cls.d, "b.json")
        assert run("baseline.py", "rest", "--baseline", cls.b, "--eeg", os.path.join(cls.d, "rest_eeg.csv"), "--fs", "250", "--eye", os.path.join(cls.d, "rest_eye.csv"))[0] == 0
        for i in (1, 2, 3):
            assert run("baseline.py", "training", "--baseline", cls.b, "--events", os.path.join(cls.d, f"train_{i}_events.csv"), "--eye", os.path.join(cls.d, f"train_{i}_eye.csv"))[0] == 0
        cls.base = json.load(open(cls.b))
        x0, fs, names = sim.synth_eeg("rest", seed=1, dur=120)
        cls.names, cls.fs = names, fs
        cls.profile = json.load(open(os.path.join(cls.d, "profile.json")))

    def passive(self, state, seed=7, baseline=True, reference=None, profile=None):
        x = sim.synth_eeg(state, seed=seed, dur=30)
        e = sim.synth_eye(state, seed=seed, dur=45)
        return simple_score.score_simple(self.base if baseline else None, x, e, None, reference, profile)

    def training(self, state, seed=3, baseline=True, **kw):
        s = sim.simulate_session(state, seed=seed, duration=120)
        task_ = dict(changes=s["changes"], clicks=s["clicks"], ball=None)
        return simple_score.score_simple(self.base if baseline else None, s["eeg"], s["eye"], task_, **kw)

    def test_passive_scores_order_and_scale(self):
        sc = {st: self.passive(st)["simple_score"] for st in ("focused", "rest", "wandering", "tired")}
        self.assertGreater(sc["focused"], sc["rest"] + 5); self.assertGreater(sc["rest"], sc["wandering"] + 5); self.assertGreater(sc["wandering"], sc["tired"])
        self.assertTrue(35 <= sc["rest"] <= 70, sc)                                         # near the user's own level
        r = self.passive("rest")
        self.assertEqual(r["mode"], "passive"); self.assertEqual(set(r["weights_used"]), {"eeg", "eye"})
        self.assertAlmostEqual(r["weights_used"]["eye"], 0.35 / 0.55, places=3)             # click absent -> renormalised base weights

    def test_training_scores_order_and_all_three_modalities(self):
        sc = {st: self.training(st) for st in ("focused", "lapsing", "tired")}
        self.assertGreater(sc["focused"]["simple_score"], sc["lapsing"]["simple_score"] + 10)
        self.assertGreater(sc["lapsing"]["simple_score"], sc["tired"]["simple_score"])
        r = sc["focused"]
        self.assertEqual(r["mode"], "training"); self.assertEqual(r["weights_used"], {"click": 0.45, "eye": 0.35, "eeg": 0.2})
        m = sum(r["weights_used"][k] * r["component_strength_sd"][k] for k in r["weights_used"])     # the printed formula reproduces the score
        self.assertAlmostEqual(scoring.to_score(m), r["simple_score"], delta=0.2)
        self.assertEqual(r["baseline_basis"]["click"], "own training baseline")
        self.assertGreaterEqual(len(r["timeline"]), 3)

    def test_every_modality_is_optional(self):
        s = sim.simulate_session("focused", seed=3, duration=120)
        t_ = dict(changes=s["changes"], clicks=s["clicks"], ball=None)
        no_eeg = simple_score.score_simple(self.base, None, s["eye"], t_)
        self.assertEqual(no_eeg["weights_used"], {"click": round(0.45 / 0.8, 4), "eye": round(0.35 / 0.8, 4)})
        few = simple_score.score_simple(self.base, s["eeg"], s["eye"], dict(changes=s["changes"][:2], clicks=s["clicks"][:2], ball=None))
        self.assertNotIn("click", few["weights_used"]); self.assertIn("too few colour changes", " ".join(few["notes"]))
        poor = s["eye"].copy(); poor.loc[poor.index % 2 == 0, ["gaze_x", "gaze_y"]] = np.nan
        full = simple_score.score_simple(self.base, s["eeg"], s["eye"], t_)["weights_used"]["eye"]
        self.assertLess(simple_score.score_simple(self.base, s["eeg"], poor, t_)["weights_used"]["eye"], full)    # lower quality -> lower weight
        self.assertIn("error", simple_score.score_simple(self.base))

    def test_no_baseline_uses_placeholders_and_skips_eeg(self):
        r = self.passive("rest", baseline=False)
        self.assertNotIn("eeg", r["components"]); self.assertEqual(r["baseline_basis"]["eye"], "population placeholders")
        self.assertIn("no baseline or reference", " ".join(r["notes"]))

    def test_cold_start_reference_simple_vs_complex(self):
        x, fs, names = sim.synth_eeg("rest", seed=1, dur=120)
        feats, _ = eeg.eeg_windows(x, fs, names)
        df = pd.DataFrame(feats)
        ref = dict(schema="focus-reference/2", age_mu=40.0, age_sd=10.0, features={
            f: dict(coef=[float(df[f].mean() + (0.5 if f == "alpha_rel" else 0.0)), -0.3, 0, 0, 0, 0, 0], sd=0.2, plain_mean=float(df[f].mean()),
                    plain_sd=0.2, oof_r2=0.2, covariates_used=True, n=100) for f in ("alpha_rel", "tbr", "beta_rel")})
        young, old = dict(age=22, sex="F"), dict(age=62, sex="F")
        sx = sim.synth_eeg("rest", seed=7, dur=30)
        simple = simple_score.score_simple(None, sx, None, None, ref)
        self.assertEqual(simple["baseline_basis"]["eeg"], "population reference (not age/sex adjusted)")
        cy = complex_score.score_complex(young, None, sx, None, None, ref); co = complex_score.score_complex(old, None, sx, None, None, ref)
        self.assertEqual(cy["simple_score"], simple["simple_score"]); self.assertEqual(co["simple_score"], simple["simple_score"])   # simple never uses age/sex
        self.assertNotEqual(cy["complex_score"], co["complex_score"]); self.assertIn("applied", cy["age_sex_note"])
        self.assertEqual(cy["baseline_basis"]["eeg"], "population reference (not age/sex adjusted)")                             # simple block unchanged

    def test_complex_is_simple_plus_shifts(self):
        for mode in ("passive", "training"):
            if mode == "passive":
                args = (self.base, sim.synth_eeg("wandering", seed=7, dur=30), sim.synth_eye("wandering", seed=7, dur=45), None)
            else:
                s = sim.simulate_session("lapsing", seed=3, duration=120)
                args = (self.base, s["eeg"], s["eye"], dict(changes=s["changes"], clicks=s["clicks"], ball=None))
            simple = simple_score.score_simple(*args)
            self.assertEqual(complex_score.score_complex({}, *args)["complex_score"], simple["simple_score"])            # empty profile: identical
            c = complex_score.score_complex(self.profile, *args)
            self.assertEqual(c["simple_score"], simple["simple_score"]); self.assertEqual(c["components"], simple["components"])
            shifts = context.context_shift(self.profile)
            self.assertAlmostEqual(c["complex_mean_strength_sd"], simple["mean_strength_sd"] + shifts["total"], places=9)
            self.assertAlmostEqual(c["complex_score"], scoring.to_score(simple["mean_strength_sd"] + shifts["total"]), delta=0.06)
            self.assertIn("not applied", c["age_sex_note"])

    def test_context_shifts(self):
        c = lambda **h: context.context_shift(dict(hours_since=h))["total"]                      # noqa: E731
        self.assertGreater(c(caffeine=1), 0); self.assertLess(c(alcohol=2), 0)
        self.assertAlmostEqual(c(caffeine=0), 0.15); self.assertAlmostEqual(c(caffeine=5), 0.075); self.assertLess(c(caffeine=60), 0.001)   # half-life 5 h
        self.assertAlmostEqual(c(alcohol=12), -0.125)
        hi = context.context_shift(dict(asrs_items=[3] * 18))["asrs"]; lo = context.context_shift(dict(asrs_items=[0] * 18))["asrs"]
        self.assertLess(hi, 0); self.assertGreater(lo, 0)
        self.assertEqual(context.context_shift(dict(asrs_items=[3] * 5))["asrs"], 0.0)            # fewer than 12 answered: ignored
        self.assertEqual(context.context_shift(None)["total"], 0.0)

    def test_cli_scripts_agree(self):
        common = ["--baseline", self.b, "--eeg", os.path.join(self.d, "session_eeg.csv"), "--fs", "250",
                  "--eye", os.path.join(self.d, "session_eye.csv"), "--events", os.path.join(self.d, "session_events.csv"), "--no-timeline"]
        s = cli_json("simple_score.py", *common)
        c = cli_json("complex_score.py", "--profile", os.path.join(self.d, "profile.json"), *common)
        self.assertEqual(s["simple_score"], c["simple_score"]); self.assertEqual(s["components"], c["components"])
        self.assertEqual(s["weights_used"], c["weights_used"]); self.assertEqual(s["formula"]["simple_score"], "clip(50 + 25 * m_total, 1, 100)")
        self.assertEqual(c["kind"], "complex"); self.assertIn("complex_score", c)
        p = cli_json("simple_score.py", "--baseline", self.b, "--eeg", os.path.join(self.d, "passive_eeg.csv"), "--fs", "250", "--eye", os.path.join(self.d, "passive_eye.csv"))
        self.assertEqual(p["mode"], "passive"); self.assertNotIn("timeline", p)
        code, out, err = run("simple_score.py")
        self.assertEqual(json.loads(out)["error"], "no input given")


class TestSignalQuality(unittest.TestCase):
    fs, names = 250, ["Fp1", "Fp2", "F3", "F4", "O1", "O2"]

    def test_clean_data_has_no_false_alarms(self):
        flagged = n = 0
        for seed in range(4):
            x, _, _ = sq.make_eeg(self.names, self.fs, 90, seed=seed)
            for s in range(0, x.shape[1] - 4 * self.fs, 2 * self.fs):
                a = sq.assess(x[:, s:s + 4 * self.fs], self.fs, self.names)
                for nm in self.names:
                    n += 1; flagged += bool(a["channels"][nm]["issues"])
        self.assertLess(flagged / n, 0.01)

    def test_faults_detected_and_monitor_prompts(self):
        rng = np.random.default_rng(0)
        n = self.fs * 4; t = np.arange(n) / self.fs
        faults = {"flat": lambda x: 0.05 * rng.normal(size=n), "line": lambda x: x + 30 * np.sin(2 * np.pi * 60 * t),
                  "muscle": lambda x: x + sq.bandpass(rng.normal(size=n), self.fs, 30, 45) * 30, "drift": lambda x: x + 150 * np.sin(2 * np.pi * 0.4 * t)}
        for code, fn in faults.items():
            x, _, _ = sq.make_eeg(self.names, self.fs, 4, seed=3)
            x[2] = fn(x[2])
            self.assertIn(code, sq.assess(x, self.fs, self.names)["channels"]["F3"]["issues"], code)
        x, _, _ = sq.make_eeg(self.names, self.fs, 40, seed=1)
        x[0, 10 * self.fs:30 * self.fs] = 0.05 * rng.normal(size=20 * self.fs)
        mon = sq.QualityMonitor(self.fs, self.names)
        prompts = [(r["t"], p) for k in range(0, x.shape[1], 62) for r in [mon.update(x[:, k:k + 62])] if r for p in r["new_prompts"]]
        self.assertEqual(len(prompts), 1); self.assertEqual(prompts[0][1]["code"], "flat"); self.assertIn("left forehead", prompts[0][1]["text"])

    def test_camera_checks(self):
        blinks = np.arange(3, 57, 4.0)
        x, fs, names = sq.make_eeg(self.names, self.fs, 60, seed=2, blink_times=blinks + 0.08, eyes_closed=[(20, 28), (40, 48)])
        eeg_d = {n: x[i] for i, n in enumerate(names)}
        ear = sq.ear_trace(60, blinks, closed=[(20, 28), (40, 48)])
        b = sq.blink_sync_check(eeg_d, fs, 0.0, ear)
        self.assertAlmostEqual(b["lag_s"], 0.08, delta=0.03); self.assertTrue(b["channels"]["Fp1"]["sees_blinks"])
        self.assertEqual(sq.eyes_closed_alpha_check(eeg_d, fs, 0.0, ear)["status"], "ok")


class TestCameraCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cv2, mediapipe as mp, skimage  # noqa: F401
            if not hasattr(mp, "solutions"):
                raise ImportError
        except ImportError:
            raise unittest.SkipTest("needs mediapipe==0.10.x, opencv, scikit-image")
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import camera_capture
        cls.cc = camera_capture

    def test_landmarks_on_real_face(self):
        import mediapipe as mp
        from skimage import data
        img = data.astronaut(); h, w = img.shape[:2]
        lm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True).process(img).multi_face_landmarks[0].landmark
        r = self.cc.landmarks_to_row(lm, w, h)
        self.assertTrue(0.22 < r["ear"] < 0.45 and abs(r["head_roll_deg"]) < 15, r)


if __name__ == "__main__":
    unittest.main()
