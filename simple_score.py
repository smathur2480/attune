#!/usr/bin/env python3
"""
simple_score.py -- the SIMPLE focus score (1-100) from EEG patterns, camera eye patterns and (in training mode) clicks.

No personal information is used here (no age, sex, substances, questionnaires): the score answers "how focused is this person,
compared with their OWN baseline?"  (50 = their usual level, +/-25 points per standard deviation.)
complex_score.py imports score_simple() from this file and only ADDS personal-context shifts, so the two can never disagree.

MODES (chosen automatically)
  passive  : EEG + camera eye patterns (blinks, eyelid closure, gaze on task, fixations)            -> no task needed
  training : EEG + eye-tracking of the orbiting ball (pursuit) + clicks on colour changes           -> pass --events

THE FORMULA (identical everywhere; implemented once in focus_core/scoring.py, numbers in focus_core/constants.py)
  1. strength   = clip( direction * (value - baseline_mean) / max(baseline_sd, floor), -3, +3 )          per metric, SD units
  2. m_modality = sum(w * strength) / sum(w)                                                           click | eye | eeg
  3. W_k        = BASE_WEIGHTS[k] * quality[k]      (click .45, eye .35, eeg .20; quality 0..1 from the amount of usable data)
     m_total    = sum(W_k * m_k) / sum(W_k)         (a missing or poor modality drops out; the rest renormalise)
  4. simple score = clip( 50 + 25 * m_total, 1, 100 )

BASELINE USED PER MODALITY (built by baseline.py)
  EEG   : baseline["eeg"] (resting windows).  No baseline -> population reference if --reference is given, else EEG is skipped.
  eye   : passive -> baseline["eye_rest"];  training -> baseline["training"].   No baseline -> population placeholders (constants.py).
  click : baseline["training"].   No baseline -> population placeholders.

USAGE
  python simple_score.py --baseline baseline.json --eeg window.csv --fs 250 --eye window_eye.csv                       # passive
  python simple_score.py --baseline baseline.json --eeg s.csv --fs 250 --eye s_eye.csv --events s_events.csv            # training
  as a module:  from simple_score import score_simple;  score_simple(baseline=..., eeg=(x, fs, names), eye=df, task=dict(...))
  Inputs: EEG CSV (header = channel names, microvolts) or EDF/BDF; eye CSV (t, ear, gaze_x, gaze_y, face_ok; tools/camera_capture.py);
  events CSV (type = change|click, t). All files must share ONE clock starting at 0.

LITERATURE  (full references: REFERENCES.md; the short keys are also used in focus_core/constants.py)
  EEG patterns of focused vs mind-wandering states
    [Kam2022]  Kam JWY, Rahnuma T, Park YE, Hart CM (2022). Electrophysiological markers of mind wandering: a systematic review.
               NeuroImage 258. Spectral effects: more delta/theta/alpha and less beta during mind wandering; less robust than ERPs.
    [Groot2021] Groot JM, Boayue NM, Csifcsak G, Boekel W, et al. (2021). Probing the neural signature of mind wandering with
               simultaneous fMRI-EEG and pupillometry. NeuroImage 224:117412. Alpha/theta/delta up, beta unchanged.
    [Jin2019]  Jin CY, Borst JP, van Vugt MK (2019). Predicting task-general mind-wandering with EEG (PMID 30850931). Alpha power
               was the most predictive marker.
    [Arnau2019] Arnau S (2019). The electrophysiological signature of mind wandering. bioRxiv 10.1101/819805 (preprint).
    [vanSon2019] van Son D, et al. (2019). Frontal EEG theta/beta ratio during mind wandering episodes. Biological Psychology
               140:19-27. Frontal TBR higher during mind wandering (n=26).
    [Frontiers2025] Frontiers in Neuroscience (2025) article 1654827 (authors not retrieved); [bioRxiv2023] doi:10.1101/2023.03.21.533634
               (authors not retrieved): alpha up in task-unrelated thought; frontal theta also rises with external task demands.
  Camera eye patterns (blinks, eyelid closure, gaze)
    [VanOrden2000] Van Orden KF, Jung T-P, Makeig S (2000). Combined eye activity measures accurately estimate changes in sustained
               visual task performance. Biological Psychology 52:221-240.
    [Schleicher2008] Schleicher R, Galley N, Briest S, Galley L (2008). Blinks and saccades as indicators of fatigue in sleepiness
               warnings: looking tired? Ergonomics. Blink duration, lid reopening delay, blink interval = best sleepiness indicators.
    [Soukupova2016] Soukupova T, Cech J (2016). Real-time eye blink detection using facial landmarks. CVWW 2016 (eye aspect ratio).
    [GarciaAgundez2018] Garcia-Agundez A, et al. (2018). Eye aspect ratio based blink rate detection... IWBBIO (82%/75% precision/recall).
    [AttentionLapse2023] arXiv:2304.10588. Looking away, long fixation, eye closure, high blink rate as lapse patterns.
  Training task: pursuit and lapses
    [Fransson2008] Fransson P-A, Patel M, Magnusson M (2008). Effects of 24-hour and 36-hour sleep deprivation on smooth pursuit and
               saccadic eye movements. J Vestib Res 18(4). Pursuit gain, accuracy and saccade velocity worsened.
    [Frontiers2022] Frontiers in Neuroscience (2022) 10.3389/fnins.2022.1067722 (authors not retrieved): pursuit gain loss under sleep
               loss and low-dose alcohol.
    [Peiris2006] Peiris et al. (2006) J Sleep Res 15:291-300; [Jones2010] Jones et al. (2010) IEEE EMBS: lapses = flat/incoherent
               tracking > 0.5 s; with eye closure = microsleep, without = attention lapse.
  Training task: clicks
    [PVT2020]  "Response speed measurements on the psychomotor vigilance test: how precise is precise enough?" Sleep 2020
               (PMID 32556295): lapses = RT >= 500 ms.   [QbTest] QbTech documentation: RT variation most sensitive; <150 ms anticipatory.
    [gradCPT2013] doi:10.3758/s13414-012-0413-x: commission errors and RT variability increase with time on task.
    [SART1998] "Circadian rhythm of sleepiness and vigilance studied in man" (1998): commission errors rise with sleepiness.
    [Shelat2024] Shelat S, Schooler JW, Giesbrecht B (2024). Predicting attentional lapses using response time speed in continuous
               performance tasks. Frontiers in Cognition 3:1460349 (within-person response-time boundaries).
CAUTION: these are population-level findings; none is validated for scoring an individual. Weights are judgement calls. Wellness
estimate only -- not a diagnosis or medical advice.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from focus_core import clicks as clicks_mod  # noqa: E402
from focus_core import constants as K  # noqa: E402
from focus_core import eeg as eeg_mod  # noqa: E402
from focus_core import eye as eye_mod  # noqa: E402
from focus_core import fileio as fio  # noqa: E402
from focus_core import pursuit as pursuit_mod  # noqa: E402
from focus_core import scoring  # noqa: E402
from focus_core import task as task_mod  # noqa: E402


def _entries(block):
    return (block or {}).get("features")


def _modalities(baseline, eeg, eye, task, reference, profile, t0=None, t1=None):
    """Metrics -> strengths -> modality means and quality, for the whole input or for the time slice [t0, t1).
    Returns dict modality -> dict(m, quality, strengths, metrics, basis, info) plus a list of notes."""
    baseline = baseline or {}
    out, notes = {}, []
    # ---- EEG -------------------------------------------------------------------------------------------------------
    if eeg is not None:
        x, fs, names = eeg
        seg = x if t0 is None else x[:, int(t0 * fs):int(t1 * fs)]
        s, q, cur = eeg_mod.eeg_strengths(seg, fs, names, baseline=baseline.get("eeg"), reference=reference, profile=profile)
        if s:
            out["eeg"] = dict(m=scoring.modality_mean(s, K.EEG_FEATURES), quality=scoring.quality_eeg(q["n_clean"], q["n_windows"]),
                              strengths=s, metrics=cur, basis=q.get("basis"), info=dict(windows=q["n_windows"], clean=q["n_clean"]))
        elif t0 is None:
            notes.append("no usable EEG: " + ("no baseline or reference to compare with" if q.get("n_clean") else f"no clean windows ({q.get('reasons')})"))
    # ---- eye ---------------------------------------------------------------------------------------------------------
    sub = None
    if eye is not None:
        sub = eye if t0 is None else eye[(eye["t"] >= t0) & (eye["t"] < t1)].reset_index(drop=True)
    if sub is not None and len(sub) > 30:
        if task is not None:
            ball = task.get("ball")
            if ball is None:
                ball = task_mod.ball_frame(eye["t"].to_numpy(float), task.get("period", K.ORBIT_PERIOD_S), task.get("radius", K.ORBIT_RADIUS))
            open_level = (baseline.get("eye_rest") or {}).get("open_level")
            m, q = pursuit_mod.analyze_pursuit(sub, ball, open_level=open_level)
            if m:
                s = scoring.strengths(m, K.eye_table("training"), _entries(baseline.get("training")))
                if s:
                    out["eye"] = dict(m=scoring.modality_mean(s, K.eye_table("training")), quality=scoring.quality_eye(q["valid_frac"]), strengths=s,
                                      metrics=m, basis="own training baseline" if _entries(baseline.get("training")) else "population placeholders",
                                      info=dict(valid_frac=round(q["valid_frac"], 3)))
            elif t0 is None:
                notes.append(f"no pursuit score: {q.get('reason', 'no usable gaze')}")
        else:
            s, q, f = eye_mod.eye_strengths(sub, baseline.get("eye_rest"))
            if s:
                out["eye"] = dict(m=scoring.modality_mean(s, K.eye_table("passive")), quality=scoring.quality_eye(q["valid_frac"]), strengths=s,
                                  metrics={k: float(v) for k, v in f.items() if np.isfinite(v)}, basis=q["basis"],
                                  info=dict(valid_frac=round(q["valid_frac"], 3)))
            elif t0 is None:
                notes.append(f"no eye score: {q.get('reason', 'no usable face data')}")
    # ---- clicks (training only) -----------------------------------------------------------------------------------------
    if task is not None:
        ch = np.asarray(task["changes"], float)
        ck = np.asarray(task["clicks"], float)
        dur = task.get("duration_s") or float(max(ch.max() + 1.5 if len(ch) else 0, ck.max() if len(ck) else 0))
        if t0 is not None:
            ch, ck, dur = ch[(ch >= t0) & (ch < t1)], ck[(ck >= t0) & (ck < t1 + K.RESP_WINDOW_S[1])], t1 - t0
        if len(ch) >= K.CLICK_MIN_TARGETS:
            cm = clicks_mod.analyze_clicks(ch, ck, duration_s=dur)
            s = scoring.strengths(cm, K.CLICK_FEATURES, _entries(baseline.get("training")))
            if s:
                out["click"] = dict(m=scoring.modality_mean(s, K.CLICK_FEATURES), quality=scoring.quality_click(cm["n_targets"]), strengths=s,
                                    metrics={k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in cm.items()},
                                    basis="own training baseline" if _entries(baseline.get("training")) else "population placeholders",
                                    info=dict(n_targets=cm["n_targets"]))
        elif t0 is None:
            notes.append(f"too few colour changes for a click score (need >= {K.CLICK_MIN_TARGETS})")
    return out, notes


def score_simple(baseline=None, eeg=None, eye=None, task=None, reference=None, profile=None, timeline=True):
    """Simple focus score.
    baseline : dict from baseline.json (or None)          eeg : (x[C,T] uV, fs, channel names) or None
    eye      : DataFrame t, ear, gaze_x, gaze_y, face_ok   task: None (passive) or dict(changes=[s], clicks=[s], ball=DataFrame|None,
               period=8.0, radius=0.3, duration_s=None) for a training session
    reference: population EEG reference (used only if there is no personal EEG baseline)
    profile  : ONLY used to pick the age/sex-matched reference when there is no EEG baseline; complex_score.py passes it, the
               simple score never does (profile=None).
    """
    if eeg is None and eye is None and task is None:
        return dict(schema=K.SCHEMA_SCORE, kind="simple", error="no input given", notes=[K.DISCLAIMER])
    mode = "training" if task is not None else "passive"
    parts, notes = _modalities(baseline, eeg, eye, task, reference, profile)
    mods = {k: v["m"] for k, v in parts.items()}
    qual = {k: v["quality"] for k, v in parts.items()}
    m_total, weights = scoring.combine(mods, qual)
    if m_total is None:
        return dict(schema=K.SCHEMA_SCORE, kind="simple", mode=mode, error="no usable signals", notes=notes + [K.DISCLAIMER])
    present, against = [], []
    for k, v in parts.items():
        table = K.eye_table("training" if mode == "training" else "passive") if k == "eye" else K.TABLES[k]
        p, a = scoring.explain(v["strengths"], table)
        present += p
        against += a
    if len(mods) < 3:
        notes.append("modalities used: " + ", ".join(sorted(mods)) + (" (click needs a training session)" if "click" not in mods and mode == "passive" else ""))
    if len(mods) > 1 and max(mods.values()) - min(mods.values()) > 1.5:
        notes.append("components disagree by more than 1.5 SD: read the sub-scores, not just the total")
    result = dict(schema=K.SCHEMA_SCORE, kind="simple", mode=mode, simple_score=round(scoring.to_score(m_total), 1),
                  mean_strength_sd=float(m_total),
                  components={k: round(scoring.to_score(m), 1) for k, m in mods.items()},
                  component_strength_sd={k: round(float(m), 3) for k, m in mods.items()},
                  weights_used=weights, quality={k: round(q, 3) for k, q in qual.items()},
                  strengths={k: {a: round(b, 2) for a, b in v["strengths"].items()} for k, v in parts.items()},
                  metrics={k: {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v["metrics"].items()} for k, v in parts.items()},
                  baseline_basis={k: v["basis"] for k, v in parts.items()}, patterns_present=present,
                  patterns_absent_or_reversed=against, formula=scoring.formula_description(), notes=notes)
    if mode == "training" and timeline and eye is not None:
        dur = float(eye["t"].max())
        tl = []
        for t0 in np.arange(0, dur - 14.0, K.TIMELINE_WINDOW_S):
            wp, _ = _modalities(baseline, eeg, eye, task, reference, profile, t0, min(t0 + K.TIMELINE_WINDOW_S, dur))
            wm, wq = {k: v["m"] for k, v in wp.items()}, {k: v["quality"] for k, v in wp.items()}
            wt, _w = scoring.combine(wm, wq)
            if wt is not None:
                tl.append(dict(t_start=float(t0), score=round(scoring.to_score(wt), 1), components={k: round(scoring.to_score(m), 1) for k, m in wm.items()}))
        result["timeline"] = tl
    result["notes"].append(K.DISCLAIMER)
    return result


def load_inputs(a):
    """Shared CLI input handling for simple_score.py and complex_score.py."""
    baseline = fio.read_json(a.baseline)
    reference = fio.read_json(a.reference)
    eeg = fio.load_eeg(a.eeg, a.fs) if a.eeg else None
    eye = fio.load_eye(a.eye) if a.eye else None
    task = None
    if a.events:
        changes, clicks = fio.load_events(a.events)
        task = dict(changes=changes, clicks=clicks, ball=(fio.read_csv(a.ball) if getattr(a, "ball", None) else None),
                    period=a.orbit_period, radius=a.orbit_radius)
    return baseline, eeg, eye, task, reference


def add_input_args(ap):
    ap.add_argument("--baseline", help="baseline.json from baseline.py")
    ap.add_argument("--reference", help="reference.json (population EEG norms) for users without an EEG baseline")
    ap.add_argument("--eeg", help="EEG segment (CSV with channel-name header in uV, or EDF/BDF)")
    ap.add_argument("--fs", type=float, help="EEG sampling rate in Hz (CSV input)")
    ap.add_argument("--eye", help="camera CSV: t, ear, gaze_x, gaze_y, face_ok")
    ap.add_argument("--events", help="training session events CSV: type (change|click), t  -> training mode")
    ap.add_argument("--ball", help="optional ball trajectory CSV (t, bx, by) if the ball does not follow the default circular orbit")
    ap.add_argument("--orbit-period", type=float, default=K.ORBIT_PERIOD_S)
    ap.add_argument("--orbit-radius", type=float, default=K.ORBIT_RADIUS)
    ap.add_argument("--no-timeline", action="store_true")
    ap.add_argument("--out", help="also write the JSON result to this file")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_args(ap)
    a = ap.parse_args(argv)
    baseline, eeg, eye, task, reference = load_inputs(a)
    res = score_simple(baseline, eeg, eye, task, reference, timeline=not a.no_timeline)
    text = json.dumps(res, indent=1, default=float)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
