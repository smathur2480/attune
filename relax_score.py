#!/usr/bin/env python3
"""
relax_score.py -- RELAXATION / MEDITATION score (1-100) from EEG (alpha, theta) and, as a PLANNED feature, heart-rate variability from a
wearable watch (ECG strip, RR intervals or PPG pulse intervals). Same formula and conventions as simple_score.py (focus): the score says
"how relaxed are you compared with YOUR usual level", 50 = usual, +/-25 points per standard deviation.

THE FORMULA (constants in focus_core/constants.py; implemented in focus_core/scoring.py -- the same functions as the focus scores)
  1. strength   = clip( direction * (value - baseline_mean) / max(baseline_sd, floor), -3, +3 )       per metric, SD units, + = more relaxed
  2. m_modality = sum(w * strength) / sum(w)                                                          eeg | heart
  3. W_k        = RELAX_BASE_WEIGHTS[k] * quality[k]        (eeg .5, heart .5; a missing/poor modality drops out, the rest renormalise)
     m_total    = sum(W_k * m_k) / sum(W_k)
  4. m_adj      = m_total - 1.5   if the DROWSINESS GUARD fires, else m_total
     relax score = clip( 50 + 25 * m_adj, 1, 100 )

THE RULES FLOW (every step is recorded in the output under "flow")
  0 INPUT GATE   need >= 1 usable modality (EEG with a MATCHING-condition baseline, or heart with a heart baseline); else no score.
  1 CONDITION    eyes closed or open? (camera: share of frames with eyelid-aspect-ratio < 20% of YOUR open level; or declared with --condition).
                 Closing the eyes raises alpha by itself (Berger effect), so EEG is compared with the matching baseline:
                 eyes closed -> baseline["eeg_closed"], eyes open -> baseline["eeg"]. No matching baseline -> EEG is NOT scored.
  2 QUALITY      EEG: share of clean windows; muscle-contaminated windows are counted (jaw/forehead tension is the opposite of relaxing).
                 Heart: valid-beat share, >= 30 s, PPG discounted x0.7.
  3 PATTERNS     EEG: alpha UP (weight 1.0), frontal theta UP (0.5) vs the eyes-matched baseline.
                 (theta is measured against NON-alpha power, so a fall in alpha alone cannot raise it.)
                 Heart: RMSSD UP (1.0), heart rate DOWN (0.8), HF power UP (0.5, only if breathing >= 9/min and segment >= 2 min).
  4 DROWSINESS   relaxed-alert is not falling asleep. FIRES if
     GUARD         eyes closed: frontal theta strength >= +1 SD AND alpha strength < +0.5 SD   (alpha not elevated while theta rises)
                   eyes open  : eyelid closure (PERCLOS) >= +1 SD above your resting level OR blink duration >= +1.5 SD.
                   Relaxed alertness raises alpha AND theta together; drowsiness lowers alpha as theta rises.
  5 STATE        drowsy (guard)  |  deeply relaxed candidate: score >= 70 and alpha >= +1 SD (must hold >= 2 min: focus_core.trigger.sustained)
                 |  relaxed: >= 58  |  your usual level: >= 42  |  tense / aroused: below 42.
  6 SUGGESTION   slow breathing (~6/min) is the only literature-backed suggestion; others are marked as UX suggestions without evidence.

USAGE
  python baseline.py rest --baseline baseline.json --eeg rest_closed.csv --fs 250 --condition eyes_closed     # once: eyes-closed baseline
  python baseline.py rest --baseline baseline.json --eeg rest_eeg.csv --fs 250 --eye rest_eye.csv            # eyes-open baseline + camera
  python baseline.py rest --baseline baseline.json --heart rr.csv                                            # PLANNED: watch HRV baseline
  python relax_score.py --baseline baseline.json --eeg session.csv --fs 250 --eye session_eye.csv [--heart rr.csv | --heart-ecg ecg.csv]
  Use >= 60 s of EEG (>= 2 min if you want the heart's breathing-rate rule to apply). Wellness estimate only -- not a diagnosis.

LITERATURE  (full references: REFERENCES.md; keys also in focus_core/constants.py)
  EEG of relaxation / meditation
    [Lomas2015]  Lomas T, Ivtzan I, Fu CHY (2015). A systematic review of the neurophysiology of mindfulness on EEG oscillations.
                 Neurosci Biobehav Rev (56 papers, 1715 subjects): mindfulness was most commonly associated with enhanced alpha AND theta power
                 compared with an EYES-CLOSED resting state (not uniformly), possibly signifying relaxed alertness.
    [Cahn2006]   Cahn BR, Polich J (2006). Meditation states and traits: EEG, ERP, and neuroimaging studies. Psychol Bull 132(2):180-211:
                 overall EEG slowing, theta and alpha activation related to proficiency of practice; effects variable.
    [Meditation2018] Front Neurosci 2018;12:178, doi:10.3389/fnins.2018.00178, "Review of the neural oscillations underlying meditation"
                 (Goodarzi A, Girgis F, et al.; full author list not retrieved): focused-attention and open-monitoring meditation increase anterior
                 theta; alpha rises in posterior regions in both; anterior alpha differs by practice.
    [AlphaThetaDepth] "Alpha and theta brain oscillations play a complementary role in inducing levels of altered consciousness during meditation
                 practice" (authors/venue not retrieved): alpha and frontal-midline theta are the bands most robustly linked with meditation;
                 alpha amplitude followed meditation depth while theta deactivated as it deepened (=> theta gets a lower weight).
  Relaxed vs drowsy
    [Chokroverty] Chokroverty S, Sleep Disorders Medicine, ch. 18: alpha cessation defines the wake-to-sleep transition; slow rolling eye
                 movements appear near sleep onset.
    [Magosso2007] Magosso E, Ursino M, Provini F, Montagna P (2007). Wavelet analysis of electroencephalographic and electro-oculographic
                 changes during the sleep onset period (15 healthy subjects): slow eye movements preceded alpha blocking and theta intrusion by
                 10-20 min (a planned early-warning feature: slow eye movements from the Fp1-Fp2 pair; not implemented).
    [Strauss2022] Strauss M, Sitt JD, Naccache L, Raimondo F (2022). NeuroImage 251:119003 (title not retrieved): drowsiness is a distinct period
                 between wakefulness and sleep marked by alpha and theta changes that track falling responsiveness.
    [Schleicher2008] Schleicher R, et al. (2008). Blinks and saccades as indicators of fatigue in sleepiness warnings. Ergonomics: blink duration,
                 lid reopening delay, blink interval were the best sleepiness indicators (basis of the eyes-open guard).
    [Patent5813993] US patent 5,813,993 (weak source): with eyes closed sleepiness reduces alpha and increases theta; with eyes open increased
                 alpha/theta can accompany sleepiness (hence the camera-based guard when the eyes are open).
  Heart (PLANNED) -- full list and caveats in focus_core/heart.py
    [Laborde2022] Laborde S, Allen MS, Borges U, et al. (2022). Effects of voluntary slow breathing on heart rate and heart rate variability: a
                 systematic review and meta-analysis (223 studies).   [Lehrer2014] Lehrer PM, Gevirtz R (2014). Heart rate variability
                 biofeedback: how and why does it work? Front Psychol.   [SevozCouche2022] Sevoz-Couche C, Laborde S (2022). HRV and slow-paced
                 breathing: when coherence meets resonance. Neurosci Biobehav Rev (breathing at ~0.1 Hz increases vagally mediated HRV).
CAUTION: literature effects are population-level and variable; thresholds and weights are untuned placeholders; EEG alpha/theta are not specific to
relaxation. The heart modality has NOT been tested with a real watch. Not a diagnosis, not a treatment, not medical advice.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from focus_core import constants as K  # noqa: E402
from focus_core import eeg as eeg_mod  # noqa: E402
from focus_core import eye as eye_mod  # noqa: E402
from focus_core import fileio as fio  # noqa: E402
from focus_core import heart as heart_mod  # noqa: E402
from focus_core import scoring  # noqa: E402


def eyes_condition(eye, baseline, condition="auto"):
    """Step 1. Returns (condition 'eyes_open'|'eyes_closed'|None, info dict)."""
    if condition in ("eyes_open", "eyes_closed"):
        return condition, dict(source="declared")
    if eye is None:
        return None, dict(source="unknown", reason="no camera data and no --condition given")
    open_level = ((baseline or {}).get("eye_rest") or {}).get("open_level") or K.EAR_OPEN_DEFAULT
    ear = eye["ear"].to_numpy(float)
    ok = np.isfinite(ear) & ((eye["face_ok"].to_numpy(float) > 0) if "face_ok" in eye else True)
    if ok.sum() < 30:
        return None, dict(source="camera", reason="face not visible")
    frac = float(np.mean(ear[ok] < K.PERCLOS_FRAC * open_level))
    info = dict(source="camera", closed_fraction=round(frac, 3), open_level=round(float(open_level), 3),
                open_level_from="own baseline" if (baseline or {}).get("eye_rest") else "default")
    if frac >= K.EYES_CLOSED_FRAC:
        return "eyes_closed", info
    if frac <= K.EYES_OPEN_FRAC:
        return "eyes_open", info
    info["mixed"] = True
    return "eyes_open", info


def _eeg_part(baseline, eeg, cond, notes, flow):
    block_name = "eeg_closed" if cond == "eyes_closed" else "eeg"
    block = (baseline or {}).get(block_name)
    if eeg is None:
        return None
    if cond is None:
        notes.append("EEG not scored: eyes-open/closed unknown (give camera data or --condition)")
        return None
    if not block:
        hint = ("python baseline.py rest --eeg ... --fs ... --condition eyes_closed" if cond == "eyes_closed"
                else "python baseline.py rest --eeg ... --fs ...")
        notes.append(f"EEG not scored: no {cond.replace('_', ' ')} baseline ({block_name}). Record one first: {hint}")
        return None
    x, fs, names = eeg
    s, q, cur = eeg_mod.eeg_strengths(x, fs, names, baseline=block, table=K.RELAX_EEG_FEATURES)
    if not s:
        notes.append(f"EEG not scored: no usable windows ({q.get('reasons')})")
        return None
    emg = q.get("reasons", {}).get("emg", 0) / max(q.get("n_windows", 1), 1)
    if emg > 0.3:
        notes.append(f"muscle activity in {emg:.0%} of EEG windows (jaw / forehead tension): try to relax your face; this is the opposite of relaxing")
    return dict(m=scoring.modality_mean(s, K.RELAX_EEG_FEATURES), quality=scoring.quality_eeg(q["n_clean"], q["n_windows"]), strengths=s,
                metrics=cur, basis=f"own {cond.replace('_', ' ')} baseline", info=dict(windows=q["n_windows"], clean=q["n_clean"], emg_share=round(emg, 3)))


def _heart_part(baseline, heart, notes):
    if heart is None:
        return None
    t_end, rr, src = heart_mod.from_arrays(**heart)
    feats, hq = heart_mod.hrv_features(t_end, rr, src)
    entries = ((baseline or {}).get("heart_rest") or {}).get("features")
    if not feats:
        notes.append(f"heart not scored: {hq.get('reason')}")
        return None
    if not entries:
        notes.append("heart not scored: no heart baseline (python baseline.py rest --heart rr.csv, >= ~3 min seated rest)")
        return None
    s = scoring.strengths(feats, K.HEART_FEATURES, entries)
    if not s:
        notes.append("heart not scored: no baseline entry matches the available features")
        return None
    return dict(m=scoring.modality_mean(s, K.HEART_FEATURES), quality=heart_mod.quality_heart(hq), strengths=s,
                metrics={k: round(float(v), 3) for k, v in feats.items()}, basis="own heart baseline", info=hq)


def _drowsy_guard(cond, eeg_part, eye, baseline):
    """Step 4. Returns (fired bool, reasons list, details dict)."""
    reasons, det = [], {}
    if cond == "eyes_closed" and eeg_part:
        a, t = eeg_part["strengths"].get("alpha_rel"), eeg_part["strengths"].get("theta_ex")
        det = dict(alpha_strength=a, theta_strength=t)
        if a is not None and t is not None and t >= K.DROWSY_THETA_MIN_SD and a < K.DROWSY_ALPHA_BELOW_SD:
            reasons.append(f"theta is {t:+.1f} SD while alpha is only {a:+.1f} SD (relaxed alertness raises both; drowsiness lowers alpha as theta rises)")
    elif cond == "eyes_open" and eye is not None:
        b = (baseline or {}).get("eye_rest") or {}
        feats, _q = eye_mod.eye_features(eye, open_level=b.get("open_level"))
        if feats:
            ent = (b.get("features") or {})
            z = {}
            for name, thr in (("perclos", K.DROWSY_PERCLOS_MIN_SD), ("blink_dur_ms", K.DROWSY_BLINK_DUR_MIN_SD)):
                spec = K.EYE_FEATURES[name]
                if name in feats and np.isfinite(feats[name]):
                    mean, sd = (ent[name]["mean"], ent[name]["sd"]) if name in ent else (spec["mean"], spec["sd"])
                    z[name] = (feats[name] - mean) / max(sd, spec["floor"])
                    if z[name] >= thr:
                        reasons.append(f"{name} is {z[name]:+.1f} SD above your resting level")
            det = dict(z=z)
    return bool(reasons), reasons, det


STATE_TEXT = {"drowsy": "drowsy: drifting toward sleep rather than relaxing", "deeply_relaxed": "deeply relaxed / meditative (candidate)",
              "relaxed": "relaxed", "usual": "about your usual level", "tense": "tense or aroused"}


def score_relax(baseline=None, eeg=None, eye=None, heart=None, condition="auto"):
    """eeg: (x[C,T] uV, fs, names) | None;  eye: camera DataFrame | None;  heart: dict(t_end, rr_ms, source) or dict(ecg, fs) | None (PLANNED)."""
    flow, notes = [], []
    if eeg is None and heart is None:
        return dict(schema=K.SCHEMA_RELAX, kind="relax", error="no EEG or heart input", flow=flow, notes=[K.DISCLAIMER])
    cond, cinfo = eyes_condition(eye, baseline, condition)
    flow.append(dict(step=1, name="condition", result=cond or "unknown", detail=cinfo))
    parts = {}
    for name, part in (("eeg", _eeg_part(baseline, eeg, cond, notes, flow)), ("heart", _heart_part(baseline, heart, notes))):
        if part:
            parts[name] = part
    flow.insert(0, dict(step=0, name="input_gate", result="pass" if parts else "fail", detail=dict(usable=sorted(parts))))
    if not parts:
        return dict(schema=K.SCHEMA_RELAX, kind="relax", condition=cond, error="no usable modality", flow=flow, notes=notes + [K.DISCLAIMER])
    flow.append(dict(step=2, name="quality", result="ok", detail={k: round(v["quality"], 3) for k, v in parts.items()}))
    mods, qual = {k: v["m"] for k, v in parts.items()}, {k: v["quality"] for k, v in parts.items()}
    m_total, weights = scoring.combine(mods, qual, K.RELAX_BASE_WEIGHTS)
    if m_total is None:
        return dict(schema=K.SCHEMA_RELAX, kind="relax", condition=cond, error="all modalities had zero data quality", flow=flow, notes=notes + [K.DISCLAIMER])
    flow.append(dict(step=3, name="patterns", result="scored", detail=dict(m_total=round(float(m_total), 3), weights=weights)))
    fired, why, gdet = _drowsy_guard(cond, parts.get("eeg"), eye, baseline)
    m_adj = m_total - (K.DROWSY_PENALTY_SD if fired else 0.0)
    flow.append(dict(step=4, name="drowsiness_guard", result="FIRED" if fired else "clear", detail=dict(reasons=why, penalty_sd=K.DROWSY_PENALTY_SD if fired else 0.0)))
    score = round(scoring.to_score(m_adj), 1)
    alpha_s = (parts.get("eeg") or {}).get("strengths", {}).get("alpha_rel")
    deep = (not fired) and score >= K.RELAX_DEEP_SCORE and (alpha_s is None or alpha_s >= K.RELAX_DEEP_ALPHA_SD)
    state = ("drowsy" if fired else "relaxed" if deep else "relaxed" if score >= K.RELAX_RELAXED_SCORE
             else "usual" if score >= K.RELAX_USUAL_SCORE else "tense")
    flow.append(dict(step=5, name="state", result=state, detail=dict(score=score, deep_candidate=bool(deep),
                                                                      note="deeply relaxed only if the candidate holds >= %d s" % K.RELAX_DEEP_SECONDS)))
    present, against = [], []
    for k, v in parts.items():
        p, a = scoring.explain(v["strengths"], K.RELAX_TABLES[k])
        present += p
        against += a
    sugg = []
    hres = (parts.get("heart") or {}).get("metrics", {}).get("resp_cpm")
    if state in ("usual", "tense") and hres is not None and hres >= K.HF_MIN_BREATH_CPM:
        sugg.append(dict(text="Try slow breathing at about 6 breaths per minute for a few minutes.", evidence="[Laborde2022] [Lehrer2014] [SevozCouche2022]"))
    if state == "drowsy":
        sugg.append(dict(text="You may be drifting toward sleep. If you want to stay awake, open your eyes, sit up or take a short break.",
                         evidence="UX suggestion, not from the cited literature"))
    if state == "tense" and (parts.get("eeg") or {}).get("info", {}).get("emg_share", 0) > 0.3:
        sugg.append(dict(text="Soften your jaw, forehead and shoulders.", evidence="UX suggestion; muscle activity was detected in the EEG"))
    flow.append(dict(step=6, name="suggestion", result=f"{len(sugg)} suggestion(s)", detail={}))
    if "heart" not in parts:
        notes.append("heart (watch HRV) not used -- planned feature; the score currently rests on EEG alone" if heart is None else "heart data given but not usable (see above)")
    if cinfo.get("mixed"):
        notes.append("eyes partly closed: scored against the eyes-open baseline; alpha may be inflated by eyelid closure")
    if len(mods) > 1 and max(mods.values()) - min(mods.values()) > 1.5:
        notes.append("EEG and heart disagree by more than 1.5 SD: read the sub-scores, not just the total")
    notes.append(K.DISCLAIMER)
    return dict(schema=K.SCHEMA_RELAX, kind="relax", condition=cond, relax_score=score, state=state, state_text=STATE_TEXT[state],
                deep_candidate=bool(deep), drowsy=fired, drowsy_reasons=why, mean_strength_sd=float(m_total), adjusted_strength_sd=float(m_adj),
                components={k: round(scoring.to_score(m), 1) for k, m in mods.items()}, component_strength_sd={k: round(float(m), 3) for k, m in mods.items()},
                weights_used=weights, quality={k: round(q, 3) for k, q in qual.items()},
                strengths={k: {a: round(b, 2) for a, b in v["strengths"].items()} for k, v in parts.items()},
                metrics={k: {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v["metrics"].items()} for k, v in parts.items()},
                baseline_basis={k: v["basis"] for k, v in parts.items()}, patterns_present=present, patterns_absent_or_reversed=against,
                suggestions=sugg, flow=flow, formula=scoring.formula_description(), notes=notes)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", help="baseline.json from baseline.py (needs eeg_closed and/or eeg; heart_rest for the heart part)")
    ap.add_argument("--eeg"); ap.add_argument("--fs", type=float); ap.add_argument("--eye", help="camera CSV: t, ear, [gaze_x, gaze_y, face_ok]")
    ap.add_argument("--heart", help="watch RR intervals CSV (t, rr_ms) -- PLANNED feature")
    ap.add_argument("--heart-ecg", help="watch ECG strip CSV (column ecg)"); ap.add_argument("--heart-fs", type=float, default=250.0)
    ap.add_argument("--heart-source", choices=["ecg", "ppg"], default="ecg")
    ap.add_argument("--condition", choices=["auto", "eyes_open", "eyes_closed"], default="auto")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    baseline = fio.read_json(a.baseline)
    eeg = fio.load_eeg(a.eeg, a.fs) if a.eeg else None
    eye = fio.load_eye(a.eye) if a.eye else None
    heart = dict(fio.load_rr(a.heart), source=a.heart_source) if a.heart else (fio.load_ecg(a.heart_ecg, a.heart_fs) if a.heart_ecg else None)
    res = score_relax(baseline, eeg, eye, heart, a.condition)
    text = json.dumps(res, indent=1, default=float)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
