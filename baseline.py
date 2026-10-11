#!/usr/bin/env python3
"""
baseline.py -- build and update a user's PERSONAL BASELINE (baseline.json) from EEG, camera eye movement and task clicks.
Everything in simple_score.py / complex_score.py is measured against this file (50 = "your usual level").

SUBCOMMANDS
  rest      resting calibration (about 2 min EEG eyes open looking at the screen + the camera clip of the same period)
              -> EEG baseline (per-feature mean/SD over clean 8-s windows) and passive eye baseline (blink rate/duration, eyelid
                 closure, on-task gaze region, ...). Bad EEG channels are screened out first (focus_core/signal_quality.py).
  training  add one ordinary training session (orbiting ball + colour-change clicks)
              -> click metrics (hit rate, median RT, RT variability, lapse rate, false alarms) and eye-pursuit metrics
                 (pursuit correlation, lag, tracking error, tracking-lapse time, eyelid closure)
  reference (developer tool) fit the POPULATION EEG reference used for users with no EEG baseline yet, from TDBRAIN-derived features
  show      print what a baseline contains and how trustworthy each part is

RULES (identical for every modality; see focus_core/baselines.py)
  mean : EEG = mean over all clean windows of all rest sessions;  eye / click = mean of the per-session values
  SD   : EEG = SD over all clean windows of all rest sessions (pooled: within + between session)
         eye / click = BETWEEN-SESSION SD once >= 3 sessions exist, until then the PLACEHOLDER SD from focus_core/constants.py
  Scores then use  strength = clip(direction * (value - mean) / max(sd, floor), -3, +3)  with these means and SDs.
  Run `rest` again on other days to improve the EEG/eye baseline; run `training` at least 3 times to get personal click/pursuit SDs.
  Only ORDINARY sessions should be added (rested, normal day): the baseline is "your usual", not your best.

USAGE
  python baseline.py rest --baseline baseline.json --eeg rest_eeg.csv --fs 250 --eye rest_eye.csv
  python baseline.py training --baseline baseline.json --events train_1_events.csv --eye train_1_eye.csv
  python baseline.py show --baseline baseline.json
  python baseline.py reference --cohort cohort.csv --features eeg_front_restEO.csv --out reference.json      # developers, TDBRAIN data

LITERATURE
  Why a PERSONAL baseline
    [Schleicher2008] Schleicher R, Galley N, Briest S, Galley L (2008). Blinks and saccades as indicators of fatigue in sleepiness
        warnings: looking tired? Ergonomics. Inter-individual differences in blink/saccade variables were substantial.
    [VanOrden2000] Van Orden KF, Jung T-P, Makeig S (2000). Combined eye activity measures accurately estimate changes in sustained
        visual task performance. Biological Psychology 52:221-240. Individual models built in one session predicted performance in
        a second session on another day.
    [Shelat2024] Shelat S, Schooler JW, Giesbrecht B (2024). Predicting attentional lapses using response time speed in continuous
        performance tasks. Frontiers in Cognition 3:1460349. Within-person response-time boundaries are used to infer attentional state.
  Resting calibration and signal quality
    [Casadei2022] Casadei V, Ferrero R (2022). Online skin-electrode contact quality monitoring in wearable devices: an EEG
        application. I2MTC 2022 (mains interference relative to a reference channel tracks contact quality).
    [Sensors2019] "Quality assessment of single-channel EEG for wearable devices", Sensors 19(3):601, doi:10.3390/s19030601 (authors
        not retrieved).
    [vanDijk2022] van Dijk H, van Wingen G, Denys D, Olbrich S, van Ruth R, Arns M (2022). The two decades brainclinics research
        archive for insights in neurophysiology (TDBRAIN) database. Scientific Data 9:333 (resting-state eyes-open/closed EEG; data
        quality validated by alpha attenuation on eye opening).
    [Arns2013] Arns M, Conners CK, Kraemer HC (2013). A decade of EEG theta/beta ratio research in ADHD: a meta-analysis. J Atten
        Disord 17(5):374-383. TBR is not a reliable diagnostic -> it is used here only RELATIVE to the person's own resting level.
  Task metrics that are baselined
    [PVT2020] "Response speed measurements on the psychomotor vigilance test: how precise is precise enough?" Sleep (2020), PMID
        32556295: lapses = RT >= 500 ms; keep click timing within a few ms.   [QbTest] QbTech documentation: RT variation is the most
        sensitive attention parameter; responses < 150 ms are anticipatory.   [gradCPT2013] doi:10.3758/s13414-012-0413-x.
    [Fransson2008] Fransson P-A, Patel M, Magnusson M (2008). Effects of 24-hour and 36-hour sleep deprivation on smooth pursuit and
        saccadic eye movements. J Vestib Res 18(4).   [Peiris2006] J Sleep Res 15:291-300 and [Jones2010] IEEE EMBS 2010: lapses of
        responsiveness in continuous tracking (flat/incoherent tracking > 0.5 s).
CAUTION: a baseline is only as good as the sessions in it (sensor contact, lighting, camera position). Wellness use only.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from focus_core import baseline_ops, baselines  # noqa: E402
from focus_core import constants as K  # noqa: E402
from focus_core import fileio as fio  # noqa: E402
from focus_core import reference as ref_mod  # noqa: E402


def load_or_new(path, must_exist=False):
    if path and os.path.exists(path):
        b = fio.read_json(path)
        if b.get("schema") != K.SCHEMA_BASELINE:
            raise SystemExit(f"{path} has schema {b.get('schema')!r}, expected {K.SCHEMA_BASELINE!r}")
        return b
    if must_exist:
        raise SystemExit(f"{path} does not exist; run `baseline.py rest` first")
    return baselines.new_baseline()


def cmd_rest(a):
    b = load_or_new(a.baseline)
    if not (a.eeg or a.eye or a.heart or a.heart_ecg):
        raise SystemExit("give --eeg and/or --eye and/or --heart / --heart-ecg")
    eeg = fio.load_eeg(a.eeg, a.fs, a.channels.split(",") if a.channels else None) if a.eeg else None
    eye = fio.load_eye(a.eye) if a.eye else None
    heart = None
    if a.heart:
        heart = dict(fio.load_rr(a.heart), source=a.heart_source)
    elif a.heart_ecg:
        heart = fio.load_ecg(a.heart_ecg, a.heart_fs)
    try:
        b, notes = baseline_ops.add_rest_session(b, eeg, eye, reset=a.reset, condition=a.condition, heart=heart)
    except ValueError as e:
        raise SystemExit(str(e))
    fio.write_json(b, a.baseline)
    print("\n".join(notes))
    print(f"wrote {a.baseline}")


def cmd_training(a):
    b = load_or_new(a.baseline)
    changes, clicks = fio.load_events(a.events)
    eye = fio.load_eye(a.eye)
    ball = fio.read_csv(a.ball) if a.ball else None
    try:
        b, values, n = baseline_ops.add_training_session(b, changes, clicks, eye, ball, a.orbit_period, a.orbit_radius, force=a.force)
    except ValueError as e:
        raise SystemExit(f"{e}  (use --force to add anyway)" if str(e).startswith("session not added") else str(e))
    fio.write_json(b, a.baseline)
    print(f"added training session {n}: " + ", ".join(f"{k}={v:.3g}" for k, v in values.items()))
    print(f"personal SDs: {'active' if n >= K.MIN_SESSIONS_PERSONAL_SD else f'not yet (placeholder SDs until {K.MIN_SESSIONS_PERSONAL_SD} sessions)'}")
    print(f"wrote {a.baseline}")


def cmd_reference(a):
    ref, log = ref_mod.fit_reference(a.cohort, a.features, a.ref, a.min_r2, a.seed)
    print("\n".join(log))
    fio.write_json(ref, a.out)
    print(f"wrote {a.out} (aggregate parameters only; check your data-use agreement before committing it)")


def cmd_show(a):
    print("\n".join(baselines.summary(load_or_new(a.baseline, must_exist=True))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("rest", help="resting EEG + eye calibration")
    r.add_argument("--baseline", default="baseline.json"); r.add_argument("--eeg"); r.add_argument("--fs", type=float)
    r.add_argument("--channels", help="comma-separated channel names to use (default: all frontal/posterior 10-20 channels present)")
    r.add_argument("--eye"); r.add_argument("--reset", action="store_true", help="discard earlier rest sessions")
    r.add_argument("--condition", choices=["eyes_open", "eyes_closed"], default="eyes_open",
                   help="eyes_closed stores the EEG as the eyes-closed baseline used by the relaxation score")
    r.add_argument("--heart", help="watch RR intervals CSV (t, rr_ms): PLANNED feature, >= ~3 min of rest")
    r.add_argument("--heart-ecg", help="watch ECG strip CSV (column ecg) instead of RR intervals"); r.add_argument("--heart-fs", type=float, default=250.0)
    r.add_argument("--heart-source", choices=["ecg", "ppg"], default="ecg")
    t = sp.add_parser("training", help="add a training session")
    t.add_argument("--baseline", default="baseline.json"); t.add_argument("--events", required=True); t.add_argument("--eye", required=True)
    t.add_argument("--ball"); t.add_argument("--orbit-period", type=float, default=K.ORBIT_PERIOD_S)
    t.add_argument("--orbit-radius", type=float, default=K.ORBIT_RADIUS); t.add_argument("--force", action="store_true")
    f = sp.add_parser("reference", help="population EEG reference from TDBRAIN-derived features")
    f.add_argument("--cohort", required=True); f.add_argument("--features", required=True); f.add_argument("--ref", default="HEALTHY")
    f.add_argument("--min-r2", type=float, default=0.0); f.add_argument("--seed", type=int, default=0); f.add_argument("--out", default="reference.json")
    s = sp.add_parser("show"); s.add_argument("--baseline", default="baseline.json")
    a = ap.parse_args(argv)
    {"rest": cmd_rest, "training": cmd_training, "reference": cmd_reference, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
