"""
focus_core.baseline_ops -- baseline building as plain functions (used by baseline.py's CLI AND by the real-time engine/server).

Same rules and literature as baseline.py (see its header); nothing here is new maths. Every function takes a baseline dict and returns the
updated dict, so the caller decides where to store it (file, database). Errors are ValueError with a user-readable message.
"""
import numpy as np

from . import baselines
from . import clicks as clicks_mod
from . import constants as K
from . import eeg as eeg_mod
from . import eye as eye_mod
from . import heart as heart_mod
from . import pursuit as pursuit_mod
from . import signal_quality as sq
from . import task as task_mod


def screen_channels(x, fs, names, min_bad_share=0.5):
    """Drop channels whose signal-quality status is bad/poor in at least half of the 4-s windows (flat, railed, hum, ...)."""
    L, H = int(4 * fs), int(2 * fs)
    stat = {n: [] for n in names}
    for s in range(0, x.shape[1] - L + 1, H):
        a = sq.assess(x[:, s:s + L], fs, names)
        for n in names:
            stat[n].append(a["channels"][n]["status"])
    excluded = {}
    for n, st in stat.items():
        share = float(np.mean([v in ("bad", "poor") for v in st])) if st else 0.0
        if share >= min_bad_share:
            excluded[n] = f"{share:.0%} of windows bad/poor"
    return excluded


def add_rest_session(baseline, eeg=None, eye=None, reset=False, condition="eyes_open", heart=None):
    """eeg = (x[C,T] uV, fs, names) of a resting recording; eye = camera DataFrame of the same period (eyes-open rest only);
    heart = dict(t_end=..., rr_ms=..., source="ecg"|"ppg") or dict(ecg=..., fs=...) from a watch (PLANNED feature; >= ~3 min of rest).
    condition: "eyes_open" -> baseline["eeg"] (used by the FOCUS scores and by eyes-open relaxation),
               "eyes_closed" -> baseline["eeg_closed"] (needed by the RELAXATION score when the eyes are closed: closing the eyes raises
               alpha by itself, so eyes-closed EEG must be compared with an eyes-closed baseline).
    Returns (baseline, notes)."""
    b = baseline or baselines.new_baseline()
    notes = []
    if condition not in ("eyes_open", "eyes_closed"):
        raise ValueError("condition must be 'eyes_open' or 'eyes_closed'")
    if eeg is None and eye is None and heart is None:
        raise ValueError("give EEG, eye and/or heart data")
    if eye is not None and condition == "eyes_closed":
        raise ValueError("the camera calibration clip is only used for eyes-open rest")
    block = "eeg" if condition == "eyes_open" else "eeg_closed"
    if eeg is not None:
        x, fs, names = eeg
        cand = [c for c in names if c in K.FRONTAL or c in K.POSTERIOR]
        if not cand:
            raise ValueError(f"none of the channels {names} are frontal/posterior 10-20 sites ({K.FRONTAL + K.POSTERIOR})")
        x, names = np.asarray(x)[[names.index(c) for c in cand]], cand
        excluded = screen_channels(x, fs, names)
        front = [c for c in names if c in K.FRONTAL and c not in excluded]
        post = [c for c in names if c in K.POSTERIOR and c not in excluded]
        if not front:
            raise ValueError(f"no usable frontal channel after screening (excluded: {excluded}). Fix the sensors first.")
        e = b.get(block)
        if e and not reset and (sorted(e["front"]) != sorted(front) or sorted(e["post"]) != sorted(post)):
            raise ValueError(f"channel set changed (baseline: front={e['front']} post={e['post']}; now front={front} post={post}). "
                             "Use reset to rebuild the EEG baseline, or fix the sensors so the same channels are usable.")
        sess = eeg_mod.session_stats(x, fs, names, front=front, post=post)
        sessions = ([] if (reset or not e) else e["sessions"]) + [dict(recorded_utc=baselines.now(), features=sess["features"],
                                                                      n_windows=sess["quality"]["n_clean"])]
        b[block] = dict(fs=fs, front=front, post=post, excluded=excluded, sessions=sessions, features=baselines.pooled_eeg(sessions))
        notes.append(f"EEG ({condition}): {sess['quality']['n_clean']}/{sess['quality']['n_windows']} clean windows; channels front={front} post={post}"
                     + (f"; excluded {excluded}" if excluded else ""))
    if eye is not None:
        cal = eye_mod.calibrate_session(eye)
        r = b.get("eye_rest")
        sessions = ([] if (reset or not r) else r["sessions"]) + [dict(recorded_utc=baselines.now(), values=cal["values"],
                                                                      open_level=cal["open_level"], roi=cal["roi"])]
        b["eye_rest"] = dict(open_level=float(np.mean([s["open_level"] for s in sessions])), roi=cal["roi"], sessions=sessions,
                             features=baselines.metric_entries(sessions, K.eye_table("passive")))
        notes.append(f"eye: {cal['quality']['n_blinks']} blinks in {cal['quality']['duration_s']:.0f} s, open level {cal['open_level']:.3f}, "
                     f"on-task gaze region {None if not cal['roi'] else [round(v, 2) for v in cal['roi']]}")
    if heart is not None:
        t_end, rr, src = heart_mod.from_arrays(**heart)
        hs = heart_mod.session_stats(t_end, rr, src)
        h = b.get("heart_rest")
        sessions = ([] if (reset or not h) else h["sessions"]) + [dict(recorded_utc=baselines.now(), features=hs["features"], source=src,
                                                                      n_windows=hs["quality"]["n_windows"])]
        b["heart_rest"] = dict(sessions=sessions, features=baselines.pooled_eeg(sessions))
        notes.append(f"heart ({src}): {hs['quality']['n_windows']} usable 60-s windows; "
                     + ", ".join(f"{k} {v['mean']:.2f}" for k, v in hs["features"].items()))
    b["updated_utc"] = baselines.now()
    return b, notes


def add_training_session(baseline, changes, clicks, eye, ball=None, period=K.ORBIT_PERIOD_S, radius=K.ORBIT_RADIUS, force=False):
    """One ordinary training session (times relative to the task start). Returns (baseline, values dict, n_sessions)."""
    b = baseline or baselines.new_baseline()
    if ball is None:
        ball = task_mod.ball_frame(eye["t"].to_numpy(float), period, radius)
    cm = clicks_mod.analyze_clicks(changes, clicks, duration_s=float(eye["t"].max()))
    pm, pq = pursuit_mod.analyze_pursuit(eye, ball, open_level=(b.get("eye_rest") or {}).get("open_level"))
    problems = []
    if cm["n_targets"] < K.CLICK_Q_FULL_AT:
        problems.append(f"only {cm['n_targets']} colour changes (need >= {K.CLICK_Q_FULL_AT})")
    if pm is None:
        problems.append(f"no usable gaze: {pq.get('reason')}")
    elif pq["valid_frac"] < K.EYE_FULL_VALID_FRAC:
        problems.append(f"only {pq['valid_frac']:.0%} of frames had a usable face+gaze (need >= {K.EYE_FULL_VALID_FRAC:.0%})")
    if problems and not force:
        raise ValueError("session not added: " + "; ".join(problems))
    values = {k: float(v) for k, v in {**{n: cm.get(n) for n in K.CLICK_FEATURES}, **{n: (pm or {}).get(n) for n in K.eye_table("training")}}.items()
              if v is not None and np.isfinite(v)}
    t = b.get("training") or dict(sessions=[], features={})
    t["sessions"].append(dict(recorded_utc=baselines.now(), values=values, n_targets=cm["n_targets"]))
    t["features"] = baselines.metric_entries(t["sessions"], {**K.CLICK_FEATURES, **K.eye_table("training")})
    b["training"] = t
    b["updated_utc"] = baselines.now()
    return b, values, len(t["sessions"])
