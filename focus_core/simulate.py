"""
focus_core.simulate -- SYNTHETIC EEG / camera / task data for demos and tests. Not real physiology: use it to exercise the pipeline,
never to judge the score.   python -m focus_core.simulate write-demo --out examples
States: rest | focused | wandering | tired (passive);  focused | lapsing | tired (training).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from . import constants as K
from .signal_quality import make_eeg  # noqa: F401  (re-exported: pink-noise EEG with blink artefacts)
from .signal_quality import pink
from .task import make_schedule, orbit

STATE_EEG = {"rest": (1.0, 1.0, 1.0), "focused": (0.8, 0.9, 1.25), "wandering": (1.3, 1.2, 0.8), "tired": (1.6, 1.4, 0.7),
             # relaxation states (alpha, theta, beta gains). EYES-CLOSED rest has ~2x the alpha of eyes-open rest (Berger effect).
             "rest_closed": (2.0, 1.0, 1.0), "relaxed_closed": (2.7, 1.4, 0.8), "drowsy_closed": (1.3, 2.3, 0.7), "aroused_closed": (1.4, 0.9, 1.5),
             "relaxed_open": (1.5, 1.25, 0.9), "deep_closed": (3.4, 1.5, 0.8)}   # alpha, theta, beta gain


def synth_eeg(state="rest", fs=250, dur=60.0, seed=0, names=("Fp1", "Fp2", "F3", "F4", "O1", "O2")):
    """1/f background (10 uV rms) + alpha 10 Hz / theta 6 Hz / beta 20 Hz oscillations whose gains depend on the state."""
    rng = np.random.default_rng(seed)
    ga, gt, gb = STATE_EEG[state]
    n = int(dur * fs)
    t = np.arange(n) / fs
    rows = []
    for nm in names:
        post = nm.startswith("O")
        x = pink(n, fs, rng, 1.0, 10.0)
        x += 6 * (2.0 if post else 1.0) * ga * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 6)) * (0.7 + 0.3 * rng.random())
        x += 4 * gt * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 6))
        x += 2.5 * gb * np.sin(2 * np.pi * 20.0 * t + rng.uniform(0, 6))
        rows.append(x)
    return np.array(rows), float(fs), list(names)


STATE_EYE = {"rest": dict(rate=14, dur=150, micro=0, off=0.04), "focused": dict(rate=8, dur=120, micro=0, off=0.02),
             "wandering": dict(rate=17, dur=170, micro=0, off=0.15), "tired": dict(rate=26, dur=400, micro=2, off=0.12)}


def synth_eye(state="rest", fps=30, dur=60, seed=0, off=None):
    """Camera signals for PASSIVE monitoring: ear, gaze_x/gaze_y, face_ok. off = share of away-from-screen fixations (x2)."""
    rng = np.random.default_rng(seed)
    cfg = dict(STATE_EYE[state])
    if off is not None:
        cfg["off"] = off
    t = np.arange(int(dur * fps)) / fps
    ear = 0.30 + rng.normal(0, 0.006, t.size)
    n = max(1, int(cfg["rate"] * dur / 60))
    centres = np.sort(rng.choice(np.arange(1.0, dur - 1.0, 1.2), n, replace=False))
    events = [(c, cfg["dur"]) for c in centres] + [(c, 1500.0) for c in rng.choice(np.arange(3, dur - 3, 7.0), cfg["micro"], replace=False)]
    for c, d in events:
        ear -= 0.25 * np.exp(-0.5 * ((t - c) / (d / 1000.0 / 2.42)) ** 2)
    gx, gy = np.full(t.size, 0.5), np.full(t.size, 0.5)
    pos = np.array([0.5, 0.5])
    for i in range(t.size):
        if i % int(2.0 * fps) == 0:
            away = rng.random() < cfg["off"] * 2.0
            pos = np.array([rng.uniform(0.9, 1.0), 0.5]) if away else np.array([rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)])
        gx[i], gy[i] = pos[0] + rng.normal(0, 0.004), pos[1] + rng.normal(0, 0.004)
    return pd.DataFrame(dict(t=t, ear=ear, gaze_x=gx, gaze_y=gy, face_ok=1))


SIM = {
    "focused": dict(rt=330, rt_sd=45, miss=0.02, fa_per_min=0.3, lag=0.15, noise=0.03, lapses_per_min=0.3, lapse_s=(0.6, 1.0), micro=0, blink=10, eeg="focused"),
    "lapsing": dict(rt=395, rt_sd=85, miss=0.05, fa_per_min=0.6, lag=0.18, noise=0.035, lapses_per_min=1.5, lapse_s=(0.8, 2.0), micro=0, blink=14, eeg="wandering"),
    "tired": dict(rt=520, rt_sd=160, miss=0.15, fa_per_min=0.6, lag=0.25, noise=0.05, lapses_per_min=3.0, lapse_s=(0.8, 2.5), micro=2, blink=24, eeg="tired"),
}


def simulate_session(state="focused", duration=180.0, fps=30, seed=0, period=K.ORBIT_PERIOD_S, radius=K.ORBIT_RADIUS, jitter=0.0):
    """A training session: ball trajectory, colour changes, clicks, camera signals and EEG.
    jitter > 0 makes the person's performance vary from session to session (relative SD of the task parameters)."""
    rng = np.random.default_rng(seed)
    c = dict(SIM[state])
    if jitter:
        for key in ("rt", "rt_sd", "lag", "noise", "lapses_per_min", "fa_per_min"):
            c[key] = c[key] * float(np.clip(1 + jitter * rng.normal(), 0.5, 1.8))
        c["miss"] = float(np.clip(c["miss"] * (1 + 2 * jitter * rng.normal()), 0.0, 0.5))
    t = np.arange(0, duration, 1.0 / fps)
    bx, by = orbit(t, period, radius)
    lagk = int(c["lag"] * fps)
    gx = 0.5 + 0.10 * (np.r_[np.full(lagk, bx[0]), bx[:len(bx) - lagk]] - 0.5) / radius + rng.normal(0, c["noise"], t.size)
    gy = 0.5 + 0.10 * (np.r_[np.full(lagk, by[0]), by[:len(by) - lagk]] - 0.5) / radius + rng.normal(0, c["noise"], t.size)
    ear = 0.30 + rng.normal(0, 0.006, t.size)
    for b in np.sort(rng.uniform(1, duration - 1, int(c["blink"] * duration / 60))):
        ear -= 0.25 * np.exp(-0.5 * ((t - b) / (0.15 / 2.42)) ** 2)
    starts = np.sort(rng.uniform(5, duration - 5, int(c["lapses_per_min"] * duration / 60)))
    for i, s0 in enumerate(starts):
        d = rng.uniform(*c["lapse_s"])
        i0, i1 = int(s0 * fps), int((s0 + d) * fps)
        gx[i0:i1], gy[i0:i1] = gx[i0], gy[i0]
        if i < c["micro"]:
            ear[i0:i1] = 0.04
    changes = make_schedule(duration, seed=seed + 100)
    clicks = [ch + max(0.2, rng.normal(c["rt"], c["rt_sd"])) / 1000.0 for ch in changes if rng.random() > c["miss"]]
    clicks += list(rng.uniform(2, duration - 1, rng.poisson(c["fa_per_min"] * duration / 60)))
    return dict(ball=pd.DataFrame(dict(t=t, bx=bx, by=by)), changes=changes, clicks=sorted(float(x) for x in clicks),
                eye=pd.DataFrame(dict(t=t, ear=ear, gaze_x=gx, gaze_y=gy, face_ok=1)), eeg=synth_eeg(c["eeg"], 250, duration, seed),
                duration=duration)


def save_eeg_csv(eeg, path):
    x, _fs, names = eeg
    pd.DataFrame(x.T, columns=names).to_csv(path, index=False)


def save_session(sess, prefix):
    ev = pd.DataFrame([dict(type="change", t=c) for c in sess["changes"]] + [dict(type="click", t=c) for c in sess["clicks"]])
    ev.to_csv(prefix + "_events.csv", index=False)
    sess["eye"].to_csv(prefix + "_eye.csv", index=False)
    save_eeg_csv(sess["eeg"], prefix + "_eeg.csv")


def write_demo(out):
    """Write a complete set of synthetic input files so baseline.py / simple_score.py / complex_score.py can be run end to end."""
    os.makedirs(out, exist_ok=True)
    save_eeg_csv(synth_eeg("rest", seed=1, dur=120), os.path.join(out, "rest_eeg.csv"))
    synth_eye("rest", seed=1, dur=90, off=0.0).to_csv(os.path.join(out, "rest_eye.csv"), index=False)
    for i, sd in enumerate((11, 12, 13), 1):                                  # three ordinary training sessions = the personal baseline
        save_session(simulate_session("focused", seed=sd, jitter=0.2), os.path.join(out, f"train_{i}"))
    save_eeg_csv(synth_eeg("wandering", seed=7, dur=30), os.path.join(out, "passive_eeg.csv"))        # a drifting-attention moment
    synth_eye("wandering", seed=7, dur=45).to_csv(os.path.join(out, "passive_eye.csv"), index=False)
    save_session(simulate_session("lapsing", seed=3), os.path.join(out, "session"))                    # a lapsing training session
    with open(os.path.join(out, "profile.json"), "w") as fh:
        json.dump(dict(age=29, sex="F", hours_since={"caffeine": 2, "cigarette": None, "alcohol": 20},
                       asrs_items=[2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 2, 3, 1, 2, 1, 2, 2]), fh, indent=1)
    return sorted(os.listdir(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("write-demo")
    w.add_argument("--out", default="examples")
    a = ap.parse_args()
    print("wrote", write_demo(a.out))


def iter_stream(eeg=None, eye=None, events=(), chunk_s=0.25, t0=0.0):
    """Yield ('eeg', t_start, chunk) / ('eye', rows) / ('event', kind, t) in time order, like a live app would deliver them.
    eeg: (x[C,T], fs, names); eye: DataFrame (t, ...); events: [(kind, t), ...]. t0 shifts everything onto the app clock."""
    items = []
    if eeg is not None:
        x, fs, _names = eeg
        step = int(chunk_s * fs)
        for k in range(0, x.shape[1] - step + 1, step):
            items.append(((k + step) / fs + t0, 0, ("eeg", k / fs + t0, x[:, k:k + step])))          # delivered when the chunk is complete
    if eye is not None:
        rows = eye.to_dict("records")
        for i in range(0, len(rows), 3):                                                              # ~10 deliveries per second at 30 fps
            batch = [dict(r, t=r["t"] + t0) for r in rows[i:i + 3]]
            items.append((batch[-1]["t"], 1, ("eye", batch)))
    for kind, t in events:
        items.append((t + t0, 2, ("event", kind, t + t0)))
    for _t, _o, item in sorted(items, key=lambda z: (z[0], z[1])):
        yield item


# ---------------------------------------------------------------------------------------------------------------------
# relaxation study helpers: eyes-closed camera trace, heart (RR / ECG)
# ---------------------------------------------------------------------------------------------------------------------
def synth_eye_closed(dur=60, fps=30, seed=0, closed_frac=1.0):
    """Camera trace of a person sitting with eyes closed (EAR ~0.04); closed_frac < 1 makes some stretches open."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fps)) / fps
    ear = np.where(rng.random(t.size) < 0, 0.30, 0.04) + rng.normal(0, 0.004, t.size)
    if closed_frac < 1.0:
        k = int(len(t) * (1 - closed_frac))
        ear[:k] = 0.30 + rng.normal(0, 0.006, k)
    return pd.DataFrame(dict(t=t, ear=ear, face_ok=1))


def synth_eye_sleepy_open(dur=60, fps=30, seed=0):
    """Eyes open but drowsy: frequent slow, long blinks and partial closures (long PERCLOS)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fps)) / fps
    ear = 0.30 + rng.normal(0, 0.006, t.size)
    for c in np.sort(rng.uniform(1, dur - 1, int(22 * dur / 60))):
        ear -= 0.25 * np.exp(-0.5 * ((t - c) / (0.55 / 2.42)) ** 2)
    for c in rng.uniform(3, dur - 3, max(1, int(dur / 12))):
        ear[(t >= c) & (t < c + 1.2)] = 0.05
    return pd.DataFrame(dict(t=t, ear=ear, face_ok=1))


HEART = {"rest": dict(rr=850.0, amp=22.0, f=0.25, noise=14.0), "relaxed": dict(rr=950.0, amp=45.0, f=0.25, noise=14.0),
         "relaxed_paced": dict(rr=950.0, amp=55.0, f=0.10, noise=14.0), "aroused": dict(rr=700.0, amp=11.0, f=0.30, noise=10.0)}


def synth_rr(state="rest", dur=180.0, seed=0, ectopic=0):
    """Beat times and RR intervals (ms) with respiratory sinus arrhythmia at the breathing frequency f. ectopic = number of early beats."""
    rng = np.random.default_rng(seed)
    c = HEART[state]
    t, rrs, t_beats = 0.0, [], [0.0]
    while t < dur:
        rr = c["rr"] + c["amp"] * np.sin(2 * np.pi * c["f"] * t) + rng.normal(0, c["noise"])
        rrs.append(rr)
        t += rr / 1000.0
        t_beats.append(t)
    rrs = np.array(rrs)
    tb = np.array(t_beats[1:])
    if ectopic:
        for i in rng.choice(np.arange(10, len(rrs) - 10), ectopic, replace=False):
            rrs[i] *= 0.55                                      # premature beat ...
            rrs[i + 1] *= 1.45                                  # ... followed by a compensatory pause
    tb = np.cumsum(rrs) / 1000.0
    return tb, rrs


def synth_ecg(beat_times, dur, fs=250, seed=0):
    """Single-lead ECG-like strip: R waves (+ small T waves), baseline wander and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fs)) / fs
    x = 0.3 * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, 0.04, t.size)
    for b in beat_times:
        if b < dur:
            x += 1.0 * np.exp(-0.5 * ((t - b) / 0.012) ** 2) + 0.18 * np.exp(-0.5 * ((t - b - 0.25) / 0.045) ** 2)
    return x
