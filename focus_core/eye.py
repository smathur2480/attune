"""
focus_core.eye -- camera eye features for PASSIVE monitoring (blinks, eyelid closure, gaze on task, fixations, saccades).

Input: DataFrame with t (s), ear (eye aspect ratio, mean of both eyes), optional gaze_x / gaze_y (0-1) and face_ok (0/1) --
produced by tools/camera_capture.py (MediaPipe FaceMesh) or the app's own landmark pipeline.

Blink: run of frames with EAR < 60% of the open level that reaches < 40% of it and lasts >= 50 ms; duration = time below 60%.
PERCLOS-type closure: share of frames with EAR < 20% of the open level (~80% closed).
off_task_frac: share of gaze samples outside the on-task region (learned from the calibration clip).
long_fix_frac / saccade_rate: I-DT fixations on gaze (only meaningful in scanning/reading tasks; webcam gaze is crude).

LITERATURE
  [VanOrden2000] Van Orden KF, Jung T-P, Makeig S (2000). Combined eye activity measures accurately estimate changes in sustained
                 visual task performance. Biological Psychology 52:221-240. Blink, fixation and pupil measures together tracked
                 fatigue-related performance changes (9 participants) with individualised models validated across sessions.
  [Schleicher2008] Schleicher R, Galley N, Briest S, Galley L (2008). Blinks and saccades as indicators of fatigue in sleepiness
                 warnings: looking tired? Ergonomics. In 129 participants blink duration, lid-reopening delay, blink interval and lid
                 closure speed were the best sleepiness indicators; saccade and fixation parameters also changed; inter-individual
                 differences were substantial (=> the PERSONAL baseline).
  [FAA1994]      FAA Office of Aerospace Medicine technical reports (1994, 1996), "Blinks, saccades, and fixation pauses during
                 vigilance task performance" (authors not retrieved): time-on-task effects on blink and saccade measures.
  [Soukupova2016] Soukupova T, Cech J (2016). Real-time eye blink detection using facial landmarks. 21st Computer Vision Winter
                 Workshop. Eye aspect ratio (EAR) from landmarks as a single openness value; blinks from its temporal pattern.
  [GarciaAgundez2018] Garcia-Agundez A, et al. (2018). Eye aspect ratio based blink rate detection and its potential use for
                 Parkinson's disease. IWBBIO 2018. Webcam EAR blink detection: 82% precision / 75% recall in 10 healthy users.
  [Danisman2010] Danisman T, et al. (2010). Drowsy driver detection system using eye blink patterns. ICMWI 2010 (webcam blink
                 detection, 94% accuracy on the JZU database).
  [AttentionLapse2023] arXiv:2304.10588, "Detecting worker attention lapses in human-robot interaction" (secondary source): long
                 fixation on one point, looking away, slow/irregular saccades, eye closure and a high blink rate as lapse patterns;
                 also notes a person can mind-wander while fixating correctly (gaze alone is not proof of focus).
CAUTION: webcam EAR blink detection is imperfect (~75-82% recall in the cited test); frame rate limits blink-duration resolution.
"""
import numpy as np
import pandas as pd

from . import constants as K
from . import scoring


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def idt_fixations(x, y, fps, thr=0.04, min_s=0.1):
    n, w, out, i = len(x), max(2, int(round(min_s * fps))), [], 0
    while i + w <= n:
        xs, ys = x[i:i + w], y[i:i + w]
        if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
            i += 1
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        if (x1 - x0) + (y1 - y0) > thr:
            i += 1
            continue
        j = i + w
        while j < n and np.isfinite(x[j]) and np.isfinite(y[j]):
            nx0, nx1, ny0, ny1 = min(x0, x[j]), max(x1, x[j]), min(y0, y[j]), max(y1, y[j])
            if (nx1 - nx0) + (ny1 - ny0) > thr:
                break
            x0, x1, y0, y1, j = nx0, nx1, ny0, ny1, j + 1
        out.append((i, j, float(np.mean(x[i:j])), float(np.mean(y[i:j]))))
        i = j
    return out


def eye_features(df, open_level=None, roi=(0.0, 1.0, 0.0, 1.0), scan_task=False):
    """Returns (features dict | None, quality dict). quality["valid_frac"] = share of frames with a usable face."""
    t, ear = df["t"].to_numpy(float), df["ear"].to_numpy(float)
    ok = np.isfinite(ear) & ((df["face_ok"].to_numpy(float) > 0) if "face_ok" in df else True)
    fps = 1.0 / float(np.median(np.diff(t)))
    q = dict(fps=round(fps, 1), valid_frac=float(ok.mean()), duration_s=float(t[-1] - t[0]))
    if ok.mean() < K.EYE_MIN_FACE_FRAC or q["duration_s"] < K.EYE_MIN_DURATION_S:
        q["reason"] = f"face not visible enough or segment shorter than {K.EYE_MIN_DURATION_S:g} s"
        return None, q
    open_lvl = float(open_level) if open_level else float(np.nanpercentile(ear[ok], 90))
    e = np.where(ok, ear, np.nan)
    below = np.nan_to_num(e, nan=np.inf) < K.BLINK_CLOSE_FRAC * open_lvl
    blinks = [(en - s) / fps * 1000.0 for s, en in runs(below)
              if np.nanmin(e[s:en]) < K.BLINK_MIN_DEPTH * open_lvl and (en - s) / fps >= 0.05]
    minutes = ok.sum() / fps / 60.0
    feats = dict(blink_rate=len(blinks) / minutes if minutes > 0 else float("nan"),
                 blink_dur_ms=float(np.mean(blinks)) if blinks else float("nan"),
                 perclos=float(np.mean(np.nan_to_num(e, nan=np.inf)[ok] < K.PERCLOS_FRAC * open_lvl)))
    if {"gaze_x", "gaze_y"} <= set(df.columns):
        gx, gy = df["gaze_x"].to_numpy(float), df["gaze_y"].to_numpy(float)
        g = ok & np.isfinite(gx) & np.isfinite(gy)
        if g.sum() > 0.5 * ok.sum():
            x0, x1, y0, y1 = roi
            inside = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
            feats["off_task_frac"] = float(1 - inside[g].mean())
            fx = idt_fixations(np.where(g, gx, np.nan), np.where(g, gy, np.nan), fps)
            if fx:
                dur = np.array([(b - a) / fps for a, b, _, _ in fx])
                feats["saccade_rate"] = (len(fx) - 1) / minutes if minutes > 0 else float("nan")
                if scan_task:
                    feats["long_fix_frac"] = float(dur[dur >= 1.0].sum() / (g.sum() / fps))
    q.update(open_level=round(open_lvl, 4), n_blinks=len(blinks))
    return feats, q


def learn_roi(df, pad=0.03):
    """On-task gaze region = 3rd-97th percentile of gaze during a calibration clip in which the user looked at the screen."""
    if not {"gaze_x", "gaze_y"} <= set(df.columns):
        return None
    g = df[(df["face_ok"] > 0) if "face_ok" in df else slice(None)].dropna(subset=["gaze_x", "gaze_y"])
    if len(g) < 30:
        return None
    x0, x1 = np.percentile(g.gaze_x, [3, 97])
    y0, y1 = np.percentile(g.gaze_y, [3, 97])
    return [float(max(0, x0 - pad)), float(min(1, x1 + pad)), float(max(0, y0 - pad)), float(min(1, y1 + pad))]


def calibrate_session(df):
    """One resting calibration clip -> dict(open_level, roi, values{feature: value}, quality). Input to baseline.py."""
    roi = learn_roi(df)
    feats, q = eye_features(df, roi=tuple(roi) if roi else (0.0, 1.0, 0.0, 1.0))
    if feats is None:
        raise ValueError(f"eye calibration failed: {q}")
    return dict(open_level=q["open_level"], roi=roi, values={k: float(v) for k, v in feats.items() if np.isfinite(v)}, quality=q)


def eye_strengths(df, baseline=None, scan_task=False):
    """PASSIVE eye strengths. baseline: the "eye_rest" block of baseline.json (None -> population placeholders).
    Returns (strengths, quality, features)."""
    b = baseline or {}
    roi = tuple(b["roi"]) if b.get("roi") else (0.0, 1.0, 0.0, 1.0)
    feats, q = eye_features(df, open_level=b.get("open_level"), roi=roi, scan_task=scan_task)
    if feats is None:
        return {}, q, {}
    s = scoring.strengths(feats, K.eye_table("passive"), b.get("features"))
    q["basis"] = "own baseline" if b.get("features") else "population placeholders"
    return s, q, feats
