"""
focus_core.pursuit -- how well gaze follows the orbiting ball (TRAINING mode eye metrics).

Input: eye DataFrame (t, gaze_x, gaze_y, [ear, face_ok]) and ball DataFrame (t, bx, by) on the same clock.
Metrics
  pursuit_corr    gaze-vs-ball trajectory correlation at the best lag (0-600 ms), mean of the x and y correlations
  lag_ms          that best lag
  tracking_error  median |mapped gaze - ball| / orbit radius, after an affine gaze->ball map fitted robustly (best 70% of frames,
                  so lapses do not bend the map); absorbs the unknown camera-to-screen calibration
  lapse_time_frac share of time in tracking LAPSES = >= 0.5 s of flat/incoherent tracking (mapped gaze moves < 35% of what the ball
                  does over 0.5 s, or is > 1 orbit radius away). Counted as a MICROSLEEP if the eyelids are closed > 50% of the lapse
                  (EAR < 35% of open level), otherwise an ATTENTION lapse.
  perclos         eyelid-closure share (same definition as focus_core.eye)

LITERATURE
  [Fransson2008] Fransson P-A, Patel M, Magnusson M (2008). Effects of 24-hour and 36-hour sleep deprivation on smooth pursuit and
                 saccadic eye movements. J Vestib Res 18(4), doi:10.3233/VES-2008-18404. Sleep deprivation worsened smooth-pursuit
                 gain, accuracy and saccade velocity.
  [Frontiers2022] "Differential saccade-pursuit coordination under sleep loss and low-dose alcohol", Frontiers in Neuroscience (2022)
                 doi:10.3389/fnins.2022.1067722 (authors not retrieved): pursuit gain loss under sleep loss and low-dose alcohol; pursuit
                 and catch-up saccades work together during tracking.
  [Peiris2006]   Peiris et al. (2006). Lapses of responsiveness during an extended visuomotor tracking task in non-sleep-deprived
                 subjects. J Sleep Res 15:291-300 (lapses identified by flat/incoherent tracking).
  [Jones2010]    Jones et al. (2010). Lapses of responsiveness: characteristics, detection, and underlying mechanisms. Proc IEEE EMBS
                 Annual Int Conf. Lapses last ~0.5-15 s; flat/incoherent tracking > 500 ms WITH eye closure = microsleep, WITHOUT =
                 attention lapse.
CAUTION: webcam gaze is an eye-in-head ratio with head movement not removed; true pursuit GAIN cannot be measured, hence the
correlation / error / lapse metrics. All thresholds are untuned starting points.
"""
import math

import numpy as np
import pandas as pd

from . import constants as K
from .eye import runs


def _pearson(a, b):
    if len(a) < 10 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def analyze_pursuit(eye, ball, open_level=None, max_lag_s=K.PURSUIT_MAX_LAG_S, lapse_s=K.TRACK_LAPSE_S,
                    err_thr=K.TRACK_ERR_THR, flat_ratio=K.TRACK_FLAT_RATIO):
    """Returns (metrics dict | None, quality dict). quality["valid_frac"] = share of frames with usable face AND gaze."""
    t = eye["t"].to_numpy(float)
    gx, gy = eye["gaze_x"].to_numpy(float), eye["gaze_y"].to_numpy(float)
    ok = np.isfinite(gx) & np.isfinite(gy) & ((eye["face_ok"].to_numpy(float) > 0) if "face_ok" in eye else True)
    bx = np.interp(t, ball["t"].to_numpy(float), ball["bx"].to_numpy(float))
    by = np.interp(t, ball["t"].to_numpy(float), ball["by"].to_numpy(float))
    fps = 1.0 / float(np.median(np.diff(t)))
    q = dict(fps=round(fps, 1), valid_frac=float(ok.mean()), duration_s=float(t[-1] - t[0]))
    if ok.sum() < K.PURSUIT_MIN_USABLE_S * fps or q["duration_s"] < K.PURSUIT_MIN_USABLE_S:
        q["reason"] = f"less than {K.PURSUIT_MIN_USABLE_S:g} s of usable gaze"
        return None, q
    best = (-2.0, 0)
    for lag in range(0, int(max_lag_s * fps) + 1):          # gaze at frame k follows the ball at frame k - lag
        sel = ok[lag:] & ok[:len(ok) - lag] if lag else ok
        a, b = (gx[lag:], bx[:len(bx) - lag]) if lag else (gx, bx)
        c, d = (gy[lag:], by[:len(by) - lag]) if lag else (gy, by)
        vals = [v for v in (_pearson(a[sel], b[sel]), _pearson(c[sel], d[sel])) if np.isfinite(v)]
        r = float(np.mean(vals)) if vals else float("nan")
        if np.isfinite(r) and r > best[0]:
            best = (r, lag)
    corr, lag = best
    k = np.arange(lag, len(t))
    sel = ok[k] & ok[k - lag]
    G = np.c_[np.ones(sel.sum()), gx[k][sel], gy[k][sel]]
    T = np.c_[bx[k - lag][sel], by[k - lag][sel]]
    coef = np.linalg.lstsq(G, T, rcond=None)[0]
    for _ in range(2):                                        # robust refit on the best 70% of frames
        res = np.linalg.norm(G @ coef - T, axis=1)
        keep = res <= np.quantile(res, 0.7)
        coef = np.linalg.lstsq(G[keep], T[keep], rcond=None)[0]
    radius = math.sqrt(float(np.var(bx) + np.var(by)))
    ok_lag = ok & np.r_[np.zeros(lag, bool), ok[:len(ok) - lag]] if lag else ok
    err = np.full(len(t), np.nan)
    err[k] = np.linalg.norm(np.c_[np.ones(len(k)), gx[k], gy[k]] @ coef - np.c_[bx[k - lag], by[k - lag]], axis=1) / radius
    err[~ok_lag] = np.nan
    w = max(1, int(0.25 * fps))
    err_s = pd.Series(err).rolling(w, min_periods=1, center=True).mean().to_numpy()
    good = np.isfinite(err_s)
    mapped = np.full((len(t), 2), np.nan)                     # "flat / incoherent tracking": gaze barely moves while the ball does
    mapped[k] = np.c_[np.ones(len(k)), gx[k], gy[k]] @ coef
    mapped[~ok] = np.nan
    mapped = pd.DataFrame(mapped).rolling(5, min_periods=3, center=True).mean().to_numpy()
    ballsh = np.c_[np.r_[np.full(lag, np.nan), bx[:len(bx) - lag]] if lag else bx, np.r_[np.full(lag, np.nan), by[:len(by) - lag]] if lag else by]
    h = max(2, int(0.25 * fps))
    dg, db = np.full(len(t), np.nan), np.full(len(t), np.nan)
    dg[h:-h] = np.linalg.norm(mapped[2 * h:] - mapped[:-2 * h], axis=1)
    db[h:-h] = np.linalg.norm(ballsh[2 * h:] - ballsh[:-2 * h], axis=1)
    ratio = dg / np.where(db > 1e-6, db, np.nan)
    flagged = good & ((err_s > err_thr) | (np.isfinite(ratio) & (ratio < flat_ratio)))
    ear = eye["ear"].to_numpy(float) if "ear" in eye else None
    open_lvl = (float(open_level) if open_level else float(np.nanpercentile(ear[ok], 90))) if ear is not None and ok.any() else None
    lapses, micro, attn = [], 0, 0
    for s, e in runs(flagged):
        if (e - s) / fps >= lapse_s:
            lapses.append((s, e))
            if open_lvl:
                closed = float(np.mean(ear[s:e] < 0.35 * open_lvl))
                micro += closed > 0.5
                attn += closed <= 0.5
            else:
                attn += 1
    m = dict(pursuit_corr=float(corr), lag_ms=lag / fps * 1000.0, tracking_error=float(np.nanmedian(err_s)),
             lapse_time_frac=float(sum(e - s for s, e in lapses) / max(good.sum(), 1)), n_tracking_lapses=len(lapses),
             n_attention_lapses=int(attn), n_microsleeps=int(micro))
    if ear is not None and open_lvl:
        m["perclos"] = float(np.mean(ear[ok] < K.PERCLOS_FRAC * open_lvl))
    return m, q
