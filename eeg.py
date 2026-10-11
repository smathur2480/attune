"""
focus_core.eeg -- EEG window features and their strengths versus a baseline.

Features per 8-s window (log10 of relative power; frontal channels unless noted):
  alpha_rel = log10( alpha(8-12 Hz) / total(1-40 Hz) )      frontal + posterior channels together
  tbr       = log10( theta(4-8) / beta(13-30) )             frontal
  beta_rel  = log10( beta / total )                         frontal
  slow_rel  = log10( slow(1-8) / total )                    frontal
  theta_ex  = log10( theta(4-8) / (total - alpha) )         frontal        (used by the RELAXATION score; independent of alpha)
Windows with peak-to-peak > 150 uV (frontal) / 300 uV (posterior) are rejected; windows with muscle contamination
(35-45 Hz power > 0.6 x 13-30 Hz power) keep alpha/slow but drop the beta-based features (tbr, beta_rel).
A scoring segment's features are the MEDIAN over its clean windows; strength = vs the baseline (focus_core.scoring.strength).

LITERATURE (why these features and these directions; see constants.EEG_FEATURES)
  [Kam2022]      Kam JWY, Rahnuma T, Park YE, Hart CM (2022). Electrophysiological markers of mind wandering: a systematic review.
                 NeuroImage 258 (PMID 35700946). Spectral findings are less robust than ERP findings but point to more delta/theta/
                 alpha and less beta during mind wandering than on-task; results vary with task context.
  [Groot2021]    Groot JM, Boayue NM, Csifcsak G, Boekel W, et al. (2021). Probing the neural signature of mind wandering with
                 simultaneous fMRI-EEG and pupillometry. NeuroImage 224:117412. Widespread alpha/theta/delta increase but NO beta change.
  [Jin2019]      Jin CY, Borst JP, van Vugt MK (2019). Predicting task-general mind-wandering with EEG (PMID 30850931). Alpha power was
                 the most predictive EEG marker (classification accuracy 0.50-0.85).
  [Arnau2019]    Arnau S (2019, bioRxiv preprint doi:10.1101/819805). The electrophysiological signature of mind wandering. Alpha power
                 was higher on trials where people reported mind wandering.
  [vanSon2019]   van Son D, De Blasio FM, Fogarty JS, Angelidis A, Barry RJ, Putman P (2019). Frontal EEG theta/beta ratio during mind
                 wandering episodes. Biological Psychology 140:19-27. Frontal TBR higher during mind wandering than on-task (n=26).
  [Frontiers2025] Frontiers in Neuroscience (2025), article 1654827 (authors not retrieved): mind wandering showed more 8-10 Hz alpha
                 over frontal-posterior regions.
  [bioRxiv2023]  bioRxiv doi:10.1101/2023.03.21.533634 (authors not retrieved): greater alpha during task-unrelated thought in many
                 studies (some contradict); frontal theta also INCREASES with externally directed attentional demands -> low weight
                 for slow_rel. The engagement index beta/(alpha+theta) is not used: evidence for it was mixed.
CAUTION: these are population-level contrasts from mind-wandering vs on-task studies; none is validated for scoring an individual.
"""
import math

import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapezoid

from . import constants as K
from . import scoring
from .reference import reference_expectation


def bandpass(x, fs, lo=0.5, hi=48.0):
    hi = min(hi, 0.45 * fs)
    sos = signal.butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def band_power(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(trapezoid(p[m], f[m])) if m.sum() > 1 else float("nan")


def window_features(xw, fs, n_front):
    """xw: (C, T) band-passed uV, first n_front rows frontal. Returns (features | None, reason)."""
    pp = xw.max(1) - xw.min(1)
    lim = np.array([K.EEG_REJECT_FRONT_UV] * n_front + [K.EEG_REJECT_POST_UV] * (xw.shape[0] - n_front))
    if (pp > lim).any():
        return None, "artefact"
    f, P = signal.welch(xw, fs, nperseg=min(int(2 * fs), xw.shape[1]), axis=-1)
    Pf, Pall = P[:n_front].mean(0), P.mean(0)
    B = K.EEG_BANDS
    tot_f, tot_all = band_power(f, Pf, *B["total"]), band_power(f, Pall, *B["total"])
    th, be, sl = band_power(f, Pf, *B["theta"]), band_power(f, Pf, *B["beta"]), band_power(f, Pf, *B["slow"])
    al_f = band_power(f, Pf, *B["alpha"])
    out = dict(alpha_rel=math.log10(band_power(f, Pall, *B["alpha"]) / tot_all), slow_rel=math.log10(sl / tot_f),
               tbr=math.log10(th / be), beta_rel=math.log10(be / tot_f),
               theta_ex=math.log10(th / max(tot_f - al_f, 1e-12)))      # theta vs NON-alpha power: a change in alpha alone cannot move it
    emg = band_power(f, Pf, *B["emg"]) / be if fs >= 100 else 0.0
    if emg > K.EEG_EMG_MAX:
        out["tbr"] = out["beta_rel"] = float("nan")
        return out, "emg"
    return out, "ok"


def eeg_windows(x, fs, names, win_s=K.EEG_WIN_S, hop_s=K.EEG_HOP_S, front=None, post=None):
    """x: (C, T) uV. Returns (list of feature dicts for usable windows, quality dict)."""
    front = [c for c in (front if front is not None else K.FRONTAL) if c in names]
    post = [c for c in (post if post is not None else K.POSTERIOR) if c in names]
    if not front:
        raise ValueError(f"no frontal channels among {names}")
    sel = [names.index(c) for c in front + post]
    xf = bandpass(np.asarray(x, float)[sel], fs)
    L, H = int(win_s * fs), int(hop_s * fs)
    if xf.shape[1] < L:
        L = H = xf.shape[1]
        if L < int(K.EEG_MIN_SEGMENT_S * fs):
            return [], dict(n_windows=0, n_clean=0, reasons={"too short": 1}, front=front, post=post)
    starts = list(range(0, xf.shape[1] - L + 1, H))
    feats, reasons = [], {}
    for s in starts:
        ft, why = window_features(xf[:, s:s + L], fs, len(front))
        reasons[why] = reasons.get(why, 0) + 1
        if ft is not None:
            feats.append(ft)
    return feats, dict(n_windows=len(starts), n_clean=len(feats), reasons=reasons, front=front, post=post)


def session_stats(x, fs, names, front=None, post=None):
    """Per-feature mean / SD / n over the clean windows of ONE resting recording (input to baseline.py)."""
    feats, q = eeg_windows(x, fs, names, front=front, post=post)
    if len(feats) < K.EEG_MIN_CLEAN_WINDOWS_BASELINE:
        raise ValueError(f"need >= {K.EEG_MIN_CLEAN_WINDOWS_BASELINE} clean {K.EEG_WIN_S:g}-s windows for a baseline, got {len(feats)} ({q['reasons']})")
    df = pd.DataFrame(feats)
    stats = {c: dict(mean=float(df[c].mean()), sd=float(df[c].std(ddof=1)), n=int(df[c].notna().sum())) for c in df.columns}
    return dict(features=stats, quality=q)


def eeg_strengths(x, fs, names, baseline=None, reference=None, profile=None, table=None):
    """Strength of each EEG feature for a scoring segment.
    baseline: the "eeg" block of baseline.json (preferred; also fixes WHICH channels are used).
    reference: population reference (focus_core.reference), used only when there is no baseline; `profile` (age/sex/habits) makes it
               age/sex/habit-matched (complex score), profile=None gives the plain population reference (simple score).
    table: which feature table to score (default the FOCUS table; the relaxation score passes K.RELAX_EEG_FEATURES).
    Returns (strengths dict, quality dict, median features dict)."""
    table = table or K.EEG_FEATURES
    front = baseline.get("front") if baseline else None
    post = baseline.get("post") if baseline else None
    feats, q = eeg_windows(x, fs, names, front=front, post=post)
    if not feats:
        return {}, q, {}
    df = pd.DataFrame(feats)
    cur = {c: float(df[c].median()) for c in df.columns if df[c].notna().any()}
    entries, basis = {}, "none"
    for f in table:
        if f not in cur:
            continue
        if baseline and f in baseline.get("features", {}):
            b = baseline["features"][f]
            entries[f], basis = dict(mean=b["mean"], sd=b["sd"]), "own baseline"
        elif reference and f in reference.get("features", {}):
            mean, sd, basis = reference_expectation(reference, f, profile)
            entries[f] = dict(mean=mean, sd=sd)
    s = scoring.strengths(cur, table, entries)
    q["basis"] = basis
    return s, q, cur
