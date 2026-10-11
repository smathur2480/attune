"""
focus_core.heart -- heart-rate variability (HRV) from a wearable watch.   *** PLANNED FEATURE: not yet validated with a real watch ***

INPUT (any one of)
  * RR intervals: arrays t (s, beat times) and rr_ms  -- what many watch / health-platform APIs can expose (check yours)
  * raw single-lead ECG strip + sampling rate         -- R-peaks are detected here (rr_from_ecg)
  * pulse intervals from PPG (source="ppg")           -- accepted, but weighted lower (PPG-derived pulse-rate variability)
What a watch really gives you (verify for the device you target):
  * Consumer smartwatch ECGs studied so far are single-lead, 30-second recordings [WatchECG2024] -> spot checks, not continuous.
  * Continuous monitoring on a watch is normally PPG, which is validated at rest and degrades with motion [Taoum2022] [PRVMeta].
FEATURES (per segment)
  hr_mean   = 60000 / mean(RR)                                               beats per minute
  ln_rmssd  = ln( sqrt(mean(diff(RR)^2)) )  over ADJACENT valid beats only    needs >= 30 s; the primary vagal index
  ln_hf     = ln( HF power 0.15-0.4 Hz )  -- ONLY if segment >= 120 s AND the estimated breathing rate >= 9 breaths/min
  resp_cpm  = breathing rate estimated from the RR spectrum (peak 0.07-0.5 Hz)  -- descriptive, needs >= 120 s
Why RMSSD first and HF only sometimes: RMSSD is the standard short-term time-domain vagal index and stays usable on very short recordings;
when someone breathes slower than ~9 breaths/min (e.g. paced breathing at ~6/min = 0.1 Hz) vagally mediated variability moves from the HF
band into the LF band, so HF power would FALL even though vagal activity rises [Laborde2021] [SevozCouche2022].
Baseline: 60-s windows (hop 30 s) over a >= 3-minute resting recording -> per-feature mean/SD, pooled across sessions exactly like the EEG
windows (focus_core/baselines.py). HRV depends on age, fitness, posture, time of day and device, so NO population placeholders exist.

LITERATURE
  [Laborde2017] Laborde S, Mosley E, Thayer JF (2017). Heart rate variability and cardiac vagal tone in psychophysiological research -
      recommendations for experiment planning, data analysis, and data reporting. Front Psychol 8:213. (title/venue from memory; verify)
  [TaskForce1996] Task Force of the European Society of Cardiology and NASPE (1996). Heart rate variability: standards of measurement,
      physiological interpretation and clinical use. (cited from memory; verify) -- short-term recordings of 2-5 min; SDNN/RMSSD time-domain.
  [Shaffer2017] Shaffer F, Ginsberg JP (2017). An overview of heart rate variability metrics and norms. Front Public Health 5:258.
      (venue from memory; verify)
  [Laborde2021] Laborde S, Iskra M, et al. (2021). Slow-paced breathing: influence of inhalation/exhalation ratio and of respiratory pauses on
      cardiac vagal activity. Sustainability 13:7775. Above ~9 breaths/min vagal activity shows in HF HRV; below it shifts to LF.
  [SevozCouche2022] Sevoz-Couche C, Laborde S (2022). Heart rate variability and slow-paced breathing: when coherence meets resonance.
      Neurosci Biobehav Rev. Breathing at the resonant frequency (~0.1 Hz) increases cardiac oscillations / vagally mediated HRV.
  [JMIR2020] "A novel smartphone app for the measurement of ultra-short-term and short-term heart rate variability: validity and reliability
      study", JMIR mHealth uHealth 2020;8(7):e18761 (authors not retrieved): 1-min recordings suggested for time-domain indices.
  [SmartwatchHRV2019] arXiv:1907.07496, "Improving heart rate variability measurements from consumer smartwatches with machine learning"
      (authors not retrieved): RMSSD reported reliable on 10/30/60-s samples, unlike SDNN or frequency measures; wrist trackers and ECG
      sometimes disagree.
  [WatchECG2024] Computing in Cardiology 2024 abstract 419 (authors not retrieved): Withings ScanWatch and Fitbit Sense single-lead 30-s ECGs vs
      medical-grade ECG in 20 people: almost identical mean HR and HRV.
  [Taoum2022] Taoum A, Bisiaux A, Tilquin F, Le Guillou Y, et al. (2022). Validity of ultra-short-term HRV analysis using PPG - a preliminary
      study. Sensors 22(18):7047 (5 people, 18 nights; PPG accuracy validated at rest, lower with motion).
  [PRVMeta] "Accuracy of photoplethysmography-derived pulse rate variability compared with electrocardiography-derived heart rate variability:
      a systematic review and meta-analysis" (authors not retrieved): PRV can differ from HRV through pulse transit time, vascular tone, motion
      and sensor contact.
CAUTION: artefact rules (300-2000 ms, > 20% jump from the local median) are common conventions chosen here, not tuned on watch data.
"""
import math

import numpy as np
from scipy import interpolate, signal
from scipy.integrate import trapezoid

from . import constants as K


# ---------------------------------------------------------------------------------------------------------------------
# beats
# ---------------------------------------------------------------------------------------------------------------------
def rr_from_ecg(ecg, fs):
    """R-peak detection for a single-lead strip (simplified Pan-Tompkins: band-pass 5-15 Hz, squared derivative, 150-ms integration,
    adaptive threshold, 250-ms refractory, peak refinement). Returns (beat_times_s, rr_ms) -- rr_ms[i] is the interval ENDING at beat i+1."""
    x = np.asarray(ecg, float)
    x = x - np.median(x)
    sos = signal.butter(3, [5.0, min(15.0, 0.45 * fs)], btype="band", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)
    d = np.gradient(y) ** 2
    w = max(1, int(0.15 * fs))
    integ = np.convolve(d, np.ones(w) / w, mode="same")
    thr = 0.3 * np.percentile(integ, 99)
    cand, _ = signal.find_peaks(integ, height=thr, distance=int(0.25 * fs))
    beats = []
    half = int(0.06 * fs)
    for c in cand:
        a, b = max(0, c - 2 * half), min(len(y), c + half)
        beats.append(a + int(np.argmax(np.abs(y[a:b]))))
    beats = np.unique(np.array(beats, int))
    t = beats / fs
    return t, np.diff(t) * 1000.0


def rr_from_beats(t):
    t = np.asarray(t, float)
    return np.diff(t) * 1000.0


# ---------------------------------------------------------------------------------------------------------------------
# cleaning and features
# ---------------------------------------------------------------------------------------------------------------------
def clean_rr(t_end, rr_ms):
    """t_end[i] = time of the beat that ENDS interval i. Marks an interval valid if 300 <= RR <= 2000 ms and within 20% of the median of
    its neighbours. Returns (valid mask, valid_fraction)."""
    rr = np.asarray(rr_ms, float)
    if len(rr) == 0:
        return np.zeros(0, bool), 0.0
    ok = (rr >= K.RR_MIN_MS) & (rr <= K.RR_MAX_MS)
    med = np.full(len(rr), np.nan)
    for i in range(len(rr)):
        nb = rr[max(0, i - 5):i + 6]
        nb = nb[(nb >= K.RR_MIN_MS) & (nb <= K.RR_MAX_MS)]
        if len(nb):
            med[i] = np.median(nb)
    ok &= np.isfinite(med) & (np.abs(rr - med) <= K.RR_JUMP_FRAC * med)
    return ok, float(ok.mean())


def _adjacent_diffs(rr, ok):
    d = np.diff(rr)
    good = ok[1:] & ok[:-1]                                        # only intervals that were neighbours in the original beat sequence
    return d[good]


def _spectrum(t_end, rr, ok, fs_resample=4.0):
    tt, vv = np.asarray(t_end)[ok], np.asarray(rr)[ok]
    if len(tt) < 16:
        return None
    grid = np.arange(tt[0], tt[-1], 1.0 / fs_resample)
    if len(grid) < int(64 * fs_resample):
        return None
    v = interpolate.interp1d(tt, vv, kind="linear")(grid)
    v = signal.detrend(v)
    f, p = signal.welch(v, fs_resample, nperseg=min(len(v), int(64 * fs_resample)))
    return f, p


def hrv_features(t_end, rr_ms, source="ecg"):
    """Features for one segment. Returns (features dict, quality dict). Features missing for a segment are simply absent."""
    t_end, rr = np.asarray(t_end, float), np.asarray(rr_ms, float)
    ok, vfrac = clean_rr(t_end, rr)
    dur = float(t_end[-1] - (t_end[0] - rr[0] / 1000.0)) if len(rr) else 0.0
    q = dict(source=source, duration_s=round(dur, 1), n_beats=int(len(rr)), valid_frac=round(vfrac, 3))
    if ok.sum() < 10 or dur < K.HRV_MIN_S:
        q["reason"] = f"needs >= {K.HRV_MIN_S:g} s with >= 10 valid beats"
        return {}, q
    rv = rr[ok]
    feats = dict(hr_mean=float(60000.0 / rv.mean()))
    d = _adjacent_diffs(rr, ok)
    if len(d) >= 8:
        feats["ln_rmssd"] = float(math.log(max(math.sqrt(np.mean(d ** 2)), 1e-3)))
        feats["sdnn_ms"] = float(np.std(rv, ddof=1))                      # descriptive only
    if dur >= K.HRV_FREQ_MIN_S:
        sp = _spectrum(t_end, rr, ok)
        if sp is not None:
            f, p = sp
            band = lambda lo, hi: float(trapezoid(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])) if ((f >= lo) & (f < hi)).sum() > 1 else float("nan")   # noqa: E731
            resp = (f >= 0.07) & (f <= 0.5)
            if resp.any():
                feats["resp_cpm"] = float(f[resp][np.argmax(p[resp])] * 60.0)
            if feats.get("resp_cpm", 0.0) >= K.HF_MIN_BREATH_CPM:
                hf = band(0.15, 0.4)
                if np.isfinite(hf) and hf > 0:
                    feats["ln_hf"] = float(math.log(hf))
    return feats, q


def quality_heart(q):
    """Data-quality factor 0..1 for the heart modality: usable-beat share, duration (full at 60 s), and a PPG discount."""
    if "reason" in q:
        return 0.0
    v = float(np.clip(q["valid_frac"] / K.HEART_FULL_VALID_FRAC, 0, 1)) * float(np.clip(q["duration_s"] / 60.0, 0, 1))
    return v * (K.PPG_QUALITY_FACTOR if q.get("source") == "ppg" else 1.0)


def session_stats(t_end, rr_ms, source="ecg", win_s=K.HEART_WINDOW_S, hop_s=K.HEART_HOP_S):
    """Baseline input: per-feature mean/SD/n over 60-s windows of ONE resting recording (>= 3 usable windows)."""
    t_end, rr = np.asarray(t_end, float), np.asarray(rr_ms, float)
    t0 = t_end[0] - rr[0] / 1000.0
    rows = []
    a = t0
    while a + win_s <= t_end[-1] + 1e-6:
        sel = (t_end > a) & (t_end <= a + win_s)
        if sel.sum() >= 10:
            f, q = hrv_features(t_end[sel], rr[sel], source)
            if "ln_rmssd" in f:
                rows.append(f)
        a += hop_s
    if len(rows) < K.HEART_MIN_BASELINE_WINDOWS:
        raise ValueError(f"heart baseline needs >= {K.HEART_MIN_BASELINE_WINDOWS} usable {win_s:g}-s windows "
                         f"(about {int((K.HEART_MIN_BASELINE_WINDOWS - 1) * hop_s + win_s)} s of clean RR data); got {len(rows)}")
    stats = {}
    for name in ("hr_mean", "ln_rmssd", "ln_hf"):
        v = np.array([r[name] for r in rows if name in r])
        if len(v) >= 2:
            stats[name] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), n=int(len(v)))
    return dict(features=stats, quality=dict(n_windows=len(rows), source=source))


def from_arrays(t_end=None, rr_ms=None, ecg=None, fs=None, source="ecg"):
    """Normalise any heart input to (t_end, rr_ms, source). ecg: strip in any unit + sampling rate fs."""
    if ecg is not None:
        t, rr = rr_from_ecg(ecg, fs)
        return t[1:], rr, "ecg"
    return np.asarray(t_end, float), np.asarray(rr_ms, float), source
