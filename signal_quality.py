#!/usr/bin/env python3
"""
Real-time EEG signal-quality feedback with "how to fix it" prompts, plus camera-assisted verification.

WHAT IT CAN AND CANNOT DO
  * Without impedance hardware, contact quality is INFERRED from the signal itself (flat/stuck channel, mains-hum pickup,
    muscle noise, drift, jumps, bridged channels, "noise instead of EEG"). That is what this does, per channel, every ~1 s.
  * The camera cannot see electrode contact. It helps by supplying KNOWN events to test the EEG against:
      - blink sync:  every camera-detected blink must show up as a bump in the forehead channels (also gives left/right balance
                     and the EEG<->camera clock offset, which the training mode needs anyway);
      - eyes-closed alpha: when the camera sees the eyes close, back-of-head (or frontal) alpha should rise (Berger effect);
      - head motion: head movement seen by the camera explains motion artefacts ("hold still").
  * Not implemented: recognising the headset on the face and telling you the band is "too high" (needs a headset-specific model).
  If your headset reports electrode impedance, use that directly and treat this as a second opinion.

DEFAULT THRESHOLDS ARE STARTING POINTS. They were not tuned on dry-electrode hardware. Use `percentiles` on lab-quality files
and `calibrate` on a recording you know is good to set thresholds for YOUR device.


LITERATURE (why these checks)
  [Casadei2022]  Casadei V, Ferrero R (2022). Online skin-electrode contact quality monitoring in wearable devices: an EEG
                 application. I2MTC 2022. Contact is monitored from line-frequency interference RELATIVE to a reference channel,
                 which catches electrode displacement, gel drying and pressure changes (hence the relative mains-hum check).
  [Sensors2019]  "Quality assessment of single-channel EEG for wearable devices", Sensors 2019, 19(3):601, doi:10.3390/s19030601
                 (authors not retrieved): wearable EEG is more exposed to movement, blink, muscle and power-line artefacts.
  [vanDijk2022]  van Dijk H, et al. (2022). TDBRAIN database. Scientific Data 9:333 -- data quality was validated by checking that
                 alpha power falls from eyes-closed to eyes-open (basis of the camera-assisted eyes-closed alpha check).
  Dry-electrode noise studies (Purdue SPARC lab, ISCAS paper; title not retrieved) report that more contact pressure and area
  reduce noise by up to 50 dB (basis of the "press firmly" prompt).

USAGE
  python -m focus_core.signal_quality demo                                        # simulated stream with injected faults and the prompts it gives
  python -m focus_core.signal_quality file --eeg rec.csv --fs 250 [--ear eye.csv] # replay a recording through the monitor
  python -m focus_core.signal_quality percentiles --bids /path/TDBRAIN_Dataset_V3_1 --task restEO --limit 40 --channels Fp1,Fp2,F3,F4,O1,O2
  python -m focus_core.signal_quality calibrate --eeg good.csv --fs 250 --out thresholds.json

Python API (what the app calls):
  mon = QualityMonitor(fs=250, names=["Fp1","Fp2","F3","F4","O1","O2"])
  report = mon.update(chunk)        # chunk: (channels, samples) in microvolts; returns a report about once per second, else None
  report["new_prompts"]             # show/speak these;  report["resolved"] -> "looks good now";  report["overall"]["ready"]
"""
import argparse
import json
import math
import os
import sys

try:
    import numpy as np
    import pandas as pd
    import scipy  # noqa: F401
except ImportError as _e:
    raise SystemExit(f"Missing Python package: {_e.name}.\nInstall with:  python -m pip install numpy pandas scipy")
from scipy import signal
from scipy.integrate import trapezoid

FRIENDLY = {"Fp1": "left forehead", "Fp2": "right forehead", "F3": "left upper forehead", "F4": "right upper forehead",
            "AF7": "left outer forehead", "AF8": "right outer forehead",
            "F7": "left temple", "F8": "right temple", "Fz": "centre forehead", "O1": "back of head, left",
            "O2": "back of head, right", "Oz": "back of head, centre", "P3": "left back-top", "P4": "right back-top",
            "Pz": "centre back-top", "A1": "left ear clip", "A2": "right ear clip"}
FRONTAL = {"Fp1", "Fp2", "AF7", "AF8", "F3", "F4", "F7", "F8", "Fz"}
POSTERIOR = {"O1", "O2", "Oz", "P3", "P4", "Pz", "P7", "P8"}

THRESH = dict(flat_std_uv=0.3, rail_frac=0.02, line_mod=0.25, line_strong=0.6, line_rel_factor=5.0, line_rel_min=0.08,
              rms_mod_uv=80.0, rms_strong_uv=150.0, emg_mod=4.0, emg_strong=10.0, drift_mod=15.0, drift_strong=60.0,
              spike_z=7.0, exponent_min=0.2, bridged_corr=0.995, common_mode_corr=0.9)
PENALTY = {"line": 35, "line_strong": 60, "line_rel": 30, "high_amplitude": 25, "high_amplitude_strong": 45, "muscle": 25,
           "muscle_strong": 45, "drift": 25, "drift_strong": 40, "spikes": 25, "noise_like": 35, "bridged": 40}
SUBSUMES = {"muscle": {"noise_like", "spikes", "high_amplitude"}, "line": {"noise_like", "spikes"}, "drift": {"high_amplitude"}}
PERSIST = {"flat": 2, "railed": 2, "reference_ground": 3, "bridged": 2}      # hard faults act fast; soft ones need longer
PRIORITY = ["reference_ground", "flat", "railed", "bridged", "line", "line_rel", "drift", "high_amplitude", "muscle", "spikes", "noise_like"]


def friendly(ch):
    return f"{FRIENDLY[ch]} ({ch})" if ch in FRIENDLY else ch


def bandpass(x, fs, lo, hi):
    hi = min(hi, 0.45 * fs)
    sos = signal.butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def _band(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(trapezoid(p[m], f[m])) if m.sum() > 1 else float("nan")


def detect_mains(x, fs, default=60.0):
    """Pick 50 or 60 Hz by which shows the stronger narrow peak above its neighbours."""
    if fs < 130:
        return default
    f, p = signal.welch(x - x.mean(axis=-1, keepdims=True), fs, nperseg=min(x.shape[-1], int(2 * fs)), axis=-1)
    p = p.mean(0)
    best, score = default, 3.0
    for m in (50.0, 60.0):
        peak = p[(f >= m - 0.6) & (f <= m + 0.6)].mean()
        floor = np.median(p[((f >= m - 6) & (f <= m - 2)) | ((f >= m + 2) & (f <= m + 6))])
        if floor > 0 and peak / floor > score:
            best, score = m, peak / floor
    return best


# ---------------------------------------------------------------------------------------------------------------------
# per-window analysis (stateless)
# ---------------------------------------------------------------------------------------------------------------------
def window_metrics(x, fs, mains_hz=60.0):
    """x: (C, T) raw microvolts for ~4 s. Returns list of per-channel metric dicts and the correlation matrix."""
    C, T = x.shape
    xd = signal.detrend(x, axis=-1, type="linear")
    y = bandpass(xd, fs, 1.0, 45.0)                                            # for amplitude and correlations only
    nper = min(T, int(2 * fs))
    f, P = signal.welch(xd, fs, nperseg=nper, axis=-1)                         # spectra from the unfiltered signal: keeps mains hum
    f0, P0 = signal.welch(xd, fs, nperseg=T, axis=-1)                         # finest resolution for slow drift
    yh = bandpass(xd, fs, 15.0, 45.0) if fs >= 100 else None
    out = []
    for i in range(C):
        raw = x[i]
        m = dict(std_raw=float(raw.std()), rail_frac=float(np.mean(raw == raw.max()) + np.mean(raw == raw.min())) if raw.std() > 0 else 0.0)
        m["rms_uv"] = float(1.4826 * np.median(np.abs(y[i] - np.median(y[i]))))
        tot = _band(f, P[i], 1, 40)
        m["line_ratio"] = (_band(f, P[i], mains_hz - 1, mains_hz + 1) / tot) if (fs > 2 * (mains_hz + 1.5) and tot > 0) else 0.0
        sel = ((f >= 2) & (f <= 4)) | ((f >= 18) & (f <= 28))
        if fs >= 100 and sel.sum() > 4 and np.all(P[i][sel] > 0):
            slope, icpt = np.polyfit(np.log10(f[sel]), np.log10(P[i][sel]), 1)
            hf = (f >= 30) & (f < 45)
            pred = 10 ** (icpt + slope * np.log10(f[hf]))
            m["emg_excess"] = float(np.sum(P[i][hf]) / np.sum(pred))        # ~1 for clean 1/f EEG, >> 1 with muscle noise
            m["exponent"] = float(-slope)
        else:
            m["emg_excess"], m["exponent"] = 1.0, float("nan")
        lowdrift = _band(f0, P0[i], 0.25, 1.0)
        m["drift_ratio"] = lowdrift / _band(f0, P0[i], 1, 30) if _band(f0, P0[i], 1, 30) > 0 else 0.0
        if yh is not None:
            d = np.diff(yh[i])
            m["spike_z"] = float(np.max(np.abs(d - np.median(d))) / (1.4826 * np.median(np.abs(d - np.median(d))) + 1e-9))
        else:
            m["spike_z"] = 0.0
        out.append(m)
    rec = max(int(1.0 * fs), 16)                                         # the newest second must agree: filters short blips
    fr, Pr = signal.welch(xd[:, -rec:], fs, nperseg=rec, axis=-1)
    for i in range(C):
        m = out[i]
        tot = _band(fr, Pr[i], 1, 40)
        m["line_recent"] = (_band(fr, Pr[i], mains_hz - 1.5, mains_hz + 1.5) / tot) if (fs > 2 * (mains_hz + 1.5) and tot > 0) else 0.0
        m["rms_recent"] = float(1.4826 * np.median(np.abs(y[i, -rec:] - np.median(y[i, -rec:]))))
        sel = ((fr >= 2) & (fr <= 4)) | ((fr >= 18) & (fr <= 28))
        if fs >= 100 and sel.sum() > 4 and np.all(Pr[i][sel] > 0):
            sl, ic = np.polyfit(np.log10(fr[sel]), np.log10(Pr[i][sel]), 1)
            hf = (fr >= 30) & (fr < 45)
            m["emg_recent"] = float(np.sum(Pr[i][hf]) / np.sum(10 ** (ic + sl * np.log10(fr[hf])))) if hf.sum() else 1.0
        else:
            m["emg_recent"] = 1.0
        if yh is not None:
            d_all = np.diff(yh[i]); med = np.median(d_all); mad = 1.4826 * np.median(np.abs(d_all - med)) + 1e-9
            m["spike_recent"] = float(np.max(np.abs(d_all[-rec:] - med)) / mad)
        else:
            m["spike_recent"] = 0.0
    with np.errstate(all="ignore"):
        R = np.corrcoef(y) if C > 1 else np.ones((1, 1))
    return out, np.nan_to_num(R)


def assess(x, fs, names, thr=None, mains_hz=60.0):
    """Returns dict: channels {name: score,status,issues,metrics}, global_issues, mean_abs_corr."""
    t = {**THRESH, **(thr or {})}
    mets, R = window_metrics(x, fs, mains_hz)
    C = len(names)
    flat = [m["std_raw"] < t["flat_std_uv"] for m in mets]
    lines = [m["line_ratio"] for m in mets]
    med_line = float(np.median([l for l, fl in zip(lines, flat) if not fl])) if not all(flat) else 0.0
    res, codes_by = {}, {}
    for i, (nm, m) in enumerate(zip(names, mets)):
        m = dict(m)
        for k_long, k_new in (("line_ratio", "line_recent"), ("rms_uv", "rms_recent"), ("emg_excess", "emg_recent"), ("spike_z", "spike_recent")):
            m[k_long + "_eff"] = min(m[k_long], m[k_new])
        codes = {}
        if flat[i]:
            codes["flat"] = 100
        elif m["rail_frac"] > t["rail_frac"]:
            codes["railed"] = 100
        else:
            if m["line_ratio_eff"] > t["line_strong"]:
                codes["line"] = PENALTY["line_strong"]
            elif m["line_ratio_eff"] > t["line_mod"]:
                codes["line"] = PENALTY["line"]
            elif m["line_ratio_eff"] > t["line_rel_min"] and m["line_ratio_eff"] > t["line_rel_factor"] * max(med_line, 1e-6) and C >= 3:
                codes["line_rel"] = PENALTY["line_rel"]
            if m["rms_uv_eff"] > t["rms_strong_uv"]:
                codes["high_amplitude"] = PENALTY["high_amplitude_strong"]
            elif m["rms_uv_eff"] > t["rms_mod_uv"]:
                codes["high_amplitude"] = PENALTY["high_amplitude"]
            if m["emg_excess_eff"] > t["emg_strong"]:
                codes["muscle"] = PENALTY["muscle_strong"]
            elif m["emg_excess_eff"] > t["emg_mod"]:
                codes["muscle"] = PENALTY["muscle"]
            if m["drift_ratio"] > t["drift_strong"]:
                codes["drift"] = PENALTY["drift_strong"]
            elif m["drift_ratio"] > t["drift_mod"]:
                codes["drift"] = PENALTY["drift"]
            if m["spike_z_eff"] > t["spike_z"]:
                codes["spikes"] = PENALTY["spikes"]
            if np.isfinite(m["exponent"]) and m["exponent"] < t["exponent_min"]:
                codes["noise_like"] = PENALTY["noise_like"]
        res[nm] = dict(issues=[], metrics=m)
        codes_by[nm] = codes
    for i in range(C):                                        # bridged channels
        for j in range(i + 1, C):
            if not flat[i] and not flat[j] and R[i, j] > t["bridged_corr"]:
                for nm in (names[i], names[j]):
                    codes_by[nm]["bridged"] = PENALTY["bridged"]
    pen = {}
    for nm, codes in codes_by.items():
        for cause, subs in SUBSUMES.items():                  # a muscle burst also looks like spikes/noise: report the cause only
            if cause in codes:
                for sub in subs:
                    codes.pop(sub, None)
        res[nm]["issues"] = list(codes)
        pen[nm] = sum(codes.values())
    glob = []
    n_line = sum(1 for m, fl in zip(mets, flat) if not fl and min(m["line_ratio"], m["line_recent"]) > t["line_mod"])
    off = R[np.triu_indices(C, 1)] if C > 1 else np.array([0.0])
    mean_corr = float(np.mean(np.abs(off))) if C >= 3 else 0.0
    if C >= 3 and (n_line >= max(3, math.ceil(0.6 * C)) or mean_corr > t["common_mode_corr"]):
        glob.append("reference_ground")
    for nm in names:
        sc = float(np.clip(100 - pen[nm], 0, 100))
        bad = any(i in ("flat", "railed") for i in res[nm]["issues"])
        res[nm]["score"] = sc
        res[nm]["status"] = "bad" if (bad or sc < 25) else "poor" if sc < 50 else "ok" if sc < 75 else "good"
    return dict(channels=res, global_issues=glob, mean_abs_corr=mean_corr)


# ---------------------------------------------------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------------------------------------------------
def prompt_text(code, chans):
    who = ", ".join(friendly(c) for c in chans)
    return {
        "reference_ground": "Check the ear-clip reference and the forehead ground first: press them against clean skin and move hair out of the way. "
                            "Most sensors are picking up the same interference, so fixing this fixes many channels at once.",
        "flat": f"{who}: no signal. Press the sensor firmly against bare skin (part hair underneath it) and check the connector or cable.",
        "railed": f"{who}: signal is stuck at its limit. Re-seat the sensor and make sure it is not touching metal or a loose cable.",
        "bridged": f"{who} read almost identically: the sensors may be touching each other, or the reference/ground is not connected. "
                   "Separate them and check the ear clip.",
        "line": f"{who}: strong mains-power hum, usually poor skin contact. Press firmly, clean the skin, and move away from chargers and power cables.",
        "line_rel": f"{who}: noisier than the other sensors, so its contact is worse. Reposition it and press firmly.",
        "drift": f"{who}: slow drifting, often sweat or a loose fit. Tighten the band and wait about 30 seconds.",
        "high_amplitude": f"{who}: very large signal, usually movement or poor contact. Hold still and re-seat the sensor.",
        "muscle": f"{who}: muscle activity. Relax your forehead, jaw and eyebrows, and do not clench your teeth.",
        "spikes": f"{who}: sudden jumps, the sensor may be shifting. Tighten the band and keep your head still.",
        "noise_like": f"{who}: looks like noise rather than brain signal. Re-seat it and press firmly; if it persists the sensor may be faulty.",
    }[code]


class QualityMonitor:
    """Streaming monitor: feed chunks, get a report about once per hop. Debounced so prompts do not flicker."""

    def __init__(self, fs, names, window_s=4.0, hop_s=1.0, mains_hz=60.0, thresholds=None, persist=2, persist_clear=3,
                 cooldown_s=20.0, max_new_prompts=2, min_good_front=2):
        self.fs, self.names = float(fs), list(names)
        self.N, self.H = int(window_s * fs), int(hop_s * fs)
        self.buf = np.zeros((len(names), 0))
        self.mains_hz, self.thr = mains_hz, thresholds
        self.persist, self.persist_clear, self.cooldown, self.max_new = persist, persist_clear, cooldown_s, max_new_prompts
        self.min_good_front = min_good_front
        self.n_seen, self.since, self.state, self.settle = 0, 0, {}, {}

    def update(self, chunk):
        chunk = np.asarray(chunk, float)
        self.buf = np.concatenate([self.buf, chunk], axis=1)[:, -self.N:]
        self.n_seen += chunk.shape[1]; self.since += chunk.shape[1]
        if self.buf.shape[1] < self.N or self.since < self.H:
            return None
        self.since = 0
        t = self.n_seen / self.fs
        a = assess(self.buf, self.fs, self.names, self.thr, self.mains_hz)
        flagged = {}                                           # (code, channel|None) -> True this window
        for g in a["global_issues"]:
            flagged[(g, None)] = True
        for nm, r in a["channels"].items():
            for code in r["issues"]:
                flagged[(code, nm)] = True
        new, resolved = [], []
        for key in set(flagged) | set(self.state):
            st = self.state.setdefault(key, dict(count=0, clear=0, active=False, last=-1e9, reminders=0))
            if key in flagged:
                st["count"] += 1; st["clear"] = 0
            else:
                st["clear"] += 1; st["count"] = 0
            if not st["active"] and st["count"] >= PERSIST.get(key[0], self.persist + 1):
                st["active"], st["last"], st["reminders"] = True, t, 0
                new.append(key)
            elif st["active"] and key in flagged and t - st["last"] >= self.cooldown and st["reminders"] < 2:
                st["last"] = t; st["reminders"] += 1
                new.append(key)
            elif st["active"] and st["clear"] >= self.persist_clear:
                st["active"] = False
                resolved.append(key)
                if key[0] in ("flat", "railed", "reference_ground", "bridged"):        # transients while a sensor re-seats: do not nag
                    for ch in (self.names if key[1] is None else [key[1]]):
                        self.settle[ch] = t + 6.0
        active = [k for k, s in self.state.items() if s["active"]]
        new = [k for k in new if not (k[1] and self.settle.get(k[1], 0) > t and k[0] not in ("flat", "railed", "bridged"))]
        glob_active = any(k[0] == "reference_ground" for k in active)
        if glob_active:                                          # mains hum / drift on many channels is a symptom of the reference
            new = [k for k in new if k[0] not in ("line", "line_rel", "drift")]
        order = lambda k: (PRIORITY.index(k[0]), k[1] or "")      # noqa: E731
        new.sort(key=order)
        prompts = []
        for code in dict.fromkeys(k[0] for k in new):
            chans = [k[1] for k in new if k[0] == code and k[1]]
            prompts.append(dict(code=code, channels=chans, text=prompt_text(code, chans) if (chans or code == "reference_ground") else ""))
        grouped = {}
        for k in sorted(resolved, key=order):
            grouped.setdefault(k[1], []).append(k[0].replace("_", " "))
        res_msgs = ["%s: looks good now (%s cleared)." % (friendly(ch) if ch else "The reference/ground", ", ".join(codes))
                    for ch, codes in grouped.items()]
        front_ok = [n for n in self.names if n in FRONTAL and a["channels"][n]["status"] in ("good", "ok")]
        n_front = sum(1 for n in self.names if n in FRONTAL)
        scores = [a["channels"][n]["score"] for n in self.names]
        overall_status = "good" if min(scores) >= 75 else "ok" if np.mean(scores) >= 50 else "poor"
        ready = len(front_ok) >= min(self.min_good_front, n_front) and not glob_active
        return dict(t=round(t, 2), overall=dict(status=overall_status, mean_score=round(float(np.mean(scores)), 1), ready=bool(ready),
                                                 quality=round(float(np.mean(scores)) / 100.0, 3)),
                    channels={n: dict(score=a["channels"][n]["score"], status=a["channels"][n]["status"], issues=a["channels"][n]["issues"])
                              for n in self.names},
                    global_issues=a["global_issues"], new_prompts=prompts[:self.max_new], queued_prompts=prompts[self.max_new:],
                    resolved=res_msgs, active_issues=[dict(code=k[0], channel=k[1]) for k in sorted(active, key=order)])


# ---------------------------------------------------------------------------------------------------------------------
# camera-assisted checks
# ---------------------------------------------------------------------------------------------------------------------
def usable_channels(report, min_status="ok"):
    """Channels whose latest status is at least min_status ('good' or 'ok'): pass these on to focus_patterns / training so that
    bad sensors do not feed the focus score. Returns (names, quality 0-1)."""
    allowed = {"good"} if min_status == "good" else {"good", "ok"}
    names = [c for c, v in report["channels"].items() if v["status"] in allowed]
    return names, report["overall"]["quality"]


def blink_times_from_ear(ear_df, closed_frac=0.6, depth=0.4):
    t = ear_df["t"].to_numpy(float); e = ear_df["ear"].to_numpy(float)
    ok = np.isfinite(e) & ((ear_df["face_ok"].to_numpy(float) > 0) if "face_ok" in ear_df else True)
    if ok.sum() < 30:
        return np.array([]), np.nan
    open_lvl = float(np.nanpercentile(e[ok], 90))
    below = np.nan_to_num(np.where(ok, e, np.nan), nan=np.inf) < closed_frac * open_lvl
    d = np.diff(np.r_[0, below.astype(int), 0])
    out = []
    for s, en in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        dur = t[min(en, len(t) - 1)] - t[s]
        if np.nanmin(e[s:en]) < depth * open_lvl and 0.05 <= dur <= 1.0:
            out.append(t[s + int(np.nanargmin(e[s:en]))])
    return np.array(out), open_lvl


def _epoch_avg(y, fs, t0, events, lo=-0.4, hi=0.4):
    a, b = int(lo * fs), int(hi * fs)
    eps = []
    for ev in events:
        k = int(round((ev - t0) * fs))
        if k + a >= 0 and k + b < len(y):
            seg = y[k + a:k + b]
            eps.append(seg - seg[:int(0.1 * fs)].mean())
    return (np.mean(eps, 0), len(eps)) if eps else (None, 0)


def blink_sync_check(eeg, fs, t0, ear_df, search_s=0.3, min_blinks=6, n_null=150, seed=0, z_ok=4.0, sym_range=(0.5, 2.0)):
    """eeg: {channel: 1-D array in uV, starting at EEG time t0}; ear_df: camera t, ear on the SAME clock.
    Tests that camera-detected blinks appear in the (frontal) EEG, estimates the EEG<->camera lag, and compares left/right."""
    blinks, _ = blink_times_from_ear(ear_df)
    if len(blinks) < min_blinks:
        return dict(status="need_blinks", n_blinks=int(len(blinks)),
                    prompt="Blink normally for about 15 seconds so the forehead sensors can be verified.")
    rng = np.random.default_rng(seed)
    ys = {c: bandpass(np.asarray(x, float), fs, 0.5, 10.0) for c, x in eeg.items()}
    dur = len(next(iter(ys.values()))) / fs
    def amp(y, ev):                                            # size of the blink-locked average AT the event time (+-40 ms)
        r, n_ev = _epoch_avg(y, fs, t0, ev)
        if r is None:
            return float("nan")
        mid = len(r) // 2
        w = max(1, int(0.02 * fs))
        return float(np.max(np.abs(r[mid - w:mid + w + 1])))
    # lag = where the blink-locked average (all channels, rectified, smoothed) peaks, searched within +-search_s
    a, b = int(-0.4 * fs), int(0.4 * fs)
    avgs = []
    for y in ys.values():
        r, _n = _epoch_avg(y, fs, t0, blinks)
        if r is not None:
            avgs.append(np.abs(r) / (np.std(r) + 1e-9))
    prof = np.mean(avgs, 0) if avgs else np.zeros(b - a)
    k = max(1, int(0.04 * fs))
    prof = np.convolve(prof, np.ones(k) / k, mode="same")
    mid = len(prof) // 2
    lo_i, hi_i = mid - int(search_s * fs), mid + int(search_s * fs)
    lag = float((np.argmax(prof[lo_i:hi_i]) + lo_i - mid) / fs)
    chans = {}
    valid_t = np.arange(t0 + 1.0, t0 + dur - 1.5, 0.05)
    valid_t = valid_t[[np.min(np.abs(blinks - v)) > 1.0 for v in valid_t]] if len(valid_t) else valid_t
    for c, y in ys.items():
        a_obs = amp(y, blinks + lag)
        null = [amp(y, rng.choice(valid_t, len(blinks), replace=False)) for _ in range(n_null)] if len(valid_t) > len(blinks) else [np.nan]
        mu, sd = float(np.nanmean(null)), float(np.nanstd(null) + 1e-9)
        z = float((a_obs - mu) / sd)
        chans[c] = dict(amp_uv=round(a_obs, 1), z=round(z, 1), sees_blinks=bool(z >= z_ok))
    sym = {}
    pairs = (("Fp1", "Fp2"), ("AF7", "AF8"), ("F3", "F4"), ("F7", "F8"), ("O1", "O2"))
    prompts = []
    for l, r in pairs:
        if l in chans and r in chans and max(chans[l]["amp_uv"], chans[r]["amp_uv"]) > 10:
            ratio = chans[l]["amp_uv"] / max(chans[r]["amp_uv"], 1e-6)
            sym[f"{l}/{r}"] = round(ratio, 2)
            if ratio < sym_range[0]:
                prompts.append(f"{friendly(l)} registers blinks much more weakly than {friendly(r)}: its contact is probably worse. Re-seat and press it firmly.")
            elif ratio > sym_range[1]:
                prompts.append(f"{friendly(r)} registers blinks much more weakly than {friendly(l)}: its contact is probably worse. Re-seat and press it firmly.")
    for c in chans:
        if c in ("Fp1", "Fp2") and not chans[c]["sees_blinks"]:
            prompts.append(f"{friendly(c)} did not register your blinks. Re-seat it and press firmly against the skin.")
    sync = "ok" if abs(lag) <= 0.15 else "check_clock_sync"
    return dict(status="ok" if not prompts else "fix", n_blinks=int(len(blinks)), lag_s=round(lag, 3), clock_sync=sync,
                channels=chans, symmetry=sym, prompts=prompts)


def _segments(t, mask, min_s, trim_s):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    out = []
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        a, b = t[s] + trim_s, t[min(e, len(t) - 1)] - trim_s
        if b - a >= min_s - 2 * trim_s:
            out.append((a, b))
    return out


def eyes_closed_alpha_check(eeg, fs, t0, ear_df, min_seg_s=3.0, trim_s=0.5, min_db=1.5):
    """Alpha (8-12 Hz) should rise when the camera sees the eyes close. Uses posterior channels if present, else all."""
    t = ear_df["t"].to_numpy(float); e = ear_df["ear"].to_numpy(float)
    ok = np.isfinite(e) & ((ear_df["face_ok"].to_numpy(float) > 0) if "face_ok" in ear_df else True)
    if ok.sum() < 30:
        return dict(status="no_camera", prompt="Face the camera so the eye-closure check can run.")
    op = float(np.nanpercentile(e[ok], 90))
    closed = _segments(t, ok & (e < 0.3 * op), min_seg_s, trim_s)
    opened = _segments(t, ok & (e > 0.8 * op), min_seg_s, trim_s)
    if not closed:
        return dict(status="need_eyes_closed", prompt="Close your eyes for about 6 seconds, then open them. This verifies the sensors.")
    if not opened:
        return dict(status="need_eyes_open", prompt="Keep your eyes open and still for about 6 seconds.")
    chans = [c for c in eeg if c in POSTERIOR] or list(eeg)

    def psd(segs, c):
        ps = []
        for a, b in segs:
            seg = np.asarray(eeg[c][int((a - t0) * fs):int((b - t0) * fs)], float)
            if len(seg) >= int(1.5 * fs):
                f, p = signal.welch(signal.detrend(seg), fs, nperseg=min(len(seg), int(2 * fs)))
                ps.append(_band(f, p, 8, 12))
        return float(np.mean(ps)) if ps else float("nan")
    res = {}
    for c in chans:
        pc, po = psd(closed, c), psd(opened, c)
        res[c] = dict(db=round(10 * math.log10(pc / po), 2) if pc > 0 and po > 0 else float("nan"))
        res[c]["ok"] = bool(res[c]["db"] >= min_db) if np.isfinite(res[c]["db"]) else False
    bad = [c for c, r in res.items() if not r["ok"]]
    where = "back-of-head" if any(c in POSTERIOR for c in chans) else "these"
    return dict(status="ok" if not bad else "fix", channels=res, n_closed=len(closed), n_open=len(opened),
                prompts=[f"{friendly(c)} did not show the usual rise in alpha when your eyes closed. Part the hair under the {where} sensors and press them firmly."
                         for c in bad])


def head_motion_check(cam_df, window_s=10.0, max_speed=0.15, max_roll_deg_s=25.0):
    """cam_df needs t, head_x, head_y (0-1 image coords) and optionally head_roll_deg (camera_capture.py writes them)."""
    d = cam_df[cam_df["t"] >= cam_df["t"].max() - window_s].dropna(subset=["head_x", "head_y"])
    if len(d) < 10:
        return dict(status="no_data")
    dt = np.diff(d["t"].to_numpy(float))
    v = np.hypot(np.diff(d["head_x"].to_numpy(float)), np.diff(d["head_y"].to_numpy(float))) / np.maximum(dt, 1e-3)
    rms = float(np.sqrt(np.mean(v ** 2)))
    out = dict(status="ok", rms_speed=round(rms, 3))
    if "head_roll_deg" in d:
        out["rms_roll_deg_s"] = round(float(np.sqrt(np.mean((np.diff(d["head_roll_deg"].to_numpy(float)) / np.maximum(dt, 1e-3)) ** 2))), 2)
    if rms > max_speed or out.get("rms_roll_deg_s", 0) > max_roll_deg_s:
        out["status"] = "moving"
        out["prompt"] = "You are moving your head a lot, which creates electrode artefacts. Hold still for a moment."
    return out


def camera_ok_check(cam_df, min_face=0.8):
    frac = float(np.mean((cam_df["face_ok"] > 0) if "face_ok" in cam_df else np.isfinite(cam_df["ear"])))
    return dict(status="ok" if frac >= min_face else "fix", face_visible=round(frac, 2),
                prompt=None if frac >= min_face else "Your face is not clearly visible to the camera: face it, improve the lighting and keep your face in frame.")


# ---------------------------------------------------------------------------------------------------------------------
# simulation (demo + tests)
# ---------------------------------------------------------------------------------------------------------------------
def pink(n, fs, rng, exponent=1.0, rms=10.0):
    f = np.fft.rfftfreq(n, 1 / fs)
    spec = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) / np.maximum(f, 0.5) ** (exponent / 2)
    spec[f < 0.5] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std() * rms


def make_eeg(names=("Fp1", "Fp2", "F3", "F4", "O1", "O2"), fs=250, dur=60.0, seed=0, blink_times=(), blink_amp=None, eyes_closed=()):
    """Plausible multichannel EEG: 1/f background, alpha (strong at the back and when eyes are closed), optional blink artefacts."""
    rng = np.random.default_rng(seed)
    n = int(dur * fs); t = np.arange(n) / fs
    amps = blink_amp or {"Fp1": 120.0, "Fp2": 110.0, "F3": 40.0, "F4": 38.0}
    closed = np.zeros(n, bool)
    for a, b in eyes_closed:
        closed[int(a * fs):int(b * fs)] = True
    x = []
    for nm in names:
        post = nm in POSTERIOR
        y = pink(n, fs, rng, 1.0, 10.0)
        al = (8.0 if post else 2.5) * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 6))
        y += al * np.where(closed, 3.0, 1.0)
        for b in blink_times:
            if nm in amps:
                y += amps[nm] * np.exp(-0.5 * ((t - b) / 0.06) ** 2)
        x.append(y)
    return np.array(x), float(fs), list(names)


def ear_trace(duration, blink_times, closed=(), fps=30, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / fps)
    ear = 0.30 + rng.normal(0, 0.006, t.size)
    for b in blink_times:
        ear -= 0.25 * np.exp(-0.5 * ((t - b) / (0.15 / 2.42)) ** 2)
    for a, b in closed:
        ear[(t >= a) & (t < b)] = 0.04
    return pd.DataFrame(dict(t=t, ear=ear, face_ok=1))


def cmd_demo(a):
    fs, names = 250, ["Fp1", "Fp2", "F3", "F4", "O1", "O2"]
    x, _, _ = make_eeg(names, fs, 70, seed=1)
    rng = np.random.default_rng(5)
    tt = np.arange(x.shape[1]) / fs
    ev = []
    def seg(a, b):
        return (tt >= a) & (tt < b)
    x[0, seg(10, 38)] = 0.05 * rng.normal(size=int(seg(10, 38).sum()))                                  # Fp1 sensor lifts off
    ev.append("10 s: Fp1 sensor lifts off (flat)")
    x[2] += np.where(seg(20, 50), 40 * np.sin(2 * np.pi * 60 * tt), 0)                                  # F3 hum
    ev.append("20 s: F3 picks up 60 Hz hum (poor contact)")
    for i in range(4):                                                                                    # jaw clench
        x[i] += np.where(seg(30, 36), bandpass(rng.normal(size=tt.size), fs, 30, 45) * 40, 0)
    ev.append("30-36 s: jaw/forehead muscle burst")
    for i in range(6):                                                                                    # reference clip comes off
        x[i] += np.where(seg(55, 70), 60 * np.sin(2 * np.pi * 60 * tt), 0) + np.where(seg(55, 70), 80 * np.sin(2 * np.pi * 0.4 * tt), 0)
    ev.append("55 s: ear-clip reference comes off (hum + drift on every channel)")
    print("Simulated 70 s stream. Injected faults:\n  " + "\n  ".join(ev) + "\n\nWhat the monitor says (time, message):\n")
    mon = QualityMonitor(fs, names, cooldown_s=25)
    for k in range(0, x.shape[1], int(0.25 * fs)):
        r = mon.update(x[:, k:k + int(0.25 * fs)])
        if r is None:
            continue
        for p in r["new_prompts"]:
            print(f"  {r['t']:5.1f}s  PROMPT   [{p['code']}] {p['text']}")
        for m in r["resolved"]:
            print(f"  {r['t']:5.1f}s  RESOLVED {m}")
    print("\nCamera checks on a simulated 60 s clip (EEG lags the camera by 80 ms, Fp1 contact is weak):")
    blinks = np.arange(3, 57, 4.0)
    x2, _, _ = make_eeg(names, fs, 60, seed=2, blink_times=blinks + 0.08, blink_amp={"Fp1": 35.0, "Fp2": 110.0, "F3": 40.0, "F4": 38.0},
                        eyes_closed=[(20, 28), (40, 48)])
    ear = ear_trace(60, blinks, closed=[(20, 28), (40, 48)])
    b = blink_sync_check({n: x2[i] for i, n in enumerate(names)}, fs, 0.0, ear)
    print(f"  blink sync: lag {b['lag_s']:+.2f} s, left/right ratios {b['symmetry']}")
    for p in b["prompts"]:
        print("   -", p)
    al = eyes_closed_alpha_check({n: x2[i] for i, n in enumerate(names)}, fs, 0.0, ear)
    db_txt = ", ".join("%s %+.1f dB" % (c, r["db"]) for c, r in al["channels"].items())
    print("  eyes-closed alpha: %s (%s)" % (al["status"], db_txt))
    for p in al["prompts"]:
        print("   -", p)


def _load(a):
    from . import fileio as fio
    x, fs, names = fio.load_eeg(a.eeg, a.fs)
    if a.channels:
        keep = a.channels.split(",")
        x = x[[names.index(c) for c in keep]]; names = keep
    return x, fs, names


def cmd_file(a):
    x, fs, names = _load(a)
    mon = QualityMonitor(fs, names, mains_hz=a.mains)
    stat = {n: {"good": 0, "ok": 0, "poor": 0, "bad": 0} for n in names}
    issues, n = {}, 0
    for k in range(0, x.shape[1], int(0.5 * fs)):
        r = mon.update(x[:, k:k + int(0.5 * fs)])
        if r is None:
            continue
        n += 1
        for c, v in r["channels"].items():
            stat[c][v["status"]] += 1
            for i in v["issues"]:
                issues[(c, i)] = issues.get((c, i), 0) + 1
        for p in r["new_prompts"]:
            print(f"  {r['t']:6.1f}s  [{p['code']}] {p['text']}")
    print(f"\n{n} one-second reports. Share of time each channel was good / ok / poor / bad:")
    for c, s in stat.items():
        tot = max(sum(s.values()), 1)
        print(f"  {c:5s} " + "  ".join(f"{k} {100 * v / tot:3.0f}%" for k, v in s.items()))
    if issues:
        print("Most frequent issues:", sorted(((f"{c}:{i}", v) for (c, i), v in issues.items()), key=lambda t: -t[1])[:6])
    if a.ear:
        ear = pd.read_csv(a.ear)
        eeg = {nm: x[i] for i, nm in enumerate(names)}
        checks = [("camera", lambda: camera_ok_check(ear)), ("blink sync", lambda: blink_sync_check(eeg, fs, a.t0, ear)),
                  ("eyes-closed alpha", lambda: eyes_closed_alpha_check(eeg, fs, a.t0, ear))]
        if {"head_x", "head_y"} <= set(ear.columns):
            checks.append(("head motion", lambda: head_motion_check(ear)))
        for title, fn in checks:
            r = fn(); print(f"\n{title}: {r.get('status')}", {k: v for k, v in r.items() if k not in ('status', 'channels')})


def cmd_percentiles(a):
    import glob, re
    import mne
    mne.set_log_level("ERROR")
    chans = a.channels.split(",")
    files = sorted(glob.glob(os.path.join(a.bids, "sub-*", "ses-*", "eeg", f"*_task-{a.task}_eeg.bdf")))[:a.limit or None]
    rows = []
    for p in files:
        raw = mne.io.read_raw_bdf(p, preload=True, verbose="ERROR")
        if any(c not in raw.ch_names for c in chans):
            continue
        raw.pick(chans); raw.resample(250.0, verbose="ERROR")
        x = raw.get_data() * 1e6
        for s in range(int(2 * 250), x.shape[1] - 4 * 250, 4 * 250):
            mets, _ = window_metrics(x[:, s:s + 4 * 250], 250.0, a.mains)
            rows += [m for m in mets]
    df = pd.DataFrame(rows)
    print(f"{len(files)} recordings, {len(df)} channel-windows (lab-quality gel caps: expect FEW real faults)")
    print(df.quantile([0.5, 0.9, 0.95, 0.99]).round(3).T.to_string())
    print("\nCurrent defaults:", {k: v for k, v in THRESH.items() if k in ('line_mod', 'emg_mod', 'drift_mod', 'rms_mod_uv', 'spike_z', 'exponent_min')})


def cmd_calibrate(a):
    x, fs, names = _load(a)
    rows = []
    for s in range(0, x.shape[1] - int(4 * fs), int(2 * fs)):
        rows += window_metrics(x[:, s:s + int(4 * fs)], fs, a.mains)[0]
    df = pd.DataFrame(rows)
    q99 = df.quantile(0.99)
    thr = dict(line_mod=float(max(THRESH["line_mod"] * 0.2, q99["line_ratio"] * 2.0)), emg_mod=float(max(2.0, q99["emg_excess"] * 1.5)),
               drift_mod=float(max(5.0, q99["drift_ratio"] * 2.0)), rms_mod_uv=float(max(40.0, q99["rms_uv"] * 1.8)),
               spike_z=float(max(10.0, q99["spike_z"] * 1.5)))
    thr["line_strong"], thr["emg_strong"] = thr["line_mod"] * 2.4, thr["emg_mod"] * 2.4
    thr["drift_strong"], thr["rms_strong_uv"] = thr["drift_mod"] * 4, thr["rms_mod_uv"] * 1.9
    with open(a.out, "w") as fh:
        json.dump(thr, fh, indent=1)
    print(f"wrote {a.out} from {len(df)} channel-windows of a recording you said was good:\n", json.dumps(thr, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("demo")
    f = sp.add_parser("file"); f.add_argument("--eeg", required=True); f.add_argument("--fs", type=float); f.add_argument("--channels")
    f.add_argument("--ear"); f.add_argument("--t0", type=float, default=0.0); f.add_argument("--mains", type=float, default=60.0)
    p = sp.add_parser("percentiles"); p.add_argument("--bids", required=True); p.add_argument("--task", default="restEO")
    p.add_argument("--limit", type=int, default=40); p.add_argument("--channels", default="Fp1,Fp2,F3,F4,O1,O2")
    p.add_argument("--mains", type=float, default=50.0)
    c = sp.add_parser("calibrate"); c.add_argument("--eeg", required=True); c.add_argument("--fs", type=float); c.add_argument("--channels")
    c.add_argument("--mains", type=float, default=60.0); c.add_argument("--out", default="thresholds.json")
    a = ap.parse_args()
    {"demo": cmd_demo, "file": cmd_file, "percentiles": cmd_percentiles, "calibrate": cmd_calibrate}[a.cmd](a)


if __name__ == "__main__":
    main()
