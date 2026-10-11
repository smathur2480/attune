"""focus_core.fileio -- file loading/saving shared by the three scripts (CSV / EDF / BDF EEG, JSON, eye and event CSVs)."""
import json
import os
import tempfile

import numpy as np
import pandas as pd


def load_eeg(path, fs=None, channels=None):
    """-> (x (C, T) in microvolts, fs, names). CSV: header = channel names, one row per sample, needs fs. EDF/BDF: needs mne."""
    p = path.lower()
    if p.endswith((".edf", ".bdf")):
        import mne
        raw = (mne.io.read_raw_bdf if p.endswith(".bdf") else mne.io.read_raw_edf)(path, preload=True, verbose="ERROR")
        x, fs, names = raw.get_data() * 1e6, float(raw.info["sfreq"]), list(raw.ch_names)
    else:
        df = pd.read_csv(path)
        if fs is None:
            raise ValueError("fs (sampling rate in Hz) is required for CSV EEG input")
        x, fs, names = df.to_numpy(float).T, float(fs), list(df.columns)
    if channels:
        x = x[[names.index(c) for c in channels]]
        names = list(channels)
    return x, fs, names


def load_eye(path):
    df = pd.read_csv(path)
    for c in ("t", "ear"):
        if c not in df.columns:
            raise ValueError(f"{path}: eye CSV needs columns t, ear (and optionally gaze_x, gaze_y, face_ok)")
    return df


def read_csv(path):
    return pd.read_csv(path)


def load_events(path):
    """events.csv with columns type (change|click) and t (s) -> (changes, clicks) as float arrays."""
    ev = pd.read_csv(path)
    return ev[ev["type"] == "change"].t.to_numpy(float), ev[ev["type"] == "click"].t.to_numpy(float)


def load_rr(path):
    """CSV with columns t (beat time in s) and rr_ms (interval ending at that beat) -> dict for baseline_ops / relax_score."""
    df = pd.read_csv(path)
    if not {"t", "rr_ms"} <= set(df.columns):
        raise ValueError(f"{path}: RR CSV needs columns t, rr_ms")
    return dict(t_end=df["t"].to_numpy(float), rr_ms=df["rr_ms"].to_numpy(float))


def load_ecg(path, fs):
    """CSV with a column named ecg (or a single unnamed column) -> dict(ecg=..., fs=...)."""
    df = pd.read_csv(path)
    col = "ecg" if "ecg" in df.columns else df.columns[0]
    return dict(ecg=df[col].to_numpy(float), fs=float(fs))


def read_json(path):
    if not path:
        return None
    with open(path) as fh:
        return json.load(fh)


def write_json(obj, path):
    """Atomic write so a crash never leaves a half-written baseline.json in the repo."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(tmp, path)
