"""
focus_core.baselines -- how a personal baseline is built and stored (baseline.json). One rule set for every modality.

baseline.json
  eeg       {fs, front, post, excluded, sessions[{features{f:{mean,sd,n}}}], features{f:{mean,sd,n,n_sessions}}}
  eeg_closed same as eeg, from an EYES-CLOSED resting recording (relaxation score)
  heart_rest {sessions[{features{f:{mean,sd,n}}}], features{...}}  HRV from a watch, 60-s windows pooled like the EEG (PLANNED feature)
  eye_rest  {open_level, roi, sessions[{values{}}], features{metric:{mean,sd,sd_source,n_sessions}}}      (passive monitoring)
  training  {sessions[{values{}}], features{metric:{...}}}                                                (click + pursuit task)

RULES (used for every modality, so scores are comparable)
  mean : EEG = mean over ALL clean windows of all sessions; eye / click = mean of the per-session values.
  SD   : EEG = SD over all clean windows of all sessions (exact pooled variance: within-session + between-session parts).
         eye / click = BETWEEN-SESSION SD once >= 3 sessions exist, otherwise the PLACEHOLDER SD from constants.py.
  At scoring time the SD is floored at the metric's `floor`, and strengths are clipped to +-3 (focus_core.scoring).

LITERATURE (why a personal baseline)
  [Schleicher2008]  substantial inter-individual differences in blink/saccade variables (Ergonomics 2008).
  [VanOrden2000]    individualised eye-activity models derived in one session predicted performance in another session.
  [Shelat2024]      within-person response-time boundaries used to infer attentional state (Frontiers in Cognition 3:1460349).
"""
import datetime as _dt

import numpy as np

from . import constants as K


def now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_baseline():
    return dict(schema=K.SCHEMA_BASELINE, created_utc=now(), updated_utc=now(), eeg=None, eye_rest=None, training=None)


def pooled_eeg(sessions):
    """Exact mean and SD over the union of windows of several sessions, from each session's (mean, sd, n)."""
    out = {}
    feats = sorted({f for s in sessions for f in s["features"]})
    for f in feats:
        parts = [s["features"][f] for s in sessions if f in s["features"] and s["features"][f]["n"] >= 2]
        n = sum(p["n"] for p in parts)
        mean = sum(p["n"] * p["mean"] for p in parts) / n
        ss = sum((p["n"] - 1) * p["sd"] ** 2 + p["n"] * (p["mean"] - mean) ** 2 for p in parts)
        out[f] = dict(mean=float(mean), sd=float(np.sqrt(ss / max(n - 1, 1))), n=int(n), n_sessions=len(parts))
    return out


def metric_entries(sessions, table):
    """Per-metric {mean, sd, sd_source, n_sessions} from per-session values (eye / click metrics)."""
    out = {}
    for name, spec in table.items():
        vals = [s["values"][name] for s in sessions if name in s["values"] and np.isfinite(s["values"][name])]
        if not vals:
            continue
        if len(vals) >= K.MIN_SESSIONS_PERSONAL_SD:
            sd, src = float(np.std(vals, ddof=1)), "between-session"
        else:
            sd, src = float(spec["sd"]), "placeholder"
        out[name] = dict(mean=float(np.mean(vals)), sd=sd, sd_source=src, n_sessions=len(vals))
    return out


def summary(b):
    """Human-readable lines describing what a baseline contains and how trustworthy each part is."""
    lines = [f"schema {b.get('schema')}   created {b.get('created_utc')}   updated {b.get('updated_utc')}"]
    e = b.get("eeg")
    lines.append("EEG       : none" if not e else
                 f"EEG       : {len(e['sessions'])} session(s), channels front={e['front']} post={e['post']}, excluded={e.get('excluded', {})}")
    ec = b.get("eeg_closed")
    lines.append("EEG closed: " + ("none (needed for the relaxation score with eyes closed)" if not ec else
                                   f"{len(ec['sessions'])} session(s), channels front={ec['front']} post={ec['post']}"))
    hr = b.get("heart_rest")
    lines.append("heart rest: " + ("none (planned watch feature)" if not hr else f"{len(hr['sessions'])} session(s), features {sorted(hr['features'])}"))
    r = b.get("eye_rest")
    if r:
        srcs = sorted({v["sd_source"] for v in r["features"].values()})
        lines.append(f"eye (rest): {len(r['sessions'])} session(s), SD source {srcs}, on-task region {r.get('roi')}")
    else:
        lines.append("eye (rest): none (population placeholders will be used)")
    t = b.get("training")
    if t:
        srcs = sorted({v["sd_source"] for v in t["features"].values()})
        lines.append(f"training  : {len(t['sessions'])} session(s), SD source {srcs}"
                     + ("" if len(t["sessions"]) >= K.MIN_SESSIONS_PERSONAL_SD else f"  (personal SD from {K.MIN_SESSIONS_PERSONAL_SD} sessions)"))
    else:
        lines.append("training  : none (population placeholders will be used)")
    return lines
