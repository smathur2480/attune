"""
focus_core.scoring -- the formulas, implemented once.  See focus_core/constants.py for the written-out formula.

Used by simple_score.py (and therefore complex_score.py, which calls simple_score) and by baseline.py (for quality checks).
"""
import math

import numpy as np

from . import constants as K


def _finite(x):
    return x is not None and isinstance(x, (int, float, np.floating, np.integer)) and math.isfinite(float(x))


# 1. strength of one metric ----------------------------------------------------------------------------------------------
def strength(value, spec, entry=None):
    """strength = clip( direction * (value - mean) / max(sd, floor), -3, +3 ).
    mean/sd: the user's baseline entry {"mean","sd"} if given, else the placeholder mean/sd in the metric spec.
    Returns None when the value or its reference mean/sd is not available."""
    mean, sd = (entry["mean"], entry["sd"]) if entry is not None else (spec.get("mean"), spec.get("sd"))
    if not (_finite(value) and _finite(mean) and _finite(sd)):
        return None
    z = spec["direction"] * (float(value) - float(mean)) / max(float(sd), spec["floor"])
    return float(np.clip(z, -K.STRENGTH_CLIP, K.STRENGTH_CLIP))


def strengths(metrics, table, entries=None):
    """Strength of every metric in `table` that is present in `metrics`. `entries`: {metric: {"mean","sd"}} from a baseline."""
    out = {}
    for name, spec in table.items():
        s = strength(metrics.get(name), spec, (entries or {}).get(name))
        if s is not None:
            out[name] = s
    return out


# 2. modality score ------------------------------------------------------------------------------------------------------
def modality_mean(strs, table):
    """m_modality = sum(w * strength) / sum(w) over the metrics that exist. None if there are none."""
    den = sum(table[k]["weight"] for k in strs)
    return sum(table[k]["weight"] * v for k, v in strs.items()) / den if den else None


# 3. data quality and combination ----------------------------------------------------------------------------------------
def quality_click(n_targets):
    return float(np.clip((n_targets - K.CLICK_Q_ZERO_AT) / (K.CLICK_Q_FULL_AT - K.CLICK_Q_ZERO_AT), 0, 1))


def quality_eye(valid_frac):
    return float(np.clip(valid_frac / K.EYE_FULL_VALID_FRAC, 0, 1))


def quality_eeg(n_clean, n_windows):
    return float(np.clip(n_clean / max(n_windows, 1) / K.EEG_FULL_CLEAN_FRAC, 0, 1))


def combine(mods, quality, base_weights=None):
    """W_k = base_weight_k * quality[k];  m_total = sum(W_k * m_k) / sum(W_k).   base_weights: K.BASE_WEIGHTS (focus) or K.RELAX_BASE_WEIGHTS.
    Returns (m_total | None, {modality: normalised weight actually used})."""
    bw = base_weights or K.BASE_WEIGHTS
    w = {k: bw[k] * quality.get(k, 0.0) for k, m in mods.items() if m is not None}
    w = {k: v for k, v in w.items() if v > 0}
    tot = sum(w.values())
    if not tot:
        return None, {}
    return sum(w[k] * mods[k] for k in w) / tot, {k: round(v / tot, 4) for k, v in w.items()}


# 4. score mapping -------------------------------------------------------------------------------------------------------
def to_score(m):
    """score = clip(50 + 25 * m, 1, 100)"""
    return float(np.clip(K.SCORE_CENTER + K.SCORE_PER_SD * m, K.SCORE_MIN, K.SCORE_MAX))


# helpers ----------------------------------------------------------------------------------------------------------------
def explain(strs, table, hi=0.5):
    """Plain-language lists: patterns clearly present (>= +hi SD) and clearly absent/reversed (<= -hi SD)."""
    present = [f"{table[k]['text']} ({v:+.1f} SD)" for k, v in sorted(strs.items(), key=lambda t: -t[1]) if v >= hi]
    against = [f"{table[k]['against']} ({v:+.1f} SD)" for k, v in sorted(strs.items(), key=lambda t: t[1]) if v <= -hi]
    return present, against


def formula_description():
    """Machine-readable copy of the formula so the app/website can display exactly what is computed."""
    return {
        "strength": "clip(direction * (value - mean) / max(sd, floor), -%g, +%g)" % (K.STRENGTH_CLIP, K.STRENGTH_CLIP),
        "modality": "m_k = sum(w_metric * strength) / sum(w_metric)",
        "combine": "W_k = base_weight_k * quality_k ; m_total = sum(W_k * m_k) / sum(W_k)",
        "simple_score": "clip(%g + %g * m_total, %g, %g)" % (K.SCORE_CENTER, K.SCORE_PER_SD, K.SCORE_MIN, K.SCORE_MAX),
        "complex_score": "clip(%g + %g * (m_total + shift_substances + shift_asrs), %g, %g)" % (K.SCORE_CENTER, K.SCORE_PER_SD, K.SCORE_MIN, K.SCORE_MAX),
        "base_weights": dict(K.BASE_WEIGHTS),
        "relax_score": "clip(%g + %g * (m_total - drowsy_penalty), %g, %g) with m_total from relax base weights %s" % (
            K.SCORE_CENTER, K.SCORE_PER_SD, K.SCORE_MIN, K.SCORE_MAX, dict(K.RELAX_BASE_WEIGHTS)),
    }
