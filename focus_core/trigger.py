"""
focus_core.trigger -- when to OFFER the training mode from the passive focus score (the user can also start it manually).

Rule: offer training when the smoothed passive score stays below a threshold for `min_seconds` in a row. The threshold is the user's own
15th percentile of past passive scores once >= 30 exist (a within-person boundary), otherwise an absolute 35.

LITERATURE
  [Shelat2024] Shelat S, Schooler JW, Giesbrecht B (2024). Predicting attentional lapses using response time speed in continuous
      performance tasks. Frontiers in Cognition 3:1460349. Real-time triggering designs have used simple within-person boundaries to
      isolate suboptimal attention -- the personal-percentile threshold follows that idea (the 15th percentile and 45 s are untuned choices).
"""
import numpy as np

from .eye import runs


def should_suggest_training(times, scores, personal_scores=None, abs_threshold=35.0, personal_pct=15.0, min_seconds=45.0):
    """Returns (trigger bool, threshold used, time the condition was first met | None)."""
    t, s = np.asarray(times, float), np.asarray(scores, float)
    if len(t) == 0:
        return False, abs_threshold, None
    thr = float(np.percentile(personal_scores, personal_pct)) if personal_scores is not None and len(personal_scores) >= 30 else abs_threshold
    for a, b in runs(s < thr):
        if t[b - 1] - t[a] >= min_seconds:
            return True, thr, float(t[a] + min_seconds)
    return False, thr, None


def sustained(times, flags, min_seconds):
    """True if `flags` (e.g. the per-update 'deep relaxation candidate' booleans) stay True for >= min_seconds in a row at the END of the series.
    Used to confirm 'deeply relaxed / meditative' (default 120 s) before the app says so."""
    t, f = np.asarray(times, float), np.asarray(flags, bool)
    if len(t) == 0 or not f[-1]:
        return False
    i = len(f) - 1
    while i > 0 and f[i - 1]:
        i -= 1
    return bool(t[-1] - t[i] >= min_seconds)
