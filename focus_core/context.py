"""
focus_core.context -- personal-context shifts used ONLY by the COMPLEX score (complex_score.py).

complex m = simple m_total + shift_substances + shift_asrs          (all in SD units; see constants.py for the full formula)

Substances (hours since last use, None = not used):  shift = amplitude * 0.5 ** (hours / half_life), hours clipped to 0..72
  caffeine  +0.15 SD, half-life  5 h     nicotine/cigarette +0.10 SD, half-life 2 h     alcohol -0.25 SD, half-life 12 h
ASRS (18 items, 0-4, >= 12 answered):  shift = -0.30 * clip((mean/4 - 0.35) / 0.20, -2, +2)

LITERATURE (directions only -- the SIZES are placeholders to be calibrated in a pilot)
  [Heishman2010]  Heishman SJ, Kleykamp BA, Singleton EG (2010). Meta-analysis of the acute effects of nicotine and smoking on
                  human performance. Psychopharmacology.  -> nicotine: significant effects on motor abilities, attention, memory
                  that likely reflect true enhancement, not just withdrawal relief.
  [Myers2013]     Myers CS, Taylor RC, Salmeron BJ, Waters AJ, Heishman SJ (2013). Nicotine enhances alerting, but not executive,
                  attention in smokers and nonsmokers. Nicotine & Tobacco Research.
  [Caffeine60mg]  "Effects of a single, oral 60 mg caffeine dose on attention in healthy adult subjects" (authors/journal not
                  retrieved) -> improved sustained attention, peak saccadic velocity and reaction time in 82 adults aged 40-60.
  [Frontiers2022] "Differential saccade-pursuit coordination under sleep loss and low-dose alcohol", Frontiers in Neuroscience
                  (2022), doi:10.3389/fnins.2022.1067722 (authors not retrieved) -> low-dose alcohol and acute sleep loss both
                  produced a large loss of pursuit gain; a caffeine countermeasure reduced the pursuit deficit after sleep loss.
  [Kessler2005]   Kessler RC, Adler L, Ames M, et al. (2005). The World Health Organization Adult ADHD Self-Report Scale (ASRS).
                  Psychological Medicine 35:245-256.  (cited from memory; verify before publication)
NOTE: the ASRS is a screening questionnaire; here it only nudges the EXPECTED level of focus. It is never used to label anyone.
"""
import numpy as np

from . import constants as K


def substance_shift(hours_since):
    total, details = 0.0, {}
    for name, (amp, half_life) in K.SUBSTANCES.items():
        h = (hours_since or {}).get(name)
        if h is None:
            continue
        s = amp * 0.5 ** (min(max(float(h), 0.0), K.SUBSTANCE_MAX_HOURS) / half_life)
        details[name] = round(float(s), 4)
        total += s
    return float(total), details


def asrs_shift(items):
    vals = [x for x in (items or []) if x is not None]
    if len(vals) < K.ASRS_MIN_ITEMS:
        return 0.0, None
    mean01 = float(np.mean(vals)) / 4.0
    z = float(np.clip((mean01 - K.ASRS_POP_MEAN) / K.ASRS_POP_SD, -K.ASRS_CLIP_SD, K.ASRS_CLIP_SD))
    return K.ASRS_SHIFT_PER_SD * z, round(mean01, 3)


def context_shift(profile):
    """profile: {"hours_since": {...}, "asrs_items": [...]}  ->  dict with the total shift and its parts (SD units)."""
    profile = profile or {}
    s_sub, det = substance_shift(profile.get("hours_since"))
    s_asrs, mean01 = asrs_shift(profile.get("asrs_items"))
    return dict(total=float(s_sub + s_asrs), substances_total=s_sub, substances=det, asrs=float(s_asrs), asrs_mean01=mean01)
