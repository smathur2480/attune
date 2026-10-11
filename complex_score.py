#!/usr/bin/env python3
"""
complex_score.py -- the COMPLEX focus score: the SAME pipeline as simple_score.py, personalised with age, sex, substance use and
(optionally) ASRS answers.  It imports score_simple() from simple_score.py, so every metric, strength, weight and quality rule is
shared; this file only adds the personal-context step.

  complex m   = m_total(simple pipeline) + shift_substances + shift_asrs
  complex score = clip( 50 + 25 * complex m, 1, 100 )                                  (same mapping as the simple score)

WHERE EACH PERSONAL FACTOR ENTERS (nothing is applied twice)
  substance use  shift_substances = sum over substances of  amplitude * 0.5 ** (hours_since_use / half_life)     [SD units]
                   caffeine +0.15 SD (half-life 5 h) | nicotine/cigarette +0.10 SD (2 h) | alcohol -0.25 SD (12 h); hours clipped 0-72
  ASRS           shift_asrs = -0.30 * clip( (mean_item_score/4 - 0.35) / 0.20, -2, +2 )   if >= 12 of 18 items answered
                   (higher self-reported symptom burden lowers the EXPECTED focus; never used to label a person)
  age, sex       change the EEG expectation only for a user with NO personal EEG baseline yet: the EEG features are then compared with
                 an age/sex/habit-matched population reference (reference.json) instead of the plain population reference. Once a
                 personal EEG baseline exists it already contains the user's own age/sex effects, so no further adjustment is made.
                 The reference uses a covariate model for a feature only if it improved cross-validated R^2 (in our TDBRAIN run it
                 rarely did -- age/sex/habits explained little of these EEG features).
ALL SHIFT SIZES ARE PLACEHOLDERS: the DIRECTIONS follow the literature below, the MAGNITUDES must be calibrated in a pilot.

USAGE
  python complex_score.py --profile profile.json --baseline baseline.json --eeg window.csv --fs 250 --eye window_eye.csv
  python complex_score.py --profile profile.json --baseline baseline.json --eeg s.csv --fs 250 --eye s_eye.csv --events s_events.csv
  profile.json: {"age": 29, "sex": "F", "hours_since": {"caffeine": 2, "cigarette": null, "alcohol": 20}, "asrs_items": [18 x 0-4]}
  (hours_since: hours since last use; null/missing = not used.)  All other arguments are the same as simple_score.py.
  The output contains BOTH scores, the shifts, and a note saying how age/sex were used.

LITERATURE  (substances, questionnaire, reference norms; the base pipeline's references are listed in simple_score.py)
  Substances
    [Heishman2010] Heishman SJ, Kleykamp BA, Singleton EG (2010). Meta-analysis of the acute effects of nicotine and smoking on human
        performance. Psychopharmacology. Nicotine had significant effects on motor abilities, attention and memory that likely reflect
        true enhancement, not just withdrawal relief.
    [Myers2013] Myers CS, Taylor RC, Salmeron BJ, Waters AJ, Heishman SJ (2013). Nicotine enhances alerting, but not executive,
        attention in smokers and nonsmokers. Nicotine & Tobacco Research.
    [Mancuso2001] Mancuso G, Lejeune M, Ansseau M (2001). Cigarette smoking and attention: processing speed or specific effects?
        Psychopharmacology 155(4).
    [Caffeine60mg] "Effects of a single, oral 60 mg caffeine dose on attention in healthy adult subjects" (authors/journal not
        retrieved): improved sustained attention, peak saccadic velocity and reaction time in 82 adults aged 40-60.
    [Frontiers2022] "Differential saccade-pursuit coordination under sleep loss and low-dose alcohol", Frontiers in Neuroscience
        (2022), doi:10.3389/fnins.2022.1067722 (authors not retrieved): low-dose alcohol and acute sleep loss both produced a large
        loss of pursuit gain; caffeine reduced the pursuit deficit after sleep loss. (Alcohol's direction is also supported by general
        knowledge; the only other source found was a thesis whose results could not be read.)
  Questionnaire
    [Kessler2005] Kessler RC, Adler L, Ames M, et al. (2005). The World Health Organization Adult ADHD Self-Report Scale (ASRS).
        Psychological Medicine 35:245-256 (cited from memory; verify). A SCREENING questionnaire: here it only nudges the expected level.
  Age / sex reference
    [vanDijk2022] van Dijk H, et al. (2022). TDBRAIN database. Scientific Data 9:333 (source of the reference norms; credit required).
    [Arns2013] Arns M, Conners CK, Kraemer HC (2013). A decade of EEG theta/beta ratio research in ADHD: a meta-analysis. J Atten Disord
        17(5):374-383 (TBR depends on age and is not a reliable diagnostic).
CAUTION: wellness estimate only -- not a diagnosis or medical advice. Do not use the ASRS shift or any score to label people.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from focus_core import constants as K  # noqa: E402
from focus_core import context  # noqa: E402
from focus_core import fileio as fio  # noqa: E402
from focus_core import scoring  # noqa: E402
from simple_score import add_input_args, load_inputs, score_simple  # noqa: E402


def score_complex(profile, baseline=None, eeg=None, eye=None, task=None, reference=None, timeline=True):
    """Complex focus score. Returns the simple result (unchanged) plus complex_score and the shifts that produced it."""
    simple = score_simple(baseline, eeg, eye, task, reference, profile=None, timeline=timeline)          # the plain simple score
    if "error" in simple:
        return {**simple, "kind": "complex"}
    cold_eeg = eeg is not None and not (baseline or {}).get("eeg") and reference is not None
    base = score_simple(baseline, eeg, eye, task, reference, profile=profile, timeline=False) if cold_eeg else simple
    shifts = context.context_shift(profile)
    m_complex = base["mean_strength_sd"] + shifts["total"]
    if cold_eeg:
        age_sex = "applied: EEG compared with an age/sex/habit-matched population reference (no personal EEG baseline yet)"
    elif profile and (profile.get("age") is not None or profile.get("sex")):
        age_sex = "not applied: scores are relative to the user's own baseline, which already reflects their age and sex"
    else:
        age_sex = "no age/sex given"
    result = dict(simple)
    result.update(kind="complex", complex_score=round(scoring.to_score(m_complex), 1), complex_mean_strength_sd=float(m_complex),
                  shifts_sd=dict(total=round(shifts["total"], 4), substances=shifts["substances"], substances_total=round(shifts["substances_total"], 4),
                                 asrs=round(shifts["asrs"], 4), asrs_mean01=shifts["asrs_mean01"]),
                  age_sex_note=age_sex)
    result["formula"] = scoring.formula_description()
    result["notes"] = [n for n in simple["notes"] if n != K.DISCLAIMER] + [
        "complex-score shifts use literature DIRECTIONS with placeholder SIZES (constants.SUBSTANCES / ASRS_*)", K.DISCLAIMER]
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_args(ap)
    ap.add_argument("--profile", required=True, help="profile.json: age, sex, hours_since{}, asrs_items[]")
    a = ap.parse_args(argv)
    baseline, eeg, eye, task, reference = load_inputs(a)
    profile = fio.read_json(a.profile)
    res = score_complex(profile, baseline, eeg, eye, task, reference, timeline=not a.no_timeline)
    text = json.dumps(res, indent=1, default=float)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
