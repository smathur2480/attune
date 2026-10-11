"""
focus_core.constants -- the SINGLE SOURCE OF TRUTH for every number used in the focus scores.

baseline.py, simple_score.py and complex_score.py all import from here, so the formulas cannot drift apart.
Change a weight or threshold HERE and all three scripts change together.

THE FORMULA (identical in all three scripts)
--------------------------------------------
1. strength of one metric (SD units, + = more focused):
       strength = clip( direction * (value - mean) / max(sd, floor),  -3, +3 )
   mean/sd come from the user's OWN baseline (baseline.py); until a user has enough data, from the placeholder values below.

2. modality score (modality = click | eye | eeg): weighted mean of its metric strengths
       m_modality = sum(w_metric * strength_metric) / sum(w_metric)            (only metrics that could be computed)

3. combine modalities with base weights scaled by data quality (missing / poor modalities drop out, the rest renormalise)
       W_modality = BASE_WEIGHTS[modality] * quality[modality]
       m_total    = sum(W_modality * m_modality) / sum(W_modality)

4. SIMPLE score  = clip( 50 + 25 * m_total, 1, 100 )                              (simple_score.py)
   COMPLEX score = clip( 50 + 25 * (m_total + shift_substances + shift_asrs), 1, 100 )   (complex_score.py)
   (50 = the user's own typical level; every SD better/worse moves the score 25 points.)

PLACEHOLDERS: every weight, the 25-points-per-SD scale, the context-shift sizes and the default mean/SD values below are
judgement calls or placeholders, NOT fitted values. Replace them with pilot-study estimates when available.

LITERATURE (full references in REFERENCES.md; the short keys in `refs=[...]` below point there)
  EEG patterns : Kam2022, Groot2021, Jin2019, Arnau2019, vanSon2019, Frontiers2025, bioRxiv2023
  Eye / camera : VanOrden2000, Schleicher2008, FAA1994, Soukupova2016, GarciaAgundez2018, Danisman2010, AttentionLapse2023,
                 Fransson2008, Frontiers2022, Peiris2006, Jones2010
  Click task   : PVT2020, QbTest, gradCPT2013, SART1998, Shelat2024
  Context      : Heishman2010, Myers2013, Caffeine60mg, Mancuso2001, Frontiers2022, Kessler2005
  Reference    : vanDijk2022 (TDBRAIN), Arns2013
"""

SCHEMA_BASELINE = "focus-baseline/2"
SCHEMA_SCORE = "focus-score/2"
DISCLAIMER = ("Wellness estimate for a demo; not a diagnosis or medical advice. Pattern directions come from population-level "
              "studies and have not been validated for individuals; weights and shifts are placeholders.")

# ---------------------------------------------------------------------------------------------------------------------
# score mapping and combination  (see THE FORMULA above)
# ---------------------------------------------------------------------------------------------------------------------
SCORE_CENTER = 50.0                 # score when the user is exactly at their own baseline
SCORE_PER_SD = 25.0                 # points per SD of mean strength (a presentation choice)
SCORE_MIN, SCORE_MAX = 1.0, 100.0
STRENGTH_CLIP = 3.0                 # one extreme metric cannot dominate
BASE_WEIGHTS = {"click": 0.45, "eye": 0.35, "eeg": 0.20}     # judgement: click = direct outcome; EEG = least validated
MIN_SESSIONS_PERSONAL_SD = 3        # between-session SD is used only from this many baseline sessions on

# data-quality ramps (quality in 0..1 multiplies the base weight)
CLICK_Q_ZERO_AT, CLICK_Q_FULL_AT = 2, 10      # quality = clip((n_targets - 2) / 8, 0, 1); a click score needs >= 3 targets
CLICK_MIN_TARGETS = 3
EYE_FULL_VALID_FRAC = 0.8                     # quality = clip(share of frames with usable face+gaze / 0.8, 0, 1)
EEG_FULL_CLEAN_FRAC = 0.7                     # quality = clip(share of clean windows / 0.7, 0, 1)

# ---------------------------------------------------------------------------------------------------------------------
# EEG feature extraction
# ---------------------------------------------------------------------------------------------------------------------
FRONTAL = ["Fp1", "Fp2", "AF7", "AF8", "F3", "F4", "F7", "F8", "Fz"]
POSTERIOR = ["O1", "O2", "Oz", "P3", "P4", "Pz", "P7", "P8"]
EEG_WIN_S, EEG_HOP_S = 8.0, 4.0
EEG_MIN_SEGMENT_S = 4.0                       # shorter scoring segments are not used
EEG_MIN_CLEAN_WINDOWS_BASELINE = 5
EEG_REJECT_FRONT_UV, EEG_REJECT_POST_UV = 150.0, 300.0     # peak-to-peak artefact limits per window
EEG_EMG_MAX = 0.6                             # (35-45 Hz power)/(13-30 Hz power) above this -> beta-based features unusable
EEG_BANDS = dict(slow=(1, 8), theta=(4, 8), alpha=(8, 12), beta=(13, 30), emg=(35, 45), total=(1, 40))

# ---------------------------------------------------------------------------------------------------------------------
# camera eye features
# ---------------------------------------------------------------------------------------------------------------------
BLINK_CLOSE_FRAC = 0.60             # eye-aspect-ratio below 60% of the open level = lid closing
BLINK_MIN_DEPTH = 0.40              # a blink must reach below 40% of the open level
PERCLOS_FRAC = 0.20                 # "closed" = below 20% of open level  (PERCLOS-type, ~80% closed)
LONG_BLINK_MS = 300.0
EYE_MIN_FACE_FRAC = 0.6
EYE_MIN_DURATION_S = 15.0

# ---------------------------------------------------------------------------------------------------------------------
# training task (orbiting ball + colour-change clicks)
# ---------------------------------------------------------------------------------------------------------------------
ORBIT_PERIOD_S, ORBIT_RADIUS = 8.0, 0.3
RESP_WINDOW_S = (0.15, 1.5)         # response window after a colour change; earlier clicks = anticipations
LAPSE_MS = 500.0                    # PVT convention: RT >= 500 ms (or no response) = lapse
PURSUIT_MAX_LAG_S = 0.6
TRACK_LAPSE_S = 0.5                 # flat/incoherent tracking lasting >= 0.5 s = tracking lapse
TRACK_ERR_THR = 1.0                 # ... or gaze more than 1 orbit radius from the ball
TRACK_FLAT_RATIO = 0.35             # ... or gaze moves < 35% of what the ball does
TIMELINE_WINDOW_S = 30.0
PURSUIT_MIN_USABLE_S = 20.0

# ---------------------------------------------------------------------------------------------------------------------
# METRIC TABLES.  direction +1 = higher is better; weight = relative weight inside its modality; floor = smallest SD allowed;
# mean / sd = POPULATION PLACEHOLDERS used until the user has a personal baseline (None for EEG: needs a baseline/reference).
# ---------------------------------------------------------------------------------------------------------------------
EEG_FEATURES = {
    "alpha_rel": dict(label="alpha_suppression", direction=-1, weight=1.0, floor=0.08, mean=None, sd=None,
                      text="alpha power is below your resting level", against="alpha power is above your resting level",
                      refs=["Kam2022", "Groot2021", "Jin2019", "Arnau2019", "Frontiers2025", "bioRxiv2023"]),
    "tbr": dict(label="low_theta_beta", direction=-1, weight=0.8, floor=0.08, mean=None, sd=None,
                text="frontal theta/beta ratio is below your resting level", against="frontal theta/beta ratio is above your resting level",
                refs=["vanSon2019", "Kam2022"]),
    "beta_rel": dict(label="beta_engagement", direction=+1, weight=0.5, floor=0.08, mean=None, sd=None,
                     text="frontal beta is above your resting level", against="frontal beta is below your resting level",
                     refs=["Kam2022", "Groot2021"]),            # Groot2021 found NO beta change: hence the lower weight
    "slow_rel": dict(label="low_slow_wave", direction=-1, weight=0.4, floor=0.08, mean=None, sd=None,
                     text="slow-wave (1-8 Hz) power is below your resting level", against="slow-wave (1-8 Hz) power is above your resting level",
                     refs=["Kam2022", "Groot2021", "bioRxiv2023"]),   # bioRxiv2023: frontal theta also RISES with external task demands
}
EYE_FEATURES = {
    "blink_rate": dict(contexts=("passive",), direction=-1, weight=1.0, floor=1.5, mean=14.0, sd=4.0,
                       text="you blink less often than at rest", against="you blink more often than at rest",
                       refs=["VanOrden2000", "Schleicher2008", "FAA1994", "AttentionLapse2023"]),
    "blink_dur_ms": dict(contexts=("passive",), direction=-1, weight=0.8, floor=15.0, mean=150.0, sd=30.0,
                         text="your blinks are short", against="your blinks are longer than at rest",
                         refs=["Schleicher2008", "VanOrden2000"]),
    "perclos": dict(contexts=("passive", "training"), direction=-1, weight=1.0, floor=0.01, mean=0.02, sd=0.02,
                    text="your eyes stay open (little eyelid closure)", against="your eyelids are closing more than at rest",
                    refs=["Schleicher2008", "Jones2010", "ZhengLu2017"]),
    "off_task_frac": dict(contexts=("passive",), direction=-1, weight=0.8, floor=0.03, mean=0.10, sd=0.08,
                          text="your gaze stays on the task area", against="your gaze is leaving the task area",
                          refs=["AttentionLapse2023"]),
    "long_fix_frac": dict(contexts=("passive",), direction=-1, weight=0.4, floor=0.05, mean=0.15, sd=0.10,
                          text="you are not staring at one point", against="you are staring at one point for long stretches",
                          refs=["AttentionLapse2023", "Schleicher2008"]),
    "saccade_rate": dict(contexts=("passive",), direction=+1, weight=0.3, floor=3.0, mean=20.0, sd=8.0,
                         text="your gaze keeps moving normally", against="your gaze is barely moving",
                         refs=["VanOrden2000", "Schleicher2008", "Fransson2008"]),
    "pursuit_corr": dict(contexts=("training",), direction=+1, weight=1.0, floor=0.04, mean=0.80, sd=0.12,
                         text="your eyes follow the ball closely", against="your eyes are not following the ball well",
                         refs=["Fransson2008", "Frontiers2022"]),
    "lag_ms": dict(contexts=("training",), direction=-1, weight=0.4, floor=30.0, mean=200.0, sd=80.0,
                   text="your eyes keep up with the ball", against="your eyes are lagging behind the ball",
                   refs=["Fransson2008", "Frontiers2022"]),
    "tracking_error": dict(contexts=("training",), direction=-1, weight=0.8, floor=0.05, mean=0.50, sd=0.20,
                           text="your gaze stays close to the ball", against="your gaze is drifting away from the ball",
                           refs=["Fransson2008"]),
    "lapse_time_frac": dict(contexts=("training",), direction=-1, weight=1.0, floor=0.02, mean=0.03, sd=0.05,
                            text="you have no stretches of lost tracking", against="you have stretches where your eyes stop following the ball",
                            refs=["Peiris2006", "Jones2010"]),
}
CLICK_FEATURES = {
    "hit_rate": dict(direction=+1, weight=1.0, floor=0.02, mean=0.95, sd=0.06,
                     text="you catch the colour changes", against="you are missing colour changes",
                     refs=["PVT2020", "QbTest"]),
    "median_rt_ms": dict(direction=-1, weight=0.8, floor=25.0, mean=380.0, sd=90.0,
                         text="your reactions are fast", against="your reactions are slower than usual",
                         refs=["PVT2020", "Shelat2024"]),
    "rt_sd_ms": dict(direction=-1, weight=1.0, floor=15.0, mean=90.0, sd=45.0,
                     text="your reaction times are steady", against="your reaction times are erratic",
                     refs=["QbTest", "gradCPT2013", "Shelat2024"]),
    "lapse_rate": dict(direction=-1, weight=1.0, floor=0.02, mean=0.05, sd=0.08,
                       text="you rarely lapse (miss or react after 500 ms)", against="you are lapsing (missing or reacting after 500 ms)",
                       refs=["PVT2020", "QbTest", "Peiris2006"]),
    "false_alarm_per_min": dict(direction=-1, weight=0.5, floor=0.2, mean=0.5, sd=0.8,
                                text="you rarely click without a change", against="you are clicking without a change or too early",
                                refs=["QbTest", "gradCPT2013", "SART1998"]),
}
TABLES = {"eeg": EEG_FEATURES, "eye": EYE_FEATURES, "click": CLICK_FEATURES}


def eye_table(context):
    """Eye metrics that belong to a context: 'passive' (rest / monitoring) or 'training' (orbiting-ball task)."""
    return {k: v for k, v in EYE_FEATURES.items() if context in v["contexts"]}


# ---------------------------------------------------------------------------------------------------------------------
# COMPLEX-score context shifts (added to m_total, in SD units). Directions from the literature, SIZES ARE PLACEHOLDERS.
# ---------------------------------------------------------------------------------------------------------------------
# (shift in SD units, half-life in hours): caffeine/nicotine improve alerting/sustained attention, alcohol impairs tracking.
# Heishman2010, Myers2013 (nicotine); Caffeine60mg, Frontiers2022 (caffeine/alcohol on pursuit); Mancuso2001 (smoking)
SUBSTANCES = {"caffeine": (+0.15, 5.0), "cigarette": (+0.10, 2.0), "alcohol": (-0.25, 12.0)}
SUBSTANCE_MAX_HOURS = 72.0
# ASRS-v1.1 (18 items scored 0-4): higher symptom burden -> lower expected focus. Kessler2005. All three numbers are PLACEHOLDERS.
ASRS_POP_MEAN, ASRS_POP_SD, ASRS_SHIFT_PER_SD, ASRS_CLIP_SD, ASRS_MIN_ITEMS = 0.35, 0.20, -0.30, 2.0, 12


# =====================================================================================================================
# RELAXATION / MEDITATION SCORE (relax_score.py).  Same formula as the focus scores (see THE FORMULA at the top):
#   strength = clip(direction * (value - mean) / max(sd, floor), -3, +3), measured against the baseline of the MATCHING CONDITION
#   m_modality = weighted mean of strengths;  W_k = RELAX_BASE_WEIGHTS[k] * quality[k];  m_total = sum(W_k m_k) / sum(W_k)
#   m_adj = m_total - DROWSY_PENALTY_SD   if the drowsiness guard fires, else m_total
#   relax score = clip( 50 + 25 * m_adj, 1, 100 )         (50 = the user's usual level for that eyes-open / eyes-closed condition)
# Modalities: eeg (alpha, theta), heart (HRV from a watch -- PLANNED feature). The camera is not a scoring modality: it decides the
# eyes-open/closed condition and feeds the drowsiness guard. Everything numeric below is a judgement call or placeholder.
#
# LITERATURE KEYS used below (full references in REFERENCES.md)
#   EEG   : Lomas2015, Cahn2006, Meditation2018, AlphaThetaDepth
#   heart : Laborde2017, Laborde2022, SevozCouche2022, Lehrer2014, Laborde2021, TaskForce1996, Shaffer2017
#   watch : WatchECG2024, Taoum2022, PRVMeta, JMIR2020, SmartwatchHRV2019
#   drowsy: Chokroverty, Magosso2007, Strauss2022, Schleicher2008, Patent5813993
# =====================================================================================================================
SCHEMA_RELAX = "relax-score/1"
RELAX_BASE_WEIGHTS = {"eeg": 0.5, "heart": 0.5}      # judgement; the heart share is unused until a watch is connected (weights renormalise)

RELAX_EEG_FEATURES = {
    # alpha and theta are the two bands most consistently RAISED in meditation vs an eyes-closed rest (Lomas2015, Cahn2006, Meditation2018)
    "alpha_rel": dict(label="alpha_up", direction=+1, weight=1.0, floor=0.08, mean=None, sd=None,
                      text="alpha power is above your usual resting level", against="alpha power is not above your usual resting level",
                      refs=["Lomas2015", "Cahn2006", "Meditation2018", "AlphaThetaDepth"]),
    # theta (measured against NON-alpha power so that an alpha change alone cannot move it) also rises in sleepiness and may DEACTIVATE as
    # meditation deepens (AlphaThetaDepth) -> lower weight + drowsiness guard
    "theta_ex": dict(label="theta_up", direction=+1, weight=0.5, floor=0.08, mean=None, sd=None,
                      text="frontal theta is above your usual resting level", against="frontal theta is not above your usual resting level",
                      refs=["Lomas2015", "Cahn2006", "Meditation2018", "AlphaThetaDepth"]),
}
# Heart features: NO population placeholders (HRV depends strongly on age, fitness, posture, device) -> a personal baseline is required.
HEART_FEATURES = {
    "ln_rmssd": dict(direction=+1, weight=1.0, floor=0.15, mean=None, sd=None,
                     text="your heart-rate variability (RMSSD) is above your usual resting level",
                     against="your heart-rate variability (RMSSD) is below your usual resting level",
                     refs=["Laborde2017", "Laborde2022", "Shaffer2017", "JMIR2020"]),
    "hr_mean": dict(direction=-1, weight=0.8, floor=3.0, mean=None, sd=None,
                    text="your heart rate is below your usual resting level", against="your heart rate is above your usual resting level",
                    refs=["Laborde2022", "Laborde2017"]),
    # used ONLY when the estimated breathing rate is >= HF_MIN_BREATH_CPM and the segment is long enough (see focus_core/heart.py)
    "ln_hf": dict(direction=+1, weight=0.5, floor=0.3, mean=None, sd=None,
                  text="your high-frequency HRV (breathing-linked) is above your usual resting level",
                  against="your high-frequency HRV (breathing-linked) is below your usual resting level",
                  refs=["Laborde2021", "TaskForce1996", "Laborde2022"]),
}
RELAX_TABLES = {"eeg": RELAX_EEG_FEATURES, "heart": HEART_FEATURES}

# ---- eyes condition (decides which resting baseline the EEG is compared with; Berger effect: closing the eyes raises alpha by itself)
EYES_CLOSED_FRAC = 0.70          # share of frames with EAR < 20% of the open level at/above which the segment counts as "eyes closed"
EYES_OPEN_FRAC = 0.20            # at/below which it counts as "eyes open"; in between = mixed (scored as eyes open, with a note)
EAR_OPEN_DEFAULT = 0.30          # used only if no camera baseline gives the user's own open-eye level

# ---- drowsiness guard (relaxed-alert must not be confused with falling asleep)
DROWSY_THETA_MIN_SD = 1.0        # eyes closed: theta strength >= this AND alpha strength < DROWSY_ALPHA_BELOW_SD  -> drowsy
DROWSY_ALPHA_BELOW_SD = 0.5
DROWSY_PERCLOS_MIN_SD = 1.0      # eyes open: PERCLOS strength (vs the resting baseline, drowsy = HIGHER) >= this ...
DROWSY_BLINK_DUR_MIN_SD = 1.5    # ... or blink-duration strength >= this
DROWSY_PENALTY_SD = 1.5          # subtracted from m_total when the guard fires

# ---- state labels from the relax score (+ pattern conditions); untuned
RELAX_DEEP_SCORE, RELAX_DEEP_ALPHA_SD, RELAX_DEEP_SECONDS = 70.0, 1.0, 120.0   # sustained >= 2 min for "deeply relaxed / meditative"
RELAX_RELAXED_SCORE, RELAX_USUAL_SCORE = 58.0, 42.0

# ---- heart processing (focus_core/heart.py)
RR_MIN_MS, RR_MAX_MS, RR_JUMP_FRAC = 300.0, 2000.0, 0.20       # plausibility limits; beats deviating > 20% from the local median are dropped
HRV_MIN_S = 30.0                  # RMSSD / mean HR need >= 30 s (RMSSD stays reliable down to 10-60 s in the ultra-short literature)
HRV_FREQ_MIN_S = 120.0            # breathing-rate and HF/LF estimates need >= 2 min
HF_MIN_BREATH_CPM = 9.0           # HF power is only a vagal index above ~9 breaths/min (Laborde2021); slower breathing moves it to LF
HEART_WINDOW_S, HEART_HOP_S, HEART_MIN_BASELINE_WINDOWS = 60.0, 30.0, 3
HEART_FULL_VALID_FRAC = 0.85
PPG_QUALITY_FACTOR = 0.7          # pulse-rate variability from wrist PPG is less trustworthy than ECG-derived HRV (PRVMeta, Taoum2022)
