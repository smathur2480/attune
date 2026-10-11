> **FUTURE IDEA -- experimental, not part of the core product.** Kept for possible expansion; not wired into the real-time engine, not validated on real data.

# Relaxation / meditation score

`relax_score.py` answers "how relaxed are you compared with *your* usual level?" with the same conventions as the focus score
(50 = usual, +/-25 points per standard deviation, strengths clipped at +/-3, quality-scaled weights). It is a separate axis from focus:
a person can be calm and alert, calm and drowsy, or tense and focused, so the app should show the two scores independently.

## Rules flow
```
           EEG segment (>= 60 s)      camera (eyes open/closed, blinks)      watch HRV (PLANNED, >= 30 s; 2 min for breathing rate)
                    │                           │                                        │
 0 INPUT GATE  ─────┴───────────────────────────┴────────────────────────────────────────┘  need >= 1 usable modality, else "no score"
                    │
 1 CONDITION    eyes closed? ── share of frames with eyelid ratio < 20% of YOUR open level (>= 70% closed / <= 20% open; mixed -> open + note)
                    │            └─ Closing the eyes raises alpha by itself, so:  closed -> baseline["eeg_closed"],  open -> baseline["eeg"]
                    │               (no matching baseline -> EEG is NOT scored; unknown condition -> EEG is NOT scored)
                    ▼
 2 QUALITY      EEG: clean-window share (muscle activity is counted and reported);  heart: valid-beat share, duration, PPG x0.7
                    ▼
 3 PATTERNS     EEG  alpha UP (w 1.0) · frontal theta UP vs non-alpha power (w 0.5)
                Heart RMSSD UP (1.0) · heart rate DOWN (0.8) · HF power UP (0.5, only if breathing >= 9/min and segment >= 2 min)
                → strengths → modality means → W_k = {eeg .5, heart .5} × quality → m_total
                    ▼
 4 DROWSINESS   eyes closed: theta >= +1 SD AND alpha < +0.5 SD         (alpha not elevated while theta rises)
   GUARD        eyes open  : PERCLOS >= +1 SD or blink duration >= +1.5 SD above your resting level      → m_adj = m_total − 1.5
                    ▼
 5 STATE        drowsy | relaxed (score ≥ 58) | usual (≥ 42) | tense (< 42);  "deeply relaxed / meditative" only if score ≥ 70, alpha ≥ +1 SD
                and it holds ≥ 2 minutes (focus_core.trigger.sustained) -- a single update never claims it
                    ▼
 6 SUGGESTION   slow breathing ~6/min (literature-backed, needs heart data); other suggestions are labelled "UX suggestion, no evidence"
 relax score = clip(50 + 25 × m_adj, 1, 100)
```
The output's `flow` array records every step, so the app can show *why* it said what it said.

## Why each rule exists (details and sources: `REFERENCES.md`)
| Rule | Evidence / reasoning |
|---|---|
| Alpha and theta both UP | Mindfulness EEG reviews: alpha and theta most commonly enhanced vs an eyes-closed rest, not uniformly [Lomas2015] [Cahn2006] [Meditation2018] |
| Theta weight only 0.5 | Theta also rises in sleepiness and deactivated as meditation deepened in one study [AlphaThetaDepth] |
| Eyes-matched baseline | The reviews compare with an *eyes-closed* rest; eyes-closed EEG vs an eyes-open baseline would reward merely closing the eyes (tested) |
| Drowsiness guard | Sleepiness with eyes closed lowers alpha and raises theta [Chokroverty] [Patent5813993] [Strauss2022]; blink duration is a sleepiness indicator when eyes are open [Schleicher2008] |
| Theta measured vs non-alpha power | Relative theta rose mechanically when alpha fell and falsely triggered the guard in our simulation; this version cannot |
| RMSSD primary, HF conditional | RMSSD is the standard short-term vagal index, reliable on short recordings; below ~9 breaths/min vagal activity shifts from HF to LF [Laborde2021] [SmartwatchHRV2019] |
| Slow-breathing suggestion | Voluntary slow breathing affects HRV in 223 studies' meta-analysis [Laborde2022]; ~0.1 Hz resonance increases vagally mediated HRV [SevozCouche2022] [Lehrer2014] |

## Eyes-closed and heart baselines
```bash
python baseline.py rest --baseline baseline.json --eeg rest_open.csv --fs 250 --eye rest_eye.csv            # eyes open (also used by focus)
python baseline.py rest --baseline baseline.json --eeg rest_closed.csv --fs 250 --condition eyes_closed    # 2 min, eyes closed, NOT meditating
python baseline.py rest --baseline baseline.json --heart rr_rest.csv                                       # >= ~3 min seated rest (watch RR intervals)
python relax_score.py --baseline baseline.json --eeg session.csv --fs 250 --eye session_eye.csv [--heart rr.csv | --heart-ecg ecg.csv --heart-fs 250]
```
Baseline = your *ordinary* rest, so 50 means "your usual", and a good meditation session lands well above it.

## Connecting a watch (PLANNED -- not tested with real hardware)
What the literature found about watches: consumer smartwatch ECGs compared so far are single-lead **30-second spot recordings** [WatchECG2024];
continuous monitoring on a watch is normally **PPG**, which is validated at rest and degrades with motion [Taoum2022] [PRVMeta]; RMSSD is the
feature that survives short recordings [SmartwatchHRV2019] [JMIR2020]. Practical consequences already built in: 30-s minimum, RMSSD-first, PPG weight x0.7,
HF only when breathing allows, no population placeholders (a personal heart baseline is required).

Still to do before this works in the app:
1. **Pick the watch and check what its API exposes** (HealthKit, Health Connect or a vendor SDK): raw ECG strip, RR/inter-beat intervals, or only heart rate.
   Many platforms give only averaged heart rate, which is *not enough* for HRV. `focus_core/heart.py` accepts RR arrays, an ECG strip + sampling rate, or PPG pulse intervals.
2. **Transport:** add a `{"type":"heart","t_end":[...],"rr_ms":[...],"source":"ecg"|"ppg"}` message to `server/app.py` and a `push_heart()` plus a relaxation mode to
   `realtime.FocusEngine` (window >= 60 s). *Not done yet*; `relax_score.score_relax()` can already be called per window from any backend.
3. **Clocks:** watch timestamps must be mapped to the headset/camera clock (see `docs/REALTIME.md`).
4. **Protocol for users:** seated, still, same time of day for baselines (HRV depends on posture, fitness, age, time of day); discard motion periods for PPG.
5. **Validate:** compare watch-derived RMSSD with a chest-strap/ECG reference in a small pilot before trusting the heart half of the score.

## What is not validated
* All weights, thresholds (58/42/70, +1/+1.5 SD, 70%/20% eyelid closure) and the 1.5-SD drowsiness penalty are untuned placeholders.
* Alpha/theta changes are not specific to relaxation; the guard uses a simple rule and was tested on **synthetic** EEG only.
* Not implemented: slow-eye-movement detection (an early drowsiness cue) from Fp1-Fp2, meditation-style differences (focused attention vs open monitoring),
  a "complex" relaxation score with caffeine/age shifts (no sourced effect sizes found), and a relaxation mode in the real-time engine.
* Wellness estimate only: not a diagnosis, treatment or medical advice.
