# Focus score engine

Rule-based (no machine learning) focus scores from **EEG patterns**, **camera eye movement** and **task clicks**.
Three scripts share one formula; everything they share lives in `focus_core/`.

| Script | Job |
|---|---|
| `baseline.py` | Build/update a user's **personal baseline** (`baseline.json`) from resting EEG, camera calibration and training sessions |
| `simple_score.py` | **Simple score** (1-100): EEG + eye (+ clicks in training mode) vs. the user's own baseline |
| `complex_score.py` | **Complex score**: the simple pipeline (imported, not copied) + age/sex/substance/ASRS personalisation |
| `relax_score.py` | **EXPERIMENTAL, future idea, not part of the core product** -- relaxation / meditation score: eyes-matched EEG (alpha, theta) + a drowsiness guard + planned watch-HRV; same formula, separate axis (see `docs/RELAXATION.md`) |
| `realtime.py` | **Real-time engine**: buffers EEG / camera / click streams and re-scores the latest window with the same functions (see `docs/REALTIME.md`) |
| `server/app.py` | WebSocket service template around the engine (+ `server/client_demo.py`, a working reference client) |

> Wellness estimate only. Not a diagnosis or medical advice. Pattern directions come from population-level studies and are **not validated for
> individuals**; all weights and context-shift sizes are placeholders. Avoid diagnostic/treatment claims in the product (a theta/beta ADHD
> diagnostic aid has been FDA-cleared, which shows how close mental-health claims are to medical-device territory).

> **Scope note:** the core product is the *focus* score (`baseline.py`, `simple_score.py`, `complex_score.py`, `realtime.py`). The relaxation / meditation
> score (`relax_score.py`, `focus_core/heart.py`, `docs/RELAXATION.md`) is kept only as a future-expansion idea: the construct is vague, it is not wired into the
> real-time engine or server, and the watch-heart part has never been tested with a real device. Nothing in the focus pipeline depends on it.

## Layout
```
baseline.py  simple_score.py  complex_score.py        the three main scripts (CLI + importable functions)
realtime.py                                           live engine (stateful, framework-agnostic)
server/app.py  server/client_demo.py                  WebSocket service (shared sessions: headset bridge + website) + protocol client
bridge/openbci_bridge.py                              OpenBCI Cyton / Cyton+Daisy (BrainFlow) -> server, clock-synced
web-client/attune-client.js                           dependency-free JS client for the website (browser / Node 22)
docs/OPENBCI.md                                       connecting the headset, the website and the GitHub repo (start here for hardware)
AGENTS.md  .github/workflows/tests.yml  requirements-{server,bridge,dev}.txt
docs/REALTIME.md                                      how to connect the app (architecture, clocks, protocol, scaling)
docs/RELAXATION.md                                    relaxation rules flow, literature, planned watch ECG/HRV
focus_core/
  constants.py     SINGLE SOURCE OF TRUTH: formula constants, weights, metric tables, thresholds, literature keys
  scoring.py       strength, modality mean, quality ramps, combination, score mapping
  eeg.py  eye.py  pursuit.py  clicks.py  task.py       signal -> metrics
  baselines.py     how baseline.json is built (rules for mean/SD)
  context.py       complex-score shifts       reference.py   age/sex population reference (cold start)
  heart.py         PLANNED watch HRV: ECG R-peaks / RR cleaning / RMSSD, breathing rate (not validated with a real watch)
  signal_quality.py  real-time electrode-quality prompts + camera-assisted checks    trigger.py  when to offer training
  baseline_ops.py  baseline building as functions (used by baseline.py, the engine and the server)
  simulate.py      synthetic data for demos/tests      fileio.py  file helpers
tools/camera_capture.py   webcam/video -> eye + head-pose CSV (MediaPipe)
tests/test_focus.py       python -m unittest discover -s tests
REFERENCES.md             full literature list (short keys used in code comments)
```

## Quick start (synthetic demo)
```bash
pip install -r requirements.txt
python -m focus_core.simulate write-demo --out examples          # synthetic inputs
python baseline.py rest --baseline baseline.json --eeg examples/rest_eeg.csv --fs 250 --eye examples/rest_eye.csv
for i in 1 2 3; do python baseline.py training --baseline baseline.json \
    --events examples/train_${i}_events.csv --eye examples/train_${i}_eye.csv; done
python baseline.py show --baseline baseline.json
python simple_score.py  --baseline baseline.json --eeg examples/passive_eeg.csv --fs 250 --eye examples/passive_eye.csv
python simple_score.py  --baseline baseline.json --eeg examples/session_eeg.csv --fs 250 --eye examples/session_eye.csv --events examples/session_events.csv
python complex_score.py --profile examples/profile.json --baseline baseline.json \
    --eeg examples/session_eeg.csv --fs 250 --eye examples/session_eye.csv --events examples/session_events.csv
```

## The formula (identical in all three scripts)
```
1  strength   = clip( direction * (value - baseline_mean) / max(baseline_sd, floor), -3, +3 )      per metric, SD units, + = more focused
2  m_modality = sum(w_metric * strength) / sum(w_metric)                                           modality = click | eye | eeg
3  W_k        = BASE_WEIGHTS[k] * quality[k]            BASE_WEIGHTS = click .45, eye .35, eeg .20
   m_total    = sum(W_k * m_k) / sum(W_k)               (missing / poor modalities drop out; the rest renormalise)
4  SIMPLE  score = clip( 50 + 25 * m_total, 1, 100 )
   COMPLEX score = clip( 50 + 25 * (m_total + shift_substances + shift_asrs), 1, 100 )
```
50 = the user's own usual level; each SD better/worse moves the score 25 points.
Illustrative numbers (training session): click strengths -0.68, -0.65, -0.54, -3.0 (clipped), +0.62 with weights 1.0, 0.8, 1.0, 1.0, 0.5 give
m_click = -1.03; with m_eye = +0.27 and m_eeg = -1.65: m_total = .45(-1.03) + .35(.27) + .20(-1.65) = -0.70, so the simple score is 50 + 25(-0.70) = 32.5.

* **Quality**: click = clip((n_targets - 2) / 8, 0, 1) (a click score needs >= 3 targets); eye = clip(usable-frame share / 0.8, 0, 1);
  EEG = clip(clean-window share / 0.7, 0, 1).
* **Modes**: *passive* (no task): eye metrics = blinks, eyelid closure, gaze on task, fixations/saccades; *training* (orbiting ball + colour-change
  clicks): eye metrics = pursuit correlation, lag, tracking error, tracking-lapse time, eyelid closure; clicks = hit rate, median RT, RT variability,
  lapse rate, false alarms. EEG is the same in both.
* **Baseline rules** (`focus_core/baselines.py`): mean = mean over sessions (EEG: over all clean windows); SD = EEG pooled over windows, eye/click
  between-session SD from 3 sessions on, placeholder SD before that; floors protect against tiny SDs.
* **Where age/sex enter** (complex only): only for a user with *no* personal EEG baseline yet, via an age/sex/habit-matched population reference
  (`baseline.py reference`, built from TDBRAIN-derived features). A personal baseline already contains the user's own age/sex effects. The
  reference falls back to plain mean/SD for any feature where covariates do not improve cross-validated R^2 (rare gains in our TDBRAIN run).
* **Substances / ASRS**: `shift = amplitude * 0.5 ** (hours / half_life)` (caffeine +0.15 SD/5 h, nicotine +0.10/2 h, alcohol -0.25/12 h);
  ASRS `-0.30 * clip((mean/4 - 0.35)/0.20, -2, 2)`. Directions from the literature, sizes are placeholders.

To change anything numeric, edit `focus_core/constants.py` only; all three scripts pick it up. Bump `SCHEMA_SCORE` if output semantics change.

## Data contracts
Inputs (one shared clock starting at 0): EEG CSV (header = channel names, microvolts, + `--fs`) or EDF/BDF; eye CSV `t, ear, gaze_x, gaze_y, face_ok`
(+ `head_x, head_y, head_roll_deg, head_yaw` from `tools/camera_capture.py`); events CSV `type (change|click), t`; `profile.json`:
`{"age": 29, "sex": "F", "hours_since": {"caffeine": 2, "cigarette": null, "alcohol": 20}, "asrs_items": [18 x 0-4]}`.

Outputs are JSON on stdout (and `--out`). Stable keys: `schema` (`focus-score/2`), `kind` (`simple|complex`), `mode` (`passive|training`),
`simple_score`, `complex_score` (complex only), `components`, `weights_used`, `quality`, `strengths`, `metrics`, `baseline_basis`, `patterns_present`,
`patterns_absent_or_reversed`, `timeline` (training), `shifts_sd` + `age_sex_note` (complex), `formula` (the exact formulas, for display), `notes`.
`baseline.json` has `schema` `focus-baseline/2` with blocks `eeg`, `eye_rest`, `training`.

Relaxation output (`relax-score/1`): `relax_score`, `state` (`drowsy|relaxed|usual|tense`), `deep_candidate`, `drowsy` + `drowsy_reasons`, `components`, `weights_used`, `flow`
(the step-by-step rules trace), `suggestions` (each with its evidence), `formula`, `notes`. Baselines gain `eeg_closed` and `heart_rest` blocks.

Downstream use: call the scripts and read the JSON, or import `simple_score.score_simple(...)` / `complex_score.score_complex(...)` directly.
Real-time electrode prompts: `focus_core.signal_quality.QualityMonitor` (see its docstring); `usable_channels(report)` gives the channels to trust.

## Git hygiene
Do **not** commit: `baseline*.json` (personal data), TDBRAIN-derived participant-level files, raw recordings (see `.gitignore`). `reference.json`
holds only aggregate coefficients, but check the TDBRAIN data-use agreement before committing it, and credit van Dijk et al. (2022).

## Validation status (be honest in the product and the pitch)
* Verified on **synthetic** data only (tests): formula arithmetic, baseline rules, ordering of simulated states, signal-quality detection, script agreement.
* TDBRAIN checks so far (separate analysis code, not in this repo): frontal TBR and blink/saccade features did **not** separate ADHD from healthy
  adults and did not track oddball performance proxies; they are moderately reliable within person (test-retest ICC ~0.4-0.9, 16-17 people). The
  within-person task-vs-rest check of the EEG patterns has not been reported yet.
* Not yet tested: real headset signals, webcam blink/gaze dynamics (`camera_capture.py` was tested on a still photo and a still-image video only),
  the orbit/click task with real users, all default thresholds and weights.
* Suggested next step: a small pilot (10-20 people, on-task probes) to replace judgement weights and shift sizes with estimates.

## Headset + website (OpenBCI Cyton Daisy)
`pip install -r requirements-server.txt -r requirements-bridge.txt`, start `uvicorn server.app:app`, run `python bridge/openbci_bridge.py --board cyton_daisy --serial-port COM3 --user-id u1`
(or `--board synthetic` without hardware), and connect the website with `web-client/attune-client.js` using the same user id. Full guide, clock-sync details and an honest list of what is untested: `docs/OPENBCI.md`.

## Real-time use
`realtime.FocusEngine` (push EEG chunks, camera rows and click events; get score / quality / suggestion messages back) and the WebSocket template in
`server/app.py`. Read `docs/REALTIME.md` first; run `python server/client_demo.py` to see the whole flow (calibration, live scoring with a sensor
failure, a training round) without any hardware.

## Tests
`python -m unittest discover -s tests -v` (camera tests are skipped unless `mediapipe==0.10.14`, OpenCV and scikit-image are installed; server tests
need `fastapi` and `httpx`).
