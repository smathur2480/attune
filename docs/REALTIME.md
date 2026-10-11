> **Update:** the server now has shared sessions (a headset bridge with role `device` and the website with role `app` feed one engine), a clock-sync message and a `status` message. The protocol header in `server/app.py` and `docs/OPENBCI.md` supersede the single-connection description below.

# Real-time integration guide

The three scripts are batch tools (file in, JSON out). For a live app they sit behind `realtime.FocusEngine`, which keeps rolling buffers of the
three streams and re-scores the latest window with the *same* `score_simple` / `score_complex` functions. `server/app.py` puts the engine behind a
WebSocket; `server/client_demo.py` is a working reference client.

```
 DEVICE (phone / browser)                                       SERVER (Python)
 ┌────────────────────────┐   EEG chunks (µV, 4/s)            ┌────────────────────────────────────────────┐
 │ EEG headset  (BLE/USB) ├─────────────────────────────────▶ │ server/app.py  (one FocusEngine per socket)│
 │ Camera ► face landmarks│   eye rows (EAR, gaze, head, 30/s)│   ├─ quality loop (1 Hz)  ─▶ "quality" prompts   │
 │   (on device: no video │ ─────────────────────────────────▶│   ├─ score loop (every 5 s) ─▶ "score"           │
 │    leaves the phone)   │   click / colour-change events    │   ├─ trigger ─▶ "suggest_training"               │
 │ Task UI (ball, clicks) ├─────────────────────────────────▶ │   └─ calibration / training workflows            │
 │ UI: score, prompts     │ ◀── JSON messages ────────────────┤ baselines saved in YOUR database (not in git)│
 └────────────────────────┘                                   └────────────────────────────────────────────┘
```

## 1. Where things run (decisions)
| Piece | Recommendation | Why |
|---|---|---|
| Face landmarks / eye aspect ratio / gaze / head pose | **On the device**; send only numbers | Video never leaves the phone (privacy, bandwidth); 30 rows/s of ~8 numbers is tiny |
| Scoring engine | Server (Python) or on-device Python if the app is Python-based | Needs numpy/scipy; ~11-14 ms per update (measured on a laptop) |
| Baselines | Server-side database, one document per user; never in git | Personal data. `baseline.json` schema is `focus-baseline/2` |
| Constants / formula | `focus_core/constants.py` only | One place to change weights for every script and the live engine |

`tools/camera_capture.py` shows the exact maths (6-point EAR, iris-in-eye gaze ratios, nose/roll/yaw head pose) using the legacy Python MediaPipe API
(`mediapipe==0.10.14`). In a browser or on a phone use your platform's face-landmark SDK (e.g. MediaPipe Face Landmarker, ML Kit, ARKit) and
re-implement the same formulas. **Verify the landmark indices against your SDK's landmark map** and re-check blink thresholds, because camera frame
rate, resolution and landmark model differ between SDKs (the defaults were tested on one still photo only).

## 2. Clocks and timing (the most common real-time bug)
* Use **one monotonic clock** for everything (e.g. `performance.now()/1000`). Send `t` in seconds.
* **EEG**: stamp the first chunk with its arrival time minus its duration, then just keep sending `t_start` for each chunk. The engine advances EEG time by
  the **sample count**, so bursty Bluetooth packets add no jitter; `t_start` is only used to detect dropouts (gap > 0.25 s -> buffer reset + `eeg_gap` warning).
* **Camera**: use the frame's capture/presentation timestamp (`requestVideoFrameCallback` in browsers), not "now when my code ran".
* **Colour change**: stamp the event when the frame showing the new colour is *presented*; a constant display latency is absorbed by the personal
  baseline, but *variable* latency inflates reaction-time variability (cited timing precision for PVT-style metrics: bias within about +-5 ms, SD <= 10 ms;
  see `REFERENCES.md` [PVT2020]).
* After connecting, `blink_sync_check` (`focus_core/signal_quality.py`) can estimate any EEG-vs-camera offset from blinks (it recovered a simulated 80 ms offset).

## 3. Cadence, window, smoothing
| Parameter (default) | Effect |
|---|---|
| `window_s` = 30 | The score describes the last 30 s, so it **lags** reality by roughly half a window; shorter = faster but noisier (30 s holds only ~5 blinks) |
| `update_s` = 5 | New score every 5 s (compute is negligible; the limit is statistical, not CPU) |
| `smoothing` = 0.5 | EMA in strength space, then the same `clip(50 + 25 m, 1, 100)` mapping. `score_raw` is also sent |
| `warmup_s` = 15 | No score before 15 s of data (and again after any stream gap); training uses 20 s |
Quality loop: one report per second; prompts are debounced (hard faults ~4-5 s, soft ones longer) so the UI does not flicker.

## 4. Messages
Full protocol: header of `server/app.py`. Server -> client types: `ready`, `quality`, `score`, `score_unavailable`, `suggest_training`, `warning`,
`training_started`, `training_summary`, `rest_started`, `rest_calibrated`, `rest_failed`, `pong`, `error`, optional `ack`.

| Message | Put it in the UI as |
|---|---|
| `score` (`score`, `components`, `weights_used`, `patterns_present`, `paused`, `prompts`) | The gauge (0-100, 50 = the user's usual), three sub-bars (click/eye/EEG), "what's helping / hurting" text, small notice when a part is paused |
| `quality` (`channels`, `prompts`, `resolved`, `ready`) | Electrode map (green/amber/red per channel), the fix-it prompt text, a "ready" gate before starting a session |
| `suggest_training` | Offer the training round (the user can also start it manually) |
| `training_summary` (`result.timeline`) | End-of-round screen with the 30-s timeline of the score |
| `warning` (`eeg_gap`, `eye_gap`) | "Connection to the headset/camera was interrupted" |
While a channel the baseline relies on is bad, `score.paused.eeg` explains why and the score continues from the camera (and clicks) alone.

## 5. Minimal browser client (untested sketch -- the protocol itself is tested by `server/client_demo.py`)
```js
const ws = new WebSocket("wss://your-host/ws");                       // always TLS in production
ws.onopen = () => ws.send(JSON.stringify({type: "hello", user_id: "u1", fs: 250,
                                          channels: ["Fp1","Fp2","F3","F4","O1","O2"], token: AUTH_TOKEN}));
ws.onmessage = (e) => { const m = JSON.parse(e.data); render(m); };    // m.type: score | quality | ...

function sendEEG(tStart, f32, nSamples) {                              // f32: Float32Array, channel-major (channels x samples), microvolts
  const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  let s = ""; for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  ws.send(JSON.stringify({type: "eeg", t_start: tStart, samples: nSamples, data: btoa(s)}));
}
function sendEye(rows) { ws.send(JSON.stringify({type: "eye", rows})); }   // [{t, ear, gaze_x, gaze_y, face_ok, head_x, head_y, head_roll_deg}, ...]
function sendEvent(kind, t) { ws.send(JSON.stringify({type: "event", kind, t})); }   // kind: "change" | "click" (training only)
// workflows:  {type:"rest", action:"start"|"finish"}   {type:"training", action:"start"|"finish", add_to_baseline:true}
```

## 6. Running and scaling
```bash
pip install -r requirements.txt fastapi uvicorn
FOCUS_BASELINE_DIR=data/baselines FOCUS_API_TOKEN=change-me uvicorn server.app:app --host 0.0.0.0 --port 8000
python server/client_demo.py                                  # in-process demo of the full flow
python server/client_demo.py --url ws://localhost:8000/ws     # against the running server (pip install websockets)
```
* Engine state lives in memory per connection. With several workers or a load balancer, use **sticky sessions** (or one process per user group).
  If the socket drops, the client reconnects, sends `hello` again (the stored baseline is reloaded) and the engine warms up for 15 s.
* Measured (one laptop core, synthetic data): ~11 ms per passive update (every 5 s), ~14 ms per training update, ~19 ms per second of EEG for the quality
  loop, i.e. roughly 2% of one core per user. Scoring runs in a worker thread so the event loop stays responsive; use several processes for many users.
* Replace the file store in `server/app.py` with your database; replace the token check with real authentication; keep everything behind TLS.

## 7. Not done / not verified
* No real headset, Bluetooth stack, browser camera or phone was used -- all tests use synthetic streams (14 engine/server tests + the script tests).
* Landmark SDK differences, headset-specific sample-rate/units/reference (check that your stream is in microvolts and the channel names are 10-20 names),
  and all default thresholds and weights still need pilot data.
* No automatic reconnection logic, rate limiting, or multi-user load test.
