# Connecting the OpenBCI Cyton Daisy, the website and this repo

```
 ┌─────────────────────── the user's laptop ───────────────────────┐              ┌──────────────── optional ────────────────┐
 │  OpenBCI Cyton + Daisy ─RF─▶ USB dongle ─▶ bridge/openbci_bridge.py ──WebSocket──┐                                          │
 │   (16 ch, 125 Hz, battery)    (BrainFlow)        role "device"                  ▼                                          │
 │                                                                         server/app.py  ──▶ database (baselines, scores)    │
 │  Website (browser): camera, clicks, UI ───────────WebSocket (role "app")────────▲   (FocusEngine per session)              │
 │   web-client/attune-client.js                                                                                              │
 └──────────────────────────────────────────────────────────────────────────────────┘              └───────────────────────────┘
```
The bridge and the website are two clients of **one session** on the server: the bridge sends EEG, the website sends camera rows and clicks, and
both receive every score and electrode-quality message. If the server runs on the same laptop, raw EEG never leaves it.

## 1. Hardware facts we rely on
* `BoardIds.CYTON_DAISY_BOARD` (= 2) in BrainFlow, `serial_port` = the dongle's port. 16 EEG channels, **125 Hz per channel** (a plain Cyton is 8 channels at 250 Hz).
  The bridge reads the rate from BrainFlow instead of hard-coding it. BrainFlow gives EEG in microvolts and epoch-second timestamps.
* Channel numbers in `--map` are the numbers printed on the boards: Cyton 1-8, Daisy 9-16. Unmapped channels are ignored.
* **Our montage:** four frontal dry electrodes on Cyton channels 1-4 (`1=Fp1,2=Fp2,3=AF7,4=AF8`) and the ear clip on the board's reference and BIAS pins.
  Hair-covered sites (O1/O2, etc.) are not needed. Check OpenBCI's current guide for which pins take the reference and BIAS electrodes.
* **Safety (OpenBCI guidance, please confirm in their current docs):** battery power only while electrodes are on a person; use the USB dongle;
  never connect the board itself to a computer with a USB cable while it is attached to someone.
* Dongle port: Windows = a COM port in Device Manager (e.g. `COM3`); macOS = `/dev/cu.usbserial-*`; Linux = `/dev/ttyUSB0` (your user may need the `dialout` group).

## 2. Run it (three terminals)
```bash
pip install -r requirements-server.txt -r requirements-bridge.txt
# 1) the engine server (same laptop = raw EEG stays local)
FOCUS_BASELINE_DIR=data/baselines uvicorn server.app:app --host 127.0.0.1 --port 8000
# 2) the headset bridge (real hardware)
python bridge/openbci_bridge.py --board cyton_daisy --serial-port COM3 --user-id u1 --map "1=Fp1,2=Fp2,3=AF7,4=AF8"
#    no hardware? BrainFlow's synthetic board (sine waves, so the quality coach will say "looks like noise"; that is correct):
python bridge/openbci_bridge.py --board synthetic --user-id u1 --duration 60
# 3) the website: connect as role "app" with the same user id / session id (see section 3)
```
The bridge prints electrode prompts ("left outer forehead (AF7): no signal. Press the sensor firmly…") and the clock offset, and retries if the server drops.

## 3. The website (role "app") with `web-client/attune-client.js`
```js
import { AttuneClient } from "./attune-client.js";
const c = new AttuneClient({ url: "ws://127.0.0.1:8000/ws", userId: "u1" });      // sessionId defaults to userId; use the same one as the bridge
c.on("quality", (m) => showElectrodeMap(m.channels, m.prompts, m.ready));        // electrode map + fix-it text
c.on("score",   (m) => showScore(m.score, m.components, m.patterns_present, m.paused));
c.on("device_connected",    () => setHeadsetStatus("connected"));
c.on("device_disconnected", () => setHeadsetStatus("lost"));
const ready = await c.connect();                    // joins, measures the clock offset to the server
setHeadsetStatus(ready.engine_ready ? "connected" : "waiting for the headset");
// camera loop (about 30 fps), from your face-landmark SDK:
c.sendFrame({ ear, gazeX, gazeY, headX, headY, headRollDeg, faceOk: 1 });         // stamped with server time automatically
// task: when the frame that shows the new colour is presented, and on every click:
c.sendEvent("change");  c.sendEvent("click");
c.startRest(); /* ... 2 min ... */ c.finishRest();                                 // resting calibration
c.startTraining(); /* ... */ c.finishTraining({ addToBaseline: true });          // orbiting-ball round
```
UI states to build: *waiting for headset* (`ready.engine_ready === false`), *connected*, *electrodes need fixing* (`quality.ready === false` + prompts), *score paused for EEG*
(`score.paused.eeg`), *headset lost* (`device_disconnected`). The website never talks to the headset and never computes a score.

## 4. How the real-time syncing works
| Problem | Mechanism |
|---|---|
| EEG, camera and clicks come from different clocks (or machines) | Each client measures its offset to the **server clock** (NTP-style: median of the lowest-round-trip half of 9 pings) and stamps everything as `local time + offset`. Same laptop: offset is about 0 ms (tested). |
| Bluetooth/USB packets arrive in bursts | EEG time inside the engine advances by **sample count**, so burst jitter adds nothing. |
| Dropped packets or a stalled laptop | The bridge counts the headset's packet numbers and sends `t_start` after a loss (and on the first chunk, or if host and sample clocks drift apart by more than 0.2 s); the engine then restarts its buffer and warm-up and tells the app (`eeg_gap`). |
| Slow clock drift between the headset crystal and the laptop | A few ms per minute: harmless for 30-s windows and 3-minute rounds; re-anchored automatically past 0.2 s. |
| Stale data after a disconnect | The engine refuses to score from a stream that stopped (`paused` notes), and the app is told when the headset leaves. |
| Colour-change timing | Stamp when the frame showing the colour is **presented**, not when code set it; constant display delay is absorbed by the personal baseline, variable delay is not. |

At 125 Hz the usable EEG band is below 62.5 Hz (we use 1-45 Hz). 60 Hz hum sits just below that limit and the quality monitor still detects it (tested);
this required fixing a gate that would have disabled hum detection below 128 Hz.

## 5. The GitHub repo
```
repo/
  baseline.py simple_score.py complex_score.py relax_score.py realtime.py     scoring engine (Python)
  focus_core/                                                                  shared formulas, features, quality monitor
  server/app.py                                                                WebSocket service with shared sessions
  bridge/openbci_bridge.py                                                     OpenBCI -> server
  web-client/attune-client.js                                                  the website's connection (copy or import)
  web/                                                                         <- put the Codex-built website here (monorepo), or keep it in its own repo
  docs/  tests/  AGENTS.md  .github/workflows/tests.yml  requirements*.txt
```
* **Monorepo (recommended for a hackathon):** move the website into `web/` and import `../web-client/attune-client.js`; one clone runs everything; CI covers it all.
* **Two repos:** copy `web-client/attune-client.js` into the website repo (it has no dependencies) and treat the protocol in `server/app.py`'s header as the contract.
* **Codex:** `AGENTS.md` tells coding agents the rules (never reimplement the score; use the client; no secrets/data in git). Codex reads `AGENTS.md` (verify in its current docs).
* **CI:** `.github/workflows/tests.yml` runs all tests on Python 3.11/3.12 with Node 22 (written for GitHub Actions; not run from the authoring environment).
* **Secrets:** `FOCUS_API_TOKEN`, database credentials and `.env` stay out of git (GitHub Actions secrets / host environment variables).

## 6. Where to run the server
| Choice | Raw EEG leaves the laptop? | Notes |
|---|---|---|
| Server on the same laptop (`127.0.0.1`) | No | Best for the privacy claim and for the demo (no network risk). The website must be served to that browser too (any static host works; a page served over `https` may refuse `ws://127.0.0.1` in some browsers: serve it from localhost for the demo, or use `wss://` with a certificate). |
| Hosted server (`wss://`) | Yes (the bridge uploads raw EEG) | Add real authentication, TLS, consent and retention rules first. Needs sticky sessions if you run several workers (a session's engine lives in memory). |
The privacy slide's "raw EEG stays on the phone" is a design target: the engine would have to move onto the device (it is cheap: about 11 ms per update).

## 7. What is and is not tested
Tested here (synthetic data): the bridge process with BrainFlow's synthetic board and a real server on a real socket (9 s, about 2,265 samples, 0 dropped packets, 0.0 ms offset); shared sessions
(device first, app first, role limits, mismatches, disconnects, cleanup); the JavaScript client under Node 22; 125 Hz scoring and hum detection; AF7/AF8 handling.
**Not tested:** a real Cyton Daisy, the dongle and drivers, real packet loss, BrainFlow timestamp behaviour on real hardware, real dry-electrode contact, browsers and phones,
different-machine clock sync over a real network, and the GitHub Actions workflow. Plan a short hardware session to check: channel numbering against `--map`, 125 Hz confirmed in the
bridge's first line, the ear-clip reference/BIAS wiring, and the quality coach's thresholds on real signals (they were tuned on simulated EEG).

## 8. Troubleshooting
* `BrainFlow is not usable (No module named 'pkg_resources')` -> `pip install "setuptools<81"` (in `requirements-bridge.txt`).
* Bridge says it cannot open the board -> wrong `--serial-port`, dongle not plugged in, board switched off or battery flat, or another program (the OpenBCI GUI) holds the port.
* Website sees `waiting` messages -> the bridge has not joined yet, or it uses a different `session_id`/`user_id`.
* `no usable frontal channel after screening` while calibrating -> fix the electrodes first (the bridge console shows the prompts).
* Score paused for EEG -> a channel the baseline relies on is bad or the headset stream stalled; `score.paused.eeg` says which.
