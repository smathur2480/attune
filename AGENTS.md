# Instructions for coding agents (Codex, Copilot, etc.)

This repo is the AttUne focus-score engine. Read `README.md`, then `docs/OPENBCI.md` (how the headset, server and website connect) before changing anything.

* **Never reimplement the score in the website.** Scores come from `simple_score.py` / `complex_score.py` through `realtime.FocusEngine`, served by `server/app.py`.
  All numbers (weights, thresholds) live in `focus_core/constants.py` and nowhere else.
* **Website <-> server protocol:** use `web-client/attune-client.js` (do not hand-roll the WebSocket messages). Every timestamp you send must be on the
  server clock: `client.now()`. Stamp a colour change when the frame showing it is presented.
* **The website is a client with role "app"; the headset bridge (`bridge/openbci_bridge.py`) is role "device".** Both join the same `session_id`.
* **Never put credentials, raw EEG, baselines or participant data in git** (`.gitignore` covers `baseline*.json`, `data/`, recordings).
* **Run the tests before every commit:** `pip install -r requirements-dev.txt && python -m unittest discover -s tests`.
* Wellness wording only: never claim diagnosis, treatment or medical-device status in any UI text.
