#!/usr/bin/env python3
"""
realtime.py -- the REAL-TIME ENGINE: turns the batch scoring scripts into a live, stateful service component.

The three scripts are file-in / JSON-out. A live app needs something that (1) keeps rolling buffers of three streams (EEG chunks, camera
frames, click events), (2) re-scores the last N seconds every few seconds with the SAME functions (simple_score.score_simple /
complex_score.score_complex -- nothing is re-implemented here), (3) pauses the EEG part when electrodes are bad and tells the user how to fix
them, and (4) runs the calibration / training workflows. That is FocusEngine. It is framework-agnostic: call it from a Python backend directly,
or put it behind the WebSocket service in server/app.py.

USAGE (one engine per user session; call from ONE thread or task; each call returns a list of message dicts to send to the UI)
    eng = FocusEngine(fs=250, channels=["Fp1","Fp2","F3","F4","O1","O2"], baseline=baseline_dict, profile=profile_dict_or_None)
    msgs  = eng.push_eeg(chunk, t_start)        # chunk: (channels, samples) microvolts; t_start: seconds on the shared clock
    msgs += eng.push_eye([{"t":..,"ear":..,"gaze_x":..,"gaze_y":..,"face_ok":1,"head_x":..,"head_y":..,"head_roll_deg":..}, ...])
    msgs += eng.push_event("click", t)          # training only: "change" (colour changed) / "click"
    eng.start_training(t) ... eng.finish_training(add_to_baseline=True)       # orbiting-ball session
    eng.start_rest(t) ... eng.finish_rest()                                    # 2-minute eyes-open resting calibration
Message types: quality | score | score_unavailable | suggest_training | warning | training_started | training_summary | rest_started |
               rest_calibrated | rest_failed

CLOCKS (the most common real-time bug): EEG chunks, camera frames and clicks must carry timestamps from ONE clock (e.g. performance.now()/1000).
EEG time is advanced by the SAMPLE COUNT (so bursty Bluetooth packets do not add jitter); t_start only detects dropouts (gap > gap_tol_s ->
buffers reset + warning). Stamp the colour change at the moment the frame showing it is PRESENTED, not when code set the colour.
  [PVT2020] "Response speed measurements on the psychomotor vigilance test: how precise is precise enough?" Sleep 2020 (PMID 32556295,
      authors not retrieved): reaction-time metrics need timing bias within about +-5 ms (SD <= 10 ms).
  [Casadei2022] Casadei V, Ferrero R (2022). Online skin-electrode contact quality monitoring in wearable devices: an EEG application.
      I2MTC 2022: contact quality should be monitored CONTINUOUSLY during recording, not only before it (hence the 1-s quality loop).
  [Shelat2024] Shelat S, Schooler JW, Giesbrecht B (2024). Frontiers in Cognition 3:1460349: real-time triggering from within-person
      boundaries (see focus_core/trigger.py).

LATENCY / SMOOTHING: the score describes the last `window_s` (default 30 s) of data, so it LAGS reality by roughly half a window; shorter windows
react faster but are noisier (a 30-s window holds only ~5 blinks). Scores are updated every `update_s` (default 5 s) and exponentially smoothed
in strength space (smoothing=0.5), then mapped with the SAME clip(50 + 25 m, 1, 100). Compute cost measured on a laptop: ~11 ms per passive
update, ~14 ms per training update, ~19 ms per second of EEG for the quality monitor.
Wellness estimate only -- not a diagnosis or medical advice.
"""
import copy

import numpy as np
import pandas as pd

from complex_score import score_complex
from focus_core import baseline_ops
from focus_core import constants as K
from focus_core import scoring
from focus_core import signal_quality as sq
from focus_core import trigger
from simple_score import score_simple


class FocusEngine:
    def __init__(self, fs, channels, baseline=None, reference=None, profile=None, *, window_s=30.0, update_s=5.0, smoothing=0.5,
                 warmup_s=15.0, gap_tol_s=0.25, eye_gap_s=1.0, stale_s=2.0, orbit_period=K.ORBIT_PERIOD_S, orbit_radius=K.ORBIT_RADIUS, mains_hz=60.0,
                 thresholds=None, suggest_training=True, trigger_cooldown_s=600.0, personal_scores=None):
        self.fs, self.channels = float(fs), list(channels)
        self.baseline, self.reference, self.profile = baseline, reference, profile
        self.window_s, self.update_s, self.alpha, self.warmup_s = float(window_s), float(update_s), float(smoothing), float(warmup_s)
        self.gap_tol_s, self.orbit_period, self.orbit_radius = float(gap_tol_s), orbit_period, orbit_radius
        self.eye_gap_s, self.stale_s = float(eye_gap_s), float(stale_s)
        self.suggest, self.cooldown = suggest_training, trigger_cooldown_s
        self.keep_s = max(self.window_s + 10.0, 60.0)
        self.use_complex = profile is not None
        self.mains_hz, self.thresholds = mains_hz, thresholds
        base_ch = ((baseline or {}).get("eeg") or {}).get("front", []) + ((baseline or {}).get("eeg") or {}).get("post", [])
        missing = [c for c in base_ch if c not in self.channels]
        if missing:
            raise ValueError(f"baseline uses channels {missing} that the headset stream does not provide")
        self._reset_streams()
        self.mode = "passive"
        self._train = self._rest = None
        self._m_smooth = None
        self._last_tick = None
        self._qreport = None
        self._qstate = None
        self._hist, self._personal = [], list(personal_scores or [])
        self._last_suggest = -1e9
        self.last_training_session = None

    # ------------------------------------------------------------------------------------------------ internal state
    def _reset_streams(self):
        self._x = np.zeros((len(self.channels), 0))
        self._eeg_t_end = None
        self._eye = []
        self._now = 0.0
        self._first_t = None
        self._qmon = sq.QualityMonitor(self.fs, self.channels, mains_hz=self.mains_hz, thresholds=self.thresholds)

    def _keep_from(self):
        sess = self._train or self._rest
        return (sess["t0"] - 1.0) if sess else (self._now - self.keep_s)

    def _trim(self):
        lo = self._keep_from()
        if self._eeg_t_end is not None and self._x.shape[1]:
            t0e = self._eeg_t_end - self._x.shape[1] / self.fs
            drop = int((lo - t0e) * self.fs)
            if drop > 0:
                self._x = self._x[:, drop:]
        if self._eye and self._eye[0]["t"] < lo:
            self._eye = [r for r in self._eye if r["t"] >= lo]

    def _see(self, t):
        self._now = max(self._now, float(t))
        if self._first_t is None:
            self._first_t = float(t)

    def _baseline_channels(self):
        e = (self.baseline or {}).get("eeg")
        if e:
            return list(e["front"]) + list(e["post"])
        return [c for c in self.channels if c in K.FRONTAL or c in K.POSTERIOR]

    # ------------------------------------------------------------------------------------------------ inputs
    def push_eeg(self, chunk, t_start=None):
        chunk = np.asarray(chunk, float)
        if chunk.ndim != 2 or chunk.shape[0] != len(self.channels):
            raise ValueError(f"EEG chunk must be (channels={len(self.channels)}, samples); got {chunk.shape}")
        msgs = []
        if self._eeg_t_end is None:
            self._eeg_t_end = 0.0 if t_start is None else float(t_start)
            self._see(self._eeg_t_end)
        elif t_start is not None and abs(float(t_start) - self._eeg_t_end) > self.gap_tol_s:
            msgs.append(dict(type="warning", code="eeg_gap", t=float(t_start),
                             text=f"EEG data was interrupted for {float(t_start) - self._eeg_t_end:.1f} s; restarting the signal buffer."))
            self._x = np.zeros((len(self.channels), 0))
            self._qmon = sq.QualityMonitor(self.fs, self.channels, mains_hz=self.mains_hz, thresholds=self.thresholds)
            self._eeg_t_end = float(t_start)
            self._first_t = None                                   # warm up again before scoring
            self._last_tick = None
        self._x = np.concatenate([self._x, chunk], axis=1)
        self._eeg_t_end += chunk.shape[1] / self.fs
        self._see(self._eeg_t_end)
        rep = self._qmon.update(chunk)
        if rep is not None:
            self._qreport = rep
            key = (rep["overall"]["status"], rep["overall"]["ready"])
            if rep["new_prompts"] or rep["resolved"] or key != self._qstate:
                msgs.append(dict(type="quality", t=round(self._eeg_t_end, 2), status=rep["overall"]["status"], ready=rep["overall"]["ready"],
                                 quality=rep["overall"]["quality"], channels=rep["channels"], prompts=rep["new_prompts"],
                                 resolved=rep["resolved"], active=rep["active_issues"]))
            self._qstate = key
        self._trim()
        return msgs + self._maybe_tick()

    def push_eye(self, rows):
        rows = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else list(rows)
        msgs = []
        for r in rows:
            r = dict(r)
            r.setdefault("face_ok", 1)
            if self._eye and r["t"] - self._eye[-1]["t"] > self.eye_gap_s:
                msgs.append(dict(type="warning", code="eye_gap", t=float(r["t"]),
                                 text=f"Camera data was interrupted for {r['t'] - self._eye[-1]['t']:.1f} s; restarting the camera buffer."))
                self._eye = []
                self._first_t = None
                self._last_tick = None
            self._eye.append(r)
            self._see(r["t"])
        self._trim()
        return msgs + self._maybe_tick()

    def push_event(self, kind, t):
        if kind not in ("change", "click"):
            raise ValueError("event kind must be 'change' or 'click'")
        if not self._train:
            return [dict(type="warning", code="event_outside_training", text=f"{kind} event ignored: no training session is running.")]
        t = float(t)
        if t < self._train["t0"]:
            return []
        self._see(t)
        self._train["changes" if kind == "change" else "clicks"].append(t - self._train["t0"])
        return []

    # ------------------------------------------------------------------------------------------------ scoring
    def _eye_df(self, t_lo, t_hi, shift=0.0):
        rows = [r for r in self._eye if t_lo - 1e-6 <= r["t"] <= t_hi]
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["t"] = df["t"] - shift
        return df.reset_index(drop=True)

    def _eeg_slice(self, t_lo):
        if self._eeg_t_end is None or not self._x.shape[1]:
            return None
        t0e = self._eeg_t_end - self._x.shape[1] / self.fs
        seg = self._x[:, max(0, int(round((t_lo - t0e) * self.fs))):]
        return (seg, self.fs, self.channels) if seg.shape[1] >= K.EEG_MIN_SEGMENT_S * self.fs else None

    def _gate_eeg(self, eeg):
        """Pause the EEG part while a channel the baseline relies on is bad/poor."""
        if eeg is None or not self._qreport:
            return eeg, None
        bad = [c for c in self._baseline_channels() if self._qreport["channels"][c]["status"] in ("bad", "poor")]
        if bad:
            return None, "EEG paused until these sensors are fixed: " + ", ".join(sq.friendly(c) for c in bad)
        return eeg, None

    def _score(self, eeg, eye, task, timeline):
        if self.use_complex:
            return score_complex(self.profile, self.baseline, eeg, eye, task, self.reference, timeline=timeline)
        return score_simple(self.baseline, eeg, eye, task, self.reference, timeline=timeline)

    def _due(self):
        if self.mode == "rest" or self._first_t is None:
            return False
        t0 = self._train["t0"] if self._train else self._first_t
        warm = K.PURSUIT_MIN_USABLE_S if self._train else self.warmup_s
        if self._now - t0 < warm:
            return False
        return self._last_tick is None or self._now - self._last_tick >= self.update_s

    def _maybe_tick(self):
        if not self._due():
            return []
        self._last_tick = self._now
        return self._tick()

    def _tick(self):
        now = self._now
        eye_last = self._eye[-1]["t"] if self._eye else now                  # each stream's window ends at ITS OWN newest sample
        eeg_last = self._eeg_t_end if self._eeg_t_end is not None else now
        lo_e, lo_y = eeg_last - self.window_s, eye_last - self.window_s
        shift, task = 0.0, None
        if self._train:
            t0 = self._train["t0"]
            lo_e, lo_y, shift = max(lo_e, t0), max(lo_y, t0), t0
            lo, hi = lo_y - t0, eye_last - t0
            task = dict(changes=[c for c in self._train["changes"] if lo <= c <= hi - K.RESP_WINDOW_S[1]],      # only targets whose response window has closed
                        clicks=[c for c in self._train["clicks"] if lo <= c <= hi], ball=None, period=self.orbit_period,
                        radius=self.orbit_radius, duration_s=hi - lo)
        eeg, paused = self._gate_eeg(self._eeg_slice(lo_e))
        stalled = {}
        if eeg is not None and now - self._eeg_t_end > self.stale_s:
            eeg, stalled["eeg"] = None, f"EEG stream stalled for {now - self._eeg_t_end:.0f} s"
        eye = self._eye_df(lo_y, eye_last + 1e-9, shift)
        if eye is not None and (len(eye) < 30 or now - (eye["t"].iloc[-1] + shift) > self.stale_s):
            eye, stalled["eye"] = None, "camera stream stalled or too short"
        paused = ({"eeg": paused} if paused else {}) | stalled
        res = self._score(eeg, eye, task, timeline=False)
        # camera-side advice
        prompts = []
        if eye is not None:
            cam = sq.camera_ok_check(eye)
            if cam["status"] != "ok":
                prompts.append(cam["prompt"])
            if {"head_x", "head_y"} <= set(eye.columns):
                hm = sq.head_motion_check(eye)
                if hm.get("status") == "moving":
                    prompts.append(hm["prompt"])
        if "error" in res:
            return [dict(type="score_unavailable", t=now, mode=self.mode, reason=res["error"], notes=res.get("notes", []),
                         paused=paused, prompts=prompts)]
        key = "complex_mean_strength_sd" if self.use_complex else "mean_strength_sd"
        m = res[key]
        self._m_smooth = m if self._m_smooth is None else self.alpha * m + (1 - self.alpha) * self._m_smooth
        smooth = round(scoring.to_score(self._m_smooth), 1)
        msg = dict(type="score", t=now, mode=self.mode, kind=res["kind"], score=smooth, score_raw=res["complex_score" if self.use_complex else "simple_score"],
                   simple_score=res["simple_score"], components=res["components"], weights_used=res["weights_used"], quality=res["quality"],
                   baseline_basis=res["baseline_basis"], patterns_present=res["patterns_present"],
                   patterns_absent_or_reversed=res["patterns_absent_or_reversed"], window_s=round(min(self.window_s, now - (self._train["t0"] if self._train else self._first_t)), 1),
                   paused=paused, prompts=prompts, notes=[n for n in res["notes"] if n != K.DISCLAIMER])
        if self.use_complex:
            msg.update(complex_score=res["complex_score"], shifts_sd=res["shifts_sd"], age_sex_note=res["age_sex_note"])
        out = [msg]
        if self.mode == "passive":
            self._hist.append((now, smooth))
            self._personal.append(smooth)
            self._hist = [h for h in self._hist if h[0] >= now - 300.0]
            if self.suggest and now - self._last_suggest >= self.cooldown:
                ts, ss = zip(*self._hist)
                hit, thr, when = trigger.should_suggest_training(ts, ss, personal_scores=self._personal[:-len(self._hist)] or None)
                if hit:
                    self._last_suggest = now
                    out.append(dict(type="suggest_training", t=now, threshold=round(thr, 1),
                                    text="Your focus has been low for a while. Want to try a short focus-training round?"))
        return out

    # ------------------------------------------------------------------------------------------------ training workflow
    def start_training(self, t=None):
        t0 = self._now if t is None else float(t)
        self._train = dict(t0=t0, changes=[], clicks=[])
        self._see(t0)
        self.mode, self._last_tick, self._m_smooth = "training", None, None
        return [dict(type="training_started", t0=t0, orbit_period=self.orbit_period, orbit_radius=self.orbit_radius)]

    def finish_training(self, add_to_baseline=False, force=False):
        if not self._train:
            return [dict(type="warning", code="no_training", text="No training session is running.")]
        tr, now = self._train, self._now
        t0 = tr["t0"]
        eeg, paused = self._gate_eeg(self._eeg_slice(t0))
        eye = self._eye_df(t0, now, t0)
        task = dict(changes=tr["changes"], clicks=tr["clicks"], ball=None, period=self.orbit_period, radius=self.orbit_radius, duration_s=now - t0)
        res = self._score(eeg, eye, task, timeline=True) if eye is not None else dict(error="no camera data during the session")
        self.last_training_session = dict(changes=list(tr["changes"]), clicks=list(tr["clicks"]), eye=eye)
        msg = dict(type="training_summary", duration_s=round(now - t0, 1), result=res, paused=({"eeg": paused} if paused else {}), baseline_updated=False)
        if add_to_baseline and eye is not None:
            try:
                self.baseline, values, n = baseline_ops.add_training_session(copy.deepcopy(self.baseline), tr["changes"], tr["clicks"], eye,
                                                                            period=self.orbit_period, radius=self.orbit_radius, force=force)
                msg.update(baseline_updated=True, baseline=self.baseline, baseline_sessions=n)
            except ValueError as e:
                msg["baseline_error"] = str(e)
        self._train, self.mode, self._last_tick, self._m_smooth = None, "passive", None, None
        self._trim()
        return [msg]

    # ------------------------------------------------------------------------------------------------ resting calibration workflow
    def start_rest(self, t=None):
        t0 = self._now if t is None else float(t)
        self._rest = dict(t0=t0)
        self._see(t0)
        self.mode = "rest"
        return [dict(type="rest_started", t0=t0, text="Sit still, eyes open, looking at the screen for about 2 minutes.")]

    def finish_rest(self, reset=False):
        if not self._rest:
            return [dict(type="warning", code="no_rest", text="No resting calibration is running.")]
        t0, now = self._rest["t0"], self._now
        eeg, eye = self._eeg_slice(t0), self._eye_df(t0, now)
        self._rest, self.mode, self._last_tick = None, "passive", None
        try:
            if eeg is None and eye is None:
                raise ValueError("no data was received during the calibration")
            b, notes = baseline_ops.add_rest_session(copy.deepcopy(self.baseline), eeg, eye, reset=reset)
        except ValueError as e:
            return [dict(type="rest_failed", reason=str(e))]
        self.baseline = b
        return [dict(type="rest_calibrated", notes=notes, baseline=b)]

    # ------------------------------------------------------------------------------------------------ misc
    def update_profile(self, profile):
        self.profile, self.use_complex = profile, profile is not None

    def status(self):
        return dict(mode=self.mode, now=round(self._now, 2), eeg_buffer_s=round(self._x.shape[1] / self.fs, 1), eye_frames=len(self._eye),
                    has_baseline=bool(self.baseline), quality=(self._qreport or {}).get("overall"), last_tick=self._last_tick)
