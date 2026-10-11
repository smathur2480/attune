"""
server/app.py -- WebSocket service around realtime.FocusEngine, with SHARED SESSIONS so a headset bridge and a website can feed ONE engine.

Run (from the repo root):   pip install -r requirements-server.txt
                            FOCUS_BASELINE_DIR=data/baselines uvicorn server.app:app --host 127.0.0.1 --port 8000
Endpoint: ws://host:8000/ws        Health: GET /health

WHO CONNECTS (both join the same session_id)
  role "device": the headset bridge (bridge/openbci_bridge.py). Streams EEG; supplies fs and channel names. May send only: eeg, clock, status, ping.
  role "app"   : the website. Sends camera rows, click events, training/rest commands; receives every score / quality / warning message.
  The engine is created by the first hello that carries fs + channels (normally the device). If the app connects first it gets
  {"type":"ready","engine_ready":false}; messages that need the engine are answered with {"type":"waiting"} until the device joins, and
  everyone then receives {"type":"device_connected", fs, channels}.

PROTOCOL (JSON text frames). Client -> server:
  {"type":"hello","user_id":"u1","session_id":"u1","role":"app"|"device","fs":125,"channels":["Fp1","Fp2","AF7","AF8"],"token":"...",
   "profile":{...}|null,"window_s":30,"update_s":5,"acks":false}       (any message may also carry "ack":true to get {"type":"ack"} after its replies)
  {"type":"clock","t":<client epoch seconds>}   -> {"type":"clock","t":<echo>,"server_t":<server epoch seconds>}   (to sender only)
        NTP-style: offset = server_t - (t_send + t_recv)/2. Stamp EVERYTHING you send as  client_time + offset  so the headset, the camera and the
        clicks share the SERVER's clock even when they run on different machines.
  {"type":"status"} -> {"type":"status", ...engine status..., "connections":{"app":n,"device":m}}        (to sender only)
  {"type":"eeg","t_start":12.0,"data":"<base64 float32 little-endian, channel-major>","samples":62}  (t_start optional: omit it unless there was a dropout;
       or {"type":"eeg","t_start":12.0,"samples":[[...],[...]]} as plain JSON lists)
  {"type":"eye","rows":[{"t":12.01,"ear":0.31,"gaze_x":0.5,"gaze_y":0.5,"face_ok":1,"head_x":..,"head_y":..,"head_roll_deg":..}, ...]}
  {"type":"event","kind":"change"|"click","t":15.2}                            (training only)
  {"type":"training","action":"start"|"finish","t":..,"add_to_baseline":true}   {"type":"rest","action":"start"|"finish","t":..,"reset":false}
  {"type":"profile","profile":{...}}   {"type":"ping"} -> {"type":"pong"}
Server -> client: every FocusEngine message is sent to ALL connections of the session (score, quality, suggest_training, training_summary, ...), plus
  ready, device_connected, device_disconnected, waiting, status, clock, pong, error, ack.
Personal data stays server-side: baselines are saved to FOCUS_BASELINE_DIR/<user_id>.json (REPLACE WITH YOUR DATABASE); clients only receive a short summary.
Set FOCUS_API_TOKEN to require the hello token (a placeholder for real authentication; use TLS: wss://).
Engine calls run in worker threads under a per-session lock, so the event loop stays responsive and a session's calls never overlap.
"""
import asyncio
import base64
import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402

from focus_core import baselines, fileio  # noqa: E402
from realtime import FocusEngine  # noqa: E402

BASELINE_DIR = os.environ.get("FOCUS_BASELINE_DIR", "data/baselines")
REFERENCE_PATH = os.environ.get("FOCUS_REFERENCE", "")
API_TOKEN = os.environ.get("FOCUS_API_TOKEN", "")
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")          # ids become file names / keys: no path tricks
DEVICE_ALLOWED = {"eeg", "clock", "status", "ping"}
NEEDS_ENGINE = {"eeg", "eye", "event", "training", "rest", "profile"}

app = FastAPI(title="Focus engine")


class Session:
    def __init__(self, sid, user_id):
        self.sid, self.user_id = sid, user_id
        self.engine, self.profile, self.config, self.kw = None, None, None, {}
        self.lock = threading.Lock()                  # engine calls never overlap (they run in worker threads)
        self.conns = {}                               # websocket -> {"role": ..., "acks": bool}


SESSIONS = {}


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(SESSIONS)}


def _baseline_path(user_id):
    return os.path.join(BASELINE_DIR, f"{user_id}.json")


def _decode_eeg(msg, n_channels):
    if "data" in msg:
        raw = np.frombuffer(base64.b64decode(msg["data"]), dtype="<f4")
        n = int(msg["samples"])
        if raw.size != n_channels * n:
            raise ValueError(f"EEG payload has {raw.size} values, expected {n_channels} x {n}")
        return raw.reshape(n_channels, n).astype(float)
    return np.asarray(msg["samples"], float)


def _make_engine(sess, fs, channels, kw):
    uid = sess.user_id
    baseline = fileio.read_json(_baseline_path(uid)) if os.path.exists(_baseline_path(uid)) else None
    reference = fileio.read_json(REFERENCE_PATH) if REFERENCE_PATH and os.path.exists(REFERENCE_PATH) else None
    sess.engine = FocusEngine(float(fs), list(channels), baseline=baseline, reference=reference, profile=sess.profile, **kw)
    sess.config = dict(fs=float(fs), channels=list(channels))


def _handle(sess, msg):
    """Synchronous engine call (runs in a worker thread, under the session lock). Returns the outbound messages."""
    engine, t, out = sess.engine, msg.get("type"), []
    with sess.lock:
        if t == "eeg":
            out = engine.push_eeg(_decode_eeg(msg, len(engine.channels)), msg.get("t_start"))
        elif t == "eye":
            out = engine.push_eye(msg["rows"])
        elif t == "event":
            out = engine.push_event(msg["kind"], msg["t"])
        elif t == "training":
            if msg["action"] == "start":
                out = engine.start_training(msg.get("t"))
            else:
                out = engine.finish_training(add_to_baseline=bool(msg.get("add_to_baseline")), force=bool(msg.get("force")))
        elif t == "rest":
            out = engine.start_rest(msg.get("t")) if msg["action"] == "start" else engine.finish_rest(reset=bool(msg.get("reset")))
        elif t == "profile":
            sess.profile = msg.get("profile")
            engine.update_profile(sess.profile)
        else:
            raise ValueError(f"unknown message type {t!r}")
        cleaned = []
        for m in out:                                  # persist baseline changes server-side; never send the personal baseline out
            if m.get("baseline"):
                os.makedirs(BASELINE_DIR, exist_ok=True)
                fileio.write_json(m["baseline"], _baseline_path(sess.user_id))
                m = {k: v for k, v in m.items() if k != "baseline"}
                m["baseline_summary"] = baselines.summary(engine.baseline)
            cleaned.append(m)
    return cleaned


async def _send(ws, m):
    try:
        await ws.send_json(m)
        return True
    except Exception:
        return False


async def _broadcast(sess, msgs):
    for ws in list(sess.conns):
        for m in msgs:
            if not await _send(ws, m):
                sess.conns.pop(ws, None)
                break


def _counts(sess):
    c = {"app": 0, "device": 0}
    for info in sess.conns.values():
        c[info["role"]] += 1
    return c


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    sess, me = None, None
    try:
        while True:
            want_ack = False
            try:
                msg = json.loads(await websocket.receive_text())
                want_ack = isinstance(msg, dict) and bool(msg.get("ack"))
                t = msg.get("type")
                if t == "hello":
                    if sess is not None:
                        raise ValueError("hello already sent on this connection")
                    if API_TOKEN and msg.get("token") != API_TOKEN:
                        await websocket.send_json(dict(type="error", text="authentication failed"))
                        await websocket.close(code=4401)
                        return
                    uid, sid, role = str(msg.get("user_id", "")), str(msg.get("session_id") or msg.get("user_id", "")), msg.get("role", "app")
                    if not ID_RE.match(uid) or not ID_RE.match(sid):
                        raise ValueError("user_id and session_id must match [A-Za-z0-9_-]{1,64}")
                    if role not in ("app", "device"):
                        raise ValueError("role must be 'app' or 'device'")
                    cur = SESSIONS.get(sid)
                    if cur is not None and cur.user_id != uid:
                        raise ValueError("this session belongs to another user")
                    new = cur or Session(sid, uid)
                    kw = {k: msg[k] for k in ("window_s", "update_s", "smoothing", "warmup_s") if k in msg}
                    engine_created = False
                    if "fs" in msg and "channels" in msg:
                        if new.engine is None:
                            _make_engine(new, msg["fs"], msg["channels"], kw)
                            engine_created = True
                        elif new.config["fs"] != float(msg["fs"]) or new.config["channels"] != list(msg["channels"]):
                            raise ValueError(f"session already runs with fs={new.config['fs']:g} and channels={new.config['channels']}")
                    if msg.get("profile") is not None:
                        new.profile = msg["profile"]
                        if new.engine is not None:
                            new.engine.update_profile(new.profile)
                    SESSIONS[sid] = new
                    sess, me = new, {"role": role, "acks": bool(msg.get("acks"))}
                    sess.conns[websocket] = me
                    eng = sess.engine
                    await websocket.send_json(dict(type="ready", role=role, session_id=sid, engine_ready=eng is not None,
                                                   has_baseline=bool(eng and eng.baseline),
                                                   baseline_summary=baselines.summary(eng.baseline) if (eng and eng.baseline) else [],
                                                   config=sess.config, engine=eng.status() if eng else None))
                    if engine_created:
                        await _broadcast(sess, [dict(type="device_connected", **sess.config)])
                elif sess is None:
                    raise ValueError("send a hello message first")
                elif t == "clock":
                    await websocket.send_json(dict(type="clock", t=msg.get("t"), server_t=time.time()))
                elif t == "ping":
                    await websocket.send_json(dict(type="pong"))
                elif t == "status":
                    st = sess.engine.status() if sess.engine else dict(engine_ready=False)
                    await websocket.send_json(dict(type="status", **st, connections=_counts(sess)))
                else:
                    if me["role"] == "device" and t not in DEVICE_ALLOWED:
                        raise ValueError(f"a device connection may only send {sorted(DEVICE_ALLOWED)}")
                    if t in NEEDS_ENGINE and sess.engine is None:
                        await websocket.send_json(dict(type="waiting", text="No headset is connected to this session yet."))
                    else:
                        await _broadcast(sess, await asyncio.to_thread(_handle, sess, msg))
            except WebSocketDisconnect:
                raise
            except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as e:          # bad input must not kill the session
                await _send(websocket, dict(type="error", text=f"{type(e).__name__}: {e}"))
            if me is not None and (me["acks"] or want_ack):
                await _send(websocket, dict(type="ack"))
            elif me is None and want_ack:
                await _send(websocket, dict(type="ack"))
    except WebSocketDisconnect:
        pass
    finally:
        if sess is not None:
            sess.conns.pop(websocket, None)
            if me and me["role"] == "device" and sess.conns:
                await _broadcast(sess, [dict(type="device_disconnected")])
            if not sess.conns and SESSIONS.get(sess.sid) is sess:
                SESSIONS.pop(sess.sid, None)
