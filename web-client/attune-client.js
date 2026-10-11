/**
 * web-client/attune-client.js -- dependency-free ES module for the website (browser or Node >= 22).
 * Talks to server/app.py over one WebSocket. Protocol: see the header of server/app.py and docs/OPENBCI.md.
 *
 *   import { AttuneClient } from "./attune-client.js";
 *   const c = new AttuneClient({ url: "ws://127.0.0.1:8000/ws", userId: "u1" });          // role "app" (the website)
 *   c.on("score", (m) => render(m.score, m.components));  c.on("quality", (m) => showPrompts(m.prompts));
 *   const ready = await c.connect();                       // joins the session, measures the clock offset to the server
 *   // camera loop (30 fps): c.sendFrame({ ear, gazeX, gazeY, headX, headY, headRollDeg, faceOk: 1 });   // stamped with server time
 *   // training:  c.startTraining();  c.sendEvent("change");  c.sendEvent("click");  c.finishTraining({ addToBaseline: true });
 *
 * CLOCKS: every timestamp you send must be on the SERVER's clock. c.now() = this device's high-resolution epoch time + the offset measured in
 * connect() (NTP-style, median of the lowest-round-trip samples). Stamp a colour change when the frame showing it is PRESENTED, and use the
 * camera frame's capture time when you have it (pass `t: c.toServerTime(epochSeconds)`).
 * A web page that streams the headset itself (Web Bluetooth) can use role "device" with fs and channels and call sendEeg().
 */
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function toBase64(u8) {
  if (typeof btoa === "function") {
    let s = "";
    for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode(...u8.subarray(i, i + 0x8000));
    return btoa(s);
  }
  return Buffer.from(u8).toString("base64");
}

export class AttuneClient {
  constructor(o) {
    this.o = { role: "app", sessionId: null, token: null, profile: null, fs: null, channels: null, windowS: null, updateS: null, ...o };
    this.offset = 0;
    this.ws = null;
    this.ready = null;
    this._handlers = {};
    this._clockWaiters = new Map();
    this._statusWaiters = [];
    this._eyeBuf = [];
    this._timer = null;
  }

  on(type, cb) { (this._handlers[type] ||= []).push(cb); return this; }
  _emit(type, m) { for (const cb of this._handlers[type] || []) cb(m); }
  _send(m) { if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(m)); }

  /** Server time in epoch seconds (valid after connect()). */
  now() { return (performance.timeOrigin + performance.now()) / 1000 + this.offset; }
  /** Convert a time on this device's clock (epoch seconds) to server time. */
  toServerTime(epochSeconds) { return epochSeconds + this.offset; }

  connect() {
    return new Promise((resolve, reject) => {
      const o = this.o;
      const ws = (this.ws = new WebSocket(o.url));
      ws.onopen = () => this._send({
        type: "hello", user_id: o.userId, session_id: o.sessionId || o.userId, role: o.role, token: o.token || undefined, profile: o.profile || undefined,
        fs: o.fs || undefined, channels: o.channels || undefined, window_s: o.windowS || undefined, update_s: o.updateS || undefined,
      });
      ws.onerror = () => reject(new Error("WebSocket error: is the server running at " + o.url + "?"));
      ws.onclose = () => { clearInterval(this._timer); this._emit("close", {}); };
      ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data);
        if (m.type === "ready") { this.ready = m; resolve(m); }
        else if (m.type === "error" && !this.ready) reject(new Error(m.text));
        else if (m.type === "clock") { const w = this._clockWaiters.get(m.t); if (w) { this._clockWaiters.delete(m.t); w(m); } }
        else if (m.type === "status") { const w = this._statusWaiters.shift(); if (w) w(m); }
        this._emit(m.type, m);
        this._emit("*", m);
      };
    }).then(async (ready) => {
      await this.syncClock();
      this._timer = setInterval(() => this._flushEye(), 100);
      if (this._timer.unref) this._timer.unref();
      return ready;
    });
  }

  /** NTP-style offset to the server clock; keeps the lowest-round-trip half of n samples. */
  async syncClock(n = 9) {
    const samples = [];
    for (let i = 0; i < n; i++) {
      const t0 = Date.now() / 1000;                                 // plain epoch seconds
      const reply = new Promise((res) => this._clockWaiters.set(t0, res));
      this._send({ type: "clock", t: t0 });
      const m = await Promise.race([reply, sleep(3000).then(() => null)]);
      if (!m) continue;
      const t1 = Date.now() / 1000;
      samples.push([t1 - t0, m.server_t - (t0 + t1) / 2]);
      await sleep(15);
    }
    if (!samples.length) throw new Error("clock sync failed: no replies from the server");
    samples.sort((a, b) => a[0] - b[0]);
    const best = samples.slice(0, Math.max(3, Math.floor(n / 2))).map((s) => s[1]).sort((a, b) => a - b);
    // offset between OUR high-resolution clock and the server's: use Date.now() based estimate, then express it relative to performance.timeOrigin+now
    const hiNow = (performance.timeOrigin + performance.now()) / 1000;
    this.offset = best[Math.floor(best.length / 2)] + (Date.now() / 1000 - hiNow);
    this.rttMs = samples[0][0] * 1000;
    return this.offset;
  }

  /** One camera frame. Times default to "now" on the server clock. */
  sendFrame(f) {
    this._eyeBuf.push({
      t: f.t ?? this.now(), ear: f.ear, gaze_x: f.gazeX ?? null, gaze_y: f.gazeY ?? null, face_ok: f.faceOk ?? 1,
      head_x: f.headX ?? null, head_y: f.headY ?? null, head_roll_deg: f.headRollDeg ?? null,
    });
  }
  _flushEye() { if (this._eyeBuf.length) { this._send({ type: "eye", rows: this._eyeBuf }); this._eyeBuf = []; } }
  flush() { this._flushEye(); }

  sendEvent(kind, t) { this._send({ type: "event", kind, t: t ?? this.now() }); }
  startTraining(t) { this._send({ type: "training", action: "start", t: t ?? this.now() }); }
  finishTraining({ addToBaseline = false, force = false } = {}) { this._flushEye(); this._send({ type: "training", action: "finish", add_to_baseline: addToBaseline, force }); }
  startRest(t) { this._send({ type: "rest", action: "start", t: t ?? this.now() }); }
  finishRest({ reset = false } = {}) { this._flushEye(); this._send({ type: "rest", action: "finish", reset }); }
  setProfile(profile) { this._send({ type: "profile", profile }); }

  /** role "device" only: EEG as a Float32Array, channel-major (channels x samples), microvolts. tStart (server time) only on the first chunk or after a dropout. */
  sendEeg(f32, samples, tStart) {
    const u8 = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
    const m = { type: "eeg", samples, data: toBase64(u8) };
    if (tStart !== undefined) m.t_start = tStart;
    this._send(m);
  }

  status() { return new Promise((res) => { this._statusWaiters.push(res); this._send({ type: "status" }); }); }
  close() { clearInterval(this._timer); if (this.ws) this.ws.close(); }
}
