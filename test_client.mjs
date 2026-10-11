// Node >= 22 test of attune-client.js against a running server. Usage: node web-client/test_client.mjs ws://127.0.0.1:PORT/ws
import { AttuneClient } from "./attune-client.js";
const url = process.argv[2];
const out = {};
const CH = ["Fp1", "Fp2", "AF7", "AF8"], FS = 125;
const dev = new AttuneClient({ url, userId: "jsuser", role: "device", fs: FS, channels: CH });
const app = new AttuneClient({ url, userId: "jsuser" });
const seen = {};
app.on("*", (m) => { seen[m.type] = (seen[m.type] || 0) + 1; });
out.appReadyFirst = (await app.connect()).engine_ready;                  // the website joins before the headset
out.appOffsetMs = Math.round(app.offset * 1000);
const dr = await dev.connect();
out.devReady = dr.engine_ready;
await new Promise((r) => setTimeout(r, 200));
out.deviceConnectedSeen = !!seen.device_connected;
// 6 seconds of fake EEG (microvolts), channel-major, in 0.25 s chunks, first chunk stamped with server time
const n = Math.round(FS * 0.25);
let t0 = dev.now();
for (let k = 0; k < 24; k++) {
  const f = new Float32Array(CH.length * n);
  for (let c = 0; c < CH.length; c++) for (let i = 0; i < n; i++) f[c * n + i] = 10 * Math.sin(2 * Math.PI * 10 * (k * n + i) / FS + c) + 3 * Math.sin(1.7 * (k * n + i) + 5 * c);
  dev.sendEeg(f, n, k === 0 ? t0 : undefined);
}
for (let i = 0; i < 40; i++) app.sendFrame({ ear: 0.3, gazeX: 0.5, gazeY: 0.5, headX: 0.5, headY: 0.5, headRollDeg: 0 });
app.flush();
await new Promise((r) => setTimeout(r, 500));
const st = await app.status();
out.eegBufferS = st.eeg_buffer_s; out.eyeFrames = st.eye_frames; out.connections = st.connections; out.qualitySeen = !!seen.quality;
app.startTraining();
app.sendEvent("change"); app.sendEvent("click");
await new Promise((r) => setTimeout(r, 300));
out.modeAfterStart = (await app.status()).mode;
app.finishTraining();
await new Promise((r) => setTimeout(r, 500));
out.trainingSummarySeen = !!seen.training_summary;
out.nowIsServerTime = Math.abs(app.now() - Date.now() / 1000 - app.offset) < 0.05;
console.log("RESULT " + JSON.stringify(out));
app.close(); dev.close();
setTimeout(() => process.exit(0), 200);
