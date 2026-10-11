"""
focus_core.clicks -- metrics from the colour-change click task.

Matching: each colour change is matched to the FIRST unused click 150 ms - 1.5 s later (and before the next change).
Unmatched clicks (anticipations < 150 ms, or clicks with no change) are false alarms.
Metrics: hit_rate, median_rt_ms, rt_sd_ms, lapse_rate = (misses + hits with RT >= 500 ms) / targets, false_alarm_per_min.

LITERATURE
  [PVT2020]  "Response speed measurements on the psychomotor vigilance test: how precise is precise enough?" Sleep (2020), PMID
             32556295 (authors not retrieved). Defines PVT lapses as RT >= 500 ms; a timing bias within +-5 ms (SD <= 10 ms) is the
             suggested maximum -- so stamp clicks and frames on one clock.
  [QbTest]   QbTech, "Attention & impulse control parameters" (commercial documentation, qbtech.com): reaction-time variation is
             described as the most sensitive attention parameter; omissions reflect lapses; responses < 150 ms are anticipatory.
  [gradCPT2013] doi:10.3758/s13414-012-0413-x (gradual-onset continuous performance task): commission errors and RT variability rose
             over a 12-minute vigilance task.
  [SART1998] "Circadian rhythm of sleepiness and vigilance studied in man" (1998, MRC CBU bibliography; authors not retrieved):
             commission errors increased with sleepiness, and fast responses were mostly errors.
  [Shelat2024] Shelat S, Schooler JW, Giesbrecht B (2024). Predicting attentional lapses using response time speed in continuous
             performance tasks. Frontiers in Cognition 3:1460349 -- within-person response-time boundaries have been used to infer
             attentional state (here: the personal baseline plays that role).
NOTE: the 500 ms lapse cut-off is the simple-reaction PVT convention; this task (colour change while tracking) may need its own.
"""
import numpy as np

from . import constants as K


def analyze_clicks(change_t, click_t, duration_s=None, window=K.RESP_WINDOW_S, lapse_ms=K.LAPSE_MS):
    change = np.sort(np.asarray(change_t, float))
    clicks = np.sort(np.asarray(click_t, float))
    used = np.zeros(len(clicks), bool)
    rts = []
    for i, c in enumerate(change):
        end = min(c + window[1], change[i + 1] if i + 1 < len(change) else np.inf)
        cand = np.where((~used) & (clicks >= c + window[0]) & (clicks <= end))[0]
        if len(cand):
            used[cand[0]] = True
            rts.append((clicks[cand[0]] - c) * 1000.0)
    n = len(change)
    rts = np.array(rts)
    dur = duration_s if duration_s else (float(change[-1] + window[1]) if n else 0.0)
    out = dict(n_targets=n, n_hits=len(rts), hit_rate=len(rts) / n if n else float("nan"), false_alarms=int((~used).sum()),
               false_alarm_per_min=float((~used).sum() / (dur / 60.0)) if dur > 0 else float("nan"))
    if len(rts):
        out.update(median_rt_ms=float(np.median(rts)), rt_sd_ms=float(np.std(rts, ddof=1)) if len(rts) >= 3 else float("nan"),
                   lapse_rate=float((n - len(rts) + (rts >= lapse_ms).sum()) / n))
    else:
        out.update(median_rt_ms=float("nan"), rt_sd_ms=float("nan"), lapse_rate=1.0 if n else float("nan"))
    return out
