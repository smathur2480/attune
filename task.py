"""
focus_core.task -- the training task geometry and schedule (shared by the app, the simulator and the scorers).

A ball orbits the screen centre on a circle (x = 0.5 + r cos(2 pi t / T), y = 0.5 + r sin(2 pi t / T), t = 0 at task start) and
changes colour at random times (gaps 3-9 s). The user follows it with their eyes and clicks at each colour change.
These design values are choices, NOT validated settings. A 3-minute PVT-style task stays sensitive to sleep loss (documented in
clinical-trial protocols using a 3-minute PVT); the orbit/pursuit design follows the tracking-task literature:
  [Peiris2006] Peiris et al. (2006) J Sleep Res 15:291-300 -- lapses of responsiveness occur during extended visuomotor tracking
               even in non-sleep-deprived people (continuous tracking tasks expose lapses that discrete tasks can miss).
"""
import numpy as np
import pandas as pd

from . import constants as K


def make_schedule(duration_s=180.0, seed=0, first=4.0, gap=(3.0, 9.0)):
    rng = np.random.default_rng(seed)
    t, out = first, []
    while t < duration_s - 2.0:
        out.append(round(float(t), 3))
        t += rng.uniform(*gap)
    return out


def orbit(t, period=K.ORBIT_PERIOD_S, radius=K.ORBIT_RADIUS, center=(0.5, 0.5), phase=0.0):
    ph = 2 * np.pi * np.asarray(t, float) / period + phase
    return center[0] + radius * np.cos(ph), center[1] + radius * np.sin(ph)


def ball_frame(t, period=K.ORBIT_PERIOD_S, radius=K.ORBIT_RADIUS):
    bx, by = orbit(t, period, radius)
    return pd.DataFrame(dict(t=np.asarray(t, float), bx=bx, by=by))
