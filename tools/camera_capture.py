#!/usr/bin/env python3
"""
Camera -> eye-signal CSV for focus_patterns.py  (webcam or video file; runs locally, no frames are stored or uploaded).

  python camera_capture.py --seconds 60 --out eye.csv                 # default webcam, 60 s
  python camera_capture.py --video clip.mp4 --out eye.csv             # an existing video

Output columns:  t (s), ear (eye aspect ratio, mean of both eyes), gaze_x, gaze_y (EYE-IN-HEAD iris ratios, 0-1, NOT screen
coordinates), face_ok (0/1), and head pose for signal_quality.py: head_x, head_y (nose tip, 0-1 image coordinates),
head_roll_deg (tilt of the line between the outer eye corners) and head_yaw (nose offset from the eye midpoint, in eye-distance units).
  * ear: open eyes are typically ~0.25-0.35, a blink drops it toward 0.05-0.1 (6-point EAR formula on FaceMesh landmarks).
  * gaze_x/gaze_y: iris position inside the eye opening (0.5 = centred). Head movement is NOT removed, so treat these as a
    coarse "where am I looking" signal. In focus_patterns.py use  calibrate --roi-from-gaze  to learn the on-task region from a
    calibration clip in which the user looks at the screen.

TESTED HERE: landmark indices and the EAR / gaze maths on a real portrait photo, and the video pipeline on a still-image
video. NOT tested here: real blinks/eye movements on live video, different lighting, glasses, phone cameras, frame-rate drift.
Test it with a short webcam clip of yourself blinking and check that ear dips at each blink. Needs:
  python -m pip install "mediapipe==0.10.14" opencv-python-headless numpy pandas
(mediapipe >= 0.10.15 removed the 'solutions' API this script uses.)
"""
import argparse
import sys
import time

try:
    import numpy as np
    import pandas as pd
except ImportError as _e:
    raise SystemExit(f"Missing Python package: {_e.name}.  python -m pip install numpy pandas")

# FaceMesh landmark indices (478 points with refine_landmarks=True). Verified on a portrait: 468 lies between 33 and 133
# (the subject's right eye, image left), 473 between 362 and 263 (the subject's left eye, image right).
R_EAR, L_EAR = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]
R_CORNERS, L_CORNERS = (33, 133), (362, 263)
R_LIDS, L_LIDS = (159, 145), (386, 374)             # upper, lower
R_IRIS, L_IRIS = 468, 473
NOSE_TIP, R_OUTER, L_OUTER = 1, 33, 263


def _pt(lm, i, w, h):
    return np.array([lm[i].x * w, lm[i].y * h])


def ear_of(lm, idx, w, h):
    p1, p2, p3, p4, p5, p6 = [_pt(lm, i, w, h) for i in idx]
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4) + 1e-9)


def landmarks_to_row(lm, w, h):
    """lm: sequence of landmarks with .x/.y in 0-1 image coordinates. Returns dict(ear, gaze_x, gaze_y)."""
    ears = [ear_of(lm, R_EAR, w, h), ear_of(lm, L_EAR, w, h)]
    gx, gy = [], []
    for corners, lids, iris in ((R_CORNERS, R_LIDS, R_IRIS), (L_CORNERS, L_LIDS, L_IRIS)):
        c0, c1 = _pt(lm, corners[0], w, h), _pt(lm, corners[1], w, h)
        up, lo, ir = _pt(lm, lids[0], w, h), _pt(lm, lids[1], w, h), _pt(lm, iris, w, h)
        xmin, xmax = min(c0[0], c1[0]), max(c0[0], c1[0])
        gx.append(float(np.clip((ir[0] - xmin) / (xmax - xmin + 1e-9), 0, 1)))
        gy.append(float(np.clip((ir[1] - up[1]) / (lo[1] - up[1] + 1e-9), 0, 1)))
    ear = float(np.mean(ears))
    blinking = ear < 0.15                                    # iris is hidden / unreliable during a blink
    ro, lo, nose = _pt(lm, R_OUTER, w, h), _pt(lm, L_OUTER, w, h), _pt(lm, NOSE_TIP, w, h)
    eye_dist = float(np.linalg.norm(lo - ro)) + 1e-9
    return dict(ear=ear, gaze_x=np.nan if blinking else float(np.mean(gx)), gaze_y=np.nan if blinking else float(np.mean(gy)),
                head_x=float(lm[NOSE_TIP].x), head_y=float(lm[NOSE_TIP].y),
                head_roll_deg=float(np.degrees(np.arctan2(lo[1] - ro[1], lo[0] - ro[0]))),
                head_yaw=float((nose[0] - (ro[0] + lo[0]) / 2) / eye_dist))


def run(source=0, video=None, seconds=60.0, out="eye.csv"):
    try:
        import cv2
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            raise ImportError("mediapipe has no 'solutions' API; install mediapipe==0.10.14")
    except ImportError as e:
        raise SystemExit(f"{e}\nInstall with: python -m pip install \"mediapipe==0.10.14\" opencv-python-headless")
    cap = cv2.VideoCapture(video if video else source)
    if not cap.isOpened():
        raise SystemExit("could not open the camera / video")
    fps_file = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                         min_detection_confidence=0.5, min_tracking_confidence=0.5)
    rows, i, t0 = [], 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = (i / fps_file) if video else (time.time() - t0)
        if t > seconds:
            break
        h, w = frame.shape[:2]
        res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            r = landmarks_to_row(res.multi_face_landmarks[0].landmark, w, h)
            rows.append(dict(t=t, face_ok=1, **r))
        else:
            rows.append(dict(t=t, ear=np.nan, gaze_x=np.nan, gaze_y=np.nan, face_ok=0,
                             head_x=np.nan, head_y=np.nan, head_roll_deg=np.nan, head_yaw=np.nan))
        i += 1
    cap.release()
    df = pd.DataFrame(rows, columns=["t", "ear", "gaze_x", "gaze_y", "face_ok", "head_x", "head_y", "head_roll_deg", "head_yaw"])
    df.to_csv(out, index=False)
    dur = df.t.iloc[-1] - df.t.iloc[0] if len(df) > 1 else 0.0
    print(f"wrote {out}: {len(df)} frames over {dur:.1f} s (~{len(df) / max(dur, 1e-9):.1f} fps), "
          f"face found in {df.face_ok.mean():.0%} of frames, median EAR {df.ear.median():.3f}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=int, default=0, help="webcam index")
    ap.add_argument("--video", help="video file instead of the webcam")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", default="eye.csv")
    a = ap.parse_args()
    run(a.source, a.video, a.seconds, a.out)
