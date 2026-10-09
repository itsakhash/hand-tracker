"""
Hand tracker using your webcam (MediaPipe Tasks HandLandmarker + OpenCV).

Shows:
  - the 21 hand landmarks and bones drawn on the live video
  - finger count (0-5)
  - pinch amount (thumb tip to index tip, normalized by hand size) as a bar
  - a fading trail behind your index fingertip (air drawing)
  - FPS

Keys: q = quit, c = clear trail

Setup:
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python hand_tracker.py

The first run downloads the hand landmark model (about 8 MB) into this folder
as hand_landmarker.task. On macOS, allow camera access for Terminal (or your
IDE) when prompted: System Settings > Privacy & Security > Camera.
If the wrong camera opens, run with --camera 1.
"""
import argparse
import math
import os
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

# MediaPipe landmark indices
WRIST, THUMB_IP, THUMB_TIP = 0, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# Which landmarks connect to which (the hand "skeleton")
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm edge
]


def ensure_model():
    """Download the model file the first time the script runs."""
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand landmark model (one time)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise SystemExit(
            f"Could not download the model: {e}\n"
            "Download hand_landmarker.task manually from the MediaPipe Hand Landmarker "
            "page (Models section) and put it next to hand_tracker.py."
        )
    print("Model saved.")


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def count_fingers(pts):
    """Count extended fingers from pixel landmarks (hand roughly upright)."""
    count = 0
    # Four fingers: tip is above its PIP joint (smaller y) when extended
    for tip, pip in [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                     (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]:
        if pts[tip][1] < pts[pip][1]:
            count += 1
    # Thumb: extended if the tip is farther from the pinky base than the IP joint is
    if dist(pts[THUMB_TIP], pts[PINKY_MCP]) > dist(pts[THUMB_IP], pts[PINKY_MCP]):
        count += 1
    return count


def draw_hand(frame, pts):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, (int(pts[a][0]), int(pts[a][1])),
                 (int(pts[b][0]), int(pts[b][1])), (255, 255, 255), 2)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=1)
    args = ap.parse_args()

    ensure_model()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Check permissions or try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    trail = deque(maxlen=40)
    pinch_smooth = 0.0
    prev_t = time.time()
    start_t = prev_t
    last_ts = -1
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view feels natural
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - start_t) * 1000)
        ts = max(ts, last_ts + 1)  # timestamps must strictly increase
        last_ts = ts
        result = landmarker.detect_for_video(mp_image, ts)

        if result.hand_landmarks:
            pts = [(p.x * w, p.y * h) for p in result.hand_landmarks[0]]
            draw_hand(frame, pts)

            fingers = count_fingers(pts)

            # Pinch: 0 = closed, 1 = fully open, normalized by palm size
            palm = dist(pts[WRIST], pts[MIDDLE_MCP])
            raw = dist(pts[THUMB_TIP], pts[INDEX_TIP]) / max(palm, 1e-6)
            raw = float(np.clip(raw / 1.2, 0.0, 1.0))
            pinch_smooth = 0.7 * pinch_smooth + 0.3 * raw  # exponential smoothing

            tip = (int(pts[INDEX_TIP][0]), int(pts[INDEX_TIP][1]))
            trail.append(tip)

            cv2.putText(frame, f"Fingers: {fingers}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.rectangle(frame, (20, 80), (220, 105), (255, 255, 255), 2)
            cv2.rectangle(frame, (20, 80), (20 + int(200 * pinch_smooth), 105),
                          (0, 200, 255), -1)
            cv2.putText(frame, "pinch", (230, 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            trail.clear()
            cv2.putText(frame, "No hand detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Fading trail behind the fingertip
        for i in range(1, len(trail)):
            thickness = max(1, int(8 * i / len(trail)))
            cv2.line(frame, trail[i - 1], trail[i], (255, 120, 0), thickness)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
        prev_t = now
        cv2.putText(frame, f"{fps:.0f} FPS", (w - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Hand tracker (q to quit, c to clear)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            trail.clear()

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
