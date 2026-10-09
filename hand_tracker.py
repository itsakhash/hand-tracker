"""
Hand tracker + air drawing canvas (MediaPipe Tasks HandLandmarker + OpenCV).

Gestures:
  POINT       draw with your index fingertip
  OPEN PALM   move without drawing (a cursor shows where you are)
  PEACE       switch to the next color
  FIST        erase (an eraser circle follows your palm)

Also shown: recognized gesture, per-finger states (T I M R P), finger count,
pinch bar, and FPS.

Keys: q = quit, c = clear drawing, s = save drawing as a transparent PNG
      in ./drawings/, [ and ] = thinner / thicker brush

Speed options:
  --width / --height   capture size (default 640x480; hand tracking does not need HD)
  --profile            print where the time goes (ms per stage) every 2 seconds

Setup:
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python hand_tracker.py

The first run downloads the hand landmark model (about 8 MB) into this folder
as hand_landmarker.task. If the download fails (common SSL issue on macOS),
download it with curl instead (see README). On macOS, allow camera access for
Terminal (or your IDE): System Settings > Privacy & Security > Camera.
If the wrong camera opens, run with --camera 1.
"""
import argparse
import math
import os
import threading
import time
import urllib.request
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
DRAWINGS_DIR = os.path.join(HERE, "drawings")

# MediaPipe landmark indices
WRIST, THUMB_IP, THUMB_TIP = 0, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
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

# Tunable thresholds
THUMB_OUT_RATIO = 0.5    # thumb tip must be this many palm-lengths from the index base
THUMB_UP_MARGIN = 0.3    # thumb tip must be this many palm-lengths above the index base
SMOOTH_FRAMES = 7        # gesture is decided by a vote over this many frames
MIN_VOTES = 4            # votes needed to switch to a new gesture

# Drawing settings
COLORS = [("CYAN", (255, 255, 0)), ("GREEN", (0, 255, 0)), ("MAGENTA", (255, 0, 255)),
          ("YELLOW", (0, 255, 255)), ("RED", (0, 0, 255)), ("WHITE", (255, 255, 255))]
FINGER_SMOOTHING = 0.5   # 0..1, higher = follows the fingertip more tightly (more jitter)
COLOR_COOLDOWN = 1.0     # seconds between color changes
ERASER_RATIO = 0.8       # eraser radius as a fraction of palm length


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
            f"Download it with:  curl -L -o hand_landmarker.task {MODEL_URL}\n"
            "and put it next to hand_tracker.py."
        )
    print("Model saved.")


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_states(pts):
    """Return [thumb, index, middle, ring, pinky] as booleans (True = extended).

    Four fingers: the tip is farther from the wrist than its PIP joint, which
    works for any hand orientation. Thumb: the tip is far enough from the
    base of the index finger (relative to palm size).
    """
    wrist = pts[WRIST]
    palm = max(dist(pts[WRIST], pts[MIDDLE_MCP]), 1e-6)
    thumb = dist(pts[THUMB_TIP], pts[INDEX_MCP]) > THUMB_OUT_RATIO * palm
    others = [dist(pts[tip], wrist) > dist(pts[pip], wrist)
              for tip, pip in [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                               (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]]
    return [thumb] + others


def thumb_points_up(pts):
    """True if the thumb tip is clearly above the base of the index finger."""
    palm = max(dist(pts[WRIST], pts[MIDDLE_MCP]), 1e-6)
    return pts[THUMB_TIP][1] < pts[INDEX_MCP][1] - THUMB_UP_MARGIN * palm


def classify_gesture(pts, states=None):
    """Return a gesture name for one frame of landmarks."""
    if states is None:
        states = finger_states(pts)
    thumb, index, middle, ring, pinky = states
    four = (index, middle, ring, pinky)

    if thumb and not any(four) and thumb_points_up(pts):
        return "THUMBS UP"
    if all(four):
        return "OPEN PALM"
    if not any(four):
        return "FIST"
    if four == (True, False, False, False):
        return "POINT"
    if four == (True, True, False, False):
        return "PEACE"
    if four == (True, False, False, True):
        return "ROCK"
    return "UNKNOWN"


class GestureSmoother:
    """Majority vote over recent frames so the label doesn't flicker."""

    def __init__(self, size=SMOOTH_FRAMES, min_votes=MIN_VOTES):
        self.history = deque(maxlen=size)
        self.min_votes = min_votes
        self.current = "NONE"

    def update(self, gesture):
        self.history.append(gesture)
        winner, votes = Counter(self.history).most_common(1)[0]
        if votes >= self.min_votes:
            self.current = winner
        return self.current

    def reset(self):
        self.history.clear()
        self.current = "NONE"


class CameraStream:
    """Reads the camera on a background thread so the main loop never waits on it.

    The main loop asks for the newest frame it has not seen yet; if processing
    is slower than the camera, older frames are dropped instead of piling up,
    which keeps latency low.
    """

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.running = False
                break
            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def read(self, last_id=-1, timeout=5.0):
        """Wait for a frame newer than last_id. Returns (ok, frame, frame_id)."""
        deadline = time.time() + timeout
        while True:
            with self.lock:
                if self.frame is not None and self.frame_id != last_id:
                    return True, self.frame.copy(), self.frame_id
                ended = not self.running
            if ended or time.time() > deadline:
                return False, None, last_id
            time.sleep(0.001)

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class DrawingCanvas:
    """A persistent drawing layer controlled by hand gestures."""

    def __init__(self, thickness=6):
        self.canvas = None
        self.thickness = thickness
        self.color_idx = 0
        self.prev = None          # smoothed fingertip position while a stroke is active
        self.last_gesture = "NONE"
        self.last_color_change = -1e9
        self.cursor = None        # where to draw the hover cursor
        self.eraser = None        # (center, radius) while erasing

    @property
    def color(self):
        return COLORS[self.color_idx][1]

    @property
    def color_name(self):
        return COLORS[self.color_idx][0]

    def ensure(self, shape):
        if self.canvas is None or self.canvas.shape != shape:
            self.canvas = np.zeros(shape, dtype=np.uint8)
            self.prev = None

    def clear(self):
        if self.canvas is not None:
            self.canvas[:] = 0
        self.prev = None

    def lift(self):
        """Hand left the frame: end the current stroke."""
        self.prev = None
        self.cursor = None
        self.eraser = None
        self.last_gesture = "NONE"

    def change_thickness(self, delta):
        self.thickness = int(np.clip(self.thickness + delta, 2, 30))

    def next_color(self, now):
        if now - self.last_color_change >= COLOR_COOLDOWN:
            self.color_idx = (self.color_idx + 1) % len(COLORS)
            self.last_color_change = now
            return True
        return False

    def update(self, gesture, pts, now):
        """Apply one frame of hand input to the canvas."""
        self.eraser = None
        tip = np.array(pts[INDEX_TIP], dtype=float)

        if gesture == "POINT":
            if self.prev is None:
                self.prev = tip
            else:
                smoothed = FINGER_SMOOTHING * tip + (1 - FINGER_SMOOTHING) * self.prev
                cv2.line(self.canvas, tuple(int(v) for v in self.prev),
                         tuple(int(v) for v in smoothed), self.color,
                         self.thickness, cv2.LINE_AA)
                self.prev = smoothed
            self.cursor = tuple(int(v) for v in self.prev)
        else:
            self.prev = None
            self.cursor = tuple(int(v) for v in tip)
            if gesture == "FIST":
                palm = dist(pts[WRIST], pts[MIDDLE_MCP])
                ids = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
                cx = sum(pts[i][0] for i in ids) / len(ids)
                cy = sum(pts[i][1] for i in ids) / len(ids)
                radius = max(10, int(ERASER_RATIO * palm))
                center = (int(cx), int(cy))
                cv2.circle(self.canvas, center, radius, (0, 0, 0), -1)
                self.eraser = (center, radius)
                self.cursor = None
            elif gesture == "PEACE" and self.last_gesture != "PEACE":
                self.next_color(now)

        self.last_gesture = gesture

    def mask(self):
        """255 where something is drawn, 0 elsewhere (single channel)."""
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    def composite(self, frame):
        """Paint the drawing onto the video frame (in place).

        cv2.copyTo with a mask is several times faster than NumPy boolean
        indexing on full frames, which matters at video frame rates.
        """
        cv2.copyTo(self.canvas, self.mask(), frame)

    def draw_overlays(self, frame):
        """Hover cursor and eraser outline, drawn on the video frame."""
        if self.cursor is not None:
            cv2.circle(frame, self.cursor, max(6, self.thickness), self.color, 2)
        if self.eraser is not None:
            cv2.circle(frame, self.eraser[0], self.eraser[1], (255, 255, 255), 2)

    def save(self):
        """Save the drawing as a transparent PNG and return its path."""
        os.makedirs(DRAWINGS_DIR, exist_ok=True)
        bgra = np.dstack([self.canvas, self.mask()])
        path = os.path.join(DRAWINGS_DIR, time.strftime("drawing_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(path, bgra)
        return path


def draw_hand(frame, pts):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, (int(pts[a][0]), int(pts[a][1])),
                 (int(pts[b][0]), int(pts[b][1])), (255, 255, 255), 2)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--profile", action="store_true",
                    help="print average ms per stage every 2 seconds")
    args = ap.parse_args()

    ensure_model()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Check permissions or try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
          f"driver reports {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
    stream = CameraStream(cap)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    smoother = GestureSmoother()
    drawing = DrawingCanvas()
    pinch_smooth = 0.0
    prev_t = time.time()
    start_t = prev_t
    last_ts = -1
    last_id = -1
    fps = 0.0
    toast = ("", 0.0)  # (message, time it was set)
    timing = {"read": 0.0, "detect": 0.0, "draw": 0.0, "show": 0.0}
    timing_frames = 0
    timing_since = time.time()

    while True:
        t0 = time.perf_counter()
        ok, frame, last_id = stream.read(last_id)
        if not ok:
            print("Camera stopped delivering frames.")
            break
        frame = cv2.flip(frame, 1)  # mirror view feels natural
        h, w = frame.shape[:2]
        drawing.ensure(frame.shape)
        t1 = time.perf_counter()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.time() - start_t) * 1000)
        ts = max(ts, last_ts + 1)  # timestamps must strictly increase
        last_ts = ts
        result = landmarker.detect_for_video(mp_image, ts)
        t2 = time.perf_counter()

        pts = None
        gesture = "NONE"
        if result.hand_landmarks:
            pts = [(p.x * w, p.y * h) for p in result.hand_landmarks[0]]
            states = finger_states(pts)
            gesture = smoother.update(classify_gesture(pts, states))
            drawing.update(gesture, pts, time.time())
        else:
            smoother.reset()
            drawing.lift()

        # Drawing goes under the hand skeleton and the text
        drawing.composite(frame)
        drawing.draw_overlays(frame)

        if pts is not None:
            draw_hand(frame, pts)

            # Pinch: 0 = closed, 1 = fully open, normalized by palm size
            palm = dist(pts[WRIST], pts[MIDDLE_MCP])
            raw = dist(pts[THUMB_TIP], pts[INDEX_TIP]) / max(palm, 1e-6)
            raw = float(np.clip(raw / 1.2, 0.0, 1.0))
            pinch_smooth = 0.7 * pinch_smooth + 0.3 * raw  # exponential smoothing

            cv2.putText(frame, gesture, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)
            cv2.putText(frame, f"Fingers: {sum(states)}", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            debug = " ".join(c if s else "-" for c, s in zip("TIMRP", states))
            cv2.putText(frame, debug, (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.rectangle(frame, (20, 160), (220, 185), (255, 255, 255), 2)
            cv2.rectangle(frame, (20, 160), (20 + int(200 * pinch_smooth), 185),
                          (0, 200, 255), -1)
            cv2.putText(frame, "pinch", (230, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No hand detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Current color swatch + brush size
        cv2.rectangle(frame, (w - 190, 60), (w - 150, 100), drawing.color, -1)
        cv2.rectangle(frame, (w - 190, 60), (w - 150, 100), (255, 255, 255), 2)
        cv2.putText(frame, f"{drawing.color_name}  {drawing.thickness}px", (w - 140, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Help line and short messages
        cv2.putText(frame, "POINT draw | PEACE color | FIST erase | PALM move | "
                           "c clear | s save | [ ] size | q quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if toast[0] and time.time() - toast[1] < 2.0:
            cv2.putText(frame, toast[0], (20, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
        prev_t = now
        cv2.putText(frame, f"{fps:.0f} FPS", (w - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        t3 = time.perf_counter()

        cv2.imshow("Hand tracker (q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        t4 = time.perf_counter()

        if args.profile:
            timing["read"] += t1 - t0
            timing["detect"] += t2 - t1
            timing["draw"] += t3 - t2
            timing["show"] += t4 - t3
            timing_frames += 1
            if time.time() - timing_since >= 2.0:
                n = max(timing_frames, 1)
                print("ms/frame  " + "  ".join(f"{k}={v / n * 1000:.1f}" for k, v in timing.items())
                      + f"  ({timing_frames / (time.time() - timing_since):.1f} fps)")
                timing = {k: 0.0 for k in timing}
                timing_frames = 0
                timing_since = time.time()

        if key == ord("q"):
            break
        elif key == ord("c"):
            drawing.clear()
        elif key == ord("s"):
            path = drawing.save()
            toast = (f"Saved {os.path.basename(path)}", time.time())
            print("Saved", path)
        elif key == ord("["):
            drawing.change_thickness(-2)
        elif key == ord("]"):
            drawing.change_thickness(2)

    landmarker.close()
    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
