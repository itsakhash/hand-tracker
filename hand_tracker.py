"""
Hand tracker + air drawing + virtual mouse (MediaPipe Tasks HandLandmarker + OpenCV).

DRAW mode (default):
  POINT       draw with your index fingertip
  OPEN PALM   move without drawing (a cursor shows where you are)
  PEACE       switch to the next color
  FIST        erase (an eraser circle follows your palm)

MOUSE mode (press m, or hold THUMBS UP for 1 second):
  move hand                 moves the real mouse cursor (inside the green box)
  pinch thumb + index       left click; hold the pinch and move to drag
  pinch thumb + middle      right click
  PEACE + move up/down      scroll
  FIST                      pause the cursor
  hold THUMBS UP (1 s)      leave mouse mode
  Emergency stop: slam your real mouse into the top-left screen corner.

Keys: q = quit, m = toggle mouse mode, c = clear drawing, s = save drawing as a
      transparent PNG in ./drawings/, [ and ] = thinner / thicker brush

Options:
  --camera N           which camera to use (1 if an iPhone grabs index 0)
  --width / --height   capture size (default 640x480)
  --profile            print where the time goes (ms per stage) every 2 seconds
  --cursor-lag SEC     how quickly the real cursor chases your hand (default 0.04;
                       lower = snappier but less smooth, higher = smoother but laggier)
  --direct-cursor      move the cursor straight from each camera frame (old behavior)
  --still-radius PX    cursor ignores wobbles smaller than this many screen pixels
                       (default 8; raise it if the cursor shakes, 0 turns it off)
  --click-lookback SEC clicks land where the cursor was this long before your fingers
                       closed (default 0.2; raise it if clicks land past the target)

Mouse mode needs:  pip install pyautogui
On macOS also allow your terminal (Terminal, iTerm, VS Code...) under
System Settings > Privacy & Security > Accessibility, then restart the terminal.

Setup:
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python hand_tracker.py

The first run downloads the hand landmark model (about 8 MB) into this folder
as hand_landmarker.task. If the download fails (common SSL issue on macOS),
download it with curl instead (see README).
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

# Gesture thresholds
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

# Mouse settings (all in "palm lengths" or fractions, so distance from the camera doesn't matter)
ACTIVE_MARGIN_X = 0.2    # the part of the frame (each side) NOT used for pointing
ACTIVE_MARGIN_Y = 0.2
SCREEN_EDGE_PAD = 5      # px kept away from screen corners (the top-left corner is the failsafe)
PINCH_ON = 0.25          # thumb-to-fingertip distance / palm length that counts as a pinch
PINCH_OFF = 0.40         # must open past this to release (hysteresis stops flicker)
PINCH_FRAMES = 2         # consecutive pinched frames required before a click starts
PINCH_MIN_REACH = 1.0    # index tip must be this far from the wrist (palm lengths), so a fist is not a click
RIGHT_CLICK_COOLDOWN = 0.6
ONE_EURO_MIN_CUTOFF = 1.0  # Hz: lower = steadier cursor when still (but more lag)
ONE_EURO_BETA = 12.0       # higher = less lag when moving fast
ONE_EURO_D_CUTOFF = 1.0    # Hz: smoothing of the speed estimate
CURSOR_FOLLOW_TAU = 0.04   # seconds: how quickly the real cursor chases its target
CURSOR_RATE_HZ = 120       # how often the cursor thread updates the real cursor
CURSOR_DEADZONE_PX = 2   # ignore cursor moves smaller than this
STILL_RADIUS_PX = 8      # cursor ignores wobbles smaller than this (screen pixels)
PINCH_LOOKBACK = 0.20    # a click aims where the cursor was this many seconds earlier
LOOKBACK_MAX_PX = 80     # ...but only if the cursor has not travelled farther than this since
DRAG_START_PX = 14       # during a click the cursor stays locked until you move this far
RELEASE_FREEZE = 0.15    # seconds the cursor stays put after you let go of a click
RIGHT_CLICK_FREEZE = 0.4 # seconds the cursor stays put after a right click
SCROLL_GAIN = 8.0        # scroll clicks per palm length of vertical hand movement
SCROLL_DIRECTION = 1     # set to -1 if scrolling feels backwards
MODE_TOGGLE_HOLD = 1.0   # seconds of THUMBS UP to switch mouse mode on/off


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
        """Hand left the frame (or another mode took over): end the current stroke."""
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


# --------------------------------------------------------------------------
# Virtual mouse
# --------------------------------------------------------------------------
def map_to_screen(point, frame_w, frame_h, screen_w, screen_h,
                  margin_x=ACTIVE_MARGIN_X, margin_y=ACTIVE_MARGIN_Y):
    """Map a point in the camera frame to normalized (u, v) in 0..1 across the active box.

    Only the middle part of the frame is used, so you can reach every screen
    edge without your hand leaving the camera view.
    """
    x0, x1 = margin_x * frame_w, (1 - margin_x) * frame_w
    y0, y1 = margin_y * frame_h, (1 - margin_y) * frame_h
    u = (point[0] - x0) / max(x1 - x0, 1e-6)
    v = (point[1] - y0) / max(y1 - y0, 1e-6)
    return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))


def to_pixels(u, v, screen_w, screen_h, pad=SCREEN_EDGE_PAD):
    """Normalized (u, v) to screen pixels, kept away from the corners."""
    x = pad + u * (screen_w - 1 - 2 * pad)
    y = pad + v * (screen_h - 1 - 2 * pad)
    return int(round(x)), int(round(y))


class OneEuroFilter2D:
    """The One Euro filter for a 2D pointer.

    A standard way to smooth noisy tracking: it removes jitter when the hand is
    nearly still and lets fast movements through with little lag. It uses real
    timestamps, so it behaves the same at 15 fps and at 30 fps.
    """

    def __init__(self, min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA,
                 d_cutoff=ONE_EURO_D_CUTOFF):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.t = None
        self.raw = None
        self.x = None
        self.dx = (0.0, 0.0)

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, u, v, t):
        if self.x is None or self.t is None:
            self.t, self.raw, self.x, self.dx = t, (u, v), (u, v), (0.0, 0.0)
            return self.x
        dt = max(t - self.t, 1e-3)
        a_d = self._alpha(self.d_cutoff, dt)
        raw_dx = ((u - self.raw[0]) / dt, (v - self.raw[1]) / dt)
        self.dx = (self.dx[0] + a_d * (raw_dx[0] - self.dx[0]),
                   self.dx[1] + a_d * (raw_dx[1] - self.dx[1]))
        speed = math.hypot(*self.dx)
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self.x = (self.x[0] + a * (u - self.x[0]), self.x[1] + a * (v - self.x[1]))
        self.t, self.raw = t, (u, v)
        return self.x


class PinchDetector:
    """Turns a changing distance ratio into clean 'down' / 'up' events.

    Hysteresis (different on/off thresholds) and a frame count stop a noisy
    measurement from producing accidental clicks.
    """

    def __init__(self, on=PINCH_ON, off=PINCH_OFF, frames=PINCH_FRAMES):
        self.on, self.off, self.frames = on, off, frames
        self.active = False
        self.count = 0

    @property
    def engaged(self):
        """True while a pinch is held or just starting."""
        return self.active or self.count > 0

    def reset(self):
        self.active = False
        self.count = 0

    def update(self, ratio, valid=True):
        if self.active:
            if ratio > self.off or not valid:
                self.active = False
                self.count = 0
                return "up"
            return None
        if valid and ratio < self.on:
            self.count += 1
            if self.count >= self.frames:
                self.active = True
                self.count = 0
                return "down"
        else:
            self.count = 0
        return None


class HoldDetector:
    """Fires once when a gesture has been held long enough; re-arms when it changes."""

    def __init__(self, target="THUMBS UP", hold=MODE_TOGGLE_HOLD):
        self.target, self.hold = target, hold
        self.start = None
        self.fired = False

    def reset(self):
        self.start = None
        self.fired = False

    def update(self, gesture, now):
        if gesture != self.target:
            self.reset()
            return False
        if self.start is None:
            self.start = now
        if not self.fired and now - self.start >= self.hold:
            self.fired = True
            return True
        return False

    def progress(self, now):
        """0..1 while the gesture is being held (for an on-screen bar)."""
        if self.start is None or self.fired:
            return 0.0
        return min((now - self.start) / self.hold, 1.0)


class PyAutoGuiBackend:
    """The real mouse, through pyautogui (imported only when mouse mode is first used)."""

    def __init__(self):
        try:
            import pyautogui
        except ImportError:
            raise RuntimeError("pyautogui is not installed. Run: pip install pyautogui")
        pyautogui.FAILSAFE = True   # slam the mouse into the top-left corner to stop
        pyautogui.PAUSE = 0         # no built-in delay after each call
        self.pg = pyautogui

    def size(self):
        s = self.pg.size()
        return int(s[0]), int(s[1])

    def move_to(self, x, y):
        self.pg.moveTo(x, y)

    def mouse_down(self):
        self.pg.mouseDown()

    def mouse_up(self):
        # Releasing a button must always work, even with the cursor in the failsafe corner,
        # otherwise the button could stay stuck down.
        old = self.pg.FAILSAFE
        self.pg.FAILSAFE = False
        try:
            self.pg.mouseUp()
        finally:
            self.pg.FAILSAFE = old

    def click(self, button="left"):
        self.pg.click(button=button)

    def scroll(self, clicks):
        self.pg.scroll(clicks)


class StillnessLeash:
    """Ignores small wobbles so the cursor holds still when your hand does.

    The output only moves once the input leaves a small circle around it, and
    then it is pulled along at the edge of that circle. So there is no jump when
    you start moving, and no shaking while you hold still.
    """

    def __init__(self, radius=STILL_RADIUS_PX):
        self.radius = radius
        self.anchor = None

    def reset(self):
        self.anchor = None

    def set(self, px):
        self.anchor = (float(px[0]), float(px[1]))

    def update(self, px):
        if self.anchor is None or self.radius <= 0:
            self.anchor = (float(px[0]), float(px[1]))
        else:
            dx, dy = px[0] - self.anchor[0], px[1] - self.anchor[1]
            d = math.hypot(dx, dy)
            if d > self.radius:
                k = (d - self.radius) / d
                self.anchor = (self.anchor[0] + dx * k, self.anchor[1] + dy * k)
        return int(round(self.anchor[0])), int(round(self.anchor[1]))


class CursorDriver:
    """Moves the real cursor smoothly between camera frames.

    The camera only reports a new hand position 15-30 times a second, which
    makes a cursor moved straight from those positions look choppy. This runs
    on its own thread at ~120 Hz and glides the cursor toward the newest target
    (an exponential chase), so motion looks continuous.
    """

    def __init__(self, backend, tau=CURSOR_FOLLOW_TAU, rate_hz=CURSOR_RATE_HZ):
        self.backend = backend
        self.tau = max(tau, 1e-3)
        self.period = 1.0 / rate_hz
        self.lock = threading.Lock()
        self.target = None
        self.pos = None
        self.last_sent = None
        self.running = False
        self.thread = None
        self.error = None

    def reset(self):
        with self.lock:
            self.target = None
            self.pos = None
            self.last_sent = None
        self.error = None

    def set_target(self, px):
        with self.lock:
            self.target = (float(px[0]), float(px[1]))

    def step(self, dt):
        """Advance the cursor by dt seconds. Returns the new pixel position if it moved."""
        with self.lock:
            if self.target is None:
                return None
            if self.pos is None:
                self.pos = self.target
            else:
                a = 1.0 - math.exp(-dt / self.tau)
                self.pos = (self.pos[0] + a * (self.target[0] - self.pos[0]),
                            self.pos[1] + a * (self.target[1] - self.pos[1]))
            out = (int(round(self.pos[0])), int(round(self.pos[1])))
            if out == self.last_sent:
                return None
            self.last_sent = out
            self.backend.move_to(*out)
        return out

    def snap(self, px):
        """Put the cursor exactly on px right now (used just before a click)."""
        with self.lock:
            self.target = (float(px[0]), float(px[1]))
            self.pos = self.target
            out = (int(round(px[0])), int(round(px[1])))
            self.last_sent = out
            self.backend.move_to(*out)
        return out

    def _run(self):
        prev = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            dt, prev = now - prev, now
            try:
                self.step(dt)
            except Exception as e:      # for example pyautogui's failsafe corner
                self.error = e
                self.running = False
                return
            time.sleep(self.period)

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.error = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)


class MouseController:
    """Turns hand landmarks into mouse actions on a backend (real or fake)."""

    def __init__(self, backend, driver=None, still_radius=STILL_RADIUS_PX, lookback=PINCH_LOOKBACK):
        self.backend = backend
        self.driver = driver      # if given, its thread does the actual cursor moving
        self.lookback = lookback
        self.screen_w, self.screen_h = backend.size()
        self.smoother = OneEuroFilter2D()
        self.leash = StillnessLeash(still_radius)
        self.left = PinchDetector()
        self.right = PinchDetector()
        self.left_down = False
        self.last_right_click = -1e9
        self.last_px = None
        self.history = deque()    # (time, px) of recent cursor targets
        self.click_lock = None    # where the current click was aimed; held until a drag starts
        self.click_ref = None     # where the pointer was when the pinch closed (drag is measured from here)
        self.freeze_until = 0.0
        self.prev_scroll_y = None
        self.scroll_accum = 0.0

    def _forget_motion(self):
        self.smoother.reset()
        self.leash.reset()
        self.last_px = None
        self.history.clear()
        self.click_lock = None
        self.click_ref = None
        self.freeze_until = 0.0
        self.prev_scroll_y = None
        self.scroll_accum = 0.0

    def reset(self):
        self.release_all()
        self.left.reset()
        self.right.reset()
        self._forget_motion()

    def release_all(self):
        """Never leave the mouse button held down."""
        if self.left_down:
            self.backend.mouse_up()
            self.left_down = False

    def hand_lost(self):
        self.release_all()
        self.left.reset()
        self.right.reset()
        self._forget_motion()

    def _lookback_point(self, now):
        """Where the cursor was aimed `lookback` seconds ago, before your fingers started closing."""
        if not self.history:
            return None
        wanted = now - self.lookback
        point = self.history[0][1]
        for t, px in self.history:
            if t <= wanted:
                point = px
            else:
                break
        if dist(point, self.history[-1][1]) > LOOKBACK_MAX_PX:
            return None           # the hand was moving fast: the current spot is the intended one
        return point

    def _aim(self, point):
        """Put the cursor exactly on `point` right now."""
        if point is None:
            return
        if self.driver is not None:
            self.driver.snap(point)
        elif point != self.last_px:
            self.backend.move_to(*point)
        self.last_px = point
        self.leash.set(point)

    def _send(self, px):
        if self.driver is not None:
            self.driver.set_target(px)
            self.last_px = px
        elif (self.last_px is None or abs(px[0] - self.last_px[0]) >= CURSOR_DEADZONE_PX
                or abs(px[1] - self.last_px[1]) >= CURSOR_DEADZONE_PX):
            self.backend.move_to(*px)
            self.last_px = px

    def update(self, gesture, pts, frame_w, frame_h, now):
        palm = max(dist(pts[WRIST], pts[MIDDLE_MCP]), 1e-6)
        reach_ok = dist(pts[INDEX_TIP], pts[WRIST]) > PINCH_MIN_REACH * palm
        index_ratio = dist(pts[THUMB_TIP], pts[INDEX_TIP]) / palm
        middle_ratio = dist(pts[THUMB_TIP], pts[MIDDLE_TIP]) / palm

        # Pointer position: midpoint of thumb and index tips, smoothed every frame
        mid = ((pts[THUMB_TIP][0] + pts[INDEX_TIP][0]) / 2,
               (pts[THUMB_TIP][1] + pts[INDEX_TIP][1]) / 2)
        u, v = map_to_screen(mid, frame_w, frame_h, self.screen_w, self.screen_h)
        inst_px = to_pixels(u, v, self.screen_w, self.screen_h)      # unsmoothed: used to detect real movement
        u, v = self.smoother.update(u, v, now)
        raw_px = to_pixels(u, v, self.screen_w, self.screen_h)       # smoothed: where the cursor should go

        # Left button: pinch down / up (hold and move to drag)
        event = self.left.update(index_ratio, reach_ok)
        if event == "down" and not self.left_down:
            # Closing your fingers nudges the pointer, so aim where it was just before.
            point = self._lookback_point(now)
            if point is None:
                point = self.last_px
            if point is not None:
                self._aim(point)
                self.click_lock = point
                self.click_ref = inst_px
            self.backend.mouse_down()
            self.left_down = True
        elif event == "up":
            self.release_all()
            if self.click_lock is not None:      # a plain click, not a drag
                self.freeze_until = now + RELEASE_FREEZE
                self.click_lock = None

        # Right click: thumb + middle finger, once per pinch
        if not self.left.active:
            event = self.right.update(middle_ratio, reach_ok)
            if event == "down" and now - self.last_right_click >= RIGHT_CLICK_COOLDOWN:
                self._aim(self._lookback_point(now))
                self.backend.click("right")
                self.last_right_click = now
                self.freeze_until = now + RIGHT_CLICK_FREEZE
        else:
            self.right.reset()

        # Scroll: PEACE sign, move hand up or down
        if gesture == "PEACE" and not self.left.active:
            y = (pts[INDEX_TIP][1] + pts[MIDDLE_TIP][1]) / 2
            if self.prev_scroll_y is not None:
                self.scroll_accum += (self.prev_scroll_y - y) / palm * SCROLL_GAIN * SCROLL_DIRECTION
                clicks = int(self.scroll_accum)
                if clicks != 0:
                    self.backend.scroll(clicks)
                    self.scroll_accum -= clicks
            self.prev_scroll_y = y
        else:
            self.prev_scroll_y = None
            self.scroll_accum = 0.0

        # Move the cursor. Frozen during FIST (pause), PEACE (scroll), a forming
        # right-click pinch, and briefly after a click.
        moving = ((gesture not in ("FIST", "PEACE") or self.left.active)
                  and not self.right.engaged and now >= self.freeze_until)
        if moving:
            # While a click is in progress the cursor stays on its target until your
            # hand clearly moves away from where it was when the pinch closed. (Compared
            # unsmoothed, so the smoothing filter catching up is not mistaken for a drag.)
            if self.click_lock is not None and dist(inst_px, self.click_ref) > DRAG_START_PX:
                self.click_lock = None
            if self.click_lock is None:
                px = self.leash.update(raw_px)
                self.history.append((now, px))
                while self.history and now - self.history[0][0] > 1.0:
                    self.history.popleft()
                self._send(px)


class MouseMode:
    """Owns the on/off state of mouse mode and creates the backend on first use."""

    def __init__(self, backend_factory=PyAutoGuiBackend, use_driver=False, tau=CURSOR_FOLLOW_TAU,
                 still_radius=STILL_RADIUS_PX, lookback=PINCH_LOOKBACK):
        self.factory = backend_factory
        self.use_driver = use_driver
        self.tau = tau
        self.still_radius = still_radius
        self.lookback = lookback
        self.controller = None
        self.driver = None
        self.on = False

    def enable(self):
        """Returns (ok, message)."""
        if self.controller is None:
            try:
                backend = self.factory()
                self.driver = CursorDriver(backend, tau=self.tau) if self.use_driver else None
                self.controller = MouseController(backend, self.driver,
                                                  still_radius=self.still_radius,
                                                  lookback=self.lookback)
            except Exception as e:
                return False, str(e)
        self.controller.reset()
        if self.driver is not None:
            self.driver.reset()
            self.driver.start()
        self.on = True
        return True, "Mouse mode ON"

    def disable(self, message="Mouse mode OFF"):
        if self.driver is not None:
            self.driver.stop()
        if self.controller is not None:
            try:
                self.controller.release_all()
            except Exception:
                self.controller.left_down = False  # nothing more we can do; don't crash
        self.on = False
        return message

    def toggle(self):
        if self.on:
            return True, self.disable()
        return self.enable()

    def update(self, gesture, pts, frame_w, frame_h, now):
        """Run one frame. Any backend error (failsafe corner, permissions) turns the mode off."""
        if self.driver is not None and self.driver.error is not None:
            return self.disable(f"Mouse mode OFF ({type(self.driver.error).__name__})")
        try:
            self.controller.update(gesture, pts, frame_w, frame_h, now)
        except Exception as e:
            return self.disable(f"Mouse mode OFF ({type(e).__name__})")
        return None

    def hand_lost(self):
        if self.on and self.controller is not None:
            self.controller.hand_lost()


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
    ap.add_argument("--cursor-lag", type=float, default=CURSOR_FOLLOW_TAU,
                    help="seconds the real cursor takes to chase your hand (lower = snappier)")
    ap.add_argument("--direct-cursor", action="store_true",
                    help="move the cursor straight from each camera frame (old behavior)")
    ap.add_argument("--still-radius", type=float, default=STILL_RADIUS_PX,
                    help="ignore cursor wobbles smaller than this many screen pixels (0 = off)")
    ap.add_argument("--click-lookback", type=float, default=PINCH_LOOKBACK,
                    help="clicks land where the cursor was this many seconds before the pinch")
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
    mouse = MouseMode(use_driver=not args.direct_cursor, tau=args.cursor_lag,
                      still_radius=args.still_radius, lookback=args.click_lookback)
    hold = HoldDetector()
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

    def show_toast(message):
        nonlocal toast
        toast = (message, time.time())
        print(message)

    def toggle_mouse():
        ok, message = mouse.toggle()
        show_toast(message)
        drawing.lift()

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
        now_t = time.time()
        if result.hand_landmarks:
            pts = [(p.x * w, p.y * h) for p in result.hand_landmarks[0]]
            states = finger_states(pts)
            gesture = smoother.update(classify_gesture(pts, states))

            if mouse.on:
                drawing.lift()
                message = mouse.update(gesture, pts, w, h, now_t)
                if message:
                    show_toast(message)
            else:
                drawing.update(gesture, pts, now_t)

            if hold.update(gesture, now_t):
                toggle_mouse()
        else:
            smoother.reset()
            drawing.lift()
            mouse.hand_lost()
            hold.reset()

        # Drawing goes under the hand skeleton and the text
        drawing.composite(frame)
        drawing.draw_overlays(frame)

        if mouse.on:
            x0, x1 = int(ACTIVE_MARGIN_X * w), int((1 - ACTIVE_MARGIN_X) * w)
            y0, y1 = int(ACTIVE_MARGIN_Y * h), int((1 - ACTIVE_MARGIN_Y) * h)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

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

            # Progress bar while holding THUMBS UP to switch modes
            progress = hold.progress(now_t)
            if progress > 0:
                cv2.rectangle(frame, (20, 195), (220, 215), (255, 255, 255), 2)
                cv2.rectangle(frame, (20, 195), (20 + int(200 * progress), 215),
                              (0, 255, 0), -1)
        else:
            cv2.putText(frame, "No hand detected", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Mode banner, color swatch and brush size
        if mouse.on:
            cv2.putText(frame, "MOUSE MODE", (w - 230, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        else:
            cv2.rectangle(frame, (w - 190, 60), (w - 150, 100), drawing.color, -1)
            cv2.rectangle(frame, (w - 190, 60), (w - 150, 100), (255, 255, 255), 2)
            cv2.putText(frame, f"{drawing.color_name}  {drawing.thickness}px", (w - 140, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Help text (two lines so it fits narrow frames) and short messages
        if mouse.on:
            help1 = "pinch=click/drag | thumb+middle=right click | PEACE+move=scroll"
            help2 = "FIST=pause | hold THUMBS UP or press m = exit | q quit"
        else:
            help1 = "POINT draw | PEACE color | FIST erase | PALM move | c clear | s save"
            help2 = "m or hold THUMBS UP = mouse mode | [ ] size | q quit"
        cv2.putText(frame, help1, (10, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, help2, (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        if toast[0] and time.time() - toast[1] < 2.0:
            cv2.putText(frame, toast[0], (20, h - 65),
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
        elif key == ord("m"):
            toggle_mouse()
        elif key == ord("c"):
            drawing.clear()
        elif key == ord("s"):
            path = drawing.save()
            show_toast(f"Saved {os.path.basename(path)}")
        elif key == ord("["):
            drawing.change_thickness(-2)
        elif key == ord("]"):
            drawing.change_thickness(2)

    mouse.disable()
    landmarker.close()
    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
