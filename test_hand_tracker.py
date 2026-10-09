"""
Tests for hand_tracker.py. No camera, model file, or pyautogui needed.

Run from the project folder (same folder as hand_tracker.py):
    python -m unittest -v test_hand_tracker.py
or run everything named test_*.py:
    python -m unittest discover -v
"""
import math
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import cv2
import numpy as np

import hand_tracker as ht


# --------------------------------------------------------------------------
# Synthetic hands (image coordinates: y grows downward; palm length ~105 px)
# --------------------------------------------------------------------------
WR = (100, 300)
MCP = {"I": (85, 200), "M": (100, 195), "R": (115, 200), "P": (130, 210)}
IDX = {"I": (5, 6, 7, 8), "M": (9, 10, 11, 12), "R": (13, 14, 15, 16), "P": (17, 18, 19, 20)}


def build_hand(extended, thumb):
    """extended: letters of the extended fingers, e.g. "IM". thumb: "up", "out" or "tucked"."""
    pts = [(0.0, 0.0)] * 21
    pts[0] = WR
    for f, (mcp_i, pip_i, dip_i, tip_i) in IDX.items():
        x, y = MCP[f]
        pts[mcp_i] = (x, y)
        if f in extended:   # straight up
            pts[pip_i], pts[dip_i], pts[tip_i] = (x, y - 60), (x, y - 85), (x, y - 105)
        else:               # curled toward the palm
            pts[pip_i], pts[dip_i], pts[tip_i] = (x, y - 25), (x, y - 5), (x, y + 15)
    if thumb == "up":
        pts[1], pts[2], pts[3], pts[4] = (82, 285), (78, 245), (78, 190), (78, 120)
    elif thumb == "out":
        pts[1], pts[2], pts[3], pts[4] = (80, 285), (55, 265), (35, 250), (15, 235)
    else:  # tucked across the palm
        pts[1], pts[2], pts[3], pts[4] = (85, 285), (95, 260), (105, 240), (110, 230)
    return pts


def rotate(pts, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    cx, cy = WR
    return [((x - cx) * c - (y - cy) * s + cx, (x - cx) * s + (y - cy) * c + cy) for x, y in pts]


def simple_hand(tip=(300, 200), palm_center=(300, 400)):
    """A hand with just enough landmarks for the drawing code (palm length 100)."""
    pts = [(0.0, 0.0)] * 21
    pts[ht.WRIST] = (palm_center[0], palm_center[1] + 60)
    pts[ht.MIDDLE_MCP] = (palm_center[0], palm_center[1] - 40)
    pts[ht.INDEX_MCP] = (palm_center[0] - 15, palm_center[1] - 35)
    pts[ht.RING_MCP] = (palm_center[0] + 15, palm_center[1] - 35)
    pts[ht.PINKY_MCP] = (palm_center[0] + 30, palm_center[1] - 25)
    pts[ht.INDEX_TIP] = tip
    return pts


def mouse_hand(index_tip=(300, 260), thumb_tip=(230, 330), middle_tip=(320, 250)):
    """A hand for mouse tests. Palm length is exactly 100; defaults are an open hand."""
    pts = [(0.0, 0.0)] * 21
    pts[ht.WRIST] = (300, 460)
    pts[ht.MIDDLE_MCP] = (300, 360)
    pts[ht.INDEX_MCP] = (285, 365)
    pts[ht.RING_MCP] = (315, 365)
    pts[ht.PINKY_MCP] = (330, 375)
    pts[ht.INDEX_TIP] = index_tip
    pts[ht.THUMB_TIP] = thumb_tip
    pts[ht.MIDDLE_TIP] = middle_tip
    return pts


def left_pinch_hand():
    return mouse_hand(index_tip=(300, 260), thumb_tip=(298, 262))     # thumb meets index


def right_pinch_hand():
    return mouse_hand(index_tip=(250, 250), thumb_tip=(315, 255), middle_tip=(320, 250))


def fist_pinch_hand():
    """Fist with the thumb resting on the curled index finger: must NOT count as a click."""
    return mouse_hand(index_tip=(300, 420), thumb_tip=(295, 415), middle_tip=(305, 415))


def shifted(pts, dx, dy):
    return [(x + dx, y + dy) for x, y in pts]


class FakeBackend:
    """Records mouse actions instead of moving the real mouse."""

    def __init__(self, size=(1920, 1080)):
        self._size = size
        self.calls = []

    def size(self):
        return self._size

    def move_to(self, x, y):
        self.calls.append(("move", x, y))

    def mouse_down(self):
        self.calls.append(("down",))

    def mouse_up(self):
        self.calls.append(("up",))

    def click(self, button="left"):
        self.calls.append(("click", button))

    def scroll(self, clicks):
        self.calls.append(("scroll", clicks))

    def names(self):
        return [c[0] for c in self.calls]

    def count(self, name):
        return self.names().count(name)


# --------------------------------------------------------------------------
class TestHelpers(unittest.TestCase):
    def test_dist(self):
        self.assertAlmostEqual(ht.dist((0, 0), (3, 4)), 5.0)
        self.assertEqual(ht.dist((2, 2), (2, 2)), 0.0)


class TestGestures(unittest.TestCase):
    def check(self, expected, extended, thumb):
        pts = build_hand(extended, thumb)
        self.assertEqual(ht.classify_gesture(pts), expected,
                         f"states={ht.finger_states(pts)}")

    def test_open_palm_thumb_out(self):
        self.check("OPEN PALM", "IMRP", "out")

    def test_open_palm_thumb_tucked(self):
        self.check("OPEN PALM", "IMRP", "tucked")

    def test_fist(self):
        self.check("FIST", "", "tucked")

    def test_thumbs_up(self):
        self.check("THUMBS UP", "", "up")

    def test_point(self):
        self.check("POINT", "I", "tucked")
        self.check("POINT", "I", "out")

    def test_peace(self):
        self.check("PEACE", "IM", "tucked")

    def test_rock(self):
        self.check("ROCK", "IP", "out")

    def test_unknown(self):
        self.check("UNKNOWN", "IMR", "tucked")

    def test_finger_states_open_hand(self):
        self.assertEqual(ht.finger_states(build_hand("IMRP", "out")), [True] * 5)

    def test_finger_states_fist(self):
        self.assertEqual(ht.finger_states(build_hand("", "tucked")), [False] * 5)

    def test_rotation_invariance_of_four_fingers(self):
        pts = build_hand("IM", "tucked")
        for angle in (0, 45, 90, 180, 270):
            self.assertEqual(ht.classify_gesture(rotate(pts, angle)), "PEACE", f"angle={angle}")


class TestGestureSmoother(unittest.TestCase):
    def test_single_glitch_is_ignored(self):
        sm = ht.GestureSmoother()
        for _ in range(5):
            sm.update("FIST")
        self.assertEqual(sm.update("PEACE"), "FIST")

    def test_switches_after_enough_votes(self):
        sm = ht.GestureSmoother()
        for _ in range(7):
            sm.update("FIST")
        out = [sm.update("PEACE") for _ in range(7)]
        self.assertEqual(out[-1], "PEACE")

    def test_starts_as_none_and_resets(self):
        sm = ht.GestureSmoother()
        self.assertEqual(sm.current, "NONE")
        for _ in range(5):
            sm.update("POINT")
        sm.reset()
        self.assertEqual(sm.current, "NONE")


class TestDrawingCanvas(unittest.TestCase):
    def setUp(self):
        self.d = ht.DrawingCanvas(thickness=6)
        self.d.ensure((480, 640, 3))
        self.t = 100.0

    def step(self, gesture, pts):
        self.d.update(gesture, pts, self.t)
        self.t += 0.03

    def test_point_draws_a_continuous_line(self):
        for x in range(100, 301, 10):
            self.step("POINT", simple_hand(tip=(x, 200)))
        mask = self.d.canvas.any(axis=2)
        self.assertGreater(mask.sum(), 500)
        ys, xs = np.nonzero(mask)
        self.assertGreaterEqual(ys.min(), 190)
        self.assertLessEqual(ys.max(), 210)
        self.assertLessEqual(xs.min(), 100)
        self.assertGreaterEqual(xs.max(), 290)

    def test_open_palm_does_not_draw_and_lifts_stroke(self):
        self.step("POINT", simple_hand(tip=(100, 200)))
        self.step("POINT", simple_hand(tip=(150, 200)))
        before = self.d.canvas.copy()
        for x in range(200, 500, 20):
            self.step("OPEN PALM", simple_hand(tip=(x, 300)))
        np.testing.assert_array_equal(before, self.d.canvas)
        self.assertIsNone(self.d.prev)

    def test_new_stroke_does_not_connect_to_old_one(self):
        self.step("POINT", simple_hand(tip=(100, 200)))
        self.step("POINT", simple_hand(tip=(150, 200)))
        self.step("OPEN PALM", simple_hand(tip=(500, 300)))
        self.step("POINT", simple_hand(tip=(500, 300)))
        self.step("POINT", simple_hand(tip=(520, 300)))
        # nothing drawn between the end of stroke one and the start of stroke two
        self.assertFalse(self.d.canvas[200, 250:450].any())

    def test_fist_erases_under_palm(self):
        self.d.canvas[380:420, 280:320] = (0, 255, 0)
        self.step("FIST", simple_hand(palm_center=(300, 400)))
        self.assertFalse(self.d.canvas[400, 300].any())
        self.assertIsNotNone(self.d.eraser)

    def test_peace_changes_color_once_per_entry(self):
        start = self.d.color_idx
        self.d.update("PEACE", simple_hand(), 1000.0)
        after_first = self.d.color_idx
        for k in range(5):  # holding the peace sign must not keep cycling
            self.d.update("PEACE", simple_hand(), 1000.1 + 0.1 * k)
        self.assertEqual(after_first, (start + 1) % len(ht.COLORS))
        self.assertEqual(self.d.color_idx, after_first)

    def test_color_cooldown_blocks_rapid_changes(self):
        self.d.update("PEACE", simple_hand(), 1000.0)
        idx = self.d.color_idx
        self.d.update("FIST", simple_hand(), 1000.2)
        self.d.update("PEACE", simple_hand(), 1000.3)
        self.assertEqual(self.d.color_idx, idx)
        self.d.update("FIST", simple_hand(), 1002.0)
        self.d.update("PEACE", simple_hand(), 1002.1)
        self.assertEqual(self.d.color_idx, (idx + 1) % len(ht.COLORS))

    def test_thickness_is_clamped(self):
        self.d.change_thickness(-100)
        self.assertEqual(self.d.thickness, 2)
        self.d.change_thickness(100)
        self.assertEqual(self.d.thickness, 30)

    def test_lift_and_clear(self):
        self.step("POINT", simple_hand(tip=(100, 200)))
        self.step("POINT", simple_hand(tip=(150, 200)))
        self.d.lift()
        self.assertIsNone(self.d.prev)
        self.assertIsNone(self.d.cursor)
        self.d.clear()
        self.assertFalse(self.d.canvas.any())

    def test_composite_paints_only_drawn_pixels(self):
        frame = np.full((480, 640, 3), 50, np.uint8)
        self.d.canvas[10:20, 10:20] = (0, 0, 255)
        self.d.composite(frame)
        self.assertEqual(tuple(frame[15, 15]), (0, 0, 255))
        self.assertEqual(tuple(frame[100, 100]), (50, 50, 50))

    def test_composite_matches_boolean_mask_version(self):
        frame = np.random.randint(0, 255, (480, 640, 3), np.uint8)
        for x in range(50, 400, 25):
            self.step("POINT", simple_hand(tip=(x, 100 + x // 3)))
        expected = frame.copy()
        m = self.d.canvas.any(axis=2)
        expected[m] = self.d.canvas[m]
        actual = frame.copy()
        self.d.composite(actual)
        np.testing.assert_array_equal(actual, expected)

    def test_save_writes_transparent_png(self):
        self.d.canvas[10:20, 10:20] = (0, 0, 255)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(ht, "DRAWINGS_DIR", tmp):
                path = self.d.save()
            self.assertTrue(os.path.exists(path))
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.assertEqual(img.shape[2], 4)
        self.assertEqual(img[15, 15, 3], 255)
        self.assertEqual(img[100, 100, 3], 0)

    def test_ensure_resets_on_size_change(self):
        self.d.canvas[10:20, 10:20] = (0, 0, 255)
        self.d.ensure((240, 320, 3))
        self.assertEqual(self.d.canvas.shape, (240, 320, 3))
        self.assertFalse(self.d.canvas.any())


class FakeCap:
    """Stands in for cv2.VideoCapture: yields numbered frames."""

    def __init__(self, n_frames=None, delay=0.002):
        self.n = 0
        self.n_frames = n_frames
        self.delay = delay
        self.released = False

    def read(self):
        if self.n_frames is not None and self.n >= self.n_frames:
            return False, None
        time.sleep(self.delay)
        self.n += 1
        return True, np.full((4, 4, 3), self.n % 256, np.uint8)

    def release(self):
        self.released = True


class TestCameraStream(unittest.TestCase):
    def test_returns_newest_frames_and_never_the_same_one_twice(self):
        stream = ht.CameraStream(FakeCap())
        try:
            ok, frame, fid = stream.read(-1, timeout=2.0)
            self.assertTrue(ok)
            seen = {fid}
            for _ in range(5):
                ok, frame, fid2 = stream.read(fid, timeout=2.0)
                self.assertTrue(ok)
                self.assertNotEqual(fid2, fid)
                seen.add(fid2)
                fid = fid2
            self.assertEqual(len(seen), 6)
        finally:
            stream.stop()

    def test_slow_consumer_drops_old_frames(self):
        stream = ht.CameraStream(FakeCap(delay=0.001))
        try:
            ok, _, first = stream.read(-1, timeout=2.0)
            time.sleep(0.1)  # camera keeps producing while we are "busy"
            ok, _, second = stream.read(first, timeout=2.0)
            self.assertTrue(ok)
            self.assertGreater(second - first, 5)  # skipped ahead to a recent frame
        finally:
            stream.stop()

    def test_reports_end_of_stream(self):
        stream = ht.CameraStream(FakeCap(n_frames=3))
        try:
            fid, got = -1, 0
            while True:
                ok, _, fid = stream.read(fid, timeout=1.0)
                if not ok:
                    break
                got += 1
            self.assertGreaterEqual(got, 1)
            self.assertLessEqual(got, 3)
        finally:
            stream.stop()

    def test_times_out_when_no_frames_arrive(self):
        stream = ht.CameraStream(FakeCap(n_frames=0))
        try:
            start = time.time()
            ok, frame, _ = stream.read(-1, timeout=0.3)
            self.assertFalse(ok)
            self.assertIsNone(frame)
            self.assertLess(time.time() - start, 1.5)
        finally:
            stream.stop()

    def test_stop_releases_camera_and_ends_thread(self):
        cap = FakeCap()
        stream = ht.CameraStream(cap)
        stream.stop()
        self.assertTrue(cap.released)
        self.assertFalse(stream.thread.is_alive())


# --------------------------------------------------------------------------
# Virtual mouse
# --------------------------------------------------------------------------
class TestScreenMapping(unittest.TestCase):
    def test_center_maps_to_center(self):
        u, v = ht.map_to_screen((320, 240), 640, 480, 1920, 1080)
        self.assertAlmostEqual(u, 0.5)
        self.assertAlmostEqual(v, 0.5)

    def test_active_box_edges_map_to_screen_edges(self):
        u, v = ht.map_to_screen((0.2 * 640, 0.2 * 480), 640, 480, 1920, 1080)
        self.assertAlmostEqual(u, 0.0)
        self.assertAlmostEqual(v, 0.0)
        u, v = ht.map_to_screen((0.8 * 640, 0.8 * 480), 640, 480, 1920, 1080)
        self.assertAlmostEqual(u, 1.0)
        self.assertAlmostEqual(v, 1.0)

    def test_outside_the_box_is_clamped(self):
        self.assertEqual(ht.map_to_screen((0, 0), 640, 480, 1920, 1080), (0.0, 0.0))
        self.assertEqual(ht.map_to_screen((640, 480), 640, 480, 1920, 1080), (1.0, 1.0))

    def test_pixels_never_touch_the_failsafe_corner(self):
        self.assertEqual(ht.to_pixels(0.0, 0.0, 1920, 1080), (ht.SCREEN_EDGE_PAD, ht.SCREEN_EDGE_PAD))
        x, y = ht.to_pixels(1.0, 1.0, 1920, 1080)
        self.assertEqual((x, y), (1920 - 1 - ht.SCREEN_EDGE_PAD, 1080 - 1 - ht.SCREEN_EDGE_PAD))
        self.assertGreater(ht.SCREEN_EDGE_PAD, 0)


class TestCursorSmoother(unittest.TestCase):
    def test_first_value_passes_through(self):
        s = ht.CursorSmoother()
        self.assertEqual(s.update(0.3, 0.7), (0.3, 0.7))

    def test_small_jitter_is_damped(self):
        s = ht.CursorSmoother()
        s.update(0.5, 0.5)
        x, _ = s.update(0.51, 0.5)
        self.assertLess(abs(x - 0.5), 0.003)      # moved less than a third of the jitter

    def test_fast_movement_is_followed_closely(self):
        s = ht.CursorSmoother()
        s.update(0.5, 0.5)
        x, _ = s.update(0.9, 0.5)
        self.assertGreater(x, 0.75)               # most of a big jump comes through at once

    def test_converges_to_a_steady_target(self):
        s = ht.CursorSmoother()
        s.update(0.2, 0.2)
        for _ in range(100):
            pos = s.update(0.8, 0.6)
        self.assertAlmostEqual(pos[0], 0.8, places=3)
        self.assertAlmostEqual(pos[1], 0.6, places=3)

    def test_reset_forgets_position(self):
        s = ht.CursorSmoother()
        s.update(0.2, 0.2)
        s.reset()
        self.assertEqual(s.update(0.9, 0.9), (0.9, 0.9))


class TestPinchDetector(unittest.TestCase):
    def test_needs_consecutive_frames_to_press(self):
        p = ht.PinchDetector(on=0.25, off=0.4, frames=2)
        self.assertIsNone(p.update(0.1))
        self.assertEqual(p.update(0.1), "down")

    def test_single_glitch_frame_does_not_click(self):
        p = ht.PinchDetector(frames=2)
        self.assertIsNone(p.update(0.1))
        self.assertIsNone(p.update(1.0))
        self.assertIsNone(p.update(0.1))
        self.assertFalse(p.active)

    def test_hysteresis_holds_between_thresholds(self):
        p = ht.PinchDetector(on=0.25, off=0.4, frames=1)
        self.assertEqual(p.update(0.1), "down")
        self.assertIsNone(p.update(0.3))          # between on and off: still pinched
        self.assertTrue(p.active)
        self.assertEqual(p.update(0.5), "up")
        self.assertFalse(p.active)

    def test_only_one_down_per_pinch(self):
        p = ht.PinchDetector(frames=1)
        events = [p.update(0.1) for _ in range(5)]
        self.assertEqual(events.count("down"), 1)

    def test_invalid_input_blocks_and_releases(self):
        p = ht.PinchDetector(frames=1)
        self.assertIsNone(p.update(0.05, valid=False))
        self.assertEqual(p.update(0.05, valid=True), "down")
        self.assertEqual(p.update(0.05, valid=False), "up")

    def test_engaged_covers_press_in_progress(self):
        p = ht.PinchDetector(frames=3)
        p.update(0.1)
        self.assertTrue(p.engaged)
        self.assertFalse(p.active)


class TestHoldDetector(unittest.TestCase):
    def test_fires_once_after_the_hold_time(self):
        h = ht.HoldDetector(hold=1.0)
        self.assertFalse(h.update("THUMBS UP", 0.0))
        self.assertFalse(h.update("THUMBS UP", 0.5))
        self.assertTrue(h.update("THUMBS UP", 1.0))
        self.assertFalse(h.update("THUMBS UP", 1.5))     # does not repeat while held

    def test_rearms_after_the_gesture_changes(self):
        h = ht.HoldDetector(hold=1.0)
        h.update("THUMBS UP", 0.0)
        self.assertTrue(h.update("THUMBS UP", 1.0))
        h.update("FIST", 1.2)
        self.assertFalse(h.update("THUMBS UP", 2.0))
        self.assertTrue(h.update("THUMBS UP", 3.0))

    def test_interrupted_hold_does_not_fire(self):
        h = ht.HoldDetector(hold=1.0)
        h.update("THUMBS UP", 0.0)
        h.update("OPEN PALM", 0.8)
        self.assertFalse(h.update("THUMBS UP", 1.2))

    def test_progress_bar_value(self):
        h = ht.HoldDetector(hold=2.0)
        self.assertEqual(h.progress(0.0), 0.0)
        h.update("THUMBS UP", 0.0)
        self.assertAlmostEqual(h.progress(1.0), 0.5)
        h.update("THUMBS UP", 2.0)
        self.assertEqual(h.progress(2.5), 0.0)


class TestMouseController(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.m = ht.MouseController(self.backend)
        self.t = 10.0

    def run_frame(self, gesture, pts, dt=0.033):
        self.m.update(gesture, pts, 640, 480, self.t)
        self.t += dt

    def test_hand_moves_the_cursor_inside_the_screen(self):
        self.run_frame("OPEN PALM", mouse_hand())
        self.assertEqual(self.backend.count("move"), 1)
        _, x, y = self.backend.calls[0]
        self.assertTrue(ht.SCREEN_EDGE_PAD <= x <= 1920 - 1 - ht.SCREEN_EDGE_PAD)
        self.assertTrue(ht.SCREEN_EDGE_PAD <= y <= 1080 - 1 - ht.SCREEN_EDGE_PAD)

    def test_hand_at_corner_of_view_never_hits_failsafe_corner(self):
        pts = shifted(mouse_hand(), -400, -400)
        self.run_frame("OPEN PALM", pts)
        _, x, y = self.backend.calls[0]
        self.assertGreaterEqual(x, ht.SCREEN_EDGE_PAD)
        self.assertGreaterEqual(y, ht.SCREEN_EDGE_PAD)

    def test_tiny_movements_are_ignored(self):
        for _ in range(5):
            self.run_frame("OPEN PALM", mouse_hand())
        self.assertEqual(self.backend.count("move"), 1)

    def test_moving_the_hand_moves_the_cursor_the_same_way(self):
        self.run_frame("OPEN PALM", mouse_hand())
        for _ in range(30):
            self.run_frame("OPEN PALM", shifted(mouse_hand(), 80, 0))
        xs = [c[1] for c in self.backend.calls if c[0] == "move"]
        self.assertGreater(xs[-1], xs[0] + 300)

    def test_left_pinch_presses_once_and_release_lifts(self):
        self.run_frame("OPEN PALM", mouse_hand())
        for _ in range(5):
            self.run_frame("POINT", left_pinch_hand())
        self.assertEqual(self.backend.count("down"), 1)
        self.assertEqual(self.backend.count("up"), 0)
        self.run_frame("POINT", mouse_hand())
        self.assertEqual(self.backend.count("up"), 1)
        names = self.backend.names()
        self.assertLess(names.index("down"), names.index("up"))

    def test_dragging_moves_the_cursor_while_pressed(self):
        self.run_frame("OPEN PALM", mouse_hand())
        for _ in range(3):
            self.run_frame("POINT", left_pinch_hand())
        moves_before = self.backend.count("move")
        for _ in range(20):
            self.run_frame("POINT", shifted(left_pinch_hand(), 80, 0))
        self.assertGreater(self.backend.count("move"), moves_before)
        self.assertEqual(self.backend.count("up"), 0)

    def test_losing_the_hand_releases_the_button(self):
        for _ in range(3):
            self.run_frame("POINT", left_pinch_hand())
        self.assertTrue(self.m.left_down)
        self.m.hand_lost()
        self.assertFalse(self.m.left_down)
        self.assertEqual(self.backend.count("up"), 1)

    def test_release_all_does_nothing_when_nothing_is_held(self):
        self.m.release_all()
        self.assertEqual(self.backend.count("up"), 0)

    def test_fist_pauses_the_cursor(self):
        self.run_frame("OPEN PALM", mouse_hand())
        moves = self.backend.count("move")
        for _ in range(10):
            self.run_frame("FIST", shifted(mouse_hand(), 100, 50))
        self.assertEqual(self.backend.count("move"), moves)

    def test_fist_with_thumb_on_finger_is_not_a_click(self):
        for _ in range(10):
            self.run_frame("FIST", fist_pinch_hand())
        self.assertEqual(self.backend.count("down"), 0)
        self.assertEqual(self.backend.count("click"), 0)

    def test_peace_scrolls_up_when_the_hand_goes_up(self):
        self.run_frame("PEACE", mouse_hand(index_tip=(290, 260), middle_tip=(310, 260)))
        self.run_frame("PEACE", mouse_hand(index_tip=(290, 210), middle_tip=(310, 210)))
        self.assertIn(("scroll", 4), self.backend.calls)
        self.assertEqual(self.backend.count("move"), 0)          # cursor stays put while scrolling

    def test_peace_scrolls_down_when_the_hand_goes_down(self):
        self.run_frame("PEACE", mouse_hand(index_tip=(290, 210), middle_tip=(310, 210)))
        self.run_frame("PEACE", mouse_hand(index_tip=(290, 235), middle_tip=(310, 235)))
        self.assertIn(("scroll", -2), self.backend.calls)

    def test_scroll_resets_between_peace_signs(self):
        self.run_frame("PEACE", mouse_hand(index_tip=(290, 260), middle_tip=(310, 260)))
        self.run_frame("OPEN PALM", mouse_hand())
        self.run_frame("PEACE", mouse_hand(index_tip=(290, 100), middle_tip=(310, 100)))
        self.assertEqual(self.backend.count("scroll"), 0)       # no jump from the old position

    def test_right_click_once_per_pinch(self):
        self.run_frame("OPEN PALM", mouse_hand())
        for _ in range(6):
            self.run_frame("UNKNOWN", right_pinch_hand())
        self.assertEqual(self.backend.calls.count(("click", "right")), 1)
        self.assertEqual(self.backend.count("down"), 0)

    def test_right_click_cooldown_then_allowed_again(self):
        for _ in range(3):
            self.run_frame("UNKNOWN", right_pinch_hand(), dt=0.05)       # first right click
        self.run_frame("OPEN PALM", mouse_hand(), dt=0.05)
        for _ in range(3):
            self.run_frame("UNKNOWN", right_pinch_hand(), dt=0.05)       # too soon
        self.assertEqual(self.backend.calls.count(("click", "right")), 1)
        self.run_frame("OPEN PALM", mouse_hand(), dt=1.0)
        for _ in range(3):
            self.run_frame("UNKNOWN", right_pinch_hand(), dt=0.05)       # after cooldown
        self.assertEqual(self.backend.calls.count(("click", "right")), 2)

    def test_cursor_is_frozen_while_a_right_click_pinch_forms(self):
        self.run_frame("OPEN PALM", mouse_hand())
        moves = self.backend.count("move")
        for _ in range(6):
            self.run_frame("UNKNOWN", right_pinch_hand())
        self.assertEqual(self.backend.count("move"), moves)

    def test_left_pinch_takes_priority_over_right(self):
        both = mouse_hand(index_tip=(300, 260), thumb_tip=(300, 262), middle_tip=(301, 261))
        for _ in range(6):
            self.run_frame("UNKNOWN", both)
        self.assertEqual(self.backend.count("down"), 1)
        self.assertEqual(self.backend.count("click"), 0)


class TestMouseMode(unittest.TestCase):
    def make(self, backend=None):
        backend = backend or FakeBackend()
        return ht.MouseMode(backend_factory=lambda: backend), backend

    def test_toggle_on_and_off(self):
        mode, _ = self.make()
        self.assertFalse(mode.on)
        ok, msg = mode.toggle()
        self.assertTrue(ok)
        self.assertTrue(mode.on)
        ok, msg = mode.toggle()
        self.assertFalse(mode.on)

    def test_missing_pyautogui_gives_a_helpful_message_and_stays_off(self):
        def broken():
            raise RuntimeError("pyautogui is not installed. Run: pip install pyautogui")
        mode = ht.MouseMode(backend_factory=broken)
        ok, msg = mode.enable()
        self.assertFalse(ok)
        self.assertFalse(mode.on)
        self.assertIn("pip install pyautogui", msg)

    def test_disable_releases_a_held_button(self):
        mode, backend = self.make()
        mode.enable()
        for i in range(4):
            mode.update("POINT", left_pinch_hand(), 640, 480, 10.0 + i * 0.03)
        self.assertEqual(backend.count("down"), 1)
        mode.disable()
        self.assertEqual(backend.count("up"), 1)
        self.assertFalse(mode.on)

    def test_backend_error_turns_mouse_mode_off(self):
        backend = FakeBackend()

        def boom(x, y):
            raise RuntimeError("failsafe triggered")
        backend.move_to = boom
        mode, _ = self.make(backend)
        mode.enable()
        message = mode.update("OPEN PALM", mouse_hand(), 640, 480, 10.0)
        self.assertIsNotNone(message)
        self.assertFalse(mode.on)

    def test_disable_survives_a_failing_release(self):
        backend = FakeBackend()
        mode, _ = self.make(backend)
        mode.enable()
        for i in range(4):
            mode.update("POINT", left_pinch_hand(), 640, 480, 10.0 + i * 0.03)

        def boom():
            raise RuntimeError("cannot release")
        backend.mouse_up = boom
        mode.disable()                     # must not raise
        self.assertFalse(mode.on)

    def test_hand_lost_is_harmless_when_mode_is_off(self):
        mode, backend = self.make()
        mode.hand_lost()
        self.assertEqual(backend.calls, [])

    def test_re_enabling_starts_clean(self):
        mode, backend = self.make()
        mode.enable()
        mode.update("OPEN PALM", mouse_hand(), 640, 480, 10.0)
        mode.disable()
        mode.enable()
        self.assertIsNone(mode.controller.last_px)
        self.assertFalse(mode.controller.left_down)


class TestPyAutoGuiBackend(unittest.TestCase):
    def test_import_error_becomes_install_hint(self):
        with mock.patch.dict(sys.modules, {"pyautogui": None}):
            with self.assertRaises(RuntimeError) as ctx:
                ht.PyAutoGuiBackend()
        self.assertIn("pip install pyautogui", str(ctx.exception))

    def test_failsafe_is_on_and_pause_is_zero(self):
        fake = types.SimpleNamespace(FAILSAFE=False, PAUSE=0.1)
        with mock.patch.dict(sys.modules, {"pyautogui": fake}):
            ht.PyAutoGuiBackend()
        self.assertTrue(fake.FAILSAFE)
        self.assertEqual(fake.PAUSE, 0)

    def test_mouse_up_works_even_with_failsafe_and_restores_it(self):
        seen = {}

        def mouse_up():
            seen["failsafe_during_call"] = fake.FAILSAFE
        fake = types.SimpleNamespace(FAILSAFE=True, PAUSE=0, mouseUp=mouse_up)
        with mock.patch.dict(sys.modules, {"pyautogui": fake}):
            backend = ht.PyAutoGuiBackend()
            backend.mouse_up()
        self.assertFalse(seen["failsafe_during_call"])
        self.assertTrue(fake.FAILSAFE)

    def test_mouse_up_restores_failsafe_after_an_error(self):
        def mouse_up():
            raise OSError("boom")
        fake = types.SimpleNamespace(FAILSAFE=True, PAUSE=0, mouseUp=mouse_up)
        with mock.patch.dict(sys.modules, {"pyautogui": fake}):
            backend = ht.PyAutoGuiBackend()
            with self.assertRaises(OSError):
                backend.mouse_up()
        self.assertTrue(fake.FAILSAFE)

    def test_size_is_returned_as_plain_ints(self):
        fake = types.SimpleNamespace(FAILSAFE=True, PAUSE=0, size=lambda: (1440, 900))
        with mock.patch.dict(sys.modules, {"pyautogui": fake}):
            self.assertEqual(ht.PyAutoGuiBackend().size(), (1440, 900))


if __name__ == "__main__":
    unittest.main(verbosity=2)
