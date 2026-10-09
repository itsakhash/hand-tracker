"""
Tests for hand_tracker.py (and its use of gesture_model.py). No camera, model file, or pyautogui needed.

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


def hand_at(mid=(265, 295), spread=1.0):
    """Open hand whose pointer sits at `mid`; spread 1.0 = open, ~0.05 = thumb and index closed."""
    index_tip = (mid[0] + 35 * spread, mid[1] - 35 * spread)
    thumb_tip = (mid[0] - 35 * spread, mid[1] + 35 * spread)
    return mouse_hand(index_tip=index_tip, thumb_tip=thumb_tip)


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


class TestOneEuroFilter(unittest.TestCase):
    DT = 1 / 30

    def feed(self, f, u, v, n, t0=0.0):
        pos = None
        for i in range(n):
            pos = f.update(u, v, t0 + i * self.DT)
        return pos

    def test_first_value_passes_through(self):
        f = ht.OneEuroFilter2D()
        self.assertEqual(f.update(0.3, 0.7, 0.0), (0.3, 0.7))

    def test_small_jitter_is_damped(self):
        f = ht.OneEuroFilter2D()
        self.feed(f, 0.5, 0.5, 30)
        x, _ = f.update(0.51, 0.5, 30 * self.DT)
        self.assertGreater(x, 0.5)
        self.assertLess(x - 0.5, 0.004)           # less than half of the 0.01 jitter gets through

    def test_fast_movement_is_followed_closely(self):
        f = ht.OneEuroFilter2D()
        self.feed(f, 0.5, 0.5, 30)
        x, _ = f.update(0.9, 0.5, 30 * self.DT)
        self.assertGreater(x, 0.75)               # most of a big jump comes through at once

    def test_converges_to_a_steady_target(self):
        f = ht.OneEuroFilter2D()
        f.update(0.2, 0.2, 0.0)
        pos = self.feed(f, 0.8, 0.6, 200, t0=self.DT)
        self.assertAlmostEqual(pos[0], 0.8, places=3)
        self.assertAlmostEqual(pos[1], 0.6, places=3)

    def test_reset_forgets_position(self):
        f = ht.OneEuroFilter2D()
        f.update(0.2, 0.2, 0.0)
        f.reset()
        self.assertEqual(f.update(0.9, 0.9, 5.0), (0.9, 0.9))

    def test_behaves_the_same_at_different_frame_rates(self):
        finals = []
        for fps in (15, 30):
            f = ht.OneEuroFilter2D()
            pos = None
            for i in range(fps + 1):
                t = i / fps
                pos = f.update(0.2 + 0.6 * t, 0.5, t)
            finals.append(pos[0])
        self.assertLess(abs(finals[0] - finals[1]), 0.03)

    def test_duplicate_timestamps_do_not_crash(self):
        f = ht.OneEuroFilter2D()
        f.update(0.5, 0.5, 1.0)
        pos = f.update(0.6, 0.5, 1.0)
        self.assertTrue(all(math.isfinite(c) for c in pos))


class TestCursorDriver(unittest.TestCase):
    def make(self, tau=0.04):
        backend = FakeBackend()
        return ht.CursorDriver(backend, tau=tau), backend

    def wait_for(self, condition, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline and not condition():
            time.sleep(0.005)
        return condition()

    def test_no_target_means_no_movement(self):
        d, b = self.make()
        self.assertIsNone(d.step(0.01))
        self.assertEqual(b.calls, [])

    def test_first_target_snaps_there(self):
        d, b = self.make()
        d.set_target((100, 200))
        self.assertEqual(d.step(0.008), (100, 200))
        self.assertEqual(b.calls, [("move", 100, 200)])

    def test_chases_a_new_target_exponentially(self):
        d, b = self.make(tau=0.04)
        d.set_target((0, 0))
        d.step(0.01)
        d.set_target((100, 0))
        out = d.step(0.04)                         # one time constant: ~63% of the way
        self.assertEqual(out[0], 63)

    def test_glides_through_many_intermediate_positions(self):
        d, b = self.make()
        d.set_target((0, 0))
        d.step(0.01)
        d.set_target((300, 0))
        outs = [d.step(1 / 120) for _ in range(20)]
        xs = [o[0] for o in outs if o]
        self.assertGreater(len(set(xs)), 10)       # smooth, not one big jump
        self.assertEqual(xs, sorted(xs))
        self.assertLessEqual(xs[-1], 300)

    def test_stops_sending_moves_once_it_has_arrived(self):
        d, b = self.make()
        d.set_target((50, 50))
        for _ in range(300):
            d.step(0.01)
        n = len(b.calls)
        d.step(0.01)
        self.assertEqual(len(b.calls), n)
        self.assertEqual(b.calls[-1], ("move", 50, 50))

    def test_real_thread_moves_the_cursor_and_stops(self):
        d, b = self.make(tau=0.01)
        d.start()
        d.set_target((10, 20))
        self.assertTrue(self.wait_for(lambda: b.count("move") > 0))
        self.assertEqual(b.calls[0], ("move", 10, 20))
        d.stop()
        self.assertFalse(d.thread.is_alive())

    def test_start_twice_does_not_create_two_threads(self):
        d, b = self.make()
        d.start()
        first = d.thread
        d.start()
        self.assertIs(d.thread, first)
        d.stop()

    def test_backend_error_is_recorded_and_the_thread_ends(self):
        backend = FakeBackend()

        def boom(x, y):
            raise RuntimeError("failsafe")
        backend.move_to = boom
        d = ht.CursorDriver(backend)
        d.start()
        d.set_target((1, 1))
        self.assertTrue(self.wait_for(lambda: d.error is not None))
        self.assertIsInstance(d.error, RuntimeError)
        d.thread.join(timeout=1.0)
        self.assertFalse(d.thread.is_alive())

    def test_reset_clears_target_position_and_error(self):
        d, b = self.make()
        d.set_target((5, 5))
        d.step(0.01)
        d.error = RuntimeError("old")
        d.reset()
        self.assertIsNone(d.target)
        self.assertIsNone(d.error)
        self.assertIsNone(d.step(0.01))


class TestStillnessLeash(unittest.TestCase):
    def test_wobble_inside_the_circle_changes_nothing(self):
        s = ht.StillnessLeash(radius=8)
        first = s.update((100, 100))
        for px in [(103, 98), (96, 104), (105, 101), (99, 95), (107, 100)]:
            self.assertEqual(s.update(px), first)

    def test_leaving_the_circle_pulls_the_output_along_at_its_edge(self):
        s = ht.StillnessLeash(radius=8)
        s.update((0, 0))
        self.assertEqual(s.update((20, 0)), (12, 0))     # follows, trailing by the radius
        self.assertEqual(s.update((10, 0)), (12, 0))     # back inside the circle: stays

    def test_no_jump_when_starting_to_move(self):
        s = ht.StillnessLeash(radius=8)
        s.update((0, 0))
        self.assertEqual(s.update((9, 0)), (1, 0))       # only the part beyond the circle

    def test_radius_zero_passes_everything_through(self):
        s = ht.StillnessLeash(radius=0)
        s.update((0, 0))
        self.assertEqual(s.update((3, 4)), (3, 4))

    def test_set_and_reset(self):
        s = ht.StillnessLeash(radius=8)
        s.update((0, 0))
        s.set((500, 500))
        self.assertEqual(s.update((503, 502)), (500, 500))
        s.reset()
        self.assertEqual(s.update((7, 7)), (7, 7))


class TestCursorDriverSnap(unittest.TestCase):
    def test_snap_moves_immediately_and_nothing_chases_it_back(self):
        backend = FakeBackend()
        d = ht.CursorDriver(backend, tau=0.04)
        d.set_target((0, 0))
        d.step(0.01)
        d.set_target((300, 0))
        d.step(0.02)                                     # mid-chase
        self.assertEqual(d.snap((120, 40)), (120, 40))
        self.assertEqual(backend.calls[-1], ("move", 120, 40))
        n = len(backend.calls)
        self.assertIsNone(d.step(0.05))                  # target is the snap point now
        self.assertEqual(len(backend.calls), n)

    def test_snap_while_the_thread_is_running_sticks(self):
        backend = FakeBackend()
        d = ht.CursorDriver(backend, tau=0.01)
        d.start()
        d.set_target((0, 0))
        time.sleep(0.05)
        d.set_target((500, 500))
        d.snap((50, 60))
        time.sleep(0.15)
        d.stop()
        self.assertEqual(backend.calls[-1], ("move", 50, 60))


class TestMouseStability(unittest.TestCase):
    """The wobble and click-accuracy fixes, with a fake mouse."""

    def setUp(self):
        self.backend = FakeBackend()
        self.m = ht.MouseController(self.backend)
        self.t = 10.0

    def frame(self, gesture, pts, dt=0.033):
        self.m.update(gesture, pts, 640, 480, self.t)
        self.t += dt

    def moves(self):
        return [c for c in self.backend.calls if c[0] == "move"]

    def test_jitter_while_the_hand_is_still_does_not_move_the_cursor(self):
        self.frame("OPEN PALM", hand_at())
        n = len(self.moves())
        for i in range(60):
            wobble = 1.5 if i % 2 else -1.5
            self.frame("OPEN PALM", hand_at((265 + wobble, 295 - wobble), 1.0))
        self.assertEqual(len(self.moves()), n)

    def test_real_movement_still_gets_through(self):
        self.frame("OPEN PALM", hand_at())
        n = len(self.moves())
        for _ in range(20):
            self.frame("OPEN PALM", hand_at((305, 295), 1.0))
        self.assertGreater(len(self.moves()), n)

    def test_leash_can_be_turned_off(self):
        m = ht.MouseController(self.backend, still_radius=0)
        m.update("OPEN PALM", hand_at(), 640, 480, 10.0)
        before = len(self.moves())
        for i in range(10):
            m.update("OPEN PALM", hand_at((265 + 3 * (i % 2), 295), 1.0), 640, 480, 10.1 + i * 0.033)
        self.assertGreater(len(self.moves()), before)

    def close_fingers_while_drifting(self, drift_per_frame=4):
        """Hold still, then close thumb and index while the pointer drifts sideways."""
        for _ in range(10):
            self.frame("POINT", hand_at())
        before = self.m.last_px
        x = 265
        for spread in (0.6, 0.3, 0.05, 0.05):
            x += drift_per_frame
            self.frame("POINT", hand_at((x, 295), spread))
        return before, x

    def test_click_lands_where_the_cursor_was_before_the_fingers_closed(self):
        before, _ = self.close_fingers_while_drifting()
        self.assertEqual(self.backend.count("down"), 1)
        i = self.backend.names().index("down")
        last_move_before_click = [c for c in self.backend.calls[:i] if c[0] == "move"][-1]
        self.assertEqual((last_move_before_click[1], last_move_before_click[2]), before)

    def test_cursor_stays_locked_during_the_click_and_just_after_release(self):
        _, x = self.close_fingers_while_drifting()
        n = len(self.moves())
        for i in range(5):                                         # tiny wobble while pinched
            self.frame("POINT", hand_at((x + (2 if i % 2 else -2), 295), 0.05))
        self.assertEqual(len(self.moves()), n)
        self.frame("POINT", hand_at((x + 10, 295), 1.0))           # fingers open: release
        self.assertEqual(self.backend.count("up"), 1)
        for _ in range(3):                                         # still inside the freeze
            self.frame("POINT", hand_at((x + 10, 295), 1.0))
        self.assertEqual(len(self.moves()), n)
        self.t += 0.3                                              # freeze is over
        for _ in range(5):
            self.frame("POINT", hand_at((x + 10, 295), 1.0))
        self.assertGreater(len(self.moves()), n)

    def test_moving_far_while_pinched_becomes_a_drag(self):
        self.close_fingers_while_drifting(drift_per_frame=0)
        n = len(self.moves())
        for _ in range(10):
            self.frame("POINT", hand_at((320, 295), 0.05))
        self.assertGreater(len(self.moves()), n)
        self.assertEqual(self.backend.count("up"), 0)
        self.assertIsNone(self.m.click_lock)

    def test_fast_motion_is_not_snapped_back_when_you_click(self):
        for i in range(8):                                         # brisk sweep to the right
            self.frame("OPEN PALM", hand_at((200 + 25 * i, 295), 1.0))
        for _ in range(3):
            self.frame("POINT", hand_at((375, 295), 0.05))
        self.assertEqual(self.backend.count("down"), 1)
        xs = [c[1] for c in self.moves()]
        self.assertEqual(xs, sorted(xs))                           # never jumps backward

    def test_lookback_of_zero_clicks_at_the_current_spot(self):
        m = ht.MouseController(self.backend, lookback=0.0)
        t = 10.0
        for _ in range(10):
            m.update("POINT", hand_at(), 640, 480, t)
            t += 0.033
        x = 265
        for spread in (0.6, 0.3, 0.05, 0.05):
            x += 6
            m.update("POINT", hand_at((x, 295), spread), 640, 480, t)
            t += 0.033
        i = self.backend.names().index("down")
        last_move_before_click = [c for c in self.backend.calls[:i] if c[0] == "move"][-1]
        self.assertEqual((last_move_before_click[1], last_move_before_click[2]), m.last_px)

    def test_right_click_freezes_the_cursor_afterwards(self):
        for _ in range(10):
            self.frame("POINT", hand_at())
        before = self.m.last_px
        for _ in range(4):
            self.frame("UNKNOWN", right_pinch_hand())
        self.assertEqual(self.backend.calls.count(("click", "right")), 1)
        self.assertEqual(self.m.last_px, before)
        n = len(self.moves())
        for _ in range(8):                                         # 0.26 s: still frozen
            self.frame("OPEN PALM", hand_at((400, 295), 1.0))
        self.assertEqual(len(self.moves()), n)
        self.t += 0.5
        for _ in range(5):
            self.frame("OPEN PALM", hand_at((400, 295), 1.0))
        self.assertGreater(len(self.moves()), n)

    def test_losing_the_hand_clears_click_state(self):
        self.close_fingers_while_drifting()
        self.m.hand_lost()
        self.assertIsNone(self.m.click_lock)
        self.assertEqual(len(self.m.history), 0)
        self.assertFalse(self.m.left_down)


class TestStabilityWithDriver(unittest.TestCase):
    def test_click_snaps_the_real_cursor_before_pressing(self):
        backend = FakeBackend()
        driver = ht.CursorDriver(backend)
        m = ht.MouseController(backend, driver)
        t = 10.0
        for _ in range(10):
            m.update("POINT", hand_at(), 640, 480, t)
            driver.step(0.033)
            t += 0.033
        before = driver.target
        x = 265
        for spread in (0.6, 0.3, 0.05, 0.05):
            x += 4
            m.update("POINT", hand_at((x, 295), spread), 640, 480, t)
            driver.step(0.033)
            t += 0.033
        names = backend.names()
        i = names.index("down")
        self.assertEqual(backend.calls[i - 1], ("move", int(before[0]), int(before[1])))


class TestMouseModeSettings(unittest.TestCase):
    def test_stability_settings_reach_the_controller(self):
        backend = FakeBackend()
        mode = ht.MouseMode(backend_factory=lambda: backend, still_radius=12, lookback=0.3)
        mode.enable()
        self.assertEqual(mode.controller.leash.radius, 12)
        self.assertEqual(mode.controller.lookback, 0.3)


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


class TestMouseControllerWithDriver(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.driver = ht.CursorDriver(self.backend)
        self.m = ht.MouseController(self.backend, self.driver)

    def test_controller_sets_a_target_and_the_driver_does_the_moving(self):
        self.m.update("OPEN PALM", mouse_hand(), 640, 480, 10.0)
        self.assertEqual(self.backend.count("move"), 0)
        self.assertIsNotNone(self.driver.target)
        self.driver.step(0.01)
        self.assertEqual(self.backend.count("move"), 1)

    def test_target_stays_inside_the_screen(self):
        for i, dx in enumerate((-600, 0, 600)):
            self.m.update("OPEN PALM", shifted(mouse_hand(), dx, -300), 640, 480, 10.0 + i)
            x, y = self.driver.target
            self.assertTrue(ht.SCREEN_EDGE_PAD <= x <= 1920 - 1 - ht.SCREEN_EDGE_PAD)
            self.assertTrue(ht.SCREEN_EDGE_PAD <= y <= 1080 - 1 - ht.SCREEN_EDGE_PAD)

    def test_clicks_still_go_straight_to_the_backend(self):
        for i in range(4):
            self.m.update("POINT", left_pinch_hand(), 640, 480, 10.0 + i * 0.03)
        self.assertEqual(self.backend.count("down"), 1)

    def test_fist_does_not_change_the_target(self):
        self.m.update("OPEN PALM", mouse_hand(), 640, 480, 10.0)
        before = self.driver.target
        for i in range(5):
            self.m.update("FIST", shifted(mouse_hand(), 150, 80), 640, 480, 10.1 + i * 0.03)
        self.assertEqual(self.driver.target, before)

    def test_moving_the_hand_moves_the_target_the_same_way(self):
        self.m.update("OPEN PALM", mouse_hand(), 640, 480, 10.0)
        start_x = self.driver.target[0]
        for i in range(30):
            self.m.update("OPEN PALM", shifted(mouse_hand(), 80, 0), 640, 480, 10.1 + i * 0.033)
        self.assertGreater(self.driver.target[0], start_x + 300)


class TestMouseModeWithDriver(unittest.TestCase):
    def wait_for(self, condition, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline and not condition():
            time.sleep(0.005)
        return condition()

    def make(self, backend=None, tau=ht.CURSOR_FOLLOW_TAU):
        backend = backend or FakeBackend()
        return ht.MouseMode(backend_factory=lambda: backend, use_driver=True, tau=tau), backend

    def test_enable_starts_the_driver_and_disable_stops_it(self):
        mode, _ = self.make()
        mode.enable()
        self.assertTrue(mode.driver.thread.is_alive())
        mode.disable()
        self.assertFalse(mode.driver.thread.is_alive())
        self.assertFalse(mode.on)

    def test_cursor_moves_through_the_thread(self):
        mode, backend = self.make()
        mode.enable()
        mode.update("OPEN PALM", mouse_hand(), 640, 480, 10.0)
        self.assertTrue(self.wait_for(lambda: backend.count("move") > 0))
        mode.disable()

    def test_driver_error_turns_mouse_mode_off(self):
        backend = FakeBackend()

        def boom(x, y):
            raise RuntimeError("failsafe")
        backend.move_to = boom
        mode, _ = self.make(backend)
        mode.enable()
        mode.update("OPEN PALM", mouse_hand(), 640, 480, 10.0)
        self.assertTrue(self.wait_for(lambda: mode.driver.error is not None))
        message = mode.update("OPEN PALM", mouse_hand(), 640, 480, 10.1)
        self.assertIsNotNone(message)
        self.assertFalse(mode.on)

    def test_can_be_enabled_again_after_an_error(self):
        backend = FakeBackend()
        good_move = backend.move_to

        def boom(x, y):
            raise RuntimeError("failsafe")
        backend.move_to = boom
        mode, _ = self.make(backend)
        mode.enable()
        mode.update("OPEN PALM", mouse_hand(), 640, 480, 10.0)
        self.assertTrue(self.wait_for(lambda: mode.driver.error is not None))
        mode.update("OPEN PALM", mouse_hand(), 640, 480, 10.1)
        backend.move_to = good_move
        ok, _ = mode.enable()
        self.assertTrue(ok)
        self.assertIsNone(mode.driver.error)
        self.assertTrue(mode.driver.thread.is_alive())
        mode.disable()

    def test_cursor_lag_setting_reaches_the_driver(self):
        mode, _ = self.make(tau=0.2)
        mode.enable()
        self.assertEqual(mode.driver.tau, 0.2)
        mode.disable()

    def test_button_is_released_when_mode_is_disabled_mid_drag(self):
        mode, backend = self.make()
        mode.enable()
        for i in range(4):
            mode.update("POINT", left_pinch_hand(), 640, 480, 10.0 + i * 0.03)
        self.assertEqual(backend.count("down"), 1)
        mode.disable()
        self.assertEqual(backend.count("up"), 1)


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


class FakeLandmarker:
    """Stands in for MediaPipe's HandLandmarker."""

    def __init__(self, hands=None):
        self.hands = hands          # list of (x, y) in 0..1, or None for "no hand"
        self.timestamps = []
        self.closed = False

    def detect_for_video(self, image, ts):
        self.timestamps.append(ts)
        if self.hands is None:
            return types.SimpleNamespace(hand_landmarks=[])
        lm = [types.SimpleNamespace(x=x, y=y) for x, y in self.hands]
        return types.SimpleNamespace(hand_landmarks=[lm])

    def close(self):
        self.closed = True


class TestHandDetector(unittest.TestCase):
    FRAME = np.zeros((200, 400, 3), dtype=np.uint8)

    def test_landmarks_are_scaled_to_pixels(self):
        det = ht.HandDetector(FakeLandmarker([(0.5, 0.25)] * 21))
        pts = det.detect(self.FRAME, now=det.start_t + 1)
        self.assertEqual(len(pts), 21)
        self.assertAlmostEqual(pts[0][0], 200.0)   # 0.5 * width 400
        self.assertAlmostEqual(pts[0][1], 50.0)    # 0.25 * height 200

    def test_no_hand_returns_none(self):
        det = ht.HandDetector(FakeLandmarker(None))
        self.assertIsNone(det.detect(self.FRAME))

    def test_timestamps_strictly_increase_even_if_the_clock_stalls(self):
        fake = FakeLandmarker(None)
        det = ht.HandDetector(fake)
        for _ in range(5):
            det.detect(self.FRAME, now=det.start_t + 1.0)   # same instant every time
        self.assertEqual(fake.timestamps, sorted(set(fake.timestamps)))
        self.assertEqual(len(fake.timestamps), 5)

    def test_timestamps_follow_the_clock_in_milliseconds(self):
        fake = FakeLandmarker(None)
        det = ht.HandDetector(fake)
        det.detect(self.FRAME, now=det.start_t + 2.5)
        self.assertEqual(fake.timestamps[-1], 2500)

    def test_close_closes_the_landmarker(self):
        fake = FakeLandmarker(None)
        ht.HandDetector(fake).close()
        self.assertTrue(fake.closed)


class TestOpenCamera(unittest.TestCase):
    def test_exits_with_a_hint_when_the_camera_will_not_open(self):
        cap = mock.Mock()
        cap.isOpened.return_value = False
        with mock.patch.object(ht.cv2, "VideoCapture", return_value=cap) as vc:
            with self.assertRaisesRegex(SystemExit, "--camera 1"):
                ht.open_camera(3, 640, 480)
        vc.assert_called_once_with(3)

    def test_requests_the_size_and_a_small_buffer(self):
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 30
        with mock.patch.object(ht.cv2, "VideoCapture", return_value=cap):
            self.assertIs(ht.open_camera(0, 800, 600), cap)
        cap.set.assert_any_call(ht.cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set.assert_any_call(ht.cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set.assert_any_call(ht.cv2.CAP_PROP_BUFFERSIZE, 1)


class FakeLearned:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def classify(self, pts):
        self.calls += 1
        return self.answer


class TestGestureSource(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "m.joblib")

    def tearDown(self):
        self.dir.cleanup()

    def make_model_file(self):
        with open(self.path, "wb") as f:
            f.write(b"x")

    def test_starts_with_the_rules(self):
        src = ht.GestureSource(self.path, loader=lambda p, c: FakeLearned("PEACE"))
        self.assertEqual(src.mode, "rules")
        self.assertEqual(src.classify(build_hand("I", "tucked")), "POINT")

    def test_missing_model_file_stays_on_rules_with_a_hint(self):
        src = ht.GestureSource(self.path, loader=lambda p, c: FakeLearned("PEACE"))
        ok, message = src.toggle()
        self.assertFalse(ok)
        self.assertIn("train_gestures.py", message)
        self.assertEqual(src.mode, "rules")

    def test_toggle_switches_to_the_learned_model_and_back(self):
        self.make_model_file()
        learned = FakeLearned("PEACE")
        src = ht.GestureSource(self.path, loader=lambda p, c: learned)
        self.assertTrue(src.toggle()[0])
        self.assertEqual(src.mode, "learned")
        self.assertEqual(src.classify(build_hand("I", "tucked")), "PEACE")
        self.assertEqual(learned.calls, 1)
        src.toggle()
        self.assertEqual(src.mode, "rules")
        self.assertEqual(src.classify(build_hand("I", "tucked")), "POINT")

    def test_model_is_loaded_only_once(self):
        self.make_model_file()
        loads = []
        src = ht.GestureSource(self.path, loader=lambda p, c: loads.append(p) or FakeLearned("FIST"))
        for _ in range(4):
            src.toggle()
        self.assertEqual(len(loads), 1)

    def test_loader_receives_the_path_and_confidence(self):
        self.make_model_file()
        seen = []
        src = ht.GestureSource(self.path, loader=lambda p, c: seen.append((p, c)) or FakeLearned("FIST"),
                               min_confidence=0.8)
        src.use_learned()
        self.assertEqual(seen, [(self.path, 0.8)])

    def test_a_broken_model_file_keeps_the_rules_and_reports_why(self):
        self.make_model_file()

        def broken(path, conf):
            raise ValueError("bad file")
        src = ht.GestureSource(self.path, loader=broken)
        ok, message = src.use_learned()
        self.assertFalse(ok)
        self.assertIn("bad file", message)
        self.assertEqual(src.mode, "rules")
        self.assertEqual(src.classify(build_hand("", "tucked")), "FIST")

    def test_works_with_a_real_trained_model(self):
        import gesture_model
        rng = np.random.default_rng(0)
        X, y = [], []
        for label, ext, thumb in [("FIST", "", "tucked"), ("OPEN PALM", "IMRP", "out"),
                                  ("POINT", "I", "tucked")]:
            for _ in range(40):
                a = np.array(build_hand(ext, thumb)) + rng.normal(0, 1.5, (21, 2))
                X.append(a)
                y.append(label)
        norm = gesture_model.normalize_batch(np.array(X))
        model = gesture_model.train_model(gesture_model.features_from_normalized(norm), y)
        gesture_model.save_model(model, list(model.classes_), self.path)
        src = ht.GestureSource(self.path)
        self.assertTrue(src.use_learned()[0])
        self.assertEqual(src.classify(build_hand("I", "tucked")), "POINT")
        self.assertEqual(src.classify(build_hand("IMRP", "out")), "OPEN PALM")



if __name__ == "__main__":
    unittest.main(verbosity=2)
