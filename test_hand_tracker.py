"""
Tests for hand_tracker.py. No camera or model file needed.

Run from the project folder (same folder as hand_tracker.py):
    python -m unittest -v test_hand_tracker.py
or run everything named test_*.py:
    python -m unittest discover -v
"""
import math
import os
import tempfile
import threading
import time
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
