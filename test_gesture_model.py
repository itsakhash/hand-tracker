"""
Tests for gesture_model.py. No camera needed; needs numpy and scikit-learn.

Run from the project folder (same folder as gesture_model.py):
    python -m unittest -v test_gesture_model.py
"""
import math
import os
import tempfile
import unittest

import numpy as np

import gesture_model as gm


# --------------------------------------------------------------------------
# Synthetic hands (image coordinates, palm length ~105 px)
# --------------------------------------------------------------------------
WR = (100.0, 300.0)
MCP = {"I": (85, 200), "M": (100, 195), "R": (115, 200), "P": (130, 210)}
IDX = {"I": (5, 6, 7, 8), "M": (9, 10, 11, 12), "R": (13, 14, 15, 16), "P": (17, 18, 19, 20)}
POSES = {
    "FIST": ("", "tucked"),
    "OPEN PALM": ("IMRP", "out"),
    "POINT": ("I", "tucked"),
    "PEACE": ("IM", "tucked"),
    "THUMBS UP": ("", "up"),
}


def build_hand(extended, thumb):
    pts = [(0.0, 0.0)] * 21
    pts[0] = WR
    for f, (mcp_i, pip_i, dip_i, tip_i) in IDX.items():
        x, y = MCP[f]
        pts[mcp_i] = (x, y)
        if f in extended:
            pts[pip_i], pts[dip_i], pts[tip_i] = (x, y - 60), (x, y - 85), (x, y - 105)
        else:
            pts[pip_i], pts[dip_i], pts[tip_i] = (x, y - 25), (x, y - 5), (x, y + 15)
    if thumb == "up":
        pts[1], pts[2], pts[3], pts[4] = (82, 285), (78, 245), (78, 190), (78, 120)
    elif thumb == "out":
        pts[1], pts[2], pts[3], pts[4] = (80, 285), (55, 265), (35, 250), (15, 235)
    else:
        pts[1], pts[2], pts[3], pts[4] = (85, 285), (95, 260), (105, 240), (110, 230)
    return pts


def variant(pts, rng, tilt=0.0, mirror=False):
    """Move, scale, tilt (degrees) and jitter a hand like a person would."""
    a = np.array(pts, dtype=float)
    if mirror:
        a[:, 0] = -a[:, 0]
    a -= a[0]
    ang = math.radians(tilt + rng.uniform(-6, 6))
    c, s = math.cos(ang), math.sin(ang)
    a = a @ np.array([[c, s], [-s, c]])
    a *= rng.uniform(0.6, 1.6)
    a += rng.normal(0, 1.5, a.shape)
    a += rng.uniform(100, 400, 2)
    return [tuple(p) for p in a]


def make_rows(labels, takes=3, per_take=25, seed=0, mirror=False, run="t"):
    rng = np.random.default_rng(seed)
    rows = []
    for label in labels:
        ext, thumb = POSES[label]
        base = build_hand(ext, thumb)
        for take in range(takes):
            tilt = rng.uniform(-10, 10)
            for _ in range(per_take):
                rows.append((label, f"{run}-{label}-{take}", variant(base, rng, tilt, mirror)))
    return rows


def write_dataset(path, labels, **kw):
    gm.append_rows(path, make_rows(labels, **kw))


# --------------------------------------------------------------------------
class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.hand = build_hand("IM", "tucked")

    def test_shape_is_forty_features_per_hand(self):
        self.assertEqual(gm.landmarks_to_features(self.hand).shape, (1, gm.FEATURE_DIM))
        self.assertEqual(gm.FEATURE_DIM, 40)

    def test_batch_input(self):
        batch = np.array([self.hand, self.hand, self.hand])
        self.assertEqual(gm.landmarks_to_features(batch).shape, (3, 40))

    def test_wrist_is_origin_and_palm_length_is_one(self):
        norm = gm.normalize_batch(self.hand)[0]
        np.testing.assert_allclose(norm[gm.WRIST], [0, 0], atol=1e-9)
        self.assertAlmostEqual(float(np.hypot(*norm[gm.MIDDLE_MCP])), 1.0)

    def test_translation_does_not_matter(self):
        moved = [(x + 123, y - 77) for x, y in self.hand]
        np.testing.assert_allclose(gm.landmarks_to_features(self.hand),
                                   gm.landmarks_to_features(moved), atol=1e-9)

    def test_scale_does_not_matter(self):
        big = [(x * 2.5, y * 2.5) for x, y in self.hand]
        np.testing.assert_allclose(gm.landmarks_to_features(self.hand),
                                   gm.landmarks_to_features(big), atol=1e-9)

    def test_orientation_does_matter(self):
        flipped = [(x, -y) for x, y in self.hand]   # upside down
        self.assertFalse(np.allclose(gm.landmarks_to_features(self.hand),
                                     gm.landmarks_to_features(flipped)))

    def test_degenerate_hand_does_not_divide_by_zero(self):
        feats = gm.landmarks_to_features([(5.0, 5.0)] * 21)
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_wrong_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            gm.landmarks_to_features([(0, 0)] * 20)


class TestDatasetCsv(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "sub", "g.csv")

    def tearDown(self):
        self.dir.cleanup()

    def test_round_trip(self):
        rows = make_rows(["FIST", "POINT"], takes=2, per_take=3)
        self.assertEqual(gm.append_rows(self.path, rows), 12)
        ds = gm.load_dataset(self.path)
        self.assertEqual(len(ds), 12)
        self.assertEqual(ds.labels[0], "FIST")
        self.assertEqual(ds.sessions[0], rows[0][1])
        np.testing.assert_allclose(ds.points[0], np.array(rows[0][2]), atol=1e-3)

    def test_creates_missing_folder(self):
        gm.append_rows(self.path, make_rows(["FIST"], takes=1, per_take=1))
        self.assertTrue(os.path.exists(self.path))

    def test_header_is_written_once(self):
        gm.append_rows(self.path, make_rows(["FIST"], takes=1, per_take=2))
        gm.append_rows(self.path, make_rows(["POINT"], takes=1, per_take=2))
        with open(self.path) as f:
            self.assertEqual(sum(1 for line in f if line.startswith("label,")), 1)
        self.assertEqual(len(gm.load_dataset(self.path)), 4)

    def test_nothing_to_write_creates_nothing(self):
        self.assertEqual(gm.append_rows(self.path, []), 0)
        self.assertFalse(os.path.exists(self.path))

    def test_wrong_landmark_count_is_rejected(self):
        with self.assertRaises(ValueError):
            gm.append_rows(self.path, [("FIST", "s", [(0, 0)] * 5)])

    def test_summary_counts_samples_and_sessions(self):
        gm.append_rows(self.path, make_rows(["FIST"], takes=3, per_take=4))
        self.assertEqual(gm.load_dataset(self.path).summary(), {"FIST": (12, 3)})

    def test_bad_header_is_rejected(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w") as f:
            f.write("a,b,c\n1,2,3\n")
        with self.assertRaises(ValueError):
            gm.load_dataset(self.path)

    def test_short_row_names_the_line(self):
        gm.append_rows(self.path, make_rows(["FIST"], takes=1, per_take=1))
        with open(self.path, "a") as f:
            f.write("FIST,s,1,2\n")
        with self.assertRaisesRegex(ValueError, "line 3"):
            gm.load_dataset(self.path)

    def test_non_numeric_value_is_rejected(self):
        gm.append_rows(self.path, make_rows(["FIST"], takes=1, per_take=1))
        with open(self.path, "a") as f:
            f.write("FIST,s," + ",".join(["x"] * 42) + "\n")
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            gm.load_dataset(self.path)

    def test_header_only_file_is_empty_error(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w") as f:
            f.write(",".join(gm.csv_header()) + "\n")
        with self.assertRaisesRegex(ValueError, "no samples"):
            gm.load_dataset(self.path)


class TestSplitBySession(unittest.TestCase):
    def setUp(self):
        rows = make_rows(["FIST", "POINT"], takes=4, per_take=5)
        self.labels = np.array([r[0] for r in rows], dtype=object)
        self.sessions = np.array([r[1] for r in rows], dtype=object)

    def test_no_session_appears_on_both_sides(self):
        train, test, _ = gm.split_by_session(self.labels, self.sessions)
        self.assertFalse(set(self.sessions[train]) & set(self.sessions[test]))

    def test_every_sample_is_used_exactly_once(self):
        train, test, _ = gm.split_by_session(self.labels, self.sessions)
        self.assertEqual(sorted(train.tolist() + test.tolist()), list(range(len(self.labels))))

    def test_every_label_gets_test_data_and_keeps_train_data(self):
        train, test, _ = gm.split_by_session(self.labels, self.sessions)
        for label in ("FIST", "POINT"):
            self.assertIn(label, self.labels[test])
            self.assertIn(label, self.labels[train])

    def test_fraction_is_respected(self):
        _, test, _ = gm.split_by_session(self.labels, self.sessions, test_fraction=0.5)
        self.assertEqual(len(set(self.sessions[test])), 4)   # 2 of 4 takes, for each of 2 labels

    def test_never_holds_out_every_session(self):
        train, _, _ = gm.split_by_session(self.labels, self.sessions, test_fraction=1.0)
        self.assertEqual(len(set(self.sessions[train])), 2)  # one take per label stays

    def test_same_seed_same_split_and_other_seed_can_differ(self):
        a = gm.split_by_session(self.labels, self.sessions, seed=1)[1]
        b = gm.split_by_session(self.labels, self.sessions, seed=1)[1]
        np.testing.assert_array_equal(a, b)
        others = [tuple(gm.split_by_session(self.labels, self.sessions, seed=s)[1]) for s in range(8)]
        self.assertGreater(len(set(others)), 1)

    def test_single_session_label_goes_to_training_with_a_warning(self):
        labels = np.array(["FIST"] * 5 + ["POINT"] * 5, dtype=object)
        sessions = np.array(["a"] * 5 + ["b", "b", "c", "c", "c"], dtype=object)
        train, test, warnings = gm.split_by_session(labels, sessions)
        self.assertTrue(all(labels[i] == "POINT" for i in test))
        self.assertEqual(sum(labels[train] == "FIST"), 5)
        self.assertEqual(len(warnings), 1)
        self.assertIn("FIST", warnings[0])


class TestAugment(unittest.TestCase):
    def setUp(self):
        hands = [build_hand("IM", "tucked"), build_hand("", "up")]
        self.norm = gm.normalize_batch(np.array(hands))
        self.labels = np.array(["PEACE", "THUMBS UP"], dtype=object)

    def test_output_size_and_labels(self):
        X, y = gm.augment(self.norm, self.labels, copies=3)
        self.assertEqual(X.shape, (2 * (2 + 3), 40))
        self.assertEqual(len(y), 10)
        self.assertEqual(sorted(set(y.tolist())), ["PEACE", "THUMBS UP"])

    def test_originals_come_first_unchanged(self):
        X, _ = gm.augment(self.norm, self.labels)
        np.testing.assert_allclose(X[:2], gm.features_from_normalized(self.norm))

    def test_mirror_copy_flips_x_only(self):
        X, _ = gm.augment(self.norm, self.labels)
        orig = X[:2].reshape(2, 20, 2)
        mirrored = X[2:4].reshape(2, 20, 2)
        np.testing.assert_allclose(mirrored[..., 0], -orig[..., 0])
        np.testing.assert_allclose(mirrored[..., 1], orig[..., 1])

    def test_random_copies_differ_but_stay_close(self):
        X, _ = gm.augment(self.norm, self.labels, copies=1)
        extra = X[4:6]
        base = np.abs(X[:2])
        self.assertFalse(np.allclose(extra, X[:2]))
        self.assertTrue(np.all(np.abs(np.abs(extra) - base) < 0.8))

    def test_seed_makes_it_repeatable(self):
        a, _ = gm.augment(self.norm, self.labels, seed=5)
        b, _ = gm.augment(self.norm, self.labels, seed=5)
        c, _ = gm.augment(self.norm, self.labels, seed=6)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_no_copies_means_only_originals_and_mirrors(self):
        X, y = gm.augment(self.norm, self.labels, copies=0)
        self.assertEqual(len(X), 4)


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        self.assertAlmostEqual(gm.accuracy(["a", "b", "a", "a"], ["a", "b", "b", "a"]), 0.75)
        self.assertEqual(gm.accuracy([], []), 0.0)

    def test_confusion_matrix_rows_are_truth(self):
        cm = gm.confusion_matrix(["a", "a", "b", "b", "b"], ["a", "b", "b", "b", "a"], ["a", "b"])
        np.testing.assert_array_equal(cm, [[1, 1], [1, 2]])

    def test_confusion_ignores_unknown_labels(self):
        cm = gm.confusion_matrix(["a", "a"], ["a", "zzz"], ["a", "b"])
        self.assertEqual(cm.sum(), 1)

    def test_format_mentions_every_class(self):
        text = gm.format_confusion(np.array([[3, 0], [1, 2]]), ["FIST", "POINT"])
        self.assertIn("FIST", text)
        self.assertIn("POINT", text)


class FakeModel:
    def __init__(self, probs, classes=("FIST", "POINT")):
        self.probs, self.classes_ = np.array([probs]), np.array(classes)
        self.seen = None

    def predict_proba(self, X):
        self.seen = X
        return self.probs


class TestLearnedClassifier(unittest.TestCase):
    def test_confident_prediction_returns_label(self):
        clf = gm.LearnedGestureClassifier(FakeModel([0.1, 0.9]), min_confidence=0.6)
        self.assertEqual(clf.classify(build_hand("I", "tucked")), "POINT")

    def test_unsure_prediction_returns_unknown(self):
        clf = gm.LearnedGestureClassifier(FakeModel([0.45, 0.55]), min_confidence=0.6)
        self.assertEqual(clf.classify(build_hand("I", "tucked")), "UNKNOWN")

    def test_model_receives_the_normalized_features(self):
        model = FakeModel([0.9, 0.1])
        gm.LearnedGestureClassifier(model).classify(build_hand("I", "tucked"))
        np.testing.assert_allclose(model.seen, gm.landmarks_to_features(build_hand("I", "tucked")))


class TestTrainingAndModelFiles(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.data = os.path.join(self.dir.name, "d.csv")
        self.out = os.path.join(self.dir.name, "m", "model.joblib")

    def tearDown(self):
        self.dir.cleanup()

    def quiet(self):
        self.lines = []
        return self.lines.append

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            gm.make_classifier("magic")

    def test_single_class_is_rejected(self):
        with self.assertRaises(ValueError):
            gm.train_model(np.zeros((4, 40)), ["FIST"] * 4)

    def test_every_model_kind_learns_unseen_takes(self):
        for kind in gm.MODEL_KINDS:
            with self.subTest(kind=kind):
                write_dataset(self.data, list(POSES), takes=3, per_take=20, seed=1)
                result = gm.train_and_report(self.data, self.out, kind=kind, log=self.quiet())
                self.assertGreater(result["holdout_accuracy"], 0.9, kind)
                os.remove(self.data)

    def test_saved_model_loads_and_predicts(self):
        write_dataset(self.data, list(POSES), seed=2)
        gm.train_and_report(self.data, self.out, log=self.quiet())
        clf = gm.LearnedGestureClassifier.from_file(self.out)
        rng = np.random.default_rng(99)
        for label, (ext, thumb) in POSES.items():
            self.assertEqual(clf.classify(variant(build_hand(ext, thumb), rng)), label)

    def test_model_file_has_metadata(self):
        write_dataset(self.data, ["FIST", "POINT"], seed=3)
        gm.train_and_report(self.data, self.out, log=self.quiet())
        info = gm.load_model(self.out)
        self.assertEqual(info["classes"], ["FIST", "POINT"])
        self.assertEqual(info["feature_dim"], 40)
        self.assertIsNotNone(info["holdout_accuracy"])

    def test_left_hand_works_after_training_on_right_hand_only(self):
        write_dataset(self.data, list(POSES), seed=4)
        gm.train_and_report(self.data, self.out, log=self.quiet())
        clf = gm.LearnedGestureClassifier.from_file(self.out)
        rng = np.random.default_rng(7)
        hits = 0
        total = 0
        for label, (ext, thumb) in POSES.items():
            for _ in range(10):
                total += 1
                hits += clf.classify(variant(build_hand(ext, thumb), rng, mirror=True)) == label
        self.assertGreater(hits / total, 0.9)

    def test_baseline_is_scored_on_the_same_held_out_data(self):
        write_dataset(self.data, ["FIST", "POINT"], seed=5)
        seen = []
        result = gm.train_and_report(self.data, self.out, baseline=lambda p: seen.append(p) or "FIST",
                                     log=self.quiet())
        self.assertTrue(seen)
        self.assertEqual(len(seen[0]), 21)
        self.assertAlmostEqual(result["baseline_accuracy"], 0.5, places=1)

    def test_single_take_gestures_warn_and_skip_the_honest_test(self):
        write_dataset(self.data, ["FIST", "POINT"], takes=1, seed=6)
        log = self.quiet()
        result = gm.train_and_report(self.data, self.out, log=log)
        self.assertIsNone(result["holdout_accuracy"])
        self.assertTrue(result["warnings"])
        self.assertTrue(os.path.exists(self.out))
        self.assertTrue(any("at least 2" in line for line in self.lines))

    def test_one_gesture_cannot_train(self):
        write_dataset(self.data, ["FIST"], seed=7)
        with self.assertRaises(ValueError):
            gm.train_and_report(self.data, self.out, log=self.quiet())

    def test_garbage_model_file_is_rejected(self):
        import joblib
        os.makedirs(os.path.dirname(self.out))
        joblib.dump({"hello": 1}, self.out)
        with self.assertRaisesRegex(ValueError, "not a gesture model"):
            gm.load_model(self.out)

    def test_old_feature_layout_is_rejected(self):
        import joblib
        os.makedirs(os.path.dirname(self.out))
        joblib.dump({"format": gm.MODEL_FORMAT, "model": None, "classes": [], "feature_dim": 12}, self.out)
        with self.assertRaisesRegex(ValueError, "retrain"):
            gm.load_model(self.out)


class TestSessionRecorder(unittest.TestCase):
    HAND = build_hand("I", "tucked")

    def make(self, **kw):
        kw.setdefault("countdown", 3.0)
        kw.setdefault("take_seconds", 10.0)
        kw.setdefault("sample_interval", 0.1)
        return gm.SessionRecorder(["FIST", "OPEN PALM", "POINT"], run_id="r", **kw)

    def test_needs_labels(self):
        with self.assertRaises(ValueError):
            gm.SessionRecorder([])

    def test_starts_idle_and_stores_nothing(self):
        rec = self.make()
        self.assertEqual(rec.state(0), "idle")
        self.assertFalse(rec.add(self.HAND, 0))

    def test_countdown_then_recording(self):
        rec = self.make()
        rec.toggle(0)
        self.assertEqual(rec.state(1), "countdown")
        self.assertFalse(rec.add(self.HAND, 1))
        self.assertEqual(rec.state(3.0), "recording")
        self.assertTrue(rec.add(self.HAND, 3.0))

    def test_seconds_left(self):
        rec = self.make()
        rec.toggle(0)
        self.assertAlmostEqual(rec.seconds_left(1), 2.0)
        self.assertAlmostEqual(rec.seconds_left(5), 8.0)

    def test_samples_are_rate_limited(self):
        rec = self.make()
        rec.toggle(0)
        stored = sum(rec.add(self.HAND, 3.0 + i * 0.01) for i in range(100))   # 1 s at 100 fps
        self.assertIn(stored, (10, 11))

    def test_take_stops_automatically(self):
        rec = self.make()
        rec.toggle(0)
        self.assertTrue(rec.add(self.HAND, 12.9))
        self.assertEqual(rec.state(13.0), "idle")
        self.assertFalse(rec.add(self.HAND, 13.5))

    def test_toggle_stops_a_take_in_progress(self):
        rec = self.make()
        rec.toggle(0)
        rec.add(self.HAND, 4)
        rec.toggle(5)
        self.assertEqual(rec.state(5), "idle")
        self.assertEqual(len(rec.rows), 1)

    def test_stopping_during_countdown_records_nothing(self):
        rec = self.make()
        rec.toggle(0)
        rec.toggle(1)
        self.assertFalse(rec.add(self.HAND, 5))

    def test_each_take_gets_its_own_session(self):
        rec = self.make()
        for start in (0, 20):
            rec.toggle(start)
            rec.add(self.HAND, start + 4)
            rec.toggle(start + 5)
        sessions = [r[1] for r in rec.rows]
        self.assertEqual(sessions, ["r-FIST-1", "r-FIST-2"])

    def test_session_ids_have_no_spaces(self):
        rec = self.make()
        rec.select(1)
        rec.toggle(0)
        rec.add(self.HAND, 4)
        self.assertEqual(rec.rows[0][1], "r-OPEN_PALM-1")
        self.assertEqual(rec.rows[0][0], "OPEN PALM")

    def test_cannot_change_gesture_during_a_take(self):
        rec = self.make()
        rec.toggle(0)
        self.assertFalse(rec.select(2))
        self.assertEqual(rec.label, "FIST")
        rec.toggle(1)
        self.assertTrue(rec.select(2))
        self.assertEqual(rec.label, "POINT")
        self.assertFalse(rec.select(9))

    def test_undo_removes_only_the_last_take(self):
        rec = self.make()
        for start in (0, 20):
            rec.toggle(start)
            for i in range(5):
                rec.add(self.HAND, start + 4 + i)
            rec.toggle(start + 10)
        self.assertEqual(rec.undo_last_take(), 5)
        self.assertEqual(len(rec.rows), 5)
        self.assertEqual({r[1] for r in rec.rows}, {"r-FIST-1"})
        self.assertEqual(rec.undo_last_take(), 0)

    def test_counts_and_drain(self):
        rec = self.make()
        rec.toggle(0)
        rec.add(self.HAND, 4)
        rec.add(self.HAND, 4.5)
        self.assertEqual(rec.counts(), {"FIST": 2})
        rows = rec.drain()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rec.rows, [])
        self.assertEqual(rec.counts(), {})

    def test_recorded_rows_save_straight_to_csv(self):
        rec = self.make()
        rec.toggle(0)
        rec.add(self.HAND, 4)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.csv")
            self.assertEqual(gm.append_rows(path, rec.drain()), 1)
            self.assertEqual(gm.load_dataset(path).labels[0], "FIST")


if __name__ == "__main__":
    unittest.main(verbosity=2)
