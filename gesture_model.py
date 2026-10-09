"""
Train-your-own gesture classifier: data, features, training, and prediction.

This file needs only numpy to import. scikit-learn and joblib are loaded when you
actually train or load a model, so the rest of the project runs without them.

The pipeline:
  1. collect_gestures.py records labelled hand landmarks into datasets/gestures.csv
     (every recording "take" gets its own session id).
  2. train_gestures.py loads that CSV, holds out WHOLE SESSIONS for testing, trains a
     classifier on the rest, prints accuracy and a confusion matrix, then saves
     models/gesture_model.joblib.
  3. hand_tracker.py loads the model (press g) and uses LearnedGestureClassifier.

Why split by session? Neighboring video frames are almost identical. If frames from
one take land in both the training and test sets, the test is really checking
memorization and the score is inflated. Holding out whole takes is an honest test.

Only load .joblib models that you trained yourself: joblib files are pickles, and
loading someone else's can run their code.
"""
import csv
import os
import time

import numpy as np

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9
FEATURE_DIM = (NUM_LANDMARKS - 1) * 2   # the wrist is always (0, 0), so it is dropped
MODEL_FORMAT = 1
MODEL_KINDS = ("forest", "knn", "mlp")

# Gesture names the app reacts to. Your labels must use these exact spellings
# for drawing and mouse mode to respond to the learned gestures.
DEFAULT_LABELS = ["OPEN PALM", "FIST", "POINT", "PEACE", "THUMBS UP", "UNKNOWN"]


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def normalize_batch(points):
    """(N, 21, 2) pixel landmarks -> landmarks centered on the wrist and divided by
    palm length (wrist to middle-finger base), so distance from the camera and
    position in the frame stop mattering. Orientation is kept on purpose: the
    difference between THUMBS UP and a thumb pointing sideways matters."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 2:
        pts = pts[None]
    if pts.shape[1:] != (NUM_LANDMARKS, 2):
        raise ValueError(f"expected landmarks of shape (21, 2), got {pts.shape[1:]}")
    centered = pts - pts[:, WRIST:WRIST + 1, :]
    palm = np.hypot(centered[:, MIDDLE_MCP, 0], centered[:, MIDDLE_MCP, 1])
    palm = np.where(palm < 1e-6, 1.0, palm)
    return centered / palm[:, None, None]


def features_from_normalized(normalized):
    """(N, 21, 2) normalized landmarks -> (N, 40) feature rows."""
    return np.asarray(normalized, dtype=float)[:, 1:, :].reshape(len(normalized), FEATURE_DIM)


def landmarks_to_features(points):
    """Landmarks (21 (x, y) pairs, or an (N, 21, 2) array) -> (N, 40) feature rows."""
    return features_from_normalized(normalize_batch(points))


# --------------------------------------------------------------------------
# Dataset (CSV)
# --------------------------------------------------------------------------
def csv_header():
    cols = ["label", "session"]
    for i in range(NUM_LANDMARKS):
        cols += [f"x{i}", f"y{i}"]
    return cols


def append_rows(path, rows):
    """Append (label, session, pts) rows to a CSV, creating it (and its folder) if needed.
    Returns how many rows were written."""
    rows = list(rows)
    if not rows:
        return 0
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(csv_header())
        for label, session, pts in rows:
            flat = [round(float(v), 3) for p in pts for v in p]
            if len(flat) != NUM_LANDMARKS * 2:
                raise ValueError("each sample needs exactly 21 landmarks")
            writer.writerow([label, session] + flat)
    return len(rows)


class Dataset:
    def __init__(self, labels, sessions, points):
        self.labels = np.asarray(labels, dtype=object)
        self.sessions = np.asarray(sessions, dtype=object)
        self.points = np.asarray(points, dtype=float).reshape(-1, NUM_LANDMARKS, 2)

    def __len__(self):
        return len(self.labels)

    def classes(self):
        return sorted(set(self.labels.tolist()))

    def summary(self):
        """{label: (samples, sessions)}"""
        out = {}
        for label in self.classes():
            mask = self.labels == label
            out[label] = (int(mask.sum()), len(set(self.sessions[mask].tolist())))
        return out


def load_dataset(path):
    """Read a CSV written by append_rows. Raises ValueError on bad or empty data."""
    labels, sessions, points = [], [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != csv_header():
            raise ValueError(f"{path} does not look like a gesture dataset (bad header)")
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != 2 + NUM_LANDMARKS * 2:
                raise ValueError(f"{path} line {line_no}: expected {2 + NUM_LANDMARKS * 2} columns, got {len(row)}")
            try:
                values = [float(v) for v in row[2:]]
            except ValueError:
                raise ValueError(f"{path} line {line_no}: non-numeric landmark value") from None
            labels.append(row[0])
            sessions.append(row[1])
            points.append(values)
    if not labels:
        raise ValueError(f"{path} has no samples")
    return Dataset(labels, sessions, points)


# --------------------------------------------------------------------------
# Splitting and augmentation
# --------------------------------------------------------------------------
def split_by_session(labels, sessions, test_fraction=0.3, seed=0):
    """Hold out whole sessions for testing. Returns (train_idx, test_idx, warnings).

    For each label with at least two sessions, about test_fraction of its sessions
    (at least one, never all) go to the test set. A label with a single session
    can't be split without leaking, so it goes entirely to training and a warning
    is returned.
    """
    labels = np.asarray(labels, dtype=object)
    sessions = np.asarray(sessions, dtype=object)
    rng = np.random.default_rng(seed)
    train, test, warnings = [], [], []
    for label in sorted(set(labels.tolist())):
        label_idx = np.where(labels == label)[0]
        ids = sorted(set(sessions[label_idx].tolist()))
        if len(ids) < 2:
            warnings.append(f"'{label}' has only {len(ids)} session; record at least 2 takes "
                            "so it can be tested honestly (all of it is used for training)")
            train.extend(label_idx.tolist())
            continue
        n_test = int(min(max(1, round(test_fraction * len(ids))), len(ids) - 1))
        held = set(rng.permutation(ids)[:n_test].tolist())
        for i in label_idx:
            (test if sessions[i] in held else train).append(int(i))
    return np.array(sorted(train), dtype=int), np.array(sorted(test), dtype=int), warnings


def augment(normalized, labels, copies=2, max_rotation=12.0, noise=0.015, seed=0):
    """Make more training data from normalized landmarks (N, 21, 2).

    Returns (features, labels) containing every original, a mirrored copy of every
    original (so a model trained on a right hand also understands a left hand), and
    `copies` randomly rotated / jittered / sometimes-mirrored versions of each.
    """
    normalized = np.asarray(normalized, dtype=float)
    labels = np.asarray(labels, dtype=object)
    rng = np.random.default_rng(seed)
    mirror = normalized * np.array([-1.0, 1.0])
    batches = [normalized, mirror]
    out_labels = [labels, labels]
    for _ in range(max(0, copies)):
        angles = np.radians(rng.uniform(-max_rotation, max_rotation, len(normalized)))
        c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
        x, y = normalized[:, :, 0], normalized[:, :, 1]
        rotated = np.stack([x * c - y * s, x * s + y * c], axis=2)
        flip = rng.random(len(normalized)) < 0.5
        rotated[flip] *= np.array([-1.0, 1.0])
        rotated += rng.normal(0.0, noise, rotated.shape)
        rotated[:, WRIST, :] = 0.0
        batches.append(rotated)
        out_labels.append(labels)
    all_norm = np.concatenate(batches)
    return features_from_normalized(all_norm), np.concatenate(out_labels)


# --------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------
def make_classifier(kind="forest", seed=0):
    """Build an untrained scikit-learn classifier."""
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown model kind '{kind}', choose from {MODEL_KINDS}")
    if kind == "forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    if kind == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=seed))


def train_model(features, labels, kind="forest", seed=0):
    labels = np.asarray(labels, dtype=object)
    if len(set(labels.tolist())) < 2:
        raise ValueError("need at least 2 different gestures to train")
    clf = make_classifier(kind, seed)
    clf.fit(np.asarray(features, dtype=float), labels.astype(str))
    return clf


def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true, y_pred, classes):
    """Rows = true label, columns = predicted label (predictions outside `classes` are ignored)."""
    index = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            cm[index[t], index[p]] += 1
    return cm


def format_confusion(cm, classes):
    width = max(len(c) for c in classes) + 2
    head = " " * width + " ".join(f"{c[:8]:>8}" for c in classes)
    lines = ["rows = true gesture, columns = predicted", head]
    for i, c in enumerate(classes):
        lines.append(f"{c:<{width}}" + " ".join(f"{v:>8d}" for v in cm[i]))
    return "\n".join(lines)


def save_model(model, classes, path, kind="forest", holdout_accuracy=None):
    import joblib
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    joblib.dump({"format": MODEL_FORMAT, "model": model, "classes": list(classes),
                 "feature_dim": FEATURE_DIM, "kind": kind,
                 "holdout_accuracy": holdout_accuracy, "saved_at": time.time()}, path)


def load_model(path):
    """Returns the saved dict. Only open files you trained yourself (they are pickles)."""
    import joblib
    data = joblib.load(path)
    if not isinstance(data, dict) or data.get("format") != MODEL_FORMAT:
        raise ValueError(f"{path} is not a gesture model this version understands")
    if data.get("feature_dim") != FEATURE_DIM:
        raise ValueError(f"{path} was trained with a different feature layout; retrain it")
    return data


def train_and_report(data_path, out_path, kind="forest", test_fraction=0.3, seed=0,
                     baseline=None, log=print):
    """The whole training run. Returns a dict with the numbers (used by train_gestures.py).

    baseline: optional function taking raw landmarks and returning a gesture name,
    used to score the hand-written rules on the same held-out takes.
    """
    ds = load_dataset(data_path)
    log(f"Loaded {len(ds)} samples from {data_path}")
    for label, (n, s) in ds.summary().items():
        log(f"  {label:<12} {n:>6} samples in {s} session(s)")
    classes = ds.classes()
    if len(classes) < 2:
        raise ValueError("need at least 2 different gestures to train")

    train_idx, test_idx, warnings = split_by_session(ds.labels, ds.sessions, test_fraction, seed)
    for w in warnings:
        log(f"WARNING: {w}")

    norm = normalize_batch(ds.points)
    result = {"samples": len(ds), "classes": classes, "holdout_accuracy": None,
              "baseline_accuracy": None, "warnings": warnings, "model_path": out_path}

    if len(test_idx) and len(set(ds.labels[train_idx].tolist())) >= 2:
        X_train, y_train = augment(norm[train_idx], ds.labels[train_idx], seed=seed)
        model = train_model(X_train, y_train, kind, seed)
        pred = model.predict(features_from_normalized(norm[test_idx]))
        y_test = ds.labels[test_idx].astype(str)
        acc = accuracy(y_test, pred)
        result["holdout_accuracy"] = acc
        log(f"\nHeld-out accuracy ({len(test_idx)} samples from unseen takes): {acc * 100:.1f}%")
        log(format_confusion(confusion_matrix(y_test, pred, classes), classes))
        if baseline is not None:
            rule_pred = [baseline([tuple(p) for p in ds.points[i]]) for i in test_idx]
            base = accuracy(y_test, rule_pred)
            result["baseline_accuracy"] = base
            log(f"Hand-written rules on the same samples: {base * 100:.1f}%")
    else:
        log("\nNo held-out takes available, so accuracy can't be measured honestly. "
            "Record each gesture in at least 2 separate takes.")

    X_all, y_all = augment(norm, ds.labels, seed=seed)
    final = train_model(X_all, y_all, kind, seed)
    save_model(final, [str(c) for c in final.classes_], out_path, kind, result["holdout_accuracy"])
    log(f"\nSaved the final model (trained on all takes) to {out_path}")
    return result


# --------------------------------------------------------------------------
# Using a model
# --------------------------------------------------------------------------
class LearnedGestureClassifier:
    """Wraps a trained model: landmarks in, gesture name out ("UNKNOWN" if unsure)."""

    def __init__(self, model, min_confidence=0.6):
        self.model = model
        self.min_confidence = min_confidence

    @classmethod
    def from_file(cls, path, min_confidence=0.6):
        return cls(load_model(path)["model"], min_confidence)

    def classify(self, pts):
        probs = self.model.predict_proba(landmarks_to_features(pts))[0]
        best = int(np.argmax(probs))
        if probs[best] < self.min_confidence:
            return "UNKNOWN"
        return str(self.model.classes_[best])


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------
class SessionRecorder:
    """Handles the timing of recording takes: countdown, recording, auto-stop.

    Every take gets its own session id. Samples are rate-limited (sample_interval)
    so you don't store 30 near-identical frames per second.
    """

    def __init__(self, labels, countdown=3.0, take_seconds=10.0, sample_interval=0.1, run_id=None):
        if not labels:
            raise ValueError("need at least one label")
        self.labels = list(labels)
        self.countdown = countdown
        self.take_seconds = take_seconds
        self.sample_interval = sample_interval
        self.run_id = run_id or time.strftime("%y%m%d-%H%M%S")
        self.selected = 0
        self.rows = []              # unsaved (label, session, pts)
        self.take_counts = {}       # label -> takes started this run
        self._phase = "idle"
        self._phase_start = 0.0
        self._session = None
        self._last_sample = -1e9
        self._last_session = None

    @property
    def label(self):
        return self.labels[self.selected]

    def select(self, index):
        """Choose the label to record. Ignored while a take is in progress."""
        if self._phase == "idle" and 0 <= index < len(self.labels):
            self.selected = index
            return True
        return False

    def toggle(self, now):
        """Start a take (after the countdown), or stop the one in progress."""
        if self._phase == "idle":
            n = self.take_counts.get(self.label, 0) + 1
            self.take_counts[self.label] = n
            self._session = f"{self.run_id}-{self.label.replace(' ', '_')}-{n}"
            self._phase, self._phase_start = "countdown", now
            self._last_sample = -1e9
        else:
            self._phase = "idle"
        return self._phase

    def state(self, now):
        """'idle', 'countdown' or 'recording' (moves to the next phase when time is up)."""
        if self._phase == "countdown" and now - self._phase_start >= self.countdown:
            self._phase, self._phase_start = "recording", self._phase_start + self.countdown
        if self._phase == "recording" and now - self._phase_start >= self.take_seconds:
            self._phase = "idle"
        return self._phase

    def seconds_left(self, now):
        phase = self.state(now)
        if phase == "countdown":
            return max(0.0, self.countdown - (now - self._phase_start))
        if phase == "recording":
            return max(0.0, self.take_seconds - (now - self._phase_start))
        return 0.0

    def add(self, pts, now):
        """Offer one frame of landmarks. Returns True if it was stored."""
        if self.state(now) != "recording":
            return False
        if now - self._last_sample < self.sample_interval:
            return False
        self._last_sample = now
        self.rows.append((self.label, self._session, [tuple(map(float, p)) for p in pts]))
        self._last_session = self._session
        return True

    def counts(self):
        """{label: unsaved sample count}"""
        out = {}
        for label, _, _ in self.rows:
            out[label] = out.get(label, 0) + 1
        return out

    def undo_last_take(self):
        """Throw away the unsaved samples of the most recent take. Returns how many."""
        if self._last_session is None:
            return 0
        before = len(self.rows)
        self.rows = [r for r in self.rows if r[1] != self._last_session]
        self._last_session = None
        return before - len(self.rows)

    def drain(self):
        rows, self.rows = self.rows, []
        self._last_session = None
        return rows
