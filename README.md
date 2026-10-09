# hand-tracker

Webcam hand tracking with MediaPipe and OpenCV. One program, four things you can do with your hand:

- **Gestures** - FIST, OPEN PALM, POINT, PEACE, ROCK, THUMBS UP, recognized from 21 hand landmarks.
- **Air drawing** - draw on the video with your index finger.
- **Virtual mouse** - move, click, drag, right-click and scroll your real computer with your hand.
- **Your own gesture classifier** - record examples, train a small scikit-learn model, and swap it in for the hand-written rules.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python hand_tracker.py            # add --camera 1 if an iPhone grabs camera 0
```

The first run downloads the hand landmark model (about 8 MB) as `hand_landmarker.task`.
If that fails with an SSL error (common on macOS), download it yourself:

```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Draw mode (default)

| Gesture | Action |
|---|---|
| POINT | draw with the index fingertip |
| OPEN PALM | move without drawing |
| PEACE | next color |
| FIST | erase |

Keys: `q` quit, `m` mouse mode, `g` rules / learned gestures, `c` clear, `s` save a transparent PNG to `drawings/`, `[` `]` brush size.

## Mouse mode

Press `m`, or hold THUMBS UP for 1 second.

| Hand | Action |
|---|---|
| move inside the green box | move the cursor |
| pinch thumb + index | left click (hold and move to drag) |
| pinch thumb + middle | right click |
| PEACE + move up/down | scroll |
| FIST | pause the cursor |
| hold THUMBS UP | leave mouse mode |

Emergency stop: slam your real mouse into the top-left screen corner.
On macOS, allow your terminal under System Settings > Privacy & Security > Accessibility, then restart it.

Tuning flags: `--still-radius PX`, `--click-lookback SEC`, `--cursor-lag SEC`, `--direct-cursor`, `--profile`.

## Train your own gesture classifier

```bash
python collect_gestures.py     # 1. record labelled examples -> datasets/gestures.csv
python train_gestures.py       # 2. train, test, save       -> models/gesture_model.joblib
python hand_tracker.py --learned   # 3. use it (or press g while running)
```

**Recording.** Press a number key to pick a gesture, then SPACE: after a 3 second countdown it records
for 10 seconds. Hold the gesture while slowly moving, tilting and re-positioning your hand. Do at least
**2 takes of every gesture**, ideally at different times or in different lighting. Record an `UNKNOWN` take
of random in-between hand shapes. Press `u` to throw away a bad take, `s` to save, `q` to save and quit.

**How it works.**
- Features: the 20 non-wrist landmarks, shifted so the wrist is the origin and divided by palm length, so distance from the camera and position in the frame don't matter (hand tilt does, on purpose).
- Augmentation: every example is also mirrored (so a model trained on your right hand works on your left) and copied a few times with small rotations and jitter.
- Honest testing: neighboring video frames are nearly identical, so splitting frames randomly would leak and inflate the score. Instead, whole takes ("sessions") are held out, and the model is scored only on takes it never saw. The trainer prints a confusion matrix and compares against the hand-written rules on the same samples.
- Low confidence (< 0.6) becomes `UNKNOWN`.
- Try other models with `--kind forest|knn|mlp`.

**Label spelling matters.** Drawing and mouse mode react to `OPEN PALM`, `FIST`, `POINT`, `PEACE`, `THUMBS UP`. `ROCK` and `UNKNOWN` show up on screen but trigger nothing.

**Safety.** `.joblib` files are pickles - only load models you trained yourself.

## Tests

No camera needed:

```bash
python -m unittest -v test_hand_tracker.py test_gesture_model.py
```

## Project layout

| File | Purpose |
|---|---|
| `hand_tracker.py` | main app: detection, gestures, drawing, mouse mode |
| `gesture_model.py` | features, dataset, training, prediction, recording logic |
| `collect_gestures.py` | record training data |
| `train_gestures.py` | train and evaluate a model |
| `test_hand_tracker.py`, `test_gesture_model.py` | unit tests |
| `datasets/` | recorded landmarks (committed) |
| `models/` | trained models (git-ignored) |
