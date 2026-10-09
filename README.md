# Hand Tracker

Real-time hand tracking from a webcam using MediaPipe and OpenCV.
Recognizes gestures and lets you draw in the air.

## Gestures

| Gesture | Action |
|---|---|
| POINT | draw with your index fingertip |
| OPEN PALM | move without drawing |
| PEACE | next color |
| FIST | erase |

Also recognized: ROCK and THUMBS UP. The screen shows the gesture, which
fingers are extended (T I M R P), a finger count, and a pinch meter.

Keys: `q` quit, `c` clear, `s` save drawing (transparent PNG in `drawings/`),
`[` / `]` brush size.

## Setup (macOS / Linux)

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Setup (Windows)

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

## Run

    python hand_tracker.py

The first run downloads the hand landmark model (`hand_landmarker.task`).
If the download fails with an SSL error (common on macOS), download it with:

    curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

Use `--camera 1` if the wrong camera opens (for example, an iPhone via
Continuity Camera). On macOS, allow camera access for your terminal in
System Settings > Privacy & Security > Camera.
