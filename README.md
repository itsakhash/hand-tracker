# Hand Tracker

Real-time hand tracking from a webcam using MediaPipe and OpenCV.
Shows landmarks, finger count, a pinch meter, and a fingertip trail.

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

Keys: `q` quit, `c` clear trail. Use `--camera 1` if the wrong camera opens.
On macOS, allow camera access for your terminal in
System Settings > Privacy & Security > Camera.
