"""
Record labelled hand landmarks to train your own gesture classifier.

    python collect_gestures.py                 # records into datasets/gestures.csv
    python collect_gestures.py --camera 1      # if an iPhone grabs camera 0

How to record (this is the part that decides how good the model is):
  1. Press a number key (1, 2, 3...) to choose which gesture you are about to record.
  2. Press SPACE. After a 3 second countdown it records for 10 seconds (SPACE again
     stops early). Hold the gesture, and slowly move your hand around, tilt it,
     and move it nearer and farther while you do.
  3. Do at least 2 takes of every gesture, ideally on different occasions or with
     different lighting and hand positions. The trainer tests on takes it has never
     seen, so it needs at least 2 per gesture.
  4. Record an UNKNOWN take of random in-between hand shapes so the model learns to
     say "none of the above".
  5. If a take went wrong, press u to throw it away (only before saving).

Keys: 1-9 choose gesture | SPACE start/stop a take | u undo last take |
      s save to disk | q save and quit

The labels must be spelled like the app's gestures (OPEN PALM, FIST, POINT, PEACE,
THUMBS UP) for drawing and mouse mode to react to them. ROCK (index + pinky up) is
recognized and shown on screen but does not trigger anything yet. Use --labels to change
the list.
"""
import argparse
import os
import time

import cv2

import gesture_model
import hand_tracker as ht

DEFAULT_DATASET = os.path.join(ht.HERE, "datasets", "gestures.csv")


def save_rows(recorder, path):
    """Write the unsaved samples to the CSV. Returns how many were saved."""
    return gesture_model.append_rows(path, recorder.drain())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--data", default=DEFAULT_DATASET, help="CSV file to append to")
    ap.add_argument("--labels", nargs="+", default=gesture_model.DEFAULT_LABELS,
                    help='gestures to record, e.g. --labels "OPEN PALM" FIST POINT')
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--take-seconds", type=float, default=10.0)
    args = ap.parse_args()
    if len(args.labels) > 9:
        raise SystemExit("At most 9 labels (they are chosen with the number keys).")

    recorder = gesture_model.SessionRecorder(args.labels, args.countdown, args.take_seconds)
    detector = ht.HandDetector()
    stream = ht.CameraStream(ht.open_camera(args.camera, args.width, args.height))
    saved_total = 0
    toast = ("", 0.0)
    last_id = -1

    while True:
        ok, frame, last_id = stream.read(last_id)
        if not ok:
            print("Camera stopped delivering frames.")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        pts = detector.detect(frame)
        if pts is not None:
            ht.draw_hand(frame, pts)
            recorder.add(pts, now)

        state = recorder.state(now)
        color = {"idle": (255, 255, 255), "countdown": (0, 200, 255), "recording": (0, 0, 255)}[state]
        cv2.putText(frame, recorder.label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        if state == "countdown":
            cv2.putText(frame, f"Get ready... {recorder.seconds_left(now):.0f}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        elif state == "recording":
            tag = "REC" if pts is not None else "REC (no hand!)"
            cv2.putText(frame, f"{tag} {recorder.seconds_left(now):.0f}s", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            cv2.putText(frame, "SPACE to record", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        counts = recorder.counts()
        for i, label in enumerate(recorder.labels):
            mark = ">" if i == recorder.selected else " "
            cv2.putText(frame, f"{mark}{i + 1} {label}: {counts.get(label, 0)} unsaved",
                        (w - 290, 30 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if i == recorder.selected else (200, 200, 200), 1)
        cv2.putText(frame, f"saved this run: {saved_total}", (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, "1-9 choose | SPACE take | u undo | s save | q save+quit",
                    (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if toast[0] and now - toast[1] < 2.0:
            cv2.putText(frame, toast[0], (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Collect gestures", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            recorder.toggle(now)
        elif ord("1") <= key <= ord("9"):
            recorder.select(key - ord("1"))
        elif key == ord("u"):
            toast = (f"Discarded {recorder.undo_last_take()} samples", now)
        elif key == ord("s"):
            n = save_rows(recorder, args.data)
            saved_total += n
            toast = (f"Saved {n} samples", now)

    saved_total += save_rows(recorder, args.data)
    print(f"Saved {saved_total} samples to {args.data}")
    detector.close()
    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
