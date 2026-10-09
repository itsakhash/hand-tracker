"""
Train a gesture classifier from the data recorded by collect_gestures.py.

    python train_gestures.py
    python train_gestures.py --kind knn          # try another model: forest, knn, mlp
    python train_gestures.py --data datasets/gestures.csv --out models/gesture_model.joblib

It prints how many samples/takes each gesture has, then tests on takes the model
never saw (so the score is honest), shows a confusion matrix, compares against the
hand-written rules, and finally saves a model trained on everything.
Then run:  python hand_tracker.py   and press g.
"""
import argparse
import os

import gesture_model

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "datasets", "gestures.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "models", "gesture_model.joblib"))
    ap.add_argument("--kind", choices=gesture_model.MODEL_KINDS, default="forest")
    ap.add_argument("--test-fraction", type=float, default=0.3,
                    help="fraction of each gesture's takes held out for testing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"No dataset at {args.data}. Run collect_gestures.py first.")

    baseline = None
    try:
        from hand_tracker import classify_gesture
        baseline = classify_gesture
    except ImportError:
        pass  # opencv/mediapipe missing: just skip the comparison

    try:
        gesture_model.train_and_report(args.data, args.out, args.kind, args.test_fraction,
                                       args.seed, baseline)
    except ImportError as e:
        raise SystemExit(f"{e}\nInstall the training library with:  pip install scikit-learn")
    except ValueError as e:
        raise SystemExit(f"Cannot train: {e}")


if __name__ == "__main__":
    main()
