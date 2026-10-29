"""
Sign Bridge - build the AI model (collect samples + train)

Step 1 - collect your own sign samples:  python train_model.py collect ISL
Step 2 - train the model:                python train_model.py train ISL
Step 3 - use it:  run_sign_to_speech("ISL") in prototype_main.py

The signs to record come from VOCAB in prototype_main.py.
Tips for good samples: good light, plain background, move your hand slightly
and change distance a little while recording, and record in 2-3 sessions.

Note: the accuracy printed after training is measured on held-out samples
from the same person and session, so real-world accuracy will usually be
lower. Test it live with different people and lighting before trusting it.

Install:  pip install opencv-python mediapipe scikit-learn numpy
"""

import os
import pickle
import sys

import numpy as np

import prototype_main as core

DATA_DIR = "data"
SAMPLES_PER_SIGN = 150


# ------------------------------------------------------------------
# STEP 1 - COLLECT SAMPLES FROM THE WEBCAM
# ------------------------------------------------------------------
def collect(language="ISL"):
    import cv2
    import mediapipe as mp

    os.makedirs(DATA_DIR, exist_ok=True)
    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
    cap = cv2.VideoCapture(0)
    X, y = [], []

    def show(frame, text):
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Collect signs", frame)
        return cv2.waitKey(1) & 0xFF

    for sign in core.VOCAB[language]:
        # wait until the person is ready
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            key = show(frame, f'Next sign: "{sign}"  -  press SPACE to record, Q to stop')
            if key == ord(" "):
                break
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                print("Stopped early - nothing saved.")
                return

        # record samples (only frames where a hand is found)
        count = 0
        while count < SAMPLES_PER_SIGN:
            ok, frame = cap.read()
            if not ok:
                break
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                X.append(core.extract_features(result.multi_hand_landmarks[0]))
                y.append(sign)
                count += 1
            show(frame, f'Recording "{sign}": {count}/{SAMPLES_PER_SIGN}  (move your hand a little)')

    cap.release()
    cv2.destroyAllWindows()
    path = os.path.join(DATA_DIR, f"{language}.npz")
    np.savez(path, X=np.array(X), y=np.array(y))
    print(f"Saved {len(X)} samples to {path}")


# ------------------------------------------------------------------
# STEP 2 - TRAIN THE MODEL
# ------------------------------------------------------------------
def train_from_arrays(X, y):
    """Train a classifier on feature vectors X and sign names y.
    Returns (model, accuracy on a 20% held-out test set)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model, model.score(X_test, y_test)


def train(language="ISL"):
    data = np.load(os.path.join(DATA_DIR, f"{language}.npz"))
    model, accuracy = train_from_arrays(data["X"], data["y"])
    print("Signs learnt:", ", ".join(str(c) for c in model.classes_))
    print(f"Accuracy on held-out samples: {accuracy:.1%}")

    path = core.MODEL_PATHS[language]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("collect", "train"):
        print("Usage: python train_model.py collect|train [ISL|ASL]")
        sys.exit(1)
    lang = sys.argv[2] if len(sys.argv) > 2 else "ISL"
    collect(lang) if sys.argv[1] == "collect" else train(lang)
