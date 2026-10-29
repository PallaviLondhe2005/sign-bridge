"""
Sign Bridge - Prototype Core Logic (main modules only, not the full app)
Multi-Language Sign Language Learning & Converter App

How the AI part works
  1. MediaPipe finds 21 landmarks on the hand in every camera frame.
  2. We turn them into 63 numbers (normalised, so position/size don't matter).
  3. A machine-learning classifier (trained by train_model.py on YOUR sign
     samples) predicts which sign it is, with a confidence score.

Modules in this file
  1. Multi-language sign vocabulary
  2. Hand-landmark feature extraction  (camera -> numbers)
  3. Sign recognition                  (numbers -> sign label)   <- the AI model
  4. Text-to-speech                    (Sign -> Speech)
  5. Text-to-sign lookup               (Speech/Text -> Sign)
  6. Main loop for the Sign -> Speech screen

Install:  pip install opencv-python mediapipe scikit-learn numpy pyttsx3
"""

import pickle
import re
import numpy as np

# ------------------------------------------------------------------
# 1. MULTI-LANGUAGE SIGN VOCABULARY
#    One list per regional sign language. The same list is used for
#    (a) collecting training samples, (b) recognising signs and
#    (c) finding sign clips, so they can never get out of sync.
#    To add a language: add a new key with its list, then collect and
#    train for it. Edit a list to change which signs the app knows.
# ------------------------------------------------------------------
_COMMON_SIGNS = ["hello", "thank you", "i love you", "help", "please", "sorry",
                 "water", "food", "doctor", "emergency", "teacher", "understand"]

VOCAB = {
    "ISL": list(_COMMON_SIGNS),
    "ASL": list(_COMMON_SIGNS),
}

MODEL_PATHS = {"ISL": "models/isl_model.pkl", "ASL": "models/asl_model.pkl"}


def clip_path(language, sign):
    """Animation clip for a whole sign, e.g. clips/isl/thank_you.mp4"""
    return f"clips/{language.lower()}/{sign.replace(' ', '_')}.mp4"


def letter_clip(language, ch):
    """Clip for one fingerspelled letter, e.g. clips/isl/letters/m.mp4"""
    return f"clips/{language.lower()}/letters/{ch}.mp4"


# ------------------------------------------------------------------
# 2. FEATURE EXTRACTION  (21 hand landmarks -> 63 numbers)
#    Landmarks are made relative to the wrist and scaled, so the same
#    sign gives similar numbers whether the hand is near/far or
#    left/right on screen.
# ------------------------------------------------------------------
def normalize_landmarks(points):
    """points: list of 21 (x, y, z) tuples -> flat numpy array of 63 values."""
    pts = np.array(points, dtype=float)          # shape (21, 3)
    pts = pts - pts[0]                           # wrist becomes origin
    scale = np.max(np.abs(pts)) or 1.0           # avoid divide-by-zero
    return (pts / scale).flatten()


def extract_features(hand_landmarks):
    """Convert a MediaPipe hand_landmarks object into the feature vector."""
    points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
    return normalize_landmarks(points)


# ------------------------------------------------------------------
# 3. SIGN RECOGNITION  (the trained AI model)
# ------------------------------------------------------------------
def load_model(language):
    with open(MODEL_PATHS[language], "rb") as f:
        return pickle.load(f)


def recognise_sign(features, model, min_confidence=0.70):
    """Return (sign_text, confidence), or (None, confidence) if unsure.
    The model stores the sign names itself (model.classes_), so the
    answer is always the right label for what it was trained on."""
    probs = model.predict_proba([features])[0]
    best = int(np.argmax(probs))
    confidence = float(probs[best])
    if confidence < min_confidence:
        return None, confidence                  # don't guess on weak matches
    return str(model.classes_[best]), confidence


# ------------------------------------------------------------------
# 4. TEXT-TO-SPEECH  (Sign -> Speech output)
# ------------------------------------------------------------------
def speak(text):
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# ------------------------------------------------------------------
# 5. TEXT-TO-SIGN LOOKUP  (Speech/Text -> Sign animation clips)
#    Longest match first, so "i love you" is one sign, not three words.
#    Unknown words are fingerspelled letter by letter, so the user
#    always gets an output instead of the word being silently skipped.
# ------------------------------------------------------------------
def text_to_sign(text, language):
    """Return (clips_to_play, words_that_had_to_be_fingerspelled)."""
    words = re.findall(r"[a-z']+", text.lower())
    signs = sorted(VOCAB[language], key=lambda s: -len(s.split()))
    clips, fingerspelled, i = [], [], 0
    while i < len(words):
        for sign in signs:
            parts = sign.split()
            if words[i:i + len(parts)] == parts:
                clips.append(clip_path(language, sign))
                i += len(parts)
                break
        else:
            word = words[i]
            fingerspelled.append(word)
            clips.extend(letter_clip(language, ch) for ch in word if ch.isalpha())
            i += 1
    return clips, fingerspelled


# ------------------------------------------------------------------
# 6. MAIN LOOP - SIGN -> SPEECH SCREEN
# ------------------------------------------------------------------
def run_sign_to_speech(language="ISL"):
    import cv2
    import mediapipe as mp

    model = load_model(language)
    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
    cap = cv2.VideoCapture(0)
    last_spoken, stable_count, candidate = None, 0, None

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            features = extract_features(result.multi_hand_landmarks[0])
            sign, conf = recognise_sign(features, model)

            # Only accept a sign once it stays the same for 10 frames
            # (avoids flickering / accidental recognitions)
            if sign and sign == candidate:
                stable_count += 1
            else:
                candidate, stable_count = sign, 1

            if sign and stable_count >= 10 and sign != last_spoken:
                print(f"Recognised: {sign} ({conf:.0%})")
                speak(sign)
                last_spoken = sign

        cv2.imshow("Sign -> Speech", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Demo of the Speech/Text -> Sign direction (no camera needed)
    clips, spelled = text_to_sign("Hello, I love you. Help me!", "ISL")
    print("Play clips:", clips)
    print("Fingerspelled words:", spelled)

    # Demo of the Sign -> Speech direction (needs camera + trained model):
    #   1) python train_model.py collect ISL
    #   2) python train_model.py train ISL
    #   3) run_sign_to_speech("ISL")
