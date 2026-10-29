"""
Test cases for the Sign Bridge prototype core logic and AI training pipeline.
Run:  python test_prototype.py

TC01-TC09 run without a camera.
TC10-TC12 (camera recognition, speech output, UI navigation) must be done by
hand on a laptop/phone - see the report for the table to fill in.
"""
import os
import pickle
import tempfile

import numpy as np

import prototype_main as p
import train_model as t


class StubModel:
    """Pretends to be a trained model that returns fixed probabilities."""
    def __init__(self, best_index, best_prob, language="ISL"):
        self.classes_ = np.array(p.VOCAB[language])
        n = len(self.classes_)
        rest = (1.0 - best_prob) / (n - 1)
        probs = [rest] * n
        probs[best_index] = best_prob
        self.probs = probs

    def predict_proba(self, X):
        return [self.probs]


def fake_hand(offset=0.0, size=1.0):
    return [(0.5 + offset + i * 0.01 * size, 0.6 - offset - i * 0.02 * size, 0.0) for i in range(21)]


def tc01():
    f = p.normalize_landmarks(fake_hand())
    return f.shape == (63,) and np.allclose(f[:3], 0), f"{f.shape[0]} features, wrist at origin"


def tc02():
    a = p.normalize_landmarks(fake_hand(0.0, 1.0))
    b = p.normalize_landmarks(fake_hand(-0.3, 2.0))
    return np.allclose(a, b), "same features for different hand position/size"


def tc03():
    f = p.normalize_landmarks(fake_hand())
    sign, conf = p.recognise_sign(f, StubModel(1, 0.85))          # index 1 = 'thank you'
    return sign == "thank you", f"returned '{sign}' ({conf:.0%})"


def tc04():
    f = p.normalize_landmarks(fake_hand())
    sign, conf = p.recognise_sign(f, StubModel(0, 0.30))
    return sign is None, f"returned {sign} at {conf:.0%} confidence (below 70% threshold)"


def tc05():
    clips, spelled = p.text_to_sign("hello help", "ISL")
    return clips == ["clips/isl/hello.mp4", "clips/isl/help.mp4"] and spelled == [], f"{len(clips)} clips, none fingerspelled"


def tc06():
    clips, spelled = p.text_to_sign("hello me", "ISL")
    ok = spelled == ["me"] and clips[-2:] == ["clips/isl/letters/m.mp4", "clips/isl/letters/e.mp4"]
    return ok, f"'me' fingerspelled as {len(clips) - 1} letter clips" if ok else f"clips={clips}, spelled={spelled}"


def tc07():
    clips, _ = p.text_to_sign("hello", "ASL")
    return clips == ["clips/asl/hello.mp4"], f"ASL clip returned: {clips[0]}"


def tc08():
    clips, spelled = p.text_to_sign("I love you, thank you!", "ISL")
    ok = clips == ["clips/isl/i_love_you.mp4", "clips/isl/thank_you.mp4"] and spelled == []
    return ok, f"{len(clips)} whole-sign clips, punctuation ignored, nothing spelled out" if ok else f"clips={clips}, spelled={spelled}"


def tc09():
    """Whole AI pipeline on synthetic hand data: train -> save -> load -> recognise."""
    rng = np.random.default_rng(0)
    signs = p.VOCAB["ISL"]
    shapes = {s: rng.uniform(-0.5, 0.5, size=(21, 3)) for s in signs}   # one 'hand shape' per sign

    def sample(sign):
        pts = shapes[sign] + rng.normal(0, 0.03, size=(21, 3))           # natural variation
        pts = pts * rng.uniform(0.8, 1.3) + rng.uniform(-0.2, 0.2, size=3)  # size / position change
        return p.normalize_landmarks([tuple(r) for r in pts])

    X = np.array([sample(s) for s in signs for _ in range(40)])
    y = np.array([s for s in signs for _ in range(40)])
    model, accuracy = t.train_from_arrays(X, y)

    with tempfile.TemporaryDirectory() as d:                              # save + load round trip
        path = os.path.join(d, "m.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        with open(path, "rb") as f:
            loaded = pickle.load(f)

    right = 0
    for s in signs:
        sign, _ = p.recognise_sign(sample(s), loaded)
        right += (sign == s)
    ok = accuracy >= 0.90 and right >= len(signs) - 1
    return ok, f"{len(signs)} signs, held-out accuracy {accuracy:.0%}, {right}/{len(signs)} fresh samples recognised after save/load"


def run_all():
    results = []
    for name, fn in [("TC01", tc01), ("TC02", tc02), ("TC03", tc03), ("TC04", tc04), ("TC05", tc05),
                     ("TC06", tc06), ("TC07", tc07), ("TC08", tc08), ("TC09", tc09)]:
        ok, detail = fn()
        results.append((name, "PASS" if ok else "FAIL", detail))
        print(f"{name}: {'PASS' if ok else 'FAIL'}  - {detail}")
    return results


if __name__ == "__main__":
    run_all()
