# Sign Bridge

Multi-language sign language learning and converter app (SIH PS2).
Helps deaf and mute individuals communicate and learn without an interpreter:

- **Speech/Text to Sign:** a sentence becomes a sequence of signs. Whole phrases such as "i love you" are matched first; words without a sign are fingerspelled letter by letter.
- **Sign to Speech/Text:** the webcam finds 21 hand points (MediaPipe), they become 63 normalised numbers, and a trained model predicts the sign. A sign is accepted only when the model is at least 70% sure and the sign stays the same for 10 frames; then it is shown and spoken.
- Vocabulary of 12 signs for ISL and ASL (edit `VOCAB` to change it).

This is a Design Thinking lab prototype (Experiments 6 to 10).

## Files

| File | What it is |
|---|---|
| `prototype_main.py` | Core logic (Python): features, recognition, text to sign, live loop |
| `train_model.py` | Collect sign samples from the webcam and train the Random Forest model |
| `test_prototype.py` | 9 logic tests |
| `prototype_main.cpp` | C++ version of the core logic (k-NN model, self-tests) |
| `sign_bridge_live.html` | Browser version with live webcam recognition (open in Chrome) |
| `sign_bridge_prototype.html` | Browser demo of text to sign and the 70% confidence rule |

## Run (Python)

```
pip install opencv-python mediapipe scikit-learn numpy pyttsx3
python prototype_main.py          # text to sign demo, no camera needed
python test_prototype.py          # 9 tests
python train_model.py collect ISL # record your own signs with the webcam
python train_model.py train ISL   # train the model
```

## Run (C++)

```
g++ -std=c++17 -O2 -Wall -Wextra prototype_main.cpp -o prototype
./prototype
```

## Status and limits

- The logic tests pass (9 of 9). The training test uses synthetic hand data, so it shows that the pipeline works, not accuracy on real signs.
- The live webcam recognition has not been validated on real signs yet. Real accuracy depends on the samples you record.
- Single-hand static signs only. Sign animation clips are placeholders.

## Author

Pallavi Maruti Londhe, CS2167, RBT25CS069, RSCOE Pune.
