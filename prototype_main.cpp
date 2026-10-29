/*
 * Sign Bridge - Prototype Core Logic in C++ (main modules only, not the full app)
 * Multi-Language Sign Language Learning & Converter App
 *
 * How the AI part works
 *   1. A hand gives 21 landmarks (x, y, z)  ->  normalised into 63 numbers.
 *   2. A k-Nearest-Neighbours (k-NN) classifier, trained on labelled sign
 *      samples, predicts the sign. Confidence = share of the k nearest
 *      samples that agree. Below 70% the program refuses to guess.
 *
 * Note: finding hand landmarks in a camera image (MediaPipe) is done outside
 * this file. Here the landmarks come from sample data (CSV) or from the
 * built-in synthetic generator used for the demo and self-tests.
 *
 * Build:  g++ -std=c++17 -O2 -Wall -Wextra prototype_main.cpp -o prototype
 * Run:    ./prototype               demo + self-tests (nothing else needed)
 *         ./prototype samples.csv   also train on your CSV and print accuracy
 *         CSV line format:  label,f1,f2,...,f63
 */

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ------------------------------------------------------------------
// 1. MULTI-LANGUAGE SIGN VOCABULARY
//    One list per regional sign language. Edit a list to change which
//    signs the app knows. To add a language: add a new key + its list.
// ------------------------------------------------------------------
const vector<string> COMMON_SIGNS = {
    "hello", "thank you", "i love you", "help", "please", "sorry",
    "water", "food", "doctor", "emergency", "teacher", "understand"};

const map<string, vector<string>> VOCAB = {
    {"ISL", COMMON_SIGNS},
    {"ASL", COMMON_SIGNS},
};

string lower(string s) {
    transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return tolower(c); });
    return s;
}

string clipPath(const string& language, string sign) {
    replace(sign.begin(), sign.end(), ' ', '_');
    return "clips/" + lower(language) + "/" + sign + ".mp4";
}

string letterClip(const string& language, char ch) {
    return "clips/" + lower(language) + "/letters/" + string(1, ch) + ".mp4";
}

// ------------------------------------------------------------------
// 2. FEATURE EXTRACTION  (21 hand landmarks -> 63 numbers)
//    Landmarks are made relative to the wrist and scaled, so the same
//    sign gives similar numbers wherever the hand is and however big.
// ------------------------------------------------------------------
using Landmark = array<double, 3>;

vector<double> normalizeLandmarks(const vector<Landmark>& pts) {
    const Landmark wrist = pts[0];
    double scale = 0.0;
    for (const auto& p : pts)
        for (int i = 0; i < 3; i++) scale = max(scale, fabs(p[i] - wrist[i]));
    if (scale == 0.0) scale = 1.0;                 // avoid divide-by-zero

    vector<double> features;
    features.reserve(pts.size() * 3);
    for (const auto& p : pts)
        for (int i = 0; i < 3; i++) features.push_back((p[i] - wrist[i]) / scale);
    return features;
}

// ------------------------------------------------------------------
// 3. SIGN RECOGNITION  (the AI model: k-Nearest-Neighbours)
// ------------------------------------------------------------------
struct Sample {
    vector<double> features;
    string label;
};

struct Prediction {
    bool confident;      // false = not sure, do not speak
    string sign;
    double confidence;   // 0.0 - 1.0
};

class KnnModel {
public:
    explicit KnnModel(int k = 5) : k_(k) {}

    void train(const vector<Sample>& samples) { samples_ = samples; }

    Prediction predict(const vector<double>& f, double minConfidence = 0.70) const {
        if (samples_.empty()) return {false, "", 0.0};

        // distance from the new hand to every training sample
        vector<pair<double, size_t>> dist;
        for (size_t i = 0; i < samples_.size(); i++) {
            double d = 0.0;
            for (size_t j = 0; j < f.size(); j++) {
                double diff = f[j] - samples_[i].features[j];
                d += diff * diff;
            }
            dist.push_back({d, i});
        }
        const size_t k = min(static_cast<size_t>(k_), dist.size());
        partial_sort(dist.begin(), dist.begin() + k, dist.end());

        // the k nearest samples vote
        map<string, int> votes;
        for (size_t i = 0; i < k; i++) votes[samples_[dist[i].second].label]++;
        string best;
        int bestVotes = 0;
        for (const auto& v : votes)
            if (v.second > bestVotes) { best = v.first; bestVotes = v.second; }

        double confidence = static_cast<double>(bestVotes) / static_cast<double>(k);
        if (confidence < minConfidence) return {false, "", confidence};   // don't guess
        return {true, best, confidence};
    }

private:
    int k_;
    vector<Sample> samples_;
};

// ------------------------------------------------------------------
// 4. SPEECH OUTPUT  (Sign -> Speech)
//    Prints the text. On a real device call the OS text-to-speech here,
//    e.g. system(("espeak \"" + text + "\"").c_str()) on Linux.
// ------------------------------------------------------------------
void speak(const string& text) { cout << "[SPEAK] " << text << "\n"; }

// ------------------------------------------------------------------
// 5. STABLE-FRAME FILTER  (same idea as the live camera loop)
//    A sign is accepted only after it stays the same for N frames, and
//    it is not repeated while the same sign stays in view.
// ------------------------------------------------------------------
class StableFilter {
public:
    explicit StableFilter(int framesNeeded = 10) : need_(framesNeeded) {}

    // returns true (and sets 'out') when a sign should be spoken now
    bool update(const Prediction& p, string& out) {
        if (!p.confident) { candidate_.clear(); count_ = 0; return false; }
        if (p.sign == candidate_) count_++;
        else { candidate_ = p.sign; count_ = 1; }

        if (count_ >= need_ && candidate_ != lastSpoken_) {
            lastSpoken_ = candidate_;
            out = candidate_;
            return true;
        }
        return false;
    }

private:
    int need_;
    int count_ = 0;
    string candidate_, lastSpoken_;
};

// ------------------------------------------------------------------
// 6. TEXT-TO-SIGN LOOKUP  (Speech/Text -> Sign animation clips)
//    Longest match first, so "i love you" is one sign, not three words.
//    Unknown words are fingerspelled letter by letter, so the user
//    always gets an output instead of the word being silently skipped.
// ------------------------------------------------------------------
vector<string> splitOnSpaces(const string& s) {
    istringstream in(s);
    vector<string> parts;
    string w;
    while (in >> w) parts.push_back(w);
    return parts;
}

vector<string> tokenize(const string& text) {      // keeps letters and '
    vector<string> words;
    string cur;
    for (unsigned char c : text) {
        if (isalpha(c) || c == '\'') cur += static_cast<char>(tolower(c));
        else if (!cur.empty()) { words.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) words.push_back(cur);
    return words;
}

struct SignPlan {
    vector<string> clips;           // clips to play, in order
    vector<string> fingerspelled;   // words that had to be spelled out
};

SignPlan textToSign(const string& text, const string& language) {
    const vector<string> words = tokenize(text);

    vector<pair<string, vector<string>>> signs;     // (sign, its words)
    for (const auto& s : VOCAB.at(language)) signs.push_back({s, splitOnSpaces(s)});
    stable_sort(signs.begin(), signs.end(), [](const auto& a, const auto& b) {
        return a.second.size() > b.second.size();   // longest sign first
    });

    SignPlan plan;
    size_t i = 0;
    while (i < words.size()) {
        bool matched = false;
        for (const auto& s : signs) {
            const size_t n = s.second.size();
            if (i + n <= words.size() && equal(s.second.begin(), s.second.end(), words.begin() + i)) {
                plan.clips.push_back(clipPath(language, s.first));
                i += n;
                matched = true;
                break;
            }
        }
        if (!matched) {                              // unknown word: spell it out
            plan.fingerspelled.push_back(words[i]);
            for (char ch : words[i])
                if (isalpha(static_cast<unsigned char>(ch))) plan.clips.push_back(letterClip(language, ch));
            i++;
        }
    }
    return plan;
}

// ------------------------------------------------------------------
// DATA HELPERS: CSV loading, synthetic hands, accuracy
// ------------------------------------------------------------------
vector<Sample> loadCsv(const string& path) {
    ifstream in(path);
    vector<Sample> out;
    string line;
    while (getline(in, line)) {
        stringstream ss(line);
        Sample s;
        getline(ss, s.label, ',');
        string tok;
        try {
            while (getline(ss, tok, ',')) s.features.push_back(stod(tok));
        } catch (...) { continue; }                  // skip header / bad lines
        if (s.features.size() == 63) out.push_back(s);
    }
    return out;
}

// Makes fake hands for the demo/tests: one random "hand shape" per sign,
// plus natural variation, size change and position change.
class SyntheticHands {
public:
    explicit SyntheticHands(const vector<string>& signs) : rng_(0) {
        uniform_real_distribution<double> u(-0.5, 0.5);
        for (const auto& s : signs) {
            vector<Landmark> shape(21);
            for (auto& p : shape) for (auto& v : p) v = u(rng_);
            shapes_[s] = shape;
        }
    }
    vector<double> sample(const string& sign) {
        normal_distribution<double> noise(0.0, 0.03);
        uniform_real_distribution<double> sizeChange(0.8, 1.3), move(-0.2, 0.2);
        const double scale = sizeChange(rng_);
        const array<double, 3> shift = {move(rng_), move(rng_), move(rng_)};
        vector<Landmark> pts = shapes_.at(sign);
        for (auto& p : pts) for (int i = 0; i < 3; i++) p[i] = (p[i] + noise(rng_)) * scale + shift[i];
        return normalizeLandmarks(pts);
    }
private:
    mt19937 rng_;
    map<string, vector<Landmark>> shapes_;
};

// every 5th sample is held out for testing, the rest are used to train
double trainAndScore(KnnModel& model, const vector<Sample>& all) {
    vector<Sample> train, test;
    for (size_t i = 0; i < all.size(); i++) (i % 5 == 0 ? test : train).push_back(all[i]);
    model.train(train);
    if (test.empty()) return 0.0;
    int right = 0;
    for (const auto& s : test) right += (model.predict(s.features, 0.0).sign == s.label);
    return static_cast<double>(right) / static_cast<double>(test.size());
}

// ------------------------------------------------------------------
// SELF-TESTS (same cases as the Python version, no camera needed)
// ------------------------------------------------------------------
int passed = 0, total = 0;

void check(const string& id, bool ok, const string& detail) {
    total++;
    passed += ok;
    cout << id << ": " << (ok ? "PASS" : "FAIL") << "  - " << detail << "\n";
}

vector<Landmark> fakeHand(double offset = 0.0, double size = 1.0) {
    vector<Landmark> pts;
    for (int i = 0; i < 21; i++)
        pts.push_back({0.5 + offset + i * 0.01 * size, 0.6 - offset - i * 0.02 * size, 0.0});
    return pts;
}

void runTests() {
    // TC01 / TC02: feature extraction
    auto f = normalizeLandmarks(fakeHand());
    check("TC01", f.size() == 63 && f[0] == 0.0 && f[1] == 0.0 && f[2] == 0.0,
          to_string(f.size()) + " features, wrist at origin");

    auto a = normalizeLandmarks(fakeHand(0.0, 1.0)), b = normalizeLandmarks(fakeHand(-0.3, 2.0));
    double maxDiff = 0.0;
    for (size_t i = 0; i < a.size(); i++) maxDiff = max(maxDiff, fabs(a[i] - b[i]));
    check("TC02", maxDiff < 1e-9, "same features for different hand position/size");

    // TC03 / TC04: recognition with a tiny hand-made model
    KnnModel tiny(5);
    vector<Sample> s;
    for (int i = 0; i < 5; i++) s.push_back({vector<double>(63, 1.0 + i * 0.001), "thank you"});
    for (int i = 0; i < 5; i++) s.push_back({vector<double>(63, -1.0 - i * 0.001), "help"});
    tiny.train(s);
    auto p1 = tiny.predict(vector<double>(63, 1.0));
    check("TC03", p1.confident && p1.sign == "thank you",
          "returned '" + p1.sign + "' (" + to_string(static_cast<int>(p1.confidence * 100)) + "%)");

    KnnModel mixed(5);                              // neighbours split 3 vs 2 -> 60%
    vector<Sample> m;
    for (int i = 0; i < 3; i++) m.push_back({vector<double>(63, 0.0 + i * 0.001), "hello"});
    for (int i = 0; i < 2; i++) m.push_back({vector<double>(63, 0.01 + i * 0.001), "sorry"});
    mixed.train(m);
    auto p2 = mixed.predict(vector<double>(63, 0.0));
    check("TC04", !p2.confident,
          "no sign returned at " + to_string(static_cast<int>(p2.confidence * 100)) + "% confidence (below 70% threshold)");

    // TC05 - TC08: text to sign
    auto t5 = textToSign("hello help", "ISL");
    check("TC05", t5.clips == vector<string>{"clips/isl/hello.mp4", "clips/isl/help.mp4"} && t5.fingerspelled.empty(),
          to_string(t5.clips.size()) + " clips, none fingerspelled");

    auto t6 = textToSign("hello me", "ISL");
    check("TC06", t6.fingerspelled == vector<string>{"me"} && t6.clips.size() == 3,
          "'me' fingerspelled as " + to_string(t6.clips.size() - 1) + " letter clips");

    auto t7 = textToSign("hello", "ASL");
    check("TC07", t7.clips == vector<string>{"clips/asl/hello.mp4"}, "ASL clip returned: " + t7.clips[0]);

    auto t8 = textToSign("I love you, thank you!", "ISL");
    check("TC08", t8.clips == vector<string>{"clips/isl/i_love_you.mp4", "clips/isl/thank_you.mp4"} && t8.fingerspelled.empty(),
          to_string(t8.clips.size()) + " whole-sign clips, punctuation ignored, nothing spelled out");

    // TC09: whole AI pipeline on synthetic hands: train -> score -> recognise
    SyntheticHands hands(COMMON_SIGNS);
    vector<Sample> all;
    for (int i = 0; i < 40; i++)
        for (const auto& sign : COMMON_SIGNS) all.push_back({hands.sample(sign), sign});
    KnnModel model(5);
    double acc = trainAndScore(model, all);
    int right = 0;
    for (const auto& sign : COMMON_SIGNS) right += (model.predict(hands.sample(sign)).sign == sign);
    check("TC09", acc >= 0.90 && right >= static_cast<int>(COMMON_SIGNS.size()) - 1,
          to_string(COMMON_SIGNS.size()) + " signs, held-out accuracy " + to_string(static_cast<int>(acc * 100)) +
              "%, " + to_string(right) + "/" + to_string(COMMON_SIGNS.size()) + " fresh samples recognised");

    // TC10: stable-frame filter speaks once, only after 10 consistent frames
    StableFilter filter(10);
    int spoken = 0, firstAt = -1;
    string out;
    for (int frame = 1; frame <= 30; frame++)
        if (filter.update({true, "help", 0.9}, out)) { spoken++; if (firstAt < 0) firstAt = frame; }
    check("TC10", spoken == 1 && firstAt == 10,
          "'help' spoken " + to_string(spoken) + " time(s), first at frame " + to_string(firstAt));
}

// ------------------------------------------------------------------
int main(int argc, char** argv) {
    cout << fixed << setprecision(0);

    // Demo: text -> sign (no camera needed)
    cout << "=== Demo: text to sign ===\n";
    SignPlan plan = textToSign("Hello, I love you. Help me!", "ISL");
    cout << "Play clips:";
    for (const auto& c : plan.clips) cout << "\n  " << c;
    cout << "\nFingerspelled words:";
    for (const auto& w : plan.fingerspelled) cout << " " << w;
    cout << "\n\n";

    // Demo: sign -> speech with synthetic hands
    cout << "=== Demo: sign to speech (synthetic hands) ===\n";
    SyntheticHands hands(COMMON_SIGNS);
    vector<Sample> all;
    for (int i = 0; i < 40; i++)
        for (const auto& sign : COMMON_SIGNS) all.push_back({hands.sample(sign), sign});
    KnnModel model(5);
    cout << "Held-out accuracy: " << trainAndScore(model, all) * 100 << "%\n";
    StableFilter filter(10);
    string said;
    for (int frame = 1; frame <= 15; frame++) {
        Prediction p = model.predict(hands.sample("help"));
        if (filter.update(p, said)) {
            cout << "Recognised: " << said << " (" << p.confidence * 100 << "%) at frame " << frame << "\n";
            speak(said);
        }
    }

    // Optional: train on the user's own CSV
    if (argc > 1) {
        cout << "\n=== Your data: " << argv[1] << " ===\n";
        vector<Sample> mine = loadCsv(argv[1]);
        if (mine.empty()) {
            cout << "No valid samples found (need: label,f1..f63 per line).\n";
        } else {
            KnnModel mymodel(5);
            cout << mine.size() << " samples loaded. Held-out accuracy: "
                 << trainAndScore(mymodel, mine) * 100 << "%\n";
        }
    }

    cout << "\n=== Self-tests ===\n";
    runTests();
    cout << passed << " of " << total << " tests passed\n";
    return passed == total ? 0 : 1;
}
