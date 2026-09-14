# ASL Smart Glasses — Real-Time Sign Language Detection

Thesis project (AUC): real-time American Sign Language recognition intended to
run on smart glasses. The recognition pipeline is **fully on-device** — camera
→ MediaPipe hand landmarks → scikit-learn classifier → text → speech. There is
no cloud inference step.

Design objectives for the deployed system: **fast, reliable, lightweight**.

## Pipeline

```
camera frame
   ↓
MediaPipe Hand Landmarker  →  21 3D hand landmarks (one hand)
   ↓
   ├── motion.py       trajectory rules  →  J / Z
   └── normalize       hand-frame features (42 floats)
          ↓
       MLPClassifier   →  24 static letters + word signs
   ↓
debouncer.py   one commit per continuous hold
   ↓
nlp_bridge.py  SymSpell correction / word segmentation
   ↓
audio.py       text-to-speech
```

### Feature representation

Static classification uses a **hand-frame normalization**: origin at the wrist,
unit scale = wrist→middle-knuckle distance, rotated so the thumb lands on +x
(which handles left vs. right hand automatically), flattened to 42 floats
(21 landmarks × x,y). This makes the classifier invariant to hand position,
scale and in-plane rotation.

> **Keep this consistent.** The same normalization appears in `collect_data.py`,
> `main.py` and `evaluate.py`. Changing it in one place without the others makes
> the trained model and live inference disagree silently.

## Setup

```bash
pip install -r requirements.txt
```

`hand_landmarker.task` (the MediaPipe model) is committed to the repo, so no
extra download is needed.

Text-to-speech in `audio.py` shells out to Windows PowerShell `System.Speech`;
on Linux/macOS everything else works, but speech output will not.

## Usage

| Command | What it does | Keys |
| --- | --- | --- |
| `python main.py` | Live translation | `q` quit · `r` reset sentence · `s` speak current text |
| `python collect_data.py` | Record labeled landmark data → appends to `landmark_data.csv` | `[` / `]` change label · `SPACE` record · `q` save & quit |
| `python train_classifier.py` | Train the MLP → `landmark_model.joblib` + `landmark_labels.json` | — |
| `python evaluate.py` | Accuracy benchmark against known target letters → appends to `eval_results.csv` | `SPACE` start 4s capture · `n` skip · `q` quit |

`collect_data.py` **appends** and never overwrites, so data from multiple people
and sessions accumulates. Multi-person collection is encouraged — more hand
shapes and sizes generalize better.

`evaluate.py` is the real accuracy number, distinct from the training/validation
split printed by `train_classifier.py`: it runs the *whole* live pipeline
(motion detection, debouncing and all) against a known target. Always label the
run with its condition (background / lighting / person) when prompted — results
accumulate across runs so conditions stay comparable.

## Vocabulary

- **24 static letters** — A–Y excluding J and Z.
- **J and Z** — motion signs, detected from fingertip trajectory in `motion.py`
  rather than by the classifier. Timing uses wall-clock milliseconds, not frame
  counts, so gesture timing survives a frame-rate change on other hardware.
- **Word signs** — `ILY`, `IHATEYOU`, `HELLO`. The classifier treats a label as
  an opaque string, so whole-word handshapes need no architectural change.

> **Academic honesty note:** `ILY` is the real ASL sign and is genuinely static.
> `IHATEYOU` and `HELLO` are mapped to **custom, self-chosen static handshapes**,
> not the real ASL signs (real HATE has a flicking motion; real HELLO has arm
> movement — neither is static). These should be described in the writeup as a
> *custom static-vocabulary extension the architecture supports*, not as
> recognition of real ASL HATE/HELLO.

### Adding a word sign

1. Add the label to `WORDS` in `collect_data.py`
2. Add its display phrase to `WORD_SIGNS` in `nlp_bridge.py`
3. Collect data
4. Retrain

Nothing else changes. Two-handed signs (e.g. CALM DOWN) are **out of scope** —
the pipeline tracks one hand (`num_hands=1`); a second hand is a real
architecture change.

## Results

`evaluate.py` runs to date, full alphabet including J/Z:

| Condition | Score | Miss |
| --- | --- | --- |
| `green_bg_dim` | 25/26 (96.2%) | G read as Q |
| `white_bg_dim` | 25/26 (96.2%) | Z never fired (read as X) |
| **Combined** | **50/52 (96.2%)** | |

G was read correctly in the second run, so the G/Q confusion is not (yet) a
recurring failure. See `NOTES.md` for the open issues behind both misses.

## Repository layout

| File | Role |
| --- | --- |
| `main.py` | Live pipeline |
| `collect_data.py` | Labeled landmark data collection |
| `train_classifier.py` | MLP training + per-class metrics |
| `evaluate.py` | End-to-end accuracy benchmark |
| `motion.py` | J/Z trajectory detection |
| `debouncer.py` | One commit per continuous hold |
| `nlp_bridge.py` | SymSpell correction, word-sign lookup |
| `audio.py` | Windows TTS output |
| `landmark_data.csv` | Training data (21,526 rows, 27 classes) |
| `landmark_model.joblib` / `landmark_labels.json` | Trained model + label order |
| `eval_results.csv` | Accumulated evaluation log |
| `hand_landmarker.task` | MediaPipe hand landmarker model |

### Branches

- **`main`** — the ML pipeline. This is the thesis basis.
- **`trial`** — a rule-based, non-ML geometric classifier (`rules.py` +
  `motion.py`). Kept as a **fallback / last resort only**. It proved the
  landmark-geometry concept before `main` adopted an ML version of the same idea.
