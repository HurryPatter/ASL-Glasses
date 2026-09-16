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
| `python main.py` | Live translation | `q` quit · `r` reset sentence · `s` speak · `d` motion debug readout |
| `python collect_data.py` | Record labeled landmark data → appends to `landmark_data.csv` | `[` / `]` change label · `SPACE` record · `q` save & quit |
| `python train_classifier.py` | Train the MLP, reporting cross-person accuracy → `landmark_model.joblib` + `landmark_labels.json` | `--quick` skips the report |
| `python evaluate.py` | Accuracy benchmark against known target letters; asks who is signing and under what condition → appends to `eval_results.csv` | `SPACE` start 4s capture · `n` skip · `q` quit |

`collect_data.py` asks who is signing and **appends** (never overwrites), so data
from multiple people accumulates.

**Collect from more people, not more frames per person.** Measured on this
dataset, testing against a signer the model had never seen:

| Signers in training | Mean cross-person accuracy |
| --- | --- |
| 3 | 79.3% |
| 4 | **84.0%** |

Adding the fourth signer was worth **+4.7 points** for one 10-minute session.
More rows from people already in the set is worth nothing by comparison
(4,352 → 14,509 rows from the same two people moved accuracy 80.2% → 80.4%).

So aim for **~100 rows per label per person**, then move on to the next person.

**What to vary while recording** — and this is counter-intuitive, because the
normalization erases most of what looks like variation:

| Variation | Effect on the 42 features |
| --- | --- |
| Moving the hand around the frame | **none** (~1e-16) |
| Moving closer or further away | **none** |
| Rotating it in the image plane | **none** |
| **Tilting it toward / away from the camera** | real |
| **Varying finger curl, thumb position** | real |

The hand frame puts the origin at the wrist, scales by wrist->middle-knuckle and
rotates the thumb onto +x, so position, distance and in-plane rotation are
removed *by construction* — the same invariance that makes G and Q collapse
together. So **tilt the hand and vary the handshape**; waving it around the
frame produces duplicate feature vectors however different the picture looks.

`evaluate.py` records **both the signer and the condition on every row**, so a
run spanning several people and several environments can be split by either
afterwards — one label for the whole run cannot tell "this person struggles"
from "this lighting is hard". It is the real accuracy number, distinct from the
training/validation split printed by `train_classifier.py`: it runs the *whole* live pipeline
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

`evaluate.py` scores with a substring test, so a committed string of
`QQQQQQQQQQ` counts as a correct `Q`. Both runs above predate the debouncer fix
and contain many such strings — see `NOTES.md`.

### Model accuracy, measured honestly

`train_classifier.py` reports **leave-one-person-out** accuracy: train on
everyone else, test on a signer the model has never seen.

| Measurement | Result |
| --- | --- |
| Random 15% row split | 98.6% — **inflated, do not quote** |
| **Held-out person, mean** | **84.0%** |
| — held out Nourhan | 89.2% |
| — held out Riad | 88.0% |
| — held out Omar | 83.8% |
| — held out Laila | 75.2% |

Four signers, 24,496 rows. The spread between signers is itself a result: the
system's accuracy depends on how close a new signer's hand geometry is to those
already collected, which is the argument for collecting more people rather than
more frames.

The gap is near-duplicate leakage: frames inside one recording burst are about
5× closer to each other than two random frames of the same label, so a shuffled
split trains on frame 200 and tests on frame 201. The held-out-person number is
what a stranger at a demo experiences.

`ILY`, `IHATEYOU` and `HELLO` are **still excluded** from the figure above. The
original rows were recorded by three signers in one sitting with no boundary in
the file showing where one stops, so they are marked `unknown` (trained on in
every fold, never tested on). Riad is the only identified signer for them, and
cross-person testing needs two. One more person signing those three closes it
— about 90 seconds of recording.

The largest cross-person confusions are **K↔P** (415/410) and **S↔N** (345/242)
— both bigger than G→Q (146), which the in-sample number hid entirely.

## Tests

```bash
python -m unittest discover -p "test_*.py" -v
```

These run automatically on every push and pull request via GitHub Actions
(`.github/workflows/tests.yml`), so a regression shows up as a red ✗ on the
pull request. The runner has no camera and no trained model, so CI covers
logic regressions only — recognition accuracy still comes from running
`evaluate.py` by hand.

`debouncer.py` is pure logic, so it is tested offline against synthesised frame
streams with explicit timestamps — no camera, MediaPipe or trained model
required. `python test_debouncer.py` additionally replays the repeated strings
recorded in `eval_results.csv` through both the old and the new implementation.

## Repository layout

| File | Role |
| --- | --- |
| `main.py` | Live pipeline |
| `collect_data.py` | Labeled landmark data collection |
| `train_classifier.py` | MLP training + per-class metrics |
| `evaluate.py` | End-to-end accuracy benchmark |
| `motion.py` | J/Z trajectory detection (needs ~15fps, see `NOTES.md`) |
| `debouncer.py` | One commit per letter run (wall-clock timed) |
| `nlp_bridge.py` | SymSpell correction, word-sign lookup |
| `audio.py` | Windows TTS output |
| `test_debouncer.py` | Offline debouncer tests (no camera needed) |
| `test_dataset.py` | Offline schema/grouping tests |
| `test_motion.py` | Offline motion-rule tests (synthetic landmarks) |
| `dataset.py` | Dataset schema + person grouping (stdlib only) |
| `hands.py` | Two-hand + orientation feature layer (Veronica stage 1, stdlib only) |
| `location.py` | Face-anchored sign location (Veronica stage 2, stdlib only) |
| `sequence.py` | Clip → fixed-length sign vector (Veronica stage 3, stdlib only) |
| `signset.py` | Sign dataset schema, vocabulary, archive→row (Veronica stage 4, stdlib only) |
| `collect_signs.py` | Two-handed sign clip collection (Veronica stage 4) |
| `rebuild_signs.py` | Regenerate the training CSV from the raw clip archive |
| `folds.py` | Leave-one-person-out fold construction (Veronica stage 5, stdlib only) |
| `train_signs.py` | Sign classifier training + cross-person report (Veronica stage 5) |
| `gloss.py` | ASL gloss → English (Veronica stage 7, stdlib only) |
| `segment.py` | Continuous sign segmentation (Veronica stage 6, stdlib only) |
| `capture.py` | The single MediaPipe adapter seam (Veronica) |
| `config.py` | Persisted capture conventions (Veronica, stdlib only) |
| `demo_veronica.py` | Live Veronica pipeline — runs without a trained model |
| `check_setup.py` | Preflight: deps, models, camera fps, handedness convention |
| `test_hands.py` | Offline feature-layer tests (no camera needed) |
| `test_location.py` | Offline location-layer tests (no camera needed) |
| `test_sequence.py` | Offline movement-layer tests (no camera needed) |
| `test_signset.py` | Offline sign-schema / reconstruction tests (no camera needed) |
| `test_folds.py` | Offline fold-construction tests (no camera needed) |
| `test_gloss.py` | Offline gloss→English tests (no camera needed) |
| `test_segment.py` | Offline continuous-segmentation tests (no camera needed) |
| `backfill_person.py` | One-off: adds `person` to pre-existing rows |
| `landmark_data.csv` | Training data (21,526 rows, 27 classes, 3 people) |
| `landmark_model.joblib` / `landmark_labels.json` | Trained model + label order |
| `eval_results.csv` | Accumulated evaluation log |
| `hand_landmarker.task` | MediaPipe hand landmarker model |

### Branches

- **`main`** — the ML pipeline. This is the thesis basis.
- **`trial`** — a rule-based, non-ML geometric classifier (`rules.py` +
  `motion.py`). Kept as a **fallback / last resort only**. It proved the
  landmark-geometry concept before `main` adopted an ML version of the same idea.
- **`project-veronica`** — the extension from fingerspelling to actual ASL:
  two hands, orientation, location, movement and phrases. See `VERONICA.md`
  for the staged plan. It adds files rather than rewriting the pipeline, so
  the thesis basis here keeps running throughout.

> Note that `main` is currently **behind** the pipeline work — the debouncer
> fix, the J/Z travel gate and the cross-person accuracy methodology were
> merged onto the feature branch, not onto `main`. Veronica is cut from that
> tip, not from `main`.
