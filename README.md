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
| `python evaluate.py` | Accuracy benchmark against known target letters; asks who is signing and under what condition → appends to `eval_results.csv`. `--width/--height/--fps` emulate slower hardware | `SPACE` start 4s capture · `n` skip · `q` quit |
| `python hw_report.py` | Summarise `eval_results.csv` by hardware profile | — |

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

**Keep takes short and disciplined.** One signer recorded twice, first in ~20s
takes and later in ~8s takes, is read by a model that has never seen her at
74.4% and 92.1% respectively — same hands, same letters. Long takes drift into
poses far from where other signers sit. Roughly 100 rows (~8 seconds) per label
is both enough data and short enough to stay consistent.

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

Five signers, 30,807 rows, averaged over three random seeds (see the note on
seed variance below).

| Held out | All 27 labels | Letters | Word signs |
| --- | --- | --- | --- |
| Hagar | 93.5% | 92.7% | 100% |
| Nourhan | 90.8% | 90.8% | — |
| Riad | 89.7% | 88.5% | 99.1% |
| Omar | 83.0% | 83.0% | — |
| Laila | 79.6% | 78.9% | 97.4% |
| **Mean** | **87.3%** | **86.8%** | **~98.8%** |

| Measurement | Result |
| --- | --- |
| Random 15% row split | 98.6% — **inflated, do not quote** |
| **Held-out person, letters** | **86.8%** |
| **Held-out person, all 27** | **87.3%** |

**The word signs transfer far better than the alphabet** (~98.8% vs 86.8%), and
that is structural rather than luck: `ILY`, `IHATEYOU` and `HELLO` are grossly
distinct handshapes, while the alphabet contains pairs the normalization
genuinely collapses (see below). It argues for growing the word-sign vocabulary
rather than chasing the last points on fingerspelling.

**Quote a seed-averaged figure, never a single run.** Refitting one fold with a
different `random_state` moves it by up to 5.4 points on the same data. Every
number above is the mean of three seeds; single-run figures carry roughly
+/-2-3 points of noise, which is wider than most of the differences anyone would
want to draw conclusions from.

The gap is near-duplicate leakage: frames inside one recording burst are about
5× closer to each other than two random frames of the same label, so a shuffled
split trains on frame 200 and tests on frame 201. The held-out-person number is
what a stranger at a demo experiences.

`ILY`, `IHATEYOU` and `HELLO` are now validated across people: Riad, Hagar and
Laila have each signed them with the signer recorded. The original 1,299 rows
stay marked `unknown` (three signers in one sitting, no boundary in the file
showing where one stops) and are trained on in every fold but never tested on.

The largest cross-person confusions are **K↔P** (415/410) and **S↔N** (345/242)
— both bigger than G→Q (146), which the in-sample number hid entirely.

## Running on a Raspberry Pi

Setup and the first measurement, in order. **Raspberry Pi OS must be 64-bit** —
the MediaPipe wheels are arm64 only.

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3-venv git libgl1 libglib2.0-0 libegl1 libgles2

git clone https://github.com/HurryPatter/ASL-Glasses.git
cd ASL-Glasses

python3 -m venv .venv            # required: Pi OS Bookworm refuses system-wide pip
source .venv/bin/activate
pip install -r requirements.txt

python bench_pi.py --seconds 120          # the number that decides the board
python bench_pi.py --width 320 --height 240 --seconds 120
```

`libegl1`/`libgles2` are needed by MediaPipe 1.0's native library even with no
display attached; the Desktop image has them, a Lite image may not, and the
failure (`OSError: libEGL.so.1: cannot open shared object file`) is not obvious.

**`requirements.txt` pins exact versions, deliberately.** The model is a pickle
and scikit-learn only guarantees one loads correctly under the version that
wrote it, so the model and the pins are a matched pair: change them together
with a retrain, never separately. The pinned set is verified to resolve on the
Pi 5 under both Python 3.11 (Bookworm) and 3.13 (Trixie).

`bench_pi.py` runs the hand landmarker with no GUI and no classifier, printing
fps every 10s along with core temperature. Run it for minutes, not seconds: a
Pi throttles as it heats, and the sustained figure is the one that matters.

| Measured fps | What works |
| --- | --- |
| >= 25-30 | motion signs (J/Z) too — full alphabet |
| 12-25 | static letters and word signs; J/Z unreliable |
| < 12 | static letters degrade as well, badly at low resolution |

**Known blockers on Linux, both expected:**

- `audio.py` shells out to Windows PowerShell and will not run. Replace with
  `espeak-ng` (tiny, robotic) or `piper` (neural, much better for a demo).
  Porting it also removes the unescaped-interpolation bug in that file.
- A **CSI ribbon camera** is not visible to `cv2.VideoCapture` on Bookworm,
  which uses libcamera; that needs `picamera2`. A **USB webcam** works with the
  existing code unchanged, so start there.
- The Pi 5 has no 3.5mm jack (the PCIe slot took its place). Use a USB audio
  adapter or an I2S DAC.

`main.py`, `collect_data.py` and `evaluate.py` all open a preview window, so
they need a desktop session or VNC. `bench_pi.py` does not, which is why it is
the first thing to run.

## Choosing the embedded target

The hardware decision turns on two numbers that can be measured on the
development laptop, for free, before anything is bought:

```bash
python evaluate.py                                # baseline
python evaluate.py --width 320 --height 240       # does accuracy survive low resolution?
python evaluate.py --letters JZ --repeat 10 --fps 15   # where do the motion signs die?
python evaluate.py --letters JZ --repeat 10 --fps 20
python evaluate.py --letters JZ --repeat 10            # uncapped, for comparison
python hw_report.py                               # compare them
```

**Press X to void a capture you fumbled.** A forgotten or wrong handshape is
your error, not the pipeline's, and on a 25-capture run one of them moves the
result by four points — enough to swamp the effect being measured. X deletes
the row just written and re-queues the letter.

**Use `--letters JZ --repeat 10` for frame-rate questions.** A full alphabet
pass yields one J and one Z per five-minute run, which is the worst possible
sampling for the only signs that clearly degrade with frame rate. The letter
set cycles rather than blocks (J Z J Z ... not J J ... Z Z) so fatigue does not
load onto one sign.

`--fps` drops frames that arrive early rather than sleeping, so it emulates a
board that cannot keep up. Timestamps stay real throughout, which is why
`debouncer.py` and `motion.py` were written against the wall clock rather than
frame counts — they behave under emulation exactly as they would on slow
hardware.

**There is a hard floor at ~10.8fps.** `motion.py` needs 8 trajectory samples
inside a 650ms window, so below `(8-1)*1000/650` **J and Z cannot fire at all**,
however well the gesture is performed. Static letters have no such cliff. That
is why `hw_report.py` breaks them out separately: a profile whose J/Z column
collapses while static holds is hitting the floor, not losing accuracy.

Resolution matters more than it looks: the pipeline consumes landmarks, not
pixels, and reported figures put MediaPipe at 25+fps at 320x240 against 8-15 at
default — a larger effect than the gap between candidate boards. If accuracy
holds at 320x240, the hardware requirement drops sharply.

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
