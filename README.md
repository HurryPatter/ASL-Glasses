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
| `python hw_report.py` | Summarise `eval_results.csv` by device and hardware profile | — |

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

`evaluate.py` records **the signer, the condition and the device on every row**
(the device is detected: `pi5` on the Pi, `laptop` otherwise), so a
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

`evaluate.py` runs to date at the camera's own resolution and frame rate,
full alphabet including J/Z (the emulated low-resolution / low-fps runs are
under *Choosing the embedded target*; `python hw_report.py` prints both):

| Device | Signer | Condition | Score | Miss |
| --- | --- | --- | --- | --- |
| laptop | (unrecorded) | `green_bg_dim` | 25/26 | G read as Q |
| laptop | (unrecorded) | `white_bg_dim` | 25/26 | Z read as X |
| laptop | Nourhan | `green_bg_indoor` | 25/26 | W read as X |
| laptop | Laila | `outdoor` | 24/26 | S, U nothing committed |
| laptop | Hagar | `indoor_brown_bg` | 25/26 | G nothing committed |
| laptop | Omar | `indoor_green_bg` | 25/26 | Q read as Z |
| **laptop** | | **all** | **149/156 (95.5%)** | |
| **Pi 5** | Omar | `indoor_white_bg` | **25/26** | Z read as X |

No single letter fails consistently; Z read as X is the one miss that recurs
(and appears as a leading X in several correct Z captures). See `NOTES.md`.

`evaluate.py` scores with a substring test, so a committed string of
`QQQQQQQQQQ` counts as a correct `Q`. The first two runs predate the debouncer
fix and contain many such strings — see `NOTES.md`.

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

Use the `pi5` branch. **Raspberry Pi OS must be 64-bit** — the MediaPipe wheels
are arm64 only, and on 32-bit `pip` fails with an error that never says so.

```bash
git clone -b pi5 https://github.com/HurryPatter/ASL-Glasses.git
cd ASL-Glasses
bash setup_pi.sh            # add --piper for the natural-sounding voice
source .venv/bin/activate   # every new terminal

python camera.py                       # camera gives a picture?
speaker-test -t wav -c 2 -l 1          # hear anything at all?
python audio.py "hello"                # speech works?
python bench_pi.py --seconds 120       # THE number: fps and ms per frame
```

`setup_pi.sh` checks the architecture, installs the system packages, builds the
venv (Pi OS refuses system-wide `pip`), installs the pinned requirements, and
finishes by confirming the model loads with no version warning and naming the
speech backend it found. Safe to re-run.

Three system packages are easy to miss and are included: **`libegl1`/`libgles2`**
(MediaPipe 1.0's native library needs libEGL even with no display — a Lite image
may lack it, and `OSError: libEGL.so.1` does not say what to install),
**`espeak-ng`** (speech), and **`alsa-utils`** (`aplay` and `speaker-test`).

**`requirements.txt` pins exact versions, deliberately.** The model is a pickle
and scikit-learn only guarantees one loads correctly under the version that
wrote it, so the model and the pins are a matched pair: change them together
with a retrain, never separately. The pinned set is verified to resolve on the
Pi 5 under both Python 3.11 (Bookworm) and 3.13 (Trixie).

### Speech

`audio.py` picks a backend automatically: Windows' built-in voice on the
laptops, `espeak-ng` on the Pi, and **Piper** (neural, far more natural — worth it
for the defense) if you ran `setup_pi.sh --piper` and set

```bash
export ASL_PIPER_MODEL=$HOME/ASL-Glasses/voices/en_US-lessac-medium.onnx
```

Force a backend with `ASL_TTS=espeak|piper|sapi|say|print`. With no TTS
installed it prints the text rather than crashing, so losing speech never takes
recognition down with it. Text is passed as data (stdin or an environment
variable), never spliced into a command, so apostrophes and quotes are safe.

The Pi 5 has **no 3.5mm jack**. Use a USB audio adapter, then make it the
default output: `aplay -l` lists devices, and on the desktop, right-click the
speaker icon on the taskbar to choose it.

### Benchmark first

`bench_pi.py` runs the hand landmarker with no GUI and no classifier, printing
fps every 10s along with core temperature. Run it for minutes, not seconds: a
Pi throttles as it heats, and the sustained figure is the one that matters.
**Keep a hand in view**: with no hand MediaPipe skips the landmark model, so an
empty run measures the cheaper path (the script warns when that happens).

It also prints the per-frame processing time (median and 95th percentile),
split by hand / no hand. The ribbon camera stops at 30fps, so on a Pi 5 fps
reads ~30 whatever the headroom; the processing time is what shows how far
below the camera's 33 ms frame interval the board actually is.

| Measured fps | What works |
| --- | --- |
| >= 25-30 | motion signs (J/Z) too — full alphabet |
| 12-25 | static letters and word signs; J/Z unreliable |
| < 12 | static letters degrade as well, badly at low resolution |

First Pi 5 results (ribbon camera, 640x480): `bench_pi.py` held 29.9fps for
120s at 55-60°C, i.e. the camera's 30fps ceiling with no throttling; the full
`evaluate.py` pipeline ran at ~25fps and scored 25/26, every static letter
correct and Z committed as X. One signer, one run: a first data point, not
yet the thesis figure.

### Camera

Both kinds work, through `camera.py`. A **ribbon (CSI) camera** is used if one
is attached, otherwise a **USB webcam**; force either with `ASL_CAMERA=csi` or
`ASL_CAMERA=usb`. Check with `python camera.py`.

A ribbon camera cannot go through `cv2.VideoCapture`: Pi OS drives it with
libcamera, and on a Pi 5 `/dev/video0` is the raw sensor front end. It goes
through Picamera2 instead, fixed at 30fps so the camera cannot slow itself
down in dim light and starve the J/Z detector. Picamera2 is only distributed
through apt, which is why `setup_pi.sh` builds the venv with
`--system-site-packages`; the pinned packages still install into the venv and
take precedence.

**Dim rooms look dark on the ribbon camera, by design.** At a fixed 30fps no
exposure can be longer than 33ms, so where a webcam would drop to 15fps and
look bright, the ribbon camera raises gain instead (darker, noisier, but J/Z
keep working). `python camera.py` saves `camera_check.jpg` and says whether
the camera has hit that limit (`LIGHT-LIMITED`). If it has, put light on the
hand; `ASL_CAMERA_EV=1 python main.py` (or `2`) brightens further using gain.
Camera Module 3's autofocus is set to continuous so a hand at arm's length
stays sharp.

**Open: the first Pi run rendered grey curtains deep blue.** A red/blue swap
would leave grey grey, so this points at white balance (camera module, tuning
or mixed lighting) rather than channel order. To settle it, compare
`camera_check.jpg` from `python camera.py` (our pipeline, which also prints the
white-balance temperature) with `rpicam-still -o still.jpg` (the Pi's own app,
bypassing our code), and note the module from `rpicam-hello --list-cameras`.

**The Pi 5's camera sockets are 22-pin, smaller than older Pis' 15-pin.** A
standard Camera Module needs a **22-to-15-pin adapter cable**. Connect it only
with the Pi powered off.

### Network

Raspberry Pi Imager can only set up Wi-Fi that takes a network name and a
password. **Campus networks that ask for a username as well (WPA2-Enterprise)
cannot be set in Imager** — the Pi never connects. Set it up on a phone hotspot
(or Ethernet), then add the campus network from the Pi's desktop Wi-Fi menu,
which does support it.

### Display

`main.py`, `collect_data.py` and `evaluate.py` open a preview window, so they
need the desktop: a monitor, or Screen Sharing in Raspberry Pi Connect.
`bench_pi.py`, `audio.py` and `camera.py` do not.

### Working remotely (no monitor)

- **Raspberry Pi Connect** (connect.raspberrypi.com, signed in with the same
  Raspberry Pi ID the Pi is linked to): *Remote shell* for the command line,
  *Screen sharing* for the desktop. Works from any network, campus included.
  If the Pi is not listed, link it from a terminal on the Pi:
  `rpi-connect on && rpi-connect signin`, then `loginctl enable-linger` so it
  survives reboots. The site's "New auth key" page is for flashing a new SD
  card in Imager — not needed for a Pi that is already running.
- **SSH** from a laptop on the same network: `ssh thesis@aslpi.local`.
- **Close Screen Sharing before benchmarking** — streaming the desktop costs
  CPU and lowers the fps being measured. Run `bench_pi.py` over SSH or the
  Remote shell; for a thesis-grade `evaluate.py` run, prefer a monitor.

### Getting results off the Pi

The Remote shell cannot download files. Either copy them from the laptop,
in PowerShell (not inside an SSH session — `scp` runs on the receiving
machine, and a Windows path typed on the Pi turns into `/home/thesisDownloads`):

```powershell
scp thesis@aslpi.local:~/ASL-Glasses/eval_results.csv $HOME\Downloads\
```

or give the Pi an SSH key on GitHub once (`ssh-keygen -t ed25519`, add
`~/.ssh/id_ed25519.pub` under GitHub → Settings → SSH keys, then
`git remote set-url origin git@github.com:HurryPatter/ASL-Glasses.git`) and
commit and push from the Pi like anywhere else.

`eval_results.csv` records a **`device`** column (`pi5` on the Pi, `laptop`
otherwise; `ASL_DEVICE` overrides), so Pi and laptop runs are never pooled.

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
| `audio.py` | Text-to-speech: Windows voice, espeak-ng or Piper, auto-detected |
| `setup_pi.sh` | One-command Raspberry Pi setup |
| `camera.py` | Opens a ribbon camera or USB webcam behind one interface; `python camera.py` checks exposure and colour |
| `bench_pi.py` | Landmarker throughput and per-frame latency on new hardware |
| `hwprofile.py` | Frame-rate emulation, device detection, results-file layout |
| `hw_report.py` | `eval_results.csv` summarised by device and hardware profile |
| `test_debouncer.py` | Offline debouncer tests (no camera needed) |
| `test_dataset.py` | Offline schema/grouping tests |
| `test_motion.py` | Offline motion-rule tests (synthetic landmarks) |
| `test_audio.py` | Offline speech tests (no speaker needed) |
| `test_camera.py` | Offline camera-selection tests (no camera needed) |
| `test_nlp_bridge.py` | Spell-correction tests |
| `test_hwprofile.py` | Frame limiter, device detection, results migration |
| `dataset.py` | Dataset schema + person grouping (stdlib only) |
| `backfill_person.py` | One-off: adds `person` to pre-existing rows |
| `landmark_data.csv` | Training data (30,807 rows, 27 classes, 5 signers) |
| `landmark_model.joblib` / `landmark_labels.json` | Trained model + label order |
| `eval_results.csv` | Accumulated evaluation log |
| `hand_landmarker.task` | MediaPipe hand landmarker model |

### Branches

- **`main`** — the ML pipeline. This is the thesis basis.
- **`pi5`** — `main` plus everything needed to run on the Raspberry Pi 5
  (ribbon camera, Linux speech, pinned dependencies, hardware emulation and
  benchmarking). This is what runs on the Pi; it merges back into `main`.
- **`hardware-emulation`** — the laptop-side emulation work, since folded
  into `pi5`.
- **`trial`** — a rule-based, non-ML geometric classifier (`rules.py` +
  `motion.py`). Kept as a **fallback / last resort only**. It proved the
  landmark-geometry concept before `main` adopted an ML version of the same idea.
