# Project notes — history, known issues, next steps

Working notes for the ASL Smart Glasses pipeline. See `README.md` for how to
run things; this file is the "why it looks like this" and "what's still broken"
record. **Thesis due late December 2026** — the glasses build needs to be
running on embedded hardware by then.

## Architecture history

1. **Pixel CNN (original).** Cropped hand image → skin-color HSV segmentation →
   Keras CNN trained on Sign Language MNIST. Real problems: background and
   lighting sensitivity (only worked against a white background), a
   false-positive-prone skin-color detector, and aspect-ratio distortion from
   non-square crops.
2. **MediaPipe rewrite.** Replaced skin-color segmentation with the MediaPipe
   Hand Landmarker — 21 3D landmarks per frame instead of a raw pixel crop.
3. **Rule-based branch (`trial`).** `rules.py` classifies letters purely from
   hand-crafted geometric tests on normalized landmarks, no trained model at
   all. Kept as a fallback, since the thesis basis is ML.
4. **Landmark ML pipeline (`main`, current).** Normalized landmarks → a small
   scikit-learn `MLPClassifier` (64, 32). Deliberately **not** TensorFlow/Keras:
   TF's native DLL failed to load on the Windows / Microsoft-Store-Python dev
   setup, a 42-input MLP does not need a deep learning framework, and dropping
   TF fits the "lightweight" objective. J/Z still come from `motion.py`.
5. **Result.** The pipeline is fully on-device — MediaPipe + scikit-learn, no
   cloud call, no TensorFlow.

> **The thesis slides are out of date.** They still describe a cloud-inference
> CNN architecture. What was actually built (fully on-device) is both accurate
> and a stronger story. Update before the defense.

## Known issues

### FIXED: reported accuracy was measured the wrong way

`train_classifier.py` used a random 15% row split. Frames inside one recording
burst are near-duplicates of a single held pose — measured on the collected
data, consecutive frames in a burst sit about 5x closer together than two random
frames of the same label. A shuffled split therefore trains on frame 200 and
tests on frame 201, measuring memorisation rather than recognition.

Measured on the three signers, on letters only:

| Measurement | Accuracy |
| --- | --- |
| Random 15% row split | 98.6% |
| Leave-one-person-out | **79.3%** (omar 81.9, laila 74.0, nourhan 83.1) |

It also answers how much more data to collect:

| Training set | Accuracy on an unseen signer |
| --- | --- |
| 1 person | 71.9% |
| 2 people | 81.7% |
| 2,176 rows (2 people) | 77.4% |
| 4,352 rows | 80.2% |
| 7,254 rows | 81.3% |
| 14,509 rows | 80.4% |

Rows saturate around 4,000; people do not. The second signer was worth about
+10 points. Collect **~100 rows per label per person** and then recruit the next
person — roughly 8-10 people is the conventional target, though with only
three signers that extrapolation is not something this data proves.

`landmark_data.csv` now carries a `person` column (`backfill_person.py`
attributed the historical rows from the recording structure), `collect_data.py`
asks who is signing, and `train_classifier.py` reports leave-one-person-out as
the headline with the random-split figure printed only as an explicitly
inflated comparison.

**`ILY`, `IHATEYOU` and `HELLO` cannot be validated across people.** All three
signers recorded them in one sitting, and since each label is a single
continuous run there is no boundary in the file marking who signed what. Their
rows are marked `unknown`: trained on in every fold, never tested on. Guessing
an attribution would corrupt the grouping every number here depends on.
Re-collect them with the signer recorded.

Cross-person confusions rank differently from the in-sample ones, which is
itself a result: **K->P 415, P->K 410, S->N 345, N->S 242**, all larger than
**G->Q 146** (and G->X 168 is larger still). The K/P and S/N pairs were
invisible in the 98.6% figure and are now the biggest accuracy problem —
bigger than the G/Q question that had been the open one.

### FIXED: a single steady hold committed the same letter many times

`eval_results.csv` recorded committed strings like `QQQQQQQQQQ`, `GGGGGGGGGGG`
and `IIIIIIIIII` from one continuous hold. The old debouncer treated `""` as
just another label, so any frame without a reading reset the hold and the
letter re-committed a few frames later. `main.py` produces `""` for three
separate reasons, none of which mean the signer let go:

1. confidence at or below 0.85 (`main.py:132`),
2. `is_motion_candidate()` suppressing static classification (`main.py:119`),
3. tracking dropout past `MISSED_FRAME_TOLERANCE` (`main.py:158`).

Source 2 explains the worst strings. Checking `_is_point_shape` /
`_is_i_shape` over all 21,526 rows of `landmark_data.csv`: X (100%), L (99%),
I (99%), Y (98%), G (98%) and Q (84%) all satisfy the motion gate while held
perfectly still, and those are exactly the letters with the longest repeat
runs. A/B/C/F/W never trip it and never repeated. The gate only needs 0.08
hand-widths of drift over 180ms, which ordinary tremor clears.

`debouncer.py` now distinguishes absence of evidence (`""`) from evidence of a
different handshape, and commits once per letter *run* rather than per hold.
See `test_debouncer.py`, which replays the recorded strings through both the
old and new implementations.

**Still worth tuning separately:** `is_motion_candidate()` is over-broad. It
fires on six static letters, and the debouncer now absorbs the consequences
rather than removing the cause. Raising `deadband` or requiring sustained
directional motion would cut the blank frames at the source.

**`evaluate.py` scores `target in committed`** (line 148), a substring test, so
`QQQQQQQQQQ` counted as correct. The 96.2% figure is blind to this class of
bug. Tightening it to exact-match once the fix is validated on camera would
give a more defensible thesis number, and re-scoring the existing
`eval_results.csv` under exact-match is a free before/after measurement.

### FIXED: a held Q could emit Z repeatedly

Observed live: holding Q with the palm toward the camera produced `Z Z Z Z`.
With the back of the hand toward the camera it did not.

Q is a pointing handshape by this module's test — 84% of the recorded Q
samples pass `_is_point_shape` — so holding a Q puts the Z trajectory rules
under continuous evaluation. Palm forward, the curled fingers are
self-occluded, MediaPipe's estimate of them wobbles, and `_is_z()` measured the
zigzag on the **index fingertip alone**, so a fingertip oscillating in place
could satisfy "three alternating strokes" by chance. `_fire()` clears the
buffer, so it re-armed and fired roughly 3-4 times a second.

It also erased the correct answer: `commit_motion()` drops static commits made
inside the gesture span, so the false Z deleted the Q that had been recognised
properly. That is why the raw output showed `Z Z Z Z` and not `Q Z Z Z Z`.

The fix is a gate, not a threshold tweak: `_is_z()` now also requires the
**whole hand** to have travelled `z_travel` (0.3 hand widths) horizontally,
measured on the centroid of all 21 landmarks via the new
`MotionDetector.hand_travel()`. Averaging 21 landmarks cancels most
per-landmark jitter while a real translation survives, so it separates "the
hand went somewhere" from "the fingers wobbled in place". A hand that has not
moved has not signed a motion letter, however noisy its landmarks — retuning
`z_stroke` would only have moved the noise floor.

`debug_info()` now reports `hand_travel` so the threshold can be tuned against
a real Z rather than guessed.

**This was already intermittent before the fix** — it stopped reproducing
after an unrelated retrain, which cannot have affected it: `motion.py` never
consults the model, and `motion_detector.update()` runs at `main.py:116`,
before any `predict_proba` call. The jitter that triggered it depends on
lighting, hand angle and distance, so absence of the symptom was never
evidence of a fix.

### Motion signs need about 11fps at all (constrains the hardware choice)

`min_samples` (8) has to be met *inside* `window_ms` (650ms), so the camera
must deliver 8 samples within 650ms: `(8-1)*1000/650` — about **10.8fps**.
Below that the buffer is pruned by time before it ever fills, and **no J or Z
can fire however the gesture is performed**. Verified in `test_motion.py`:
fires at 30 and 15fps, never at 10.

This is a floor on the embedded target, not a tuning preference. Whatever
hardware is chosen has to sustain ~15fps of MediaPipe inference with headroom,
or the motion letters stop existing — and lowering `min_samples` to
compensate buys noise, since the floor is there to stop 2-3 samples firing a
gesture.

### Z occasionally never fires (read as X instead)

Seen in the `white_bg_dim` run: expected `Z`, committed `XX`. In the earlier
`green_bg_dim` run the same sign committed `XZZZ` — an X leaked through before
Z fired. So this is flaky rather than dead.

Likely root cause is the **window length**, in `motion.py`:

- `MotionDetector.window_ms` defaults to **650 ms**, and the buffer is pruned by
  time on every `update()`.
- `_is_z()` requires **three alternating strokes plus net downward drift to all
  be visible in the buffer simultaneously**.
- A deliberately-drawn Z takes most people noticeably longer than 650 ms. By the
  time the third stroke completes, the samples from the first stroke have
  already been pruned, so `_strokes()` never sees three completed strokes at
  once, and `drifted_down` under-measures the total drop because it is computed
  across a truncated span.

This also explains why **J is reliable in both runs and Z is not** — J is a
quick flick that fits inside 650 ms, whereas Z is three deliberate strokes.

Suggested fix (**not implemented** — needs a camera to validate): give Z its own
longer window rather than widening `window_ms` for both, since J does not need
it and a longer J window risks false positives:

```python
def __init__(self, window_ms=650, z_window_ms=1200, ...):
```

...then have `_is_z()` evaluate over the full buffer while `_is_j()` keeps
looking only at the trailing `window_ms`, and prune at `max(window_ms,
z_window_ms)`. The travel gate above should land first and has: widening the
window increases false-positive exposure, which is exactly what the gate
bounds. Instrument first: `debug_info()` already reports `z_strokes` and
`z_drift` live, so watch those while signing Z to confirm the stroke count is
what is actually falling short before changing thresholds.

### G/Q confusion — watch, do not fix yet

The first evaluation run (`green_bg_dim`) missed only G, reading it as Q.
Hypothesis: G and Q are the same handshape distinguished only by orientation,
and the hand-frame normalization deliberately removes rotation, which would make
them genuinely indistinguishable in feature space.

**The second run (`white_bg_dim`) read G correctly** (11 consecutive G commits),
so this is not currently a recurring failure. A fix was drafted — a runtime-only
orientation override, no retraining needed — and deliberately **not**
implemented: it is not worth the added runtime complexity on one miss. Revisit
only if G→Q recurs across more runs.

### `audio.py` interpolates text into a PowerShell command unescaped

```python
f'$s.Speak("{text}")'
```

A corrected string containing a quote, `;` or `$` can break or inject into the
shell command. Note that the NLP layer can legitimately produce apostrophes, so
this is reachable in normal use, not just adversarially. Not yet fixed.

### Hardware target still undecided

True MCU (ESP32/STM32-class) vs. small Linux SBC (Pi Zero 2 W / Jetson-Nano
class). This matters a lot: MediaPipe needs real compute and does not run on
bare MCUs, so an MCU target likely means the glasses only do capture and
streaming while MediaPipe and inference run on a paired phone or server — which
would give up the "fully on-device" result. Needs to be locked down soon.

## Deliberate non-changes

- **`debouncer.py` commits once per letter run.** Do **not** change this to a
  periodic-repeat scheme — that was tried and reverted. Double letters (the
  "LL" in HELLO) come from the signer briefly bouncing out of the shape and
  back in; that bounce must now last longer than `blank_ms` (250ms) to read
  as two runs.
- **`motion.py` uses wall-clock time windows, not frame counts.** A frame-count
  window means a different real-world duration on every device; embedded
  hardware will not match the dev laptop's fps. Keep it time-based.
- **One hand only** (`num_hands=1`). Two-handed signs are an architecture
  change, not a quick add.

### Rejected word-sign candidates

- **CALM DOWN** — two-handed with real movement. Out of scope (see above).
- **HATE, OK (as real ASL signs)** — both involve genuine movement or a
  handshape transition (HATE: a flick; OK: O→K). The informal "OK circle"
  gesture is also literally identical to the ASL letter F, so it cannot be a
  distinct class anyway. These would need a short two-state transition detector
  like `motion.py`, not a static pose — same risk class as J/Z.

## Current status

- Full pipeline working end-to-end, fully on-device.
- Training data: **21,526 rows across 27 classes** — 24 static letters plus
  `HELLO` (490), `IHATEYOU` (468) and `ILY` (341) — collected from at least two
  people's hands.
- Word signs are collected, trained and live in `landmark_labels.json`.
- Evaluation: two full runs, **50/52 combined (96.2%)**. See README.

## Next steps

1. Collection session with the rest of the team. Target ~100 rows per label
   per person and as many people as possible (8-10); the three word signs
   need a second signer most urgently. Rows per person stop paying off past
   roughly 100/label — see the accuracy issue above.
2. Retrain on the combined dataset.
3. More `evaluate.py` rounds across people and conditions, to build a
   defensible accuracy number and to settle whether G/Q is real.
4. Instrument and fix the Z window (above) — now safer, since the travel
   gate bounds the false positives that widening the window would invite.
5. Validate the debouncer fix on camera — specifically the "LL" in HELLO,
   which now needs a bounce longer than `blank_ms` (250ms). If doubles are
   hard to produce, lower `blank_ms`; if single holds still repeat, raise it.
6. **Decide the embedded hardware target (MCU vs. SBC)** — gates the port off
   the dev laptop.
7. Update the thesis slides to the actual on-device architecture.
