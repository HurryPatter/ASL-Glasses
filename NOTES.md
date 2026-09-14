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
z_window_ms)`. Instrument first: `debug_info()` already reports `z_strokes` and
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

- **`debouncer.py` commits once per continuous hold** (`frame_count ==
  min_frames`). Do **not** change this to a periodic-repeat scheme — that was
  tried and reverted. Double letters (the "LL" in HELLO) are meant to come from
  the signer briefly bouncing out of the shape and back in, producing two
  separate holds and therefore two separate commits.
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

1. More letter data from teammates — multi-person, multi-environment session.
   Class counts are currently uneven (A 618 … G 1045); the word signs are the
   thinnest classes at 341–490 and would benefit most.
2. Retrain on the combined dataset.
3. More `evaluate.py` rounds across people and conditions, to build a
   defensible accuracy number and to settle whether G/Q is real.
4. Instrument and fix the Z window (above) — this is the one confirmed
   recognition bug with a concrete diagnosis.
5. **Decide the embedded hardware target (MCU vs. SBC)** — gates the port off
   the dev laptop.
6. Update the thesis slides to the actual on-device architecture.
