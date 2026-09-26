# Project Veronica — design

How the pipeline works and why it is built this way. For what is done and what
is next, see **[VERONICA.md](VERONICA.md)**.

1. [Data flow](#data-flow)
2. [The representation](#the-representation)
3. [Decisions worth knowing about](#decisions-worth-knowing-about)
4. [Things that bit us on real hardware](#things-that-bit-us-on-real-hardware)
5. [Performance](#performance)
6. [Testing](#testing)

---

## Data flow

```
camera frame
   │
   ├─ capture.py ──── MediaPipe hand landmarker (2 hands) + face detector
   │                  the ONLY module that imports MediaPipe
   │
   ├─ hands.py ────── which hand is which, handshape, orientation,
   │                  the relationship between two hands
   │
   ├─ location.py ─── where the hands are, anchored to the face
   │                        ↓
   │                  107-float frame vector
   │
   ├─ sequence.py ─── a window of frames -> one 371-float sign vector
   │
   ├─ segment.py ──── when to classify, and when to commit a sign
   │                        ↓
   │                  a stream of glosses:  YESTERDAY ME GO SCHOOL
   │
   └─ gloss.py ────── ASL grammar -> English:  "Yesterday, I went to school."
```

| File | Role | Deps |
| --- | --- | --- |
| `capture.py` | MediaPipe adapter — builds detectors, converts their output | MediaPipe |
| `hands.py` | Hand geometry, handedness, the frame vector | stdlib |
| `location.py` | Face-anchored sign location | stdlib |
| `sequence.py` | Clip → fixed-length sign vector | stdlib |
| `segment.py` | Continuous signing: where one sign ends | stdlib |
| `gloss.py` | ASL gloss → English | stdlib |
| `signset.py` | Dataset schema, vocabulary, archive → training row | stdlib |
| `folds.py` | Leave-one-person-out fold construction | stdlib |
| `config.py` | Persisted capture conventions | stdlib |

**Everything except `capture.py` is standard-library only.** That is not
tidiness. CI installs nothing, so a dependency-free core is a core whose logic
is actually tested on every push. A test enforces it, because an `import numpy`
in one of these would not fail loudly — the affected suite would error while
the workflow still looked green.

`capture.py` exists as a single seam for a second reason: **MediaPipe Tasks
reports a face bounding box in pixels while reporting hand landmarks normalized
to [0, 1]**. A pixel box read as normalized puts the face past the corner of the
frame, and every location feature becomes a large, stable, plausible-looking
number. That conversion belongs in exactly one place.

---

## The representation

### The frame vector — 107 floats, one per camera frame

```
[  0: 46)  dominant hand      present · 42 shape · orient_cos · orient_sin · mirrored
[ 46: 92)  non-dominant hand  same layout
[ 92:100)  relationship       both_present · dx · dy · distance · size_ratio
                              · rel_cos · rel_sin · contact
[100:107)  location           face_present · dom(x, y, depth) · non(x, y, depth)
```

**The 42 shape floats are unchanged from `main`.** Verified against all 24,496
existing letter rows: the standard-library port agrees with the numpy original
to 1.8e-15, so the existing dataset and model stay valid.

**Orientation is what the hand frame threw away.** `normalize_landmarks()`
rotates the thumb onto +x, which makes handshape invariant to how the hand is
turned — and also makes G and Q genuinely indistinguishable. Veronica records
the rotation alongside the shape instead of discarding it, as `(cos, sin)`
rather than an angle so that 359° and 1° stay adjacent instead of sitting at
opposite ends of the input range.

**Every block leads with a presence flag.** Zeros are a perfectly valid hand
pose, so without the flag "this hand is missing" and "this hand is at the
origin, unrotated" are the same vector. Same distinction `debouncer.py` had to
make between absence of evidence and evidence of absence.

**Contact is its own feature** because contact is phonemic in ASL — signs
differing only in whether the hands touch are distinct signs — and wrist
separation cannot see it, since two hands can have distant wrists and touching
fingertips.

**Everything relational and locational is in hand-widths or face-widths, never
pixels.** A hand and a face are fixed physical sizes, so dividing by them
cancels camera distance: no depth sensor, no calibration, no second network.
That is what keeps the "fully on-device, lightweight" result intact.

**Face *width* is the unit, not height.** ASL uses head movement grammatically,
and pitching the head compresses a face box's apparent height hard while
leaving its width nearly alone. A height-based unit would move every location
feature by ~45% during exactly the constructions stage 8 has to read. Both axes
divide by that one number, so the frame stays isotropic and an angle measured
in it is a real angle.

**`hand size / face width` is a free depth proxy.** Both are fixed physical
sizes, so the ratio is roughly constant *unless* the hand is nearer the camera
than the face — which is what a sign made out in neutral space does, as against
one contacting the body. A third dimension out of two flat measurements.

### The sign vector — 371 floats, one per clip

```
  2 keyframes × 84 shape floats     handshape barely moves within a sign
  8 keyframes × 23 dynamic floats   orientation, relation, location all move
  2 hands     ×  8 movement floats  net · path length · straightness
                                    · extent · peak speed · reversals
  + per-hand tracking coverage, + clip duration
```

**Each parameter is sampled at the rate it actually changes.** Resampling all
107 floats at 8 keyframes is 856 inputs against the few thousand clips a real
collection effort produces — that is not a model that generalises, it is one
that memorises, and `NOTES.md` already records what happened last time this
project measured memorisation and called it accuracy. The index sets are
derived from the layout rather than written out, so they cannot drift from it.

**Keyframes are placed by time, not by index.** The same gesture at 30 fps and
15 fps agrees to 0.05, and a clip whose samples bunch up mid-gesture resamples
correctly by time and demonstrably wrongly by index. Resampling by frame index
would quietly undo the wall-clock discipline `motion.py` established.

**Interpolated rotations are renormalized.** Blending two unit vectors linearly
gives the chord, not the arc, so a rotation halfway between two keyframes would
otherwise read as a *smaller* rotation — and the error grows with the gap, so
it is worst on exactly the low-frame-rate hardware this has to survive.

**Straightness and reversals are explicit** because ASL makes those
distinctions. A circle returns to its start, so net displacement is near zero
while path length is not — straightness is what separates circular movement
from straight. Reversals count the repetitions many signs carry as part of
their form rather than as emphasis, with a deadband so jitter never becomes a
reversal.

**Trajectory is self-relative; location is body-anchored; both are kept.** The
movement summary measures displacement from where the clip started, in
hand-widths, so it works with no face in frame. The keyframed location block
carries absolute position on the body when a face is there. "The hand arced
downward and reversed twice" and "it did so at the chin" are different facts,
and neither is derivable from the other.

> One consequence worth knowing, because it is easy to trip over: **with no
> face detected, a hand crossing the entire frame produces identical frame
> vectors throughout.** Position is normalized away by construction. The
> trajectory is the only thing that knows it moved, which is why the wrist
> anchors are passed into `SignBuffer.add()` separately rather than read back
> out of the features.

---

## Decisions worth knowing about

### Collect raw, derive features

Collection writes **two** files:

| File | What it is |
| --- | --- |
| `veronica_clips.jsonl` | raw landmarks, one clip per line — **the archive** |
| `veronica_signs.csv` | the 371-float rows, one per whole clip — **derived, regenerable**, used by `inspect_signs.py` |

`train_signs.py` reads the archive directly rather than the CSV, cutting each
clip into live-length windows, so training always reflects the current feature
code even if nobody has run `rebuild_signs.py`.

Sequencing the feature layers before collection protected the signers' time
exactly once; every later change to the representation would have invalidated
every row. Keeping the raw landmarks makes that `python rebuild_signs.py`.

The letter dataset learned this in reverse: `normalize_landmarks()` ran before
anything reached disk, so orientation is gone from those 24,496 rows for good
and no script can recover it. **The archive is the record; the CSV is a build
artifact.** Measured at about 6× the CSV, in a format close to the best case
for the compression git already applies to blobs.

**This has already paid off three times**: a wrong handedness convention, a
change to hand assignment, and the addition of gap bridging and jitter
smoothing were all applied to already-recorded clips with nobody re-signing.

### The acting hand is observed, not asked

`collect_signs.py` asks the signer which hand leads. The glasses meet a
stranger and cannot ask anyone anything, so a canonical space keyed on that
answer is a space the deployed system can never enter.

Whichever hand travels further is the acting hand: with one hand tracked, that
one; with two, the one that moves more — which is also what separates the
acting hand from the base hand in an asymmetric two-handed sign. Both are
readable from the video alone.

Handedness is not phonemic in ASL — a left-handed signer's sign is the mirror
image and means the same thing — so canonicalising on the acting hand loses no
distinction and gains the one that matters: **a sign made with either hand
lands in the same place.** Measured on 250 collected clips, the signing hand
lands in the dominant slot 94% of the time, against 50% before.

Offline that is `signset.acting_hand()`; live it is `hands.ActingHandTracker`,
which decays travel in wall-clock milliseconds so a signer switching hands
mid-conversation is followed rather than outvoted by their own history.

### Continuous signing does not cut first

The obvious approach — find the pauses, or the minima in hand velocity, and cut
there — does not work, for a reason with a name. **Movement epenthesis:** the
transition between two signs is itself movement, and looks like a sign to
anything watching for movement. Signers also do not pause between signs the way
speakers pause between words, so the pauses a cutter would look for are
frequently not there at all.

So `segment.py` classifies a trailing window continuously and commits when the
answer is stable — the same shape as the problem `debouncer.py` already solves
for letters, one level up. Its central distinction generalises exactly:

```
""       no confident reading   -- absence of evidence
"EAT"    a different sign       -- evidence the last one ended
```

Treating the first as a boundary is what produced committed strings like
`QQQQQQQQQQ` from one steady hold. **`Debouncer` is reused, not
reimplemented** — 24 tests, replaying real recorded failure strings, and every
property it has is one this needs: commit once per run, blanks must persist
before releasing, a different label takes over faster than a blank because it
is real evidence, and a genuine release re-arms the same label so `AGAIN AGAIN`
commits twice.

Three defences against epenthesis, in order of how much work they do:

1. **The `_REST` class** — a transition classified as `_REST` produces no
   reading at all. The main defence, and why stage 4 collects `_REST` even
   though nobody signs it.
2. **The confidence floor** — a window spanning two signs is a clean example of
   neither, so the classifier should be unsure.
3. **The hold requirement** — a spurious label must survive several consecutive
   evaluations, and a transition is brief.

There is a test for the honest limit too: a classifier that is confidently
wrong for long enough commits, and no threshold here fixes that.

Evaluation is **strided** — the buffer takes every frame, classification runs
every `stride_ms`. Classifying every frame would mean a full `sign_vector()`
build plus a forward pass at camera rate, on hardware that also has to sustain
MediaPipe.

### Gloss is not English

A recogniser outputs the sequence of signs made. ASL has no copula, marks tense
with a time word rather than a verb inflection, puts wh-words at the end, and
uses no articles.

| Gloss | English |
| --- | --- |
| `YESTERDAY ME GO SCHOOL` | Yesterday, I went to school. |
| `YOU NAME WHAT` | What is your name? |
| `ME SORRY` | I am sorry. |
| `ME NEED GO BATHROOM` | I need to go to the bathroom. |

Rules rather than a model, because of the **failure mode** rather than the
accuracy. When the transformer cannot parse something it returns the gloss,
which is still readable — a Deaf signer reading `ME GO STORE` loses nothing,
while a fluent English sentence that says the wrong thing is worse than no
translation at all. A learned translator needs a parallel corpus this project
does not have and fails in the opposite direction, by inventing fluent text.

`HE-SHE` renders as **"they"**. The ASL sign is a *point*, which carries no
gender; choosing "he" or "she" would invent information the signer did not
give, every time it appears.

The failure this layer guards hardest against is **dropping a sign**. Awkward
English is visibly awkward; a missing sign looks like a clean translation of
something else. Two of the five bugs found by running the first real
vocabulary through it did exactly that.

### Honest measurement

`folds.py` holds everything that decides what goes in which fold, standard
library only and tested in CI. The sklearn call is the easy part to get right;
**a bug in fold construction does not crash and does not look wrong — it
produces a number that is simply too high**, which is the one failure mode this
project has already been bitten by (98.6% shuffled against 84.0% held-out).

Three ways a fold can quietly lie, all handled and all reported *before* the
number rather than after it:

1. **A label only one person signed** is absent from training on that person's
   fold, so it scores ~0 for a reason unrelated to the model.
2. **`_REST` counted as a sign** — it is much easier than any sign, so
   including it inflates the headline. Reported separately.
3. **Folds that are not comparable**, because people signed different numbers
   of labels. Averaging without saying so hides it.

Clips are far more independent than frames were, but the same trap exists one
level up: twenty HELLOs by one person in one sitting are much more like each
other than like anyone else's. Grouping by person closes both.

`train_signs.py` also puts a `StandardScaler` ahead of the MLP. The 42 letter
inputs were all the same kind of quantity in one range, so scaling did nothing;
these 371 are not — landmark coordinates near ±3, clamped distances to 6,
cosines to 1, a duration to 3 — and un-scaled the widest block dominates the
first layer through its units alone.

---

## Things that bit us on real hardware

Each of these passed every offline test and only appeared with a camera
attached. They are recorded because the pattern is the point: the parts of this
system that cannot be tested in CI are exactly the parts that broke.

**MediaPipe's handedness convention runs the opposite way on this build.** It
reports `Left` for a raised right hand while the displayed feed is genuinely
mirrored (the hand sits at x≈0.85, the right side of the frame). Nothing
downstream can detect a wrong convention — the features stay entirely
plausible, training succeeds, and the number at the end is merely wrong. It is
now a setting (`config.py`), recorded into every clip, and correctable across a
whole archive with `rebuild_signs.py --mirrored-input`.

`check_setup.py` reports **two** facts rather than one — what MediaPipe called
the hand, *and* which side of the frame it appeared on — because the label
alone cannot separate two root causes with different implications: a MediaPipe
whose convention runs the other way, and a camera that already delivers a
mirrored feed which `cv2.flip` un-mirrors. The fix is the same flag either way,
but the distinction matters on the glasses, where the camera changes.

**MediaPipe's handedness also flips mid-clip**, most readily when a palm turns
away — which ASL does constantly, since orientation is one of the five
parameters. Trusting the per-frame label let a hand change identity *mid-sign*,
which in a two-handed sign swaps the dominant and non-dominant blocks partway
through and produces a feature vector describing a sign nobody made. Identity
is now resolved over a whole clip: hands are followed by position and each
track takes the majority label of its own frames. Following position rather
than side-of-frame is deliberate, because hands cross in ASL.

**Tracking drops out worst during fast movement**, since motion blur and
self-occlusion cluster inside the gesture rather than spreading evenly. A
keyframe near a dropout used to interpolate between a real hand and an all-zero
block, producing a half-scale hand at an impossible position — a pose nobody
made, blended out of a pose and an absence. `bridge_gaps()` holds the last
tracked geometry across the gap. The presence flags are deliberately *not*
filled: they stay honest and come out fractional after resampling, which is
exactly the signal a classifier should get.

**The first version of that fix created its own bug.** It also extended a
hand's first and last sighting out to the clip edges, so a second hand detected
for a single frame — a face or background object read as a hand — had its
geometry held across the *entire* clip. That happened in 18 of the first 250
clips, nearly all MOTHER and FATHER, where the face is closest to the hand.
Bridging now only fills a gap with a sighting on **both** sides, and only up to
300 ms: a real dropout is bridged, a ghost is not.

**The model was trained on something different from what it was shown live.**
Training used whole takes — the hand coming up, the sign, the hand going down,
a median of 2 s. The live segmenter classifies a trailing 900 ms window. On the
first 250 clips that mismatch dropped a nearest-centroid floor from 83% to
67–72%. `train_signs.py` now cuts every clip into overlapping 900 ms windows
(~1,100 from 250 clips) straight from the archive, and together with the
bridging fix the same live-style measurement reads **85.6%**. The window length
lives in one place, `sequence.SIGN_WINDOW_MS`, and a test fails if the training
windows and the live segmenter ever disagree about it again.

**A lone hand could land in the wrong slot live.** Offline, a MediaPipe label
flip is voted away over the whole clip; live there is no clip to vote over, so
one flipped frame dropped the hand into the non-dominant slot. The demo now
puts a single visible hand in the dominant slot, which is where one-handed ASL
signs are made. This is deliberately *not* in `capture.scene()`, because
`check_setup.py` uses that to test the handedness convention and the rule would
make the test pass unconditionally.

**Landmarks jitter.** A 3-point **median** rather than an average, and that
choice is the whole reason it is safe: the median of three monotonically
changing samples *is* the middle sample, so a hand moving steadily passes
through completely unchanged — no lag, no attenuation of a fast sign. Only a
sample disagreeing with both of its neighbours is replaced, which is the
definition of a spike. An average would blunt every fast movement to buy the
same protection.

**A parameter shadowed a module it imported.** `def measure(capture, ...)` grew
an `import capture` in its body; the import rebinds the name, so the camera
object became the module and `capture.read()` died — past every offline test,
only on a machine with a camera attached. There is now a test that walks every
function in the repository looking for that shape.

**A stale build artifact blocked collection.** The collector wrote the CSV
header at startup rather than on the first saved clip, so quitting without
recording left a header-only file; a later schema change turned that leftover
into a hard stop, with an error message recommending a rebuild that does
nothing when no clips exist. Schema drift is now reconciled rather than
refused, and only the one genuinely unrecoverable case stops.

---

## Performance

Measured: the per-frame cost of **everything in this repository is 0.08 ms, or
0.2% of the budget at 20 fps**. The cost is MediaPipe and camera I/O, so
optimising the feature code would be pointless.

`capture.open_camera()` prefers **DirectShow + MJPG** on Windows. OpenCV's
default Windows backend is known to deliver frames slowly and to cap or jitter
the frame rate on many webcams. It falls back to the default if DirectShow
fails, and resolution stays pinned at 640×480 to match every collected clip —
a different resolution would change tracking quality and reintroduce a
train/live mismatch. `--default-backend` on any tool switches back, and
`check_setup.py` reports which backend it got.

`capture.PeriodicFace` runs the face detector every 3rd frame and reuses the
box between — a whole second model saved on two frames in three. The tradeoff
is that the box lags by up to `every` frames during a fast head turn, which
shifts location slightly; a head moves slowly compared to hands, and location
is measured in face-widths, so a box a frame or two old is very nearly the same
box. `--face-every 1` disables it.

**Frame rate and motion blur are different problems.** More fps does not reduce
blur if the exposure stays long, and blur is what breaks tracking on quick
movements — the hand smears within a single frame and MediaPipe loses it. Light
on the signer is the cheapest fix available; shortening exposure is the next
one.

There is a frame-rate **floor**, separately: `min_samples` (8) must be met
inside a sign, so a ~700 ms sign needs about **11 fps**. `NOTES.md` derives the
same figure independently from `motion.py`'s window. Below it no clip can be
assembled and no motion letter can fire, which makes it a hardware constraint
on the embedded target rather than a tuning preference.

---

## Testing

```bash
python -m unittest discover -p "test_*.py" -v
```

341 offline tests, standard library only, running in CI on every push. No `pip
install` step: the test modules import nothing outside the standard library, so
the suite finishes in seconds.

What CI can and cannot cover is worth being explicit about. It covers the
geometry, the schema, the fold construction, the gloss grammar and the
segmentation logic. It covers **none** of MediaPipe's real-world behaviour, and
every bug in [Things that bit us](#things-that-bit-us-on-real-hardware) got
past it.

`check_setup.py` exists to cover part of that gap against a real camera: it
verifies packages and model files, runs the offline suite, then measures
throughput against the 11 fps floor and settles the handedness convention
against a hand whose identity the person confirms.

A few tests are worth calling out because of what they protect:

- **The stdlib guard** — no offline module may import numpy, MediaPipe or
  `capture`. Without it, adding such an import would not fail loudly; the
  affected suite would error while the workflow still looked green.
- **The shadowing guard** — no function may take a parameter named after a
  module it imports.
- **The gloss fuzz** — every one- and two-sign combination of the vocabulary
  must render without raising. A translator that crashes on an unanticipated
  combination takes the pipeline down mid-conversation.
- **The numpy-port check** — replaying real recorded rows through the
  standard-library normalization, confirming it matches the numpy original that
  produced the existing dataset.
