# Project Veronica — full ASL, not just fingerspelling

Branch: **`project-veronica`**, cut from the latest pipeline work (not from
`main`, which is 20 commits behind it).

`main` recognises a 27-class vocabulary: 24 static letters, J and Z from
trajectory rules, and three static word signs. It measures **84.0%
cross-person**, and for what it does it is sound. What it does is
**fingerspelling**, and fingerspelling is not how ASL is used. A Deaf signer
fingerspells names, unfamiliar proper nouns and the occasional loanword;
everything else is signs — most of them moving, about half of them two-handed,
many of them distinguished only by where on the body they are made.

Veronica is the path from "spells letters" to "reads the language". This file
is the order that happens in, and — more importantly — **why that order**.

## What a sign actually is

ASL signs decompose into five parameters. A sign is distinct from another if
*any one* of them differs. Where `main` stands today:

| Parameter | Example contrast | `main` |
| --- | --- | --- |
| **Handshape** | the 24 letters | ✅ the 42 floats |
| **Orientation** | palm up vs. palm down | ❌ normalized away |
| **Location** | FATHER (forehead) vs. MOTHER (chin) | ❌ normalized away |
| **Movement** | a path through space and time | ❌ invisible to one frame |
| **Non-manual markers** | brow raise = yes/no question | ❌ not a hand |

Four of five are missing. That is not a gap in the model — it is a gap in the
**representation**, and no amount of extra training data closes it. Two signs
that differ only in location produce byte-identical feature vectors today; the
classifier is being asked to distinguish things it cannot see. `NOTES.md`
already records this from the inside: G and Q are confusable precisely because
they differ only in orientation, which the hand frame deletes by construction.

So Veronica is mostly a representation project, and only then a modelling one.

## Why the stages are in this order

Stages 1–3 all change **what a recorded row contains**. Stage 4 spends the one
resource that cannot be re-spent cheaply: other people's time, in person, one
session each. `NOTES.md` is unambiguous that people are the binding constraint
— a fourth signer was worth +4.7 points while tripling the rows from existing
signers was worth nothing.

**So every schema change lands before anyone is asked to sign into a camera.**
Collect against a half-finished feature vector and the choice is re-collecting
from everyone or throwing the new parameter away. That is the entire reason
this is sequenced rather than parallel, and it is why stage 4 is a hard gate
rather than a suggestion.

---

## Stage 1 — Two hands and orientation ✅ **done**

`hands.py`, `test_hands.py`.

Two 46-float hand blocks plus an 8-float relational block — 100 floats, which
stage 2 extends to 107. Recovers **orientation** (the parameter the hand frame discarded) and
adds everything about a **second hand** — separation, relative rotation,
relative size, and contact, all measured in hand-widths so they survive camera
distance without a depth sensor or a body model.

Decisions worth knowing about:

- **The 42 shape floats are unchanged.** Verified against all 24,496 recorded
  rows: the standard-library port agrees with the numpy original to 1.8e-15.
  The existing dataset and the existing model stay valid.
- **Hands are ordered dominant-first, not left-then-right**, and a
  left-dominant signer's scene is mirrored into right-dominant space. ASL is
  handedness-symmetric — a left-handed signer signs the mirror image, with the
  same meaning. Without this, every sign a left-handed signer makes is an
  unseen class. It costs one negation.
- **Standard library only**, like `dataset.py`, so all 38 tests run in CI.

### The mirror hazard — found on real hardware, now settled and recoverable

MediaPipe reports handedness relative to an assumed mirroring of the input.
**On this project's actual dev machine that assumption runs the opposite way to
what the code assumed**: `check_setup.py` reported `Left` on 100% of frames for
a raised right hand. Caught before any clip was collected, which is the entire
reason the check exists.

It cannot be settled by reading code. It depends on the MediaPipe build *and*
on the camera, since some webcams deliver an already-mirrored feed that
`cv2.flip` then un-mirrors. So the check reports **two** facts, not one:

1. **what MediaPipe called the hand**, and
2. **which side of the frame the hand appeared on** — the person says which
   hand they raised, so this says whether the feed is actually mirrored.

Reporting only (1) is what made the first version of this check ambiguous.
Together they separate the two root causes, which matters on the glasses where
the camera changes.

Three things make it survivable rather than fatal:

- **It is a setting, not a default.** `config.py` persists it to
  `veronica_config.json`; `collect_signs.py` and `demo_veronica.py` both read
  it. `python check_setup.py --set-mirrored-input false` writes it once.
- **Every clip records the convention it was collected under.** Mixed archives
  are detected and reported.
- **A whole archive collected under the wrong convention is repairable.**
  `python rebuild_signs.py --mirrored-input true` re-derives every training row
  from the raw landmarks. Demonstrated end to end: 24/24 rows corrected, nobody
  signs again. This is the payoff for archiving raw landmarks rather than
  features.

### MediaPipe's handedness also flips mid-clip

A second finding from the same machine: the per-frame label is **not stable**.
It flips, most readily when a hand rotates so the palm turns away from the
camera — which ASL does constantly, since orientation is one of the five
parameters a sign is built from.

Trusting the per-frame label lets a hand change identity *mid-sign*. In a
two-handed sign that swaps the dominant and non-dominant blocks partway through
the clip, and the resulting feature vector describes a sign nobody made.

So identity is resolved **over the whole clip**: hands are followed by position
and each track takes the majority label of its own frames. Following position
rather than side-of-frame is deliberate — hands cross in ASL, and "the right
hand is the one further right" breaks exactly when they do.

`handedness_stability` is recorded per clip, so unstable recordings are
findable rather than silently averaged in.

## Stage 2 — Location: anchoring signs to the body ✅ **done**

`location.py`, `test_location.py`. Composed into the frame vector by
`hands.feature_vector(dominant, nondominant, face)`, now **107 floats**.

The parameter that cannot be recovered from hand landmarks at all. FATHER and
MOTHER are the same handshape, the same orientation and the same movement; one
is at the forehead and one is at the chin. There is a test pinning exactly that
down — the two produce *byte-identical* vectors without a body reference, and
differ by more than a face-width with one.

**The body reference is a face detector**, not a pose model. Most
location-contrastive signs are head-anchored, the model is small, and it leaves
the "fully on-device, lightweight" result intact — with the embedded target
still undecided, a second full pose model per frame is headroom that is not
free to spend. The deployment geometry cooperates: the glasses see the
conversation partner, so the signer's face is in frame.

Decisions worth knowing about:

- **Face *width* is the unit, not height or diagonal.** ASL uses head movement
  grammatically, and pitching the head compresses a face box's apparent height
  hard while leaving its width nearly alone. A height-based unit would move
  every location feature by ~45% during exactly the constructions stage 8 has
  to read. Both axes divide by that one number, so the frame stays isotropic
  and an angle in it is a real angle.
- **`hand size / face width` is a free depth proxy.** Both are fixed physical
  sizes, so the ratio is roughly constant for a person *unless* the hand is
  nearer the camera than the face — which is what a sign made out in neutral
  space does, as against one contacting the body. That is a usable third
  dimension out of two monocular measurements.
- **Detector-agnostic.** A face is three numbers (centre and width); nothing in
  `location.py` imports MediaPipe or knows what produced the box. So it is
  testable in CI, and a better face model later is a change at the call site.
  Landmark anchoring (eye and mouth keypoints, steadier than a box, and they
  would also give head rotation) fits this interface unchanged.
- **A lost face zeroes only the location block.** `face_present` leads it, so
  "no face detected" stays distinguishable from "both hands at the centre of
  the face", and the other 100 floats survive intact. Same distinction
  `debouncer.py` had to make between absence of evidence and evidence of
  absence.

`zone_name()` and `debug_info()` name the region a hand is in — forehead,
mouth/chin, neck/shoulder, chest — for an on-screen readout, in the same spirit
as `MotionDetector.debug_info()`. They are **diagnostics, never features**: the
cut points are a guess, and a hard-coded "chin" boundary that is slightly wrong
is worse than letting the model learn where a chin falls.

**Still to do on camera:** the face detector itself is not wired into `main.py`
yet — that happens with the stage 4 collection tool, since it is the first
thing that needs a real face in a real frame. Obtaining the MediaPipe face
detector model file is part of that step.

## Stage 3 — Movement: from frames to a sign ✅ **done**

`sequence.py`, `test_sequence.py`.

The jump `main` cannot make by adding classes. A static classifier sees one
frame; a sign is a path.

`motion.py` recognises J and Z with hand-written trajectory rules, and its own
docstring is honest that each new sign means another predicate plus its own
thresholds. Two signs in, that has already produced two documented false-fire
bugs — a held Q emitting Z from fingertip jitter, and Z failing to fire because
three strokes did not fit inside a 650ms window. Neither was a tuning mistake;
both were the approach reaching its limit.

So a clip becomes a **fixed-length vector** and the same cheap MLP classifies
it. No RNN, no TensorFlow, no per-sign thresholds — which keeps the lightweight
result and means stage 5 reuses `train_classifier.py`'s methodology unchanged.

Decisions worth knowing about:

- **Each parameter is sampled at the rate it actually changes.** Resampling all
  107 floats at 8 keyframes is 856 inputs against the few thousand clips a
  realistic collection effort produces — that is not a model that generalises,
  it is one that memorises, and `NOTES.md` already records what happened last
  time this project measured memorisation and called it accuracy. Handshape is
  near-constant within a sign, so it gets **2** keyframes; orientation,
  inter-hand relationship and body location move continuously, so they get
  **8**. Same clip, **371 floats instead of 856**.
- **Keyframes are placed by time, not by index.** `NOTES.md` lists
  wall-clock windows as a deliberate non-change; resampling by frame index
  would quietly undo it. Tested: the same gesture at 30fps and 15fps agrees to
  0.05, and a clip whose samples bunch up mid-gesture resamples correctly by
  time and wrongly by index.
- **Trajectory is self-relative; location is body-anchored; both are kept.**
  The movement summary measures displacement from where the clip started, in
  hand-widths, so it works with no face in frame. The keyframed location block
  carries absolute position on the body when a face is there. "The hand arced
  downward and reversed twice" and "it did so at the chin" are different facts
  and neither is derivable from the other.
- **Straightness and reversals are explicit.** A circle returns to its start,
  so net displacement is near zero while path length is not — straightness is
  what separates circular movement from straight, and circular movement is
  common in ASL. Reversals count the repetitions many signs carry as part of
  their form rather than as emphasis, using a deadband so that jitter never
  becomes a reversal (the 2-D generalisation of `MotionDetector._strokes()`).
- **Interpolated rotations are renormalized.** Blending two unit vectors
  linearly gives the chord, not the arc, so a rotation halfway between two
  keyframes would otherwise read as a *smaller* rotation — and the error grows
  with the gap, i.e. it is worst on exactly the low-frame-rate hardware this is
  meant to survive.

### The frame-rate floor is confirmed from a second direction

`SignBuffer.min_samples` (8) has to be met inside a real sign, so a 700ms sign
needs about **11fps** — the same figure `NOTES.md` derives for J/Z from
`motion.py`'s window, reached independently. That is a hardware constraint on
the embedded target, not a tuning preference, and it now has two derivations
behind it.

### One thing that surprised the tests, worth knowing before stage 4

With **no face detected, a hand crossing the entire frame produces identical
frame vectors throughout** — position is normalized away by construction, which
is what makes handshape recognisable anywhere in the frame. The trajectory is
the only thing that knows it moved. This is why the wrist anchors are passed
into `SignBuffer.add()` separately rather than read back out of the features,
and it is pinned down by its own test.

## Stage 4 — Collection ✅ **tool built, data not yet collected**

`collect_signs.py`, `signset.py`, `rebuild_signs.py`, `test_signset.py`.

The expensive, hard-to-repeat step. Everything above exists so it happens once.

```bash
python collect_signs.py       # SPACE to start/stop a clip, U to undo a fluffed take
python rebuild_signs.py       # regenerate the training CSV from the archive
python rebuild_signs.py --check   # what has been collected, what is missing
```

### Collect raw, derive features

Collection writes **two** files:

| File | What it is |
| --- | --- |
| `veronica_clips.jsonl` | raw landmarks, one clip per line — **the archive** |
| `veronica_signs.csv` | the 371-float training rows — **derived, regenerable** |

Sequencing stages 1–3 before collection protected the signers' time exactly
once. The next time `sequence.py`'s keyframe counts are revisited — and stages
5–8 will revisit them — feature-only rows would all be invalid and the sessions
would have to happen again. Keeping the raw landmarks makes that a script run.

The letter dataset learned this in reverse: `normalize_landmarks()` ran before
anything reached disk, so orientation is gone from those 24,496 rows for good
and no script can recover it. **The archive is the record; the CSV is a build
artifact.** Measured at ~6x the CSV, and it is JSON Lines of short repeated
keys, which is near the best case for the compression git already applies.

### What the tool does differently from `collect_data.py`

- **One clip per sign attempt**, not one row per frame. A letter is a held
  pose; a sign is a path.
- **`num_hands=2`**, and it asks for the signer's **dominant hand** — a
  left-dominant signer's scene is mirrored into right-dominant space, and
  getting that wrong is undetectable downstream.
- **Refuses to start without the face detector** unless you pass `--no-face`.
  Collecting a whole session without location data by accident is expensive
  enough to be worth a hard stop rather than a warning. Download
  `blaze_face_short_range.tflite` into the repo directory; the error message
  carries the URL.
- **Rejects takes too short to be a sign**, using `SignBuffer`'s own floors, so
  anything saved is something the live pipeline could also have classified.
- **`U` undoes the last clip**, truncating both files to recorded offsets.
  Fluffed takes are common and the alternative is keeping known-bad data.
- **`F` shows a live readout** — reported handedness, which hand is dominant,
  and the named location zone. **Press it and confirm your right hand reads as
  right before starting a real session**; this is where the mirror hazard from
  stage 1 gets caught or gets baked into the dataset.
- **Flags low face coverage** per clip rather than dropping it — filtering
  later is easy, losing a good take is not.

### What to tell the signer

The letter collector's advice was counter-intuitive because the hand frame
erased most apparent variation. **That is no longer true**: the vector now
carries orientation, face-relative location and the whole trajectory, so most
real variation reaches the features. Sign at natural speed and amplitude.

The one piece that still holds: **position in the frame and distance from the
camera are still normalized away**, now against the face. Shifting around in
your chair contributes nothing.

And the lesson that has held throughout: **more people, not more clips per
person.** A fourth signer was worth +4.7 points on the letter dataset while
tripling the rows from existing signers was worth nothing. `SUGGESTED_CLIPS` is
20 per sign per person — far below the old 100 rows/label, because a clip is a
separate attempt where frames within a burst were near-duplicates.

### The vocabulary

68 glosses in `signset.VOCABULARY`, grouped by category and freely editable —
nothing downstream knows the particular strings, exactly as `collect_data.py`'s
`WORDS` list works today. Two are worth keeping if you edit it:

- **MOTHER and FATHER**, which differ only in location and are therefore the
  working check that stage 2 earns its place.
- **`_REST`**, a not-signing class. Two minutes at collection time; stage 5
  needs something to reject garbage with and stage 6 has to tell "between
  signs" from "a sign" with nobody pressing a key. Collecting it later means
  another session with every signer.

Glosses only. **How each sign is formed is not encoded anywhere in the repo**
and should come from a dictionary and a fluent signer.

The custom `HELLO` / `IHATEYOU` handshapes from `main` are **not** carried over
— Veronica can represent movement, so the real ASL forms are collectable and
the academic-honesty note in `README.md` stops being necessary.

### Still to do before a real session

1. **Download `blaze_face_short_range.tflite`.** The tool stops without it.
2. **Verify handedness on camera** (press `F`). The stage 1 hazard.
3. **Recruit a fluent signer.** Every number in this repo so far comes from
   hearing team members performing signs — fine for letters, misleading for
   phrases, since fluent signing differs in timing and coarticulation, which is
   exactly what stages 3 and 6 model.

## Stage 5 — Classifier and honest evaluation ⚙️ **script ready, blocked on data**

`folds.py`, `train_signs.py`, `test_folds.py`.

```bash
python train_signs.py          # cross-person report, then train
python train_signs.py --quick  # skip the report
```

Same shape as `train_classifier.py`, because its methodology is the strongest
thing in this repository: **leave-one-person-out is the headline**, the random
split is printed only as an explicitly inflated comparison. On the letter data
those are 84.0% and 98.6%.

Everything deciding *what goes in which fold* lives in `folds.py`, which is
standard-library only and tested in CI. That split is deliberate: the sklearn
call is the easy part to get right, and a bug in fold construction does not
crash and does not look wrong — it just produces a number that is too high.
This project has already been bitten by that once.

Three ways a fold can quietly lie, all handled and all reported *before* the
number rather than after it:

- **A label only one person signed** is absent from training on that person's
  fold, so it scores ~0 for a reason that has nothing to do with the model.
  Excluded, as `train_classifier.py` already does.
- **`_REST` counted as a sign.** It is a real class the model must learn and
  much easier than any sign, so it is reported separately and kept out of the
  headline.
- **Folds that are not comparable.** If one person signed 68 labels and another
  20, their folds measure different tasks; averaging without saying so hides it.

Clips are far more independent than frames were — each is a separate attempt —
but the same trap exists one level up: twenty HELLOs by one person in one
sitting are much more like each other than like anyone else's. Grouping by
person closes both, which is why nothing groups by clip.

One change from the letter model: **a `StandardScaler` in front of the MLP**.
The 42 letter inputs were all the same kind of quantity in one range, so
scaling did nothing. These 371 are not — landmark coordinates near ±3, clamped
distances to 6, cosines to 1, a duration to 3, a coverage fraction to 1.
Un-scaled, the widest block dominates the first layer purely through its units.

**Expect a lower number than 84%.** More classes, harder classes, a genuinely
harder task. A lower number on real ASL is a better result than a higher one on
fingerspelling and should be reported as such.

The pairs to read first in the confusion matrix are those between signs
differing in only **one** parameter. Those say the parameter is not reaching
the classifier — a representation problem in `hands`/`location`/`sequence`, not
something more data will cure. G/Q on the letter model was exactly this.

## Stage 6 — Continuous signing ✅ **done**

`segment.py`, `test_segment.py`, and `capture.py` / `demo_veronica.py` for the
live path.

Everything before this assumes someone presses a key to mark a sign's
boundaries. Real conversation has no key.

### Why this is not boundary detection

The obvious approach — find the pauses, or the minima in hand velocity, and cut
there — does not work, for a reason with a name. **Movement epenthesis**: the
transition between two signs is itself movement, and looks like a sign to
anything watching for movement. Signers also do not pause between signs the way
speakers pause between words, so the pauses a cutter would look for are often
simply not there.

So this does not cut first. It **classifies a trailing window continuously and
commits when the answer is stable** — which is the same shape as the problem
`debouncer.py` already solves for letters, one level up. Its central
distinction generalises exactly:

    ""       no confident reading   -- absence of evidence
    "EAT"    a different sign       -- evidence the last one ended

Treating the first as a boundary is what produced `QQQQQQQQQQ` from one steady
hold. Here the same mistake fires a sign on every frame of an unrecognised
transition. So **`Debouncer` is reused, not reimplemented** — 24 tests,
replaying real recorded failure strings, and every property it has is one this
needs: commit once per run, blanks must persist before releasing, a different
label takes over faster than a blank because it is real evidence, and a genuine
release re-arms the same label so `AGAIN AGAIN` commits twice.

### Three defences against epenthesis, in order of how much work they do

1. **The `_REST` class** — a transition classified as `_REST` produces no
   reading at all. This is the main defence and the reason stage 4 collects
   `_REST` even though nobody signs it.
2. **The confidence floor** — a window spanning two signs is a clean example of
   neither, so the classifier should be unsure.
3. **The hold requirement** — a spurious label must survive several consecutive
   evaluations, and a transition is brief.

There is a test for the honest limit too: a classifier that is *confidently
wrong for long enough* commits, and no threshold in this module fixes that.
That is a model problem.

### Cost

Classifying every frame means a full `sign_vector()` build plus a forward pass
at camera rate, on hardware that also has to sustain ~15fps of MediaPipe.
Evaluation is **strided**: the buffer takes every frame, classification runs
every `stride_ms`. At 30fps with a 100ms stride that is one evaluation in
three, and the window it sees is unchanged.

**The thresholds here are structural guesses.** Continuous segmentation is the
least finished part of this pipeline, and it needs real continuous signing to
tune — which is why `debug_info()` puts every one of them on screen. The value
to watch first is `reading`: if transitions show a confident label rather than
a blank, `_REST` is not covering them, and more `_REST` clips will do more than
any threshold change.

## Stage 7 — Phrases: gloss to English ✅ **done**

`gloss.py`, `test_gloss.py`.

A sign recogniser outputs **gloss** — the sequence of signs made. Gloss is not
English, and rendering it as though it were is a mistranslation rather than a
rough edge.

| Gloss | English |
| --- | --- |
| `YESTERDAY ME GO SCHOOL` | Yesterday, I went to school. |
| `YOU NAME WHAT` | What is your name? |
| `ME TIRED` | I am tired. |
| `ME DONT-UNDERSTAND` | I do not understand. |
| `ME NEED GO BATHROOM` | I need to go to the bathroom. |
| `HELLO ME NAME L-A-I-L-A` | Hello, my name is Laila. |

It undoes the differences ASL systematically has: **no copula**, **tense as a
time marker rather than a verb inflection**, **wh-words at the end**, **no
articles**.

This turned out **not to be blocked on stage 5 at all** — it transforms gloss
strings and needs only a vocabulary, which stage 4 settled. The "after 5" in
the original plan was pipeline order mistaken for a dependency.

### Rules, not a model — and the reason is the failure mode

Over a closed vocabulary a template transformer is honest, debuggable, and
**degrades gracefully**: when it cannot parse something it returns the gloss,
which is still readable. A Deaf signer reading `ME GO STORE` loses nothing,
while a fluent English sentence that says the wrong thing is worse than no
translation at all. A learned translator needs a parallel corpus this project
does not have and fails in the opposite direction, by inventing fluent text.

Two decisions worth knowing about:

- **`HE-SHE` renders as "they".** The ASL sign is a *point*, which carries no
  gender. Choosing "he" or "she" would invent information the signer did not
  give, every time it appears. Singular "they" keeps exactly what was signed.
- **`render(tokens, question=True)` exists now** for stage 8 to fill in: a brow
  raise turns a statement into a yes/no question with no change to the hands,
  so `YOU HUNGRY` is either "You are hungry." or "Are you hungry?" depending on
  a channel this stage cannot see.

### What it deliberately does not attempt

Real ASL grammar is much richer, and pretending otherwise would be the same
mistake as quoting a shuffled-split accuracy. **Aspect** (marked by modifying a
verb's movement), **spatial agreement** (verbs moving between points assigned
to referents), **classifiers**, and **role shift** are all unhandled and
documented as such in the module. Stage 3 records the movement that carries
aspect; nothing yet maps it to meaning.

Spelling correction stays in `nlp_bridge.py` — grammar and spelling are
separate jobs, and only one of them is testable without a third-party
dictionary.

## Stage 8 — Non-manual markers

Grammar on the face. A brow raise makes a sentence a yes/no question; a
headshake negates; mouth morphemes modify. Without them a question and a
statement are the same sentence.

Deferred to last because it needs stage 2's face model already running, and
because the gloss output is useful without it. Not optional for *correct* ASL
— a signed question rendered as a statement is a mistranslation, not a
rough edge.

---

## Status

| Stage | State |
| --- | --- |
| 1 — two hands, orientation | ✅ done, 38 offline tests |
| 2 — location / body anchor | ✅ done, 25 offline tests (face detector chosen) |
| 3 — movement | ✅ done, 41 offline tests |
| 4 — collection | ✅ tool built, 26 offline tests — **data not yet collected** |
| 5 — classifier | ⚙️ script ready, 21 offline tests — **blocked on data** |
| 6 — continuous signing | ✅ done, 26 offline tests |
| 7 — gloss → English | ✅ done, 37 offline tests |
| 8 — non-manual markers | last |

`main` is untouched and still runs. Veronica adds files rather than rewriting
them until stage 4, so the thesis basis stays demonstrable throughout.

---

## Testing locally

Nothing below needs a trained model. The point of running it now is that the
parts which can only be judged in front of a real camera should be judged
*before* a collection session, not after.

### 1. Install and check

```bash
pip install -r requirements.txt
curl -LO https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
python check_setup.py
```

`check_setup.py` verifies packages and model files, runs the 266 offline tests,
opens the camera, and then checks the two things that do not announce
themselves when they are wrong:

- **Pipeline throughput.** It reports fps with *both* models running and fails
  below 11fps. That floor is derived twice over — from `motion.py`'s window in
  `NOTES.md`, and independently from `SignBuffer.min_samples`. Below it, no
  clip can be assembled and no motion letter can fire.
- **The handedness convention.** It asks you to hold up your right hand, then
  reports both what MediaPipe called it *and* which side of the frame it was
  on. If the label is inverted, every hand in your dataset would be labelled
  backwards, trained on happily, with features that stay entirely plausible.

  **This has already happened on the dev machine** — it reported `Left` for a
  right hand. Fix it once, for every tool:

  ```bash
  python check_setup.py --set-mirrored-input false
  ```

### 2. Watch the layers run

```bash
python demo_veronica.py            # add --dominant left if that is you
```

Recognition is idle without a model, but everything under it is live. Press `D`
and you get fps, the reported handedness, the buffer filling, the confidence
reading, and the named location zone each hand is in — forehead, mouth/chin,
neck/shoulder, chest. Move a hand from your forehead to your chin and watch the
zone change: that is stage 2 working, and it is the difference between FATHER
and MOTHER being separable or not.

### 3. Then collect

```bash
python collect_signs.py            # SPACE start/stop, U undo, F readout
python rebuild_signs.py --check    # what is collected, what is missing
python train_signs.py              # once two people have signed the same signs
```

### What "working" looks like before any data exists

| Check | Expect |
| --- | --- |
| `check_setup.py` | all pass; fps comfortably above 15 |
| Right hand raised | `Handedness convention` passes (set the flag if not) |
| Both hands up | `hands: 2` in the demo HUD |
| Face in frame | green box, `face: yes` |
| Hand at forehead vs chin | the zone in the `D` readout changes |
| Hands dropped out of frame | buffer empties, nothing commits |
