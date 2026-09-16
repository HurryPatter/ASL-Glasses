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

### One hazard to watch, because nothing downstream can detect it

MediaPipe labels handedness assuming a **mirrored** frame. `main.py` does
`cv2.flip(frame, 1)`, so that holds today.

It will not hold on the glasses. The dev setup has the signer looking at a
mirrored preview of themselves; the deployed glasses see a *different person,
facing the wearer*. Those two geometries are reflections of each other. Get the
convention wrong and every left hand is labelled right — trained on happily,
silently, and wrong. `assign_hands(mirrored_input=...)` makes it one explicit
flag instead of an assumption buried in three files. **Verify it on camera
before stage 4**, by signing with one hand and checking the label.

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

## Stage 3 — Movement: from frames to a sign

The jump `main` cannot make by adding classes. A static classifier sees one
frame; a sign is a path.

`motion.py` handles J and Z with hand-written rules, and its own docstring is
honest that each new sign is another predicate plus its own thresholds. That
does not scale past a handful — two signs, two rules, already two documented
false-fire bugs. A vocabulary of dozens needs the movement **learned**, not
enumerated.

The approach that fits the constraints: resample a window of frame vectors to a
fixed number of keyframes and concatenate, plus explicit velocity and path
summaries. This turns a variable-length sequence into a fixed-length vector
that the same cheap MLP can classify — no RNN, no TensorFlow, no per-sign
threshold tuning, and it keeps the "lightweight" objective that dropping
TensorFlow bought.

Two things `motion.py` already got right and that carry over verbatim:
**wall-clock windows, never frame counts** (embedded hardware will not match
the dev laptop's fps), and **gating on whole-hand travel** rather than
fingertip motion (per-landmark jitter cancels in a centroid; that gate is what
stopped a held Q from emitting Z).

The existing ~11fps floor for motion signs still applies and still constrains
the hardware choice.

**Done when:** a synthetic moving-hand sequence produces a stable fixed-length
vector, tested offline at several frame rates — the `test_motion.py` pattern.

## Stage 4 — Collection 🔒 *gated on 1–3 being frozen*

The expensive, hard-to-repeat step. Everything above exists to make sure this
happens once.

- Record **sequences**, not frames. A row becomes a clip.
- Ask for the signer's **dominant hand**, alongside the name `collect_data.py`
  already asks for.
- Start from a **realistic vocabulary**: 50–100 signs of everyday
  conversational ASL, chosen for usefulness rather than for ease of
  recognition. Full ASL is tens of thousands of signs; a defensible thesis
  result is a real, honestly-measured vocabulary, not a claim to cover the
  language.
- Keep `main`'s collection lesson: **~100 examples per sign per person, then
  recruit the next person.** Rows saturate; people do not.
- **Recruit a Deaf or fluent signer if at all possible.** Every number in this
  repo so far comes from hearing team members performing signs. That is fine
  for letters and misleading for phrases — fluent signing differs in timing,
  amplitude and coarticulation, which is exactly what stages 3 and 6 model.

The three custom word signs on `main` (`HELLO`, `IHATEYOU`) should be
**replaced by their real ASL forms** here, since Veronica can represent
movement. `README.md` currently has to carry an academic-honesty note saying
they are self-chosen handshapes rather than real ASL. That note should become
unnecessary.

Veronica writes a **new file**, not a wider `landmark_data.csv`. The historical
rows cannot supply orientation — it was discarded before anything reached disk
(there is a test pinning that down). Widening the old file would mean 24,496
rows with a silently-imputed parameter. The letter model keeps its file and its
84%, and becomes the fingerspelling component rather than being thrown away.

## Stage 5 — Classifier and honest evaluation

Same discipline as `main`, which is already the strongest methodological thing
in this repo: **leave-one-person-out is the headline**, the random split is
quoted only as an explicitly inflated comparison.

For sequences the leakage risk is worse, not better — frames within a clip are
near-duplicates *and* clips of the same sign by the same person in one sitting
are near-duplicates of each other. Group by person, as `dataset.py` already
does.

**Expect the first number to be worse than 84%.** More classes, harder classes,
and a genuinely harder task. A lower number on real ASL is a better result than
a higher one on fingerspelling, and it should be reported as such.

## Stage 6 — Continuous signing

Everything above assumes someone presses a key to mark where a sign starts.
Real conversation has no key.

The hard part is **movement epenthesis**: the transition between two signs is
itself motion, and it looks like a sign to a detector that is only watching for
movement. Signers also do not pause between signs the way speakers pause
between words.

`debouncer.py` solves the letter-sized version of this problem and solves it
well — in particular its distinction between *absence of evidence* (`""`) and
*evidence of a different handshape*, which is what stopped one held letter
committing ten times. That distinction generalises; the thresholds do not.

## Stage 7 — Phrases: gloss to English

ASL is not English word order and has no separate word for much of English
grammar. `ME STORE GO-TO FINISH` is "I went to the store." Topic-comment
ordering, no copula, aspect marked on the verb's movement rather than by an
auxiliary.

`nlp_bridge.py` currently does SymSpell correction over fingerspelled letters,
which is the right tool for spelling and the wrong tool for grammar. Gloss →
English is a reordering and inflection problem. Start with a template-based
transformer over a small, closed vocabulary — honest, debuggable, and it
degrades gracefully to a bare gloss string, which is still readable.

This is also where the **name injection already in `nlp_bridge.py`** pays off:
fingerspelling, routed from the letter model, is exactly how names arrive in a
real conversation.

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
| 3 — movement | next |
| 4 — collection | 🔒 gated on 1–3 |
| 5 — classifier | after 4 |
| 6 — continuous signing | after 5 |
| 7 — gloss → English | after 5 |
| 8 — non-manual markers | last |

`main` is untouched and still runs. Veronica adds files rather than rewriting
them until stage 4, so the thesis basis stays demonstrable throughout.
