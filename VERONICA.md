# Project Veronica — roadmap

Branch: **`project-veronica`**, cut from the latest pipeline work (not from
`main`, which is behind it).

For how any of it works and why, see **[VERONICA_DESIGN.md](VERONICA_DESIGN.md)**.
This file is what is done, what is next, and what is blocking.

## What this is

`main` recognises 27 classes: 24 static letters, J and Z from trajectory rules,
and three static word signs. It measures **84.0% cross-person**, and for what it
does it is sound. What it does is **fingerspelling**, and fingerspelling is not
how ASL is used — a Deaf signer fingerspells names and unfamiliar words, and
everything else is signs.

Veronica is the path from "spells letters" to "reads the language". `main` is
untouched and still runs: Veronica adds files rather than rewriting them.

## The five parameters

A sign is distinct from another if any one of these differs. This is what
drove the plan, because it says exactly what was missing:

| Parameter | Example contrast | `main` | Veronica |
| --- | --- | --- | --- |
| Handshape | the 24 letters | ✅ | ✅ |
| Orientation | palm up vs. palm down | ❌ | ✅ |
| Location | FATHER (forehead) vs. MOTHER (chin) | ❌ | ✅ |
| Movement | a path through space and time | ❌ | ✅ |
| Non-manual markers | brow raise = yes/no question | ❌ | ⬜ |

Four of five were missing, and that is a gap in the **representation**, not in
the model — no amount of training data closes it.

## Quick start

```bash
pip install -r requirements.txt
curl -LO https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite

python check_setup.py         # deps, models, camera fps, handedness convention
python collect_signs.py       # SPACE rec · C countdown · U undo · F readout
python inspect_signs.py       # what is collected; is the signal in the features?
python rebuild_signs.py       # regenerate the training CSV from the archive
python train_signs.py         # cross-person report, then train
python demo_veronica.py       # live: signs -> gloss -> English
```

`demo_veronica.py` runs **without a trained model** — recognition is idle but
every layer beneath it is live, so the things that can only be judged in front
of a camera can be judged before a collection session rather than after one.

## Status

| Stage | State |
| --- | --- |
| 1 — two hands, orientation | ✅ done |
| 2 — location / face anchor | ✅ done |
| 3 — movement | ✅ done |
| 4 — collection | ✅ tool built; 250 clips, 13 signs, 1 signer |
| 5 — classifier | ✅ script ready; **needs a second signer to mean anything** |
| 6 — continuous signing | ✅ done; thresholds need real continuous signing to tune |
| 7 — gloss → English | ✅ done |
| 8 — non-manual markers | ⬜ not started |

### Where the data stands

250 clips, 13 signs, one signer. A leave-one-clip-out **nearest-centroid floor
of 83.2%** (chance 7.7%) — the crudest possible classifier, so a real one should
beat it. The mistakes are phonologically sensible rather than random:
MOTHER/FATHER→HELLO are all near the head, TOMORROW→YESTERDAY differ mainly in
movement direction, YOU→ME are both pointing signs.

**None of that is an accuracy result.** One person in one sitting measures those
recordings, not the language.

## What is next, in order

1. **A second signer.** This is the only thing blocking a real number —
   leave-one-person-out needs two people, and `train_signs.py` refuses to
   report a headline without them. More clips from one person will not
   substitute; on the letter dataset a fourth signer was worth +4.7 points
   while tripling the rows from existing signers was worth nothing.
2. **More of the 68-gloss vocabulary**, whichever signs the next session can
   cover. `python rebuild_signs.py --check` lists what is missing.
3. **A fluent signer**, if one can be found. Every number here comes from
   hearing team members performing signs.
4. **Tune segmentation on real continuous signing.** The thresholds in
   `segment.py` are structural guesses; `debug_info()` puts them on screen.
5. **Stage 8, non-manual markers.** Needs the face model already running, which
   it is. `gloss.render(question=True)` is wired and waiting.

## Known limits

Recorded plainly, because claiming coverage this does not have would be the
same mistake as quoting a shuffled-split accuracy.

- **One signer.** Cross-person accuracy cannot be measured at all yet.
- **Hearing signers.** Fine for letters, misleading for phrases — fluent
  signing differs in timing and coarticulation, which is exactly what stages 3
  and 6 model.
- **No non-manual markers.** A signed question rendered as a statement is a
  mistranslation, not a rough edge.
- **No aspect, spatial agreement, classifiers or role shift.** Real ASL grammar
  is much richer than `gloss.py` attempts.
- **Segmentation thresholds are guesses** until someone signs continuously.
- **The vocabulary is glosses only.** How each sign is formed is deliberately
  not encoded anywhere in the repository and should come from a dictionary and
  a fluent signer.
- **Frame rate is 15–25 fps on the dev laptop.** Above the ~11 fps floor, with
  little headroom. See the design doc on why blur, not fps, is the thing to fix
  for quick movements.
