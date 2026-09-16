"""Schema, vocabulary, and raw→features reconstruction for the sign dataset —
Project Veronica stage 4.

Standard library only, like `dataset.py`: this is the module that turns an
archived clip back into a training row, which makes it the one place a silent
mistake would corrupt everything downstream. It runs in CI.

Collect raw, derive features
----------------------------
Collection writes **two** files:

    veronica_clips.jsonl   raw landmarks, one JSON object per clip — the archive
    veronica_signs.csv     the 371-float training rows — derived, regenerable

The whole point of sequencing stages 1–3 before collection was to avoid asking
people to sign everything twice. Storing only the feature vectors would have
protected that exactly once: the next time `sequence.py`'s keyframe counts are
revisited — and stages 5–8 will revisit them — every row would be invalid and
the sessions would have to happen again.

Keeping the raw landmarks makes that a script run instead
(`python rebuild_signs.py`). The archive is the record; the CSV is a build
artifact. `dataset.py` learned this lesson the expensive way in reverse: the
existing 24,496 rows cannot supply orientation because normalization ran
*before* anything reached disk, so that parameter is gone for good.

The archive is larger — measured at about **6x the derived CSV** — but it is
plain JSON Lines of short repeated keys, which is close to the best case for
the zlib compression git already applies to every blob. A full collection
effort (roughly 8,000 clips) projects to a few hundred MB on disk and a small
fraction of that in the repository. Commit it; revisit Git LFS only if it
actually grows past what git is comfortable with, and never solve the size by
dropping it — that is the one file that cannot be regenerated.
"""
import csv
import json
import math

import config
import hands
import location
import sequence

CLIPS_PATH = "veronica_clips.jsonl"
SIGNS_PATH = "veronica_signs.csv"

# Not signing. Worth the two minutes it costs at collection time: stage 5 needs
# something to reject garbage with, and stage 6 has to tell "between signs"
# from "a sign" with nobody pressing a key. Collecting it later means another
# session with every signer.
REST_LABEL = "_REST"

# A starting vocabulary of everyday conversational ASL — roughly 60 signs,
# chosen for how often they are actually used rather than for how easy they are
# to recognise. Full ASL is tens of thousands of signs; a defensible thesis
# result is a real vocabulary measured honestly, not a claim to cover the
# language.
#
# Edit freely — nothing downstream knows these particular strings, exactly as
# collect_data.py's WORDS list works today. Two things to keep if you do:
# MOTHER and FATHER, because they differ only in location and so are the
# working test that stage 2 earns its keep; and _REST.
#
# Glosses only. How each sign is actually formed is not encoded anywhere here
# and should come from a dictionary and a fluent signer, not from this file.
VOCABULARY = {
    "greetings": ["HELLO", "GOODBYE", "PLEASE", "THANK-YOU", "SORRY",
                  "EXCUSE-ME", "NICE-TO-MEET-YOU"],
    "pronouns": ["ME", "YOU", "HE-SHE", "WE", "THEY"],
    "questions": ["WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW"],
    "responses": ["YES", "NO", "MAYBE", "DONT-KNOW", "UNDERSTAND",
                  "DONT-UNDERSTAND"],
    "needs": ["HELP", "WANT", "NEED", "STOP", "WAIT", "FINISH"],
    "feelings": ["GOOD", "BAD", "HAPPY", "SAD", "TIRED", "SICK", "HUNGRY",
                 "THIRSTY", "HURT"],
    "people": ["MOTHER", "FATHER", "FRIEND", "FAMILY", "NAME", "DEAF",
               "HEARING"],
    "everyday": ["EAT", "DRINK", "GO", "COME", "WORK", "SCHOOL", "HOME",
                 "BATHROOM", "WATER", "MONEY"],
    "time": ["NOW", "TODAY", "TOMORROW", "YESTERDAY", "LATER"],
    "modifiers": ["MORE", "AGAIN", "SLOW", "FAST", "LOVE", "LIKE"],
    "control": [REST_LABEL],
}

SIGNS = [sign for group in VOCABULARY.values() for sign in group]

# Metadata, not features. `clip_id` is what lets a suspicious training row be
# traced back to the exact clip in the archive -- without it, a class that
# trains badly is a mystery rather than a recording to go and look at.
META_COLUMNS = ["clip_id", "label", "person", "dominant", "mirrored_input",
                "n_frames", "clip_ms", "face_coverage", "handedness_stability"]
HEADER = META_COLUMNS + sequence.SIGN_COLUMNS

# Below this, the clip has no usable location data for most of its span, so
# every location feature is mostly the zeroed no-face block. Kept rather than
# dropped -- filtering later is easy, and losing a good take is not -- but
# flagged at collection time, while re-recording still costs seconds.
MIN_FACE_COVERAGE = 0.5

# Below this, MediaPipe disagreed with itself about which hand it was looking
# at for a good part of the clip. Stabilisation repairs the identity, but a
# clip this unstable usually also has poor tracking underneath it.
MIN_HANDEDNESS_STABILITY = 0.8

_COORD_PLACES = 4      # ~0.06px on a 640px frame; far finer than the tracker


def _round(value):
    return round(float(value), _COORD_PLACES)


# ── the archive ────────────────────────────────────────────────────────────
def raw_frame(detected, face_box, timestamp_ms):
    """One frame of the archive.

    Landmarks are stored exactly as MediaPipe reported them -- normalized to
    [0, 1], un-mirrored, un-assigned -- together with the frame size, so any
    future feature layer can start from the same place this one does. Storing
    the derived hand frame instead is precisely the mistake that cost the
    letter dataset its orientation.
    """
    return {
        "t": int(timestamp_ms),
        "hands": [{"label": label,
                   "points": [[_round(x), _round(y)] for x, y in points]}
                  for points, label in detected],
        "face": None if face_box is None else [_round(v) for v in face_box],
    }


def raw_clip(clip_id, label, person, dominant, frame_w, frame_h, frames,
             mirrored_input=True):
    """One archived clip.

    `mirrored_input` is stored per clip rather than assumed, because it cannot
    be settled by reading code -- it depends on the MediaPipe build and on the
    camera, some of which deliver an already-mirrored feed. A wrong value
    labels every left hand right, undetectably.

    Storing it is what makes that recoverable. If the convention is later found
    to have been wrong, `rebuild_signs.py --mirrored-input` re-derives every
    training row from these raw landmarks under the corrected one, and nobody
    signs anything twice. Clips recorded before this field existed default to
    True, which is what they were collected under.
    """
    return {
        "clip_id": clip_id,
        "label": label,
        "person": person,
        "dominant": dominant,
        "mirrored_input": bool(mirrored_input),
        "frame_w": int(frame_w),
        "frame_h": int(frame_h),
        "frames": frames,
    }


def append_clip(path, clip):
    """One clip per line. Append-only, so a crash costs the clip in progress
    and nothing else."""
    with open(path, "a") as fh:
        fh.write(json.dumps(clip, separators=(",", ":")) + "\n")


def read_clips(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ── hand identity across a clip ────────────────────────────────────────────
# MediaPipe's per-frame handedness is not stable. It flips, most readily when a
# hand rotates so the palm turns away from the camera -- which ASL does
# constantly, since orientation is one of the five parameters a sign is built
# from. Observed directly on this project's own recordings.
#
# Trusting the per-frame label means a hand can change identity *mid-sign*. In
# a two-handed sign that swaps the dominant and non-dominant blocks partway
# through the clip, and the resulting feature vector describes a sign nobody
# made. So identity is resolved over the whole clip instead: hands are followed
# by position, and each track takes the majority label of its own frames.
#
# Following position rather than side-of-image is deliberate. Hands cross in
# ASL, and a rule like "the right hand is the one further right" breaks exactly
# when they do; continuity follows the hand through the crossing.
_MATCH_RADIUS = 0.25       # normalized image units between consecutive frames


def _wrist(hand):
    return hand["points"][hands.WRIST]


def _track_hands(frames):
    """-> (tracks, assignment), where assignment[i][j] is frame i hand j's track."""
    tracks = []            # each: {"last": (x, y), "labels": [...]}
    assignment = []

    for frame in frames:
        detections = frame.get("hands", [])
        taken = {}
        for index, hand in enumerate(detections):
            x, y = _wrist(hand)
            best, best_distance = None, _MATCH_RADIUS
            for track_index, track in enumerate(tracks):
                if track_index in taken.values():
                    continue
                tx, ty = track["last"]
                distance = math.hypot(x - tx, y - ty)
                if distance < best_distance:
                    best, best_distance = track_index, distance
            if best is None:
                tracks.append({"last": (x, y), "labels": []})
                best = len(tracks) - 1
            taken[index] = best
            tracks[best]["last"] = (x, y)
            tracks[best]["labels"].append(hand.get("label"))
        assignment.append(taken)

    return tracks, assignment


def _majority(labels):
    present = [l for l in labels if l]
    if not present:
        return None
    return max(set(present), key=present.count)


def stabilized_frames(clip):
    """The clip's frames with each hand's label replaced by its track's majority.

    Returns (frames, stability) where stability is the fraction of labelled
    detections that already agreed with their track -- 1.0 means MediaPipe
    never flipped, and a low value marks a clip worth looking at.
    """
    frames = clip.get("frames") or []
    tracks, assignment = _track_hands(frames)
    resolved = [_majority(track["labels"]) for track in tracks]

    agreed = total = 0
    out = []
    for frame, taken in zip(frames, assignment):
        rebuilt = []
        for index, hand in enumerate(frame.get("hands", [])):
            track_index = taken.get(index)
            label = resolved[track_index] if track_index is not None else hand.get("label")
            if hand.get("label"):
                total += 1
                agreed += int(hand["label"] == label)
            rebuilt.append({"label": label, "points": hand["points"]})
        copy = dict(frame)
        copy["hands"] = rebuilt
        out.append(copy)

    return out, (agreed / total if total else 1.0)


def handedness_stability(clip):
    return stabilized_frames(clip)[1]


# ── reconstruction: archive -> training row ────────────────────────────────
def clip_to_samples(clip, mirrored_input=None):
    """Rebuild the per-frame Sample stream from an archived clip.

    The one function that has to stay correct forever: every training row the
    project ever produces passes through it, and it is the only thing standing
    between the archive and needing the signers back.

    `mirrored_input` overrides what the clip recorded, which is how a whole
    archive collected under a wrong convention gets corrected without anyone
    signing again.
    """
    frame_w, frame_h = clip["frame_w"], clip["frame_h"]
    dominant_hand = clip.get("dominant", hands.RIGHT)
    if mirrored_input is None:
        # Clips predating the field were collected under the old default.
        mirrored_input = clip.get("mirrored_input", True)

    frames, _ = stabilized_frames(clip)

    samples = []
    for frame in frames:
        detected = [
            ([(x * frame_w, y * frame_h) for x, y in hand["points"]],
             hand.get("label"))
            for hand in frame.get("hands", [])
        ]
        box = frame.get("face")
        face = None if box is None else location.face_from_normalized_box(
            box[0], box[1], box[2], box[3], frame_w, frame_h)

        # canonical_scene, never assign_hands directly: it is what keeps the
        # hands and the face reflected together for a left-dominant signer.
        dom, non, face = hands.canonical_scene(
            detected, face, signer_dominant=dominant_hand,
            mirrored_input=mirrored_input)

        samples.append(sequence.Sample(
            frame["t"],
            hands.feature_vector(dom, non, face),
            hands.anchor(dom),
            hands.anchor(non),
        ))
    return samples


def face_coverage(clip):
    frames = clip.get("frames") or []
    if not frames:
        return 0.0
    return sum(1 for f in frames if f.get("face") is not None) / len(frames)


def clip_span_ms(clip):
    frames = clip.get("frames") or []
    if len(frames) < 2:
        return 0
    return frames[-1]["t"] - frames[0]["t"]


def clip_to_row(clip, mirrored_input=None):
    """An archived clip -> one CSV training row."""
    samples = clip_to_samples(clip, mirrored_input=mirrored_input)
    used = mirrored_input if mirrored_input is not None \
        else clip.get("mirrored_input", True)
    return [
        clip["clip_id"],
        clip["label"],
        clip["person"],
        clip.get("dominant", hands.RIGHT),
        bool(used),
        len(clip.get("frames") or []),
        clip_span_ms(clip),
        round(face_coverage(clip), 4),
        round(handedness_stability(clip), 4),
    ] + sequence.sign_vector(samples)


# ── the derived CSV ────────────────────────────────────────────────────────
def read_header(path):
    with open(path, newline="") as fh:
        return next(csv.reader(fh))


def write_rows(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)


def people(clips):
    return sorted({c["person"] for c in clips})


def counts_by_label(clips):
    counts = {}
    for clip in clips:
        counts[clip["label"]] = counts.get(clip["label"], 0) + 1
    return counts
