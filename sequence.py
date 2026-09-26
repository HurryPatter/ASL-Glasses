"""Movement: turning a clip into a sign — Project Veronica stage 3.

The jump `main` cannot make by adding classes. A static classifier sees one
frame; a sign is a path through space and time. Movement is the fourth of the
five parameters (VERONICA.md), and unlike the first three it is not a property
of any single frame at all.

Why this is learned, not enumerated
-----------------------------------
`motion.py` recognises J and Z with hand-written trajectory rules, and its own
docstring is honest that each new sign means another predicate plus its own
thresholds. Two signs in, that approach has already produced two documented
false-fire bugs — a held Q emitting Z from fingertip jitter, and Z failing to
fire at all because three strokes did not fit inside a 650ms window. Neither
was a tuning mistake; both were the approach reaching its limit. A vocabulary
of dozens cannot be a few dozen more hand-tuned predicates.

So: turn a variable-length clip into a **fixed-length vector** and let the same
cheap MLP classify it. No RNN, no TensorFlow, no per-sign thresholds. That
keeps the "lightweight, fully on-device" result that dropping TensorFlow bought,
and it means stage 5 reuses `train_classifier.py`'s methodology unchanged.

Sampling each parameter at the rate it actually changes
-------------------------------------------------------
Resampling all 107 floats at eight keyframes would be 856 inputs, against the
few thousand clips a realistic collection effort produces. That is not a model
that generalises; it is a model that memorises, and `NOTES.md` already records
what happens when this project measures memorisation and calls it accuracy.

A sign's **handshape** is near-constant — most signs have one, a few have two,
which is what `hands.SHAPE_INDICES` is sampled twice to capture. Its
**orientation, inter-hand relationship and body location** move continuously,
so `hands.DYNAMIC_INDICES` gets eight keyframes. Same clip, different rates,
about 370 floats instead of 856.

Two things carried over from motion.py verbatim
-----------------------------------------------
**Wall-clock time, never frame counts.** Keyframes are placed at even fractions
of the clip's real duration, so the same gesture produces the same vector at
30fps on a laptop and at 15fps on embedded hardware. `NOTES.md` lists this as a
deliberate non-change; resampling by frame index would quietly undo it.

**Whole-hand travel, not fingertip travel.** Trajectories are measured on the
wrist and scaled by hand size, so per-landmark jitter does not read as
movement. That distinction is exactly what stopped a stationary Q from firing Z.

And it inherits the same frame-rate floor: `min_samples` (8) has to be met
inside a sign, and a sign lasting ~700ms therefore needs about 11fps — the
figure NOTES.md already derives for J/Z from a different direction. That floor
is a hardware constraint, not a tuning preference.

Trajectory is self-relative, location is body-anchored, and both are kept
------------------------------------------------------------------------
The movement summary measures displacement **from where the clip started**, in
hand-widths, so it works with no face in frame. The keyframed location block
carries **absolute position on the body**, when a face is there. These answer
different questions — "the hand arced downward and reversed twice" versus "it
did so at the chin" — and a sign can need either. Neither is derivable from the
other, so both are in the vector.

Standard library only, like the layers below it.
"""
import math
from collections import namedtuple, deque

import hands

# One frame of evidence: wall-clock time, the 107-float frame vector, and each
# hand's (wrist_xy, size_px) anchor -- the anchors are what the trajectory is
# measured on, and they are deliberately not in the frame vector, because the
# frame vector normalizes position away by design.
Sample = namedtuple("Sample", "t features dom non")

SHAPE_KEYFRAMES = 2       # handshape barely moves within a sign
DYNAMIC_KEYFRAMES = 8     # orientation / relation / location move continuously
MOVEMENT_FLOATS_PER_HAND = 8

SIGN_FLOATS = (SHAPE_KEYFRAMES * len(hands.SHAPE_INDICES)
               + DYNAMIC_KEYFRAMES * len(hands.DYNAMIC_INDICES)
               + MOVEMENT_FLOATS_PER_HAND * 2
               + 2      # per-hand tracking coverage
               + 1)     # clip duration

# Bounds, for the same reason hands.MAX_SPAN exists: past these a value says
# only "a lot", while an unbounded input is free to dominate an MLP's first
# layer. All in hand-widths, or hand-widths per second.
MAX_PATH = 12.0
MAX_EXTENT = 8.0
MAX_SPEED = 20.0
MAX_REVERSALS = 8.0
MAX_DURATION_S = 3.0

# How long one sign's window is -- the span the live segmenter classifies,
# and therefore the span training windows are cut to. Kept in one place
# because a mismatch between the two cost ~12 points: training on whole takes
# (median 2s) and classifying 900ms live windows dropped a nearest-centroid
# floor from 83% to 67-72% on the first 250 clips.
SIGN_WINDOW_MS = 900

# A dropout longer than this is not bridged -- the hand is treated as having
# genuinely left, rather than its last pose being held indefinitely.
MAX_BRIDGE_MS = 300

# Ignore wander below this before calling it a change of direction. Same role
# as MotionDetector.deadband, and the same reason: a hand held still still
# drifts, and every drift has a direction.
LEG_DEADBAND = 0.15

_EPS = 1e-6


def _clamp(value, low, high):
    return low if value < low else (high if value > high else value)


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


# ── tracking gaps ──────────────────────────────────────────────────────────
def bridge_gaps(samples, max_gap_ms=MAX_BRIDGE_MS):
    """Feature vectors with short tracking dropouts filled in.

    MediaPipe drops hands most on the frames a sign needs most -- fast movement
    blurs the image, and a turning hand occludes itself -- so dropouts cluster
    inside the gesture. Without this, a keyframe landing in a dropout
    interpolates between a real hand and an all-zero block, producing a
    half-scale hand at an impossible position.

    Only a gap with a sighting on **both** sides is bridged, and only if it is
    shorter than `max_gap_ms`. That is what separates a dropout from a ghost.
    An earlier version also extended a hand's first and last sighting out to
    the clip edges, which meant a second hand detected for a single frame --
    a face or a background object read as a hand -- had its geometry held
    across the entire clip. That happened in 18 of the first 250 collected
    clips, nearly all MOTHER and FATHER, where the face is closest to the hand.

    The presence flags are never filled. They keep telling the truth, and after
    resampling they come out fractional across a bridged gap, which is the
    signal that a stretch was interpolated.
    """
    vectors = [list(s.features) for s in samples]
    times = [s.t for s in samples]
    for start, length, flag in hands.BLOCKS:
        body = slice(start + 1, start + length)
        seen = [i for i, v in enumerate(vectors) if v[flag]]
        for before, after in zip(seen, seen[1:]):
            if after - before > 1 and times[after] - times[before] <= max_gap_ms:
                for i in range(before + 1, after):
                    vectors[i][body] = vectors[before][body]
    return vectors


# ── resampling ─────────────────────────────────────────────────────────────
def resample(samples, k):
    """k frame vectors at evenly spaced times across the clip.

    Spaced by **time**, not by index, which is what makes the result the same
    for a gesture performed identically at two different frame rates. A device
    that drops frames mid-sign gets the same vector as one that does not, as
    long as it stays above the sample floor.
    """
    if not samples:
        return []
    filled = bridge_gaps(samples)
    if len(samples) == 1 or samples[-1].t <= samples[0].t:
        return [list(filled[0]) for _ in range(k)]
    if k == 1:
        return [list(filled[0])]

    start, end = samples[0].t, samples[-1].t
    out = []
    cursor = 0
    for i in range(k):
        target = start + (end - start) * i / (k - 1)
        while cursor + 2 < len(samples) and samples[cursor + 1].t < target:
            cursor += 1
        span = samples[cursor + 1].t - samples[cursor].t
        ratio = 0.0 if span <= 0 else _clamp(
            (target - samples[cursor].t) / span, 0.0, 1.0)
        blended = [a + (b - a) * ratio
                   for a, b in zip(filled[cursor], filled[cursor + 1])]
        out.append(_renormalize_unit_pairs(blended))
    return out


def _renormalize_unit_pairs(vector):
    """Put interpolated (cos, sin) pairs back on the unit circle.

    Blending two unit vectors linearly produces a shorter one -- the chord,
    not the arc. Left alone, a rotation halfway between two keyframes would
    read as a rotation of smaller magnitude rather than an intermediate angle,
    and the error grows with the gap, i.e. it is worst on exactly the
    low-frame-rate hardware this is meant to survive.
    """
    for cos_i, sin_i in hands.UNIT_PAIR_INDICES:
        norm = math.hypot(vector[cos_i], vector[sin_i])
        if norm > _EPS:
            vector[cos_i] /= norm
            vector[sin_i] /= norm
    return vector


# ── trajectory ─────────────────────────────────────────────────────────────
def trajectory(samples, which):
    """[(t, x, y)] for one hand, in hand-widths from where the clip started.

    Only frames where that hand was tracked contribute, so a dropout shortens
    the path rather than teleporting it to the origin. The scale is the
    *median* hand size across the clip, not the first frame's: one badly
    estimated frame at the start would otherwise rescale the whole trajectory.
    """
    present = [s for s in samples if getattr(s, which) is not None]
    if not present:
        return [], 0.0

    unit = _median([getattr(s, which)[1] for s in present]) or _EPS
    ox, oy = getattr(present[0], which)[0]
    path = [(s.t, (getattr(s, which)[0][0] - ox) / unit,
             (getattr(s, which)[0][1] - oy) / unit) for s in present]
    return path, len(present) / len(samples)


def _legs(path):
    """Split a 2-D path into direction vectors, ignoring jitter.

    The two-dimensional generalisation of MotionDetector._strokes(): accumulate
    displacement until it clears the deadband, emit that as one leg, reset.
    Short wander never becomes a leg, so it can never become a reversal.
    """
    legs = []
    ax = ay = 0.0
    for (_, x0, y0), (_, x1, y1) in zip(path, path[1:]):
        ax += x1 - x0
        ay += y1 - y0
        if math.hypot(ax, ay) >= LEG_DEADBAND:
            legs.append((ax, ay))
            ax = ay = 0.0
    return legs


def movement_summary(path):
    """8 floats describing the shape of one hand's path.

    Chosen so that the distinctions ASL actually makes survive:
    straightness separates a straight movement from a circular one (a circle
    returns to its start, so net displacement is near zero while path length is
    not), and reversals count the repetitions that many signs carry as part of
    their form rather than as emphasis.
    """
    if len(path) < 2:
        return [0.0] * MOVEMENT_FLOATS_PER_HAND

    xs = [p[1] for p in path]
    ys = [p[2] for p in path]

    net_x, net_y = xs[-1] - xs[0], ys[-1] - ys[0]
    net = math.hypot(net_x, net_y)

    length = 0.0
    peak = 0.0
    for (t0, x0, y0), (t1, x1, y1) in zip(path, path[1:]):
        step = math.hypot(x1 - x0, y1 - y0)
        length += step
        dt = (t1 - t0) / 1000.0
        if dt > 0:
            peak = max(peak, step / dt)

    # A path that went nowhere is trivially straight; path_length is the float
    # that says whether it moved at all, so this one does not have to.
    straightness = 1.0 if length < _EPS else _clamp(net / length, 0.0, 1.0)

    legs = _legs(path)
    reversals = sum(1 for a, b in zip(legs, legs[1:])
                    if a[0] * b[0] + a[1] * b[1] < 0)

    return [
        _clamp(net_x, -MAX_EXTENT, MAX_EXTENT),
        _clamp(net_y, -MAX_EXTENT, MAX_EXTENT),
        _clamp(length, 0.0, MAX_PATH),
        straightness,
        _clamp(max(xs) - min(xs), 0.0, MAX_EXTENT),
        _clamp(max(ys) - min(ys), 0.0, MAX_EXTENT),
        _clamp(peak, 0.0, MAX_SPEED),
        _clamp(float(reversals), 0.0, MAX_REVERSALS),
    ]


# ── the sign vector ────────────────────────────────────────────────────────
def sign_vector(samples):
    """One clip -> one fixed-length vector, whatever its length or frame rate."""
    if not samples:
        return [0.0] * SIGN_FLOATS

    shape = []
    for frame in resample(samples, SHAPE_KEYFRAMES):
        shape.extend(frame[i] for i in hands.SHAPE_INDICES)

    dynamic = []
    for frame in resample(samples, DYNAMIC_KEYFRAMES):
        dynamic.extend(frame[i] for i in hands.DYNAMIC_INDICES)

    dom_path, dom_coverage = trajectory(samples, "dom")
    non_path, non_coverage = trajectory(samples, "non")

    duration_s = _clamp((samples[-1].t - samples[0].t) / 1000.0,
                        0.0, MAX_DURATION_S)

    return (shape + dynamic
            + movement_summary(dom_path) + movement_summary(non_path)
            + [dom_coverage, non_coverage, duration_s])


def _movement_columns(prefix):
    return [f"{prefix}_net_x", f"{prefix}_net_y", f"{prefix}_path_length",
            f"{prefix}_straightness", f"{prefix}_extent_x", f"{prefix}_extent_y",
            f"{prefix}_peak_speed", f"{prefix}_reversals"]


SIGN_COLUMNS = (
    [f"k{k}_{hands.FEATURE_COLUMNS[i]}"
     for k in range(SHAPE_KEYFRAMES) for i in hands.SHAPE_INDICES]
    + [f"d{k}_{hands.FEATURE_COLUMNS[i]}"
       for k in range(DYNAMIC_KEYFRAMES) for i in hands.DYNAMIC_INDICES]
    + _movement_columns("dom_move") + _movement_columns("non_move")
    + ["dom_coverage", "non_coverage", "duration_s"]
)


# ── live buffering ─────────────────────────────────────────────────────────
class SignBuffer:
    """A rolling window of samples, for the live pipeline.

    Collection (stage 4) does not need this — a clip there is delimited by a
    keypress — but stage 6 has to decide where a sign starts and ends with
    nobody pressing anything, and this is the buffer it will decide over.

    Pruned by time, never by a maxlen, for the reason motion.py gives: a
    frame-count window is a different real-world duration on every device, and
    the embedded target will not match the dev laptop's frame rate.
    """

    def __init__(self, window_ms=2500, min_samples=8, min_span_ms=250):
        self.window_ms = window_ms       # longest clip a sign may span
        self.min_samples = min_samples   # floor, so 2-3 noisy points can't be a sign
        self.min_span_ms = min_span_ms   # ...and neither can a 30ms flicker
        self.samples = deque()

    def add(self, timestamp_ms, features, dom_anchor=None, non_anchor=None):
        self.samples.append(Sample(timestamp_ms, list(features),
                                   dom_anchor, non_anchor))
        cutoff = timestamp_ms - self.window_ms
        while self.samples and self.samples[0].t < cutoff:
            self.samples.popleft()

    def clear(self):
        self.samples.clear()

    def __len__(self):
        return len(self.samples)

    def span_ms(self):
        if len(self.samples) < 2:
            return 0
        return self.samples[-1].t - self.samples[0].t

    def ready(self):
        """Enough evidence to be worth classifying.

        Both conditions matter, and they fail on different hardware: too few
        samples means a slow camera, too short a span means a fast one caught a
        flicker. Requiring min_samples inside a real sign is what sets the
        frame-rate floor for the glasses -- about 11fps for a 700ms sign.
        """
        return len(self.samples) >= self.min_samples and self.span_ms() >= self.min_span_ms

    def vector(self):
        return sign_vector(list(self.samples)) if self.ready() else None

    def debug_info(self):
        """Live values for tuning, mirroring MotionDetector.debug_info()."""
        if not self.ready():
            return {"clip": f"{len(self.samples)}/{self.min_samples} samples, "
                            f"{self.span_ms()}/{self.min_span_ms}ms"}
        dom_path, dom_coverage = trajectory(list(self.samples), "dom")
        summary = movement_summary(dom_path)
        return {
            "clip": f"{len(self.samples)} samples / {self.span_ms()}ms",
            "coverage": f"{dom_coverage:.0%}",
            "path_length": round(summary[2], 2),
            "straightness": round(summary[3], 2),
            "peak_speed": round(summary[6], 2),
            "reversals": int(summary[7]),
        }
