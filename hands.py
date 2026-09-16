"""Two-hand, orientation-aware feature layer — Project Veronica stage 1.

The `main` pipeline describes a sign with 42 floats: one hand, in a frame that
deliberately removes position, scale and in-plane rotation. That is the right
representation for fingerspelling and it measures 84% cross-person, but it can
only ever express **one of the five parameters** linguists use to describe an
ASL sign:

    handshape    ✓ the existing 42 floats
    orientation  ✗ normalized away (this is the documented G/Q confusion)
    location     ✓ location.py, face-anchored (stage 2)
    movement     ✗ invisible to a single frame (stage 3)
    non-manuals  ✗ not a hand at all (stage 8)

This module adds the two that are recoverable from hand landmarks alone —
orientation, and everything about a *second* hand — and assembles them with the
face-anchored location block from `location.py` into the per-frame vector that
stage 3 will consume. See VERONICA.md for why that order.

Layout of the 107-float frame vector
------------------------------------
    [  0: 46)  dominant hand block
    [ 46: 92)  non-dominant hand block
    [ 92:100)  relational block      (how the hands stand to each other)
    [100:107)  location block        (where they are on the body — location.py)

A hand block is 46 floats:

    present      1    0.0 when that hand is not in frame
    shape       42    the existing hand-frame normalization, UNCHANGED
    orient_cos   1    direction of wrist->middle-knuckle, in image space
    orient_sin   1
    mirrored     1    whether the thumb-on-+x flip was applied

The 42 shape floats are bit-for-bit what `normalize_landmarks()` produces on
`main`, which is the point: the existing 24,497 rows stay meaningful, and a
Veronica model can be initialised from, or compared against, the letter model
without re-deriving anything. `orient_*` and `mirrored` are exactly the
information that normalization threw away — recorded rather than discarded, so
the classifier can use orientation where it is contrastive (G vs Q) and ignore
it where it is not.

The relational block is 8 floats and is what a one-hand pipeline cannot have at
all:

    both_present  1   0.0 unless two hands are in frame
    dx, dy        2   non-dominant wrist relative to dominant, in hand-widths
    distance      1   |(dx, dy)|
    size_ratio    1   non-dominant hand size / dominant hand size
    rel_cos       1   rotation of the non-dominant hand relative to the dominant
    rel_sin       1
    contact       1   closest approach of the two hands, in hand-widths

Everything relational is measured in **hand-widths of the dominant hand**
(wrist -> middle knuckle), never pixels. A person's hand is a fixed physical
size, so dividing by it cancels camera distance the same way the hand frame
does — and it needs no body model, no depth and no second network, which is
what keeps the "fully on-device, lightweight" result intact.

Dominance, not left/right
-------------------------
Hands are ordered **dominant first**, not left-then-right. ASL is
handedness-symmetric: a left-handed signer produces the mirror image of a
right-handed signer's sign, with the same meaning. Ordering by dominance and
mirroring left-dominant signers into right-dominant space (`mirror_scene`)
puts both in one feature space, so a left-handed signer's rows are directly
comparable instead of being a separate, unlearnable class. This costs one
negation and is the difference between left-handed data helping and hurting.

Standard library only, deliberately: CI installs nothing, and this is the layer
every later stage sits on, so it is the one that most needs tests that actually
run. It also has to keep up on whatever the glasses end up running.
"""
import math

import location

# MediaPipe hand landmark indices
WRIST = 0
THUMB_MCP = 2
MIDDLE_MCP = 9
N_POINTS = 21

SHAPE_FLOATS = 42
PER_HAND_FLOATS = 46
RELATIONAL_FLOATS = 8
LOCATION_FLOATS = location.LOCATION_FLOATS                 # 7
FEATURE_FLOATS = (PER_HAND_FLOATS * 2 + RELATIONAL_FLOATS
                  + LOCATION_FLOATS)                       # 107

DOMINANT_OFFSET = 0
NONDOMINANT_OFFSET = PER_HAND_FLOATS
RELATIONAL_OFFSET = PER_HAND_FLOATS * 2
LOCATION_OFFSET = RELATIONAL_OFFSET + RELATIONAL_FLOATS

RIGHT = "Right"
LEFT = "Left"

# Beyond a few hand-widths apart, the exact distance carries no linguistic
# information -- the hands are simply "not near each other" -- while an
# unbounded input is free to dominate an MLP's first layer. Clamping keeps
# every relational float in a comparable range without losing a distinction
# any sign actually makes.
MAX_SPAN = 6.0
MAX_SIZE_RATIO = 4.0

_EPS = 1e-6


def _clamp(value, low, high):
    return low if value < low else (high if value > high else value)


# ── adapters ───────────────────────────────────────────────────────────────
def to_pixels(landmarks, frame_w, frame_h):
    """MediaPipe landmarks -> [(x, y)] in pixels.

    The pixel conversion is not cosmetic. MediaPipe reports x and y each
    normalized to [0, 1] against a *different* denominator (width and height),
    so on a non-square frame the pair is anisotropically scaled and every angle
    in it is wrong. Undoing that here is what makes the hand frame, the
    orientation angle and the inter-hand geometry all measure the same space.
    """
    return [(lm.x * frame_w, lm.y * frame_h) for lm in landmarks]


def mirror_scene(points):
    """Reflect one hand's points about the vertical axis.

    Applied to *every* hand in the frame together, this converts a left-dominant
    signer into the right-dominant frame the features are defined in. Reflecting
    about x=0 rather than about the frame centre is intentional: every quantity
    downstream is a difference of positions, and a reflection changes those
    identically wherever the axis sits.
    """
    return [(-x, y) for x, y in points]


# ── one hand ───────────────────────────────────────────────────────────────
def hand_frame(points):
    """(shape42, orient_cos, orient_sin, mirrored, size_px) for one hand.

    `shape42` is identical to what `normalize_landmarks()` returns on `main`;
    the other three are the degrees of freedom that function discards.
    `size_px` — the wrist -> middle-knuckle length — is the unit every
    relational measurement is expressed in.
    """
    wx, wy = points[WRIST]
    mx, my = points[MIDDLE_MCP]
    vx, vy = mx - wx, my - wy
    size = math.hypot(vx, vy) or _EPS

    # Hand frame: +y along wrist->middle-knuckle, +x perpendicular to it.
    y_hat_x, y_hat_y = vx / size, vy / size
    x_hat_x, x_hat_y = y_hat_y, -y_hat_x

    local = []
    for px, py in points:
        dx, dy = px - wx, py - wy
        local.append(((dx * x_hat_x + dy * x_hat_y) / size,
                      (dx * y_hat_x + dy * y_hat_y) / size))

    # Thumb knuckle on -x means we are looking at the other hand, or the same
    # hand palm-reversed. Flipping normalizes both into one shape space -- and
    # is precisely why the shape floats alone cannot tell left from right,
    # hence recording the flag.
    mirrored = local[THUMB_MCP][0] < 0
    if mirrored:
        local = [(-x, y) for x, y in local]

    shape = [component for point in local for component in point]
    return shape, y_hat_x, y_hat_y, (1.0 if mirrored else 0.0), size


def hand_block(points):
    """46 floats describing one present hand."""
    shape, orient_cos, orient_sin, mirrored, _ = hand_frame(points)
    return [1.0] + shape + [orient_cos, orient_sin, mirrored]


def absent_hand_block():
    """46 zeros. `present` leading the block is what lets the classifier tell
    'this hand is missing' from 'this hand is at the origin, unrotated' --
    zeros alone are a perfectly valid hand pose."""
    return [0.0] * PER_HAND_FLOATS


# ── two hands ──────────────────────────────────────────────────────────────
def relational_block(dominant, nondominant):
    """8 floats describing how the two hands stand relative to each other.

    All zeros when either hand is missing, which is the common case: most of
    the ASL lexicon is one-handed, and the leading `both_present` flag is what
    stops the classifier reading those zeros as a real configuration.
    """
    if dominant is None or nondominant is None:
        return [0.0] * RELATIONAL_FLOATS

    _, dom_cos, dom_sin, _, dom_size = hand_frame(dominant)
    _, non_cos, non_sin, _, non_size = hand_frame(nondominant)
    unit = dom_size or _EPS

    dx = (nondominant[WRIST][0] - dominant[WRIST][0]) / unit
    dy = (nondominant[WRIST][1] - dominant[WRIST][1]) / unit
    distance = math.hypot(dx, dy)

    # Rotation of the non-dominant hand *relative to* the dominant one, as
    # cos/sin of the difference. A bare angle would jump by 2*pi across the
    # wrap point, and a classifier reading it as a number would see two nearly
    # identical hand configurations as maximally far apart.
    rel_cos = non_cos * dom_cos + non_sin * dom_sin
    rel_sin = non_sin * dom_cos - non_cos * dom_sin

    # Closest approach of any two landmarks. Contact is phonemic in ASL --
    # signs differing only in whether the hands touch are distinct signs -- and
    # wrist separation alone cannot see it, since two hands can have distant
    # wrists and touching fingertips.
    closest = min(math.hypot(ax - bx, ay - by)
                  for ax, ay in dominant
                  for bx, by in nondominant) / unit

    return [
        1.0,
        _clamp(dx, -MAX_SPAN, MAX_SPAN),
        _clamp(dy, -MAX_SPAN, MAX_SPAN),
        _clamp(distance, 0.0, MAX_SPAN),
        _clamp(non_size / unit, 0.0, MAX_SIZE_RATIO),
        rel_cos,
        rel_sin,
        _clamp(closest, 0.0, MAX_SPAN),
    ]


def anchor(points):
    """(wrist_xy, hand_size_px) — what location.py needs from a hand.

    Passing this rather than the landmark list is what keeps location.py from
    importing this module, so the two blocks stay independently testable.
    """
    if points is None:
        return None
    return points[WRIST], hand_frame(points)[4]


def feature_vector(dominant, nondominant, face=None):
    """The 107-float Veronica frame vector.

    Either hand may be None, and so may `face` — a detector that loses the
    signer's face for a few frames zeroes the location block rather than
    invalidating the whole vector, because the other 100 floats are still
    perfectly good. Each block leads with its own presence flag so partial
    evidence stays distinguishable from evidence of absence, which is the same
    distinction debouncer.py had to make to stop one held letter committing
    ten times.
    """
    dom = hand_block(dominant) if dominant is not None else absent_hand_block()
    non = hand_block(nondominant) if nondominant is not None else absent_hand_block()
    return (dom + non
            + relational_block(dominant, nondominant)
            + location.location_block(anchor(dominant), anchor(nondominant), face))


# ── which hand is which ────────────────────────────────────────────────────
def assign_hands(detected, signer_dominant=RIGHT, mirrored_input=True):
    """Order detected hands as (dominant, nondominant), either possibly None.

    `detected` is [(points, handedness_label)] — one entry per hand MediaPipe
    found, with its "Left"/"Right" classification.

    `mirrored_input` describes the *image*, and it is the setting most likely
    to be wrong. MediaPipe labels handedness on the assumption that the frame
    is mirrored, as in a selfie view; `main.py` does `cv2.flip(frame, 1)`, so
    that assumption holds today and the labels are used as they come. Pass
    False for an un-flipped feed and the labels are swapped instead.

    Watch this one when the pipeline moves to the glasses. The dev setup has
    the signer at their own laptop looking at a mirrored preview of themselves;
    the deployed glasses see a *different* person, facing the wearer. Those two
    geometries are reflections of each other, so a convention that is silently
    wrong produces data where every left hand is labelled right — trained on
    happily, and wrong in exactly the way nothing downstream can detect.
    """
    hands = list(detected)
    if not hands:
        return None, None

    if signer_dominant == LEFT:
        # Reflect the whole scene so a left-handed signer lands in the same
        # feature space as everyone else. The reflection also turns each hand
        # into its opposite, so the labels flip with the geometry.
        hands = [(mirror_scene(points), label) for points, label in hands]
        mirrored_input = not mirrored_input

    def resolved(label):
        if label not in (LEFT, RIGHT):
            return None
        if mirrored_input:
            return label
        return LEFT if label == RIGHT else RIGHT

    labels = [resolved(label) for _, label in hands]

    # MediaPipe can hand back two hands with the same label, usually when one
    # is partly occluded. Fall back to image position, under the same mirror
    # convention the labels use: in a mirrored frame the signer's right hand
    # appears on the right.
    if len(hands) == 2 and labels[0] == labels[1]:
        x0 = hands[0][0][WRIST][0]
        x1 = hands[1][0][WRIST][0]
        right_first = (x0 > x1) if mirrored_input else (x0 < x1)
        labels = [RIGHT, LEFT] if right_first else [LEFT, RIGHT]

    dominant = nondominant = None
    for (points, _), label in zip(hands, labels):
        if label == RIGHT and dominant is None:
            dominant = points
        elif label == LEFT and nondominant is None:
            nondominant = points

    # An unlabelled or duplicate-resolved leftover still belongs somewhere:
    # a single tracked hand is far more often the dominant one.
    if dominant is None and nondominant is None:
        dominant = hands[0][0]

    return dominant, nondominant


# ── schema ─────────────────────────────────────────────────────────────────
def _hand_columns(prefix):
    columns = [f"{prefix}_present"]
    columns += [f"{prefix}_p{i}_{axis}" for i in range(N_POINTS) for axis in ("x", "y")]
    columns += [f"{prefix}_orient_cos", f"{prefix}_orient_sin", f"{prefix}_mirrored"]
    return columns


FEATURE_COLUMNS = (
    _hand_columns("dom")
    + _hand_columns("non")
    + ["rel_both_present", "rel_dx", "rel_dy", "rel_distance",
       "rel_size_ratio", "rel_cos", "rel_sin", "rel_contact"]
    + location.LOCATION_COLUMNS
)
