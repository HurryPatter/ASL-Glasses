"""Body-anchored location — Project Veronica stage 2.

The parameter no amount of hand landmark data can recover. FATHER and MOTHER
share a handshape, an orientation and a movement; one is made at the forehead
and one at the chin. To a pipeline that sees only hands they are the same sign,
forever. Location is the third of the five parameters (see VERONICA.md) and the
one that most limits how large a vocabulary is reachable at all.

Why the face, and not a pose model
----------------------------------
A body reference has to come from somewhere, and the cheapest useful one is the
signer's face: most location-contrastive signs are anchored to the head, a face
detector is small, and it leaves the "fully on-device, lightweight" result
intact where a second full pose model per frame would put it at risk — with the
embedded target still undecided (NOTES.md), that headroom is not free to spend.

The deployment geometry cooperates. The glasses see the *conversation partner*,
so the signer's face is in frame. It would not be if the camera were egocentric
on the signer themselves.

Why face WIDTH is the unit
--------------------------
Everything here is measured in face-widths, never pixels, for the same reason
the hand frame measures in hand-widths: a face is a fixed physical size, so
dividing by it cancels camera distance without depth, calibration or a second
sensor.

Width specifically, not height, and not the box diagonal. ASL uses head
movement grammatically — a nod, a headshake — and pitching the head compresses
the apparent *height* of a face box hard while leaving its width close to
alone. A unit that shrinks whenever the signer nods would make every location
feature move during exactly the constructions stage 8 has to read.

The free depth proxy
--------------------
`hand size / face width` is worth more than it looks. Both are fixed physical
sizes, so their ratio is roughly constant for a given person — *unless* the
hand is nearer the camera than the face is, which is precisely what a sign made
out in neutral space does, as against one contacting the body. That is a usable
proxy for the third dimension out of two monocular measurements, no depth
sensor involved.

Detector-agnostic on purpose
----------------------------
Nothing here imports MediaPipe or knows which detector produced the box. A face
is three numbers — centre and width — so the layer is testable offline (CI
installs nothing) and a better face model later is a change at the call site
rather than in here. Landmark-based anchoring (eye and mouth keypoints, which
are steadier than a box and would also give head rotation) is the natural
refinement, and it fits this interface unchanged.
"""
import math

# Signing space is roughly two face-widths either side of the head and about as
# far below it. Past that the hand is out of signing space entirely, and an
# unbounded input is free to dominate an MLP's first layer -- the same reasoning
# as hands.MAX_SPAN.
MAX_REACH = 4.0

# A hand is normally well under a face-width across. Much above 1 means it is
# nearer the camera than the face; far above that is a tracking artifact.
MAX_DEPTH_RATIO = 2.5

LOCATION_FLOATS = 7

LOCATION_COLUMNS = [
    "face_present",
    "dom_loc_x", "dom_loc_y", "dom_depth",
    "non_loc_x", "non_loc_y", "non_depth",
]

_EPS = 1e-6


def _clamp(value, low, high):
    return low if value < low else (high if value > high else value)


def face_from_box(x, y, width, height):
    """A detector's bounding box -> the (cx, cy, width, height) this module uses.

    Pixels, matching hands.to_pixels(): a face box and a hand landmark have to
    be in one coordinate space before any distance between them means anything.
    """
    return (x + width / 2.0, y + height / 2.0, width, height)


def face_from_normalized_box(x, y, width, height, frame_w, frame_h):
    """Same, for a detector reporting a box in [0, 1] against frame size."""
    return face_from_box(x * frame_w, y * frame_h, width * frame_w, height * frame_h)


def locate(point, face):
    """Where `point` sits relative to the face, in face-widths.

    Both axes are divided by the *same* number, so the frame stays isotropic
    and an angle measured in it is a real angle. Dividing x by width and y by
    height — the obvious thing — would stretch the space and make "up and to
    the left" mean something different at every head pitch.
    """
    cx, cy, width, _ = face
    unit = width or _EPS
    return ((point[0] - cx) / unit, (point[1] - cy) / unit)


def location_block(dominant, nondominant, face):
    """7 floats placing both hands against the signer's face.

    `dominant` / `nondominant` are `(wrist_xy, hand_size_px)` or None —
    deliberately not landmark lists, so this module stays independent of
    hands.py rather than the two importing each other.

    All zeros when no face was detected, which happens often enough to matter:
    the partner turns their head, or steps out of frame. `face_present` leading
    the block is what stops the classifier reading that as "both hands are
    exactly at the centre of the face".
    """
    if face is None:
        return [0.0] * LOCATION_FLOATS

    _, _, width, _ = face
    unit = width or _EPS

    def place(anchor):
        if anchor is None:
            return [0.0, 0.0, 0.0]
        wrist, size = anchor
        lx, ly = locate(wrist, face)
        return [
            _clamp(lx, -MAX_REACH, MAX_REACH),
            _clamp(ly, -MAX_REACH, MAX_REACH),
            _clamp(size / unit, 0.0, MAX_DEPTH_RATIO),
        ]

    return [1.0] + place(dominant) + place(nondominant)


# ── diagnostics (does not affect the features) ─────────────────────────────
def zone_name(point, face):
    """A readable name for where a hand is, for the on-screen debug readout.

    Not a feature and never fed to a classifier: the features stay raw and let
    the model learn where a chin falls, because these cut points are a guess
    and a hard-coded "chin" boundary that is slightly wrong is worse than no
    boundary at all. This exists for the same reason MotionDetector.debug_info()
    does -- the thresholds that do matter can only be judged against a real
    person in front of a real camera, and you cannot judge what you cannot see.

    Unlike the features, this one *does* use the box height, since naming the
    forehead and the chin means knowing where the face ends.
    """
    if face is None:
        return "no face"
    cx, cy, width, height = face
    half = (height or width) / 2.0
    dy = point[1] - cy
    dx = (point[0] - cx) / (width or _EPS)

    if dy < -half:
        band = "above head"
    elif dy < -half / 3.0:
        band = "forehead"
    elif dy < half / 3.0:
        band = "eyes/nose"
    elif dy < half:
        band = "mouth/chin"
    elif dy < half + (height or width):
        band = "neck/shoulder"
    else:
        band = "chest"

    side = "centre" if abs(dx) <= 0.75 else ("signer left" if dx < 0 else "signer right")
    return f"{band}, {side}"


def debug_info(dominant, nondominant, face):
    """Live location values, mirroring MotionDetector.debug_info()."""
    if face is None:
        return {"face": "not detected"}
    cx, cy, width, height = face
    info = {"face": f"{width:.0f}x{height:.0f}px at ({cx:.0f}, {cy:.0f})"}
    for name, anchor in (("dom", dominant), ("non", nondominant)):
        if anchor is None:
            info[name] = "absent"
            continue
        wrist, size = anchor
        lx, ly = locate(wrist, face)
        info[name] = (f"({lx:+.2f}, {ly:+.2f}) faces  "
                      f"depth {size / (width or _EPS):.2f}  "
                      f"{zone_name(wrist, face)}")
    return info
