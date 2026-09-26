"""MediaPipe adapter layer — the one place that touches the tracker's output.

Everything else in Veronica works on plain tuples of floats, which is what
makes `hands.py`, `location.py`, `sequence.py`, `signset.py`, `folds.py`,
`gloss.py` and `segment.py` standard-library-only and testable in CI. This file
is the seam: it builds the detectors and converts what they return into that
form, and it is the only module that imports MediaPipe.

It exists because the conversion has a trap in it. **MediaPipe Tasks reports a
face detection's bounding box in pixels while reporting hand landmarks
normalized to [0, 1]** — two conventions out of one library. A pixel box read
as normalized puts the face somewhere past the corner of the frame, and every
location feature downstream becomes a large, stable, entirely plausible-looking
number. Nothing crashes and nothing looks wrong.

That fix belongs in exactly one place, which is the whole reason for this
module: the collector, the live demo and the setup check all read the same
detectors, and three copies of a subtle conversion is three chances to fix two
of them.
"""
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import hands

HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "blaze_face_short_range.tflite"

FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                  "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")


def open_camera(index=0, default_backend=False):
    """Open the webcam, preferring DirectShow + MJPG on Windows.

    OpenCV's default Windows backend (Media Foundation) is known to be slow to
    deliver frames and to cap or jitter the frame rate on many webcams, and
    Veronica needs every frame it can get -- motion signs stop being
    assemblable below ~11fps. DirectShow with MJPG compression usually lets the
    camera run at its full 30fps.

    Resolution is pinned at 640x480, the same as every clip collected so far:
    a different resolution changes how well MediaPipe tracks, which would be a
    fresh train/live mismatch of exactly the kind this is trying to remove.

    Falls back to the default backend if DirectShow cannot open the camera or
    returns no frame, so this cannot make things worse than before. Pass
    default_backend=True to compare the two -- `check_setup.py` reports which
    one it got and the fps it measured.
    """
    import sys
    import cv2

    if sys.platform.startswith("win") and not default_backend:
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cam.isOpened():
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cam.set(cv2.CAP_PROP_FPS, 30)
            ok, _ = cam.read()
            if ok:
                return cam
        cam.release()
    return cv2.VideoCapture(index)


def build_landmarker(num_hands=2, model_path=HAND_MODEL):
    """num_hands=2 by default -- about half the ASL lexicon is two-handed, and
    `main.py`'s num_hands=1 is what stage 1 exists to move past."""
    return mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6))


def build_face_detector(model_path=FACE_MODEL):
    return mp_vision.FaceDetector.create_from_options(
        mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            min_detection_confidence=0.5))


class PeriodicFace:
    """Run the face detector every Nth frame and reuse the last box between.

    The face detector is a whole second model per frame, and it is the largest
    cost in the live loop after hand tracking. It is also the most wasteful:
    a head moves slowly compared to hands, and location is measured in
    face-widths, so a box a frame or two old is very nearly the same box.

    The tradeoff is real but small -- during a fast head turn the box lags by
    up to `every` frames, which shifts the location features slightly. Set
    every=1 to disable.
    """

    def __init__(self, detector, every=3):
        self.detector = detector
        self.every = max(1, every)
        self.box = None
        self._count = 0

    def update(self, image, timestamp_ms, frame_w, frame_h):
        """-> the current normalized (x, y, w, h) box, or None."""
        if self.detector is None:
            return None
        if self._count % self.every == 0:
            self.box = largest_face(
                self.detector.detect_for_video(image, timestamp_ms),
                frame_w, frame_h)
        self._count += 1
        return self.box


def to_mp_image(bgr_frame):
    import cv2
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))


def detected_hands(result):
    """-> [(normalized points, handedness label)].

    Handedness is passed through as reported. That is correct while the frame
    is mirrored -- every caller does cv2.flip -- and wrong if it ever stops
    being. See hands.assign_hands(mirrored_input=...) and VERONICA.md stage 1;
    `check_setup.py` is what confirms it against a real hand, because nothing
    downstream can.
    """
    out = []
    for index, landmarks in enumerate(result.hand_landmarks):
        label = None
        if index < len(result.handedness) and result.handedness[index]:
            label = result.handedness[index][0].category_name
        out.append(([(lm.x, lm.y) for lm in landmarks], label))
    return out


def largest_face(result, frame_w, frame_h):
    """-> the biggest detected face as a **normalized** (x, y, w, h) box, or None.

    Biggest, because the conversation partner is the nearest person to the
    camera; a face in the background is not the one being signed by.

    The normalization is the part that matters -- see this module's docstring.
    The guard below is worth its three lines: a face box narrower than one
    pixel is impossible, which makes it a sound discriminator rather than a
    guess, and it means a future MediaPipe that switches conventions degrades
    instead of silently poisoning every location feature.
    """
    if not result.detections:
        return None
    best = max(result.detections,
               key=lambda d: d.bounding_box.width * d.bounding_box.height)
    box = best.bounding_box
    if box.width <= 1.0 and box.height <= 1.0:
        return (box.origin_x, box.origin_y, box.width, box.height)
    return (box.origin_x / frame_w, box.origin_y / frame_h,
            box.width / frame_w, box.height / frame_h)


def scene(detected, face_box, frame_w, frame_h, signer_dominant=hands.RIGHT,
          mirrored_input=True):
    """Normalized detections -> (dominant, nondominant, face) in pixel space.

    Goes through hands.canonical_scene(), never assign_hands() directly: that
    is what keeps the hands and the face reflected together for a left-dominant
    signer. Mirroring one without the other leaves every location feature wrong
    by twice the head's offset from the axis, with handshape and orientation
    left correct so the damage hides in the hardest block to eyeball.
    """
    import location

    pixel_hands = [([(x * frame_w, y * frame_h) for x, y in points], label)
                   for points, label in detected]
    face = None if face_box is None else location.face_from_normalized_box(
        face_box[0], face_box[1], face_box[2], face_box[3], frame_w, frame_h)
    return hands.canonical_scene(pixel_hands, face,
                                 signer_dominant=signer_dominant,
                                 mirrored_input=mirrored_input)
