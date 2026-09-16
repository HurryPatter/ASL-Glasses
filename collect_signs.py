"""
Clip collection for Project Veronica — two hands, movement, real ASL signs.

The letter collector (`collect_data.py`) records one row per *frame* of a held
pose, because a fingerspelled letter is a held pose. A sign is a path, so this
records one clip per *sign attempt*: press SPACE, sign, press SPACE again.

It writes two files (see signset.py for why):

    veronica_clips.jsonl   raw landmarks — the archive, never regenerable
    veronica_signs.csv     371-float training rows — derived, regenerable with
                           `python rebuild_signs.py`

Both are appended to, never overwritten, so data from several people
accumulates exactly as `landmark_data.csv` does today.

What to vary, and what not to
-----------------------------
The letter collector's guidance was counter-intuitive because the hand frame
erased most of what looked like variation. Here it is different, and more
ordinary: the feature vector now carries orientation, location relative to the
face, and the whole trajectory, so **most real variation now reaches the
features**. Sign at a natural speed and amplitude, and let the natural
differences between takes stand.

The one piece of the old advice that still holds: **position in the frame and
distance from the camera are still normalized away**, now against the face
rather than the hand. Shuffling around in your chair contributes nothing.

Still true, and now the binding constraint again: **collect from more people,
not more clips per person.** Measured on the letter dataset, a fourth signer
was worth +4.7 points on an unseen signer while tripling the rows from existing
signers was worth nothing.

Controls:
  [ / ]   = previous / next sign          , / . = jump a category
  SPACE   = start / stop recording a clip
  U       = undo the last clip
  F       = toggle the feature/diagnostic readout
  Q       = quit
"""
import argparse
import json
import os
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import hands
import location
import sequence
import signset

HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "blaze_face_short_range.tflite"

# Clips per sign per person. The letter dataset plateaued at roughly 4,000
# training rows total while each new *person* kept paying, so the budget is
# better spent on the next signer than on a longer session with this one.
# Clips are far more independent than frames were -- each is a separate
# attempt -- so this number is much smaller than the old 100 rows/label.
SUGGESTED_CLIPS = 20

FACE_HELP = f"""
{FACE_MODEL} not found.

Stage 2 needs a face to anchor sign location against: FATHER and MOTHER are
the same handshape, orientation and movement, differing only in forehead
versus chin. Without it every location feature is the zeroed no-face block,
and signs that differ only in where they are made become the same class.

Download MediaPipe's short-range face detector into this directory:

    https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite

Collecting a whole session without it by accident would be expensive, which
is why this stops rather than warns. If you really mean to, pass --no-face.
"""


def ask_person():
    while True:
        name = input("Who is signing? (first name, e.g. omar): ").strip().lower()
        if name and all(c.isalnum() or c in "-_" for c in name):
            return name
        print("  Please enter a single name (letters/digits, no spaces).")


def ask_dominant():
    """Which hand leads.

    Not cosmetic: a left-dominant signer's scene is mirrored into
    right-dominant space, because ASL is handedness-symmetric and without the
    mirror every sign they make is an unseen class. Getting this wrong is
    undetectable downstream -- the features stay perfectly plausible -- so it
    is asked rather than guessed.
    """
    while True:
        answer = input("Dominant (signing) hand? [R/l]: ").strip().lower() or "r"
        if answer in ("r", "right"):
            return hands.RIGHT
        if answer in ("l", "left"):
            return hands.LEFT
        print("  Please answer R or L.")


def build_landmarker():
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,                       # the whole point of stage 1
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def build_face_detector():
    options = mp_vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )
    return mp_vision.FaceDetector.create_from_options(options)


def detected_hands(result):
    """MediaPipe's result -> [(normalized points, handedness label)].

    Handedness is taken as reported. That is correct while the frame is
    mirrored -- cv2.flip below -- and wrong if it ever stops being; see
    hands.assign_hands() and VERONICA.md, since nothing downstream can detect
    the difference.
    """
    out = []
    for index, landmarks in enumerate(result.hand_landmarks):
        label = None
        if index < len(result.handedness) and result.handedness[index]:
            label = result.handedness[index][0].category_name
        out.append(([(lm.x, lm.y) for lm in landmarks], label))
    return out


def largest_face(result, frame_w, frame_h):
    """The biggest detected face, as a **normalized** (x, y, w, h) box.

    Biggest, because the conversation partner is the nearest person to the
    camera; a face in the background is not the one being signed by.

    The normalization is the part that matters. MediaPipe Tasks reports a
    detection's bounding box in *pixels*, while it reports hand landmarks
    normalized to [0, 1] -- two different conventions out of one library. The
    archive stores everything normalized, so the conversion happens here, once,
    rather than at each place a face is read back.

    Getting this wrong would not crash: a pixel box read as normalized puts the
    face somewhere off past the corner of the frame, and every location feature
    would be a large, stable, plausible-looking number. So the guard below is
    worth its three lines -- a face box narrower than one pixel is impossible,
    which makes it a sound discriminator rather than a guess.
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


def draw(frame, detected, face_box, state):
    h, w = frame.shape[:2]
    for points, label in detected:
        colour = (0, 255, 255) if label == hands.RIGHT else (255, 200, 0)
        for x, y in points:
            cv2.circle(frame, (int(x * w), int(y * h)), 2, colour, -1)

    if face_box is not None:
        fx, fy, fw, fh = face_box
        cv2.rectangle(frame, (int(fx * w), int(fy * h)),
                      (int((fx + fw) * w), int((fy + fh) * h)), (120, 220, 120), 1)
    else:
        cv2.putText(frame, "NO FACE - location features will be empty",
                    (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    recording = state["recording"]
    colour = (0, 255, 0) if recording else (0, 0, 255)
    saved = state["counts"].get(state["label"], 0)
    enough = "  ENOUGH" if saved >= SUGGESTED_CLIPS else ""

    cv2.putText(frame, f"{state['label']}  ({saved}/{SUGGESTED_CLIPS}{enough})",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 255) if enough else colour, 2)
    cv2.putText(frame, f"{state['category']}   {state['index'] + 1}/{state['total']}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(frame, "RECORDING" if recording else "paused",
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.putText(frame, f"signer: {state['person']} ({state['dominant'].lower()}-dominant)"
                       f"   hands: {len(detected)}   clips: {state['total_clips']}",
                (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    if recording:
        cv2.putText(frame, f"{state['frames']} frames / {state['span_ms']}ms",
                    (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    if state["message"]:
        cv2.putText(frame, state["message"], (10, h - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state["message_colour"], 2)

    cv2.putText(frame, "[ ] sign   , . category   SPACE rec   U undo   F info   Q quit",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


def draw_readout(frame, detected, face_box, dominant_hand, frame_w, frame_h):
    """Live feature values, so the layers can be checked against a real hand.

    Same reasoning as MotionDetector.debug_info(): the mirror convention and
    the location bands are the parts of this pipeline that can only be
    confirmed in front of a camera, and an assumption nobody can see is an
    assumption nobody checks. In particular the reported handedness here is
    the thing to verify before a real session -- raise your right hand and
    confirm it says so.
    """
    pixel_hands = [([(x * frame_w, y * frame_h) for x, y in points], label)
                   for points, label in detected]
    face = None if face_box is None else location.face_from_normalized_box(
        face_box[0], face_box[1], face_box[2], face_box[3], frame_w, frame_h)
    dom, non, face = hands.canonical_scene(pixel_hands, face,
                                           signer_dominant=dominant_hand)

    lines = [f"raw handedness: {[label for _, label in detected] or '-'}",
             f"dominant hand tracked: {'yes' if dom is not None else 'no'}",
             f"non-dominant tracked:  {'yes' if non is not None else 'no'}"]
    for key, value in location.debug_info(hands.anchor(dom), hands.anchor(non),
                                          face).items():
        lines.append(f"{key}: {value}")

    y = 210
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 200, 0), 1)
        y += 18


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-face", action="store_true",
                        help="collect without location features (rarely what you want)")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(HAND_MODEL):
        sys.exit(f"{HAND_MODEL} not found.")
    use_face = not args.no_face
    if use_face and not os.path.exists(FACE_MODEL):
        sys.exit(FACE_HELP)

    person = ask_person()
    dominant_hand = ask_dominant()

    landmarker = build_landmarker()
    face_detector = build_face_detector() if use_face else None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}.")

    labels = signset.SIGNS
    category_starts = []
    seen = 0
    for name, group in signset.VOCABULARY.items():
        category_starts.append((seen, name))
        seen += len(group)

    def category_of(index):
        current = category_starts[0][1]
        for start, name in category_starts:
            if index >= start:
                current = name
        return current

    state = {
        "label": labels[0], "index": 0, "total": len(labels),
        "category": category_of(0), "person": person, "dominant": dominant_hand,
        "recording": False, "counts": {}, "total_clips": 0,
        "frames": 0, "span_ms": 0, "message": "", "message_colour": (200, 200, 200),
    }

    def say(text, colour=(200, 200, 200)):
        state["message"] = text
        state["message_colour"] = colour
        print(text)

    session = int(time.time())
    clip_frames = []
    clip_start_ms = 0
    undo_stack = []          # (clips_offset, csv_offset, label) for U
    show_readout = False
    start = time.monotonic()

    clips_file = open(signset.CLIPS_PATH, "a")
    csv_is_new = not os.path.exists(signset.SIGNS_PATH) or \
        os.path.getsize(signset.SIGNS_PATH) == 0
    csv_file = open(signset.SIGNS_PATH, "a", newline="")
    if not csv_is_new:
        existing = signset.read_header(signset.SIGNS_PATH)
        if existing != signset.HEADER:
            sys.exit(f"{signset.SIGNS_PATH} has a different schema; not appending. "
                     f"Re-derive it with `python rebuild_signs.py`.")
    else:
        csv_file.write(",".join(signset.HEADER) + "\n")
    csv_file.flush()

    print(f"\nReady -- {person}, {dominant_hand.lower()}-dominant.")
    print(f"{len(labels)} signs. Aim for ~{SUGGESTED_CLIPS} clips each, then "
          f"recruit the next person.")
    print("Press F and check the reported handedness matches your real hand "
          "before you start.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)     # selfie view; see detected_hands()
            frame_h, frame_w = frame.shape[:2]
            timestamp_ms = int((time.monotonic() - start) * 1000)

            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hand_result = landmarker.detect_for_video(image, timestamp_ms)
            detected = detected_hands(hand_result)

            face_box = None
            if face_detector is not None:
                face_box = largest_face(
                    face_detector.detect_for_video(image, timestamp_ms),
                    frame_w, frame_h)

            if state["recording"]:
                clip_frames.append(signset.raw_frame(
                    detected, face_box, timestamp_ms - clip_start_ms))
                state["frames"] = len(clip_frames)
                state["span_ms"] = clip_frames[-1]["t"]

            draw(frame, detected, face_box, state)
            if show_readout:
                draw_readout(frame, detected, face_box, dominant_hand,
                             frame_w, frame_h)
            cv2.imshow("Veronica -- sign collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            elif key == ord(' '):
                if not state["recording"]:
                    clip_frames = []
                    clip_start_ms = timestamp_ms
                    state["recording"] = True
                    state["frames"] = state["span_ms"] = 0
                    say(f"recording {state['label']}...", (0, 220, 0))
                else:
                    state["recording"] = False
                    clip = signset.raw_clip(
                        f"{person}-{session}-{state['total_clips'] + 1}",
                        state["label"], person, dominant_hand,
                        frame_w, frame_h, clip_frames)
                    saved, note, colour = save_clip(
                        clip, clips_file, csv_file, undo_stack)
                    if saved:
                        state["counts"][state["label"]] = \
                            state["counts"].get(state["label"], 0) + 1
                        state["total_clips"] += 1
                    say(note, colour)

            elif key == ord('u'):
                say(*undo_last(undo_stack, clips_file, csv_file, state))

            elif key in (ord('['), ord(']')) and not state["recording"]:
                step = -1 if key == ord('[') else 1
                state["index"] = (state["index"] + step) % len(labels)
                state["label"] = labels[state["index"]]
                state["category"] = category_of(state["index"])
                state["message"] = ""

            elif key in (ord(','), ord('.')) and not state["recording"]:
                starts = [s for s, _ in category_starts]
                if key == ord('.'):
                    nxt = next((s for s in starts if s > state["index"]), starts[0])
                else:
                    earlier = [s for s in starts if s < state["index"]]
                    nxt = earlier[-1] if earlier else starts[-1]
                state["index"] = nxt
                state["label"] = labels[nxt]
                state["category"] = category_of(nxt)
                state["message"] = ""

            elif key == ord('f'):
                show_readout = not show_readout

    finally:
        clips_file.close()
        csv_file.close()
        landmarker.close()
        if face_detector is not None:
            face_detector.close()
        cap.release()
        cv2.destroyAllWindows()

    report(state, person)


def save_clip(clip, clips_file, csv_file, undo_stack):
    """Append one clip to both files, rejecting takes that cannot be a sign.

    The floors are SignBuffer's, so a clip that is accepted here is one the
    live pipeline could also have classified -- a clip too short or too sparse
    to be a sign at collection time would be an unlearnable training row.
    Rejecting it now costs a retake; keeping it costs accuracy that is hard to
    trace back.
    """
    samples = signset.clip_to_samples(clip)
    buffer = sequence.SignBuffer()
    for sample in samples:
        buffer.add(sample.t, sample.features, sample.dom, sample.non)
    if not buffer.ready():
        return False, (f"too short -- {len(samples)} frames / "
                       f"{signset.clip_span_ms(clip)}ms, not saved"), (0, 0, 255)

    coverage = signset.face_coverage(clip)
    row = signset.clip_to_row(clip)

    # Offsets before writing, so U can truncate both files back exactly.
    undo_stack.append((clips_file.tell(), csv_file.tell(), clip["label"]))

    # Written here rather than via signset.append_clip() because undo needs
    # this handle's offsets -- a separately opened handle has none to record.
    clips_file.write(json.dumps(clip, separators=(",", ":")) + "\n")
    clips_file.flush()
    csv_file.write(",".join(str(v) for v in row) + "\n")
    csv_file.flush()

    if coverage < signset.MIN_FACE_COVERAGE:
        return True, (f"saved, but face seen in only {coverage:.0%} of frames "
                      f"-- consider a retake"), (0, 165, 255)
    return True, f"saved {clip['label']}  ({len(samples)} frames)", (0, 220, 0)


def undo_last(undo_stack, clips_file, csv_file, state):
    """Drop the most recent clip from both files.

    Worth the bookkeeping: a fluffed take is common, and the alternative is
    either keeping known-bad data or editing two files by hand afterwards.
    Truncating to a recorded offset is exact, where deleting a last line by
    rewriting is not.
    """
    if not undo_stack:
        return "nothing to undo", (0, 165, 255)
    clips_offset, csv_offset, label = undo_stack.pop()
    for handle, offset in ((clips_file, clips_offset), (csv_file, csv_offset)):
        handle.flush()
        handle.truncate(offset)
        handle.seek(offset)
    state["counts"][label] = max(0, state["counts"].get(label, 0) - 1)
    state["total_clips"] = max(0, state["total_clips"] - 1)
    return f"undid {label}", (0, 165, 255)


def report(state, person):
    counts = {k: v for k, v in state["counts"].items() if v}
    print(f"\nSaved {state['total_clips']} clips as '{person}'")
    print(f"  archive: {signset.CLIPS_PATH}")
    print(f"  rows:    {signset.SIGNS_PATH}")
    if counts:
        print("\nPer-sign clips this session:")
        for label in sorted(counts):
            print(f"  {label:20} {counts[label]}")
    thin = [s for s in signset.SIGNS if counts.get(s, 0) < SUGGESTED_CLIPS]
    if thin:
        print(f"\nBelow {SUGGESTED_CLIPS} clips this session ({len(thin)}): "
              f"{', '.join(thin[:12])}{' ...' if len(thin) > 12 else ''}")
    print("\nRetrain from the archive with: python rebuild_signs.py")


if __name__ == "__main__":
    main()
