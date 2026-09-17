"""
Live Veronica pipeline — two hands, location, movement, continuous signing,
gloss → English.

    python demo_veronica.py                  # right-dominant signer
    python demo_veronica.py --dominant left
    python demo_veronica.py --no-face        # without location features

**This runs before you have a trained model**, and that is deliberate. Without
one, recognition is idle but every layer beneath it is live: press `D` and you
can watch the feature vector, the buffer filling, the location zone each hand
is in and the reported handedness, against a real hand in real light. Those are
the numbers that can only be judged in front of a camera, and judging them
before a collection session is much cheaper than after one.

With `veronica_model.joblib` present it runs the whole chain: frame → features
→ trailing window → classifier → continuous segmentation → gloss → English.

Controls:
  Q quit   R reset the sentence   S speak it   D toggle the readout
"""
import argparse
import json
import os
import sys
import time

import cv2

import capture
import config
import gloss
import hands
import location
import segment
import sequence

MODEL_PATH = "veronica_model.joblib"
LABELS_PATH = "veronica_labels.json"


def load_classifier():
    """-> (classify, description). Falls back to a null classifier.

    Returning a working-but-idle classifier rather than refusing to start is
    what makes this usable during stage 4: the layers below recognition are
    exactly what needs checking before anyone signs into a camera, and they do
    not need a model to be checked.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        return (lambda vector: ("", 0.0)), "no model -- recognition idle"

    import joblib
    import numpy as np

    model = joblib.load(MODEL_PATH)
    with open(LABELS_PATH) as fh:
        labels = json.load(fh)

    def classify(vector):
        probabilities = model.predict_proba([vector])[0]
        best = int(np.argmax(probabilities))
        return labels[best], float(probabilities[best])

    return classify, f"{MODEL_PATH} ({len(labels)} signs)"


def draw_hud(frame, segmenter, english, model_note, tracked, face_box):
    height, width = frame.shape[:2]

    glosses = segmenter.glosses
    line = " ".join(glosses[-8:])
    cv2.putText(frame, f"Gloss: {line}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(frame, english, (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

    cv2.putText(frame, f"hands: {tracked}   face: {'yes' if face_box else 'NO'}"
                       f"   {model_note}",
                (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (150, 150, 150), 1)
    cv2.putText(frame, "Q quit   R reset   S speak   D readout",
                (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (150, 150, 150), 1)


def draw_landmarks(frame, detected, face_box):
    height, width = frame.shape[:2]
    for points, label in detected:
        colour = (0, 255, 255) if label == hands.RIGHT else (255, 200, 0)
        for x, y in points:
            cv2.circle(frame, (int(x * width), int(y * height)), 2, colour, -1)
    if face_box is not None:
        fx, fy, fw, fh = face_box
        cv2.rectangle(frame, (int(fx * width), int(fy * height)),
                      (int((fx + fw) * width), int((fy + fh) * height)),
                      (120, 220, 120), 1)


def draw_readout(frame, segmenter, dom, non, face, detected, fps,
                 mirrored_input=True, dominant_hand=None):
    lines = [f"fps: {fps:.1f}" + ("" if fps >= 11 else "   BELOW THE 11fps FLOOR"),
             f"raw handedness: {[l for _, l in detected] or '-'}   "
             f"(mirrored_input={mirrored_input})",
             f"acting hand: {dominant_hand or '-'}   "
             f"dominant tracked: {'yes' if dom is not None else 'no'}"]
    for key, value in segmenter.debug_info().items():
        lines.append(f"{key}: {value}")
    for key, value in location.debug_info(hands.anchor(dom), hands.anchor(non),
                                          face).items():
        lines.append(f"{key}: {value}")

    y = 110
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 200, 0), 1)
        y += 18


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dominant", choices=["auto", "right", "left"],
                        default="auto",
                        help="the signer's dominant hand; 'auto' observes which "
                             "hand is doing the work, which is the only option "
                             "available when the signer is a stranger")
    parser.add_argument("--no-face", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--mirrored-input", choices=["true", "false"],
                        help="override the stored handedness convention")
    args = parser.parse_args()

    mirrored = config.mirrored_input()
    if args.mirrored_input:
        mirrored = args.mirrored_input == "true"

    if not os.path.exists(capture.HAND_MODEL):
        sys.exit(f"{capture.HAND_MODEL} not found -- run from the repo root.")
    use_face = not args.no_face
    if use_face and not os.path.exists(capture.FACE_MODEL):
        sys.exit(f"{capture.FACE_MODEL} not found. Either download it:\n"
                 f"  curl -LO {capture.FACE_MODEL_URL}\n"
                 f"or run with --no-face (location features will be empty, so "
                 f"signs\ndiffering only in where they are made collapse "
                 f"together).")

    fixed_dominant = None
    if args.dominant == "right":
        fixed_dominant = hands.RIGHT
    elif args.dominant == "left":
        fixed_dominant = hands.LEFT
    acting = hands.ActingHandTracker(mirrored_input=mirrored)
    classify, model_note = load_classifier()
    segmenter = segment.ContinuousSegmenter(classify,
                                            min_confidence=args.confidence)

    landmarker = capture.build_landmarker(num_hands=2)
    face_detector = capture.build_face_detector() if use_face else None

    audio = None
    try:
        from audio import AudioOutput
        audio = AudioOutput()
    except Exception:
        pass                       # TTS is Windows-only; the HUD still works

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}.")

    print(f"Veronica live -- dominant hand: {args.dominant}, {model_note}")
    print(f"Convention: {config.describe({'mirrored_input': mirrored})}")
    print("Press D for the readout. Q to quit.")

    show_readout = not os.path.exists(MODEL_PATH)   # default on with no model
    english = ""
    start = time.monotonic()
    frames = 0
    fps = 0.0
    last_fps_at = start

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)      # selfie view; see capture.py
            frame_h, frame_w = frame.shape[:2]
            now = time.monotonic()
            timestamp_ms = int((now - start) * 1000)

            frames += 1
            if now - last_fps_at >= 0.5:
                fps = frames / (now - last_fps_at)
                frames, last_fps_at = 0, now

            image = capture.to_mp_image(frame)
            detected = capture.detected_hands(
                landmarker.detect_for_video(image, timestamp_ms))
            face_box = None
            if face_detector is not None:
                face_box = capture.largest_face(
                    face_detector.detect_for_video(image, timestamp_ms),
                    frame_w, frame_h)

            acting.update(timestamp_ms, [
                ([(x * frame_w, y * frame_h) for x, y in points], label)
                for points, label in detected])
            dominant_hand = fixed_dominant or acting.signer_dominant()

            dom, non, face = capture.scene(detected, face_box, frame_w, frame_h,
                                           signer_dominant=dominant_hand,
                                           mirrored_input=mirrored)

            if dom is None and non is None:
                committed = segmenter.update(timestamp_ms)
            else:
                committed = segmenter.update(
                    timestamp_ms,
                    hands.feature_vector(dom, non, face),
                    hands.anchor(dom), hands.anchor(non))

            if committed:
                english = gloss.render(segmenter.glosses)
                print(f"  {' '.join(committed)}   ->   {english}")

            draw_landmarks(frame, detected, face_box)
            draw_hud(frame, segmenter, english, model_note,
                     len(detected), face_box)
            if show_readout:
                draw_readout(frame, segmenter, dom, non, face, detected, fps,
                             mirrored, dominant_hand)

            cv2.imshow("Veronica -- live", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                segmenter.reset()
                acting.reset()
                english = ""
                print("Reset.")
            elif key == ord('s'):
                if audio and english:
                    audio.speak(english)
                print(f"Spoken: {english}")
            elif key == ord('d'):
                show_readout = not show_readout
    finally:
        landmarker.close()
        if face_detector is not None:
            face_detector.close()
        cap.release()
        cv2.destroyAllWindows()

    if segmenter.glosses:
        print(f"\nGloss:   {' '.join(segmenter.glosses)}")
        print(f"English: {gloss.render(segmenter.glosses)}")


if __name__ == "__main__":
    main()
