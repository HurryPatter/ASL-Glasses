"""
Data collection tool for the landmark-based ML classifier.

Captures MediaPipe hand landmarks and normalizes them into a "hand frame"
(origin at wrist, unit length = wrist-to-middle-knuckle distance, thumb
always on +x) before logging them. This is the same normalization the
rule-based fallback branch uses -- it makes the features invariant to
hand distance-from-camera, left/right hand, and arm rotation, so the
model doesn't have to learn that from data the way a pixel-based CNN did.

J and Z are motion signs (handled separately, see motion.py on the trial
branch) and are skipped here -- this collects the 24 static letters only.

Output: landmark_data.csv, one row per captured frame:
    label, person, p0_x, p0_y, ..., p20_x, p20_y   (2 + 42 columns)

The `person` column is asked for at startup and matters more than the row
count: a random train/test split over frames measures memorisation, because
frames within one recording are near-duplicates. Accuracy is reported by
holding out a whole person, so every row needs to say who signed it.

Collect from as many different people as you can. Measured: a second person was
worth about +10 points on an unseen signer and a fourth +4.7, while tripling the
rows from the same people was worth nothing.

When recording, tilt the hand and vary the handshape. Do NOT bother moving it
around the frame or changing distance -- the normalization below removes both
exactly, so those frames are duplicates however different they look on screen.

Controls:
  [ / ]   = previous / next letter
  SPACE   = start / stop recording for the current letter
  Q       = quit and save
"""
import cv2
import csv
import os
import math
import sys
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import dataset

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTERS = [c for c in ALPHABET if c not in ("J", "Z")]  # motion signs, skipped

# Static word signs (not fingerspelled) go here. Add new ones the same way --
# any static, single-handed handshape can be collected identically to a
# letter. Multi-character labels are what tell main.py "this is a whole
# word, not a fingerspelled character" -- no other code needs to know about
# a specific new word.
WORDS = ["ILY", "IHATEYOU", "HELLO"]

LABELS = LETTERS + WORDS

LANDMARKER_PATH = "hand_landmarker.task"
OUT_PATH = "landmark_data.csv"
CAPTURE_EVERY_N_FRAMES = 2  # avoid logging near-duplicate consecutive frames

# Past this many rows for one label, a person is mostly contributing duplicate
# frames. Measured: accuracy on an unseen signer plateaued at roughly 4,000
# training rows total, so ~100 per label per person is where effort is better
# spent on another person than on a longer take.
#
# Rows are not the whole story though. What makes a row useful is variation the
# features can actually represent, and normalize_landmarks() removes position,
# distance-from-camera and in-plane rotation exactly (verified: those change the
# 42 floats by ~1e-16). Only out-of-plane tilt and genuine handshape change
# survive. So a take should tilt the hand and vary finger curl; waving it around
# the frame contributes literally nothing.
SUGGESTED_ROWS_PER_LABEL = 100


def normalize_landmarks(landmarks, frame_w, frame_h):
    """Convert 21 MediaPipe landmarks into the scale/rotation-invariant hand frame.

    Returns a flat list of 42 floats: [p0_x, p0_y, p1_x, p1_y, ..., p20_x, p20_y]
    """
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=float)
    pts[:, 0] *= frame_w
    pts[:, 1] *= frame_h

    v = pts[9] - pts[0]  # wrist -> middle knuckle
    size = np.linalg.norm(v) or 1e-6
    y_hat = v / size
    x_hat = np.array([y_hat[1], -y_hat[0]])
    local = ((pts - pts[0]) @ np.stack([x_hat, y_hat], axis=1)) / size

    if local[2][0] < 0:  # thumb knuckle landed on -x: mirror so thumb is on +x
        local[:, 0] *= -1

    return local.flatten().tolist()


def ask_person():
    """Who is signing. Recorded per row so accuracy can be measured by holding
    out a whole person rather than a random sample of near-duplicate frames."""
    while True:
        name = input("Who is signing? (first name, e.g. omar): ").strip().lower()
        if name and all(c.isalnum() or c in "-_" for c in name):
            return name
        print("  Please enter a single name (letters/digits, no spaces).")


def open_output(person):
    """Append to landmark_data.csv, refusing to mix schemas."""
    if os.path.exists(OUT_PATH):
        header = dataset.read_header(OUT_PATH)
        if not dataset.has_person_column(header):
            sys.exit(
                f"{OUT_PATH} predates the `person` column. Run "
                f"`python backfill_person.py` first, then collect."
            )
        if header != dataset.HEADER:
            sys.exit(f"{OUT_PATH} has an unexpected header; not appending to it.")
        csv_file = open(OUT_PATH, "a", newline="")
        return csv_file, csv.writer(csv_file)

    csv_file = open(OUT_PATH, "a", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(dataset.HEADER)
    return csv_file, writer


def main():
    person = ask_person()
    base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    frame_timestamp_ms = 0

    csv_file, writer = open_output(person)

    letter_idx = 0
    recording = False
    frame_count = 0
    saved_counts = {L: 0 for L in LABELS}

    print(f"Data collection ready -- signing as '{person}'.")
    print("[ / ] = prev/next letter | SPACE = start/stop recording | Q = quit")
    print(f"Aim for ~{SUGGESTED_ROWS_PER_LABEL} rows per label, then stop.")
    print("While recording, vary what the FEATURES can see:")
    print("  - TILT the hand toward and away from the camera")
    print("  - vary the handshape slightly: finger curl, thumb position")
    print("Moving it around the frame, closer/further, or rotating it in the")
    print("image plane does NOTHING -- the hand frame normalizes all three away.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        current_letter = LABELS[letter_idx]
        status_color = (0, 255, 0) if recording else (0, 0, 255)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            for lm in landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 2, (255, 255, 0), -1)

            if recording:
                frame_count += 1
                if frame_count % CAPTURE_EVERY_N_FRAMES == 0:
                    row = [current_letter, person] + normalize_landmarks(landmarks, w, h)
                    writer.writerow(row)
                    saved_counts[current_letter] += 1
        else:
            cv2.putText(frame, "No hand detected", (10, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        saved = saved_counts[current_letter]
        enough = " ENOUGH" if saved >= SUGGESTED_ROWS_PER_LABEL else ""
        cv2.putText(frame, f"{current_letter}  ({saved}/{SUGGESTED_ROWS_PER_LABEL}{enough})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255) if enough else status_color, 2)
        cv2.putText(frame, f"signer: {person}", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "REC" if recording else "PAUSED",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, "[ / ] letter   SPACE rec/pause   Q quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Landmark Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('['):
            letter_idx = (letter_idx - 1) % len(LABELS)
            recording = False
        elif key == ord(']'):
            letter_idx = (letter_idx + 1) % len(LABELS)
            recording = False
        elif key == ord(' '):
            recording = not recording

    csv_file.close()
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved to {OUT_PATH} as '{person}'")
    recorded = {k: v for k, v in saved_counts.items() if v}
    print("Per-label counts:", recorded)
    missing = [l for l in LABELS if not saved_counts[l]]
    if missing:
        print(f"Not recorded this session: {', '.join(missing)}")


if __name__ == "__main__":
    main()