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
    label, p0_x, p0_y, p1_x, p1_y, ..., p20_x, p20_y   (1 + 42 columns)

Controls:
  [ / ]   = previous / next letter
  SPACE   = start / stop recording for the current letter
  Q       = quit and save
"""
import cv2
import csv
import os
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTERS = [c for c in ALPHABET if c not in ("J", "Z")]  # motion signs, skipped

LANDMARKER_PATH = "hand_landmarker.task"
OUT_PATH = "landmark_data.csv"
CAPTURE_EVERY_N_FRAMES = 2  # avoid logging near-duplicate consecutive frames


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


def main():
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

    file_exists = os.path.exists(OUT_PATH)
    csv_file = open(OUT_PATH, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        header = ["label"] + [f"p{i}_{axis}" for i in range(21) for axis in ("x", "y")]
        writer.writerow(header)

    letter_idx = 0
    recording = False
    frame_count = 0
    saved_counts = {L: 0 for L in LETTERS}

    print("Data collection ready.")
    print("[ / ] = prev/next letter | SPACE = start/stop recording | Q = quit")
    print("Tip: vary background, lighting, and hand distance/angle between takes.")

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

        current_letter = LETTERS[letter_idx]
        status_color = (0, 255, 0) if recording else (0, 0, 255)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            for lm in landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 2, (255, 255, 0), -1)

            if recording:
                frame_count += 1
                if frame_count % CAPTURE_EVERY_N_FRAMES == 0:
                    row = [current_letter] + normalize_landmarks(landmarks, w, h)
                    writer.writerow(row)
                    saved_counts[current_letter] += 1
        else:
            cv2.putText(frame, "No hand detected", (10, h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.putText(frame, f"Letter: {current_letter}  (saved: {saved_counts[current_letter]})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        cv2.putText(frame, "REC" if recording else "PAUSED",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, "[ / ] letter   SPACE rec/pause   Q quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Landmark Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('['):
            letter_idx = (letter_idx - 1) % len(LETTERS)
            recording = False
        elif key == ord(']'):
            letter_idx = (letter_idx + 1) % len(LETTERS)
            recording = False
        elif key == ord(' '):
            recording = not recording

    csv_file.close()
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved to {OUT_PATH}")
    print("Per-letter counts:", saved_counts)


if __name__ == "__main__":
    main()
