import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from debouncer import Debouncer
from motion import MotionDetector
from nlp_bridge import NLPBridge
from audio import AudioOutput

# ── Constants ──────────────────────────────────────────────────────────────
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VALID_IDX = [i for i in range(26) if i not in [9, 25]]  # no J or Z
MODEL_PATH = "asl_model.h5"
LANDMARKER_PATH = "hand_landmarker.task"
CONFIDENCE_THRESHOLD = 0.85
BOX_PADDING = 20  # pixels added around the landmark bounding box
STILL_THRESHOLD = 0.15  # hand widths moved over 5 frames; above this, static letters are ignored
MOTION_FLASH_FRAMES = 15  # how long a recognised J/Z stays on the HUD

# ── Load components ────────────────────────────────────────────────────────
print("Loading CNN model...")
model = load_model(MODEL_PATH)

print("Loading MediaPipe hand landmarker...")
base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH)
landmarker_options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
landmarker = mp_vision.HandLandmarker.create_from_options(landmarker_options)

print("Initialising pipeline...")
debouncer = Debouncer(min_frames=3)
motion = MotionDetector(window=20)
nlp = NLPBridge(mode="academic")
audio = AudioOutput()


# ── Hand region from MediaPipe landmarks ────────────────────────────────────
def get_hand_bbox(landmarks, frame_w, frame_h, pad=BOX_PADDING):
    """Convert normalized MediaPipe landmarks into a padded, SQUARE pixel bbox.

    Forcing a square box (rather than the raw landmark extent) means the
    later cv2.resize(28, 28) scales evenly instead of squashing the hand
    into a different aspect ratio than the training images.
    """
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]

    x1, x2 = min(xs) - pad, max(xs) + pad
    y1, y2 = min(ys) - pad, max(ys) + pad

    box_w, box_h = x2 - x1, y2 - y1
    side = max(box_w, box_h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1, x2 = cx - side / 2, cx + side / 2
    y1, y2 = cy - side / 2, cy + side / 2

    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(frame_w, x2))
    y2 = int(min(frame_h, y2))
    return x1, y1, x2, y2


# ── Camera loop ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0
frame_no = 0
last_motion = ("", -MOTION_FLASH_FRAMES)  # (letter, frame it fired)

print("Camera ready.")
print("Show your hand to the camera.")
print("Controls: Q = quit | R = reset | S = speak")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # ── MediaPipe hand detection ─────────────────────────────────────────
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_timestamp_ms += 33  # approx one frame at ~30fps
    frame_no += 1
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    predicted_letter = ""
    confidence = 0.0
    motion_letter, motion_start = "", None
    hand_speed = 0.0

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        motion_letter, motion_start = motion.update(landmarks, frame_no)
        hand_speed = motion.speed(5)
        x1, y1, x2, y2 = get_hand_bbox(landmarks, w, h)
        hand_crop = frame[y1:y2, x1:x2]

        if hand_crop.size > 0:
            # ── Preprocess to match CNN training format ────────────────────
            gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28))
            normalized = resized / 255.0
            input_arr = normalized.reshape(1, 28, 28, 1)

            # ── CNN prediction ─────────────────────────────────────────────
            predictions = model.predict(input_arr, verbose=0)
            pred_idx = np.argmax(predictions)
            confidence = predictions[0][pred_idx]

            if pred_idx in VALID_IDX and confidence > CONFIDENCE_THRESHOLD:
                predicted_letter = ALPHABET[pred_idx]
            if hand_speed > STILL_THRESHOLD:
                predicted_letter = ""  # mid-motion: don't commit static guesses

        # ── Draw bounding box, landmarks and prediction ─────────────────────
        color = (0, 255, 0) if predicted_letter else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        for lm in landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (px, py), 2, (255, 255, 0), -1)
        if predicted_letter:
            cv2.putText(frame, f"{predicted_letter} ({confidence:.0%})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, color, 2)
        elif hand_speed > STILL_THRESHOLD:
            cv2.putText(frame, "moving...", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        motion.clear()  # hand left the frame; stale trajectory is meaningless
        cv2.putText(frame, "No hand detected", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # ── Debounce → NLP ─────────────────────────────────────────────────────
    raw_string = debouncer.update(predicted_letter, frame_no)
    if motion_letter:
        raw_string = debouncer.commit_motion(motion_letter, motion_start)
        last_motion = (motion_letter, frame_no)
        print(f"Motion letter: {motion_letter}")
    corrected = nlp.correct(raw_string)

    # ── HUD ────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"Raw: {raw_string[-30:]}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Corrected: {corrected}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    if frame_no - last_motion[1] < MOTION_FLASH_FRAMES:
        cv2.putText(frame, f"Motion: {last_motion[0]}", (w - 220, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, "Q=quit R=reset S=speak",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (150, 150, 150), 1)

    cv2.imshow("ASL Translator", frame)

    # ── Keyboard controls ──────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        debouncer.reset()
        print("Reset.")
    elif key == ord('s'):
        audio.speak(corrected)
        print(f"Spoken: {corrected}")

# ── Cleanup ────────────────────────────────────────────────────────────────
landmarker.close()
cap.release()
cv2.destroyAllWindows()