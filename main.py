import os
import sys
os.environ.setdefault("KERAS_BACKEND", "jax")  # MediaPipe ships JAX; avoids the TensorFlow/protobuf clash on macOS
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
BOX_PADDING = 0  # pixels around the landmark box; training images are almost all hand, so no padding
TRAIN_MEAN, TRAIN_STD = 159.0, 41.0  # Sign Language MNIST pixel statistics; live crops are matched to them
MASK_DILATE_PX = 15   # grow the landmark hull by this much (at a 270px crop; scaled with crop size)
FLAT_BACKGROUND = 255  # everything outside the hand becomes pure white; measured 44/47 on a live B vs 42/47 at 225
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
CAMERA_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # e.g. `main.py 1` when a phone sits at index 0
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    sys.exit(f"Could not open camera index {CAMERA_INDEX}. Try another index: python main.py 1")
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
    cnn_preview = None

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        motion_letter, motion_start = motion.update(landmarks, frame_no)
        hand_speed = motion.speed(5)
        x1, y1, x2, y2 = get_hand_bbox(landmarks, w, h)
        hand_crop = frame[y1:y2, x1:x2]

        if hand_crop.size > 0:
            # ── Preprocess to match CNN training format ────────────────────
            # 1. Isolate the hand: fill the convex hull of the landmarks, grow it
            #    slightly, and paint everything outside it a flat light gray.
            #    Measured on a live B: 14/47 frames correct without this, 46/47 with it.
            gray = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
            pts = np.array([[lm.x * w - x1, lm.y * h - y1] for lm in landmarks], np.int32)
            mask = np.zeros(gray.shape, np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
            k = max(3, int(MASK_DILATE_PX * gray.shape[0] / 270)) | 1
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            # 2. Match the training brightness statistics using hand pixels only,
            #    so the background never influences the hand's contrast.
            g = gray.astype(np.float32)
            hand_px = g[mask > 0]
            if hand_px.size < 50:
                hand_px = g
            g = (g - hand_px.mean()) / (hand_px.std() + 1e-6) * TRAIN_STD + TRAIN_MEAN
            g = np.where(mask > 0, g, FLAT_BACKGROUND)
            gray = np.clip(g, 0, 255).astype(np.uint8)
            # 3. The display frame is mirrored; the training photos were not.
            gray = cv2.flip(gray, 1)
            resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
            cnn_preview = cv2.cvtColor(cv2.resize(resized, (112, 112), interpolation=cv2.INTER_NEAREST),
                                       cv2.COLOR_GRAY2BGR)
            # Debug window: crop -> mask -> exact model input, each 280px, pixel-for-pixel
            P = 280
            panel_crop = cv2.resize(hand_crop, (P, P))
            panel_mask = cv2.cvtColor(cv2.resize(mask, (P, P), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            panel_in = cv2.cvtColor(cv2.resize(resized, (P, P), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            for y in range(0, P, P // 28):  # grid so each of the 28x28 cells is visible
                cv2.line(panel_in, (0, y), (P, y), (210, 210, 210), 1)
                cv2.line(panel_in, (y, 0), (y, P), (210, 210, 210), 1)
            debug_view = np.hstack([panel_crop, panel_mask, panel_in])
            for i, label in enumerate(("1. crop", "2. hand mask", "3. model input 28x28")):
                cv2.putText(debug_view, label, (i * P + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
            normalized = resized / 255.0
            input_arr = normalized.reshape(1, 28, 28, 1)

            # ── CNN prediction ─────────────────────────────────────────────
            predictions = model.predict(input_arr, verbose=0)
            pred_idx = np.argmax(predictions)
            confidence = predictions[0][pred_idx]

            if pred_idx in VALID_IDX and confidence > CONFIDENCE_THRESHOLD:
                predicted_letter = ALPHABET[pred_idx]
            top3 = np.argsort(predictions[0])[::-1][:3]
            guess_text = "   ".join(f"{ALPHABET[i]} {predictions[0][i]:.0%}" for i in top3)
            cv2.putText(debug_view, guess_text, (2 * P + 6, P - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("What the model sees", debug_view)
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
    if cnn_preview is not None:  # what the CNN actually sees, top-right corner
        frame[10:122, w - 122:w - 10] = cnn_preview
        cv2.rectangle(frame, (w - 122, 10), (w - 10, 122), (255, 255, 255), 1)
    if frame_no - last_motion[1] < MOTION_FLASH_FRAMES:
        cv2.putText(frame, f"Motion: {last_motion[0]}", (w - 340, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(frame, "Q=quit R=reset S=speak",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (150, 150, 150), 1)

    if frame_no % 30 == 0:  # once a second: what the pipeline sees
        if result.hand_landmarks:
            top3 = np.argsort(predictions[0])[::-1][:3] if hand_crop.size > 0 else []
            guesses = " ".join(f"{ALPHABET[i]}:{predictions[0][i]:.2f}" for i in top3)
            print(f"[{frame_no}] hand=yes speed={hand_speed:.2f} top3=[{guesses}] raw='{raw_string}'")
        else:
            print(f"[{frame_no}] hand=no raw='{raw_string}'")
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