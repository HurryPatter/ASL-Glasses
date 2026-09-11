import json
import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from debouncer import Debouncer
from nlp_bridge import NLPBridge
from audio import AudioOutput
from motion import MotionDetector

# ── Constants ──────────────────────────────────────────────────────────────
LANDMARKER_PATH = "hand_landmarker.task"
MODEL_PATH = "landmark_model.joblib"
LABELS_PATH = "landmark_labels.json"
CONFIDENCE_THRESHOLD = 0.85

# ── Load components ────────────────────────────────────────────────────────
print("Loading landmark classifier...")
model = joblib.load(MODEL_PATH)
with open(LABELS_PATH) as f:
    LETTERS = json.load(f)  # index -> letter, same order used at training time

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
motion_detector = MotionDetector()
nlp = NLPBridge(mode="academic")
audio = AudioOutput()


# ── Landmark normalization (must match collect_data.py exactly) ────────────
def normalize_landmarks(landmarks, frame_w, frame_h):
    """Hand frame: origin at wrist, unit = wrist->middle-knuckle, thumb on +x."""
    pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=float)
    pts[:, 0] *= frame_w
    pts[:, 1] *= frame_h

    v = pts[9] - pts[0]
    size = np.linalg.norm(v) or 1e-6
    y_hat = v / size
    x_hat = np.array([y_hat[1], -y_hat[0]])
    local = ((pts - pts[0]) @ np.stack([x_hat, y_hat], axis=1)) / size

    if local[2][0] < 0:
        local[:, 0] *= -1

    return local.flatten()


# ── Camera loop ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0
frame_no = 0

print("Camera ready.")
print("Show your hand to the camera.")
print("Controls: Q = quit | R = reset | S = speak")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    frame_no += 1

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_timestamp_ms += 33
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    predicted_letter = ""
    confidence = 0.0
    motion_letter = ""

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        # ── Check for a motion sign (J/Z) first ─────────────────────────────
        motion_letter, start_frame = motion_detector.update(landmarks, frame_no)
        if motion_letter:
            debouncer.commit_motion(motion_letter, start_frame)
        else:
            # ── Fall back to the static landmark classifier ─────────────────
            feats = normalize_landmarks(landmarks, w, h).reshape(1, -1)
            probs = model.predict_proba(feats)[0]
            pred_idx = int(np.argmax(probs))
            confidence = probs[pred_idx]
            if confidence > CONFIDENCE_THRESHOLD:
                predicted_letter = LETTERS[pred_idx]

        # ── Draw landmarks and prediction ───────────────────────────────────
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x1, y1, x2, y2 = int(min(xs)) - 20, int(min(ys)) - 20, int(max(xs)) + 20, int(max(ys)) + 20
        shown_letter = motion_letter or predicted_letter
        color = (0, 255, 255) if motion_letter else ((0, 255, 0) if predicted_letter else (0, 165, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        for lm in landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (px, py), 2, (255, 255, 0), -1)
        if shown_letter:
            label = f"{shown_letter} (motion)" if motion_letter else f"{shown_letter} ({confidence:.0%})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    else:
        motion_detector.clear()
        cv2.putText(frame, "No hand detected", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # ── Debounce (static letters only — motion letters already committed) ──
    if not motion_letter:
        raw_string = debouncer.update(predicted_letter, frame_no)
    else:
        raw_string = debouncer.confirmed_string
    corrected = nlp.correct(raw_string)

    # ── HUD ────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"Raw: {raw_string[-30:]}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Corrected: {corrected}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    cv2.putText(frame, "Q=quit R=reset S=speak",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (150, 150, 150), 1)

    cv2.imshow("ASL Translator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        debouncer.reset()
        motion_detector.clear()
        print("Reset.")
    elif key == ord('s'):
        audio.speak(corrected)
        print(f"Spoken: {corrected}")

# ── Cleanup ────────────────────────────────────────────────────────────────
landmarker.close()
cap.release()
cv2.destroyAllWindows()