import json
import time
from collections import deque, Counter

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
MISSED_FRAME_TOLERANCE = 5   # consecutive dropped-tracking frames tolerated before giving up
DISPLAY_SMOOTHING_N = 6      # how many recent guesses the on-screen readout is smoothed over

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
recent_guesses = deque(maxlen=DISPLAY_SMOOTHING_N)  # display-only, never fed to the debouncer


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


def smoothed_display_letter():
    """Majority vote over recent guesses, for the on-screen readout only.
    Purely cosmetic -- does not affect what gets committed to the debouncer."""
    if not recent_guesses:
        return "", 0.0
    letter, count = Counter(recent_guesses).most_common(1)[0]
    return letter, count / len(recent_guesses)


# ── Camera loop ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
start_time = time.monotonic()
frame_no = 0
consecutive_missed = 0

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

    # Real elapsed wall-clock time, not an assumed fixed frame duration --
    # this is what makes both MediaPipe's tracker and the motion timing
    # correct regardless of the actual camera frame rate.
    timestamp_ms = int((time.monotonic() - start_time) * 1000)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    predicted_letter = ""
    confidence = 0.0
    motion_letter = ""
    skip_debounce_this_frame = False

    if result.hand_landmarks:
        consecutive_missed = 0
        landmarks = result.hand_landmarks[0]

        # ── Check for a motion sign (J/Z) first ─────────────────────────────
        motion_letter, start_frame = motion_detector.update(landmarks, frame_no, timestamp_ms)
        if motion_letter:
            debouncer.commit_motion(motion_letter, start_frame)
        elif motion_detector.is_motion_candidate():
            # Handshape matches + fingertip is actively moving -- a J/Z
            # attempt is still mid-flight. Hold off on static classification
            # this frame so a fast-moving 'X'/'I' handshape doesn't get
            # committed as a static letter before the gesture finishes.
            pass
        else:
            # ── Fall back to the static landmark classifier ─────────────────
            feats = normalize_landmarks(landmarks, w, h).reshape(1, -1)
            probs = model.predict_proba(feats)[0]
            pred_idx = int(np.argmax(probs))
            confidence = probs[pred_idx]
            top_guess = LETTERS[pred_idx]
            if confidence > CONFIDENCE_THRESHOLD:
                predicted_letter = top_guess
                recent_guesses.append(top_guess)

            cv2.putText(frame, f"top guess: {top_guess} ({confidence:.0%})",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 1)

        # ── TEMP: motion tuning overlay — remove once J/Z are fully settled ──
        dbg = motion_detector.debug_info()
        if dbg:
            for i, (k, v) in enumerate(dbg.items()):
                cv2.putText(frame, f"{k}: {v}", (w - 260, 30 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # ── Draw landmarks + smoothed display letter ────────────────────────
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x1, y1, x2, y2 = int(min(xs)) - 20, int(min(ys)) - 20, int(max(xs)) + 20, int(max(ys)) + 20
        disp_letter, disp_ratio = smoothed_display_letter()
        shown_letter = motion_letter or disp_letter
        color = (0, 255, 255) if motion_letter else ((0, 255, 0) if disp_letter else (0, 165, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        for lm in landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (px, py), 2, (255, 255, 0), -1)
        if shown_letter:
            label = f"{shown_letter} (motion)" if motion_letter else f"{shown_letter} ({disp_ratio:.0%} steady)"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    else:
        # Tolerate a few dropped-tracking frames before disturbing anything --
        # fast gestures and brief occlusions are the most likely causes of a
        # dropout, and wiping state on frame 1 of one throws away exactly the
        # holds/gestures we most want to keep.
        consecutive_missed += 1
        recent_guesses.clear()
        if consecutive_missed >= MISSED_FRAME_TOLERANCE:
            motion_detector.clear()
        else:
            skip_debounce_this_frame = True
        cv2.putText(frame, "No hand detected", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # ── Debounce (static letters only — motion letters already committed) ──
    if motion_letter or skip_debounce_this_frame:
        raw_string = debouncer.confirmed_string
    else:
        raw_string = debouncer.update(predicted_letter, frame_no)
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
        recent_guesses.clear()
        print("Reset.")
    elif key == ord('s'):
        audio.speak(corrected)
        print(f"Spoken: {corrected}")

# ── Cleanup ────────────────────────────────────────────────────────────────
landmarker.close()
cap.release()
cv2.destroyAllWindows()