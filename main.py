import cv2
import numpy as np
from keras.models import load_model

from debouncer import Debouncer
from nlp_bridge import NLPBridge
from audio import AudioOutput

# ── Constants ──────────────────────────────────────────────────────────────
ALPHABET  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VALID_IDX = [i for i in range(26) if i not in [9, 25]]   # no J or Z
MODEL_PATH = "asl_model.h5"

# ── Load components ────────────────────────────────────────────────────────
print("Loading CNN model...")
model = load_model(MODEL_PATH)

print("Initialising pipeline...")
debouncer = Debouncer(min_frames=3)
nlp       = NLPBridge(mode="academic")
audio     = AudioOutput()

# ── Hand region detector (skin colour segmentation) ────────────────────────
def detect_hand_roi(frame, roi_box):
    """Only search for hand inside the green ROI box."""
    x1_roi, y1_roi, x2_roi, y2_roi = roi_box
    roi_frame = frame[y1_roi:y2_roi, x1_roi:x2_roi]

    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0,  40,  80],  dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 5000:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    pad = 10
    # Convert back to full frame coordinates
    x1 = max(0, x1_roi + x - pad)
    y1 = max(0, y1_roi + y - pad)
    x2 = min(frame.shape[1], x1_roi + x + w + pad)
    y2 = min(frame.shape[0], y1_roi + y + h + pad)
    return x1, y1, x2, y2

# ── Camera loop ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

frame_h, frame_w = 480, 640
roi_x1, roi_y1 = frame_w // 2, 50
roi_x2, roi_y2 = frame_w - 20, frame_h - 50

print("Camera ready.")
print("Hold your hand inside the GREEN BOX.")
print("Controls: Q = quit | R = reset | S = speak")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ── Draw ROI guide box ─────────────────────────────────────────────────
    roi_box = (roi_x1, roi_y1, roi_x2, roi_y2)
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    cv2.putText(frame, "Place hand here", (roi_x1 + 5, roi_y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    roi = detect_hand_roi(frame, roi_box)

    predicted_letter = ""

    if roi:
        x1, y1, x2, y2 = roi
        hand_crop = frame[y1:y2, x1:x2]

        if hand_crop.size > 0:
            # ── Preprocess to match CNN training format ────────────────────
            gray       = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
            resized    = cv2.resize(gray, (28, 28))
            normalized = resized / 255.0
            input_arr  = normalized.reshape(1, 28, 28, 1)

            # ── CNN prediction ─────────────────────────────────────────────
            predictions = model.predict(input_arr, verbose=0)
            pred_idx    = np.argmax(predictions)
            confidence  = predictions[0][pred_idx]

            if pred_idx in VALID_IDX and confidence > 0.85:
                predicted_letter = ALPHABET[pred_idx]

            # ── Draw bounding box and prediction ───────────────────────────
            color = (0, 255, 0) if predicted_letter else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if predicted_letter:
                cv2.putText(frame, f"{predicted_letter} ({confidence:.0%})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, color, 2)

    # ── Debounce → NLP ─────────────────────────────────────────────────────
    raw_string = debouncer.update(predicted_letter)
    corrected  = nlp.correct(raw_string)

    # ── HUD ────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"Raw:       {raw_string[-30:]}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Corrected: {corrected}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    cv2.putText(frame, "Q=quit  R=reset  S=speak",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
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
cap.release()
cv2.destroyAllWindows()