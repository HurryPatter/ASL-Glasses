"""ASL fingerspelling translator, rule-based edition (branch: trial).

Pipeline per frame:
  camera -> MediaPipe hand landmarks -> geometric rules (rules.py) -> letter
         -> Debouncer -> SymSpell word segmentation -> HUD / speech
J and Z, which need motion, come from the trajectory rules in motion.py.
No trained letter model is used anywhere.
"""
import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from rules import classify, FINGERS
from debouncer import Debouncer
from motion import MotionDetector
from nlp_bridge import NLPBridge
from audio import AudioOutput

# ── Constants ──────────────────────────────────────────────────────────────
LANDMARKER_PATH = "hand_landmarker.task"
MIN_SCORE = 0.85          # fraction of a letter's rule weight that must pass
MIN_MARGIN = 0.05         # best letter must beat the runner-up by this much
STILL_THRESHOLD = 0.15    # hand widths moved over 5 frames; above this, static letters are ignored
MOTION_FLASH_FRAMES = 15  # how long a recognised J/Z stays on the HUD
CAMERA_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # e.g. `main.py 1` when a phone sits at index 0

# Skeleton edges for drawing
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4)] + [(0, 5), (0, 17), (5, 9), (9, 13), (13, 17)]
for _f in FINGERS.values():
    EDGES += [(_f[0], _f[1]), (_f[1], _f[2]), (_f[2], _f[3])]

# ── Load components ────────────────────────────────────────────────────────
print("Loading MediaPipe hand landmarker...")
landmarker = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
))

print("Initialising pipeline...")
debouncer = Debouncer(min_frames=3)
motion = MotionDetector(window=20)
nlp = NLPBridge(mode="academic")
audio = AudioOutput()


def debug_panel(features, ranked, size=360):
    """Hand skeleton in the normalised hand frame plus rule readout and top scores."""
    panel = np.full((size, size + 300, 3), 30, np.uint8)
    p = features.p
    scale = size / 4.5
    cx, cy = size // 2, int(size * 0.8)
    pix = [(int(cx + x * scale), int(cy - y * scale)) for x, y in p]
    for a, b in EDGES:
        cv2.line(panel, pix[a], pix[b], (200, 200, 200), 2)
    for i, (x, y) in enumerate(pix):
        cv2.circle(panel, (x, y), 4, (0, 255, 255) if i == 4 else (255, 200, 0), -1)
    cv2.putText(panel, "hand frame (thumb on right)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    x0 = size + 10
    cv2.putText(panel, "rules", (x0, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    for i, line in enumerate(features.readout().split("  ")):
        cv2.putText(panel, line, (x0, 52 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(panel, "scores", (x0, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    for i, (letter, s) in enumerate(ranked[:5]):
        color = (0, 255, 0) if i == 0 and s >= MIN_SCORE else (200, 200, 200)
        cv2.putText(panel, f"{letter}  {s:.2f}", (x0, 228 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return panel


# ── Camera loop ────────────────────────────────────────────────────────────
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
    frame_timestamp_ms += 33
    frame_no += 1
    result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    predicted_letter = ""
    score = 0.0
    motion_letter, motion_start = "", None
    hand_speed = 0.0
    ranked = []

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        motion_letter, motion_start = motion.update(landmarks, frame_no)
        hand_speed = motion.speed(5)

        # ── Rule-based letter ──────────────────────────────────────────────
        predicted_letter, score, features, ranked = classify(
            landmarks, w, h, min_score=MIN_SCORE, min_margin=MIN_MARGIN)
        if hand_speed > STILL_THRESHOLD:
            predicted_letter = ""  # mid-motion: don't commit static guesses
        cv2.imshow("What the rules see", debug_panel(features, ranked))

        # ── Draw skeleton and prediction ───────────────────────────────────
        pix = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        color = (0, 255, 0) if predicted_letter else (0, 165, 255)
        for a, b in EDGES:
            cv2.line(frame, pix[a], pix[b], color, 2)
        for x, y in pix:
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)
        xs, ys = zip(*pix)
        label = f"{predicted_letter} ({score:.0%})" if predicted_letter else \
                ("moving..." if hand_speed > STILL_THRESHOLD else f"? {ranked[0][0]} {ranked[0][1]:.0%}")
        cv2.putText(frame, label, (min(xs), min(ys) - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    else:
        motion.clear()
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
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    if frame_no % 30 == 0:  # once a second: what the pipeline sees
        if ranked:
            top = " ".join(f"{l}:{s:.2f}" for l, s in ranked[:3])
            print(f"[{frame_no}] hand=yes speed={hand_speed:.2f} top3=[{top}] raw='{raw_string}'")
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
