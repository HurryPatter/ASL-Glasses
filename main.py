"""The live translator: camera -> hand landmarks -> letters -> text -> speech.

Two ways to run it:

    python main.py                 # preview window and keyboard (laptop, desk)
    python main.py --headless      # no window: the glasses

Headless is how it runs on the glasses, started at boot (see
install_service.sh). With no keyboard, the signer speaks by lowering their
hands: after --pause-ms (1.5s) with no hand in view, the text so far is
spoken and cleared (speech_trigger.py). An optional push button on the body
unit does the same on a short press and discards the text on a long one
(--button-pin, a BCM GPIO number; needs gpiozero, which Pi OS ships). The
current text is printed whenever it changes, until there is a display for it.

Window controls: Q quit, R reset, S speak (and clear), D motion debug.
"""
import argparse
import signal
import time
from collections import Counter, deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from debouncer import Debouncer
from nlp_bridge import NLPBridge
from audio import AudioOutput
from motion import MotionDetector
from speech_trigger import SpeechTrigger
import camera
from features import load_classifier, normalize_landmarks

LANDMARKER_PATH = "hand_landmarker.task"
MODEL_PATH = "landmark_model.joblib"
LABELS_PATH = "landmark_labels.json"
CONFIDENCE_THRESHOLD = 0.85
MISSED_FRAME_TOLERANCE = 5   # consecutive dropped-tracking frames tolerated before giving up
DISPLAY_SMOOTHING_N = 6      # how many recent guesses the on-screen readout is smoothed over
BUTTON_HOLD_S = 1.0          # a press at least this long discards instead of speaking


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Live ASL fingerspelling translator.")
    p.add_argument("--headless", action="store_true",
                   help="no preview window or keyboard: for the glasses")
    p.add_argument("--model", default=MODEL_PATH,
                   help=f"classifier to use (default {MODEL_PATH}), e.g. a "
                        f"first-person model from train_classifier.py --out")
    p.add_argument("--pause-ms", type=int, default=1500,
                   help="speak and clear after this long with no hand in view "
                        "(0 turns it off; default 1500)")
    p.add_argument("--button-pin", type=int, default=None,
                   help="BCM GPIO pin of a push button: short press speaks, "
                        "long press discards")
    p.add_argument("--frames", type=int, default=None,
                   help="stop after this many frames (testing, benchmarking)")
    return p.parse_args(argv)


def make_landmarker():
    return mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )


class Button:
    """Short press: speak now. Long press: discard. Callbacks only set flags;
    the camera loop acts on them, so nothing touches the pipeline from
    gpiozero's thread."""

    def __init__(self, pin):
        from gpiozero import Button as GPIOButton   # only on the Pi
        self.speak = self.discard = False
        self._held = False
        self._button = GPIOButton(pin, hold_time=BUTTON_HOLD_S)
        self._button.when_held = self._on_held
        self._button.when_released = self._on_released

    def _on_held(self):
        self._held = True
        self.discard = True

    def _on_released(self):
        if not self._held:
            self.speak = True
        self._held = False

    def take(self):
        """(speak, discard) since the last call."""
        speak, discard = self.speak, self.discard
        self.speak = self.discard = False
        return speak, discard


def open_button(pin):
    if pin is None:
        return None
    try:
        return Button(pin)
    except Exception as e:   # no gpiozero, not a Pi, pin in use
        print(f"Button on GPIO{pin} unavailable ({e}); carrying on without it.")
        return None


class Translator:
    """Everything between a camera frame and the committed text."""

    def __init__(self, model_path):
        print("Loading landmark classifier...")
        self.model, self.labels = load_classifier(model_path, LABELS_PATH)
        print("Loading MediaPipe hand landmarker...")
        self.landmarker = make_landmarker()
        self.debouncer = Debouncer()
        self.motion = MotionDetector()
        self.nlp = NLPBridge(mode="academic")
        self.recent = deque(maxlen=DISPLAY_SMOOTHING_N)  # display only
        self.missed = 0

    def reset(self):
        self.debouncer.reset()
        self.motion.clear()
        self.recent.clear()

    def text(self):
        labels = [label for label, _ in self.debouncer.commits]
        return " ".join(labels), self.nlp.correct_sequence(labels)

    def step(self, frame, frame_no, timestamp_ms):
        """Process one frame. Returns what the preview needs to draw it."""
        h, w = frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect_for_video(image, timestamp_ms)

        predicted = ""
        motion_letter = ""
        skip_debounce = False
        landmarks = None

        if result.hand_landmarks:
            self.missed = 0
            landmarks = result.hand_landmarks[0]
            # Motion signs (J/Z) first.
            motion_letter, start_frame = self.motion.update(landmarks, frame_no,
                                                            timestamp_ms)
            if motion_letter:
                self.debouncer.commit_motion(motion_letter, start_frame)
            elif self.motion.is_motion_candidate():
                # Handshape matches and the fingertip is moving: a J/Z is
                # mid-flight. Hold off on static classification so a moving
                # X/I is not committed before the gesture finishes.
                pass
            else:
                feats = normalize_landmarks(landmarks, w, h).reshape(1, -1)
                probs = self.model.predict_proba(feats)[0]
                idx = int(np.argmax(probs))
                if probs[idx] > CONFIDENCE_THRESHOLD:
                    predicted = self.labels[idx]
                    self.recent.append(predicted)
        else:
            # Tolerate a few dropped-tracking frames before disturbing
            # anything: fast gestures and brief occlusions cause most
            # dropouts, and wiping state on the first one throws away exactly
            # the holds and gestures we most want to keep.
            self.missed += 1
            self.recent.clear()
            if self.missed >= MISSED_FRAME_TOLERANCE:
                self.motion.clear()
            else:
                skip_debounce = True

        # Static letters only; motion letters were committed above.
        if not (motion_letter or skip_debounce):
            self.debouncer.update(predicted, frame_no, timestamp_ms)
        return landmarks, motion_letter

    def smoothed_display_letter(self):
        """Majority vote over recent guesses, for the preview only."""
        if not self.recent:
            return "", 0.0
        letter, count = Counter(self.recent).most_common(1)[0]
        return letter, count / len(self.recent)

    def close(self):
        self.landmarker.close()


def draw(frame, t, landmarks, motion_letter, raw, corrected, show_debug):
    h, w = frame.shape[:2]
    if landmarks is not None:
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x1, y1 = int(min(xs)) - 20, int(min(ys)) - 20
        x2, y2 = int(max(xs)) + 20, int(max(ys)) + 20
        disp_letter, disp_ratio = t.smoothed_display_letter()
        shown = motion_letter or disp_letter
        color = (0, 255, 255) if motion_letter else ((0, 255, 0) if disp_letter
                                                    else (0, 165, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (255, 255, 0), -1)
        if shown:
            label = (f"{shown} (motion)" if motion_letter
                     else f"{shown} ({disp_ratio:.0%} steady)")
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    else:
        cv2.putText(frame, "No hand detected", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.putText(frame, f"Raw: {raw[-40:]}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"Corrected: {corrected}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    cv2.putText(frame, "Q=quit R=reset S=speak D=debug", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Motion-detector readout (D). Off by default so a demo stays clean. The
    # J/Z thresholds are the only numbers in the pipeline that have to be
    # tuned against a real hand -- hand_travel in particular, which is what
    # stops a held Q from firing Z. Watch it while signing a real Z and while
    # holding a Q: z_travel belongs between the two.
    if show_debug:
        y = 150
        for key, value in t.motion.debug_info().items():
            cv2.putText(frame, f"{key}: {value}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            y += 20
        cv2.putText(frame, f"thresholds: travel>{t.motion.z_travel} "
                           f"stroke>{t.motion.z_stroke} drift>{t.motion.z_drop}",
                    (10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


def run(args):
    t = Translator(args.model)
    audio = AudioOutput()
    trigger = SpeechTrigger(args.pause_ms)
    button = open_button(args.button_pin)

    # systemd stops the service with SIGTERM; treat it like Ctrl+C so the
    # camera is released cleanly.
    stopping = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stopping.update(now=True))

    cap = camera.open_camera()   # ribbon camera if attached, else USB
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Check it with: python camera.py")
    start = time.monotonic()
    frame_no = 0
    show_debug = False
    last_printed = None

    def speak(text):
        audio.speak(text)
        print(f"SPOKEN: {text}", flush=True)
        t.reset()
        trigger.reset()

    print("Camera ready." + (" Running headless." if args.headless else ""))
    if args.pause_ms:
        print(f"Lower your hands for {args.pause_ms / 1000:g}s to speak.")
    if not args.headless:
        print("Controls: Q = quit | R = reset | S = speak | D = debug")
    else:
        audio.speak("Ready")

    try:
        while not stopping["now"]:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frame_no += 1
            # Real elapsed time, not an assumed frame duration: this is what
            # keeps MediaPipe's tracker and the motion timing right whatever
            # the camera's actual frame rate.
            now_ms = int((time.monotonic() - start) * 1000)

            landmarks, motion_letter = t.step(frame, frame_no, now_ms)
            raw, corrected = t.text()

            said = trigger.update(now_ms, landmarks is not None, corrected)
            if button:
                pressed, discard = button.take()
                if discard:
                    t.reset()
                    trigger.reset()
                    print("Discarded.", flush=True)
                    corrected = ""
                elif pressed and corrected:
                    said = corrected
            if said:
                speak(said)
                raw, corrected = "", ""

            if args.headless:
                if corrected != last_printed and corrected:
                    print(f"text: {corrected}", flush=True)
                last_printed = corrected
            else:
                draw(frame, t, landmarks, motion_letter, raw, corrected, show_debug)
                cv2.imshow("ASL Translator", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    t.reset()
                    trigger.reset()
                    print("Reset.")
                elif key == ord('s') and corrected:
                    speak(corrected)
                elif key == ord('d'):
                    show_debug = not show_debug

            if args.frames and frame_no >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        t.close()
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
    return frame_no


if __name__ == "__main__":
    run(parse_args())
