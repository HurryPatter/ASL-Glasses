"""
End-to-end evaluation harness.

Unlike train_classifier.py's validation split (which only tests the
classifier against held-out data from the SAME collection sessions), this
runs the full live pipeline -- MediaPipe -> motion detector -> static
classifier -> debouncer -- against a known target letter, the way it will
actually be used. This is what should go in the thesis as the real
accuracy number.

Run once per condition (background/lighting/distance) you want to test.
Each run appends to eval_results.csv rather than overwriting it, so you
can build up results across many conditions over multiple sessions.

Controls:
  SPACE = start the capture window for the current target letter
  N     = skip to the next letter without recording a result
  Q     = quit and print the summary so far
"""
import csv
import json
import os
import time
from collections import Counter

import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import argparse

import hwprofile
from debouncer import Debouncer
from motion import MotionDetector

LANDMARKER_PATH = "hand_landmarker.task"
MODEL_PATH = "landmark_model.joblib"
LABELS_PATH = "landmark_labels.json"
CONFIDENCE_THRESHOLD = 0.85
CAPTURE_WINDOW_S = 4.0          # how long you get per letter before it auto-advances
RESULTS_PATH = "eval_results.csv"

TARGET_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # includes J/Z, unlike collect_data.py


def normalize_landmarks(landmarks, frame_w, frame_h):
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


RESULTS_HEADER = ["person", "condition", "profile", "width", "height",
                  "fps_cap", "fps_actual", "expected", "committed", "correct"]
# Older layouts, migrated in place on first run rather than left incomparable.
LEGACY_HEADERS = [
    ["condition", "expected", "committed", "correct"],
    ["person", "condition", "expected", "committed", "correct"],
]


def ask(prompt, default):
    value = input(prompt).strip().lower()
    return value or default


def open_results():
    """Append to eval_results.csv, migrating older layouts in place.

    Two columns were added over time and both matter for comparing runs:
    `person`, without which a multi-person multi-environment run cannot be
    split by either afterwards, and the hardware-profile columns, without
    which a 320x240 run is indistinguishable from a native-resolution one.
    Rows predating a column are filled with what is actually known about them
    -- 'unknown' for the signer, native/uncapped for the profile, since every
    earlier run was at the camera's own resolution and frame rate.
    """
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, newline="") as fh:
            rows = list(csv.reader(fh))
        if rows and rows[0] in LEGACY_HEADERS:
            old = rows[0]
            pad_person = ["unknown"] if old[0] != "person" else []
            with open(RESULTS_PATH, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(RESULTS_HEADER)
                for r in rows[1:]:
                    r = pad_person + r
                    # person, condition, [profile...], expected, committed, correct
                    w.writerow(r[:2] + ["native@uncapped", "", "", "", ""] + r[2:])
            print(f"Migrated {len(rows)-1} existing rows in {RESULTS_PATH} to the "
                  f"current layout (earlier runs were native resolution, uncapped).")
        csv_file = open(RESULTS_PATH, "a", newline="")
        return csv_file, csv.writer(csv_file)

    csv_file = open(RESULTS_PATH, "a", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(RESULTS_HEADER)
    return csv_file, writer


def void_last_row(csv_file):
    """Remove the most recently written row and reopen for appending.

    Rows are flushed as each capture completes, so a run survives a crash.
    That means undo cannot just forget a buffered row -- it has to rewrite the
    file. The file is small and this happens at human speed, so the simplest
    correct thing is to close, drop the final line, and reopen.
    """
    csv_file.close()
    with open(RESULTS_PATH, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) > 1:
        rows = rows[:-1]
    with open(RESULTS_PATH, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    fh2 = open(RESULTS_PATH, "a", newline="")
    return fh2, csv.writer(fh2)


def parse_args():
    p = argparse.ArgumentParser(
        description="Accuracy benchmark. --width/--height/--fps emulate slower "
                    "hardware on this laptop, to decide the embedded target "
                    "before buying one.")
    p.add_argument("--width", type=int, default=None,
                   help="capture width, e.g. 320 (default: the camera's own)")
    p.add_argument("--height", type=int, default=None,
                   help="capture height, e.g. 240")
    p.add_argument("--letters", default=None,
                   help="only these letters, e.g. JZ. The default alphabet pass "
                        "gives one J and one Z per five-minute run, which is far "
                        "too few to measure the motion signs -- they are the ones "
                        "that fail first as the frame rate drops.")
    p.add_argument("--repeat", type=int, default=1,
                   help="repeat the letter set this many times, cycling rather "
                        "than blocking (J Z J Z ... not J J ... Z Z), so fatigue "
                        "and habituation do not load onto one sign.")
    p.add_argument("--fps", type=float, default=None,
                   help="process at most this many frames per second; extra "
                        "frames are dropped unprocessed, emulating a board "
                        f"that cannot keep up. Below "
                        f"{hwprofile.motion_floor_fps():.1f} J and Z cannot "
                        f"fire at all.")
    a = p.parse_args()
    letters = (a.letters or TARGET_LETTERS).upper()
    bad = [c for c in letters if c not in TARGET_LETTERS]
    if bad:
        p.error(f"not letters: {''.join(bad)}")
    if a.repeat < 1:
        p.error("--repeat must be at least 1")
    a.sequence = list(letters) * a.repeat
    return a


def main():
    args = parse_args()
    # Both are recorded per row so a multi-person, multi-environment run can
    # be split by either afterwards. One label for the whole run cannot tell
    # "this person struggles" from "this lighting is hard".
    person = ask("Who is signing? (first name, e.g. omar): ", "unknown")
    condition = ask("Condition? (e.g. 'outdoor_sun', 'indoor_white_bg'): ", "unlabeled")

    model = joblib.load(MODEL_PATH)
    with open(LABELS_PATH) as f:
        LETTERS = json.load(f)

    base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    landmarker = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )

    cap = cv2.VideoCapture(0)
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # Cameras snap to the nearest mode they support, so record what was
    # actually delivered rather than what was requested.
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.width and (actual_w, actual_h) != (args.width, args.height):
        print(f"NOTE: asked for {args.width}x{args.height}, camera gave "
              f"{actual_w}x{actual_h}. Logging what it gave.")
    limiter = hwprofile.FrameLimiter(args.fps)
    profile = hwprofile.describe(actual_w, actual_h, args.fps)
    if args.fps and args.fps < hwprofile.motion_floor_fps():
        print(f"NOTE: {args.fps}fps is below the {hwprofile.motion_floor_fps():.1f}fps "
              f"floor -- J and Z cannot fire at this rate, by construction.")
    start_time = time.monotonic()

    csv_file, writer = open_results()

    sequence = args.sequence
    results = []  # (expected, committed, correct) for this run's summary
    run_rows = 0  # rows this run has written, so X can never void an older one

    letter_idx = 0
    capturing = False
    capture_start = None
    debouncer = Debouncer()
    motion_detector = MotionDetector()
    frame_no = 0
    consecutive_missed = 0
    baseline_len = 0  # confirmed_string length when the capture window started

    print(f"\nRunning eval for condition: {condition} / {profile}")
    print(f"{len(sequence)} captures: {' '.join(sequence[:12])}"
          f"{' ...' if len(sequence) > 12 else ''}")
    print("SPACE = start capture | X = void the last one | N = skip | Q = quit")
    print("Press X whenever you fumble a sign -- a forgotten or wrong handshape")
    print("is your error, not the pipeline's, and it moves a 25-capture run by")
    print("four points.\n")

    while letter_idx < len(sequence):
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        if not limiter.should_process(timestamp_ms):
            # Dropped before any processing: this is what a board that cannot
            # keep up would do. Still service the keyboard so the run stays
            # controllable at low frame rates.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_no += 1

        target = sequence[letter_idx]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        predicted_letter = ""
        motion_letter = ""
        skip_debounce = False

        if capturing and result.hand_landmarks:
            consecutive_missed = 0
            landmarks = result.hand_landmarks[0]
            motion_letter, start_frame = motion_detector.update(landmarks, frame_no, timestamp_ms)
            if motion_letter:
                debouncer.commit_motion(motion_letter, start_frame)
            elif not motion_detector.is_motion_candidate():
                feats = normalize_landmarks(landmarks, w, h).reshape(1, -1)
                probs = model.predict_proba(feats)[0]
                pred_idx = int(np.argmax(probs))
                if probs[pred_idx] > CONFIDENCE_THRESHOLD:
                    predicted_letter = LETTERS[pred_idx]
        elif capturing:
            consecutive_missed += 1
            if consecutive_missed >= 5:
                motion_detector.clear()
            else:
                skip_debounce = True

        if capturing:
            if motion_letter or skip_debounce:
                pass
            else:
                debouncer.update(predicted_letter, frame_no, timestamp_ms)

            if time.monotonic() - capture_start > CAPTURE_WINDOW_S:
                committed = debouncer.confirmed_string[baseline_len:]
                correct = target in committed
                results.append((target, committed, correct))
                writer.writerow([person, condition, profile, actual_w, actual_h,
                                 args.fps or "", f"{limiter.achieved_fps():.1f}",
                                 target, committed, correct])
                csv_file.flush()
                run_rows += 1
                print(f"  {target}: got '{committed}' -> "
                      f"{'OK' if correct else 'MISS'}   (X to void)")
                capturing = False
                letter_idx += 1

        # ── HUD ──────────────────────────────────────────────────────────
        status = f"CAPTURING ({CAPTURE_WINDOW_S - (time.monotonic() - capture_start):.1f}s)" if capturing else "ready"
        cv2.putText(frame, f"Sign: {target}   [{status}]", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Progress: {letter_idx}/{len(sequence)}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "SPACE=capture  X=void last  N=skip  Q=quit", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.imshow("Evaluation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n') and not capturing:
            results.append((target, "", False))
            writer.writerow([person, condition, profile, actual_w, actual_h,
                             args.fps or "", f"{limiter.achieved_fps():.1f}",
                             target, "", False])
            csv_file.flush()
            run_rows += 1
            letter_idx += 1
        elif key == ord('x') and not capturing:
            # Only ever voids a row this run wrote, never one from an earlier
            # session that happens to be last in the file.
            if run_rows and results:
                voided = results.pop()
                csv_file, writer = void_last_row(csv_file)
                run_rows -= 1
                letter_idx -= 1
                print(f"  voided {voided[0]} -> '{voided[1]}'; sign it again")
            else:
                print("  nothing from this run to void")
        elif key == ord(' ') and not capturing:
            capturing = True
            capture_start = time.monotonic()
            baseline_len = len(debouncer.confirmed_string)
            motion_detector.clear()

    csv_file.close()
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    # ── This run's summary ───────────────────────────────────────────────
    if results:
        n_correct = sum(1 for _, _, ok in results if ok)
        print(f"\n{person} / {condition} / {profile}: "
              f"{n_correct}/{len(results)} correct ({n_correct/len(results):.1%})")
        print(f"  camera delivered {limiter.capture_fps():.1f}fps, "
              f"pipeline processed {limiter.achieved_fps():.1f}fps "
              f"({limiter.processed} of {limiter.seen} frames)")
        misses = [(t, c) for t, c, ok in results if not ok]
        if misses:
            print("Missed:", ", ".join(f"{t}->'{c}'" for t, c in misses))
    print(f"\nAll results appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
