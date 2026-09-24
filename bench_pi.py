"""Measure MediaPipe throughput on this machine. No GUI, no model, no signing.

This is the first thing to run on new hardware, because it answers the only
question that decides whether the board is usable: how many frames per second
can it push through the hand landmarker. Everything else in the pipeline is
cheap by comparison -- the classifier is a 42-input MLP and the rest is
arithmetic over 21 points.

Targets, from the emulation measurements on the laptop:

  >= 25-30 fps   motion signs (J/Z) work; full alphabet viable
  12-20 fps      static letters and word signs fine, J/Z unreliable
  < 12 fps       static letters degrade too, especially at low resolution
                 (320x240 at 10fps measured 48%, against 94% at 640x480)

Run it for at least a few minutes: a Pi throttles when it heats up, and the
number that matters is the sustained one, not the first ten seconds.

    python bench_pi.py                       # native resolution, 60s
    python bench_pi.py --width 320 --height 240
    python bench_pi.py --seconds 600         # thermal soak
"""
import argparse
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

LANDMARKER_PATH = "hand_landmarker.task"


def cpu_temp_c():
    """Pi exposes core temperature here; returns None elsewhere."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as fh:
            return int(fh.read().strip()) / 1000.0
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    landmarker = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )

    print(f"camera {w}x{h}, running {args.seconds:.0f}s. Hold a hand in view.")
    t0 = time.monotonic()
    frames = 0
    detections = 0
    window_start = t0
    window_frames = 0

    while time.monotonic() - t0 < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int((time.monotonic() - t0) * 1000)
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(image, ts)
        frames += 1
        window_frames += 1
        if result.hand_landmarks:
            detections += 1

        now = time.monotonic()
        if now - window_start >= 10.0:
            fps = window_frames / (now - window_start)
            temp = cpu_temp_c()
            temp_s = f", {temp:.0f}C" if temp is not None else ""
            print(f"  t+{now-t0:5.0f}s  {fps:5.1f} fps{temp_s}")
            window_start, window_frames = now, 0

    elapsed = time.monotonic() - t0
    landmarker.close()
    cap.release()

    fps = frames / elapsed if elapsed else 0.0
    print(f"\n{w}x{h}: {fps:.1f} fps average over {elapsed:.0f}s "
          f"({frames} frames, hand found in {detections})")
    if fps >= 25:
        print("  >= 25fps: motion signs (J/Z) should work. Full alphabet viable.")
    elif fps >= 12:
        print("  12-25fps: static letters and word signs fine, J/Z unreliable.")
        print("  Try --width 320 --height 240 to buy frames.")
    else:
        print("  < 12fps: too slow. Static letters degrade here too, badly so at")
        print("  low resolution. Try 320x240; if that does not clear 12fps, this")
        print("  board cannot run the pipeline as it stands.")
    temp = cpu_temp_c()
    if temp is not None and temp >= 80:
        print(f"  {temp:.0f}C -- thermal throttling likely. Check cooling before "
              f"trusting a low number.")


if __name__ == "__main__":
    main()
