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

Keep a hand in view for most of the run. With no hand, MediaPipe runs only
the palm detector; with one, it runs the landmark model too, and the
classifier would follow -- so a run with no hand measures the wrong path.

fps alone cannot show headroom when the camera is the limit (the ribbon
camera stops at 30), so the per-frame processing time is reported as well:
the time from a frame arriving to its landmarks coming back. That number,
not fps, is how far the board is from its ceiling.

    python bench_pi.py                       # native resolution, 60s
    python bench_pi.py --width 320 --height 240
    python bench_pi.py --seconds 600         # thermal soak
"""
import argparse
import statistics
import time

import cv2

import camera
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


def latency_summary(ms):
    """(median, 95th percentile) of a list of per-frame times, or None."""
    if not ms:
        return None
    ordered = sorted(ms)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return statistics.median(ordered), p95


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    source = camera.detect_source()
    cap = camera.open_camera(args.width, args.height, index=args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open {source} camera {args.camera}")
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

    print(f"{source} camera {w}x{h}, running {args.seconds:.0f}s. Hold a hand in view.")
    if source == "csi":
        print("  (the ribbon camera is fixed at 30fps, so ~30fps here means the "
              "Pi keeps up with it)")
    t0 = time.monotonic()
    frames = 0
    detections = 0
    hand_ms, empty_ms = [], []
    window_start = t0
    window_frames = 0

    while time.monotonic() - t0 < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int((time.monotonic() - t0) * 1000)
        start = time.perf_counter()
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(image, ts)
        spent_ms = (time.perf_counter() - start) * 1000
        frames += 1
        window_frames += 1
        if result.hand_landmarks:
            detections += 1
            hand_ms.append(spent_ms)
        else:
            empty_ms.append(spent_ms)

        now = time.monotonic()
        if now - window_start >= 10.0:
            fps = window_frames / (now - window_start)
            temp = cpu_temp_c()
            temp_s = f", {temp:.0f}C" if temp is not None else ""
            print(f"  t+{now-t0:5.0f}s  {fps:5.1f} fps{temp_s}, "
                  f"hand {detections / frames:4.0%}")
            window_start, window_frames = now, 0

    elapsed = time.monotonic() - t0
    landmarker.close()
    cap.release()

    fps = frames / elapsed if elapsed else 0.0
    found = detections / frames if frames else 0.0
    print(f"\n{w}x{h}: {fps:.1f} fps average over {elapsed:.0f}s "
          f"({frames} frames, hand found in {detections} = {found:.0%})")
    for name, ms in (("hand in view", hand_ms), ("no hand", empty_ms)):
        summary = latency_summary(ms)
        if summary:
            median, p95 = summary
            print(f"  processing, {name:12s}: median {median:5.1f} ms, "
                  f"p95 {p95:5.1f} ms  (ceiling ~{1000 / median:.0f} fps)")
    if found < 0.5:
        print("  WARNING: a hand was in view for under half the run, so this")
        print("  mostly measures the no-hand path. Re-run holding a hand up.")
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
