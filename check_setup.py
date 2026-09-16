"""
Preflight check for Project Veronica — run this before collecting anything.

Everything downstream of a collection session is expensive to redo, and two of
the failure modes in this pipeline do not announce themselves: a wrong
handedness convention labels every left hand right, and a camera below the
frame-rate floor silently produces clips no sign can be assembled from. Both
are cheap to check now and costly to discover in the data.

    python check_setup.py              # everything, including camera
    python check_setup.py --no-camera  # offline checks only
    python check_setup.py --skip-hands # camera checks without the interactive part

Exit status is non-zero if anything failed, so it can gate a script.
"""
import argparse
import os
import subprocess
import sys
import time

import config

# Named here rather than imported, because this script has to run and report
# useful failures on a machine where MediaPipe is not installed at all -- which
# is exactly the state it exists to diagnose.
FACE_MODEL = "blaze_face_short_range.tflite"
HAND_MODEL = "hand_landmarker.task"
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                  "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")

# motion.py needs 8 samples inside a 650ms window and sequence.py needs 8
# inside a sign; both land on about 11fps, derived independently. Below this
# the motion letters stop existing and clips stop being assemblable.
FPS_FLOOR = 11.0
FPS_COMFORTABLE = 15.0

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
MARK = {PASS: "  ok  ", WARN: " warn ", FAIL: " FAIL "}

results = []


def record(status, name, detail="", fix=""):
    results.append((status, name, detail, fix))
    print(f"[{MARK[status]}] {name}" + (f" -- {detail}" if detail else ""))
    if fix and status != PASS:
        for line in fix.strip().splitlines():
            print(f"           {line}")
    return status == PASS


# ── offline ────────────────────────────────────────────────────────────────
def check_python():
    version = sys.version_info
    detail = f"{version.major}.{version.minor}.{version.micro}"
    if version < (3, 9):
        return record(FAIL, "Python version", detail,
                      "Veronica uses 3.9+ syntax. Install a newer Python.")
    return record(PASS, "Python version", detail)


def check_packages():
    needed = {
        "cv2": "opencv-python", "mediapipe": "mediapipe", "numpy": "numpy",
        "sklearn": "scikit-learn", "joblib": "joblib",
        "symspellpy": "symspellpy", "pandas": "pandas",
    }
    missing = []
    for module, package in needed.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        return record(FAIL, "Python packages", f"missing {', '.join(missing)}",
                      "pip install -r requirements.txt")
    return record(PASS, "Python packages", f"all {len(needed)} present")


def check_models():
    ok = True
    if os.path.exists(HAND_MODEL):
        record(PASS, "Hand landmarker model", HAND_MODEL)
    else:
        ok = record(FAIL, "Hand landmarker model", f"{HAND_MODEL} not found",
                    "It is committed to the repo -- check you are in the "
                    "repository root.")
    if os.path.exists(FACE_MODEL):
        record(PASS, "Face detector model", FACE_MODEL)
    else:
        ok = record(FAIL, "Face detector model", f"{FACE_MODEL} not found",
                    f"Stage 2 needs a face to anchor sign location against, or\n"
                    f"FATHER and MOTHER become the same sign. Download it:\n"
                    f"  curl -LO {FACE_MODEL_URL}")
    return ok


def check_offline_tests():
    """The suite that needs no camera and no model.

    Worth running here rather than assuming: it is the only thing that
    confirms the feature layers on *this* machine behave as they do in CI.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-p", "test_*.py"],
        capture_output=True, text=True)
    tail = completed.stderr.strip().splitlines()
    count = next((l for l in tail if l.startswith("Ran ")), "")
    if completed.returncode == 0:
        return record(PASS, "Offline test suite", count or "passed")
    return record(FAIL, "Offline test suite", count or "failed",
                  "Run `python -m unittest discover -p \"test_*.py\" -v` "
                  "to see which.")


# ── camera ─────────────────────────────────────────────────────────────────
def open_camera(index):
    import cv2
    cam = cv2.VideoCapture(index)
    if not cam.isOpened():
        record(FAIL, "Camera", f"could not open camera {index}",
               "Check it is connected and not in use by another application.\n"
               "Try --camera 1 if you have more than one.")
        return None
    ok, frame = cam.read()
    if not ok:
        cam.release()
        record(FAIL, "Camera", "opened but returned no frame")
        return None
    height, width = frame.shape[:2]
    record(PASS, "Camera", f"{width}x{height}")
    return cam


def build_detectors():
    """The same detectors the collector and the demo build, via the same
    adapter -- checking a different code path than the one that will run would
    make this reassuring rather than useful."""
    import capture
    return capture.build_landmarker(num_hands=2), capture.build_face_detector()


def measure(cam, landmarker, detector, seconds, prompt=None,
            mirrored_input=True):
    """Run the real pipeline and collect what it saw.

    The camera is `cam`, not `capture`: `capture` is the MediaPipe adapter
    module this function imports, and a parameter of the same name shadows it
    the moment the import runs.
    """
    import capture
    import cv2
    import hands as hands_module

    start = time.monotonic()
    frames = 0
    handedness = []
    sides = []
    faces = 0
    two_handed = 0
    hand_frames = 0
    dominant_frames = 0

    while time.monotonic() - start < seconds:
        ok, frame = cam.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)             # selfie view, as main.py does
        frames += 1
        timestamp_ms = int((time.monotonic() - start) * 1000)
        image = capture.to_mp_image(frame)

        result = landmarker.detect_for_video(image, timestamp_ms)
        found = capture.detected_hands(result)
        if found:
            if len(found) == 2:
                two_handed += 1
            handedness += [label for _, label in found if label]
            if len(found) == 1:
                # Which side of the *displayed* frame the hand appeared on.
                # This is the independent signal: the person says which hand
                # they raised, so where it lands tells us whether the feed
                # reaching MediaPipe is mirrored -- which is the difference
                # between two root causes with different fixes.
                sides.append(found[0][0][0][0])

            # The question that actually matters: with the convention now in
            # force, does the hand they raised land in the DOMINANT block?
            # Run it through the same capture.scene() the collector uses,
            # rather than re-deriving the rule here -- a check that
            # reimplements what it is checking can agree with itself and
            # still be wrong.
            height, width = frame.shape[:2]
            dom, _, _ = capture.scene(found, None, width, height,
                                      signer_dominant=hands_module.RIGHT,
                                      mirrored_input=mirrored_input)
            hand_frames += 1
            dominant_frames += int(dom is not None)
        if detector.detect_for_video(image, timestamp_ms).detections:
            faces += 1

        if prompt:
            cv2.putText(frame, prompt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 220, 0), 2)
            remaining = seconds - (time.monotonic() - start)
            cv2.putText(frame, f"{remaining:.0f}s", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow("Veronica setup check", frame)
            cv2.waitKey(1)

    elapsed = time.monotonic() - start
    return {
        "fps": frames / elapsed if elapsed else 0.0,
        "frames": frames,
        "handedness": handedness,
        "sides": sides,
        "faces": faces,
        "two_handed": two_handed,
        "hand_frames": hand_frames,
        "dominant_frames": dominant_frames,
        "mirrored_input": mirrored_input,
    }


def check_throughput(stats):
    fps = stats["fps"]
    detail = f"{fps:.1f} fps with both models running"
    if fps < FPS_FLOOR:
        return record(FAIL, "Pipeline throughput", detail,
                      f"Below the {FPS_FLOOR:.0f}fps floor. A sign needs 8 samples "
                      f"inside it, so\nbelow this no clip can be assembled and no "
                      f"motion letter can fire.\nClose other applications, lower "
                      f"the camera resolution, or use\na faster machine before "
                      f"collecting anything.")
    if fps < FPS_COMFORTABLE:
        return record(WARN, "Pipeline throughput", detail,
                      f"Above the floor but below {FPS_COMFORTABLE:.0f}fps, so there "
                      f"is no headroom.\nA slower moment during a session will drop "
                      f"clips.")
    return record(PASS, "Pipeline throughput", detail)


def check_face(stats):
    if not stats["frames"]:
        return record(FAIL, "Face detection", "no frames captured")
    ratio = stats["faces"] / stats["frames"]
    detail = f"face found in {ratio:.0%} of frames"
    if ratio < 0.5:
        return record(FAIL, "Face detection", detail,
                      "Location features will be mostly empty, so signs "
                      "differing only in\nwhere they are made become the same "
                      "class. Sit facing the camera,\nfully in frame, with the "
                      "light in front of you rather than behind.")
    if ratio < 0.9:
        return record(WARN, "Face detection", detail,
                      "Intermittent. Check framing and lighting.")
    return record(PASS, "Face detection", detail)


def check_handedness(stats):
    """The hazard nothing downstream can detect, settled against a real hand.

    Two independent facts are needed, and reporting only the first is what
    made the earlier version of this check ambiguous:

      1. **What MediaPipe called the hand.** Its handedness is reported
         relative to an assumed mirroring of the input.
      2. **Which side of the displayed frame the hand appeared on.** The
         person says they raised their right hand, so this says whether the
         feed reaching MediaPipe is actually mirrored.

    Together they separate two root causes that look identical from (1) alone:
    a MediaPipe whose convention runs the other way, and a camera that already
    delivers a mirrored feed which cv2.flip then un-mirrors. Either way the
    fix is the same flag -- but only the pair tells you what you are looking
    at, and the second matters on the glasses, where the camera changes.
    """
    labels = stats["handedness"]
    if not labels:
        return record(FAIL, "Handedness convention", "no hand detected",
                      "Hold your hand clearly in frame and run this again.")

    rights = labels.count("Right")
    ratio = rights / len(labels)
    reported = "Right" if ratio > 0.5 else "Left"
    agreement = max(ratio, 1 - ratio)

    sides = stats.get("sides") or []
    mean_x = sum(sides) / len(sides) if sides else None
    if mean_x is None:
        where = "unknown (needed exactly one hand in frame)"
        view = None
    elif mean_x > 0.5:
        where = f"the RIGHT side of the frame (x={mean_x:.2f})"
        view = "mirrored"
    else:
        where = f"the LEFT side of the frame (x={mean_x:.2f})"
        view = "not mirrored"

    detail = f"MediaPipe said '{reported}' on {agreement:.0%} of frames; "
    detail += f"your hand was on {where}"

    convention = stats.get("mirrored_input", True)
    hand_frames = stats.get("hand_frames", 0)
    landed = stats.get("dominant_frames", 0)
    resolved = landed / hand_frames if hand_frames else 0.0

    detail = (f"raw label '{reported}' ({agreement:.0%}), hand on {where}; "
              f"mirrored_input={convention} -> "
              f"lands in the DOMINANT block {resolved:.0%} of the time")

    if agreement < 0.8:
        record(WARN, "Handedness convention", detail,
               "The raw label is mixed, which is itself a finding -- MediaPipe "
               "flips it, most\nreadily when the palm turns away. Clip "
               "reconstruction resolves identity over\na whole clip rather than "
               "per frame, so this is handled, but re-run with only\nyour right "
               "hand raised to read the convention cleanly.")
        return False

    if resolved > 0.8:
        # The raw label being 'Left' is not a problem once the convention
        # accounts for it -- what matters is where the hand ends up.
        note = ""
        if reported != "Right":
            note = (f"MediaPipe calls your right hand '{reported}' on this "
                    f"build, and the\nconvention is set to compensate. That is "
                    f"working as intended.")
        record(PASS, "Handedness convention", detail, note)
        return True

    cause = ""
    if view == "mirrored":
        cause = ("The displayed feed IS mirrored -- your right hand is on the "
                 "right --\nso MediaPipe's convention runs the opposite way on "
                 "this build.")
    elif view == "not mirrored":
        cause = ("Your right hand appeared on the LEFT of the frame, so this "
                 "camera\nalready delivers a mirrored feed and cv2.flip "
                 "un-mirrors it. Expect a\ndifferent answer on other hardware.")

    return record(FAIL, "Handedness convention", detail,
                  f"Your right hand is NOT landing in the dominant block.\n"
                  f"{cause}\n\nFlip the convention:\n"
                  f"    python check_setup.py --set-mirrored-input "
                  f"{'false' if convention else 'true'}\n"
                  f"then run this again. The value is recorded into every clip, "
                  f"so if it is\never wrong again `python rebuild_signs.py "
                  f"--mirrored-input <value>` re-derives\nevery training row "
                  f"from the raw landmarks. Nobody re-signs.")


def check_two_hands(stats):
    if stats["two_handed"]:
        return record(PASS, "Two-hand tracking",
                      f"both hands seen in {stats['two_handed']} frames")
    return record(WARN, "Two-hand tracking", "never saw two hands at once",
                  "Only one hand was in frame during the check, which is fine.\n"
                  "Confirm with both hands up before a real session -- about "
                  "half\nthe ASL lexicon is two-handed.")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--skip-hands", action="store_true",
                        help="skip the interactive handedness check")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--set-mirrored-input", choices=["true", "false"],
                        help="write the handedness convention to "
                             f"{config.CONFIG_PATH} and exit")
    args = parser.parse_args()

    if args.set_mirrored_input:
        value = args.set_mirrored_input == "true"
        saved = config.save({"mirrored_input": value})
        print(f"Wrote {config.CONFIG_PATH}: {config.describe(saved)}")
        print("collect_signs.py and demo_veronica.py now use this.")
        sys.exit(0)

    print("Project Veronica -- setup check\n")
    print(f"Convention: {config.describe()}\n")
    print("Offline")
    print("-" * 60)
    check_python()
    packages_ok = check_packages()
    models_ok = check_models()
    check_offline_tests()

    if not args.no_camera and packages_ok and models_ok:
        print("\nCamera")
        print("-" * 60)
        import cv2
        cam = open_camera(args.camera)
        if cam is not None:
            landmarker, detector = build_detectors()
            try:
                effective = config.mirrored_input()
                if args.skip_hands:
                    stats = measure(cam, landmarker, detector, args.seconds,
                                    mirrored_input=effective)
                else:
                    print("\n  >>> Hold up your RIGHT hand, facing the camera.")
                    print("  >>> Capturing for "
                          f"{args.seconds:.0f} seconds...\n")
                    stats = measure(cam, landmarker, detector, args.seconds,
                                    prompt="Hold up your RIGHT hand",
                                    mirrored_input=effective)
                check_throughput(stats)
                check_face(stats)
                if not args.skip_hands:
                    check_handedness(stats)
                    check_two_hands(stats)
            finally:
                landmarker.close()
                detector.close()
                cam.release()
                cv2.destroyAllWindows()
    elif not args.no_camera:
        print("\nCamera checks skipped -- fix the failures above first.")

    print("\n" + "=" * 60)
    failed = [r for r in results if r[0] == FAIL]
    warned = [r for r in results if r[0] == WARN]
    print(f"{len(results) - len(failed) - len(warned)} passed, "
          f"{len(warned)} warning, {len(failed)} failed")

    if failed:
        print("\nFix these before collecting:")
        for _, name, detail, _ in failed:
            print(f"  - {name}: {detail}")
        sys.exit(1)

    print("\nReady. Next:")
    print("  python demo_veronica.py    # live pipeline, no model needed yet")
    print("  python collect_signs.py    # record clips")
    sys.exit(0)


if __name__ == "__main__":
    main()
