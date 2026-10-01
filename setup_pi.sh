#!/usr/bin/env bash
# One-time setup on a Raspberry Pi 5. Run from inside the repo:
#
#     bash setup_pi.sh              # espeak-ng voice (robotic, works immediately)
#     bash setup_pi.sh --piper      # also install Piper (natural voice, ~100MB)
#
# Safe to re-run: every step is idempotent.
set -euo pipefail

WITH_PIPER=0
[[ "${1:-}" == "--piper" ]] && WITH_PIPER=1

say() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[[ -f requirements.txt && -f main.py ]] || die "run this from inside the ASL-Glasses folder"

# MediaPipe publishes arm64 wheels only. On 32-bit Pi OS, pip fails with an
# error that never mentions the real cause, so check up front.
arch="$(uname -m)"
[[ "$arch" == "aarch64" ]] || die "need 64-bit Raspberry Pi OS (this is '$arch'). Re-flash with the 64-bit image."

say "System packages"
# libegl1/libgles2: MediaPipe 1.0's native library needs libEGL even with no
#   display; a Lite image may lack it and the error does not say so.
# libgl1/libglib2.0-0: OpenCV.   espeak-ng: speech.   alsa-utils: aplay,
#   speaker-test, and the audio device list.
sudo apt-get update
# python3-picamera2/rpicam-apps: the ribbon (CSI) camera. Picamera2 is only
#   distributed through apt -- it binds to the system's libcamera -- so it
#   cannot be pip-installed into the venv; the venv is made able to see it.
sudo apt-get install -y python3-venv python3-pip git \
    libgl1 libglib2.0-0 libegl1 libgles2 \
    espeak-ng alsa-utils \
    python3-picamera2 rpicam-apps

say "Python environment (.venv)"
# Pi OS refuses system-wide pip ("externally-managed-environment"), so a venv
# is not optional. --system-site-packages lets it see the apt-installed
# Picamera2; the pinned packages below still install INTO the venv and take
# precedence over the system's copies.
if [[ -d .venv ]] && ! grep -q "include-system-site-packages = true" .venv/pyvenv.cfg; then
    echo "  existing .venv cannot see the ribbon-camera library; rebuilding it"
    rm -rf .venv
fi
[[ -d .venv ]] || python3 -m venv --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if (( WITH_PIPER )); then
    say "Piper neural voice"
    pip install piper-tts
    mkdir -p voices
    python -m piper.download_voices en_US-lessac-medium --download-dir voices
fi

say "Checking the install"
python - <<'PY'
import platform, sys, warnings
try:
    codename = dict(l.strip().split("=", 1) for l in open("/etc/os-release")
                    if "=" in l).get("VERSION_CODENAME", "?").strip('"')
except Exception:
    codename = "?"
print(f"  Pi OS {codename} | Python {platform.python_version()}")
import cv2, mediapipe, numpy, sklearn, symspellpy, joblib
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    joblib.load("landmark_model.joblib")
bad = [x for x in w if "Version" in type(x.message).__name__]
print(f"  mediapipe {mediapipe.__version__} | opencv {cv2.__version__} | "
      f"numpy {numpy.__version__} | scikit-learn {sklearn.__version__}")
print("  model loads cleanly" if not bad else
      "  WARNING: model/scikit-learn version mismatch -- see requirements.txt")
import nlp_bridge; nlp_bridge.NLPBridge()
print("  spell-correction dictionary loads")
import audio; print(f"  speech backend: {audio.detect_backend()}")

# Ribbon camera. Importing picamera2 under the venv's pinned numpy is the one
# place an incompatibility could surface (its compiled helpers are built
# against the system's numpy), so it is tried explicitly and reported plainly.
try:
    from picamera2 import Picamera2
    cams = Picamera2.global_camera_info()
    if cams:
        for i, c in enumerate(cams):
            print(f"  ribbon camera {i}: {c.get('Model', c)}")
    else:
        print("  no ribbon camera detected (USB webcam will be used if present)")
except Exception as e:
    print(f"  WARNING: ribbon-camera library failed to load: {type(e).__name__}: {e}")
    print("           send this line to be fixed; a USB webcam will still work")
import camera; print(f"  camera that will be used: {camera.detect_source()}")
PY

say "Done"
cat <<EOF

Every new terminal needs:   source .venv/bin/activate

Next:
  python camera.py                   # does the camera give a picture?
  speaker-test -t wav -c 2 -l 1      # can you hear the speaker at all?
  python audio.py "hello"            # does speech work?
  python bench_pi.py --seconds 120   # THE number: frames per second
EOF
if (( WITH_PIPER )); then
cat <<EOF

To use the Piper voice, add this line to ~/.bashrc (then open a new terminal):
  export ASL_PIPER_MODEL=$PWD/voices/en_US-lessac-medium.onnx
EOF
fi
