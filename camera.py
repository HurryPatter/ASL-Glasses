"""Open whichever camera is attached: a Pi ribbon camera, or a USB webcam.

Every script used cv2.VideoCapture(0). That works for a USB webcam but cannot
see a ribbon (CSI) camera on current Raspberry Pi OS, which drives those
through libcamera; on a Pi 5, /dev/video0 is the raw sensor front end, so
VideoCapture either fails or returns frames that are not images. The supported
route is Picamera2.

open_camera() returns an object with the subset of the cv2.VideoCapture
interface the pipeline uses -- read(), isOpened(), get(), set(), release() --
so callers do not care which kind of camera is behind it.

Which camera, in order:
  1. ASL_CAMERA=csi or ASL_CAMERA=usb, if set
  2. a ribbon camera, if Picamera2 is installed and reports one
  3. a USB webcam via OpenCV

Frames are always BGR, as OpenCV and the rest of the pipeline expect.
Picamera2's "RGB888" format is, despite the name, stored B,G,R -- which is
why it is the format asked for below.
"""
import os

import cv2

DEFAULT_SIZE = (640, 480)   # 4:3, matching the webcams the training data used


def csi_cameras():
    """Ribbon cameras Picamera2 can see; empty if none or not installed."""
    try:
        from picamera2 import Picamera2
        return Picamera2.global_camera_info()
    except Exception:
        return []


def detect_source():
    forced = os.environ.get("ASL_CAMERA", "").strip().lower()
    if forced in ("csi", "usb"):
        return forced
    return "csi" if csi_cameras() else "usb"


def open_camera(width=None, height=None, fps=30, index=0, source=None):
    """Open a camera and return a VideoCapture-like object.

    width/height of None means the camera's default for USB, or 640x480 for
    a ribbon camera. `index` selects which camera of the chosen kind.
    """
    source = source or detect_source()
    if source == "csi":
        size = (width, height) if (width and height) else DEFAULT_SIZE
        return PiCamera(size, fps, index)

    cap = cv2.VideoCapture(index)
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


class PiCamera:
    """A Picamera2 camera behind the cv2.VideoCapture interface."""

    def __init__(self, size=DEFAULT_SIZE, fps=30, index=0):
        from picamera2 import Picamera2   # only needed, and only present, on a Pi

        self.width, self.height = int(size[0]), int(size[1])
        self.fps = fps
        self._cam = Picamera2(index)
        # Fixing the frame duration pins the frame rate, so the camera does
        # not quietly slow down in dim light and starve the motion detector,
        # which needs a steady ~30fps for J and Z.
        frame_us = int(1_000_000 / fps)
        config = self._cam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls={"FrameDurationLimits": (frame_us, frame_us)},
        )
        self._cam.configure(config)
        self._cam.start()
        self._open = True

    def isOpened(self):
        return self._open

    def read(self):
        if not self._open:
            return False, None
        try:
            frame = self._cam.capture_array("main")
        except Exception:
            return False, None
        if frame.ndim == 3 and frame.shape[2] == 4:   # some formats carry padding
            frame = frame[:, :, :3]
        return True, frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def set(self, prop, value):
        # Size and rate are fixed when the camera starts; pass them to
        # open_camera() instead. Returning False matches VideoCapture's
        # behaviour for a property it could not change.
        return False

    def release(self):
        if self._open:
            self._open = False
            try:
                self._cam.stop()
            finally:
                self._cam.close()


if __name__ == "__main__":
    # Quick check on the Pi:  python camera.py
    src = detect_source()
    print(f"camera source: {src}")
    if src == "csi":
        for i, info in enumerate(csi_cameras()):
            print(f"  ribbon camera {i}: {info}")
    cap = open_camera()
    ok, frame = cap.read()
    print("frame:", frame.shape if ok else "NONE -- camera opened but gave no image")
    cap.release()
