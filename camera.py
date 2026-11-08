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

USB cameras on Linux (the Pi, and the glasses' camera). Two things go wrong
with a bare cv2.VideoCapture(0):
  - On a Pi 5, /dev/video0 and the nodes after it belong to the Pi's own
    camera hardware, present even with no ribbon camera attached, so a USB
    camera arrives as /dev/video8 or later. usb_video_nodes() finds USB
    cameras by where they hang in sysfs, and `index` picks among those.
  - A webcam lengthens its exposure in dim light by quietly dropping to 15fps,
    below what J and Z need. The camera is asked for 640x480 MJPEG at 30fps,
    and its "dynamic frame rate" control is switched off with v4l2-ctl (apt:
    v4l-utils), so it brightens with gain instead, as the ribbon camera does.
    Cameras without that control, or without v4l2-ctl, are opened anyway; the
    check in `python camera.py` says which happened.

Frames are always BGR, as OpenCV and the rest of the pipeline expect.
Picamera2's "RGB888" format is, despite the name, stored B,G,R -- which is
why it is the format asked for below.

Low light. The ribbon camera's frame rate is pinned at 30fps (see PiCamera),
which caps each exposure at 33ms: in a dim room the camera cannot brighten
the image by exposing longer, as a webcam does by quietly dropping to 15fps.
It raises sensor gain instead, which is noisier but keeps J and Z working.
`python camera.py` reports whether that limit has been reached. If it has,
the fix is light on the hand; ASL_CAMERA_EV=1 (or 2) asks the camera for a
brighter image at the cost of more noise, and works only while there is gain
left to give.
"""
import os
import subprocess
import sys

import cv2

DEFAULT_SIZE = (640, 480)   # 4:3, matching the webcams the training data used
SYSFS_V4L = "/sys/class/video4linux"
# The control that lets a UVC camera slow its frame rate for longer exposures,
# under its current name and the one older kernels use.
DYNAMIC_FPS_CONTROLS = ("exposure_dynamic_framerate", "exposure_auto_priority")

# What the last open_camera() did for a USB camera, for `python camera.py`.
last_usb = {}


def requested_ev(env=None):
    """ASL_CAMERA_EV as a float, or None if unset or not a number."""
    env = os.environ if env is None else env
    try:
        return float(env["ASL_CAMERA_EV"])
    except (KeyError, ValueError):
        return None


def exposure_report(meta, frame_us, gain_range=None):
    """Plain-language verdict on brightness from Picamera2 frame metadata.

    The question that matters is whether a dark image is the camera's choice
    or its limit: an exposure at the frame-time cap means it is already
    collecting all the light 30fps allows.
    """
    exp = meta.get("ExposureTime")
    gain = meta.get("AnalogueGain")
    if exp is None or gain is None:
        return ["no exposure metadata from the camera"]
    lines = [f"exposure {exp / 1000:.1f} ms of {frame_us / 1000:.1f} ms per frame, "
             f"gain {gain:.1f}x" + (f" (sensor max {gain_range[1]:.0f}x)"
                                   if gain_range else "")]
    if "Lux" in meta:
        lines.append(f"scene brightness ~{meta['Lux']:.0f} lux "
                     f"(an evening room ~50, an office ~300-500)")
    if "ColourTemperature" in meta:
        lines.append(f"white balance set for ~{meta['ColourTemperature']:.0f}K "
                     f"(warm indoor bulbs ~2700-3000K, daylight ~5500-6500K)")
    if exp >= 0.9 * frame_us:
        lines.append("LIGHT-LIMITED: exposure is at the 30fps cap, so brightness "
                     "now costs noise. Add light on the hand (lamp, window); "
                     "ASL_CAMERA_EV=1 brightens further using gain.")
    else:
        lines.append("exposure has headroom: the camera is choosing this "
                     "brightness, so the scene is lit well enough.")
    return lines


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


def _linux():
    return sys.platform.startswith("linux")


def usb_video_nodes(sysfs=None):
    """/dev/videoN numbers of attached USB cameras, in order.

    A UVC camera registers two nodes, video capture and metadata; sysfs gives
    the capture one index 0. Nodes not under a USB device (the Pi 5's camera
    front end and ISP) are skipped.
    """
    sysfs = sysfs or SYSFS_V4L
    try:
        names = os.listdir(sysfs)
    except OSError:
        return []
    found = []
    for name in names:
        if not name.startswith("video") or not name[5:].isdigit():
            continue
        node = os.path.join(sysfs, name)
        if "/usb" not in os.path.realpath(os.path.join(node, "device")):
            continue
        try:
            with open(os.path.join(node, "index")) as fh:
                if fh.read().strip() != "0":
                    continue
        except OSError:
            pass
        found.append(int(name[5:]))
    return sorted(found)


def lock_frame_rate(device, run=subprocess.run):
    """Stop a UVC camera dropping its frame rate in dim light.

    Returns the control name that was set, or None if the camera has neither
    name or v4l2-ctl is not installed.
    """
    for control in DYNAMIC_FPS_CONTROLS:
        try:
            r = run(["v4l2-ctl", "-d", f"/dev/video{device}", "-c", f"{control}=0"],
                    capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        if r.returncode == 0:
            return control
    return None


def open_camera(width=None, height=None, fps=30, index=0, source=None):
    """Open a camera and return a VideoCapture-like object.

    width/height of None means 640x480 for a ribbon camera or a USB camera on
    Linux, and the camera's own default elsewhere (the Windows laptops the
    front-view data was collected on). `index` selects which camera of the
    chosen kind.
    """
    source = source or detect_source()
    if source == "csi":
        size = (width, height) if (width and height) else DEFAULT_SIZE
        return PiCamera(size, fps, index)

    if not _linux():
        cap = cv2.VideoCapture(index)
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    nodes = usb_video_nodes()
    device = nodes[index] if index < len(nodes) else index
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    w, h = (width, height) if (width and height) else DEFAULT_SIZE
    # MJPEG first: uncompressed 640x480 at 30fps is close to what a USB 2.0
    # camera can send, and many only offer 30fps in MJPEG.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    last_usb.clear()
    last_usb.update(device=device, found=nodes, locked=lock_frame_rate(device))
    return cap


def fourcc_text(cap):
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    if code <= 0:                       # not opened, or the backend cannot say
        return "?"
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip("\x00") or "?"


def _af_continuous():
    try:
        from libcamera import controls
        return controls.AfModeEnum.Continuous
    except Exception:
        return 2   # libcamera's value for Continuous


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
        self.frame_us = frame_us = int(1_000_000 / fps)
        controls = {"FrameDurationLimits": (frame_us, frame_us)}
        available = getattr(self._cam, "camera_controls", {}) or {}
        # Camera Module 3 has autofocus but starts at a fixed lens position;
        # a hand at arm's length can sit outside it. Earlier modules are
        # fixed-focus and do not list the control.
        if "AfMode" in available:
            controls["AfMode"] = _af_continuous()
        ev = requested_ev()
        if ev is not None and "ExposureValue" in available:
            lo, hi = available["ExposureValue"][:2]
            controls["ExposureValue"] = max(lo, min(hi, ev))
        self.controls = controls
        self.gain_range = available.get("AnalogueGain")
        config = self._cam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            controls=controls,
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

    def metadata(self):
        """Exposure, gain, light level etc. for the latest frame."""
        return self._cam.capture_metadata()

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
    # Saves camera_check.jpg (open it, or copy it off with scp) and, for a
    # ribbon camera, says whether a dark picture is the room or the camera.
    src = detect_source()
    print(f"camera source: {src}")
    if src == "csi":
        for i, info in enumerate(csi_cameras()):
            print(f"  ribbon camera {i}: {info}")
    cap = open_camera()
    # Auto-exposure, white balance and focus take a moment to settle; the
    # first frame is not representative.
    ok, frame = False, None
    for _ in range(45):
        ok, frame = cap.read()
    if not ok:
        cap.release()
        sys.exit("frame: NONE -- camera opened but gave no image")
    print("frame:", frame.shape)
    cv2.imwrite("camera_check.jpg", frame)
    print("saved camera_check.jpg")
    if last_usb:
        print(f"USB camera /dev/video{last_usb['device']} "
              f"(USB cameras found: {last_usb['found'] or 'none'}), "
              f"format {fourcc_text(cap)}, {cap.get(cv2.CAP_PROP_FPS):.0f}fps requested")
        if last_usb["locked"]:
            print(f"  frame rate locked ({last_usb['locked']}=0): it will not "
                  f"slow down in dim light")
        else:
            print("  could not lock the frame rate (no such control, or v4l2-ctl "
                  "missing: sudo apt install v4l-utils). Check fps in a dim room "
                  "with bench_pi.py.")
    if isinstance(cap, PiCamera):
        print("controls:", {k: v for k, v in cap.controls.items()
                            if k != "FrameDurationLimits"} or "defaults")
        for line in exposure_report(cap.metadata(), cap.frame_us, cap.gain_range):
            print(" ", line)
    cap.release()
