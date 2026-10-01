"""Offline tests for camera.py -- no camera, no Pi, no OpenCV, no numpy.

Small stand-ins for cv2 and picamera2 are installed for the duration of each
test, so what is checked is camera.py's own logic: which camera it picks, and
that a ribbon camera is configured and exposed through the VideoCapture-shaped
interface the rest of the pipeline relies on.
"""
import importlib
import os
import sys
import types
import unittest
from unittest import mock


class FakeArray:
    """Just enough of a numpy array: shape, ndim, and [:, :, :3]."""
    def __init__(self, shape):
        self.shape, self.ndim = tuple(shape), len(shape)

    def __getitem__(self, key):
        return FakeArray(self.shape[:2] + (3,))


def make_fake_cv2():
    cv2 = types.ModuleType("cv2")
    cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS = 3, 4, 5

    class VideoCapture:
        instances = []

        def __init__(self, index):
            self.index, self.props = index, {}
            VideoCapture.instances.append(self)

        def set(self, prop, value):
            self.props[prop] = value
            return True

    cv2.VideoCapture = VideoCapture
    return cv2


def make_fake_picamera2(cameras, channels=3, fail_capture=False):
    mod = types.ModuleType("picamera2")

    class Picamera2:
        instances = []

        def __init__(self, index=0):
            self.index = index
            self.config = None
            self.started = self.stopped = self.closed = False
            Picamera2.instances.append(self)

        @staticmethod
        def global_camera_info():
            return cameras

        def create_video_configuration(self, main, controls):
            return {"main": main, "controls": controls}

        def configure(self, cfg): self.config = cfg
        def start(self): self.started = True
        def stop(self): self.stopped = True
        def close(self): self.closed = True

        def capture_array(self, name="main"):
            if fail_capture:
                raise RuntimeError("camera timed out")
            w, h = self.config["main"]["size"]
            return FakeArray((h, w, channels))

    mod.Picamera2 = Picamera2
    return mod


class CameraTest(unittest.TestCase):

    def load(self, cameras=None, picamera2_installed=True, **fake):
        """Import camera.py with the stand-ins in place."""
        modules = {"cv2": make_fake_cv2()}
        if picamera2_installed:
            modules["picamera2"] = make_fake_picamera2(cameras or [], **fake)
        else:
            modules["picamera2"] = None      # makes `import picamera2` fail
        patcher = mock.patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        sys.modules.pop("camera", None)
        self.addCleanup(sys.modules.pop, "camera", None)
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start(); self.addCleanup(env.stop)
        os.environ.pop("ASL_CAMERA", None)
        cam = importlib.import_module("camera")
        return cam, modules["cv2"], modules.get("picamera2")


class TestWhichCamera(CameraTest):

    def test_ribbon_camera_is_preferred_when_present(self):
        cam, _, _ = self.load(cameras=[{"Model": "imx708"}])
        self.assertEqual(cam.detect_source(), "csi")

    def test_usb_when_no_ribbon_camera(self):
        cam, _, _ = self.load(cameras=[])
        self.assertEqual(cam.detect_source(), "usb")

    def test_usb_when_picamera2_not_installed(self):
        # The development laptops: no picamera2 at all.
        cam, _, _ = self.load(picamera2_installed=False)
        self.assertEqual(cam.detect_source(), "usb")
        self.assertEqual(cam.csi_cameras(), [])

    def test_env_override_beats_detection(self):
        cam, _, _ = self.load(cameras=[{"Model": "imx708"}])
        os.environ["ASL_CAMERA"] = "usb"
        self.assertEqual(cam.detect_source(), "usb")
        os.environ["ASL_CAMERA"] = "CSI"
        self.assertEqual(cam.detect_source(), "csi")

    def test_unknown_override_is_ignored(self):
        cam, _, _ = self.load(cameras=[])
        os.environ["ASL_CAMERA"] = "webcam"
        self.assertEqual(cam.detect_source(), "usb")


class TestUsb(CameraTest):

    def test_opens_videocapture_at_the_index(self):
        cam, cv2, _ = self.load(cameras=[])
        cap = cam.open_camera(index=2)
        self.assertIsInstance(cap, cv2.VideoCapture)
        self.assertEqual(cap.index, 2)

    def test_size_applied_only_when_given(self):
        cam, cv2, _ = self.load(cameras=[])
        self.assertEqual(cam.open_camera().props, {})
        cap = cam.open_camera(320, 240)
        self.assertEqual(cap.props[cv2.CAP_PROP_FRAME_WIDTH], 320)
        self.assertEqual(cap.props[cv2.CAP_PROP_FRAME_HEIGHT], 240)


class TestRibbonCamera(CameraTest):

    def open(self, *args, **fake):
        cam, cv2, pc = self.load(cameras=[{"Model": "imx708"}], **fake)
        return cam, cv2, cam.open_camera(*args), pc.Picamera2.instances[-1]

    def test_configured_for_bgr_at_640x480_by_default(self):
        _, _, cap, picam = self.open()
        main = picam.config["main"]
        self.assertEqual(main["size"], (640, 480))
        # Picamera2's "RGB888" is stored B,G,R -- what OpenCV expects.
        self.assertEqual(main["format"], "RGB888")
        self.assertTrue(picam.started)

    def test_requested_size_is_used(self):
        _, _, cap, picam = self.open(320, 240)
        self.assertEqual(picam.config["main"]["size"], (320, 240))

    def test_frame_rate_is_pinned(self):
        _, _, cap, picam = self.open()
        self.assertEqual(picam.config["controls"]["FrameDurationLimits"],
                         (33333, 33333))

    def test_read_returns_a_3_channel_frame(self):
        _, _, cap, _ = self.open()
        ok, frame = cap.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, (480, 640, 3))

    def test_padding_channel_is_dropped(self):
        _, _, cap, _ = self.open(channels=4)
        ok, frame = cap.read()
        self.assertEqual(frame.shape, (480, 640, 3))

    def test_a_failed_capture_is_reported_not_raised(self):
        # main.py's loop ends cleanly on ok=False; an exception would crash it.
        _, _, cap, _ = self.open(fail_capture=True)
        self.assertEqual(cap.read(), (False, None))

    def test_get_reports_the_size_like_videocapture(self):
        _, cv2, cap, _ = self.open(320, 240)
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 320.0)
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 240.0)
        self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 30.0)

    def test_set_is_refused_rather_than_silently_ignored(self):
        _, cv2, cap, _ = self.open()
        self.assertFalse(cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280))

    def test_release_stops_and_closes_once(self):
        _, _, cap, picam = self.open()
        self.assertTrue(cap.isOpened())
        cap.release()
        cap.release()
        self.assertTrue(picam.stopped and picam.closed)
        self.assertFalse(cap.isOpened())
        self.assertEqual(cap.read(), (False, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
