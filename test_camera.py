"""Offline tests for camera.py -- no camera, no Pi, no OpenCV, no numpy.

Small stand-ins for cv2 and picamera2 are installed for the duration of each
test, so what is checked is camera.py's own logic: which camera it picks, and
that a ribbon camera is configured and exposed through the VideoCapture-shaped
interface the rest of the pipeline relies on.
"""
import importlib
import os
import shutil
import sys
import tempfile
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
    cv2.CAP_PROP_FOURCC, cv2.CAP_V4L2 = 6, 200
    cv2.VideoWriter_fourcc = lambda *c: sum(ord(ch) << 8 * i for i, ch in enumerate(c))

    class VideoCapture:
        instances = []

        def __init__(self, index, api=None):
            self.index, self.api, self.props = index, api, {}
            VideoCapture.instances.append(self)

        def get(self, prop):
            return float(self.props.get(prop, 0))

        def set(self, prop, value):
            self.props[prop] = value
            return True

    cv2.VideoCapture = VideoCapture
    return cv2


# Camera Module 3 (imx708) lists autofocus; earlier modules do not.
CM3_CONTROLS = {"AfMode": (0, 2, 0), "ExposureValue": (-8.0, 8.0, 0.0),
                "AnalogueGain": (1.12, 16.0, None)}


def make_fake_picamera2(cameras, channels=3, fail_capture=False,
                        camera_controls=None, metadata=None):
    mod = types.ModuleType("picamera2")

    class Picamera2:
        instances = []

        def __init__(self, index=0):
            self.index = index
            self.camera_controls = dict(camera_controls or {})
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

        def capture_metadata(self):
            return dict(metadata or {})

    mod.Picamera2 = Picamera2
    return mod


class CameraTest(unittest.TestCase):

    def load(self, cameras=None, picamera2_installed=True, **fake):
        """Import camera.py with the stand-ins in place."""
        # libcamera absent: camera.py falls back to the plain enum value.
        modules = {"cv2": make_fake_cv2(), "libcamera": None}
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
        os.environ.pop("ASL_CAMERA_EV", None)
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


class TestUsbOffLinux(CameraTest):
    """The Windows laptops the front-view data was collected on: unchanged."""

    def load(self, **kw):
        cam, cv2, pc = super().load(**kw)
        p = mock.patch.object(cam, "_linux", return_value=False)
        p.start(); self.addCleanup(p.stop)
        return cam, cv2, pc

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


def fake_sysfs(root, nodes):
    """nodes: {video number: (device path, index)} as sysfs would show them."""
    for n, (device, idx) in nodes.items():
        d = os.path.join(root, f"video{n}")
        os.makedirs(d)
        os.symlink(device, os.path.join(d, "device"))
        with open(os.path.join(d, "index"), "w") as fh:
            fh.write(f"{idx}\n")


PI5_WITH_WEBCAM = {
    0: ("/sys/devices/platform/axi/1000880000.pcie/rp1-cfe", 0),
    1: ("/sys/devices/platform/axi/1000880000.pcie/rp1-cfe", 0),
    4: ("/sys/devices/platform/axi/1000880000.pcie/pispbe", 0),
    8: ("/sys/devices/platform/axi/xhci-hcd.0/usb1/1-1/1-1:1.0", 0),
    9: ("/sys/devices/platform/axi/xhci-hcd.0/usb1/1-1/1-1:1.0", 1),   # metadata
}


class TestUsbOnLinux(CameraTest):
    """The Pi: find the USB camera, ask for 640x480 MJPEG at 30fps, lock it."""

    def load(self, nodes=PI5_WITH_WEBCAM, lock_ok=True, **kw):
        cam, cv2, pc = super().load(cameras=[], **kw)
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root)
        fake_sysfs(root, nodes)
        self.calls = []

        def run(cmd, **_):
            self.calls.append(cmd)
            ok = lock_ok and cmd[-1].startswith("exposure_dynamic_framerate")
            return types.SimpleNamespace(returncode=0 if ok else 1)

        for name, value in (("_linux", lambda: True), ("SYSFS_V4L", root)):
            p = mock.patch.object(cam, name, value)
            p.start(); self.addCleanup(p.stop)
        real_lock = self.real_lock = cam.lock_frame_rate
        p = mock.patch.object(cam, "lock_frame_rate",
                              lambda dev: real_lock(dev, run=run))
        p.start(); self.addCleanup(p.stop)
        return cam, cv2, pc

    def test_pi5_camera_nodes_are_skipped(self):
        cam, _, _ = self.load()
        self.assertEqual(cam.usb_video_nodes(), [8])

    def test_opens_the_usb_camera_not_video0(self):
        cam, cv2, _ = self.load()
        cap = cam.open_camera()
        self.assertEqual((cap.index, cap.api), (8, cv2.CAP_V4L2))

    def test_asks_for_mjpeg_640x480_at_30fps(self):
        cam, cv2, _ = self.load()
        cap = cam.open_camera()
        self.assertEqual(cam.fourcc_text(cap), "MJPG")
        self.assertEqual(cap.props[cv2.CAP_PROP_FRAME_WIDTH], 640)
        self.assertEqual(cap.props[cv2.CAP_PROP_FRAME_HEIGHT], 480)
        self.assertEqual(cap.props[cv2.CAP_PROP_FPS], 30)

    def test_frame_rate_is_locked(self):
        cam, _, _ = self.load()
        cam.open_camera()
        self.assertEqual(self.calls[0], ["v4l2-ctl", "-d", "/dev/video8", "-c",
                                         "exposure_dynamic_framerate=0"])
        self.assertEqual(cam.last_usb["locked"], "exposure_dynamic_framerate")

    def test_older_control_name_is_tried_and_failure_is_reported(self):
        cam, _, _ = self.load(lock_ok=False)
        cam.open_camera()
        self.assertEqual([c[-1] for c in self.calls],
                         ["exposure_dynamic_framerate=0", "exposure_auto_priority=0"])
        self.assertIsNone(cam.last_usb["locked"])

    def test_missing_v4l2_ctl_is_not_fatal(self):
        cam, _, _ = self.load()

        def missing(*a, **k):
            raise FileNotFoundError("v4l2-ctl")
        self.assertIsNone(self.real_lock(8, run=missing))

    def test_index_picks_among_usb_cameras(self):
        nodes = dict(PI5_WITH_WEBCAM)
        nodes[10] = ("/sys/devices/platform/axi/xhci-hcd.1/usb3/3-1/3-1:1.0", 0)
        cam, _, _ = self.load(nodes=nodes)
        self.assertEqual(cam.open_camera(index=1).index, 10)

    def test_no_usb_camera_falls_back_to_the_index(self):
        cam, _, _ = self.load(nodes={0: ("/sys/devices/platform/rp1-cfe", 0)})
        self.assertEqual(cam.open_camera().index, 0)


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


class TestLowLight(CameraTest):

    def open(self, env=None, **fake):
        cam, _, _ = self.load(cameras=[{"Model": "imx708"}], **fake)
        os.environ.update(env or {})
        return cam, cam.open_camera()

    def test_autofocus_runs_continuously_when_the_camera_has_it(self):
        _, cap = self.open(camera_controls=CM3_CONTROLS)
        self.assertEqual(cap.controls["AfMode"], 2)

    def test_fixed_focus_camera_is_not_sent_autofocus(self):
        # Asking a camera for a control it lacks is an error in libcamera.
        _, cap = self.open(camera_controls={})
        self.assertNotIn("AfMode", cap.controls)

    def test_ev_is_off_unless_asked_for(self):
        _, cap = self.open(camera_controls=CM3_CONTROLS)
        self.assertNotIn("ExposureValue", cap.controls)

    def test_ev_is_applied_and_clamped(self):
        _, cap = self.open({"ASL_CAMERA_EV": "1.5"}, camera_controls=CM3_CONTROLS)
        self.assertEqual(cap.controls["ExposureValue"], 1.5)
        _, cap = self.open({"ASL_CAMERA_EV": "20"}, camera_controls=CM3_CONTROLS)
        self.assertEqual(cap.controls["ExposureValue"], 8.0)

    def test_bad_ev_is_ignored(self):
        _, cap = self.open({"ASL_CAMERA_EV": "bright"}, camera_controls=CM3_CONTROLS)
        self.assertNotIn("ExposureValue", cap.controls)

    def test_frame_rate_stays_pinned_whatever_else_is_set(self):
        _, cap = self.open({"ASL_CAMERA_EV": "1"}, camera_controls=CM3_CONTROLS)
        self.assertEqual(cap.controls["FrameDurationLimits"], (33333, 33333))

    def test_report_flags_exposure_at_the_frame_cap(self):
        cam, cap = self.open(camera_controls=CM3_CONTROLS,
                             metadata={"ExposureTime": 33000, "AnalogueGain": 9.5,
                                       "Lux": 40, "ColourTemperature": 6800})
        text = " ".join(cam.exposure_report(cap.metadata(), cap.frame_us,
                                            cap.gain_range))
        self.assertIn("LIGHT-LIMITED", text)
        self.assertIn("16x", text)
        self.assertIn("6800K", text)

    def test_report_says_when_there_is_headroom(self):
        cam, cap = self.open(metadata={"ExposureTime": 8000, "AnalogueGain": 1.2})
        text = " ".join(cam.exposure_report(cap.metadata(), cap.frame_us))
        self.assertNotIn("LIGHT-LIMITED", text)
        self.assertIn("headroom", text)

    def test_report_without_metadata(self):
        cam, cap = self.open()
        self.assertEqual(cam.exposure_report({}, cap.frame_us),
                         ["no exposure metadata from the camera"])
