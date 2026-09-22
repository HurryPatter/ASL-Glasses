"""Offline tests for the hardware-emulation profile. Standard library only."""
import unittest

import hwprofile


class TestFrameLimiter(unittest.TestCase):

    def feed(self, limiter, camera_fps, seconds):
        """Simulate a camera delivering frames; return how many got processed."""
        step = 1000.0 / camera_fps
        n = int(camera_fps * seconds)
        return sum(1 for i in range(n) if limiter.should_process(i * step))

    def test_no_cap_processes_everything(self):
        lim = hwprofile.FrameLimiter(None)
        self.assertEqual(self.feed(lim, 30, 2), 60)
        self.assertEqual(lim.processed, 60)

    def test_cap_below_camera_rate_drops_frames(self):
        lim = hwprofile.FrameLimiter(10)
        kept = self.feed(lim, 30, 3)
        self.assertAlmostEqual(kept / 3, 10, delta=1)

    def test_cap_above_camera_rate_is_a_no_op(self):
        # Asking for 60fps from a 30fps camera cannot invent frames.
        lim = hwprofile.FrameLimiter(60)
        self.assertEqual(self.feed(lim, 30, 2), 60)

    def test_achieved_fps_reports_what_got_through(self):
        lim = hwprofile.FrameLimiter(12)
        self.feed(lim, 30, 4)
        self.assertAlmostEqual(lim.achieved_fps(), 12, delta=1.5)

    def test_capture_fps_reports_the_camera_not_the_cap(self):
        # If the camera itself is the bottleneck, no faster board helps, so
        # the two rates are recorded separately.
        lim = hwprofile.FrameLimiter(30)
        self.feed(lim, 10, 3)
        self.assertAlmostEqual(lim.capture_fps(), 10, delta=1)
        self.assertAlmostEqual(lim.achieved_fps(), 10, delta=1)

    def test_first_frame_always_processed(self):
        lim = hwprofile.FrameLimiter(1)
        self.assertTrue(lim.should_process(0.0))

    def test_no_frames_at_all(self):
        lim = hwprofile.FrameLimiter(10)
        self.assertEqual(lim.achieved_fps(), 0.0)
        self.assertEqual(lim.capture_fps(), 0.0)


class TestMotionFloor(unittest.TestCase):

    def test_floor_matches_the_motion_detector_settings(self):
        # min_samples must be met inside window_ms, so 8 samples in 650ms.
        self.assertAlmostEqual(hwprofile.motion_floor_fps(), 10.77, places=1)

    def test_floor_moves_with_its_inputs(self):
        self.assertAlmostEqual(hwprofile.motion_floor_fps(8, 1300), 5.38, places=1)

    def test_agrees_with_motion_py(self):
        """If MotionDetector's defaults change, this floor must change with it."""
        try:
            from motion import MotionDetector
        except Exception:
            self.skipTest("motion.py not importable")
        d = MotionDetector()
        self.assertAlmostEqual(
            hwprofile.motion_floor_fps(d.min_samples, d.window_ms),
            hwprofile.motion_floor_fps(), places=6)


class TestDescribe(unittest.TestCase):

    def test_labels(self):
        self.assertEqual(hwprofile.describe(320, 240, 15), "320x240@15fps")
        self.assertEqual(hwprofile.describe(None, None, None), "native@uncapped")
        self.assertEqual(hwprofile.describe(640, 480, None), "640x480@uncapped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
