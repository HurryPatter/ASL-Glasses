"""Offline tests for the motion detector -- no camera, no MediaPipe.

Landmark streams are synthesised, so these test the *geometry rules*, not
MediaPipe's real-world behaviour: a passing suite says the travel gate
separates a translating hand from a stationary jittering one, not that it
holds for every real hand. Confirming that still needs a camera.

Standard library only, so it runs in CI.

Run:  python -m unittest test_motion -v
"""
import math
import unittest

from motion import MotionDetector

FPS = 30.0
STEP_MS = 1000.0 / FPS
HAND_SIZE = 0.15   # hand spans 15% of the frame, in normalized image units


class LM:
    """Stands in for a MediaPipe NormalizedLandmark (only .x/.y are read)."""

    def __init__(self, x, y):
        self.x = x
        self.y = y


# A pointing hand in hand-local units: wrist at origin, middle knuckle one
# unit "up" (image y grows downward, hence negative). Index extended, the
# other three curled -- i.e. _is_point_shape() is satisfied, which is also
# true of a real Q.
POINTING = {
    0: (0.0, 0.0),        # wrist
    9: (0.0, -1.0),       # middle knuckle -> defines the unit
    6: (0.2, -1.2),   8: (0.3, -2.0),      # index: pip, tip (extended)
    10: (0.0, -1.3), 12: (0.0, -0.9),      # middle (curled)
    14: (-0.2, -1.25), 16: (-0.15, -0.85), # ring (curled)
    18: (-0.4, -1.1), 20: (-0.35, -0.8),   # pinky (curled)
}


def hand(cx, cy, index_tip_offset=(0.0, 0.0)):
    """21 landmarks for a pointing hand centred at (cx, cy) in image units."""
    pts = []
    for i in range(21):
        lx, ly = POINTING.get(i, (0.0, -0.5))   # unused joints: somewhere sane
        x = cx + lx * HAND_SIZE
        y = cy + ly * HAND_SIZE
        if i == 8:
            x += index_tip_offset[0] * HAND_SIZE
            y += index_tip_offset[1] * HAND_SIZE
        pts.append(LM(x, y))
    return pts


def feed(detector, frames):
    """Push (landmarks) frames at a fixed frame rate; return what fired."""
    fired = []
    for i, landmarks in enumerate(frames):
        letter, start = detector.update(landmarks, i, i * STEP_MS)
        if letter:
            fired.append(letter)
    return fired


def genuine_z(n=18, extent=2.2):
    """A Z: stroke right, diagonal down-left, stroke right. The whole hand
    translates, which is what distinguishes it from fingertip noise."""
    frames = []
    legs = [((0.0, 0.0), (extent, 0.0)),            # top bar, rightward
            ((extent, 0.0), (0.0, 1.1)),            # diagonal, down-left
            ((0.0, 1.1), (extent, 1.1))]            # bottom bar, rightward
    per_leg = max(2, n // 3)
    for (x0, y0), (x1, y1) in legs:
        for k in range(per_leg):
            t = k / (per_leg - 1) if per_leg > 1 else 1.0
            cx = 0.35 + (x0 + (x1 - x0) * t) * HAND_SIZE
            cy = 0.35 + (y0 + (y1 - y0) * t) * HAND_SIZE
            frames.append(hand(cx, cy))
    return frames


def jittering_hold(n=18, swing=0.45, drift=0.6):
    """A stationary pointing hand whose index tip oscillates and drifts down.

    This is the palm-forward Q case: the hand is held still, but the occluded
    fingertip estimate swings far enough to look like alternating strokes.
    """
    frames = []
    for k in range(n):
        phase = (k // 3) % 2          # a few frames each way
        dx = swing if phase else -swing
        dy = drift * (k / (n - 1))    # slow downward drift of the tip
        frames.append(hand(0.5, 0.4, index_tip_offset=(dx, dy)))
    return frames


class TestSyntheticHandIsWhatWeThink(unittest.TestCase):
    """If the synthetic hand doesn't satisfy the shape test, nothing below
    is testing what it claims to."""

    def test_pointing_hand_passes_the_point_shape_gate(self):
        d = MotionDetector()
        feed(d, [hand(0.5, 0.4) for _ in range(10)])
        self.assertTrue(d._shape_held("point_shape"),
                        "synthetic hand must read as a pointing handshape")

    def test_stationary_hand_has_near_zero_travel(self):
        d = MotionDetector()
        feed(d, jittering_hold())
        self.assertLess(d.hand_travel(axis=0), 0.1,
                        "a hand held in place should barely move its centroid")

    def test_genuine_z_has_large_travel(self):
        d = MotionDetector(z_travel=99)   # keep it from firing and clearing
        feed(d, genuine_z())
        self.assertGreater(d.hand_travel(axis=0), 1.0)


class TestTravelGate(unittest.TestCase):

    def test_jittering_stationary_hand_fires_z_without_the_gate(self):
        # Establishes that the synthetic jitter really does reach the Z rules,
        # so the next test is demonstrating the gate and not a dud input.
        without = MotionDetector(z_travel=0.0)
        self.assertIn("Z", feed(without, jittering_hold()),
                      "synthetic jitter should satisfy the old Z rules")

    def test_the_gate_suppresses_it(self):
        d = MotionDetector()          # default z_travel
        self.assertEqual(feed(d, jittering_hold()), [])

    def test_genuine_z_still_fires_with_the_gate(self):
        d = MotionDetector()
        self.assertIn("Z", feed(d, genuine_z()))

    def test_gate_does_not_depend_on_frame_rate(self):
        # Same wall-clock gesture sampled at laptop and embedded rates. 15fps
        # is the lowest tested because of a floor that predates this gate --
        # see test_motion_signs_need_about_11fps_at_all below.
        for fps in (30.0, 20.0, 15.0):
            frames = genuine_z(n=max(9, int(0.6 * fps)))
            d = MotionDetector()
            step = 1000.0 / fps
            fired = []
            for i, lms in enumerate(frames):
                letter, _ = d.update(lms, i, i * step)
                if letter:
                    fired.append(letter)
            with self.subTest(fps=fps):
                self.assertIn("Z", fired, f"genuine Z should fire at {fps}fps")


class TestFrameRateFloor(unittest.TestCase):
    """Not caused by the travel gate -- documented because it constrains the
    hardware choice."""

    def test_motion_signs_need_about_11fps_at_all(self):
        # min_samples (8) must be met *inside* window_ms (650), so the camera
        # has to deliver 8 samples in 650ms: (8-1)*1000/650 ~= 10.8fps. Below
        # that, no J or Z can ever fire however the gesture is performed,
        # because the buffer is pruned by time before it fills.
        d = MotionDetector()
        floor_fps = (d.min_samples - 1) * 1000.0 / d.window_ms
        self.assertAlmostEqual(floor_fps, 10.77, places=1)

        for fps, should_fire in ((30.0, True), (15.0, True), (10.0, False)):
            frames = genuine_z(n=max(9, int(0.6 * fps)))
            det = MotionDetector()
            step = 1000.0 / fps
            fired = []
            for i, lms in enumerate(frames):
                letter, _ = det.update(lms, i, i * step)
                if letter:
                    fired.append(letter)
            with self.subTest(fps=fps):
                self.assertEqual(bool(fired), should_fire)


class TestTravelPrimitive(unittest.TestCase):

    def test_empty_buffer(self):
        self.assertEqual(MotionDetector().hand_travel(), 0.0)

    def test_axis_selection(self):
        d = MotionDetector(z_travel=99)
        # Move the hand purely vertically.
        feed(d, [hand(0.5, 0.2 + 0.02 * k) for k in range(12)])
        self.assertLess(d.hand_travel(axis=0), 0.05)
        self.assertGreater(d.hand_travel(axis=1), 1.0)

    def test_lookback_measures_only_the_tail(self):
        d = MotionDetector(z_travel=99)
        # Move a long way, then hold still for the last ~200ms.
        moving = [hand(0.2 + 0.03 * k, 0.4) for k in range(12)]
        still = [hand(0.56, 0.4) for _ in range(6)]
        feed(d, moving + still)
        self.assertGreater(d.hand_travel(axis=0), 1.0)
        self.assertLess(d.hand_travel(axis=0, lookback_ms=150), 0.1)


class TestDebugInfo(unittest.TestCase):

    def test_reports_hand_travel_for_tuning(self):
        d = MotionDetector(z_travel=99)
        feed(d, genuine_z())
        info = d.debug_info()
        self.assertIn("hand_travel", info)
        self.assertGreater(info["hand_travel"], 1.0)

    def test_buffer_status_before_enough_samples(self):
        d = MotionDetector()
        feed(d, [hand(0.5, 0.4) for _ in range(3)])
        self.assertIn("buffer", d.debug_info())


if __name__ == "__main__":
    unittest.main(verbosity=2)
