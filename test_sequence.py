"""Offline tests for the movement layer -- no camera, no MediaPipe.

Clips are synthesised, so these test the *transform*: a passing suite says the
same gesture yields the same vector at 30fps and at 15fps, not that any real
sign is recognisable. That still needs a camera and stage 4's data.

Standard library only, so it runs in CI.

Run:  python -m unittest test_sequence -v
"""
import math
import unittest

import hands
import location
import sequence
from test_hands import place

FACE = location.face_from_box(260, 100, 120, 160)


def clip(path, duration_ms=800, fps=30.0, face=None, two_handed=False,
         drop=()):
    """A synthetic clip of a hand following `path(u)` for u in [0, 1].

    `drop` lists fractions of the clip where tracking is lost, so a dropout
    can be tested without pretending it never happens.
    """
    count = max(2, int(round(duration_ms / 1000.0 * fps)) + 1)
    samples = []
    for i in range(count):
        u = i / (count - 1)
        t = duration_ms * u
        if any(abs(u - d) < 1e-9 for d in drop):
            samples.append(sequence.Sample(t, hands.feature_vector(None, None, face),
                                           None, None))
            continue
        cx, cy = path(u)
        dominant = place(cx, cy)
        nondominant = place(cx - 180, cy) if two_handed else None
        samples.append(sequence.Sample(
            t,
            hands.feature_vector(dominant, nondominant, face),
            hands.anchor(dominant),
            hands.anchor(nondominant),
        ))
    return samples


STILL = lambda u: (320.0, 300.0)
STRAIGHT_DOWN = lambda u: (320.0, 200.0 + 300.0 * u)
CIRCLE = lambda u: (320.0 + 100.0 * math.cos(2 * math.pi * u),
                    300.0 + 100.0 * math.sin(2 * math.pi * u))
ZIGZAG = lambda u: (320.0 + 120.0 * math.sin(3 * math.pi * u), 300.0)


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


class TestVectorShape(unittest.TestCase):

    def test_length_is_fixed_whatever_the_clip(self):
        for duration, fps in ((300, 60.0), (800, 30.0), (2400, 12.0)):
            vector = sequence.sign_vector(clip(STRAIGHT_DOWN, duration, fps))
            self.assertEqual(len(vector), sequence.SIGN_FLOATS)

    def test_empty_clip_is_zeros_not_an_error(self):
        self.assertEqual(sequence.sign_vector([]), [0.0] * sequence.SIGN_FLOATS)

    def test_single_sample_clip(self):
        self.assertEqual(len(sequence.sign_vector(clip(STILL)[:1])),
                         sequence.SIGN_FLOATS)

    def test_columns_match_the_vector(self):
        self.assertEqual(len(sequence.SIGN_COLUMNS), sequence.SIGN_FLOATS)
        self.assertEqual(len(set(sequence.SIGN_COLUMNS)), sequence.SIGN_FLOATS)

    def test_sampling_by_rate_is_much_smaller_than_sampling_everything(self):
        # The reason handshape and the moving parameters are sampled
        # differently: the naive version is 856 inputs against the few
        # thousand clips a collection session can realistically produce.
        naive = sequence.DYNAMIC_KEYFRAMES * hands.FEATURE_FLOATS
        self.assertLess(sequence.SIGN_FLOATS, naive / 2)


class TestFrameRateInvariance(unittest.TestCase):
    """The headline property: keyframes are placed by time, not by index."""

    def test_same_gesture_at_30fps_and_15fps(self):
        fast = sequence.sign_vector(clip(STRAIGHT_DOWN, 800, 30.0))
        slow = sequence.sign_vector(clip(STRAIGHT_DOWN, 800, 15.0))
        self.assertLess(max_diff(fast, slow), 0.05)

    def test_curved_path_survives_a_halved_frame_rate(self):
        fast = sequence.sign_vector(clip(CIRCLE, 900, 30.0))
        slow = sequence.sign_vector(clip(CIRCLE, 900, 15.0))
        self.assertLess(max_diff(fast, slow), 0.20)

    def test_resampling_by_index_would_not_have_worked(self):
        # The control. A clip whose samples are bunched at the start -- a
        # camera stalling mid-gesture -- has its midpoint in a completely
        # different place by index than by time.
        even = clip(STRAIGHT_DOWN, 800, 30.0, face=FACE)
        uneven = [s for i, s in enumerate(even) if i < 6 or i % 5 == 0]
        by_time = sequence.resample(uneven, 3)[1]
        by_index = uneven[len(uneven) // 2].features
        self.assertGreater(max_diff(by_time, by_index), 0.1)

    def test_keyframes_land_at_even_fractions_of_real_time(self):
        samples = clip(STRAIGHT_DOWN, 1000, 30.0, face=FACE)
        frames = sequence.resample(samples, 3)
        y_index = hands.LOCATION_OFFSET + 2       # dom_loc_y
        first, middle, last = (f[y_index] for f in frames)
        self.assertAlmostEqual(middle, (first + last) / 2.0, places=2)


class TestResampling(unittest.TestCase):

    def test_endpoints_are_the_real_endpoints(self):
        samples = clip(STRAIGHT_DOWN, 800, 30.0, face=FACE)
        frames = sequence.resample(samples, sequence.DYNAMIC_KEYFRAMES)
        self.assertLess(max_diff(frames[0], samples[0].features), 1e-9)
        self.assertLess(max_diff(frames[-1], samples[-1].features), 1e-9)

    def test_a_still_clip_resamples_to_copies_of_itself(self):
        samples = clip(STILL, 800, 30.0, face=FACE)
        for frame in sequence.resample(samples, sequence.DYNAMIC_KEYFRAMES):
            self.assertLess(max_diff(frame, samples[0].features), 1e-9)

    def test_zero_duration_clip_does_not_divide_by_zero(self):
        one = clip(STILL)[0]
        frozen = [sequence.Sample(0.0, one.features, one.dom, one.non)] * 5
        frames = sequence.resample(frozen, 4)
        self.assertEqual(len(frames), 4)

    def test_interpolated_rotations_stay_on_the_unit_circle(self):
        # Blending two unit vectors linearly gives the chord, not the arc; a
        # rotation halfway between two keyframes must still be a rotation.
        turning = clip(lambda u: (320.0, 300.0), 800, 30.0)
        turning = [sequence.Sample(
            s.t,
            hands.feature_vector(place(320, 300, rotation=math.radians(120 * i / (len(turning) - 1))), None),
            s.dom, s.non)
            for i, s in enumerate(turning)]
        for frame in sequence.resample(turning, sequence.DYNAMIC_KEYFRAMES):
            for cos_i, sin_i in hands.UNIT_PAIR_INDICES[:1]:
                self.assertAlmostEqual(
                    math.hypot(frame[cos_i], frame[sin_i]), 1.0, places=9)


class TestWhereThePositionLives(unittest.TestCase):
    """Worth pinning down, because it is easy to trip over.

    The frame vector normalizes hand position away by construction -- that is
    what makes handshape recognisable anywhere in the frame. So with no face
    detected, a hand crossing the whole frame produces *identical* frame
    vectors throughout, and the trajectory is the only thing that knows it
    moved. With a face, the location block carries absolute body position too.
    That is why sign_vector() keeps both, and why the anchors are passed
    separately from the features rather than read out of them.
    """

    def test_without_a_face_the_frame_vectors_are_identical(self):
        samples = clip(STRAIGHT_DOWN, 800, 30.0)
        self.assertLess(max_diff(samples[0].features, samples[-1].features), 1e-9)

    def test_but_the_trajectory_still_sees_the_movement(self):
        path, _ = sequence.trajectory(clip(STRAIGHT_DOWN, 800, 30.0), "dom")
        self.assertGreater(abs(path[-1][2] - path[0][2]), 1.0)

    def test_with_a_face_the_frame_vectors_differ(self):
        samples = clip(STRAIGHT_DOWN, 800, 30.0, face=FACE)
        self.assertGreater(max_diff(samples[0].features, samples[-1].features), 1.0)


class TestTrajectory(unittest.TestCase):

    def test_measured_in_hand_widths_not_pixels(self):
        # Same gesture at half the camera distance: every pixel measurement
        # halves, and the hand halves with it, so the path must not move.
        near, _ = sequence.trajectory(clip(STRAIGHT_DOWN, 800, 30.0), "dom")
        far_clip = clip(lambda u: (160.0, 100.0 + 150.0 * u), 800, 30.0)
        far_clip = [sequence.Sample(s.t, s.features,
                                    (s.dom[0], s.dom[1] / 2.0), s.non)
                    for s in far_clip]
        far, _ = sequence.trajectory(far_clip, "dom")
        self.assertLess(abs(near[-1][2] - far[-1][2]), 0.05)

    def test_starts_at_the_origin(self):
        path, _ = sequence.trajectory(clip(CIRCLE), "dom")
        self.assertAlmostEqual(path[0][1], 0.0, places=9)
        self.assertAlmostEqual(path[0][2], 0.0, places=9)

    def test_dropout_shortens_the_path_rather_than_teleporting_it(self):
        dropped = clip(STRAIGHT_DOWN, 800, 30.0, drop=(0.5,))
        path, coverage = sequence.trajectory(dropped, "dom")
        self.assertLess(coverage, 1.0)
        steps = [math.hypot(b[1] - a[1], b[2] - a[2])
                 for a, b in zip(path, path[1:])]
        self.assertLess(max(steps), 0.5)   # no jump to the origin

    def test_coverage_reports_how_much_was_tracked(self):
        _, full = sequence.trajectory(clip(STRAIGHT_DOWN), "dom")
        self.assertEqual(full, 1.0)
        _, none = sequence.trajectory(clip(STRAIGHT_DOWN), "non")
        self.assertEqual(none, 0.0)

    def test_absent_hand_gives_an_empty_path(self):
        path, coverage = sequence.trajectory(clip(STILL), "non")
        self.assertEqual(path, [])
        self.assertEqual(coverage, 0.0)


class TestMovementSummary(unittest.TestCase):
    """The distinctions ASL actually makes have to survive the summary."""

    def summarise(self, path_fn, **kw):
        path, _ = sequence.trajectory(clip(path_fn, **kw), "dom")
        return sequence.movement_summary(path)

    def test_a_straight_movement_reads_straight(self):
        self.assertGreater(self.summarise(STRAIGHT_DOWN)[3], 0.95)

    def test_a_circle_reads_curved(self):
        # Returns to where it started: net displacement near zero, path length
        # large. Straightness is what separates the two, and circular movement
        # is common enough in ASL that it has to be visible.
        summary = self.summarise(CIRCLE)
        self.assertLess(summary[3], 0.2)
        self.assertGreater(summary[2], 3.0)

    def test_a_still_hand_has_no_path(self):
        self.assertLess(self.summarise(STILL)[2], 0.05)

    def test_repetition_shows_up_as_reversals(self):
        # Many ASL signs carry repetition as part of their form, not as
        # emphasis, so a single pass and a repeated one must differ.
        self.assertGreaterEqual(self.summarise(ZIGZAG)[7], 2.0)
        self.assertEqual(self.summarise(STRAIGHT_DOWN)[7], 0.0)

    def test_jitter_is_not_a_reversal(self):
        jitter = lambda u: (320.0 + 1.5 * math.sin(40 * math.pi * u), 300.0)
        self.assertEqual(self.summarise(jitter)[7], 0.0)

    def test_direction_is_signed(self):
        up = self.summarise(lambda u: (320.0, 500.0 - 300.0 * u))
        down = self.summarise(STRAIGHT_DOWN)
        self.assertLess(up[1], 0.0)
        self.assertGreater(down[1], 0.0)

    def test_a_faster_sign_has_a_higher_peak_speed(self):
        quick = self.summarise(STRAIGHT_DOWN, duration_ms=300)
        slow = self.summarise(STRAIGHT_DOWN, duration_ms=1500)
        self.assertGreater(quick[6], slow[6])

    def test_everything_stays_bounded(self):
        wild = lambda u: (320.0 + 6000.0 * u, 300.0 - 6000.0 * u)
        summary = self.summarise(wild, duration_ms=60)
        self.assertLessEqual(summary[2], sequence.MAX_PATH)
        self.assertLessEqual(summary[6], sequence.MAX_SPEED)
        self.assertLessEqual(abs(summary[0]), sequence.MAX_EXTENT)

    def test_short_paths_summarise_to_zeros(self):
        self.assertEqual(sequence.movement_summary([]),
                         [0.0] * sequence.MOVEMENT_FLOATS_PER_HAND)


class TestSignsAreDistinguishable(unittest.TestCase):
    """End to end: clips that differ only in one parameter must differ."""

    def test_same_handshape_different_movement(self):
        # The contrast main cannot see at all, since one frame has no path.
        down = sequence.sign_vector(clip(STRAIGHT_DOWN, 800, 30.0, face=FACE))
        circle = sequence.sign_vector(clip(CIRCLE, 800, 30.0, face=FACE))
        self.assertGreater(max_diff(down, circle), 0.5)

    def test_same_movement_different_location(self):
        high = clip(lambda u: (320.0, 120.0 + 60.0 * u), 800, 30.0, face=FACE)
        low = clip(lambda u: (320.0, 420.0 + 60.0 * u), 800, 30.0, face=FACE)
        self.assertGreater(max_diff(sequence.sign_vector(high),
                                    sequence.sign_vector(low)), 0.5)

    def test_one_handed_and_two_handed_differ(self):
        one = sequence.sign_vector(clip(STRAIGHT_DOWN, 800, 30.0, face=FACE))
        two = sequence.sign_vector(clip(STRAIGHT_DOWN, 800, 30.0, face=FACE,
                                        two_handed=True))
        self.assertGreater(max_diff(one, two), 0.5)

    def test_reversed_direction_is_a_different_sign(self):
        down = sequence.sign_vector(clip(STRAIGHT_DOWN, 800, 30.0))
        up = sequence.sign_vector(clip(lambda u: (320.0, 500.0 - 300.0 * u),
                                       800, 30.0))
        self.assertGreater(max_diff(down, up), 0.5)


class TestSignBuffer(unittest.TestCase):

    def fill(self, buffer, samples):
        for s in samples:
            buffer.add(s.t, s.features, s.dom, s.non)

    def test_prunes_by_time_not_by_count(self):
        buffer = sequence.SignBuffer(window_ms=500)
        self.fill(buffer, clip(STRAIGHT_DOWN, 2000, 30.0))
        self.assertLessEqual(buffer.span_ms(), 500)
        self.assertGreater(len(buffer), 0)

    def test_a_faster_camera_keeps_more_samples_over_the_same_span(self):
        # The point of pruning by time: the window is the same real duration
        # on both, so the faster device simply has more evidence in it.
        fast, slow = sequence.SignBuffer(), sequence.SignBuffer()
        self.fill(fast, clip(STRAIGHT_DOWN, 800, 60.0))
        self.fill(slow, clip(STRAIGHT_DOWN, 800, 15.0))
        self.assertGreater(len(fast), len(slow))
        self.assertEqual(fast.span_ms(), slow.span_ms())

    def test_not_ready_until_both_floors_are_met(self):
        buffer = sequence.SignBuffer(min_samples=8, min_span_ms=250)

        too_few = sequence.SignBuffer(min_samples=8, min_span_ms=250)
        self.fill(too_few, clip(STRAIGHT_DOWN, 800, 5.0))    # slow camera
        self.assertFalse(too_few.ready())

        too_brief = sequence.SignBuffer(min_samples=8, min_span_ms=250)
        self.fill(too_brief, clip(STRAIGHT_DOWN, 100, 120.0))  # fast flicker
        self.assertFalse(too_brief.ready())

        self.fill(buffer, clip(STRAIGHT_DOWN, 800, 30.0))
        self.assertTrue(buffer.ready())

    def test_the_frame_rate_floor_matches_the_one_notes_derives_for_motion(self):
        # min_samples inside a real sign is what sets the hardware floor.
        # NOTES.md derives ~11fps for J/Z from motion.py's window; a 700ms
        # sign here needs the same, and it is a hardware constraint rather
        # than a tuning preference.
        buffer = sequence.SignBuffer(min_samples=8)
        self.assertAlmostEqual((8 - 1) * 1000.0 / 700.0, 10.0, places=0)
        self.fill(buffer, clip(STRAIGHT_DOWN, 700, 15.0))
        self.assertTrue(buffer.ready())

        starved = sequence.SignBuffer(min_samples=8)
        self.fill(starved, clip(STRAIGHT_DOWN, 700, 8.0))
        self.assertFalse(starved.ready())

    def test_vector_is_none_until_ready(self):
        buffer = sequence.SignBuffer()
        self.assertIsNone(buffer.vector())
        self.fill(buffer, clip(STRAIGHT_DOWN, 800, 30.0))
        self.assertEqual(len(buffer.vector()), sequence.SIGN_FLOATS)

    def test_clear_empties_it(self):
        buffer = sequence.SignBuffer()
        self.fill(buffer, clip(STRAIGHT_DOWN, 800, 30.0))
        buffer.clear()
        self.assertEqual(len(buffer), 0)
        self.assertIsNone(buffer.vector())

    def test_debug_info_reports_fill_before_it_is_ready(self):
        buffer = sequence.SignBuffer()
        self.assertIn("clip", buffer.debug_info())
        self.fill(buffer, clip(ZIGZAG, 900, 30.0))
        info = buffer.debug_info()
        self.assertIn("reversals", info)
        self.assertGreaterEqual(info["reversals"], 2)


if __name__ == "__main__":
    unittest.main()


class TestTrackingGaps(unittest.TestCase):
    """MediaPipe drops hands, and does it most during fast movement -- so
    dropouts cluster inside the gesture rather than spreading evenly."""

    def test_a_gap_no_longer_blends_toward_an_empty_hand(self):
        # Without bridging, a keyframe near a gap interpolates between a real
        # hand and an all-zero block: a half-scale hand at an impossible
        # position, blended out of a pose and an absence.
        samples = clip(STRAIGHT_DOWN, 800, 30.0, face=FACE, drop=(0.5,))
        filled = sequence.bridge_gaps(samples)

        gap = next(i for i, s in enumerate(samples)
                   if s.features[hands.DOMINANT_OFFSET] == 0.0)
        shape = slice(hands.DOMINANT_OFFSET + 1,
                      hands.DOMINANT_OFFSET + hands.PER_HAND_FLOATS)
        self.assertEqual(set(samples[gap].features[shape]), {0.0})
        self.assertNotEqual(set(filled[gap][shape]), {0.0})

    def test_the_held_geometry_is_the_last_one_actually_seen(self):
        samples = clip(STRAIGHT_DOWN, 800, 30.0, face=FACE, drop=(0.5,))
        filled = sequence.bridge_gaps(samples)
        gap = next(i for i, s in enumerate(samples)
                   if s.features[hands.DOMINANT_OFFSET] == 0.0)
        shape = slice(hands.DOMINANT_OFFSET + 1,
                      hands.DOMINANT_OFFSET + hands.PER_HAND_FLOATS)
        self.assertEqual(filled[gap][shape], filled[gap - 1][shape])

    def test_presence_flags_still_tell_the_truth(self):
        # The flag is deliberately not filled: after resampling it comes out
        # fractional across a gap, which is the signal that a stretch was
        # interpolated and should be trusted less.
        samples = clip(STRAIGHT_DOWN, 800, 30.0, face=FACE, drop=(0.5,))
        filled = sequence.bridge_gaps(samples)
        flags = [v[hands.DOMINANT_OFFSET] for v in filled]
        self.assertIn(0.0, flags)

    def test_a_gap_at_the_very_start_takes_the_first_real_geometry(self):
        samples = clip(STRAIGHT_DOWN, 800, 30.0, face=FACE, drop=(0.0,))
        filled = sequence.bridge_gaps(samples)
        shape = slice(hands.DOMINANT_OFFSET + 1,
                      hands.DOMINANT_OFFSET + hands.PER_HAND_FLOATS)
        self.assertNotEqual(set(filled[0][shape]), {0.0})
        self.assertEqual(filled[0][shape], filled[1][shape])

    def test_a_clean_clip_is_untouched(self):
        samples = clip(STRAIGHT_DOWN, 800, 30.0, face=FACE)
        for sample, filled in zip(samples, sequence.bridge_gaps(samples)):
            self.assertEqual(list(sample.features), filled)

    def test_a_dropout_barely_moves_the_sign_vector(self):
        # The payoff: a clip with a hole in it should still describe the same
        # sign, because the geometry either side of the hole is real.
        clean = sequence.sign_vector(clip(STRAIGHT_DOWN, 800, 30.0, face=FACE))
        holed = sequence.sign_vector(
            clip(STRAIGHT_DOWN, 800, 30.0, face=FACE, drop=(0.4, 0.5)))
        self.assertLess(max_diff(clean, holed), 0.35)

    def test_a_hand_that_is_never_seen_stays_absent(self):
        # Bridging must not invent a hand that was not in the clip at all.
        vector = sequence.sign_vector(clip(STRAIGHT_DOWN, 800, 30.0, face=FACE))
        block = slice(hands.NONDOMINANT_OFFSET,
                      hands.NONDOMINANT_OFFSET + hands.PER_HAND_FLOATS)
        self.assertEqual(set(vector[0:0]) | {0.0}, {0.0})
        samples = clip(STRAIGHT_DOWN, 800, 30.0, face=FACE)
        for filled in sequence.bridge_gaps(samples):
            self.assertEqual(set(filled[block]), {0.0})
