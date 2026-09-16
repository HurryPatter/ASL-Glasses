"""Offline tests for the two-hand feature layer -- no camera, no MediaPipe.

Standard library only, so it runs in CI alongside the existing suites.

The interesting test here is `TestMatchesRecordedData`. `hands.hand_frame()` is
a from-scratch standard-library port of the numpy `normalize_landmarks()` that
produced all 24,497 existing rows, and a silent disagreement between the two
would invalidate the dataset and the trained model at once -- exactly the
failure README.md warns about when it says to keep the normalization
consistent. CI has no numpy to compare against, but it does have the numpy
implementation's *output*, checked into landmark_data.csv, and normalization is
idempotent: re-normalizing an already-normalized hand must return it unchanged.
So replaying real recorded rows through the port compares the two
implementations without needing both installed.

Run:  python -m unittest test_hands -v
"""
import csv
import math
import os
import unittest

import dataset
import hands

DATA_PATH = "landmark_data.csv"

# A hand in its own frame: wrist at the origin, middle knuckle one unit "up"
# (image y grows downward, hence negative). The thumb sits on the side that
# makes hand_frame() leave it alone -- i.e. this fixture is a canonical,
# un-mirrored hand, so the mirrored flag reads 0.0 and mirror_scene() is what
# flips it. The exact pose is arbitrary: these tests are about the transform,
# not about any particular handshape.
HAND_LOCAL = [
    (0.00, 0.00), (-0.25, -0.20), (-0.45, -0.45), (-0.60, -0.70), (-0.72, -0.95),
    (-0.20, -0.95), (-0.25, -1.45), (-0.27, -1.75), (-0.28, -2.00),
    (0.00, -1.00), (-0.02, -1.55), (-0.03, -1.90), (-0.03, -2.15),
    (0.20, -0.95), (0.24, -1.45), (0.26, -1.78), (0.27, -2.02),
    (0.38, -0.85), (0.45, -1.25), (0.50, -1.50), (0.53, -1.72),
]


def place(cx, cy, scale=100.0, rotation=0.0, points=None):
    """Put HAND_LOCAL into image (pixel) space at a position/scale/rotation."""
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    out = []
    for lx, ly in (points or HAND_LOCAL):
        rx = lx * cos_r - ly * sin_r
        ry = lx * sin_r + ly * cos_r
        out.append((cx + rx * scale, cy + ry * scale))
    return out


def shape_of(points):
    return hands.hand_frame(points)[0]


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


class TestVectorLayout(unittest.TestCase):

    def test_sizes_line_up(self):
        self.assertEqual(hands.FEATURE_FLOATS, 100)
        self.assertEqual(len(hands.FEATURE_COLUMNS), hands.FEATURE_FLOATS)
        self.assertEqual(len(hands.hand_block(place(300, 300))), hands.PER_HAND_FLOATS)
        self.assertEqual(len(hands.absent_hand_block()), hands.PER_HAND_FLOATS)

    def test_feature_vector_length_in_every_presence_combination(self):
        left, right = place(200, 300), place(400, 300)
        for dom, non in ((left, right), (left, None), (None, right), (None, None)):
            self.assertEqual(len(hands.feature_vector(dom, non)),
                             hands.FEATURE_FLOATS)

    def test_shape_block_is_the_legacy_42(self):
        # The shape floats sit where main's 42-float vector would, so a row
        # from landmark_data.csv can be dropped straight into a Veronica
        # vector's dominant-hand slot.
        self.assertEqual(hands.SHAPE_FLOATS, len(dataset.LANDMARK_COLUMNS))
        block = hands.hand_block(place(300, 300))
        self.assertEqual(len(block[1:1 + hands.SHAPE_FLOATS]), hands.SHAPE_FLOATS)

    def test_column_names_are_unique_and_ordered(self):
        self.assertEqual(len(set(hands.FEATURE_COLUMNS)), hands.FEATURE_FLOATS)
        self.assertEqual(hands.FEATURE_COLUMNS[0], "dom_present")
        self.assertEqual(hands.FEATURE_COLUMNS[hands.NONDOMINANT_OFFSET], "non_present")
        self.assertEqual(hands.FEATURE_COLUMNS[hands.RELATIONAL_OFFSET],
                         "rel_both_present")


class TestMatchesRecordedData(unittest.TestCase):
    """Cross-checks the standard-library port against the numpy original."""

    @unittest.skipUnless(os.path.exists(DATA_PATH), f"{DATA_PATH} not present")
    def test_renormalizing_recorded_rows_is_a_no_op(self):
        rows = []
        with open(DATA_PATH, newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if i >= 400:
                    break
                rows.append([float(row[c]) for c in dataset.LANDMARK_COLUMNS])
        self.assertTrue(rows, "no rows read")

        worst = 0.0
        for flat in rows:
            points = list(zip(flat[0::2], flat[1::2]))
            worst = max(worst, max_diff(shape_of(points), flat))
        # Pure float round-trip: anything above this is a real disagreement
        # between the two implementations, not arithmetic noise.
        self.assertLess(worst, 1e-9, f"port diverges from numpy output by {worst}")

    @unittest.skipUnless(os.path.exists(DATA_PATH), f"{DATA_PATH} not present")
    def test_recorded_rows_carry_no_orientation(self):
        # Worth pinning down: the historical rows cannot supply the new
        # orientation features, because the numpy normalization had already
        # discarded them by the time anything was written to disk. Every
        # recorded hand reads as "pointing straight up, never mirrored".
        # This is why Veronica collects its own file rather than widening the
        # old one -- see VERONICA.md.
        with open(DATA_PATH, newline="") as fh:
            row = next(csv.DictReader(fh))
        flat = [float(row[c]) for c in dataset.LANDMARK_COLUMNS]
        points = list(zip(flat[0::2], flat[1::2]))
        _, orient_cos, orient_sin, mirrored, _ = hands.hand_frame(points)
        self.assertAlmostEqual(orient_cos, 0.0, places=9)
        self.assertAlmostEqual(orient_sin, 1.0, places=9)
        self.assertEqual(mirrored, 0.0)


class TestShapeInvariance(unittest.TestCase):
    """The invariances NOTES.md measured on the numpy version must survive."""

    def setUp(self):
        self.reference = shape_of(place(320, 240))

    def test_translation_changes_nothing(self):
        self.assertLess(max_diff(shape_of(place(80, 400)), self.reference), 1e-12)

    def test_scale_changes_nothing(self):
        for scale in (40.0, 100.0, 250.0):
            self.assertLess(max_diff(shape_of(place(320, 240, scale=scale)),
                                     self.reference), 1e-12)

    def test_in_plane_rotation_changes_nothing(self):
        for degrees in (25, 90, 200, 355):
            rotated = place(320, 240, rotation=math.radians(degrees))
            self.assertLess(max_diff(shape_of(rotated), self.reference), 1e-12)

    def test_finger_curl_does_change_it(self):
        # The control for the three tests above: they would also pass if the
        # transform flattened every hand to a constant.
        curled = list(HAND_LOCAL)
        curled[8] = (-0.28, -1.20)         # index tip curled in
        moved = shape_of(place(320, 240, points=curled))
        self.assertGreater(max_diff(moved, self.reference), 0.1)


class TestOrientationIsRecovered(unittest.TestCase):
    """The parameter the hand frame throws away, now kept alongside it."""

    def test_rotation_moves_orientation_but_not_shape(self):
        upright = hands.hand_frame(place(320, 240))
        turned = hands.hand_frame(place(320, 240, rotation=math.radians(90)))
        self.assertLess(max_diff(turned[0], upright[0]), 1e-12)      # shape
        self.assertGreater(math.hypot(turned[1] - upright[1],
                                      turned[2] - upright[2]), 1.0)  # orientation

    def test_orientation_is_a_unit_vector(self):
        for degrees in (0, 37, 180, 300):
            _, c, s, _, _ = hands.hand_frame(
                place(320, 240, rotation=math.radians(degrees)))
            self.assertAlmostEqual(math.hypot(c, s), 1.0, places=9)

    def test_orientation_is_continuous_across_the_wrap_point(self):
        # cos/sin rather than a raw angle, so 359 deg and 1 deg stay adjacent
        # instead of sitting at opposite ends of the input range.
        before = hands.hand_frame(place(320, 240, rotation=math.radians(359)))
        after = hands.hand_frame(place(320, 240, rotation=math.radians(1)))
        self.assertLess(math.hypot(after[1] - before[1], after[2] - before[2]), 0.1)

    def test_mirrored_flag_distinguishes_the_two_hands(self):
        flipped = hands.mirror_scene(place(320, 240))
        self.assertEqual(hands.hand_frame(place(320, 240))[3], 0.0)
        self.assertEqual(hands.hand_frame(flipped)[3], 1.0)


class TestAbsentHands(unittest.TestCase):

    def test_absent_block_is_flagged_not_just_empty(self):
        block = hands.absent_hand_block()
        self.assertEqual(block[0], 0.0)
        self.assertEqual(set(block), {0.0})

    def test_present_hand_sets_its_flag(self):
        self.assertEqual(hands.hand_block(place(300, 300))[0], 1.0)

    def test_one_handed_sign_zeroes_the_relational_block(self):
        vector = hands.feature_vector(place(300, 300), None)
        relational = vector[hands.RELATIONAL_OFFSET:]
        self.assertEqual(set(relational), {0.0})

    def test_two_hands_set_both_present(self):
        vector = hands.feature_vector(place(200, 300), place(400, 300))
        self.assertEqual(vector[hands.RELATIONAL_OFFSET], 1.0)


class TestRelationalBlock(unittest.TestCase):

    def test_is_invariant_to_camera_distance(self):
        # Both hands twice as far away: every pixel measurement halves, but
        # the block is expressed in hand-widths, so nothing should move.
        near = hands.relational_block(place(200, 300, scale=100),
                                      place(400, 300, scale=100))
        far = hands.relational_block(place(100, 150, scale=50),
                                     place(200, 150, scale=50))
        self.assertLess(max_diff(near, far), 1e-9)

    def test_is_invariant_to_where_in_the_frame_the_pair_sits(self):
        here = hands.relational_block(place(200, 300), place(400, 300))
        there = hands.relational_block(place(500, 100), place(700, 100))
        self.assertLess(max_diff(here, there), 1e-9)

    def test_separation_is_measured_in_hand_widths(self):
        # Wrists 200px apart with a 100px hand -> 2.0 hand-widths.
        block = hands.relational_block(place(200, 300, scale=100),
                                       place(400, 300, scale=100))
        self.assertAlmostEqual(block[1], 2.0, places=6)   # dx
        self.assertAlmostEqual(block[2], 0.0, places=6)   # dy
        self.assertAlmostEqual(block[3], 2.0, places=6)   # distance

    def test_contact_separates_touching_hands_from_apart_ones(self):
        apart = hands.relational_block(place(100, 300), place(600, 300))
        touching = hands.relational_block(place(300, 300), place(330, 300))
        self.assertGreater(apart[7], 1.0)
        self.assertLess(touching[7], apart[7])

    def test_relative_rotation_is_zero_for_parallel_hands(self):
        block = hands.relational_block(place(200, 300), place(400, 300))
        self.assertAlmostEqual(block[5], 1.0, places=6)   # rel_cos
        self.assertAlmostEqual(block[6], 0.0, places=6)   # rel_sin

    def test_relative_rotation_tracks_one_hand_turning(self):
        block = hands.relational_block(
            place(200, 300),
            place(400, 300, rotation=math.radians(90)))
        self.assertAlmostEqual(block[5], 0.0, places=6)
        self.assertAlmostEqual(abs(block[6]), 1.0, places=6)

    def test_size_ratio_reports_the_smaller_hand(self):
        block = hands.relational_block(place(200, 300, scale=100),
                                       place(400, 300, scale=50))
        self.assertAlmostEqual(block[4], 0.5, places=6)

    def test_far_apart_values_are_clamped(self):
        block = hands.relational_block(place(0, 300, scale=20),
                                       place(1900, 300, scale=20))
        self.assertLessEqual(block[3], hands.MAX_SPAN)
        self.assertLessEqual(block[7], hands.MAX_SPAN)

    def test_size_ratio_is_clamped(self):
        block = hands.relational_block(place(200, 300, scale=5),
                                       place(400, 300, scale=500))
        self.assertLessEqual(block[4], hands.MAX_SIZE_RATIO)


class TestHandAssignment(unittest.TestCase):

    def setUp(self):
        self.right = place(500, 300)     # signer's right, image-right when mirrored
        self.left = place(200, 300)

    def test_labels_order_the_pair(self):
        dom, non = hands.assign_hands([(self.left, hands.LEFT),
                                       (self.right, hands.RIGHT)])
        self.assertEqual(dom, self.right)
        self.assertEqual(non, self.left)

    def test_detection_order_does_not_matter(self):
        a = hands.assign_hands([(self.left, hands.LEFT), (self.right, hands.RIGHT)])
        b = hands.assign_hands([(self.right, hands.RIGHT), (self.left, hands.LEFT)])
        self.assertEqual(a, b)

    def test_unmirrored_input_swaps_the_labels(self):
        dom, non = hands.assign_hands([(self.left, hands.LEFT),
                                       (self.right, hands.RIGHT)],
                                      mirrored_input=False)
        self.assertEqual(dom, self.left)
        self.assertEqual(non, self.right)

    def test_duplicate_labels_fall_back_to_image_position(self):
        # MediaPipe occasionally calls both hands the same thing.
        dom, non = hands.assign_hands([(self.left, hands.RIGHT),
                                       (self.right, hands.RIGHT)])
        self.assertEqual(dom, self.right)
        self.assertEqual(non, self.left)

    def test_duplicate_labels_respect_the_mirror_convention(self):
        dom, _ = hands.assign_hands([(self.left, hands.RIGHT),
                                     (self.right, hands.RIGHT)],
                                    mirrored_input=False)
        self.assertEqual(dom, self.left)

    def test_single_hand_is_dominant(self):
        dom, non = hands.assign_hands([(self.right, hands.RIGHT)])
        self.assertEqual(dom, self.right)
        self.assertIsNone(non)

    def test_lone_non_dominant_hand_stays_non_dominant(self):
        dom, non = hands.assign_hands([(self.left, hands.LEFT)])
        self.assertIsNone(dom)
        self.assertEqual(non, self.left)

    def test_missing_label_still_produces_a_hand(self):
        dom, non = hands.assign_hands([(self.right, None)])
        self.assertEqual(dom, self.right)
        self.assertIsNone(non)

    def test_no_hands(self):
        self.assertEqual(hands.assign_hands([]), (None, None))


class TestHandednessSymmetry(unittest.TestCase):
    """A left-handed signer must land in the same feature space as everyone
    else -- otherwise every sign they make is an unseen class."""

    def test_left_dominant_signer_yields_the_same_vector(self):
        right_handed = [(place(200, 300), hands.LEFT),
                        (place(500, 300), hands.RIGHT)]
        reference = hands.feature_vector(*hands.assign_hands(right_handed))

        # The same sign by a left-handed signer is its mirror image, and each
        # hand is now classified as the opposite one.
        left_handed = [(hands.mirror_scene(points),
                        hands.RIGHT if label == hands.LEFT else hands.LEFT)
                       for points, label in right_handed]
        mirrored = hands.feature_vector(
            *hands.assign_hands(left_handed, signer_dominant=hands.LEFT))

        self.assertLess(max_diff(mirrored, reference), 1e-9)

    def test_without_the_correction_they_would_not_match(self):
        # The control: dropping signer_dominant is what the naive pipeline
        # does, and this is how far apart it leaves the two signers.
        right_handed = [(place(200, 300), hands.LEFT),
                        (place(500, 300), hands.RIGHT)]
        reference = hands.feature_vector(*hands.assign_hands(right_handed))
        left_handed = [(hands.mirror_scene(points),
                        hands.RIGHT if label == hands.LEFT else hands.LEFT)
                       for points, label in right_handed]
        uncorrected = hands.feature_vector(*hands.assign_hands(left_handed))
        self.assertGreater(max_diff(uncorrected, reference), 0.5)


if __name__ == "__main__":
    unittest.main()
