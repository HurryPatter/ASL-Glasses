"""Offline tests for the face-anchored location layer -- no camera, no detector.

Standard library only, so it runs in CI. A face is three numbers here, which is
the point: location.py never learns which detector produced the box, so these
tests exercise the geometry without MediaPipe installed.

Run:  python -m unittest test_location -v
"""
import math
import unittest

import hands
import location

# A face 120px wide, 160px tall, centred at (320, 180) -- roughly where a
# seated signer's head sits in a 640x480 frame.
FACE = location.face_from_box(260, 100, 120, 160)
CX, CY, FW, FH = FACE


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


class TestFaceBox(unittest.TestCase):

    def test_box_to_centre_and_size(self):
        self.assertEqual(FACE, (320.0, 180.0, 120, 160))

    def test_normalized_box_matches_pixel_box(self):
        normalized = location.face_from_normalized_box(
            260 / 640, 100 / 480, 120 / 640, 160 / 480, 640, 480)
        self.assertLess(max_diff(normalized, FACE), 1e-9)


class TestLocate(unittest.TestCase):

    def test_face_centre_is_the_origin(self):
        self.assertEqual(location.locate((CX, CY), FACE), (0.0, 0.0))

    def test_measured_in_face_widths(self):
        # One face-width right and one face-width down.
        self.assertEqual(location.locate((CX + FW, CY + FW), FACE), (1.0, 1.0))

    def test_both_axes_share_one_unit(self):
        # An isotropic frame: a 45-degree offset has to stay at 45 degrees.
        # Dividing y by face *height* instead would stretch it, and "up and to
        # the left" would mean a different direction at every head pitch.
        lx, ly = location.locate((CX + 90, CY + 90), FACE)
        self.assertAlmostEqual(lx, ly, places=9)


class TestCameraDistance(unittest.TestCase):
    """The reason for face-widths: no depth sensor, no calibration."""

    def test_moving_the_whole_scene_further_away_changes_nothing(self):
        near = location.location_block(((CX + 120, CY + 60), 96.0), None, FACE)

        # Same geometry at half the scale, and shifted somewhere else in frame.
        far_face = location.face_from_box(430, 250, 60, 80)
        fcx, fcy, _, _ = far_face
        far = location.location_block(((fcx + 60, fcy + 30), 48.0), None, far_face)

        self.assertLess(max_diff(near, far), 1e-9)

    def test_moving_the_head_around_the_frame_changes_nothing(self):
        here = location.location_block(((CX + 120, CY + 60), 96.0), None, FACE)
        moved_face = location.face_from_box(20, 300, 120, 160)
        mcx, mcy, _, _ = moved_face
        there = location.location_block(((mcx + 120, mcy + 60), 96.0), None, moved_face)
        self.assertLess(max_diff(here, there), 1e-9)


class TestHeadPitch(unittest.TestCase):
    """Nodding is grammatical in ASL, so the unit must not move when it happens."""

    def test_a_nod_compresses_height_but_not_the_features(self):
        # Pitching the head forward squashes the box vertically; width holds.
        nodded = location.face_from_box(260, 120, 120, 110)
        upright = location.location_block(((CX + 120, CY), 96.0), None, FACE)
        ncx, ncy, _, _ = nodded
        pitched = location.location_block(((ncx + 120, ncy), 96.0), None, nodded)
        self.assertLess(max_diff(upright, pitched), 1e-9)

    def test_height_would_have_moved_it(self):
        # The control: this is the size of the error a height-based unit
        # would inject every time the signer nods.
        self.assertGreater(abs(160 / 110 - 1.0), 0.4)


class TestDepthProxy(unittest.TestCase):
    """hand size / face width: the third dimension, from two flat measurements."""

    def test_a_hand_reaching_toward_the_camera_reads_larger(self):
        at_body = location.location_block(((CX, CY + 200), 84.0), None, FACE)
        reaching = location.location_block(((CX, CY + 200), 130.0), None, FACE)
        self.assertGreater(reaching[3], at_body[3])

    def test_depth_is_clamped(self):
        absurd = location.location_block(((CX, CY), 4000.0), None, FACE)
        self.assertLessEqual(absurd[3], location.MAX_DEPTH_RATIO)

    def test_reach_is_clamped(self):
        far = location.location_block(((CX + 9000, CY - 9000), 96.0), None, FACE)
        self.assertLessEqual(abs(far[1]), location.MAX_REACH)
        self.assertLessEqual(abs(far[2]), location.MAX_REACH)


class TestLocationContrasts(unittest.TestCase):
    """The whole point of the stage: signs that differ only in where they are."""

    def test_forehead_and_chin_are_distinguishable(self):
        # FATHER vs MOTHER -- same handshape, same orientation, same movement.
        forehead = location.location_block(((CX, CY - FH / 2), 90.0), None, FACE)
        chin = location.location_block(((CX, CY + FH / 2), 90.0), None, FACE)
        self.assertGreater(abs(forehead[2] - chin[2]), 1.0)

    def test_a_hands_only_pipeline_cannot_tell_them_apart(self):
        # The control, and the reason this stage exists: with no face, the two
        # produce byte-identical vectors however far apart the hands are.
        forehead = location.location_block(((CX, CY - FH / 2), 90.0), None, None)
        chin = location.location_block(((CX, CY + FH / 2), 90.0), None, None)
        self.assertEqual(forehead, chin)


class TestMissingFace(unittest.TestCase):

    def test_no_face_zeroes_the_block(self):
        block = location.location_block(((CX, CY), 90.0), None, None)
        self.assertEqual(len(block), location.LOCATION_FLOATS)
        self.assertEqual(set(block), {0.0})

    def test_face_present_flag_separates_absent_from_centred(self):
        # Without the flag, "no face detected" and "both hands exactly at the
        # centre of the face" would be the same vector.
        absent = location.location_block(((CX, CY), 0.0), None, None)
        centred = location.location_block(((CX, CY), 0.0), None, FACE)
        self.assertEqual(absent[0], 0.0)
        self.assertEqual(centred[0], 1.0)
        self.assertNotEqual(absent, centred)

    def test_absent_hand_zeroes_only_its_own_slots(self):
        block = location.location_block(((CX + 60, CY), 90.0), None, FACE)
        self.assertEqual(block[0], 1.0)
        self.assertNotEqual(block[1:4], [0.0, 0.0, 0.0])
        self.assertEqual(block[4:7], [0.0, 0.0, 0.0])


class TestDiagnostics(unittest.TestCase):
    """Debug readout only -- never fed to a classifier."""

    def test_bands_run_head_to_chest(self):
        cases = [
            (CY - FH, "above head"),
            (CY - FH / 3, "forehead"),
            (CY, "eyes/nose"),
            (CY + FH / 3, "mouth/chin"),
            (CY + FH, "neck/shoulder"),
            (CY + FH * 2.5, "chest"),
        ]
        for y, expected in cases:
            self.assertIn(expected, location.zone_name((CX, y), FACE))

    def test_sides_are_named(self):
        self.assertIn("centre", location.zone_name((CX, CY), FACE))
        self.assertIn("signer left", location.zone_name((CX - 2 * FW, CY), FACE))
        self.assertIn("signer right", location.zone_name((CX + 2 * FW, CY), FACE))

    def test_debug_info_survives_every_absence(self):
        self.assertEqual(location.debug_info(None, None, None), {"face": "not detected"})
        info = location.debug_info(((CX, CY), 90.0), None, FACE)
        self.assertEqual(info["non"], "absent")
        self.assertIn("depth", info["dom"])


class TestFrameVectorIntegration(unittest.TestCase):
    """location.py composed into the full 107-float frame vector."""

    def setUp(self):
        from test_hands import place
        self.dominant = place(400, 380)
        self.nondominant = place(240, 380)

    def test_location_block_lands_at_its_offset(self):
        vector = hands.feature_vector(self.dominant, self.nondominant, FACE)
        self.assertEqual(len(vector), hands.FEATURE_FLOATS)
        block = vector[hands.LOCATION_OFFSET:]
        self.assertEqual(block,
                         location.location_block(hands.anchor(self.dominant),
                                                 hands.anchor(self.nondominant),
                                                 FACE))

    def test_face_is_optional(self):
        without = hands.feature_vector(self.dominant, self.nondominant)
        self.assertEqual(len(without), hands.FEATURE_FLOATS)
        self.assertEqual(set(without[hands.LOCATION_OFFSET:]), {0.0})

    def test_losing_the_face_keeps_the_other_hundred_floats(self):
        # A detector dropping the face for a few frames must not invalidate
        # handshape, orientation or the relational block.
        with_face = hands.feature_vector(self.dominant, self.nondominant, FACE)
        without = hands.feature_vector(self.dominant, self.nondominant, None)
        self.assertEqual(with_face[:hands.LOCATION_OFFSET],
                         without[:hands.LOCATION_OFFSET])

    def test_anchor_reports_wrist_and_hand_size(self):
        wrist, size = hands.anchor(self.dominant)
        self.assertEqual(wrist, self.dominant[hands.WRIST])
        self.assertAlmostEqual(size, 100.0, places=6)
        self.assertIsNone(hands.anchor(None))

    def test_location_moves_with_the_hand_while_shape_stays_put(self):
        from test_hands import place
        high = hands.feature_vector(place(400, 120), None, FACE)
        low = hands.feature_vector(place(400, 420), None, FACE)
        self.assertEqual(high[:hands.RELATIONAL_OFFSET], low[:hands.RELATIONAL_OFFSET])
        self.assertNotEqual(high[hands.LOCATION_OFFSET:], low[hands.LOCATION_OFFSET:])


if __name__ == "__main__":
    unittest.main()
