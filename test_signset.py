"""Offline tests for the sign dataset schema and archive reconstruction.

Standard library only, so it runs in CI.

`clip_to_samples()` is the function these mostly exist for. Every training row
Veronica ever produces passes through it, and it is the only thing standing
between the raw archive and needing every signer back in a room. A silent bug
in it corrupts the dataset in a way that looks entirely plausible on inspection.

Run:  python -m unittest test_signset -v
"""
import json
import os
import tempfile
import unittest

import hands
import location
import sequence
import signset
from test_hands import place

FRAME_W, FRAME_H = 640, 480


def normalized(points):
    return [[x / FRAME_W, y / FRAME_H] for x, y in points]


def mirrored_normalized(points):
    """Reflect about the frame centre, as a left-handed signer's scene is."""
    return [[1.0 - x / FRAME_W, y / FRAME_H] for x, y in points]


def frame(hand_specs, face_box, t):
    return {
        "t": t,
        "hands": [{"label": label, "points": pts} for pts, label in hand_specs],
        "face": face_box,
    }


FACE_BOX = [300 / FRAME_W, 60 / FRAME_H, 120 / FRAME_W, 160 / FRAME_H]


def clip(label="HELLO", person="omar", dominant=hands.RIGHT, frames=12,
         duration_ms=800, face=True, two_handed=True, mirror=False):
    """A synthetic archived clip of a hand travelling downward."""
    out = []
    for i in range(frames):
        u = i / (frames - 1)
        t = int(duration_ms * u)
        right = place(400, 200 + 200 * u)
        left = place(220, 200 + 200 * u)

        if mirror:
            specs = [(mirrored_normalized(right), hands.LEFT)]
            if two_handed:
                specs.append((mirrored_normalized(left), hands.RIGHT))
            box = None if not face else [
                1.0 - (FACE_BOX[0] + FACE_BOX[2]), FACE_BOX[1],
                FACE_BOX[2], FACE_BOX[3]]
        else:
            specs = [(normalized(right), hands.RIGHT)]
            if two_handed:
                specs.append((normalized(left), hands.LEFT))
            box = FACE_BOX if face else None

        out.append(frame(specs, box, t))

    return signset.raw_clip("omar-1-1", label, person, dominant,
                            FRAME_W, FRAME_H, out)


def max_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


class TestSchema(unittest.TestCase):

    def test_header_is_metadata_then_features(self):
        self.assertEqual(signset.HEADER[:len(signset.META_COLUMNS)],
                         signset.META_COLUMNS)
        self.assertEqual(len(signset.HEADER),
                         len(signset.META_COLUMNS) + sequence.SIGN_FLOATS)

    def test_column_names_are_unique(self):
        self.assertEqual(len(set(signset.HEADER)), len(signset.HEADER))

    def test_metadata_is_not_a_feature_column(self):
        # The hazard dataset.py names: a metadata column leaking into the
        # feature list silently becomes an extra input, and `person` in
        # particular would let the model identify the signer instead of the
        # sign -- inflating exactly the cross-person number that matters.
        for column in signset.META_COLUMNS:
            self.assertNotIn(column, sequence.SIGN_COLUMNS)


class TestVocabulary(unittest.TestCase):

    def test_no_duplicate_glosses(self):
        self.assertEqual(len(signset.SIGNS), len(set(signset.SIGNS)))

    def test_rest_is_included(self):
        # Stage 5 needs something to reject garbage with and stage 6 has to
        # tell "between signs" from "a sign". Collecting it later means
        # another session with every signer.
        self.assertIn(signset.REST_LABEL, signset.SIGNS)

    def test_the_location_contrast_is_in_the_vocabulary(self):
        # MOTHER and FATHER differ only in location, so they are the working
        # check that stage 2 earns its place. Losing them from the list would
        # leave nothing in the data that tests it.
        self.assertIn("MOTHER", signset.SIGNS)
        self.assertIn("FATHER", signset.SIGNS)


class TestArchiveFormat(unittest.TestCase):

    def test_frames_store_raw_normalized_landmarks(self):
        record = signset.raw_frame(
            [([(0.123456, 0.654321)], hands.RIGHT)], None, 40)
        self.assertEqual(record["t"], 40)
        self.assertEqual(record["hands"][0]["label"], hands.RIGHT)
        self.assertEqual(record["hands"][0]["points"], [[0.1235, 0.6543]])
        self.assertIsNone(record["face"])

    def test_rounding_is_finer_than_the_tracker(self):
        # 4 decimals is ~0.06px on a 640px frame.
        self.assertLess(10 ** -signset._COORD_PLACES * FRAME_W, 0.1)

    def test_round_trip_through_the_file(self):
        original = clip()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clips.jsonl")
            signset.append_clip(path, original)
            signset.append_clip(path, clip(label="YES"))
            back = list(signset.read_clips(path))
        self.assertEqual(len(back), 2)
        self.assertEqual(back[0], original)
        self.assertEqual(back[1]["label"], "YES")

    def test_blank_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clips.jsonl")
            with open(path, "w") as fh:
                fh.write(json.dumps(clip()) + "\n\n")
            self.assertEqual(len(list(signset.read_clips(path))), 1)


class TestReconstruction(unittest.TestCase):

    def test_one_sample_per_frame(self):
        source = clip(frames=15)
        samples = signset.clip_to_samples(source)
        self.assertEqual(len(samples), 15)
        self.assertEqual([s.t for s in samples],
                         [f["t"] for f in source["frames"]])

    def test_samples_carry_full_frame_vectors(self):
        for sample in signset.clip_to_samples(clip()):
            self.assertEqual(len(sample.features), hands.FEATURE_FLOATS)
            self.assertIsNotNone(sample.dom)
            self.assertIsNotNone(sample.non)

    def test_row_matches_the_header(self):
        self.assertEqual(len(signset.clip_to_row(clip())), len(signset.HEADER))

    def test_row_metadata(self):
        row = dict(zip(signset.HEADER, signset.clip_to_row(clip(frames=12))))
        self.assertEqual(row["label"], "HELLO")
        self.assertEqual(row["person"], "omar")
        self.assertEqual(row["dominant"], hands.RIGHT)
        self.assertEqual(row["n_frames"], 12)
        self.assertEqual(row["face_coverage"], 1.0)

    def test_one_handed_clip_leaves_the_non_dominant_absent(self):
        samples = signset.clip_to_samples(clip(two_handed=False))
        self.assertTrue(all(s.non is None for s in samples))

    def test_clip_with_no_hands_at_all(self):
        empty = signset.raw_clip("x-1-1", signset.REST_LABEL, "omar",
                                 hands.RIGHT, FRAME_W, FRAME_H,
                                 [frame([], FACE_BOX, t) for t in (0, 100, 200)])
        row = signset.clip_to_row(empty)
        self.assertEqual(len(row), len(signset.HEADER))

    def test_face_coverage_counts_frames_with_a_face(self):
        self.assertEqual(signset.face_coverage(clip(face=True)), 1.0)
        self.assertEqual(signset.face_coverage(clip(face=False)), 0.0)

        patchy = clip()
        for f in patchy["frames"][:6]:
            f["face"] = None
        self.assertAlmostEqual(signset.face_coverage(patchy), 0.5, places=6)

    def test_span_is_wall_clock(self):
        self.assertEqual(signset.clip_span_ms(clip(duration_ms=800)), 800)
        self.assertEqual(signset.clip_span_ms(
            signset.raw_clip("x", "A", "p", hands.RIGHT, 1, 1, [])), 0)

    def test_rebuilding_twice_gives_the_same_row(self):
        # The archive is the record and the CSV is a build artifact, so the
        # derivation has to be deterministic or a rebuild silently changes
        # the dataset under a trained model.
        source = clip()
        self.assertEqual(signset.clip_to_row(source),
                         signset.clip_to_row(source))


class TestLeftDominantSigners(unittest.TestCase):
    """The corruption that would have been invisible.

    A left-dominant signer's scene is mirrored into right-dominant space. If
    the hands are mirrored and the face is not, every location feature comes
    out wrong by twice the head's offset from the axis -- while staying
    entirely plausible. hands.canonical_scene() binds the two together and
    clip_to_samples() is required to use it.
    """

    def test_a_left_handed_signer_lands_in_the_same_feature_space(self):
        right = signset.clip_to_row(clip(dominant=hands.RIGHT))
        left = signset.clip_to_row(clip(dominant=hands.LEFT, mirror=True))
        # Metadata differs (dominant), features must not.
        n = len(signset.META_COLUMNS)
        self.assertLess(max_diff(right[n:], left[n:]), 1e-9)

    def test_mirroring_the_hands_but_not_the_face_corrupts_location(self):
        # The control: what the naive call site would have produced.
        source = clip(dominant=hands.LEFT, mirror=True)
        good = signset.clip_to_samples(source)[0].features

        f = source["frames"][0]
        detected = [([(x * FRAME_W, y * FRAME_H) for x, y in h["points"]],
                     h["label"]) for h in f["hands"]]
        face = location.face_from_normalized_box(*f["face"], FRAME_W, FRAME_H)
        dom, non = hands.assign_hands(detected, signer_dominant=hands.LEFT)
        bad = hands.feature_vector(dom, non, face)   # face NOT mirrored

        self.assertGreater(max_diff(good, bad), 1.0)

    def test_the_hand_blocks_are_unaffected_either_way(self):
        # Narrowing where the damage lands: handshape and orientation are
        # fine, only location is wrong -- which is why it is hard to spot.
        source = clip(dominant=hands.LEFT, mirror=True)
        good = signset.clip_to_samples(source)[0].features

        f = source["frames"][0]
        detected = [([(x * FRAME_W, y * FRAME_H) for x, y in h["points"]],
                     h["label"]) for h in f["hands"]]
        face = location.face_from_normalized_box(*f["face"], FRAME_W, FRAME_H)
        dom, non = hands.assign_hands(detected, signer_dominant=hands.LEFT)
        bad = hands.feature_vector(dom, non, face)

        self.assertEqual(good[:hands.LOCATION_OFFSET], bad[:hands.LOCATION_OFFSET])
        self.assertNotEqual(good[hands.LOCATION_OFFSET:], bad[hands.LOCATION_OFFSET:])


class TestNoFace(unittest.TestCase):

    def test_location_block_is_zeroed_but_the_rest_survives(self):
        with_face = signset.clip_to_samples(clip(face=True))[0].features
        without = signset.clip_to_samples(clip(face=False))[0].features
        self.assertEqual(set(without[hands.LOCATION_OFFSET:]), {0.0})
        self.assertEqual(with_face[:hands.LOCATION_OFFSET],
                         without[:hands.LOCATION_OFFSET])

    def test_a_clip_with_no_face_still_produces_a_valid_row(self):
        self.assertEqual(len(signset.clip_to_row(clip(face=False))),
                         len(signset.HEADER))


class TestCsvOutput(unittest.TestCase):

    def test_written_file_reads_back_with_the_right_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "signs.csv")
            signset.write_rows(path, [signset.clip_to_row(clip()),
                                      signset.clip_to_row(clip(label="YES"))])
            self.assertEqual(signset.read_header(path), signset.HEADER)
            with open(path) as fh:
                self.assertEqual(len(fh.readlines()), 3)

    def test_people_and_counts(self):
        clips = [clip(person="omar"), clip(person="laila", label="YES"),
                 clip(person="laila")]
        self.assertEqual(signset.people(clips), ["laila", "omar"])
        self.assertEqual(signset.counts_by_label(clips), {"HELLO": 2, "YES": 1})


if __name__ == "__main__":
    unittest.main()
