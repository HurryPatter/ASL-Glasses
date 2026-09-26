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

import config
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
         duration_ms=800, face=True, two_handed=True, mirror=False,
         mirrored_input=True):
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
                            FRAME_W, FRAME_H, out,
                            mirrored_input=mirrored_input)


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


class TestHandednessConvention(unittest.TestCase):
    """The setting that cannot be settled by reading code, only against a
    real hand on a real camera -- so it is recorded rather than assumed."""

    def test_clips_record_the_convention_they_were_collected_under(self):
        self.assertTrue(clip()["mirrored_input"])
        self.assertFalse(clip(mirrored_input=False)["mirrored_input"])

    def test_row_carries_it_as_metadata(self):
        row = dict(zip(signset.HEADER,
                       signset.clip_to_row(clip(mirrored_input=False))))
        self.assertEqual(row["mirrored_input"], False)

    def test_clips_predating_the_field_default_to_the_old_behaviour(self):
        old = clip()
        del old["mirrored_input"]
        # Must not raise, and must reproduce what it was collected under.
        self.assertEqual(signset.clip_to_row(old)[:4],
                         signset.clip_to_row(clip())[:4])

    def test_the_convention_swaps_which_hand_is_dominant(self):
        # The whole hazard in one assertion: the same landmarks, read under
        # the two conventions, put different physical hands in the dominant
        # block. Nothing downstream can tell -- both vectors are plausible.
        as_reported = signset.clip_to_samples(clip(), mirrored_input=True)
        swapped = signset.clip_to_samples(clip(), mirrored_input=False)
        self.assertGreater(
            max_diff(as_reported[0].features, swapped[0].features), 0.5)

    def test_an_archive_collected_wrong_is_repairable_without_re_signing(self):
        # The reason the archive stores raw landmarks at all. A whole
        # collection made under the wrong convention is corrected by
        # re-deriving, not by bringing the signers back.
        correct = signset.clip_to_row(clip(mirrored_input=True))
        collected_wrong = clip(mirrored_input=False)
        repaired = signset.clip_to_row(collected_wrong, mirrored_input=True)

        n = len(signset.META_COLUMNS)
        self.assertLess(max_diff(correct[n:], repaired[n:]), 1e-9)
        self.assertNotEqual(signset.clip_to_row(collected_wrong)[n:], correct[n:])


class TestHandednessStability(unittest.TestCase):
    """MediaPipe's per-frame label flips, most readily when a palm turns away.

    Trusting it per frame lets a hand change identity mid-sign, which in a
    two-handed sign swaps the dominant and non-dominant blocks partway through
    and produces a feature vector describing a sign nobody made.
    """

    def flipped(self, at, label=hands.LEFT):
        source = clip(two_handed=False)
        for f in source["frames"][at:at + 3]:
            f["hands"][0]["label"] = label
        return source

    def test_a_clean_clip_reports_full_stability(self):
        self.assertEqual(signset.handedness_stability(clip()), 1.0)

    def test_a_flip_is_detected(self):
        self.assertLess(signset.handedness_stability(self.flipped(4)), 1.0)

    def test_a_flip_is_corrected_by_clip_majority(self):
        frames, _ = signset.stabilized_frames(self.flipped(4))
        labels = {f["hands"][0]["label"] for f in frames if f["hands"]}
        self.assertEqual(labels, {hands.RIGHT})

    def test_correction_makes_the_features_match_the_clean_clip(self):
        # The point: the repaired clip must be indistinguishable from one
        # MediaPipe never got wrong.
        clean = signset.clip_to_samples(clip(two_handed=False))
        repaired = signset.clip_to_samples(self.flipped(4))
        for a, b in zip(clean, repaired):
            self.assertLess(max_diff(a.features, b.features), 1e-9)

    def test_without_correction_the_flipped_frames_would_differ(self):
        # The control, and the size of the damage: three frames out of twelve
        # describing the other hand.
        source = self.flipped(4)
        raw = source["frames"][4]
        detected = [([(x * FRAME_W, y * FRAME_H) for x, y in h["points"]],
                     h["label"]) for h in raw["hands"]]
        dom, non, _ = hands.canonical_scene(detected, None)
        self.assertIsNone(dom)          # relabelled away from the dominant slot
        self.assertIsNotNone(non)

    def test_hands_are_followed_through_a_crossing(self):
        # Hands cross in ASL, so identity must follow the hand rather than
        # the side of the frame it happens to be on.
        source = clip(two_handed=True)
        stability = signset.handedness_stability(source)
        self.assertGreaterEqual(stability, 0.9)

    def test_stability_is_reported_as_metadata(self):
        row = dict(zip(signset.HEADER, signset.clip_to_row(self.flipped(4))))
        self.assertLess(row["handedness_stability"], 1.0)
        self.assertGreater(row["handedness_stability"], 0.0)

    def test_an_empty_clip_is_trivially_stable(self):
        empty = signset.raw_clip("x", "A", "p", hands.RIGHT, 1, 1, [])
        self.assertEqual(signset.handedness_stability(empty), 1.0)


class TestConfigFile(unittest.TestCase):

    def test_defaults_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "none.json")
            self.assertEqual(config.load(path), config.DEFAULTS)

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.json")
            config.save({"mirrored_input": False}, path)
            self.assertFalse(config.load(path)["mirrored_input"])

    def test_a_corrupt_config_does_not_stop_a_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.json")
            with open(path, "w") as fh:
                fh.write("{ not json")
            self.assertEqual(config.load(path), config.DEFAULTS)

    def test_unknown_keys_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.json")
            with open(path, "w") as fh:
                json.dump({"mirrored_input": False, "nonsense": 1}, fh)
            self.assertEqual(config.load(path), {"mirrored_input": False})


class TestJitterSmoothing(unittest.TestCase):
    """A 3-point median: the median of three monotonically changing samples is
    the middle sample, so steady movement passes through untouched and only a
    sample disagreeing with both neighbours is replaced."""

    def spike(self, at, size=0.25):
        source = clip(two_handed=False)
        source["frames"][at]["hands"][0]["points"] = [
            [x + size, y] for x, y in
            source["frames"][at]["hands"][0]["points"]]
        return source

    def test_a_single_frame_spike_is_removed(self):
        spiked = self.spike(6)
        before = spiked["frames"][6]["hands"][0]["points"][0][0]
        after = signset.smooth_points(
            spiked["frames"])[6]["hands"][0]["points"][0][0]
        self.assertLess(abs(after - before), 0.25)

    def test_steady_movement_passes_through_unchanged(self):
        # The property that makes a median safe where an average is not: a
        # fast sign must not be blunted to buy spike protection.
        source = clip(two_handed=False)
        smoothed = signset.smooth_points(source["frames"])
        for original, filtered in zip(source["frames"][1:-1], smoothed[1:-1]):
            self.assertLess(
                max_diff([c for p in original["hands"][0]["points"] for c in p],
                         [c for p in filtered["hands"][0]["points"] for c in p]),
                1e-9)

    def test_peak_speed_survives_smoothing(self):
        fast = clip(two_handed=False, duration_ms=300)
        summary = sequence.movement_summary(
            sequence.trajectory(signset.clip_to_samples(fast), "dom")[0])
        self.assertGreater(summary[6], 1.0)

    def test_endpoints_are_left_alone(self):
        source = clip(two_handed=False)
        smoothed = signset.smooth_points(source["frames"])
        self.assertEqual(smoothed[0]["hands"][0]["points"],
                         source["frames"][0]["hands"][0]["points"])

    def test_a_short_clip_is_not_smoothed(self):
        source = clip(two_handed=False, frames=2)
        self.assertEqual(signset.smooth_points(source["frames"]),
                         source["frames"])

    def test_smoothing_survives_frames_with_no_hands(self):
        source = clip(two_handed=False)
        source["frames"][5]["hands"] = []
        smoothed = signset.smooth_points(source["frames"])
        self.assertEqual(smoothed[5]["hands"], [])


class TestHandCoverage(unittest.TestCase):

    def test_a_clean_clip_is_fully_covered(self):
        self.assertEqual(signset.hand_coverage(clip()), 1.0)

    def test_dropouts_lower_it(self):
        source = clip()
        for f in source["frames"][:3]:
            f["hands"] = []
        self.assertAlmostEqual(signset.hand_coverage(source), 0.75, places=6)

    def test_it_is_recorded_as_metadata(self):
        source = clip()
        for f in source["frames"][:3]:
            f["hands"] = []
        row = dict(zip(signset.HEADER, signset.clip_to_row(source)))
        self.assertLess(row["hand_coverage"], 1.0)

    def test_an_empty_clip_has_no_coverage(self):
        empty = signset.raw_clip("x", "A", "p", hands.RIGHT, 1, 1, [])
        self.assertEqual(signset.hand_coverage(empty), 0.0)


class TestActingHand(unittest.TestCase):
    """Observed, not asked -- the glasses cannot ask a stranger."""

    def test_a_one_handed_clip_reports_the_hand_that_signed_it(self):
        self.assertEqual(signset.acting_hand(clip(two_handed=False)), hands.RIGHT)

    def test_the_convention_is_applied(self):
        source = clip(two_handed=False, mirrored_input=False)
        self.assertEqual(signset.acting_hand(source), hands.LEFT)

    def test_either_hand_lands_in_the_dominant_slot(self):
        # The point of the whole thing: a sign made with either hand must end
        # up in one canonical space, since handedness is not phonemic in ASL.
        right = signset.clip_to_samples(clip(two_handed=False))
        mirrored = signset.clip_to_samples(clip(two_handed=False, mirror=True))
        for sample in (right[0], mirrored[0]):
            self.assertEqual(sample.features[hands.DOMINANT_OFFSET], 1.0)
            self.assertEqual(sample.features[hands.NONDOMINANT_OFFSET], 0.0)

    def test_an_explicit_override_still_wins(self):
        source = clip(two_handed=False)
        samples = signset.clip_to_samples(source, signer_dominant=hands.LEFT)
        self.assertEqual(samples[0].features[hands.NONDOMINANT_OFFSET], 1.0)

    def test_a_clip_with_no_hands_falls_back_to_what_was_recorded(self):
        empty = signset.raw_clip("x", "A", "p", hands.LEFT, 640, 480,
                                 [frame([], FACE_BOX, t) for t in (0, 100)])
        self.assertEqual(signset.acting_hand(empty), hands.LEFT)

    def test_it_is_recorded_as_metadata(self):
        row = dict(zip(signset.HEADER, signset.clip_to_row(clip(two_handed=False))))
        self.assertEqual(row["acting_hand"], hands.RIGHT)


class TestTrainingWindows(unittest.TestCase):
    """Train on what the live segmenter sees: windows, not whole takes."""

    def test_a_long_clip_is_cut_into_overlapping_windows(self):
        long_clip = clip(frames=60, duration_ms=2400)
        windows = signset.clip_windows(long_clip, length_ms=900, step_ms=300)
        self.assertGreaterEqual(len(windows), 5)
        for window in windows:
            self.assertLessEqual(signset.clip_span_ms(window), 900)

    def test_each_window_starts_at_zero(self):
        for window in signset.clip_windows(clip(frames=60, duration_ms=2400)):
            self.assertEqual(window["frames"][0]["t"], 0)

    def test_windows_keep_the_parent_clip_id(self):
        # So a split by clip keeps a take's overlapping windows together
        # rather than scoring near-duplicates against each other.
        parent = clip(frames=60, duration_ms=2400)
        ids = {w["clip_id"] for w in signset.clip_windows(parent)}
        self.assertEqual(ids, {parent["clip_id"]})

    def test_a_short_clip_comes_back_whole(self):
        short = clip(frames=12, duration_ms=700)
        self.assertEqual(signset.clip_windows(short), [short])

    def test_the_parent_clip_is_not_modified(self):
        parent = clip(frames=60, duration_ms=2400)
        before = json.dumps(parent, sort_keys=True)
        signset.clip_windows(parent)
        self.assertEqual(json.dumps(parent, sort_keys=True), before)

    def test_every_window_makes_a_full_row(self):
        for window in signset.clip_windows(clip(frames=60, duration_ms=2400)):
            self.assertEqual(len(signset.clip_to_row(window)), len(signset.HEADER))

    def test_training_and_live_windows_are_the_same_length(self):
        # The mismatch this exists to prevent: whole-take training against
        # 900ms live windows cost ~12 points on the first 250 clips.
        import inspect
        import segment
        default = inspect.signature(
            segment.ContinuousSegmenter.__init__).parameters["window_ms"].default
        window = inspect.signature(
            signset.clip_windows).parameters["length_ms"].default
        self.assertEqual(default, window)
        self.assertEqual(default, sequence.SIGN_WINDOW_MS)
