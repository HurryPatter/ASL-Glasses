"""Offline tests for the dataset schema and person grouping.

Standard library only -- these run in CI, which installs nothing.

Run:  python -m unittest test_dataset -v
"""
import csv
import os
import tempfile
import unittest

import dataset


class TestSchema(unittest.TestCase):

    def test_header_shape(self):
        self.assertEqual(len(dataset.LANDMARK_COLUMNS), 42)
        self.assertEqual(dataset.HEADER[:3], ["label", "person", "view"])
        self.assertEqual(len(dataset.HEADER), 45)
        self.assertEqual(len(dataset.PRE_VIEW_HEADER), 44)
        self.assertEqual(len(dataset.LEGACY_HEADER), 43)

    def test_person_is_not_a_feature_column(self):
        # The training script selects features by this list, so `person`
        # leaking into it would silently become a 43rd input feature.
        self.assertNotIn("person", dataset.LANDMARK_COLUMNS)
        self.assertNotIn("view", dataset.LANDMARK_COLUMNS)
        self.assertNotIn("label", dataset.LANDMARK_COLUMNS)

    def test_has_person_column(self):
        self.assertTrue(dataset.has_person_column(dataset.HEADER))
        self.assertFalse(dataset.has_person_column(dataset.LEGACY_HEADER))

    def test_read_header(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(dataset.HEADER)
                w.writerow(["A", "omar", "front"] + ["0.0"] * 42)
            self.assertEqual(dataset.read_header(path), dataset.HEADER)


class TestBursts(unittest.TestCase):

    def test_bursts_are_runs_of_identical_labels(self):
        self.assertEqual(dataset.bursts(["A", "A", "B", "B", "B", "A"]),
                         [("A", 0, 2), ("B", 2, 5), ("A", 5, 6)])

    def test_empty(self):
        self.assertEqual(dataset.bursts([]), [])

    def test_single_row(self):
        self.assertEqual(dataset.bursts(["A"]), [("A", 0, 1)])


class TestSessionInference(unittest.TestCase):

    def test_two_alphabet_passes(self):
        labels = ["A"] * 3 + ["B"] * 4 + ["A"] * 2 + ["B"] * 3
        self.assertEqual(dataset.infer_sessions(labels),
                         [0] * 7 + [1] * 5)

    def test_multi_frame_bursts_do_not_split_a_session(self):
        # The bug this guards: comparing row-by-row makes the second frame of
        # every burst look like a repeated label, yielding one session per row.
        labels = ["A"] * 200 + ["B"] * 200
        self.assertEqual(set(dataset.infer_sessions(labels)), {0})

    def test_labels_unique_to_one_pass_do_not_start_a_session(self):
        # The word signs were recorded once, inside one pass.
        labels = (["A"] * 2 + ["B"] * 2 + ["HELLO"] * 2
                  + ["A"] * 2 + ["B"] * 2)
        self.assertEqual(dataset.infer_sessions(labels), [0] * 6 + [1] * 4)

    def test_three_passes(self):
        labels = []
        for _ in range(3):
            for letter in "ABC":
                labels += [letter] * 5
        sessions = dataset.infer_sessions(labels)
        self.assertEqual(sorted(set(sessions)), [0, 1, 2])
        self.assertEqual(sessions.count(0), 15)

    def test_real_dataset_is_attributed(self):
        """Guards the actual committed file, not a synthetic case.

        Deliberately does not pin the set of signers: people get added, and a
        test that fails on every new contributor teaches people to edit the
        test rather than read it. What is worth pinning is that every row is
        attributed, and that the rows which *cannot* be attributed stay that
        way.
        """
        path = os.path.join(os.path.dirname(__file__), "landmark_data.csv")
        if not os.path.exists(path):
            self.skipTest("landmark_data.csv not present")
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        header, body = rows[0], rows[1:]
        self.assertEqual(header, dataset.HEADER,
                         "landmark_data.csv should carry the person column")

        # Every row: full width, and a non-empty signer.
        self.assertTrue(all(len(r) == len(dataset.HEADER) and r[1] for r in body))

        # The original signers are still present.
        people = {r[1] for r in body}
        self.assertTrue({"omar", "laila", "nourhan"} <= people)

        # The unattributable rows are the word signs recorded by three signers
        # in one sitting, with no boundary in the file showing where one stops.
        # They must stay unattributed rather than being credited to whichever
        # pass they happen to sit next to. Note the implication runs one way:
        # a word sign MAY carry a real signer (Riad recorded all three), but an
        # `unknown` row must never be a letter.
        for r in body:
            if r[1] == dataset.UNKNOWN_PERSON:
                self.assertGreater(len(r[0]), 1,
                                   f"letter {r[0]!r} should have a known signer")
        # Every row has a person, a view and the full 42 features.
        self.assertTrue(all(len(r) == 45 and r[1] for r in body))
        self.assertTrue(all(r[2] in dataset.VIEWS for r in body))



class TestView(unittest.TestCase):

    def write(self, d, header, rows):
        path = os.path.join(d, "data.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        return path

    def test_pre_view_file_gains_the_column_as_front(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(d, dataset.PRE_VIEW_HEADER,
                              [["A", "omar"] + ["0.5"] * 42,
                               ["B", "laila"] + ["0.25"] * 42])
            self.assertEqual(dataset.add_view_column(path), 2)
            with open(path, newline="") as fh:
                rows = list(csv.reader(fh))
            self.assertEqual(rows[0], dataset.HEADER)
            self.assertEqual(rows[1][:3], ["A", "omar", "front"])
            self.assertEqual(rows[2][3:], ["0.25"] * 42)   # features untouched
            self.assertEqual(dataset.add_view_column(path), 0)   # idempotent

    def test_unexpected_header_is_refused_and_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(d, ["label"] + dataset.LANDMARK_COLUMNS,
                              [["A"] + ["0.5"] * 42])
            with self.assertRaises(ValueError):
                dataset.add_view_column(path)
            self.assertEqual(dataset.read_header(path)[0:2], ["label", "p0_x"])

    def test_parse_view(self):
        self.assertEqual(dataset.parse_view("f"), "front")
        self.assertEqual(dataset.parse_view(" G "), "firstperson")
        self.assertEqual(dataset.parse_view("glasses"), "firstperson")
        self.assertIsNone(dataset.parse_view("x"))
        self.assertIsNone(dataset.parse_view(""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
