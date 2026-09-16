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
        self.assertEqual(dataset.HEADER[:2], ["label", "person"])
        self.assertEqual(len(dataset.HEADER), 44)
        self.assertEqual(len(dataset.LEGACY_HEADER), 43)

    def test_person_is_not_a_feature_column(self):
        # The training script selects features by this list, so `person`
        # leaking into it would silently become a 43rd input feature.
        self.assertNotIn("person", dataset.LANDMARK_COLUMNS)
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
                w.writerow(["A", "omar"] + ["0.0"] * 42)
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
        # Every row has a person and the full 42 features.
        self.assertTrue(all(len(r) == 44 and r[1] for r in body))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestOfflineModulesStayDependencyFree(unittest.TestCase):
    """CI installs nothing, so every module the suites import must be
    importable with the standard library alone.

    Without this, adding `import numpy` to one of them would not fail loudly:
    the import error would skip or error the affected suite while the workflow
    still looked like it ran, and the guard against logic regressions would be
    quietly gone. The tests.yml comment claims this property; this enforces it.
    """

    OFFLINE_MODULES = ["dataset", "debouncer", "motion", "hands", "location",
                       "sequence", "signset", "folds", "gloss", "segment"]
    HEAVY = {"numpy", "pandas", "cv2", "mediapipe", "sklearn",
             "joblib", "symspellpy", "scipy", "torch", "tensorflow"}

    def test_no_third_party_imports(self):
        import ast

        for name in self.OFFLINE_MODULES:
            path = f"{name}.py"
            if not os.path.exists(path):
                continue
            imported = set()
            for node in ast.walk(ast.parse(open(path).read())):
                if isinstance(node, ast.Import):
                    imported |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertFalse(
                imported & self.HEAVY,
                f"{path} imports {sorted(imported & self.HEAVY)}; CI installs "
                f"nothing, so this silently disables its tests")

    def test_capture_py_is_the_only_mediapipe_seam(self):
        """The offline layer is testable precisely because nothing in it
        touches MediaPipe. capture.py is the single adapter, which is also
        what keeps the pixel-vs-normalized face box conversion in one place
        instead of three."""
        import ast

        for name in self.OFFLINE_MODULES:
            path = f"{name}.py"
            if not os.path.exists(path):
                continue
            source = open(path).read()
            self.assertNotIn("import mediapipe", source, path)
            self.assertNotIn("import capture", source,
                             f"{path} must not depend on the MediaPipe seam")

    def test_they_actually_import(self):
        import importlib

        for name in self.OFFLINE_MODULES:
            if os.path.exists(f"{name}.py"):
                importlib.import_module(name)
