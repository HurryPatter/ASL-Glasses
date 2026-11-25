"""Tests for features.py, the hand-frame normalization. Needs numpy.

The properties checked are the ones the whole design leans on: the 42 floats
do not change when the hand moves in the image, comes closer, rotates in the
image plane, or is the other hand. And the three scripts that use the
features must import this one definition rather than carry their own copy.
"""
import ast
import math
import os
import random
import tempfile
import unittest

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

HERE = os.path.dirname(os.path.abspath(__file__))


class LM:
    def __init__(self, x, y):
        self.x, self.y = x, y


def hand(seed=0):
    """A plausible 21-point hand in pixel space, thumb on the right."""
    rng = random.Random(seed)
    pts = [(0.0, 0.0)]
    for finger, base_x in enumerate([1.2, 0.6, 0.0, -0.5, -1.0]):
        for joint in range(1, 5):
            pts.append((base_x * (1 + 0.1 * joint) + rng.uniform(-0.05, 0.05),
                        -joint * (0.9 if finger else 0.5) - 1.0 + rng.uniform(-0.05, 0.05)))
    pts[9] = (0.0, -1.6)          # middle knuckle straight above the wrist
    return pts


def to_landmarks(pts, w, h, dx=0.0, dy=0.0, scale=1.0, angle=0.0, mirror=False):
    out = []
    c, s = math.cos(angle), math.sin(angle)
    for x, y in pts:
        if mirror:
            x = -x
        x, y = scale * (c * x - s * y) + dx, scale * (s * x + c * y) + dy
        out.append(LM(x / w, y / h))      # MediaPipe gives normalised coords
    return out


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestNormalization(unittest.TestCase):

    W, H = 640, 480

    def feats(self, **kw):
        from features import normalize_landmarks
        return normalize_landmarks(to_landmarks(hand(), self.W, self.H, **kw),
                                   self.W, self.H)

    def assertSame(self, a, b):
        self.assertLess(max(abs(x - y) for x, y in zip(a, b)), 1e-9)

    def test_shape(self):
        f = self.feats(dx=300, dy=300, scale=60)
        self.assertEqual(f.shape, (42,))
        self.assertAlmostEqual(f[0], 0.0)       # wrist at the origin
        self.assertAlmostEqual(f[1], 0.0)
        self.assertAlmostEqual(math.hypot(f[18], f[19]), 1.0)   # unit = wrist->middle knuckle
        self.assertGreater(f[4], 0)             # thumb on +x

    def test_position_in_frame_does_not_matter(self):
        self.assertSame(self.feats(dx=100, dy=200, scale=60),
                        self.feats(dx=500, dy=350, scale=60))

    def test_distance_from_camera_does_not_matter(self):
        self.assertSame(self.feats(dx=300, dy=300, scale=30),
                        self.feats(dx=300, dy=300, scale=120))

    def test_in_plane_rotation_does_not_matter(self):
        self.assertSame(self.feats(dx=300, dy=300, scale=60),
                        self.feats(dx=300, dy=300, scale=60, angle=0.7))

    def test_left_and_right_hand_match(self):
        self.assertSame(self.feats(dx=300, dy=300, scale=60),
                        self.feats(dx=300, dy=300, scale=60, mirror=True))

    def test_non_square_frame_is_handled(self):
        # Coordinates are scaled by frame size before the geometry, so a hand
        # in a 4:3 frame is not squashed relative to a 16:9 one.
        from features import normalize_landmarks
        pts = hand()
        a = normalize_landmarks(to_landmarks(pts, 640, 480, dx=300, dy=240, scale=60), 640, 480)
        b = normalize_landmarks(to_landmarks(pts, 1280, 720, dx=600, dy=360, scale=60), 1280, 720)
        self.assertSame(a, b)


class TestOneDefinition(unittest.TestCase):
    """The normalization used to be copied into three files."""

    def test_scripts_import_it_rather_than_define_it(self):
        for name in ("main.py", "evaluate.py", "collect_data.py"):
            with open(os.path.join(HERE, name)) as fh:
                tree = ast.parse(fh.read())
            defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            self.assertNotIn("normalize_landmarks", defined, name)
            imported = {(n.module, a.name) for n in ast.walk(tree)
                        if isinstance(n, ast.ImportFrom) for a in n.names}
            self.assertIn(("features", "normalize_landmarks"), imported, name)



LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "HELLO", "I", "IHATEYOU",
          "ILY", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y"]


class TestOutputNames(unittest.TestCase):
    """Each model output must be named after the class it really is."""

    def test_every_label_trained(self):
        from features import output_names
        self.assertEqual(output_names(range(27), LABELS), LABELS)

    def test_model_missing_some_labels(self):
        # First-person data without the word signs: 24 outputs, 27 labels.
        # Pairing them by position read I as HELLO and L as IHATEYOU.
        from features import output_names
        classes = [i for i, l in enumerate(LABELS) if len(l) == 1]
        names = output_names(classes, LABELS)
        self.assertEqual(len(names), 24)
        self.assertEqual(names, [l for l in LABELS if len(l) == 1])
        self.assertEqual(names[8], "I")
        self.assertNotIn("HELLO", names)

    def test_names_already_one_per_output(self):
        from features import output_names
        classes = [i for i, l in enumerate(LABELS) if len(l) == 1]
        per_output = [LABELS[c] for c in classes]
        self.assertEqual(output_names(classes, per_output), per_output)

    def test_mismatch_is_refused(self):
        from features import output_names
        with self.assertRaises(ValueError):
            output_names(range(27), LABELS[:20])


try:
    import sklearn  # noqa: F401
    import joblib  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


@unittest.skipUnless(HAVE_SKLEARN and HAVE_NUMPY, "scikit-learn not installed")
class TestTrainedModelRoundTrip(unittest.TestCase):
    """Train on a subset of labels, save as train_classifier.py does, load as
    main.py does, and check the words that come out."""

    @staticmethod
    def sample(rng, c, n):
        """Easily separable stand-in features: class c lives along axis c."""
        import numpy as np
        X = rng.normal(0, 0.05, (n, 42))
        X[:, c] += 3.0
        return X

    def test_subset_model_says_the_right_words(self):
        import numpy as np
        from sklearn.neural_network import MLPClassifier
        from features import load_classifier, output_names
        rng = np.random.RandomState(0)
        present = [0, 1, 9, 13]                       # A, B, I, L: no word signs
        X = np.vstack([self.sample(rng, c, 40) for c in present])
        y = np.repeat(present, 40)
        model = MLPClassifier((16,), max_iter=500, random_state=0).fit(X, y)
        model.label_names_ = output_names(model.classes_, LABELS)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.joblib")
            import joblib
            joblib.dump(model, path)
            loaded, labels = load_classifier(path, labels_path="nonexistent.json")
        for c, name in zip(present, ["A", "B", "I", "L"]):
            probs = loaded.predict_proba(self.sample(rng, c, 1))[0]
            self.assertEqual(labels[int(np.argmax(probs))], name)

    def test_older_model_with_full_label_list_still_works(self):
        # A model saved before the fix carries all 27 names; the mapping has
        # to recover the right ones from classes_.
        import numpy as np
        from sklearn.neural_network import MLPClassifier
        from features import load_classifier
        rng = np.random.RandomState(1)
        present = [0, 9, 26]                          # A, I, Y
        X = np.vstack([self.sample(rng, c, 40) for c in present])
        model = MLPClassifier((16,), max_iter=500, random_state=0).fit(
            X, np.repeat(present, 40))
        model.label_names_ = list(LABELS)             # the old, buggy save
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.joblib")
            import joblib
            joblib.dump(model, path)
            loaded, labels = load_classifier(path)
        self.assertEqual(labels, ["A", "I", "Y"])

    def test_shipped_model_loads_with_its_labels(self):
        from features import load_classifier
        here = os.path.dirname(os.path.abspath(__file__))
        model, labels = load_classifier(os.path.join(here, "landmark_model.joblib"),
                                        os.path.join(here, "landmark_labels.json"))
        self.assertEqual(len(labels), len(model.classes_))
        self.assertEqual(labels, LABELS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
