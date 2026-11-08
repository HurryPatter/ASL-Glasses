"""Tests for nlp_bridge.py. Needs symspellpy (installed in CI for these)."""
import ast
import os
import unittest

try:
    import symspellpy  # noqa: F401
    HAVE_SYMSPELL = True
except ImportError:
    HAVE_SYMSPELL = False

HERE = os.path.dirname(os.path.abspath(__file__))


class TestNoPkgResources(unittest.TestCase):
    """setuptools 84 removed pkg_resources, and a fresh install -- the Pi's
    first -- crashed on this module's import line. Checked on the syntax tree
    rather than the text, so a comment explaining the history is allowed."""

    def test_nlp_bridge_does_not_import_pkg_resources(self):
        tree = ast.parse(open(os.path.join(HERE, "nlp_bridge.py")).read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("pkg_resources", imported)


@unittest.skipUnless(HAVE_SYMSPELL, "symspellpy not installed")
class TestCorrectSequence(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from nlp_bridge import NLPBridge
        cls.NLPBridge = NLPBridge
        cls.nlp = NLPBridge(mode="academic")   # ~3s: loads the dictionary once

    def setUp(self):
        self.nlp._last_labels = None           # isolate tests from each other

    def test_dictionary_loads_and_corrects(self):
        self.assertEqual(self.nlp.correct_sequence(list("HELLO")), "Hello")

    def test_word_signs_pass_through_untouched(self):
        self.assertEqual(self.nlp.correct_sequence(["ILY"]), "I love you")

    def test_mixed_letters_and_word_signs(self):
        out = self.nlp.correct_sequence(list("HELLO") + ["ILY"])
        self.assertEqual(out, "Hello I love you")

    def test_empty(self):
        self.assertEqual(self.nlp.correct_sequence([]), "")

    def _count_corrections(self):
        """Wrap correct() so the test can see how often real work happens."""
        calls = {"n": 0}
        original = self.nlp.correct

        def counting(s):
            calls["n"] += 1
            return original(s)
        self.nlp.correct = counting
        self.addCleanup(lambda: setattr(self.nlp, "correct", original))
        return calls

    def test_unchanged_labels_are_not_recomputed(self):
        # main.py calls this every frame; this is the whole point of the cache.
        calls = self._count_corrections()
        labels = list("HELLOWORLD")
        first = self.nlp.correct_sequence(labels)
        for _ in range(100):
            self.assertEqual(self.nlp.correct_sequence(labels), first)
        self.assertEqual(calls["n"], 1)

    def test_a_new_commit_is_recomputed(self):
        calls = self._count_corrections()
        self.nlp.correct_sequence(list("HELL"))
        self.assertEqual(self.nlp.correct_sequence(list("HELLO")), "Hello")
        self.assertEqual(calls["n"], 2)

    def test_mutating_the_callers_list_does_not_serve_a_stale_result(self):
        # The classic memoisation bug: caching by reference. main.py rebuilds
        # its list each frame, but a caller that appends in place must still
        # get a fresh answer.
        labels = list("HELL")
        self.nlp.correct_sequence(labels)
        labels.append("O")
        self.assertEqual(self.nlp.correct_sequence(labels), "Hello")

    def test_retroactive_removal_is_recomputed(self):
        # commit_motion() can drop earlier commits, so the list can shrink.
        self.nlp.correct_sequence(list("CATX"))
        self.assertEqual(self.nlp.correct_sequence(list("CAT")), "Cat")

    def test_changing_mode_invalidates_the_cache(self):
        calls = self._count_corrections()
        labels = list("LATENCY")
        self.nlp.correct_sequence(labels)
        self.nlp.set_mode("academic")
        self.nlp.correct_sequence(labels)
        self.assertEqual(calls["n"], 2)



@unittest.skipUnless(HAVE_SYMSPELL, "symspellpy not installed")
class TestFingerspellingCorrection(unittest.TestCase):
    """What a signer actually produces: one letter per hold, no spaces."""

    @classmethod
    def setUpClass(cls):
        from nlp_bridge import NLPBridge
        cls.nlp = NLPBridge(mode="academic")

    def check(self, signed, expected):
        self.assertEqual(self.nlp.correct(signed), expected)

    def test_one_letter_words_survive(self):
        # Correction used to delete them: "Am Omar", "Love You", "Bag".
        self.check("IAMOMAR", "I am Omar")
        self.check("ILOVEYOU", "I love you")
        self.check("ABAG", "A bag")
        self.check("CANIGO", "Can I go")

    def test_correctly_spelled_sentences_are_only_split(self):
        self.check("MYNAMEISLAILA", "My name is Laila")
        self.check("WHATISYOURNAME", "What is your name")
        self.check("IWANTWATER", "I want water")

    def test_missed_double_letter(self):
        # The debouncer needs the hand dropped between the two Ls of HELLO,
        # so a single L is the most likely slip. Plain correction read "Help".
        self.check("HELO", "Hello")
        self.check("APLE", "Apple")
        self.check("HELOMYNAMEISOMAR", "Hello my name is Omar")

    def test_small_slips_are_corrected(self):
        self.check("HWLLO", "Hello")
        self.check("THNAKYOU", "Thank you")
        self.check("IAMHWNGRY", "I am hungry")

    def test_noise_is_left_as_signed(self):
        # A guess is worse than the letters: XQZ used to become "xxx", and a
        # stray X after CAT became the word "a".
        self.check("XQZ", "Xqz")
        self.check("CATX", "Cat x")

    def test_names_are_capitalised(self):
        self.check("HAGAR", "Hagar")


if __name__ == "__main__":
    unittest.main(verbosity=2)
