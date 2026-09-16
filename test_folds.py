"""Offline tests for leave-one-person-out fold construction.

Standard library only, so it runs in CI.

These matter more than most. A bug here does not crash and does not look
wrong -- it produces an accuracy figure that is simply too high, which is the
one failure mode this project has already been bitten by once: the letter
model read 98.6% on a shuffled split and 84.0% on a held-out person.

Run:  python -m unittest test_folds -v
"""
import unittest

import folds
import signset


def dataset(spec):
    """{person: [labels...]} -> parallel (labels, people) lists."""
    labels, people = [], []
    for person, person_labels in spec.items():
        for label in person_labels:
            labels.append(label)
            people.append(person)
    return labels, people


TWO_PEOPLE = {
    "omar": ["HELLO", "HELLO", "MOTHER", "FATHER", signset.REST_LABEL],
    "laila": ["HELLO", "MOTHER", "MOTHER", "FATHER", signset.REST_LABEL],
}


class TestFoldConstruction(unittest.TestCase):

    def test_one_fold_per_person(self):
        labels, people = dataset(TWO_PEOPLE)
        self.assertEqual([held for held, _, _ in folds.folds(labels, people)],
                         ["laila", "omar"])

    def test_a_persons_clips_are_never_split_across_train_and_test(self):
        # The single property the whole measurement rests on.
        labels, people = dataset(TWO_PEOPLE)
        for held, train, test in folds.folds(labels, people):
            self.assertTrue(all(people[i] != held for i in train))
            self.assertTrue(all(people[i] == held for i in test))

    def test_train_and_test_are_disjoint_and_complete(self):
        labels, people = dataset(TWO_PEOPLE)
        for _, train, test in folds.folds(labels, people):
            self.assertEqual(set(train) & set(test), set())
            self.assertEqual(sorted(train + test), list(range(len(labels))))

    def test_every_clip_is_tested_exactly_once_across_all_folds(self):
        labels, people = dataset(TWO_PEOPLE)
        tested = [i for _, _, test in folds.folds(labels, people) for i in test]
        self.assertEqual(sorted(tested), list(range(len(labels))))

    def test_a_single_person_yields_a_useless_fold(self):
        # Not an error, but problems() has to say so -- training on nobody is
        # what makes the resulting number meaningless rather than merely low.
        labels, people = dataset({"omar": ["HELLO", "MOTHER"]})
        result = folds.folds(labels, people)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], [])            # nothing to train on

    def test_empty_dataset(self):
        self.assertEqual(folds.folds([], []), [])


class TestSingleSignerLabels(unittest.TestCase):
    """A label one person signed scores ~0 on their fold for a reason that has
    nothing to do with the model, and drags the headline down with it."""

    def test_detected(self):
        labels, people = dataset({
            "omar": ["HELLO", "THANK-YOU"],
            "laila": ["HELLO"],
        })
        self.assertEqual(folds.single_signer_labels(labels, people),
                         ["THANK-YOU"])

    def test_many_clips_by_one_person_is_still_one_signer(self):
        # The trap: 40 clips looks like plenty of data, and is still
        # untestable across people.
        labels, people = dataset({
            "omar": ["HELLO"] * 40 + ["MOTHER"],
            "laila": ["MOTHER"],
        })
        self.assertEqual(folds.single_signer_labels(labels, people), ["HELLO"])

    def test_mask_excludes_exactly_those_rows(self):
        labels, people = dataset({
            "omar": ["HELLO", "THANK-YOU"],
            "laila": ["HELLO"],
        })
        mask = folds.testable_mask(labels, people)
        self.assertEqual(mask, [True, False, True])

    def test_nothing_excluded_when_everyone_signed_everything(self):
        labels, people = dataset(TWO_PEOPLE)
        self.assertEqual(folds.single_signer_labels(labels, people), [])
        self.assertTrue(all(folds.testable_mask(labels, people)))


class TestRestClass(unittest.TestCase):

    def test_rest_is_separable_from_signs(self):
        labels = ["HELLO", signset.REST_LABEL, "MOTHER"]
        self.assertEqual(folds.sign_mask(labels), [True, False, True])

    def test_rest_is_flagged_as_kept_out_of_the_headline(self):
        labels, people = dataset(TWO_PEOPLE)
        notes = " ".join(folds.problems(labels, people))
        self.assertIn(signset.REST_LABEL, notes)
        self.assertIn("separately", notes)

    def test_missing_rest_is_also_flagged(self):
        labels, people = dataset({
            "omar": ["HELLO", "MOTHER"], "laila": ["HELLO", "MOTHER"]})
        notes = " ".join(folds.problems(labels, people))
        self.assertIn("No _REST clips", notes)


class TestProblems(unittest.TestCase):
    """The caveats are printed before the number, not after."""

    def test_one_person_is_reported_as_unmeasurable(self):
        labels, people = dataset({"omar": ["HELLO"] * 50})
        notes = folds.problems(labels, people)
        self.assertEqual(len(notes), 1)
        self.assertIn("cannot be measured", notes[0])

    def test_uneven_label_coverage_is_reported(self):
        labels, people = dataset({
            "omar": [f"SIGN{i}" for i in range(20)],
            "laila": [f"SIGN{i}" for i in range(3)],
        })
        notes = " ".join(folds.problems(labels, people))
        self.assertIn("different numbers of labels", notes)

    def test_even_coverage_is_not_reported(self):
        labels, people = dataset({
            "omar": [f"SIGN{i}" for i in range(20)],
            "laila": [f"SIGN{i}" for i in range(20)],
        })
        notes = " ".join(folds.problems(labels, people))
        self.assertNotIn("different numbers of labels", notes)

    def test_thin_folds_are_reported(self):
        labels, people = dataset({
            "omar": ["HELLO"] * 40, "laila": ["HELLO"] * 2})
        notes = " ".join(folds.problems(labels, people))
        self.assertIn("Thin folds", notes)
        self.assertIn("laila", notes)

    def test_a_healthy_dataset_reports_only_the_rest_note(self):
        spec = {person: [f"SIGN{i}" for i in range(20)] * 2 + [signset.REST_LABEL]
                for person in ("omar", "laila", "riad", "nourhan")}
        labels, people = dataset(spec)
        notes = folds.problems(labels, people)
        self.assertEqual(len(notes), 1)
        self.assertIn(signset.REST_LABEL, notes[0])


class TestCoverageHelpers(unittest.TestCase):

    def test_per_person_labels(self):
        labels, people = dataset(TWO_PEOPLE)
        coverage = folds.per_person_labels(labels, people)
        self.assertEqual(coverage["omar"],
                         {"HELLO", "MOTHER", "FATHER", signset.REST_LABEL})

    def test_clip_counts(self):
        labels, people = dataset(TWO_PEOPLE)
        self.assertEqual(folds.clip_counts(people), {"omar": 5, "laila": 5})

    def test_label_signers(self):
        labels, people = dataset({
            "omar": ["HELLO", "MOTHER"], "laila": ["HELLO"]})
        signers = folds.label_signers(labels, people)
        self.assertEqual(signers["HELLO"], {"omar", "laila"})
        self.assertEqual(signers["MOTHER"], {"omar"})


if __name__ == "__main__":
    unittest.main()
