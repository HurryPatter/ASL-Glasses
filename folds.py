"""Honest measurement for the sign dataset — Project Veronica stage 5.

The strongest methodological thing in this repository is that
`train_classifier.py` reports **leave-one-person-out** accuracy as the headline
and prints the random-split figure only as an explicitly inflated comparison.
On the letter data those two numbers are 84.0% and 98.6%. Quoting the wrong one
is not a rounding error; it is the difference between a result and a mistake.

This module is the part of that discipline that is pure logic, extracted so it
can be tested in CI. `train_signs.py` is a thin scikit-learn wrapper over it.
The sklearn call is the part hardest to get wrong; deciding *what goes in which
fold* is the part where a silent mistake produces a number that looks fine.

What is different about clips
-----------------------------
For the letter dataset the leakage was frames inside one recording burst —
near-duplicates of a single held pose, about 5x closer to each other than two
random frames of the same letter. A shuffled split trained on frame 200 and
tested on frame 201.

Clips are much more independent than frames: each is a separate attempt. But
the same trap is still there one level up — twenty clips of HELLO recorded by
one person in one sitting are far more like each other than like anyone else's
HELLO. **Grouping by person closes both**, because every clip a person
contributed lands in the same fold. That is why nothing here groups by clip.

Three ways a fold can quietly lie
---------------------------------
1. **A label only one person ever signed.** On the fold holding that person
   out it is absent from training entirely, so it scores ~0 and drags the
   headline down for a reason that has nothing to do with the model.
   `train_classifier.py` already excludes these; so does this.
2. **`_REST` counted as a sign.** It is a real class the model must learn, and
   it is much easier than any sign, so including it in the headline inflates
   the number. Reported separately.
3. **Folds that are not comparable.** If one person signed 68 labels and
   another 20, their folds measure different tasks and averaging them without
   saying so hides that.
"""
import signset

# Below this, a fold's number is too noisy to mean much on its own -- it goes
# in the mean, but quoting it per-person invites reading noise as a finding.
MIN_CLIPS_PER_FOLD = 30


def label_signers(labels, people):
    """label -> set of people who signed it."""
    signers = {}
    for label, person in zip(labels, people):
        signers.setdefault(label, set()).add(person)
    return signers


def single_signer_labels(labels, people):
    """Labels no two people signed, so cross-person accuracy is undefined."""
    return sorted(l for l, ps in label_signers(labels, people).items()
                  if len(ps) < 2)


def testable_mask(labels, people):
    """True for each row that can legitimately appear in a headline number."""
    excluded = set(single_signer_labels(labels, people))
    return [label not in excluded for label in labels]


def sign_mask(labels):
    """True for rows that are an actual sign, i.e. not the _REST class."""
    return [label != signset.REST_LABEL for label in labels]


def folds(labels, people):
    """[(held_out_person, train_indices, test_indices)], one per person.

    Every clip a person contributed is in their own fold and nowhere else --
    that single property is what makes the resulting number the one a stranger
    at a demo experiences, rather than a measure of memorisation.
    """
    everyone = sorted(set(people))
    out = []
    for held in everyone:
        train = [i for i, p in enumerate(people) if p != held]
        test = [i for i, p in enumerate(people) if p == held]
        out.append((held, train, test))
    return out


def per_person_labels(labels, people):
    """person -> set of labels they signed."""
    coverage = {}
    for label, person in zip(labels, people):
        coverage.setdefault(person, set()).add(label)
    return coverage


def clip_counts(people):
    counts = {}
    for person in people:
        counts[person] = counts.get(person, 0) + 1
    return counts


def problems(labels, people):
    """Everything that would make a headline number misleading, in words.

    Printed before any accuracy figure rather than after, so the caveats are
    read as conditions on the number instead of as footnotes to it.
    """
    notes = []
    everyone = sorted(set(people))

    if len(everyone) < 2:
        notes.append(
            "Only one person in the dataset -- cross-person accuracy cannot be "
            "measured at all. Any number from this data is memorisation. "
            "Collect from someone else before quoting anything.")
        return notes

    single = single_signer_labels(labels, people)
    if single:
        notes.append(
            f"{len(single)} label(s) signed by only one person, excluded from "
            f"the headline: {', '.join(single[:10])}"
            f"{' ...' if len(single) > 10 else ''}")

    coverage = per_person_labels(labels, people)
    sizes = {p: len(ls) for p, ls in coverage.items()}
    if sizes and max(sizes.values()) - min(sizes.values()) > 5:
        spread = ", ".join(f"{p}:{n}" for p, n in sorted(sizes.items()))
        notes.append(
            f"People signed different numbers of labels ({spread}). Their folds "
            f"measure different tasks, so the per-fold numbers are not directly "
            f"comparable and the mean hides that.")

    counts = clip_counts(people)
    thin = sorted(p for p, n in counts.items() if n < MIN_CLIPS_PER_FOLD)
    if thin:
        notes.append(
            f"Thin folds (<{MIN_CLIPS_PER_FOLD} clips): {', '.join(thin)}. "
            f"Their per-person accuracy is mostly noise; read the mean, not "
            f"the individual rows.")

    if signset.REST_LABEL in set(labels):
        notes.append(
            f"{signset.REST_LABEL} is easier than any real sign, so it is "
            f"reported separately and kept out of the headline.")
    else:
        notes.append(
            f"No {signset.REST_LABEL} clips. Stage 5 has nothing to reject "
            f"garbage with and stage 6 has no way to tell 'between signs' from "
            f"'a sign'. Worth two minutes at the next session.")

    return notes
