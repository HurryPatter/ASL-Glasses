"""One-off: add the `person` column to landmark data collected before it existed.

The historical landmark_data.csv has no `person` column, but the rows are in
capture order and each session was one full pass through the alphabet, so the
sessions are recoverable: a label reappearing marks the start of a new pass.

Run once. It is idempotent -- a file that already has the column is left alone.
Data collected from now on carries the person from collect_data.py and must not
be passed through here.

    python backfill_person.py [--dry-run]
"""
import csv
import shutil
import sys

import dataset

DATA_PATH = "landmark_data.csv"

# Session order in the historical file, confirmed against the recording
# structure: three complete alphabet passes, one per person.
SESSION_PEOPLE = ["omar", "laila", "nourhan"]

# The word signs sit between the second and third alphabet passes, but all
# three people took turns recording them in that one sitting. Because the rows
# are one continuous run per label, there is no boundary in the file marking
# where one signer stopped and the next began, so per-row attribution is not
# recoverable -- and guessing it would silently corrupt the grouping that every
# accuracy number depends on. They are marked `unknown`: train_classifier.py
# uses such rows for training on every fold and never tests on them.
WORD_SIGN_PERSON = dataset.UNKNOWN_PERSON


def main(dry_run=False):
    header = dataset.read_header(DATA_PATH)
    if dataset.has_person_column(header):
        print(f"{DATA_PATH} already has a `person` column; nothing to do.")
        return 0
    if header != dataset.LEGACY_HEADER:
        print(f"Unexpected header in {DATA_PATH}; refusing to guess.", file=sys.stderr)
        return 1

    with open(DATA_PATH, newline="") as fh:
        rows = list(csv.reader(fh))[1:]

    labels = [r[0] for r in rows]
    sessions = dataset.infer_sessions(labels)
    n_sessions = max(sessions) + 1

    if n_sessions != len(SESSION_PEOPLE):
        print(f"Found {n_sessions} sessions but {len(SESSION_PEOPLE)} names are "
              f"configured. Refusing to guess -- update SESSION_PEOPLE.",
              file=sys.stderr)
        return 1

    people = [SESSION_PEOPLE[s] for s in sessions]
    if WORD_SIGN_PERSON is not None:
        people = [WORD_SIGN_PERSON if len(l) > 1 else p
                  for l, p in zip(labels, people)]

    print(f"{len(rows)} rows across {n_sessions} sessions:")
    for i, name in enumerate(SESSION_PEOPLE):
        n = sum(1 for p in people if p == name)
        labels_here = sorted({l for l, s in zip(labels, sessions) if s == i})
        print(f"  {name:10} {n:6} rows, {len(labels_here):2} labels")
        words = [l for l in labels_here if len(l) > 1]
        if words:
            print(f"             word signs: {', '.join(words)}")

    if dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    shutil.copyfile(DATA_PATH, DATA_PATH + ".bak")
    with open(DATA_PATH, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(dataset.HEADER)
        for person, row in zip(people, rows):
            writer.writerow([row[0], person] + row[1:])

    print(f"\nWrote {DATA_PATH} with a `person` column "
          f"(previous file kept as {DATA_PATH}.bak).")
    return 0


if __name__ == "__main__":
    sys.exit(main(dry_run="--dry-run" in sys.argv))
