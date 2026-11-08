"""Schema and grouping helpers for landmark_data.csv.

Deliberately standard-library only: the tests for this module run in CI,
where neither pandas nor numpy is installed.

The dataset has one row per captured frame:

    label, person, view, p0_x, p0_y, ..., p20_x, p20_y      (3 + 42 columns)

`person` exists because a random train/test split over these rows badly
overstates accuracy. Frames inside one recording burst are near-duplicates of
a single held pose -- measured on the collected data, consecutive frames in a
burst are about five times closer together than two random frames of the same
letter. A shuffled split therefore puts frame 200 in training and frame 201 in
test, and measures memorisation. Holding out a whole *person* is the estimate
that matches what a new signer experiences: on the first three people that was
81.7% against the shuffled split's 98.6%.

`view` says where the camera was. Every row before the glasses is `front`: a
signer facing a laptop or desk camera. The glasses see the signer's own hands
from behind and above (`firstperson`), which is a different input to the
model, not more of the same. Keeping both in one file with the view on every
row lets training use either or both, and lets evaluation train on one view
and test on the other, which is the question the glasses chapter asks.
"""
import csv
import os

LANDMARK_COLUMNS = [f"p{i}_{axis}" for i in range(21) for axis in ("x", "y")]
HEADER = ["label", "person", "view"] + LANDMARK_COLUMNS
PRE_VIEW_HEADER = ["label", "person"] + LANDMARK_COLUMNS
LEGACY_HEADER = ["label"] + LANDMARK_COLUMNS

UNKNOWN_PERSON = "unknown"

FRONT, FIRSTPERSON = "front", "firstperson"
VIEWS = (FRONT, FIRSTPERSON)


def has_person_column(header):
    return "person" in header


def add_view_column(path):
    """Rewrite a pre-`view` landmark_data.csv in place, marking every existing
    row `front` -- true of all of them, since no first-person data existed
    before the column did. Returns the number of rows migrated (0 if the file
    is already current). Refuses any other header rather than guess."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows or rows[0] == HEADER:
        return 0
    if rows[0] != PRE_VIEW_HEADER:
        raise ValueError(f"{path}: unexpected header, not migrating it")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for r in rows[1:]:
            w.writerow(r[:2] + [FRONT] + r[2:])
    os.replace(tmp, path)      # never leave a half-written dataset behind
    return len(rows) - 1


def parse_view(text):
    """'f'/'front' -> front, 'g'/'glasses'/'firstperson' -> firstperson."""
    t = (text or "").strip().lower()
    if t in ("f", FRONT):
        return FRONT
    if t in ("g", "glasses", "fp", FIRSTPERSON):
        return FIRSTPERSON
    return None


def read_header(path):
    with open(path, newline="") as fh:
        return next(csv.reader(fh))


def infer_sessions(labels):
    """Map each row to a recording-session index, for the historical file only.

    collect_data.py appends rows in capture order, and each session was one
    pass through the alphabet, so a label reappearing means a new pass has
    started. This is how the pre-`person` rows get attributed; data collected
    from now on carries the person explicitly and must not be run through this.
    """
    sessions = [0] * len(labels)
    index = 0
    seen = set()
    for label, start, end in bursts(labels):
        if label in seen:          # this pass already covered it -> new pass
            index += 1
            seen = set()
        seen.add(label)
        for i in range(start, end):
            sessions[i] = index
    return sessions


def bursts(labels):
    """(label, start, end) for each run of identical consecutive labels.

    One burst is one uninterrupted press of SPACE in collect_data.py, i.e. one
    continuous recording of one pose -- the unit that is genuinely independent,
    as opposed to the individual frames inside it.
    """
    out = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((labels[start], start, i))
            start = i
    return out
