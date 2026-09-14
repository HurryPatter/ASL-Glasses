"""Schema and grouping helpers for landmark_data.csv.

Deliberately standard-library only: the tests for this module run in CI,
where neither pandas nor numpy is installed.

The dataset has one row per captured frame:

    label, person, p0_x, p0_y, ..., p20_x, p20_y      (2 + 42 columns)

`person` exists because a random train/test split over these rows badly
overstates accuracy. Frames inside one recording burst are near-duplicates of
a single held pose -- measured on the collected data, consecutive frames in a
burst are about five times closer together than two random frames of the same
letter. A shuffled split therefore puts frame 200 in training and frame 201 in
test, and measures memorisation. Holding out a whole *person* is the estimate
that matches what a new signer experiences: on the first three people that was
81.7% against the shuffled split's 98.6%.
"""
import csv

LANDMARK_COLUMNS = [f"p{i}_{axis}" for i in range(21) for axis in ("x", "y")]
HEADER = ["label", "person"] + LANDMARK_COLUMNS
LEGACY_HEADER = ["label"] + LANDMARK_COLUMNS

UNKNOWN_PERSON = "unknown"


def has_person_column(header):
    return "person" in header


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
