"""
What is actually in the collected clips — Project Veronica.

    python inspect_signs.py                  # what has been collected so far
    python inspect_signs.py MOTHER FATHER    # what separates two signs

The second form is the one to reach for early. `train_signs.py` answers "how
accurate is the model", which needs several people before it means anything.
This answers a different and earlier question: **is the signal even in the
features?** That needs no model, no training and no second signer, so it is
the first thing worth asking after a first session.

MOTHER and FATHER are the pair to ask it about. They share a handshape, an
orientation and a movement, and differ only in **location** — forehead versus
chin. If the features separate them, stage 2's face anchoring is doing real
work. If they do not, no amount of training data will fix it, because the
distinction is not being recorded at all.

How separation is measured
--------------------------
Two things, both deliberately simpler than a classifier:

**Effect size** (Cohen's d) per feature: how far apart the two classes' means
are, in pooled standard deviations. This says which features carry the
difference, and the *block* they come from is as interesting as the number. If
MOTHER and FATHER separate on location features, that is the designed
behaviour. If they separate on handshape instead, the two signs are being made
with different hands shapes — worth knowing, since that is not the contrast
being tested.

**Leave-one-clip-out accuracy on the single best feature**: a threshold
classifier with exactly one parameter, tested on a clip it has never seen.
One parameter is about as little as a model can have, so this is close to a
floor — whatever a real classifier does, it should beat this.

What this cannot tell you
-------------------------
Every clip here is one person in one sitting. A high number says the features
distinguish *these recordings*, not that the model will generalise to a signer
it has never seen. That is what leave-one-person-out in `train_signs.py`
measures, and it needs a second signer. Do not quote anything from here as an
accuracy result.

Standard library only.
"""
import argparse
import csv
import math
import sys

import sequence
import signset


def load_rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def block_of(column):
    """Which part of the representation a feature column comes from.

    The point of naming these: a contrast showing up in the block you expected
    is evidence the design works, and one showing up somewhere else is a
    finding about how the signs were actually performed.
    """
    body = column
    if column[0] in "kd" and "_" in column and column[1].isdigit():
        body = column.split("_", 1)[1]
    if "_loc_" in body or body.endswith("_depth") or body == "face_present":
        return "location"
    if body.startswith("rel_"):
        return "two-hand relation"
    if "_orient_" in body or body.endswith("_mirrored"):
        return "orientation"
    if body.endswith("_present"):
        return "hand presence"
    if "_move_" in body:
        return "movement"
    if body in ("dom_coverage", "non_coverage", "duration_s"):
        return "clip"
    return "handshape"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def cohens_d(a, b):
    """Standardised difference between two groups' means."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sa, sb = stdev(a), stdev(b)
    pooled = math.sqrt(((len(a) - 1) * sa ** 2 + (len(b) - 1) * sb ** 2)
                       / (len(a) + len(b) - 2))
    if pooled < 1e-9:
        return 0.0
    return (mean(a) - mean(b)) / pooled


def leave_one_out_threshold(a, b):
    """Accuracy of a one-parameter threshold rule on held-out clips.

    The threshold is the midpoint of the two class means computed *without*
    the clip being tested, so no clip contributes to the rule that judges it.
    With a single parameter there is very little to overfit, which is what
    makes this a floor rather than a flattering number.
    """
    correct = total = 0
    for group, others, sign in ((a, b, 1), (b, a, -1)):
        for i in range(len(group)):
            rest = group[:i] + group[i + 1:]
            if len(rest) < 2 or len(others) < 2:
                continue
            threshold = (mean(rest) + mean(others)) / 2.0
            predicted = sign * (group[i] - threshold) > 0
            correct += int(predicted == (sign * (mean(rest) - mean(others)) > 0))
            total += 1
    return correct / total if total else 0.0


def summarise(rows):
    labels = {}
    people = {}
    for row in rows:
        labels[row["label"]] = labels.get(row["label"], 0) + 1
        people[row["person"]] = people.get(row["person"], 0) + 1

    print(f"{len(rows)} clips, {len(labels)} signs, {len(people)} people\n")
    print(f"  {'sign':<18} clips")
    for label in sorted(labels):
        print(f"  {label:<18} {labels[label]}")
    print(f"\n  {'person':<18} clips")
    for person in sorted(people):
        print(f"  {person:<18} {people[person]}")

    def column(name):
        return [float(r[name]) for r in rows if r.get(name)]

    faces, tracked = column("face_coverage"), column("hand_coverage")
    stability = column("handedness_stability")
    if faces:
        print(f"\n  face coverage:        min {min(faces):.0%}, "
              f"mean {mean(faces):.0%}")
    if tracked:
        print(f"  hand tracking:        min {min(tracked):.0%}, "
              f"mean {mean(tracked):.0%}")
        poor = [r["clip_id"] for r in rows
                if float(r.get("hand_coverage", 1)) < 0.7]
        if poor:
            print(f"    {len(poor)} clip(s) below 70% -- the tracker lost the "
                  f"hands for much of\n    the sign. Gap bridging repairs the "
                  f"geometry either side of a dropout,\n    but it cannot "
                  f"invent the part nobody saw. Worth re-recording:")
            print(f"    {', '.join(poor[:6])}{' ...' if len(poor) > 6 else ''}")
    if stability:
        print(f"  handedness stability: min {min(stability):.0%}, "
              f"mean {mean(stability):.0%}")

    # Which slot the hands actually landed in. This is a second, independent
    # read on the handedness convention -- and a better one than the camera
    # check, because it is measured on the clips that were really recorded.
    # A one-handed signer whose clips are nearly all non-dominant has the
    # convention inverted: their signing hand is being filed as the helper.
    dom = [float(r["d0_dom_present"]) for r in rows if "d0_dom_present" in r]
    non = [float(r["d0_non_present"]) for r in rows if "d0_non_present" in r]
    if dom and non:
        print(f"\n  dominant hand tracked:     {mean(dom):.0%} of clips")
        print(f"  non-dominant hand tracked: {mean(non):.0%} of clips")
        if mean(non) > 0.7 and mean(dom) < 0.3:
            print("\n  The signing hand is landing in the NON-dominant slot on "
                  "nearly every\n  clip. For a one-handed sign that means the "
                  "handedness convention is\n  inverted -- flip it and "
                  "re-derive, no re-recording needed:")
            print("      python check_setup.py --set-mirrored-input "
                  "<the other value>")
            print("      python rebuild_signs.py --mirrored-input <same value>")

    if len(people) < 2:
        print("\n  Only one signer. Cross-person accuracy cannot be measured at "
              "all yet --\n  every number from this data measures these "
              "recordings, not the language.")
    thin = [l for l, n in labels.items() if n < 8]
    if thin:
        print(f"\n  Few clips ({', '.join(sorted(thin))}): effect sizes below "
              f"about 8 clips\n  per sign are noisy. Treat them as a direction, "
              f"not a measurement.")


def compare(rows, first, second, top=12):
    groups = {first: [], second: []}
    for row in rows:
        if row["label"] in groups:
            groups[row["label"]].append(row)

    for label, group in groups.items():
        if len(group) < 2:
            sys.exit(f"Need at least 2 clips of {label}; found {len(group)}.")

    print(f"{first} ({len(groups[first])} clips)  vs  "
          f"{second} ({len(groups[second])} clips)\n")

    scored = []
    for column in sequence.SIGN_COLUMNS:
        a = [float(r[column]) for r in groups[first]]
        b = [float(r[column]) for r in groups[second]]
        d = cohens_d(a, b)
        if d:
            scored.append((abs(d), d, column, a, b))
    scored.sort(reverse=True)

    if not scored:
        sys.exit("No feature varies between these two signs at all.")

    print(f"  {'feature':<34} {'effect':>8}  block")
    for _, d, column, _, _ in scored[:top]:
        print(f"  {column:<34} {d:>8.2f}  {block_of(column)}")

    # Which part of the representation is carrying the contrast. This is the
    # question the design makes a prediction about, so it gets its own line.
    weight = {}
    for magnitude, _, column, _, _ in scored[:40]:
        block = block_of(column)
        weight[block] = weight.get(block, 0.0) + magnitude
    total = sum(weight.values()) or 1.0
    print("\n  where the difference lives (top 40 features):")
    for block, value in sorted(weight.items(), key=lambda kv: -kv[1]):
        print(f"    {block:<20} {value / total:>5.0%}")

    magnitude, d, column, a, b = scored[0]
    accuracy = leave_one_out_threshold(a, b)
    print(f"\n  Best single feature: {column}")
    print(f"    {first:<12} mean {mean(a):+.3f}  (sd {stdev(a):.3f})")
    print(f"    {second:<12} mean {mean(b):+.3f}  (sd {stdev(b):.3f})")
    print(f"    leave-one-clip-out accuracy, one threshold: {accuracy:.0%}")

    print()
    if magnitude > 2.0:
        print("  Cleanly separated. The distinction is in the features, so a "
              "classifier\n  has something real to learn from.")
    elif magnitude > 0.8:
        print("  Separated, but not cleanly. Usable, though expect confusion "
              "between these\n  two once other signs crowd the space.")
    else:
        print("  NOT separated. A classifier cannot learn a distinction that is "
              "not in the\n  features -- more clips will not help. Check that "
              "the two signs really were\n  made differently, and that a face "
              "was detected throughout.")
    print("  One person, one sitting: this is not an accuracy result.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("signs", nargs="*", help="two labels to compare")
    parser.add_argument("--rows", default=signset.SIGNS_PATH)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    try:
        rows = load_rows(args.rows)
    except FileNotFoundError:
        sys.exit(f"{args.rows} not found. Record some clips with "
                 f"`python collect_signs.py` first.")
    if not rows:
        sys.exit(f"{args.rows} has no rows yet.")

    if len(args.signs) == 2:
        compare(rows, args.signs[0], args.signs[1], args.top)
    elif args.signs:
        sys.exit("Give exactly two sign labels to compare, or none for a summary.")
    else:
        summarise(rows)


if __name__ == "__main__":
    main()
