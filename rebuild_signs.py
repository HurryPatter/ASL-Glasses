"""
Regenerate veronica_signs.csv from the raw clip archive.

The archive (`veronica_clips.jsonl`) is the record; the CSV is a build
artifact. Every time the feature layers change -- a keyframe count in
sequence.py, a new block in hands.py, anything stage 5 onward wants to try --
this rebuilds every training row from the original landmarks instead of asking
the signers back.

That is the whole reason collection stores raw landmarks at all. The letter
dataset learned it in reverse: `normalize_landmarks()` ran before anything
reached disk, so orientation is gone from those 24,496 rows for good and no
script can recover it.

    python rebuild_signs.py                  # rebuild in place
    python rebuild_signs.py --check          # report only, write nothing
"""
import argparse
import os
import sys

import signset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="summarise the archive without writing the CSV")
    parser.add_argument("--clips", default=signset.CLIPS_PATH)
    parser.add_argument("--out", default=signset.SIGNS_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.clips):
        sys.exit(f"{args.clips} not found -- nothing collected yet. "
                 f"Run `python collect_signs.py` first.")

    clips = list(signset.read_clips(args.clips))
    if not clips:
        sys.exit(f"{args.clips} is empty.")

    rows = []
    thin_face = []
    for clip in clips:
        rows.append(signset.clip_to_row(clip))
        if signset.face_coverage(clip) < signset.MIN_FACE_COVERAGE:
            thin_face.append(clip["clip_id"])

    counts = signset.counts_by_label(clips)
    everyone = signset.people(clips)

    print(f"{len(clips)} clips, {len(counts)} signs, {len(everyone)} people: "
          f"{', '.join(everyone)}")
    print(f"{len(signset.HEADER)} columns "
          f"({len(signset.META_COLUMNS)} metadata + "
          f"{len(signset.HEADER) - len(signset.META_COLUMNS)} features)")

    # A label only one person ever signed cannot be validated across people --
    # on the fold holding that person out it is absent from training entirely.
    # train_classifier.py already excludes such labels from its headline; this
    # is the same warning early enough to do something about it.
    signers = {}
    for clip in clips:
        signers.setdefault(clip["label"], set()).add(clip["person"])
    single = sorted(l for l, ps in signers.items() if len(ps) < 2)
    if single:
        print(f"\n  NOT VALIDATED -- only one signer: {', '.join(single)}")
        print("  Cross-person accuracy cannot be measured for these.")

    missing = [s for s in signset.SIGNS if s not in counts]
    if missing:
        print(f"\n  Never recorded ({len(missing)}): {', '.join(missing[:15])}"
              f"{' ...' if len(missing) > 15 else ''}")

    if thin_face:
        print(f"\n  Low face coverage ({len(thin_face)}): location features are "
              f"mostly empty in these clips.")
        print(f"  {', '.join(thin_face[:8])}{' ...' if len(thin_face) > 8 else ''}")

    if args.check:
        print("\n--check: nothing written.")
        return

    signset.write_rows(args.out, rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
