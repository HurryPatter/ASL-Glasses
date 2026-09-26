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
    parser.add_argument("--mirrored-input", choices=["true", "false"],
                        help="re-derive every row under this handedness "
                             "convention, overriding what each clip recorded. "
                             "This is the repair path for an archive collected "
                             "under the wrong one -- nobody signs again.")
    args = parser.parse_args()
    override = None if args.mirrored_input is None else (
        args.mirrored_input == "true")

    if not os.path.exists(args.clips):
        sys.exit(f"{args.clips} not found -- nothing collected yet. "
                 f"Run `python collect_signs.py` first.")

    clips = list(signset.read_clips(args.clips))
    if not clips:
        sys.exit(f"{args.clips} is empty.")

    rows = []
    thin_face = []
    unstable = []
    poorly_tracked = []
    for clip in clips:
        rows.append(signset.clip_to_row(clip, mirrored_input=override))
        if signset.face_coverage(clip) < signset.MIN_FACE_COVERAGE:
            thin_face.append(clip["clip_id"])
        if signset.handedness_stability(clip) < signset.MIN_HANDEDNESS_STABILITY:
            unstable.append(clip["clip_id"])
        if signset.hand_coverage(clip) < signset.MIN_HAND_COVERAGE:
            poorly_tracked.append(clip["clip_id"])

    if override is not None:
        print(f"Overriding the recorded convention: mirrored_input={override}")
    conventions = {c.get("mirrored_input", True) for c in clips}
    if override is None and len(conventions) > 1:
        print("\n  MIXED CONVENTIONS in the archive: clips were collected under\n"
              "  both mirrored_input settings. Re-derive them all under one with\n"
              "  `python rebuild_signs.py --mirrored-input true|false`.")

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

    if poorly_tracked:
        print(f"\n  Poor hand tracking ({len(poorly_tracked)}): the tracker "
              f"lost the hands for\n  much of these signs. Usable, but they are "
              f"the weakest rows in the set.")
        print(f"  {', '.join(poorly_tracked[:8])}"
              f"{' ...' if len(poorly_tracked) > 8 else ''}")

    if unstable:
        print(f"\n  Unstable handedness ({len(unstable)}): MediaPipe disagreed "
              f"with itself\n  about which hand it was seeing. Identity is "
              f"resolved over the whole\n  clip, so these are usable, but the "
              f"tracking underneath is poor.")
        print(f"  {', '.join(unstable[:8])}{' ...' if len(unstable) > 8 else ''}")

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
