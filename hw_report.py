"""Summarise eval_results.csv by hardware profile.

Answers the two questions the embedded-target decision turns on:
  does accuracy survive a lower capture resolution, and how slow can the
  frame rate get before recognition degrades.

J and Z are broken out separately because they fail first and for a different
reason: they need 8 trajectory samples inside a 650ms window, so below about
10.8fps they cannot fire at all however well the gesture is performed. Static
letters have no such cliff and degrade gradually, if at all.

    python hw_report.py
"""
import collections
import csv
import sys

import hwprofile

RESULTS_PATH = "eval_results.csv"
MOTION = ("J", "Z")


def main():
    try:
        rows = list(csv.DictReader(open(RESULTS_PATH, newline="")))
    except FileNotFoundError:
        sys.exit(f"{RESULTS_PATH} not found -- run evaluate.py first.")
    if not rows or "profile" not in rows[0]:
        sys.exit(f"{RESULTS_PATH} predates the profile column; run evaluate.py "
                 f"once to migrate it.")

    by = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    fps = collections.defaultdict(list)
    for r in rows:
        prof = r["profile"] or "native@uncapped"
        ok = r["correct"] == "True"
        kind = "motion" if r["expected"] in MOTION else "static"
        by[prof]["all"][0] += ok;    by[prof]["all"][1] += 1
        by[prof][kind][0] += ok;     by[prof][kind][1] += 1
        if r.get("fps_actual"):
            try: fps[prof].append(float(r["fps_actual"]))
            except ValueError: pass

    floor = hwprofile.motion_floor_fps()
    print(f"eval_results.csv by hardware profile   (J/Z floor: {floor:.1f}fps)\n")
    print(f"  {'profile':22} {'overall':>9} {'static':>9} {'J/Z':>9} {'fps':>7}")
    for prof in sorted(by):
        a, s, m = by[prof]["all"], by[prof]["static"], by[prof]["motion"]
        f = f"{sum(fps[prof])/len(fps[prof]):.1f}" if fps.get(prof) else "-"
        def pct(c):
            return f"{c[0]}/{c[1]}" if c[1] else "-"
        print(f"  {prof:22} {pct(a):>9} {pct(s):>9} {pct(m):>9} {f:>7}")

    print("\n  overall/static/J-Z are correct/total. A profile whose J/Z column")
    print("  collapses while static holds up is hitting the motion floor, not a")
    print("  general accuracy problem.")


if __name__ == "__main__":
    main()
