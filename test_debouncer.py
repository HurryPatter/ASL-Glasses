"""Offline tests for Debouncer -- no camera, no MediaPipe, no model.

Frame streams are synthesised with explicit wall-clock timestamps, so every
timing rule is exercised deterministically at whatever frame rate we choose.

Run:  python -m unittest test_debouncer -v
      python test_debouncer.py            (same, plus the regression report)
"""
import unittest

from debouncer import Debouncer


# ── helpers ────────────────────────────────────────────────────────────────
def frames(spec, fps=30):
    """Expand a compact spec into (letter, timestamp_ms) samples.

    spec is a list of (letter, count) pairs; "" means "no reading this frame".
    """
    step = 1000.0 / fps
    out, t = [], 0.0
    for letter, count in spec:
        for _ in range(count):
            out.append((letter, t))
            t += step
    return out


def run(spec, fps=30, **kwargs):
    """Feed a spec through a Debouncer and return its committed string."""
    d = Debouncer(**kwargs)
    for i, (letter, t) in enumerate(frames(spec, fps)):
        d.update(letter, frame_no=i, now_ms=t)
    return d.confirmed_string


class LegacyDebouncer:
    """The previous implementation, kept only so the tests can demonstrate the
    regression rather than assert against a remembered description of it."""

    def __init__(self, min_frames=3):
        self.min_frames = min_frames
        self.current_letter = None
        self.frame_count = 0
        self.frame_no = 0
        self.commits = []

    @property
    def confirmed_string(self):
        return "".join(letter for letter, _ in self.commits)

    def update(self, letter, frame_no=None, now_ms=None):
        self.frame_no = self.frame_no + 1 if frame_no is None else frame_no
        if letter == self.current_letter:
            self.frame_count += 1
        else:
            self.current_letter = letter
            self.frame_count = 1
        if self.frame_count == self.min_frames and letter:
            self.commits.append((letter, self.frame_no))
        return self.confirmed_string


def run_legacy(spec, fps=30):
    d = LegacyDebouncer()
    for i, (letter, t) in enumerate(frames(spec, fps)):
        d.update(letter, frame_no=i, now_ms=t)
    return d.confirmed_string


# ── the bug this change fixes ──────────────────────────────────────────────
class TestRepeatCommitRegression(unittest.TestCase):

    def test_single_blank_frame_does_not_split_a_hold(self):
        spec = [("D", 5), ("", 1), ("D", 5)]
        self.assertEqual(run_legacy(spec), "DD")   # the bug
        self.assertEqual(run(spec), "D")           # fixed

    def test_periodic_confidence_dips_commit_once(self):
        # A 4s hold at 30fps with a 2-frame dip every 12 frames -- the pattern
        # that produced "QQQQQQQQQQ" in the white_bg_dim evaluation run.
        spec = [("Q", 12), ("", 2)] * 10
        self.assertEqual(run_legacy(spec), "Q" * 10)
        self.assertEqual(run(spec), "Q")

    def test_recorded_eval_strings_collapse_to_one_commit(self):
        # Every repeated string recorded in eval_results.csv, reproduced as a
        # steady hold punctuated by blanks.
        for letter, repeats in [("D", 8), ("I", 10), ("L", 9), ("Q", 10),
                                ("P", 10), ("X", 9), ("G", 11), ("Y", 6)]:
            spec = [(letter, 12), ("", 2)] * repeats
            with self.subTest(letter=letter):
                self.assertEqual(run_legacy(spec), letter * repeats)
                self.assertEqual(run(spec), letter)

    def test_motion_candidate_suppression_does_not_split_a_hold(self):
        # is_motion_candidate() trips on X/L/I/Y/G/Q, skipping classification
        # for scattered frames while the letter is held perfectly still.
        spec = [("X", 4), ("", 1), ("X", 3), ("", 2), ("X", 6), ("", 1), ("X", 8)]
        self.assertEqual(run_legacy(spec), "XXXX")
        self.assertEqual(run(spec), "X")

    def test_single_frame_misclassification_does_not_split_a_hold(self):
        # A stray confident frame of a different letter is below switch_ms.
        spec = [("D", 6), ("R", 1), ("D", 6)]
        self.assertEqual(run_legacy(spec), "DD")
        self.assertEqual(run(spec), "D")


class TestFlickerBetweenConfusableLetters(unittest.TestCase):
    """Both of these were found by fuzzing, not by hand."""

    def test_flicker_to_a_confusable_letter_does_not_recommit(self):
        # G/Q and R/X flicker in the recorded evaluation data. A brief flicker
        # ends the hold but must not end the letter *run*, or the original
        # letter commits a second time.
        self.assertEqual(run([("G", 10), ("Q", 3), ("G", 10)]), "G")
        self.assertEqual(run([("R", 10), ("X", 2), ("R", 10)]), "R")

    def test_a_flicker_that_is_really_held_commits_normally(self):
        self.assertEqual(run([("G", 10), ("Q", 10), ("G", 10)]), "GQG")

    def test_blanks_after_a_flicker_do_not_release_early(self):
        # Exact stream from the fuzzer (15fps): the hold sits on B while single
        # frames of C and A flicker past, so "time since the held letter was
        # seen" is already large when two blank frames arrive. Those two frames
        # are 133ms of absence, not 333ms, and must not release the run.
        spec = [("C", 6), ("B", 2), ("C", 1), ("A", 1), ("C", 1),
                ("", 2), ("C", 3)]
        self.assertEqual(run(spec, fps=15), "C")

    def test_property_no_repeat_commit_without_a_real_absence(self):
        # Randomised streams: holds, gaps, dropped frames and stray letters.
        import random
        rng = random.Random(1234)
        letters = list("ABCDEFG")
        for _ in range(4000):
            fps = rng.choice([30, 25, 20, 15, 10, 6, 5])
            step = 1000.0 / fps
            stream, t = [], 0.0
            for _ in range(rng.randint(1, 6)):
                letter = rng.choice(letters)
                for _ in range(int(rng.choice([0, 33, 100, 300, 600]) / step)):
                    stream.append(("", t)); t += step
                for _ in range(max(1, int(rng.choice([50, 150, 400, 900]) / step))):
                    r = rng.random()
                    if r < 0.12:   stream.append(("", t))
                    elif r < 0.18: stream.append((rng.choice([x for x in letters if x != letter]), t))
                    else:          stream.append((letter, t))
                    t += step
            d = Debouncer()
            for i, (lt, ts) in enumerate(stream):
                d.update(lt, frame_no=i, now_ms=ts)
            committed = [c[0] for c in d.commits]
            for a in range(len(committed) - 1):
                if committed[a] != committed[a + 1]:
                    continue
                f0, f1 = d.commits[a][1], d.commits[a + 1][1]
                run_ms = best = 0
                for lt, _ in stream[f0:f1 + 1]:
                    if lt == "":
                        run_ms += step
                        best = max(best, run_ms)
                    else:
                        run_ms = 0
                self.assertGreaterEqual(
                    best, d.blank_ms - step,
                    f"{committed[a]} committed twice with only {best:.0f}ms of "
                    f"absence between (fps={fps}): {''.join(x or '.' for x, _ in stream)}")


# ── behaviour that must be preserved ───────────────────────────────────────
class TestPreservedBehaviour(unittest.TestCase):

    def test_clean_hold_commits_once(self):
        self.assertEqual(run([("A", 30)]), "A")

    def test_hold_shorter_than_min_hold_ms_never_commits(self):
        self.assertEqual(run([("A", 2)]), "")           # 2 frames = 33ms
        self.assertEqual(run([("", 10)]), "")

    def test_deliberate_bounce_still_gives_a_double_letter(self):
        # The "LL" in HELLO: out of the shape for longer than blank_ms (250ms
        # = ~8 frames at 30fps) and back in.
        spec = [("L", 10), ("", 10), ("L", 10)]
        self.assertEqual(run(spec), "LL")

    def test_bounce_shorter_than_blank_ms_is_one_letter(self):
        spec = [("L", 10), ("", 4), ("L", 10)]          # 133ms gap
        self.assertEqual(run(spec), "L")

    def test_fingerspelling_commits_each_letter_in_order(self):
        self.assertEqual(run([("C", 8), ("A", 8), ("T", 8)]), "CAT")

    def test_letter_change_needs_no_blank_between(self):
        # Adjacent distinct letters with no gap at all still both commit.
        self.assertEqual(run([("H", 6), ("I", 6)]), "HI")

    def test_commit_records_frame_number(self):
        d = Debouncer()
        for i, (letter, t) in enumerate(frames([("A", 10)])):
            d.update(letter, frame_no=i, now_ms=t)
        self.assertEqual(len(d.commits), 1)
        self.assertEqual(d.commits[0][0], "A")
        self.assertGreaterEqual(d.commits[0][1], 3)


# ── frame-rate independence (the embedded-hardware property) ───────────────
class TestFrameRateIndependence(unittest.TestCase):

    def test_same_gesture_timing_gives_same_result_across_frame_rates(self):
        # Identical wall-clock scenario, sampled at laptop and embedded rates.
        # Every hold here is long enough to clear min_hold_frames even at 5fps;
        # below that floor the frame rate necessarily matters (see the test
        # after this one).
        scenarios = [
            [("A", 1.0), ("", 0.1), ("A", 1.0)],          # blip mid-hold -> "A"
            [("A", 1.0), ("", 0.5), ("A", 1.0)],          # real bounce   -> "AA"
            [("A", 1.0), ("", 0.2), ("A", 1.0)],          # borderline gap
            [("C", 1.0), ("A", 1.0), ("T", 1.0)],         # fingerspelled -> "CAT"
        ]
        for seconds_spec in scenarios:
            results = {}
            for fps in (30, 15, 10, 5):
                spec = [(letter, max(1, round(secs * fps)))
                        for letter, secs in seconds_spec]
                results[fps] = run(spec, fps=fps)
            with self.subTest(scenario=seconds_spec):
                self.assertEqual(len(set(results.values())), 1,
                                 f"frame-rate dependent: {results}")

    def test_legacy_was_frame_rate_dependent(self):
        # The same 0.1s blip: harmless at 5fps, splits the hold at 30fps.
        spec_fast = [("A", 30), ("", 3), ("A", 30)]
        spec_slow = [("A", 5), ("", 1), ("A", 5)]
        self.assertEqual(run_legacy(spec_fast, fps=30), "AA")
        self.assertEqual(run_legacy(spec_slow, fps=5), "AA")
        # ...whereas the hold requirement itself moves: 3 frames is 100ms at
        # 30fps but 600ms at 5fps, so a real 0.2s hold commits on one and not
        # the other.
        self.assertEqual(run_legacy([("A", 6)], fps=30), "A")
        self.assertEqual(run_legacy([("A", 1)], fps=5), "")

    def test_below_the_sample_floor_frame_rate_necessarily_matters(self):
        # A 0.5s hold is 15 samples at 30fps but only 2 at 5fps, which is under
        # min_hold_frames. This is the floor working as intended -- the same
        # tradeoff MotionDetector.min_samples makes -- and it is the real
        # constraint on how slow the embedded camera can be: min_hold_frames /
        # (shortest hold you expect to recognise) is the minimum usable fps.
        spec = [("A", 0.5)]
        self.assertEqual(run([(l, round(s * 30)) for l, s in spec], fps=30), "A")
        self.assertEqual(run([(l, round(s * 5)) for l, s in spec], fps=5), "")
        # 3 frames / 0.5s => ~6fps is the floor for recognising a 0.5s hold.
        self.assertEqual(run([("A", 3)], fps=6), "A")

    def test_low_frame_rate_floor_prevents_committing_on_two_samples(self):
        # At 5fps, 100ms is under one frame, so min_hold_frames is what stops
        # a two-sample blip from committing.
        self.assertEqual(run([("A", 2)], fps=5), "")
        self.assertEqual(run([("A", 3)], fps=5), "A")

    def test_one_dropped_frame_never_releases_even_at_low_fps(self):
        # At 3fps a single blank frame spans 333ms > blank_ms; min_blank_frames
        # is what keeps it from ending the hold.
        self.assertEqual(run([("A", 3), ("", 1), ("A", 3)], fps=3), "A")


# ── motion integration and reset ───────────────────────────────────────────
class TestMotionAndReset(unittest.TestCase):

    def _hold(self, d, letter, count, start_frame=0, fps=30):
        step = 1000.0 / fps
        for i in range(count):
            d.update(letter, frame_no=start_frame + i, now_ms=(start_frame + i) * step)
        return start_frame + count

    def test_commit_motion_drops_static_commits_inside_its_span(self):
        d = Debouncer()
        n = self._hold(d, "A", 10)                 # committed well before the gesture
        motion_start = n
        n = self._hold(d, "I", 10, start_frame=n)  # misread start of the J hook
        self.assertEqual(d.confirmed_string, "AI")
        d.commit_motion("J", motion_start)
        self.assertEqual(d.confirmed_string, "AJ")

    def test_commit_motion_clears_hold_state(self):
        d = Debouncer()
        n = self._hold(d, "I", 10)
        d.commit_motion("J", 0)
        self.assertIsNone(d.current_letter)
        self.assertFalse(d.committed)
        # The same letter can be held again straight afterwards.
        self._hold(d, "I", 10, start_frame=n)
        self.assertEqual(d.confirmed_string, "JI")

    def test_reset_clears_everything(self):
        d = Debouncer()
        self._hold(d, "A", 10)
        d.reset()
        self.assertEqual(d.confirmed_string, "")
        self.assertIsNone(d.current_letter)


if __name__ == "__main__":
    print("Recorded evaluate.py strings, replayed through both implementations")
    print("(steady hold, 2-frame blank every 12 frames, 30fps):\n")
    print(f"  {'expected':>8}  {'old':<14} {'new':<6}")
    for letter, repeats in [("D", 8), ("I", 10), ("L", 9), ("Q", 10),
                            ("P", 10), ("X", 9), ("G", 11), ("Y", 6)]:
        spec = [(letter, 12), ("", 2)] * repeats
        print(f"  {letter:>8}  {run_legacy(spec):<14} {run(spec):<6}")
    print()
    unittest.main(verbosity=2)
