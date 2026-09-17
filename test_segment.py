"""Offline tests for continuous segmentation -- no camera, no model.

Standard library only, so it runs in CI.

The classifier is scripted, which is the whole point: these test whether the
*segmentation* is right, separately from whether any model is accurate. A real
model would confound the two, and the thresholds here are structural guesses
that need real continuous signing to tune -- so what can be pinned down now is
the logic around them.

Run:  python -m unittest test_segment -v
"""
import unittest

import hands
import segment
import sequence
from signset import REST_LABEL

FPS = 30.0
STEP_MS = 1000.0 / FPS


class ScriptedClassifier:
    """Returns whatever the script says for the current time.

    `script` is [(until_ms, label, probability)], read in order.
    """

    def __init__(self, script):
        self.script = script
        self.now = 0
        self.calls = 0

    def __call__(self, vector):
        self.calls += 1
        for until, label, probability in self.script:
            if self.now < until:
                return label, probability
        return "", 0.0


def run(script, duration_ms=4000, fps=FPS, hands_present=lambda t: True, **kw):
    """Drive a segmenter through `duration_ms` of scripted classification."""
    classifier = ScriptedClassifier(script)
    seg = segment.ContinuousSegmenter(classifier, **kw)
    features = hands.feature_vector(None, None, None)
    anchor = ((320.0, 240.0), 100.0)

    step = 1000.0 / fps
    committed = []
    t = 0.0
    while t <= duration_ms:
        classifier.now = t
        if hands_present(t):
            committed += seg.update(int(t), features, anchor, None)
        else:
            committed += seg.update(int(t))
        t += step
    return seg, committed, classifier


class TestSingleSign(unittest.TestCase):

    def test_a_steady_sign_commits_once(self):
        _, committed, _ = run([(1500, "HELLO", 0.95)], duration_ms=1500)
        self.assertEqual(committed, ["HELLO"])

    def test_it_does_not_re_fire_while_still_visible(self):
        # A held sign stays in the trailing window for as long as the signer
        # holds it. Committing once per run is what stops that becoming
        # HELLO HELLO HELLO -- the letter pipeline's QQQQQQQQQQ, one level up.
        _, committed, _ = run([(4000, "HELLO", 0.95)], duration_ms=4000)
        self.assertEqual(committed, ["HELLO"])

    def test_nothing_commits_without_a_confident_reading(self):
        _, committed, _ = run([(4000, "HELLO", 0.30)])
        self.assertEqual(committed, [])

    def test_rest_never_commits(self):
        _, committed, _ = run([(4000, REST_LABEL, 0.99)])
        self.assertEqual(committed, [])


class TestSequences(unittest.TestCase):

    def test_two_different_signs_commit_in_order(self):
        _, committed, _ = run([
            (1200, "ME", 0.95),
            (2600, "WANT", 0.95),
            (4000, "WATER", 0.95),
        ])
        self.assertEqual(committed, ["ME", "WANT", "WATER"])

    def test_a_repeated_sign_commits_twice_when_released_between(self):
        # AGAIN AGAIN is two signs, not one. A genuine release re-arms the
        # same label -- the same property that makes the "LL" in HELLO work
        # in the letter pipeline.
        _, committed, _ = run([
            (1200, "AGAIN", 0.95),
            (2000, "", 0.0),         # hands drop between repetitions
            (3400, "AGAIN", 0.95),
        ])
        self.assertEqual(committed, ["AGAIN", "AGAIN"])

    def test_a_repeated_sign_with_no_gap_commits_once(self):
        # The counterpart: with no release there is one continuous run, and
        # committing twice would be inventing a sign the signer did not make.
        _, committed, _ = run([(4000, "AGAIN", 0.95)])
        self.assertEqual(committed, ["AGAIN"])

    def test_glosses_accumulate_in_order(self):
        seg, _, _ = run([
            (1200, "ME", 0.95), (2600, "TIRED", 0.95)])
        self.assertEqual(seg.glosses, ["ME", "TIRED"])


class TestMovementEpenthesis(unittest.TestCase):
    """The transition between two signs is itself movement, and looks like a
    sign to anything watching for movement."""

    def test_a_transition_classified_as_rest_produces_nothing(self):
        # The main defence, and why stage 4 collects _REST at all.
        _, committed, _ = run([
            (1200, "ME", 0.95),
            (1700, REST_LABEL, 0.90),     # the transition
            (3000, "TIRED", 0.95),
        ])
        self.assertEqual(committed, ["ME", "TIRED"])

    def test_an_unsure_transition_produces_nothing(self):
        # Second defence: a window spanning two signs is a clean example of
        # neither, so the classifier should be unsure.
        _, committed, _ = run([
            (1200, "ME", 0.95),
            (1700, "EAT", 0.45),          # garbage from a straddling window
            (3000, "TIRED", 0.95),
        ])
        self.assertEqual(committed, ["ME", "TIRED"])

    def test_a_brief_confident_spurious_label_does_not_commit(self):
        # Third defence: a label has to survive several consecutive
        # evaluations, and a transition is brief.
        _, committed, _ = run([
            (1200, "ME", 0.95),
            (1290, "EAT", 0.95),          # one evaluation's worth
            (2600, "TIRED", 0.95),
        ])
        self.assertEqual(committed, ["ME", "TIRED"])

    def test_a_sustained_confident_wrong_label_does_commit(self):
        # The honest limit. If the classifier is confidently wrong for long
        # enough, nothing here can tell -- that is a model problem, and no
        # threshold in this module fixes it.
        _, committed, _ = run([
            (1200, "ME", 0.95),
            (2600, "EAT", 0.95),
            (4000, "TIRED", 0.95),
        ])
        self.assertEqual(committed, ["ME", "EAT", "TIRED"])


class TestTrackingDropout(unittest.TestCase):

    def test_a_brief_dropout_does_not_break_a_sign(self):
        # A fast sign and a brief occlusion are the likeliest causes of a
        # dropout, so wiping state on the first missing frame throws away
        # exactly the sign being made.
        gone = lambda t: not (700 <= t < 780)
        _, committed, _ = run([(2000, "HELLO", 0.95)], duration_ms=2000,
                              hands_present=gone)
        self.assertEqual(committed, ["HELLO"])

    def test_a_long_dropout_releases_the_hold(self):
        gone = lambda t: not (1200 <= t < 2200)
        _, committed, _ = run([
            (1200, "HELLO", 0.95),
            (4000, "HELLO", 0.95),
        ], hands_present=gone)
        self.assertEqual(committed, ["HELLO", "HELLO"])

    def test_no_hands_at_all_commits_nothing(self):
        _, committed, _ = run([(4000, "HELLO", 0.95)],
                              hands_present=lambda t: False)
        self.assertEqual(committed, [])


class TestFrameRateIndependence(unittest.TestCase):
    """Timing is wall-clock, so the same signing commits the same signs
    whatever the camera manages -- the constraint motion.py and debouncer.py
    both already hold to."""

    def test_same_commits_at_30fps_and_15fps(self):
        script = [(1200, "ME", 0.95), (2600, "WANT", 0.95), (4000, "WATER", 0.95)]
        fast = run(script, fps=30.0)[1]
        slow = run(script, fps=15.0)[1]
        self.assertEqual(fast, slow)

    def test_still_works_at_the_frame_rate_floor(self):
        # ~11fps is the floor NOTES.md derives for motion signs and
        # sequence.py reaches independently. At 12fps this must still work.
        _, committed, _ = run([(1500, "HELLO", 0.95)], duration_ms=1500, fps=12.0)
        self.assertEqual(committed, ["HELLO"])

    def test_below_the_floor_it_stops_rather_than_guessing(self):
        # A sign cannot be assembled from three samples, and firing on them
        # anyway is worse than not firing: SignBuffer refuses.
        _, committed, _ = run([(1500, "HELLO", 0.95)], duration_ms=1500, fps=4.0)
        self.assertEqual(committed, [])


class TestStridedEvaluation(unittest.TestCase):
    """Classifying every frame means a full sign_vector() build plus a forward
    pass at camera rate, on hardware that has to sustain ~15fps of MediaPipe
    as well."""

    def test_evaluation_is_far_less_frequent_than_frames(self):
        seg, _, classifier = run([(4000, "HELLO", 0.95)], fps=30.0,
                                 stride_ms=100)
        self.assertLess(classifier.calls, seg.frames / 2)

    def test_the_buffer_still_sees_every_frame(self):
        # Striding must not thin the window: the vector a decision is made on
        # should be identical either way.
        seg, _, _ = run([(1000, "", 0.0)], duration_ms=1000, fps=30.0,
                        stride_ms=200)
        self.assertGreater(len(seg.buffer), 10)

    def test_a_longer_stride_means_fewer_calls(self):
        _, _, quick = run([(4000, "", 0.0)], stride_ms=100)
        _, _, lazy = run([(4000, "", 0.0)], stride_ms=400)
        self.assertLess(lazy.calls, quick.calls)


class TestResetAndDiagnostics(unittest.TestCase):

    def test_reset_clears_everything(self):
        seg, _, _ = run([(2000, "HELLO", 0.95)], duration_ms=2000)
        self.assertTrue(seg.glosses)
        seg.reset()
        self.assertEqual(seg.glosses, [])
        self.assertEqual(len(seg.buffer), 0)

    def test_debug_info_reports_the_reading_and_the_thresholds(self):
        seg, _, _ = run([(2000, "HELLO", 0.95)], duration_ms=2000)
        info = seg.debug_info()
        self.assertIn("reading", info)
        self.assertIn("thresholds", info)
        self.assertEqual(info["committed"], 1)

    def test_debug_info_marks_a_reading_below_the_floor(self):
        seg, _, _ = run([(2000, "HELLO", 0.40)], duration_ms=2000)
        self.assertIn("below floor", seg.debug_info()["reading"])

    def test_debug_info_survives_an_empty_segmenter(self):
        seg = segment.ContinuousSegmenter(lambda v: ("", 0.0))
        self.assertIn("clip", seg.debug_info())


class TestGlossIntegration(unittest.TestCase):
    """Stage 6 feeding stage 7: the committed glosses are what gets rendered."""

    def test_a_signed_sentence_comes_out_as_english(self):
        import gloss
        seg, _, _ = run([
            (1200, "YESTERDAY", 0.95),
            (2500, "ME", 0.95),
            (3800, "GO", 0.95),
            (5000, "SCHOOL", 0.95),
        ], duration_ms=5000)
        self.assertEqual(seg.glosses, ["YESTERDAY", "ME", "GO", "SCHOOL"])
        self.assertEqual(gloss.render(seg.glosses),
                         "Yesterday, I went to school.")


if __name__ == "__main__":
    unittest.main()
