"""Continuous signing: where does one sign end and the next begin —
Project Veronica stage 6.

Everything up to here assumes somebody presses a key to mark a sign's
boundaries. `collect_signs.py` does exactly that. Real conversation has no key.

Why this is not boundary detection
----------------------------------
The obvious approach is to find the boundaries first and classify between them:
look for a pause, or a minimum in hand velocity, and cut there. It does not
work, for a reason with a name — **movement epenthesis**. The transition
between two signs is itself movement, and it looks like a sign to anything
watching for movement. Signers also do not pause between signs the way speakers
pause between words, so the pauses a cutter would look for are frequently not
there at all.

So this does not cut first. It **classifies a trailing window continuously and
commits when the answer is stable**, which is the same shape as the problem
`debouncer.py` already solves for letters, one level up. That is not a
coincidence worth glossing over: its central distinction generalises exactly.

    ""       no confident reading   -- absence of evidence
    "EAT"    a different sign       -- evidence the last one ended

Treating the first as evidence of a boundary is what produced committed strings
like `QQQQQQQQQQ` from one steady hold. Here the same mistake would fire a sign
on every frame of an unrecognised transition.

So `Debouncer` is reused rather than reimplemented. It is the most thoroughly
tested component in the repository (24 tests, replaying real recorded failure
strings), and every property it has is one this needs: commit once per run, a
blank has to persist before it releases, a different label takes over faster
than a blank does because it is real evidence, and a genuine release re-arms
the same label so a repeated sign can commit twice.

What defends against epenthesis
-------------------------------
Three things, in order of how much work they do:

1. **The `_REST` class.** A transition classified as `_REST` produces no
   reading at all. This is the main defence, and it is why stage 4 collects
   `_REST` even though nobody signs it — without it a transition has to be
   classified as *some* sign, and the best the confidence floor can do is
   suppress it sometimes.
2. **The confidence floor.** A window spanning two signs is not a clean example
   of either, so the classifier should be unsure. Anything below the floor
   reads as "".
3. **The debouncer's hold requirement.** A spurious label has to survive
   several consecutive evaluations to commit, and a transition is brief.

None of these is exact, and the honest position is that continuous segmentation
is the least finished part of this pipeline. The thresholds here are structural
guesses that need real continuous signing to tune — which is why
`debug_info()` reports every one of them live.

Cost
----
Classifying every frame would mean a full `sign_vector()` build plus an MLP
forward pass at camera rate. Evaluation is **strided** instead: the buffer
takes every frame, but classification runs every `stride_ms`. At 30fps and a
100ms stride that is one evaluation in three frames, and the window it sees is
unchanged — the buffer already holds every frame either way.

Standard library only.
"""
from debouncer import Debouncer
from sequence import SIGN_WINDOW_MS, SignBuffer
from signset import REST_LABEL


class ContinuousSegmenter:
    """Sign out of a continuous stream, with nobody pressing anything.

    `classify` takes a sign vector and returns `(label, probability)`. It is
    injected rather than constructed here so this module stays free of
    scikit-learn and testable against a scripted classifier -- which is the
    only way to test the *segmentation* separately from the model's accuracy.
    """

    def __init__(self, classify, window_ms=SIGN_WINDOW_MS, stride_ms=100,
                 min_confidence=0.70, rest_label=REST_LABEL,
                 missing_tolerance_ms=300,
                 min_hold_ms=150, blank_ms=300, switch_ms=100,
                 min_hold_evals=2, min_blank_evals=2):
        self.classify = classify
        self.stride_ms = stride_ms          # how often to actually classify
        self.min_confidence = min_confidence
        self.rest_label = rest_label
        self.missing_tolerance_ms = missing_tolerance_ms

        # window_ms is roughly one sign's duration. Longer spans two signs and
        # is a clean example of neither; shorter truncates the ones that carry
        # repetition. It is the threshold most in need of real signing to tune.
        self.buffer = SignBuffer(window_ms=window_ms)

        # Frame counts inside Debouncer become *evaluation* counts here, since
        # that is the rate decisions arrive at. Two evaluations at a 100ms
        # stride is 200ms of agreement before anything commits.
        self.debouncer = Debouncer(
            min_hold_ms=min_hold_ms, blank_ms=blank_ms, switch_ms=switch_ms,
            min_hold_frames=min_hold_evals, min_blank_frames=min_blank_evals)

        self.evaluations = 0
        self.frames = 0
        self._last_eval_ms = None
        self._last_seen_ms = None
        self._committed = 0
        self._last_reading = ("", 0.0)

    # ── per-frame ───────────────────────────────────────────────────────
    def update(self, timestamp_ms, features=None, dom_anchor=None,
               non_anchor=None):
        """Feed one frame. Returns the labels committed on this frame (usually
        none), so a caller can speak or display them as they land."""
        self.frames += 1

        if features is None:
            return self._no_hands(timestamp_ms)

        self._last_seen_ms = timestamp_ms
        self.buffer.add(timestamp_ms, features, dom_anchor, non_anchor)

        if not self._due(timestamp_ms):
            return []
        return self._evaluate(timestamp_ms)

    def _no_hands(self, timestamp_ms):
        """Tracking lost. Tolerate a short dropout before disturbing anything.

        Same reasoning as main.py's MISSED_FRAME_TOLERANCE: a fast sign and a
        brief occlusion are the likeliest causes of a dropout, and wiping state
        on the first frame of one throws away exactly the sign being made. The
        tolerance is in milliseconds rather than frames so it means the same
        thing on slower hardware.
        """
        if (self._last_seen_ms is not None
                and timestamp_ms - self._last_seen_ms < self.missing_tolerance_ms):
            return []

        before = len(self.debouncer.commits)
        self.debouncer.update("", self.evaluations, timestamp_ms)
        self.buffer.clear()
        self._last_reading = ("", 0.0)
        return self._newly_committed(before)

    def _due(self, timestamp_ms):
        if self._last_eval_ms is None:
            self._last_eval_ms = timestamp_ms
            return False
        return timestamp_ms - self._last_eval_ms >= self.stride_ms

    def _evaluate(self, timestamp_ms):
        self._last_eval_ms = timestamp_ms
        vector = self.buffer.vector()
        if vector is None:                    # not enough evidence yet
            return []

        self.evaluations += 1
        label, probability = self.classify(vector)
        self._last_reading = (label, probability)

        # Absence of evidence, not evidence of a boundary -- the distinction
        # debouncer.py exists to make. An unsure classifier and a transition
        # classified as _REST both land here.
        reading = ""
        if probability >= self.min_confidence and label != self.rest_label:
            reading = label

        before = len(self.debouncer.commits)
        self.debouncer.update(reading, self.evaluations, timestamp_ms)
        committed = self._newly_committed(before)

        if committed:
            # Drop the frames that produced this sign, so its own evidence
            # cannot commit a second time or blend into the next window. Same
            # move as MotionDetector._fire(), and it doubles as a refractory
            # period while the buffer refills.
            self.buffer.clear()
        return committed

    def _newly_committed(self, before):
        new = [label for label, _ in self.debouncer.commits[before:]]
        self._committed = len(self.debouncer.commits)
        return new

    # ── output ──────────────────────────────────────────────────────────
    @property
    def glosses(self):
        return [label for label, _ in self.debouncer.commits]

    def reset(self):
        self.buffer.clear()
        self.debouncer.reset()
        self._last_eval_ms = None
        self._last_seen_ms = None
        self._last_reading = ("", 0.0)
        self._committed = 0

    def debug_info(self):
        """Live values for tuning, mirroring MotionDetector.debug_info().

        Every threshold here is a structural guess until somebody signs
        continuously in front of a camera, so all of them are on screen. The
        one to watch first is `reading`: if transitions between signs show a
        confident label rather than a blank, the _REST class is not covering
        them and more _REST clips will do more than any threshold change.
        """
        label, probability = self._last_reading
        info = dict(self.buffer.debug_info())
        info.update({
            "reading": f"{label or '-'} {probability:.2f}"
                       f"{'' if probability >= self.min_confidence else '  (below floor)'}",
            "holding": self.debouncer.current_letter or "-",
            "committed": len(self.debouncer.commits),
            "evaluations": f"{self.evaluations} of {self.frames} frames",
            "thresholds": f"conf>{self.min_confidence} stride={self.stride_ms}ms "
                          f"window={self.buffer.window_ms}ms",
        })
        return info
