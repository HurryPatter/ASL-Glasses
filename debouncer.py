"""Commit a letter once per continuous hold.

Timing here is wall-clock milliseconds, not frame counts, for the same reason
as in motion.py: a frame-count threshold means a different real-world duration
on every device, and the embedded target will not match the dev laptop's fps.
Frame counts survive only as *floors* (min_hold_frames / min_blank_frames), so
a low-fps device can't act on one or two noisy samples -- mirroring
MotionDetector.min_samples.

The central distinction this makes, which a plain "did the label change?"
debouncer cannot:

    ""      no reading this frame  -- absence of evidence
    "R"     a different handshape  -- evidence the hold actually ended

main.py produces "" for three different reasons (confidence below threshold, a
motion-gesture candidate suppressing static classification, and tracking
dropout), none of which mean the signer stopped holding the letter. Treating
those as end-of-hold is what produced committed strings like "QQQQQQQQQQ" from
a single steady hold: each blank frame reset the hold, and the letter was
re-committed a few frames later.

So a blank only releases a hold once it has persisted for blank_ms, a different
letter only takes over after switch_ms (much shorter -- it is real evidence,
and fingerspelling needs to stay responsive), and `committed` latches so one
hold can never commit twice regardless.

A genuine double letter (the "LL" in HELLO) still works the same way: the
signer bounces out of the shape and back in. That bounce now has to last longer
than blank_ms to read as two holds instead of one.
"""
import time


class Debouncer:
    def __init__(self, min_hold_ms=100, blank_ms=250, switch_ms=60,
                 min_hold_frames=3, min_blank_frames=2):
        self.min_hold_ms = min_hold_ms          # hold this long before committing
        self.blank_ms = blank_ms                # no reading this long to release a hold
        self.switch_ms = switch_ms              # a different letter this long to take over
        self.min_hold_frames = min_hold_frames  # floor: never commit on fewer samples
        self.min_blank_frames = min_blank_frames  # floor: one dropped frame never releases

        self.current_letter = None
        self.hold_start_ms = None
        self.last_reading_ms = None  # when ANY letter was last read (start of the current absence)
        self.hold_frames = 0
        self.committed = False       # latch: at most one commit per hold
        self._pending = None         # [kind, since_ms, frames]; kind is "" or a letter

        # A letter commits once per *run*, not merely once per hold. A brief
        # flicker to a confusable letter (G->Q, R->X: both are in the recorded
        # evaluation data) ends the hold without ending the run, and must not
        # let the original letter commit a second time.
        self.last_commit_letter = None
        self.released_since_commit = True

        self.frame_no = 0
        self.commits = []            # list of (letter, frame_no) in order

    @property
    def confirmed_string(self):
        return "".join(letter for letter, _ in self.commits)

    def update(self, letter, frame_no=None, now_ms=None):
        """Feed in the raw per-frame prediction ("" when there is no reading).

        Returns the confirmed string.
        """
        self.frame_no = self.frame_no + 1 if frame_no is None else frame_no
        now = time.monotonic() * 1000 if now_ms is None else now_ms

        if letter:
            self.last_reading_ms = now

        saw_current = False

        if letter and letter == self.current_letter:
            # Hold continues; any contradiction in flight was a blip.
            self._pending = None
            self.hold_frames += 1
            saw_current = True

        elif letter:
            if self.current_letter is None:
                self._start_hold(letter, now)
                saw_current = True
            else:
                _, since, frames = self._note_pending(letter, now)
                if now - since >= self.switch_ms:
                    # The new letter has already been visible for switch_ms, so
                    # count the hold from when it first appeared rather than now
                    # -- otherwise every letter change pays that latency twice.
                    self._start_hold(letter, since, frames=frames)
                    saw_current = True

        else:
            if self.current_letter is None:
                self._pending = None
            else:
                _, _, frames = self._note_pending("", now)
                # Measured from the last frame that produced ANY reading, for
                # two reasons. Measuring between blank samples instead would
                # undercount the absence by a whole frame interval, which is
                # 200ms at 5fps. And measuring from when the *held* letter was
                # last seen would count frames that read some other letter as
                # absence -- the hand is plainly visible in those, so a couple
                # of dropped frames after a flicker would wrongly release.
                if (self.last_reading_ms is not None
                        and now - self.last_reading_ms >= self.blank_ms
                        and frames >= self.min_blank_frames):
                    self._release()
            return self.confirmed_string

        if (saw_current and not self.committed
                and now - self.hold_start_ms >= self.min_hold_ms
                and self.hold_frames >= self.min_hold_frames):
            self.committed = True      # latch this hold either way
            repeat = (self.current_letter == self.last_commit_letter
                      and not self.released_since_commit)
            if not repeat:
                self.commits.append((self.current_letter, self.frame_no))
                self.last_commit_letter = self.current_letter
                self.released_since_commit = False

        return self.confirmed_string

    def commit_motion(self, letter, start_frame):
        """A motion letter (J/Z) was recognised over frames [start_frame, now].

        Any static letters committed during that span were the classifier
        misreading the start of the motion (e.g. an 'I' before the J hook),
        so drop them.
        """
        self.commits = [c for c in self.commits if c[1] < start_frame]
        self.commits.append((letter, self.frame_no))
        self.last_commit_letter = letter
        self._release()
        return self.confirmed_string

    def reset(self):
        self.commits = []
        self.last_commit_letter = None
        self._release()

    # ── internals ──────────────────────────────────────────────────────────
    def _note_pending(self, kind, now):
        """Track how long the current hold has been contradicted, and by what."""
        if self._pending is None or self._pending[0] != kind:
            self._pending = [kind, now, 1]
        else:
            self._pending[2] += 1
        return self._pending

    def _start_hold(self, letter, start_ms, frames=1):
        self.current_letter = letter
        self.hold_start_ms = start_ms
        self.hold_frames = frames
        self.committed = False
        self._pending = None

    def _release(self):
        # A genuine absence is what ends a letter run and re-arms the same
        # letter for another commit -- this is what makes "LL" possible.
        self.released_since_commit = True
        self.current_letter = None
        self.hold_start_ms = None
        self.hold_frames = 0
        self.committed = False
        self._pending = None
