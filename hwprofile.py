"""Emulating slower hardware on the development laptop.

The embedded target is undecided, and the decision turns on two numbers that
can be measured here for free, before anything is bought:

  1. Does accuracy survive a lower capture resolution? The pipeline consumes
     landmarks, not pixels, so resolution beyond what the detector needs is
     wasted work -- and reported figures put MediaPipe at 25+fps at 320x240
     against 8-15 at default, a larger effect than the gap between candidate
     boards.

  2. How slow can the frame rate get before recognition degrades? There is a
     hard floor already: motion.py needs min_samples (8) inside window_ms
     (650), i.e. (8-1)*1000/650 ~= 10.8fps, below which J and Z cannot fire at
     all. What is not known is where *static* letters start to suffer.

Throttling is done by dropping frames rather than by sleeping: the camera is
read at its own rate and frames arriving sooner than the target interval are
discarded unprocessed. That emulates a device that cannot keep up, which is
the thing being measured. Timestamps stay real wall-clock throughout, so the
time-based logic in debouncer.py and motion.py behaves exactly as it would on
slow hardware -- which is precisely why both were written against the clock
rather than against frame counts.

Standard library only: the tests for this run in CI.
"""


class FrameLimiter:
    """Pass at most `fps` frames per second through, dropping the rest.

    fps of None means no limit -- every frame is processed, which is the
    normal behaviour of the pipeline.
    """

    def __init__(self, fps=None):
        self.fps = fps
        self.interval_ms = None if not fps else 1000.0 / fps
        self.due_ms = None
        self.seen = 0
        self.processed = 0
        self.first_ms = None
        self.latest_ms = None

    def should_process(self, now_ms):
        """True if this frame should be processed, False to drop it.

        The next slot is scheduled by advancing a running deadline rather than
        by measuring from the frame that was last kept. That distinction
        matters: frames arrive only on the camera's own grid, so "keep one once
        a full interval has elapsed" quantises the result downward -- a 12fps
        cap on a 30fps camera would keep every third frame and deliver 10fps,
        silently testing a different rate than the one asked for. Accumulating
        the deadline instead alternates 2- and 3-frame gaps and averages out at
        the requested rate.
        """
        self.seen += 1
        if self.first_ms is None:
            self.first_ms = now_ms
        self.latest_ms = now_ms

        if self.interval_ms is None:
            self.processed += 1
            return True

        if self.due_ms is None:
            self.due_ms = now_ms + self.interval_ms
            self.processed += 1
            return True

        if now_ms < self.due_ms:
            return False

        # If the camera stalled and the deadline is far in the past, resync
        # rather than processing a catch-up burst that never really happened.
        if now_ms - self.due_ms > self.interval_ms:
            self.due_ms = now_ms + self.interval_ms
        else:
            self.due_ms += self.interval_ms
        self.processed += 1
        return True

    def achieved_fps(self):
        """Frames per second actually processed, over the whole run.

        This is the number to record: the cap is what was asked for, this is
        what the machine and the camera between them delivered.
        """
        if self.first_ms is None or self.latest_ms is None:
            return 0.0
        span_s = (self.latest_ms - self.first_ms) / 1000.0
        if span_s <= 0:
            return 0.0
        return self.processed / span_s

    def capture_fps(self):
        """Frames per second the camera delivered, before throttling.

        Worth recording separately: if this is already below target, the
        camera is the bottleneck and no faster board will help.
        """
        if self.first_ms is None or self.latest_ms is None:
            return 0.0
        span_s = (self.latest_ms - self.first_ms) / 1000.0
        if span_s <= 0:
            return 0.0
        return self.seen / span_s


def describe(width, height, fps):
    """Short label for a hardware profile, for logging and reporting."""
    res = f"{width}x{height}" if width and height else "native"
    rate = f"{fps}fps" if fps else "uncapped"
    return f"{res}@{rate}"


# motion.py cannot fire J or Z below this, whatever the gesture: min_samples
# has to be met inside window_ms.
def motion_floor_fps(min_samples=8, window_ms=650):
    return (min_samples - 1) * 1000.0 / window_ms
