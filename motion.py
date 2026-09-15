"""Trajectory-based detection of the two motion letters, J and Z.

The static classifier sees one frame at a time, so it can never see a path.
This module keeps a short history of MediaPipe landmarks and applies
hand-written rules to fingertip trajectories.

All distances are measured in "hand widths" (wrist -> middle knuckle), and
all timing is measured in wall-clock milliseconds rather than frame count --
that's the part that matters for portability. A frame-count window (e.g.
"last 20 frames") represents a different real-world duration on every
camera/device; a millisecond window represents the same gesture duration
everywhere, whether the camera runs at 30fps on a laptop or something
slower on embedded hardware.

Adding another motion sign
--------------------------
Each sign is a predicate over the buffer plus its own thresholds, checked in
update(). Three primitives exist to build on, and reusing them is what keeps
a new sign cheap:

  _shape_held(key)      the handshape persisted across the window
  _strokes(values)      a 1-D path split into direction runs, jitter ignored
  hand_travel(axis)     how far the whole hand actually went

hand_travel() is the one to reach for first. A sign that involves real
movement should require the hand to have moved; without that, any handshape
which happens to satisfy the shape test fires on landmark noise alone, which
is exactly how a held Q came to emit Z (see _is_z). Add per-sign thresholds
as constructor arguments -- named per sign, like z_travel -- rather than
tightening a shared one, since tightening a shared threshold to fix one sign
silently changes every other.

Two hands
---------
This class holds no module-level or class-level mutable state: the buffer and
every threshold are per-instance. So two-handed tracking means one
MotionDetector per tracked hand, fed that hand's landmarks, rather than any
change in here. What it does *not* give you is a two-handed sign -- a
predicate over the relationship between two hands needs a layer above this
that owns both detectors and can see both buffers. main.py would also need
num_hands=2 and a way to keep each hand's identity stable across frames,
which is the real work.

Cost
----
Every rule here is O(buffer) over a few dozen samples of plain Python floats,
with no allocation per frame beyond one dict, and no dependency outside the
standard library. That is deliberate: this file has to keep up on whatever
the glasses end up running.
"""
from collections import deque
import math

# MediaPipe hand landmark indices
WRIST = 0
MIDDLE_MCP = 9
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20


class MotionDetector:
    def __init__(self, window_ms=650, min_samples=8, shape_ratio=0.7,
                 j_drop=0.5, j_hook=0.25,
                 z_stroke=0.25, z_drop=0.3, z_travel=0.3, deadband=0.08):
        self.window_ms = window_ms       # how much real time a gesture attempt spans
        self.min_samples = min_samples   # floor so a low-fps device can't fire on 2-3 noisy points
        self.shape_ratio = shape_ratio   # fraction of samples that must hold the handshape
        self.j_drop = j_drop             # pinky tip must fall this far (hand widths)
        self.j_hook = j_hook             # ...then move sideways this far in the final third
        self.z_stroke = z_stroke         # each Z stroke must be at least this long
        self.z_drop = z_drop             # net downward drift across the Z
        self.z_travel = z_travel         # whole hand must cross this much (hand widths)
        self.deadband = deadband         # ignore horizontal jitter smaller than this
        self.buffer = deque()            # pruned by time, not maxlen

    # ── per-frame bookkeeping ───────────────────────────────────────────
    def update(self, landmarks, frame_no, timestamp_ms):
        """Feed one frame of landmarks + its wall-clock timestamp (ms).

        Returns (letter, start_frame) or ("", None).
        """
        wx, wy = landmarks[WRIST].x, landmarks[WRIST].y
        mx, my = landmarks[MIDDLE_MCP].x, landmarks[MIDDLE_MCP].y
        size = math.hypot(mx - wx, my - wy) or 1e-6

        rel = [((lm.x - wx) / size, (lm.y - wy) / size) for lm in landmarks]
        absx = [lm.x / size for lm in landmarks]
        absy = [lm.y / size for lm in landmarks]

        self.buffer.append({
            "t": timestamp_ms,
            "frame": frame_no,
            "i_shape": self._is_i_shape(rel),
            "point_shape": self._is_point_shape(rel),
            "pinky": (absx[PINKY_TIP], absy[PINKY_TIP]),
            "index": (absx[INDEX_TIP], absy[INDEX_TIP]),
            "centroid": (sum(absx) / len(absx), sum(absy) / len(absy)),
        })

        # Drop samples older than the window -- this is what makes the
        # window a fixed real-world duration instead of a fixed frame count.
        cutoff = timestamp_ms - self.window_ms
        while self.buffer and self.buffer[0]["t"] < cutoff:
            self.buffer.popleft()

        if len(self.buffer) < self.min_samples:
            return "", None

        if self._is_j():
            return self._fire("J")
        if self._is_z():
            return self._fire("Z")
        return "", None

    def clear(self):
        self.buffer.clear()

    def is_motion_candidate(self, lookback_ms=180):
        """True if the last lookback_ms of samples look like a J/Z attempt
        still unfolding: the right handshape held AND the fingertip is
        actively moving.

        Used to hold off on static commits (e.g. 'X') while a Z/J gesture is
        mid-flight. A genuinely held static letter isn't moving, so this
        won't suppress it -- only an in-progress gesture trips it.
        """
        if not self.buffer:
            return False
        cutoff = self.buffer[-1]["t"] - lookback_ms
        recent = [f for f in self.buffer if f["t"] >= cutoff]
        if len(recent) < 3:
            return False

        i_like = sum(f["i_shape"] for f in recent) >= self.shape_ratio * len(recent)
        point_like = sum(f["point_shape"] for f in recent) >= self.shape_ratio * len(recent)
        if not (i_like or point_like):
            return False

        key = "pinky" if i_like else "index"
        (x0, y0) = recent[0][key]
        (x1, y1) = recent[-1][key]
        moved = math.hypot(x1 - x0, y1 - y0)
        return moved > self.deadband

    # ── whole-hand motion primitive ─────────────────────────────────────
    def hand_travel(self, axis=0, lookback_ms=None):
        """How far the whole hand has moved across the window, in hand widths.

        Measured on the centroid of all 21 landmarks, which is the point of
        this: per-landmark jitter largely cancels in the average, while a real
        translation of the hand survives it intact. A single fingertip cannot
        tell those apart -- an occluded, wobbling fingertip traces the same
        oscillation as a small deliberate stroke.

        This is the primitive that separates "the hand went somewhere" from
        "the fingers wobbled in place", so any future motion sign should gate
        on it rather than re-deriving the idea. axis=0 is horizontal, axis=1
        vertical; pass lookback_ms to measure only the recent tail.
        """
        if not self.buffer:
            return 0.0
        frames = self.buffer
        if lookback_ms is not None:
            cutoff = self.buffer[-1]["t"] - lookback_ms
            frames = [f for f in self.buffer if f["t"] >= cutoff]
        vals = [f["centroid"][axis] for f in frames]
        return max(vals) - min(vals)

    # ── handshape rules (frame-independent — unchanged) ─────────────────
    @staticmethod
    def _extended(pts, tip, pip):
        return math.hypot(*pts[tip]) > math.hypot(*pts[pip])

    def _is_i_shape(self, pts):
        return (self._extended(pts, PINKY_TIP, PINKY_PIP)
                and not self._extended(pts, INDEX_TIP, INDEX_PIP)
                and not self._extended(pts, MIDDLE_TIP, MIDDLE_PIP)
                and not self._extended(pts, RING_TIP, RING_PIP))

    def _is_point_shape(self, pts):
        return (self._extended(pts, INDEX_TIP, INDEX_PIP)
                and not self._extended(pts, MIDDLE_TIP, MIDDLE_PIP)
                and not self._extended(pts, RING_TIP, RING_PIP)
                and not self._extended(pts, PINKY_TIP, PINKY_PIP))

    def _shape_held(self, key):
        held = sum(1 for f in self.buffer if f[key])
        return held >= self.shape_ratio * len(self.buffer)

    def _last_third(self):
        """Samples from the final third of the window BY TIME, not by index count."""
        t_start, t_end = self.buffer[0]["t"], self.buffer[-1]["t"]
        cutoff = t_end - (t_end - t_start) / 3
        return [f for f in self.buffer if f["t"] >= cutoff]

    # ── trajectory rules ────────────────────────────────────────────────
    def _is_j(self):
        """I handshape, pinky tip drops, then hooks sideways in the final third."""
        if not self._shape_held("i_shape"):
            return False
        ys = [f["pinky"][1] for f in self.buffer]
        dropped = (ys[-1] - ys[0]) > self.j_drop          # image y grows downward

        last_third = self._last_third()
        xs_third = [f["pinky"][0] for f in last_third]
        hooked = abs(xs_third[-1] - xs_third[0]) > self.j_hook if len(xs_third) >= 2 else False
        return dropped and hooked

    def _is_z(self):
        """Pointing handshape, index tip zigzags: three alternating strokes,
        and the whole hand actually crosses the frame while doing it.

        The travel gate is what stops a *stationary* pointing hand from firing.
        Q is a pointing handshape by this module's test (it passes
        _is_point_shape on 84% of the recorded Q samples), so holding a Q
        palm-toward-camera puts the Z rules under continuous evaluation. With
        the palm forward the curled fingers are self-occluded, MediaPipe's
        estimate of them wobbles, and a fingertip oscillating in place can
        satisfy "three alternating strokes" by chance -- observed live as a
        held Q emitting Z repeatedly, which also erased the correct Q because
        commit_motion() drops static commits inside the gesture span.

        A hand that has not gone anywhere has not signed a motion letter,
        whatever its fingertips did. That holds regardless of how noisy the
        landmarks are, which is why this is a gate rather than a threshold
        tweak -- retuning z_stroke would only move the noise floor.
        """
        if not self._shape_held("point_shape"):
            return False
        if self.hand_travel(axis=0) < self.z_travel:
            return False
        xs = [f["index"][0] for f in self.buffer]
        ys = [f["index"][1] for f in self.buffer]
        strokes = self._strokes(xs)[-3:]
        alternating = (len(strokes) == 3
                       and strokes[0] == strokes[2]
                       and strokes[0] != strokes[1])
        drifted_down = (ys[-1] - ys[0]) > self.z_drop
        return alternating and drifted_down

    def _strokes(self, xs):
        """Split a 1-D path into direction runs (+1 / -1), ignoring jitter."""
        strokes = []
        direction = 0
        start = extreme = xs[0]
        for x in xs[1:]:
            if direction == 0:
                if abs(x - start) > self.deadband:
                    direction = 1 if x > start else -1
                    extreme = x
            elif (x - extreme) * direction > 0:
                extreme = x                        # still moving the same way
            elif abs(x - extreme) > self.deadband:
                if abs(extreme - start) >= self.z_stroke:
                    strokes.append(direction)      # completed stroke
                start, extreme, direction = extreme, x, -direction
        if direction != 0 and abs(extreme - start) >= self.z_stroke:
            strokes.append(direction)
        return strokes

    def _fire(self, letter):
        start_frame = self.buffer[0]["frame"]
        self.buffer.clear()                        # never fire twice on one motion
        return letter, start_frame

    # ── diagnostics (does not affect firing) ────────────────────────────
    def debug_info(self):
        """Live values for tuning thresholds. Call every frame; returns a
        buffer-fill status until there's enough data to evaluate rules."""
        if len(self.buffer) < self.min_samples:
            return {"buffer": f"{len(self.buffer)}/{self.min_samples} samples"}

        ys = [f["pinky"][1] for f in self.buffer]
        j_dropped = ys[-1] - ys[0]
        last_third = self._last_third()
        xs_third = [f["pinky"][0] for f in last_third]
        j_hooked = abs(xs_third[-1] - xs_third[0]) if len(xs_third) >= 2 else 0.0

        xs_i = [f["index"][0] for f in self.buffer]
        ys_i = [f["index"][1] for f in self.buffer]
        strokes = self._strokes(xs_i)
        z_drift = ys_i[-1] - ys_i[0]

        span_ms = self.buffer[-1]["t"] - self.buffer[0]["t"]
        return {
            "window": f"{len(self.buffer)} samples / {span_ms:.0f}ms",
            "i_shape_held": f"{sum(f['i_shape'] for f in self.buffer)}/{len(self.buffer)}",
            "point_shape_held": f"{sum(f['point_shape'] for f in self.buffer)}/{len(self.buffer)}",
            "j_drop": round(j_dropped, 3),      # need > self.j_drop
            "j_hook": round(j_hooked, 3),       # need > self.j_hook
            "z_strokes": len(strokes),          # need 3, alternating
            "z_drift": round(z_drift, 3),       # need > self.z_drop
            "hand_travel": round(self.hand_travel(axis=0), 3),  # need > self.z_travel
        }