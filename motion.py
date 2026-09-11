"""Trajectory-based detection of the two motion letters, J and Z.

The static CNN sees one frame at a time, so it can never see a path.
This module keeps a short history of MediaPipe landmarks and applies
hand-written rules to fingertip trajectories.

All distances are measured in "hand widths" (wrist -> middle knuckle),
so the rules behave the same whether the hand is near or far from the camera.
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
    def __init__(self, window=20, shape_ratio=0.7,
                 j_drop=0.5, j_hook=0.25,
                 z_stroke=0.25, z_drop=0.3, deadband=0.08):
        self.window = window
        self.shape_ratio = shape_ratio   # fraction of frames that must hold the handshape
        self.j_drop = j_drop             # pinky tip must fall this far (hand widths)
        self.j_hook = j_hook             # ...then move sideways this far in the last third
        self.z_stroke = z_stroke         # each Z stroke must be at least this long
        self.z_drop = z_drop             # net downward drift across the Z
        self.deadband = deadband         # ignore horizontal jitter smaller than this
        self.buffer = deque(maxlen=window)

    # ── per-frame bookkeeping ───────────────────────────────────────────
    def update(self, landmarks, frame_no):
        """Feed one frame of landmarks. Returns (letter, start_frame) or ("", None)."""
        wx, wy = landmarks[WRIST].x, landmarks[WRIST].y
        mx, my = landmarks[MIDDLE_MCP].x, landmarks[MIDDLE_MCP].y
        size = math.hypot(mx - wx, my - wy) or 1e-6

        # wrist-relative points for handshape, absolute (scaled) points for paths
        rel = [((lm.x - wx) / size, (lm.y - wy) / size) for lm in landmarks]
        absx = [lm.x / size for lm in landmarks]
        absy = [lm.y / size for lm in landmarks]

        self.buffer.append({
            "frame": frame_no,
            "i_shape": self._is_i_shape(rel),
            "point_shape": self._is_point_shape(rel),
            "pinky": (absx[PINKY_TIP], absy[PINKY_TIP]),
            "index": (absx[INDEX_TIP], absy[INDEX_TIP]),
            "centroid": (sum(absx) / len(absx), sum(absy) / len(absy)),
        })

        if len(self.buffer) < self.window:
            return "", None

        if self._is_j():
            return self._fire("J")
        if self._is_z():
            return self._fire("Z")
        return "", None

    def clear(self):
        self.buffer.clear()

    def speed(self, n=5):
        """Centroid displacement over the last n frames, in hand widths."""
        if len(self.buffer) < n + 1:
            return 0.0
        (x0, y0) = self.buffer[-n - 1]["centroid"]
        (x1, y1) = self.buffer[-1]["centroid"]
        return math.hypot(x1 - x0, y1 - y0)

    # ── handshape rules ─────────────────────────────────────────────────
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

    # ── trajectory rules ────────────────────────────────────────────────
    def _is_j(self):
        """I handshape, pinky tip drops, then hooks sideways at the end."""
        if not self._shape_held("i_shape"):
            return False
        ys = [f["pinky"][1] for f in self.buffer]
        xs = [f["pinky"][0] for f in self.buffer]
        third = len(self.buffer) // 3
        dropped = (ys[-1] - ys[0]) > self.j_drop          # image y grows downward
        hooked = abs(xs[-1] - xs[-third]) > self.j_hook
        return dropped and hooked

    def _is_z(self):
        """Pointing handshape, index tip zigzags: three alternating strokes."""
        if not self._shape_held("point_shape"):
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
