"""Rule-based ASL fingerspelling from MediaPipe hand landmarks. No trained letter model.

How a person describes a letter is how this file describes it: which fingers are
straight, which are curled, where the thumb sits, which way the palm points.
Every letter is a list of (weight, condition) tests; the letter whose tests pass
best wins, if it passes well enough.

Geometry is done in a "hand frame":
  * origin at the wrist
  * y axis along wrist -> middle-finger knuckle ("up the palm")
  * x axis across the palm, flipped so the thumb is always on +x
  * unit length = wrist -> middle-knuckle distance
so the rules do not care about left/right hand, distance from the camera, or
how the arm is rotated. Palm orientation in the image is kept separately for
the letters that depend on it (G, H, P, Q point sideways or down).
"""
import math
import numpy as np

WRIST = 0
THUMB = [1, 2, 3, 4]
FINGERS = {"index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
           "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _angle(a, b, c):
    """Angle at b (degrees) between segments b->a and b->c."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _dir_angle(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


class HandFeatures:
    def __init__(self, landmarks, frame_w, frame_h):
        pts = np.array([[getattr(lm, "x", None) if hasattr(lm, "x") else lm[0],
                         getattr(lm, "y", None) if hasattr(lm, "y") else lm[1]]
                        for lm in landmarks], dtype=float)
        pts[:, 0] *= frame_w
        pts[:, 1] *= frame_h
        self.img = pts

        # palm direction in the image: 90 = up, 0 = right, -90 = down (image y grows downward)
        v = pts[9] - pts[0]
        self.palm_angle = math.degrees(math.atan2(-v[1], v[0]))
        size = np.linalg.norm(v) or 1e-6

        y_hat = v / size
        x_hat = np.array([y_hat[1], -y_hat[0]])
        local = ((pts - pts[0]) @ np.stack([x_hat, y_hat], axis=1)) / size
        if local[2][0] < 0:             # thumb knuckle on -x: mirror so thumb is on +x
            local[:, 0] *= -1
        self.p = local

        self.curl = {name: self._curl(idx) for name, idx in FINGERS.items()}
        self.curl["thumb"] = self._thumb_curl()

    # ── finger state ─────────────────────────────────────────────────────
    def _curl(self, idx):
        """0 = straight and pointing away from the palm, 1 = folded into the palm."""
        mcp, pip, dip, tip = (self.p[i] for i in idx)
        bend = 180 - _angle(mcp, pip, tip)              # 0 straight, ~120 fully folded
        c = min(1.0, max(0.0, (bend - 20) / 90))
        if tip[1] < pip[1]:                              # tip lower than middle joint: folded
            c = max(c, 0.8)
        return c

    def _thumb_curl(self):
        bend = 180 - _angle(self.p[1], self.p[2], self.p[4])
        return min(1.0, max(0.0, (bend - 20) / 70))

    def d(self, a, b):
        return float(np.linalg.norm(self.p[a] - self.p[b]))

    def spread(self, f1, f2):
        """Angle between two fingers' knuckle->tip directions, degrees."""
        a, b = FINGERS[f1], FINGERS[f2]
        return _dir_angle(self.p[a[3]] - self.p[a[0]], self.p[b[3]] - self.p[b[0]])

    def ext(self, name):
        return self.curl[name] < 0.35

    def cur(self, name):
        return self.curl[name] > 0.65

    def half(self, name):
        return 0.25 < self.curl[name] < 0.8

    # ── orientation ──────────────────────────────────────────────────────
    def up(self):
        return 45 < self.palm_angle < 135

    def down(self):
        return -135 < self.palm_angle < -45

    def sideways(self):
        return not self.up() and not self.down()

    # ── thumb placement (hand frame, thumb on +x) ────────────────────────
    @property
    def thumb(self):
        return self.p[4]

    def thumb_out(self):
        """Thumb sticking out to the side, away from the index knuckle (L, Y)."""
        return self.thumb[0] > self.p[5][0] + 0.55 and self.curl["thumb"] < 0.5

    def thumb_beside_index(self):
        """Thumb straight up alongside the index finger (A)."""
        return (self.p[5][0] + 0.1 < self.thumb[0] < self.p[5][0] + 0.7
                and self.thumb[1] > self.p[5][1] - 0.1)

    def thumb_across(self):
        """Thumb crossing over the palm toward the little finger (B, S, U, ...)."""
        return self.thumb[0] < self.p[5][0] - 0.05

    def readout(self):
        """Human-readable state, for the debug window."""
        parts = [f"{n[:3]}:{'ext' if self.ext(n) else 'cur' if self.cur(n) else 'half'}"
                 for n in FINGERS]
        parts.append(f"thumb:{'out' if self.thumb_out() else 'across' if self.thumb_across() else 'side'}")
        orient = "up" if self.up() else "down" if self.down() else "side"
        parts.append(f"palm:{orient}")
        return "  ".join(parts)


# ── letter definitions ───────────────────────────────────────────────────
def letter_tests(f):
    """Return {letter: [(weight, passed), ...]}."""
    ei, em, er, ep = f.ext("index"), f.ext("middle"), f.ext("ring"), f.ext("pinky")
    ci, cm, cr, cp = f.cur("index"), f.cur("middle"), f.cur("ring"), f.cur("pinky")
    fist = ci and cm and cr and cp
    tips_y = [f.p[i][1] for i in (8, 12, 16, 20)]
    im_spread = f.spread("index", "middle")
    mr_spread = f.spread("middle", "ring")
    thumb_index_angle = _dir_angle(f.p[4] - f.p[2], f.p[8] - f.p[5])
    crossed = f.p[8][0] < f.p[12][0]                    # index tip on the far side of middle tip

    T = {}
    T["A"] = [(3, fist), (2, f.thumb_beside_index()), (1, f.up())]
    # thumb may be folded across the palm or resting beside the index; only "out" rules B out
    T["B"] = [(3, ei and em and er and ep), (2, im_spread < 14 and mr_spread < 14),
              (2, not f.thumb_out()), (1, f.up())]
    T["C"] = [(3, all(f.half(n) for n in FINGERS)), (2, 0.45 < f.d(4, 8) < 1.3),
              (1, f.curl["thumb"] < 0.7), (1, f.up() or f.sideways())]
    T["D"] = [(3, ei and cm and cr and cp), (2, min(f.d(4, 12), f.d(4, 11)) < 0.5), (1, f.up())]
    T["E"] = [(3, fist), (2, f.thumb[1] < min(tips_y) + 0.15), (1, f.thumb_across() or abs(f.thumb[0] - f.p[5][0]) < 0.3),
              (1, f.up())]
    T["F"] = [(3, f.d(4, 8) < 0.42), (3, em and er and ep), (1, f.up())]
    T["G"] = [(3, ei and cm and cr and cp), (2, f.curl["thumb"] < 0.4 and thumb_index_angle < 40),
              (2, f.sideways())]
    T["H"] = [(3, ei and em and cr and cp), (2, im_spread < 14), (2, f.sideways())]
    T["I"] = [(3, ep and ci and cm and cr), (1, not f.thumb_out()), (1, f.up())]
    T["K"] = [(3, ei and em and cr and cp), (2, im_spread > 16), (2, f.d(4, 10) < 0.55 or f.d(4, 9) < 0.55),
              (1, f.up())]
    T["L"] = [(3, ei and cm and cr and cp), (3, f.thumb_out()), (1, 55 < thumb_index_angle < 120), (1, f.up())]
    T["M"] = [(3, fist), (2, f.thumb[0] < f.p[13][0] + 0.1), (1, f.thumb[1] < f.p[14][1]), (1, f.up())]
    T["N"] = [(3, fist), (2, f.p[13][0] - 0.05 < f.thumb[0] < f.p[9][0] + 0.15), (1, f.thumb[1] < f.p[10][1]),
              (1, f.up())]
    T["O"] = [(3, all(f.curl[n] > 0.35 for n in FINGERS)), (3, f.d(4, 8) < 0.42), (1, f.d(4, 12) < 0.7),
              (1, f.up() or f.sideways())]
    T["P"] = [(3, ei and em and cr and cp), (2, im_spread > 12), (1, f.d(4, 10) < 0.6 or f.d(4, 9) < 0.6),
              (3, f.down())]
    T["Q"] = [(3, ei and cm and cr and cp), (2, f.curl["thumb"] < 0.4 and thumb_index_angle < 40), (3, f.down())]
    T["R"] = [(3, ei and em and cr and cp), (3, crossed), (1, f.up())]
    T["S"] = [(3, fist), (2, f.thumb_across() and f.thumb[1] > f.p[6][1] - 0.6), (1, f.thumb[1] > min(tips_y)),
              (1, f.up())]
    T["T"] = [(3, fist), (2, f.d(4, 6) < 0.5 and f.p[9][0] - 0.1 < f.thumb[0] < f.p[5][0] + 0.2),
              (1, f.thumb[1] > f.p[5][1]), (1, f.up())]
    T["U"] = [(3, ei and em and cr and cp), (2, im_spread < 14), (2, f.up()), (1, f.thumb_across() or not f.thumb_out())]
    T["V"] = [(3, ei and em and cr and cp), (2, im_spread > 16), (2, f.d(4, 10) > 0.55 and f.d(4, 9) > 0.55),
              (1, f.up())]
    T["W"] = [(3, ei and em and er and cp), (2, im_spread > 10 and mr_spread > 10), (1, f.up())]
    T["X"] = [(3, 0.3 < f.curl["index"] < 0.85 and cm and cr and cp), (2, f.p[8][1] < f.p[7][1] + 0.1),
              (1, f.up())]
    T["Y"] = [(3, ep and ci and cm and cr), (3, f.thumb_out()), (1, f.up())]
    return T


def classify(landmarks, frame_w, frame_h, min_score=0.85, min_margin=0.05):
    """Returns (letter or "", score, features, ranked list of (letter, score))."""
    f = HandFeatures(landmarks, frame_w, frame_h)
    scores = {}
    for letter, tests in letter_tests(f).items():
        total = sum(w for w, _ in tests)
        scores[letter] = sum(w for w, ok in tests if ok) / total
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, s1 = ranked[0]
    s2 = ranked[1][1]
    if s1 >= min_score and (s1 - s2) >= min_margin:
        return best, s1, f, ranked
    return "", s1, f, ranked
