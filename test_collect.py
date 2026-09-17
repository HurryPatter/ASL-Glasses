"""Offline test for the hands-free countdown timing.

Standard library only. The countdown exists so two-handed signs can be
recorded alone -- both hands are up, so nobody is left to press a key -- and
its timing can otherwise only be checked with a camera in front of you, which
is the worst place to find a timing bug.

Run:  python -m unittest test_collect -v
"""
import ast
import types
import unittest

# collect_signs imports cv2, so lift just the pure timing function out of it.
_tree = ast.parse(open("collect_signs.py").read())
_fn = next(n for n in _tree.body
           if isinstance(n, ast.FunctionDef) and n.name == "due")
_module = types.ModuleType("collect_stub")
exec(compile(ast.Module(body=[_fn], type_ignores=[]), "<stub>", "exec"),
     _module.__dict__)
due = _module.due

COUNTDOWN_MS = 3000
AUTO_CLIP_MS = 2000


def idle():
    return {"recording": False, "countdown_until": None, "stop_at": None}


def armed(at=0):
    return {"recording": False, "countdown_until": at + COUNTDOWN_MS,
            "stop_at": None}


def recording(since=0):
    return {"recording": True, "countdown_until": None,
            "stop_at": since + AUTO_CLIP_MS}


class TestCountdown(unittest.TestCase):

    def test_idle_does_nothing(self):
        self.assertIsNone(due(idle(), 10_000))

    def test_armed_waits_out_the_countdown(self):
        state = armed()
        self.assertIsNone(due(state, 0))
        self.assertIsNone(due(state, COUNTDOWN_MS - 1))
        self.assertEqual(due(state, COUNTDOWN_MS), "start")

    def test_recording_stops_after_the_clip_length(self):
        state = recording()
        self.assertIsNone(due(state, AUTO_CLIP_MS - 1))
        self.assertEqual(due(state, AUTO_CLIP_MS), "stop")

    def test_a_manual_recording_never_auto_stops(self):
        # SPACE starts without a stop_at, so it runs until SPACE again.
        state = {"recording": True, "countdown_until": None, "stop_at": None}
        self.assertIsNone(due(state, 60_000))

    def test_a_slow_frame_past_the_deadline_still_fires(self):
        # The loop only sees the frames the camera delivers, so a deadline
        # can be crossed between two frames rather than landing on one.
        self.assertEqual(due(armed(), COUNTDOWN_MS + 400), "start")
        self.assertEqual(due(recording(), AUTO_CLIP_MS + 400), "stop")

    def test_a_full_hands_free_take(self):
        state = armed(at=1000)
        self.assertIsNone(due(state, 3_500))
        self.assertEqual(due(state, 4_000), "start")

        state = recording(since=4_000)
        self.assertIsNone(due(state, 5_000))
        self.assertEqual(due(state, 6_000), "stop")

    def test_the_clip_is_long_enough_to_be_a_sign(self):
        # save_clip() rejects anything under SignBuffer's floors, so an
        # auto-clip that is too short would silently never save.
        import sequence
        self.assertGreater(AUTO_CLIP_MS, sequence.SignBuffer().min_span_ms)


if __name__ == "__main__":
    unittest.main()
