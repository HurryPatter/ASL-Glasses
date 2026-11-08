"""Offline tests for speech_trigger.py. Standard library only."""
import unittest

from speech_trigger import SpeechTrigger


def run(trigger, frames, text="HELLO"):
    """frames: [(ms, hand_present)]; returns [(ms, spoken)] for each firing.
    Mirrors main.py: whatever is spoken is cleared."""
    spoken = []
    for ms, hand in frames:
        said = trigger.update(ms, hand, text)
        if said:
            spoken.append((ms, said))
            text = ""
    return spoken


def at_30fps(start, end, hand):
    return [(ms, hand) for ms in range(start, end, 33)]


class TestSpeechTrigger(unittest.TestCase):

    def test_dropping_the_hands_speaks_after_the_pause(self):
        frames = at_30fps(0, 1000, True) + at_30fps(1000, 3000, False)
        spoken = run(SpeechTrigger(1500), frames)
        self.assertEqual(len(spoken), 1)
        self.assertGreaterEqual(spoken[0][0] - 1000, 1500)
        self.assertLess(spoken[0][0] - 1000, 1600)

    def test_a_double_letter_bounce_does_not_speak(self):
        # Leaving the view for ~300ms is how a double letter is signed.
        frames = (at_30fps(0, 1000, True) + at_30fps(1000, 1300, False)
                  + at_30fps(1300, 2000, True))
        self.assertEqual(run(SpeechTrigger(1500), frames), [])

    def test_nothing_signed_means_nothing_spoken(self):
        self.assertEqual(run(SpeechTrigger(1500), at_30fps(0, 5000, False), text=""), [])

    def test_speaks_once_per_pause(self):
        frames = at_30fps(0, 500, True) + at_30fps(500, 8000, False)
        self.assertEqual(len(run(SpeechTrigger(1500), frames)), 1)

    def test_each_phrase_gets_its_own_pause(self):
        t = SpeechTrigger(1500)
        first = run(t, at_30fps(0, 500, True) + at_30fps(500, 2500, False), "HI")
        second = run(t, at_30fps(2500, 3000, True) + at_30fps(3000, 5000, False), "BYE")
        self.assertEqual([s for _, s in first + second], ["HI", "BYE"])

    def test_zero_pause_disables_it(self):
        frames = at_30fps(0, 500, True) + at_30fps(500, 5000, False)
        self.assertEqual(run(SpeechTrigger(0), frames), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
