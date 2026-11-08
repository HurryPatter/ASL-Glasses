"""When to speak, with no keyboard: the signer lowers their hands.

On the glasses there is no S key. The signer fingerspells a word or a short
phrase, drops their hands out of the camera's view, and after a pause the
text is spoken and cleared, ready for the next. The pause has to be clearly
longer than the debouncer's blank_ms (250ms), which is how long a hand must
leave the view to sign a double letter; 1.5s keeps the two from ever being
confused while still feeling immediate.

Pure logic on timestamps, like debouncer.py, so it is tested offline.
"""


class SpeechTrigger:

    def __init__(self, pause_ms=1500):
        self.pause_ms = pause_ms
        self.absent_since = None

    def update(self, now_ms, hand_present, text):
        """Called every frame. Returns the text to speak now, or None.

        Fires once per pause: the caller clears the text after speaking it,
        so nothing is repeated, and the hand has to come back and go away
        again before anything else can be spoken.
        """
        if hand_present:
            self.absent_since = None
            return None
        if self.absent_since is None:
            self.absent_since = now_ms
            return None
        if text and self.pause_ms and now_ms - self.absent_since >= self.pause_ms:
            self.absent_since = now_ms     # restart the clock for the next one
            return text
        return None

    def reset(self):
        self.absent_since = None
