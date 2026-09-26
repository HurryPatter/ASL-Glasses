from importlib.resources import files

from symspellpy import SymSpell, Verbosity

# Team names -- always recognized, independent of academic/normal mode.
TEAM_NAMES = ["omar", "hagar", "laila", "nourhan", "hassan", "crossfit"]

# Static word signs, mapped to how they should be displayed. Anything the
# classifier outputs that isn't a single letter is looked up here -- this is
# the only place a new static word sign needs to be registered once its
# training data and label exist (see collect_data.py's WORDS list).
WORD_SIGNS = {
    "ILY": "I love you",
    "IHATEYOU": "I hate you",
    "HELLO": "Hello",
}


class NLPBridge:
    def __init__(self, mode="normal"):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

        # Load the built-in Wikipedia frequency dictionary. Located with the
        # standard library rather than pkg_resources: setuptools 84 removed
        # pkg_resources entirely, and requirements.txt put no ceiling on
        # setuptools, so a fresh install -- the Pi's first one -- crashed here.
        dictionary_path = str(
            files("symspellpy") / "frequency_dictionary_en_82_765.txt"
        )
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

        self._inject_names()

        self.mode = mode
        self._inject_domain_vocab()

        # Last input and output of correct_sequence(). See there for why.
        self._last_labels = None
        self._last_result = ""

    def _inject_names(self):
        """Boost the team's names so they don't get segmented/misspelled
        away. Unlike academic-mode vocab, this always applies."""
        for name in TEAM_NAMES:
            self.sym_spell.create_dictionary_entry(name, 10**9)

    def _inject_domain_vocab(self):
        """Boost domain-specific words based on current mode."""
        if self.mode == "academic":
            engineering_words = [
                "microcontroller", "mediapipe", "convolutional", "latency",
                "debounce", "segmentation", "wearable", "inference",
                "bandwidth", "accelerometer", "gyroscope",
            ]
            for word in engineering_words:
                self.sym_spell.create_dictionary_entry(word, 10**9)

    def correct(self, raw_string):
        """Takes a fingerspelled letter string, returns corrected sentence.
        Only ever called on runs of single-character labels -- see
        correct_sequence() for input containing word-sign tokens too."""
        if not raw_string:
            return ""
        results = self.sym_spell.word_segmentation(raw_string)
        return results.corrected_string

    def correct_sequence(self, labels):
        """Takes the debouncer's ordered list of committed labels -- a mix
        of single fingerspelled letters and whole-word sign tokens (e.g.
        'ILY') -- and returns corrected, readable text.

        Word tokens are NOT run through spell-segmentation (they're already
        a complete lexical unit, not letters to be spelled out); consecutive
        letters are grouped into runs and corrected exactly as before.

        Returns the previous result unchanged when the labels have not changed.
        main.py calls this every frame, but the committed labels only change a
        few times a second, and segmentation cost grows much faster than
        linearly with sentence length -- measured at 0.8ms for 5 letters,
        9.8ms for 15 and 53.7ms for 40. Recomputing every frame meant the
        frame rate fell the longer someone signed: a 40-letter sentence cost
        more than an entire 30fps frame on the laptop, before the slower Pi,
        and J/Z stop firing well above that. It is now paid once per commit.
        """
        key = tuple(labels)
        if key == self._last_labels:
            return self._last_result

        pieces = []
        letter_buffer = []

        def flush():
            if letter_buffer:
                corrected = self.correct("".join(letter_buffer))
                if corrected:
                    pieces.append(corrected)
                letter_buffer.clear()

        for label in labels:
            if len(label) == 1:
                letter_buffer.append(label)
            else:
                flush()
                pieces.append(WORD_SIGNS.get(label, label))
        flush()

        self._last_labels = key
        self._last_result = " ".join(pieces)
        return self._last_result

    def set_mode(self, mode):
        self.mode = mode
        self._inject_domain_vocab()
        # The dictionary just changed, so a cached correction may be stale.
        self._last_labels = None