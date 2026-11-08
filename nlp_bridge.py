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
        """Takes a fingerspelled letter string, returns corrected text.
        Only ever called on runs of single-character labels -- see
        correct_sequence() for input containing word-sign tokens too.

        Three steps, cheapest and safest first:

        1. Split into words WITHOUT changing a letter. SymSpell's segmentation
           with its normal edit distance treats deleting a letter as a cheap
           correction, so it used to drop the one-letter words: IAMOMAR came
           out "Am Omar", ILOVEYOU "Love You", ABAG "Bag". At edit distance 0
           those read "I am Omar", "I love you", "A bag".
        2. Pieces that are not words get a doubled letter tried first. The
           debouncer commits one letter per hold, so the second L of HELLO
           only appears if the signer drops the hand and signs it again: a
           missed double letter is the most likely slip, and plain spelling
           correction gets it wrong (HELO came out "Help").
        3. Otherwise ordinary correction, accepted only when the change is
           small for the length of the piece, so that random letters are left
           as signed rather than turned into an unrelated word (XQZ -> "xxx").
        """
        if not raw_string:
            return ""
        tokens = self.sym_spell.word_segmentation(
            raw_string.lower(), max_edit_distance=0).corrected_string.split()
        known = [self._is_word(t) for t in tokens]

        words = []
        last_is_free = False    # words[-1] is a plain known word we may absorb
        i = 0
        while i < len(tokens):
            if known[i]:
                words.append(tokens[i])
                last_is_free = True
                i += 1
                continue
            j = i
            while j < len(tokens) and not known[j]:
                j += 1
            # A missed double letter often splits into a word plus debris
            # ("hel o" for HELO), so the doubled-letter repair also gets to
            # use the known words on either side of the unknown run.
            before = [words.pop()] if last_is_free else []
            after = [tokens[j]] if j < len(tokens) else []
            fixed = self._doubled("".join(before + tokens[i:j] + after))
            if fixed:
                words.extend(fixed.split())
                i = j + len(after)
            else:
                words.extend(before)
                words.extend(self._corrected("".join(tokens[i:j])).split())
                i = j
            last_is_free = False
        return self._format(words)

    def _is_word(self, token):
        # Every single letter is in the frequency dictionary; only these two
        # are words on their own.
        if len(token) == 1 and token not in ("i", "a"):
            return False
        return bool(self.sym_spell.lookup(token, Verbosity.TOP,
                                          max_edit_distance=0))

    def _all_words(self, text):
        seg = self.sym_spell.word_segmentation(text, max_edit_distance=0)
        tokens = seg.corrected_string.split()
        if tokens and all(self._is_word(t) for t in tokens):
            return seg.corrected_string
        return None

    def _doubled(self, span):
        """The span with one letter doubled, if that makes it all words."""
        for k in range(len(span)):
            fixed = self._all_words(span[:k + 1] + span[k] + span[k + 1:])
            if fixed:
                return fixed
        return None

    def _corrected(self, span):
        """Ordinary spelling correction, only when the change is small. A lone
        stray letter is left alone: correcting one letter means replacing it,
        which is a guess (CATX read "Cat a")."""
        if len(span) < 2:
            return span
        seg = self.sym_spell.word_segmentation(span, max_edit_distance=2)
        allowed = 1 if len(span) <= 4 else 2
        if seg.corrected_string and seg.distance_sum <= allowed:
            return seg.corrected_string
        return span

    @staticmethod
    def _format(words):
        """Sentence case, with I and the team's names capitalised."""
        out = []
        for i, w in enumerate(words):
            if w == "i" or w in TEAM_NAMES or i == 0:
                w = w[:1].upper() + w[1:]
            out.append(w)
        return " ".join(out)

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