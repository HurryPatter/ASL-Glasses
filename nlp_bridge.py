from symspellpy import SymSpell, Verbosity
import pkg_resources

# Team names -- always recognized, independent of academic/normal mode.
TEAM_NAMES = ["omar", "hagar", "laila", "nourhan", "hassan"]


class NLPBridge:
    def __init__(self, mode="normal"):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

        # Load the built-in Wikipedia frequency dictionary
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

        self._inject_names()

        self.mode = mode
        self._inject_domain_vocab()

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
        """Takes the debounced letter string, returns corrected sentence."""
        if not raw_string:
            return ""
        results = self.sym_spell.word_segmentation(raw_string)
        return results.corrected_string

    def set_mode(self, mode):
        self.mode = mode
        self._inject_domain_vocab()