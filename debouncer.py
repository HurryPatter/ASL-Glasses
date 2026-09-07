class Debouncer:
    def __init__(self, min_frames=3):
        self.min_frames = min_frames
        self.current_letter = None
        self.frame_count = 0
        self.confirmed_string = ""

    def update(self, letter):
        """Feed in the raw CNN prediction each frame. Returns confirmed string."""
        if letter == self.current_letter:
            self.frame_count += 1
        else:
            self.current_letter = letter
            self.frame_count = 1

        # Only accept a letter once it's been held for min_frames
        if self.frame_count == self.min_frames:
            self.confirmed_string += letter

        return self.confirmed_string

    def reset(self):
        self.confirmed_string = ""
        self.current_letter = None
        self.frame_count = 0