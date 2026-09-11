class Debouncer:
    def __init__(self, min_frames=3):
        self.min_frames = min_frames
        self.current_letter = None
        self.frame_count = 0
        self.frame_no = 0
        self.commits = []        # list of (letter, frame_no) in order

    @property
    def confirmed_string(self):
        return "".join(letter for letter, _ in self.commits)

    def update(self, letter, frame_no=None):
        """Feed in the raw per-frame prediction each frame. Returns confirmed string.

        Commits exactly once per continuous hold of the same letter. A real
        double letter (the "LL" in HELLO) is captured by the signer bouncing
        briefly out of the shape and back in -- that produces two separate
        holds, and therefore two separate commits, naturally.
        """
        self.frame_no = self.frame_no + 1 if frame_no is None else frame_no

        if letter == self.current_letter:
            self.frame_count += 1
        else:
            self.current_letter = letter
            self.frame_count = 1

        if self.frame_count == self.min_frames and letter:
            self.commits.append((letter, self.frame_no))

        return self.confirmed_string

    def commit_motion(self, letter, start_frame):
        """A motion letter (J/Z) was recognised over frames [start_frame, now].

        Any static letters committed during that span were the classifier
        misreading the start of the motion (e.g. an 'I' before the J hook),
        so drop them.
        """
        self.commits = [c for c in self.commits if c[1] < start_frame]
        self.commits.append((letter, self.frame_no))
        self.current_letter = None
        self.frame_count = 0
        return self.confirmed_string

    def reset(self):
        self.commits = []
        self.current_letter = None
        self.frame_count = 0