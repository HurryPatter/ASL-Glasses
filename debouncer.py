class Debouncer:
    def __init__(self, min_frames=3, repeat_interval=None):
        self.min_frames = min_frames
        # After the initial commit, how many additional frames of a
        # continuous hold before the SAME letter is allowed to commit again.
        # This is what makes a double letter (the "LL" in HELLO) register as
        # two letters even if the signer holds it as one continuous shape
        # instead of visibly bouncing between the two. Defaults to the same
        # cadence as the initial commit.
        self.repeat_interval = repeat_interval or min_frames
        self.current_letter = None
        self.frame_count = 0
        self.frame_no = 0
        self.commits = []        # list of (letter, frame_no) in order

    @property
    def confirmed_string(self):
        return "".join(letter for letter, _ in self.commits)

    def update(self, letter, frame_no=None):
        """Feed in the raw per-frame prediction each frame. Returns confirmed string."""
        self.frame_no = self.frame_no + 1 if frame_no is None else frame_no

        if letter == self.current_letter:
            self.frame_count += 1
        else:
            self.current_letter = letter
            self.frame_count = 1

        # Commit at min_frames, then again every repeat_interval frames after
        # that for as long as the same letter keeps being held.
        if letter and self.frame_count >= self.min_frames:
            frames_since_first_commit = self.frame_count - self.min_frames
            if frames_since_first_commit % self.repeat_interval == 0:
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