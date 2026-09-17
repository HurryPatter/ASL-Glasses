"""Persisted capture conventions — Project Veronica.

One setting lives here, and it is here rather than as a default argument
because getting it wrong is silent: **whether the camera feed reaching
MediaPipe is mirrored.**

MediaPipe reports handedness relative to an assumed mirroring of the input.
Which way that assumption runs depends on the MediaPipe build *and* on the
camera, since some webcams deliver an already-mirrored feed that `cv2.flip`
then un-mirrors. It cannot be settled by reading code, only by holding up a
known hand in front of the actual camera -- which is what `check_setup.py`
does, and why it writes the answer here instead of asking every tool to
rediscover it.

A wrong value labels every left hand right. Nothing downstream can detect
that: the features stay entirely plausible, training succeeds, and the number
at the end is merely wrong.

So the value is also **recorded into every recorded clip** (see signset.py).
If it is ever found to have been wrong, `rebuild_signs.py --mirrored-input`
re-derives every training row from the raw archive under the corrected
convention. Nobody has to sign anything twice.
"""
import json
import os

CONFIG_PATH = "veronica_config.json"

DEFAULTS = {
    # True  -> MediaPipe's handedness labels are used as reported.
    # False -> they are swapped before use.
    "mirrored_input": True,
}


def load(path=CONFIG_PATH):
    settings = dict(DEFAULTS)
    if os.path.exists(path):
        try:
            with open(path) as fh:
                stored = json.load(fh)
            settings.update({k: v for k, v in stored.items() if k in DEFAULTS})
        except (ValueError, OSError):
            pass          # a corrupt config must not stop a session starting
    return settings


def save(settings, path=CONFIG_PATH):
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in settings.items() if k in DEFAULTS})
    with open(path, "w") as fh:
        json.dump(merged, fh, indent=2)
        fh.write("\n")
    return merged


def mirrored_input(path=CONFIG_PATH):
    return load(path)["mirrored_input"]


def describe(settings=None):
    settings = settings or load()
    if settings["mirrored_input"]:
        return "handedness labels used as reported (mirrored_input=True)"
    return "handedness labels swapped (mirrored_input=False)"
