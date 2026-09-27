"""Text-to-speech output, on whatever machine the translator runs on.

Backends, chosen automatically unless the ASL_TTS environment variable names one:

  sapi    Windows' built-in System.Speech -- the development laptops.
  piper   Neural TTS on Linux. Far more natural than espeak; worth it for a
          demo. Opt in by pointing ASL_PIPER_MODEL at a downloaded .onnx voice.
  espeak  espeak-ng on Linux. Tiny, instant, robotic. The Pi's default because
          it installs with apt and needs nothing else.
  say     macOS.
  print   Nothing installed: print the text rather than crash the pipeline.
          Losing speech should never take recognition down with it.

Text is always passed as DATA -- through stdin or an environment variable --
and never spliced into a command line. The previous version built a PowerShell
command with f'$s.Speak("{text}")', so an apostrophe broke speech mid-sentence
("I'm" is ordinary output from the spell-corrector) and a quote or semicolon
could run arbitrary PowerShell. No backend here uses a shell at all.

speak() never blocks the camera loop: it starts the synthesiser and returns. A
new utterance stops one still playing rather than talking over it -- main.py
speaks the whole corrected sentence, so the newer one supersedes the older.
"""
import json
import os
import platform
import shutil
import subprocess

DEVNULL = subprocess.DEVNULL
PIPE = subprocess.PIPE


def detect_backend():
    """Pick the best TTS this machine actually has."""
    forced = os.environ.get("ASL_TTS")
    if forced:
        return forced
    system = platform.system()
    if system == "Windows":
        return "sapi"
    if system == "Darwin":
        return "say"
    model = os.environ.get("ASL_PIPER_MODEL")
    if model and shutil.which("piper") and shutil.which("aplay"):
        return "piper"
    if shutil.which("espeak-ng"):
        return "espeak"
    return "print"


def _piper_sample_rate(model_path):
    """Piper voices ship a <model>.onnx.json saying their sample rate; aplay
    must be told it, or the audio plays at the wrong pitch and speed."""
    try:
        with open(model_path + ".json") as fh:
            return int(json.load(fh)["audio"]["sample_rate"])
    except Exception:
        return 22050   # the rate of the common "medium" voices


class AudioOutput:
    def __init__(self, backend=None, voice="en-us", rate=160):
        self.backend = backend or detect_backend()
        self.voice = voice      # espeak voice
        self.rate = rate        # espeak words per minute
        self.last_spoken = ""
        self._procs = []
        if self.backend == "print":
            print("[audio] no text-to-speech found; printing instead. "
                  "On a Pi: sudo apt install espeak-ng")

    def speak(self, text):
        """Start speaking `text` and return immediately.

        Returns True if speech was started, False if skipped (empty, or the
        same text as last time -- main.py re-speaks on every S press).
        """
        text = (text or "").strip()
        if not text or text == self.last_spoken:
            return False
        self.last_spoken = text
        self.stop()
        self._procs = self._launch(text)
        return True

    def stop(self):
        """Silence anything still playing."""
        for p in self._procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        self._procs = []

    # ── backends ───────────────────────────────────────────────────────────
    def _launch(self, text):
        return getattr(self, "_launch_" + self.backend)(text)

    @staticmethod
    def _feed(proc, text):
        proc.stdin.write(text.encode("utf-8"))
        proc.stdin.close()

    def _launch_espeak(self, text):
        p = subprocess.Popen(
            ["espeak-ng", "-v", self.voice, "-s", str(self.rate), "--stdin"],
            stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)
        self._feed(p, text)
        return [p]

    def _launch_piper(self, text):
        model = os.environ["ASL_PIPER_MODEL"]
        synth = subprocess.Popen(
            ["piper", "-m", model, "--output-raw"],
            stdin=PIPE, stdout=PIPE, stderr=DEVNULL)
        play = subprocess.Popen(
            ["aplay", "-q", "-r", str(_piper_sample_rate(model)),
             "-f", "S16_LE", "-t", "raw", "-c", "1", "-"],
            stdin=synth.stdout, stdout=DEVNULL, stderr=DEVNULL)
        synth.stdout.close()   # so piper gets SIGPIPE if aplay is stopped
        self._feed(synth, text)
        return [synth, play]

    def _launch_sapi(self, text):
        # The script is a constant; the text reaches it only through an
        # environment variable, so nothing in it can ever be parsed as code.
        script = ("Add-Type -AssemblyName System.Speech; "
                  "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  "$s.Speak($env:ASL_TTS_TEXT)")
        p = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            env=dict(os.environ, ASL_TTS_TEXT=text),
            stdout=DEVNULL, stderr=DEVNULL)
        return [p]

    def _launch_say(self, text):
        p = subprocess.Popen(["say", "-f", "-"],
                             stdin=PIPE, stdout=DEVNULL, stderr=DEVNULL)
        self._feed(p, text)
        return [p]

    def _launch_print(self, text):
        print(f"[speak] {text}")
        return []


if __name__ == "__main__":
    # Quick check on any machine: python audio.py "hello there"
    import sys
    out = AudioOutput()
    print(f"backend: {out.backend}")
    out.speak(" ".join(sys.argv[1:]) or "Hello, I am the sign language translator")
    for p in out._procs:
        p.wait()
