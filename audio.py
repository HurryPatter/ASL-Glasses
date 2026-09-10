import subprocess
import sys


class AudioOutput:
    """Text-to-speech using whatever the operating system already ships.

    Windows : PowerShell + System.Speech
    macOS   : the built-in `say` command
    Linux   : `espeak` if installed
    A missing speech tool is reported, never fatal.
    """

    def __init__(self):
        self.last_spoken = ""

    def _command(self, text):
        if sys.platform.startswith("win"):
            safe = text.replace('"', '`"')  # escape double quotes for PowerShell
            return ["powershell", "-Command",
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f'$s.Speak("{safe}")']
        if sys.platform == "darwin":
            return ["say", text]
        return ["espeak", text]

    def speak(self, text):
        if not text or text == self.last_spoken:
            return
        self.last_spoken = text
        try:
            subprocess.Popen(self._command(text),
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print(f"[audio] no speech tool found on {sys.platform}; would have said: {text}")
