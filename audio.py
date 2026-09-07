import subprocess
import sys

class AudioOutput:
    def __init__(self):
        self.last_spoken = ""

    def speak(self, text):
        if text and text != self.last_spoken:
            self.last_spoken = text
            # Uses Windows built-in PowerShell TTS — no extra libraries needed
            cmd = (
                f'Add-Type -AssemblyName System.Speech; '
                f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                f'$s.Speak("{text}")'
            )
            subprocess.Popen(
                ["powershell", "-Command", cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )