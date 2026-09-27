"""Offline tests for audio.py -- no speakers, no TTS engines, standard library.

subprocess.Popen is replaced with a recorder, so these check what WOULD be
run: which program, which arguments, and where the text goes. The property
that matters most is that spoken text never appears in a command line.
"""
import os
import unittest
from unittest import mock

import audio

HOSTILE = 'I\'m fine"; Remove-Item -Recurse ~; echo "$HOME `whoami` $(id)'


class FakePopen:
    instances = []

    def __init__(self, args, stdin=None, stdout=None, stderr=None, env=None, **kw):
        self.args, self.env, self.kw = args, env, kw
        self.written = b""
        self.running = True
        self.terminated = False
        self.stdin = self if stdin == audio.PIPE else None
        self.stdout = self if stdout == audio.PIPE else None
        FakePopen.instances.append(self)

    # stdin/stdout stand-ins
    def write(self, data): self.written += data
    def close(self): pass

    def poll(self): return None if self.running else 0
    def terminate(self):
        self.terminated = True
        self.running = False


class Base(unittest.TestCase):
    def setUp(self):
        FakePopen.instances = []
        p = mock.patch.object(audio.subprocess, "Popen", FakePopen)
        p.start(); self.addCleanup(p.stop)
        e = mock.patch.dict(os.environ, {}, clear=False)
        e.start(); self.addCleanup(e.stop)
        for k in ("ASL_TTS", "ASL_PIPER_MODEL"):
            os.environ.pop(k, None)

    def all_argv_text(self):
        return " ".join(" ".join(p.args) for p in FakePopen.instances)


class TestDetection(Base):

    def which(self, available):
        return mock.patch.object(audio.shutil, "which",
                                 lambda n: f"/usr/bin/{n}" if n in available else None)

    def system(self, name):
        return mock.patch.object(audio.platform, "system", lambda: name)

    def test_windows_uses_sapi(self):
        with self.system("Windows"):
            self.assertEqual(audio.detect_backend(), "sapi")

    def test_mac_uses_say(self):
        with self.system("Darwin"):
            self.assertEqual(audio.detect_backend(), "say")

    def test_pi_defaults_to_espeak(self):
        with self.system("Linux"), self.which({"espeak-ng"}):
            self.assertEqual(audio.detect_backend(), "espeak")

    def test_piper_only_when_asked_for_and_installed(self):
        with self.system("Linux"), self.which({"espeak-ng", "piper", "aplay"}):
            self.assertEqual(audio.detect_backend(), "espeak")   # not asked for
            os.environ["ASL_PIPER_MODEL"] = "/v/en_US-lessac-medium.onnx"
            self.assertEqual(audio.detect_backend(), "piper")

    def test_piper_requested_but_missing_falls_back(self):
        os.environ["ASL_PIPER_MODEL"] = "/v/voice.onnx"
        with self.system("Linux"), self.which({"espeak-ng"}):
            self.assertEqual(audio.detect_backend(), "espeak")

    def test_nothing_installed_prints_rather_than_crashing(self):
        with self.system("Linux"), self.which(set()):
            self.assertEqual(audio.detect_backend(), "print")

    def test_env_override_wins(self):
        os.environ["ASL_TTS"] = "print"
        with self.system("Windows"):
            self.assertEqual(audio.detect_backend(), "print")


class TestTextIsDataNotCode(Base):
    """The old sapi backend spliced text into a PowerShell command, so an
    apostrophe broke speech and a quote could run arbitrary code."""

    def test_sapi_text_goes_through_env_never_argv(self):
        audio.AudioOutput(backend="sapi").speak(HOSTILE)
        p, = FakePopen.instances
        self.assertEqual(p.args[0], "powershell")
        self.assertNotIn(HOSTILE, " ".join(p.args))
        for fragment in ("I'm", "Remove-Item", "whoami", "$HOME"):
            self.assertNotIn(fragment, " ".join(p.args))
        self.assertEqual(p.env["ASL_TTS_TEXT"], HOSTILE)

    def test_espeak_text_goes_through_stdin(self):
        audio.AudioOutput(backend="espeak").speak(HOSTILE)
        p, = FakePopen.instances
        self.assertEqual(p.args[0], "espeak-ng")
        self.assertIn("--stdin", p.args)
        self.assertNotIn("Remove-Item", self.all_argv_text())
        self.assertEqual(p.written.decode(), HOSTILE)

    def test_say_text_goes_through_stdin(self):
        audio.AudioOutput(backend="say").speak(HOSTILE)
        p, = FakePopen.instances
        self.assertNotIn("Remove-Item", self.all_argv_text())
        self.assertEqual(p.written.decode(), HOSTILE)

    def test_piper_text_goes_to_the_synthesiser_and_audio_to_aplay(self):
        os.environ["ASL_PIPER_MODEL"] = "/v/voice.onnx"
        audio.AudioOutput(backend="piper").speak(HOSTILE)
        synth, play = FakePopen.instances
        self.assertEqual(synth.args[:3], ["piper", "-m", "/v/voice.onnx"])
        self.assertIn("--output-raw", synth.args)
        self.assertEqual(play.args[0], "aplay")
        self.assertEqual(synth.written.decode(), HOSTILE)
        self.assertNotIn("Remove-Item", self.all_argv_text())

    def test_no_backend_uses_a_shell(self):
        for backend in ("sapi", "espeak", "say"):
            FakePopen.instances = []
            audio.AudioOutput(backend=backend).speak("hello")
            for p in FakePopen.instances:
                self.assertFalse(p.kw.get("shell"), backend)
                self.assertIsInstance(p.args, list, backend)

    def test_unicode_survives(self):
        audio.AudioOutput(backend="espeak").speak("café naïve")
        self.assertEqual(FakePopen.instances[0].written.decode("utf-8"), "café naïve")


class TestBehaviour(Base):

    def test_same_text_twice_speaks_once(self):
        a = audio.AudioOutput(backend="espeak")
        self.assertTrue(a.speak("hello"))
        self.assertFalse(a.speak("hello"))
        self.assertEqual(len(FakePopen.instances), 1)

    def test_empty_or_blank_is_ignored(self):
        a = audio.AudioOutput(backend="espeak")
        self.assertFalse(a.speak(""))
        self.assertFalse(a.speak("   "))
        self.assertFalse(a.speak(None))
        self.assertEqual(FakePopen.instances, [])

    def test_new_sentence_stops_the_one_still_playing(self):
        a = audio.AudioOutput(backend="espeak")
        a.speak("hello")
        first = FakePopen.instances[0]
        a.speak("hello world")
        self.assertTrue(first.terminated)
        self.assertEqual(len(FakePopen.instances), 2)

    def test_piper_stop_silences_both_processes(self):
        os.environ["ASL_PIPER_MODEL"] = "/v/voice.onnx"
        a = audio.AudioOutput(backend="piper")
        a.speak("hello")
        a.stop()
        self.assertTrue(all(p.terminated for p in FakePopen.instances))

    def test_print_backend_never_launches_anything(self):
        with mock.patch("builtins.print"):
            a = audio.AudioOutput(backend="print")
            self.assertTrue(a.speak("hello"))
        self.assertEqual(FakePopen.instances, [])


class TestPiperSampleRate(unittest.TestCase):

    def test_reads_rate_from_the_voice_config(self):
        import json, tempfile
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "voice.onnx")
            with open(model + ".json", "w") as fh:
                json.dump({"audio": {"sample_rate": 16000}}, fh)
            self.assertEqual(audio._piper_sample_rate(model), 16000)

    def test_defaults_when_config_missing(self):
        self.assertEqual(audio._piper_sample_rate("/nonexistent.onnx"), 22050)


if __name__ == "__main__":
    unittest.main(verbosity=2)
