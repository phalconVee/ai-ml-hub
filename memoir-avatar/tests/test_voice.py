"""Voices and lip-sync: the envelope, WAV wrapping, engine choice, and a live `say` run."""

import io
import json
import platform
import shutil
import wave

import pytest

from memoir.avatar import avatar_payload
from memoir.config import SETTINGS
from memoir.voice import ElevenLabsEngine, available_engines, envelope, pcm_to_wav, speak

from .fakes import ToneEngine


def test_envelope_follows_loudness():
    pcm = ToneEngine().synthesize("x" * 50, "v")          # 1.0 s tone + 0.2 s silence
    env = envelope(pcm)
    assert len(env) == 36                                  # 1.2 s × 30 fps
    assert min(env[2:28]) > 0.9                            # mouth open while the tone plays
    assert env[-3:] == [0.0, 0.0, 0.0]                     # closed in the silence
    assert envelope(b"") == [] and envelope(b"\x00" * 4800) == [0.0] * 3


def test_wav_wrapping_round_trips():
    pcm = ToneEngine().synthesize("hi", "v")
    with wave.open(io.BytesIO(pcm_to_wav(pcm))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
        assert w.readframes(w.getnframes()) == pcm


def test_speak_with_a_server_engine_and_the_browser():
    tone = ToneEngine()
    s = speak("hello there", "say", "Samantha", engine_obj=tone)
    assert tone.calls == [("hello there", "Samantha")]
    assert s.wav.startswith(b"RIFF") and len(s.envelope) == round(s.duration_s * 30)
    b = speak("hello", "browser")
    assert b.wav == b"" and b.envelope == []


def test_avatar_payload():
    s = speak("hey", "say", engine_obj=ToneEngine())
    p = avatar_payload("id1", "hey", s, voice="Samantha")
    assert p["audio_b64"] and p["envelope"] == s.envelope and p["fps"] == SETTINGS.envelope_fps
    json.dumps(p)                                          # must be JSON-safe for the component
    browser = avatar_payload("id2", "hey", speak("hey", "browser"))
    assert browser["audio_b64"] == "" and browser["text"] == "hey"


def test_engine_list(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert "browser" in available_engines() and "elevenlabs" not in available_engines()
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    assert available_engines()[-1] == "elevenlabs"         # paid is offered last, never first


def test_elevenlabs_request_is_built_correctly():
    seen = {}

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def opener(req, timeout):
        seen.update(url=req.full_url, body=json.loads(req.data), key=req.headers["Xi-api-key"])
        return Resp(b"\x00\x00" * 100)

    pcm = ElevenLabsEngine("secret", opener=opener).synthesize("hi", "VOICE")
    assert seen["url"].endswith("/VOICE?output_format=pcm_24000")
    assert seen["body"]["text"] == "hi" and seen["key"] == "secret" and len(pcm) == 200


@pytest.mark.live
@pytest.mark.skipif(platform.system() != "Darwin" or not shutil.which("say"), reason="needs macOS say")
def test_live_macos_say_lip_sync():
    s = speak("Hello, I remember you.", "say", "Samantha")
    assert s.duration_s > 0.5
    assert max(s.envelope) > 0.9 and 0.0 in s.envelope      # it opens and it closes
