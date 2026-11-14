"""Engine choice, the ElevenLabs request shape (mocked), and live local TTS when present."""

import json
import platform
import shutil
import wave

import pytest

from podcast import voices
from podcast.config import SETTINGS
from podcast.script import Line, Script
from podcast.studio import record
from podcast.voices import ElevenLabsEngine, VoiceError, pick_engine


def test_auto_prefers_free_local_engines(monkeypatch):
    monkeypatch.setattr(voices, "available_engines", lambda: ["elevenlabs"])
    with pytest.raises(VoiceError, match="No free local voice"):
        pick_engine("auto")  # a paid engine is never picked automatically

    monkeypatch.setattr(voices, "available_engines", lambda: ["say", "elevenlabs"])
    monkeypatch.setitem(voices.ENGINES, "say", lambda: "SAY")
    assert pick_engine("auto") == "SAY"


def test_elevenlabs_needs_a_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(VoiceError, match="ELEVENLABS_API_KEY"):
        ElevenLabsEngine()


def test_elevenlabs_request_shape():
    sent = []

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"\x00\x01" * 10

    def opener(req, timeout):
        sent.append(req)
        return Resp()

    pcm = ElevenLabsEngine("test-key", opener=opener).synthesize("Hello.", "voice123")
    [req] = sent
    assert req.full_url.endswith("/v1/text-to-speech/voice123?output_format=pcm_24000")
    assert req.get_header("Xi-api-key") == "test-key" and req.get_method() == "POST"
    assert json.loads(req.data) == {"text": "Hello.", "model_id": SETTINGS.elevenlabs_model}
    assert pcm == b"\x00\x01" * 10


def test_kokoro_not_ready_without_model_files(tmp_path):
    assert not voices.kokoro_ready(tmp_path)


# ── Live checks: a real local engine on a short line (no keys, no network) ──

@pytest.mark.live
@pytest.mark.skipif(platform.system() != "Darwin" or not shutil.which("say"), reason="needs macOS say")
def test_live_macos_say_two_hosts(tmp_path):
    engine = pick_engine("say")
    s = Script("t", [Line("A", "Welcome to the show."), Line("B", "Glad to be here.")])
    ep = record(s, engine, {"A": "Samantha", "B": "Daniel"})
    out = tmp_path / "ep.wav"
    out.write_bytes(ep.wav)
    with wave.open(str(out)) as w:
        assert w.getframerate() == SETTINGS.sample_rate and w.getnchannels() == 1
    assert 1.5 < ep.seconds < 10 and all(s > 0.5 for s in ep.line_seconds)


@pytest.mark.live
@pytest.mark.skipif(not voices.kokoro_ready(), reason="Kokoro not installed / model not downloaded")
def test_live_kokoro_line():
    pcm = pick_engine("kokoro").synthesize("Hello from Kokoro.", "af_heart")
    assert len(pcm) > SETTINGS.sample_rate  # more than half a second of audio
