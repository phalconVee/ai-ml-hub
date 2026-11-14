"""Audio stitching with a fake voice, plus a live check of the local voice if present."""

from __future__ import annotations

import math
import shutil
import struct
import wave

import pytest

from sourcebook import tts


class ToneBackend:
    """Writes a short sine tone per turn (different pitch per host)."""
    name = "tone"

    def synth(self, text, voice, out):
        rate, secs, freq = 24_000, 0.2, 440 if voice == 0 else 660
        frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate)))
                          for i in range(int(rate * secs)))
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate), w.writeframes(frames)


def test_render_concatenates_with_pauses(tmp_path):
    out = tmp_path / "o.wav"
    progress = []
    secs = tts.render([("A", "hi"), ("B", "hello"), ("A", "bye")], ToneBackend(), out,
                      pause_s=0.1, on_progress=lambda d, t: progress.append((d, t)))
    assert secs == pytest.approx(3 * 0.2 + 2 * 0.1, abs=1e-3)
    with wave.open(str(out)) as w:
        assert w.getframerate() == 24_000 and w.getnframes() == int(24_000 * 0.8)
    assert progress[-1] == (3, 3)


def test_pick_backend_respects_choice(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert tts.pick_backend("none") is None
    assert tts.pick_backend("elevenlabs") is None          # no key -> not chosen
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    assert isinstance(tts.pick_backend("auto"), tts.ElevenLabsBackend)


@pytest.mark.skipif(not shutil.which("say"), reason="macOS `say` not available")
def test_live_macos_say(tmp_path):
    out = tmp_path / "say.wav"
    secs = tts.render([("A", "Welcome to the overview."), ("B", "-- Thanks, glad to be here.")],
                      tts.SayBackend(), out)
    assert secs > 1.0 and out.stat().st_size > 10_000
