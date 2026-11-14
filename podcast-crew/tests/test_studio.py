"""Voicing and stitching, with a pretend TTS that makes real PCM audio."""

import io
import wave

from podcast.config import SETTINGS
from podcast.script import Line, Script
from podcast.studio import record, silence

from .fakes import ToneEngine


def test_episode_is_a_valid_wav_with_the_right_length_and_gaps():
    s = Script("t", [Line("A", "Hello there."), Line("B", "Hi!"), Line("B", "Go on."),
                     Line("A", "[laughs]")])  # last line is empty after cleaning: skipped
    engine, progress = ToneEngine(), []
    ep = record(s, engine, {"A": "low", "B": "high"}, on_line=lambda i, n: progress.append((i, n)))

    # Each host spoke with their own voice; the stage direction was never voiced.
    assert engine.calls == [("Hello there.", "low"), ("Hi!", "high"), ("Go on.", "high")]
    assert progress == [(1, 3), (2, 3), (3, 3)]

    with wave.open(io.BytesIO(ep.wav)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SETTINGS.sample_rate)
        frames = w.getnframes()

    speech = sum(len(t) for t, _ in engine.calls) * 10            # 10 ms per character
    pauses = 2 * SETTINGS.lead_in_ms + SETTINGS.gap_turn_ms + SETTINGS.gap_same_speaker_ms
    assert frames == (speech + pauses) * SETTINGS.sample_rate // 1000
    assert abs(ep.seconds - frames / SETTINGS.sample_rate) < 1e-9
    assert ep.engine == "tone" and len(ep.line_seconds) == 3


def test_silence_is_exactly_zero_bytes():
    assert silence(1000) == b"\x00" * (SETTINGS.sample_rate * 2)


def test_odd_byte_from_an_engine_is_dropped():
    class Odd(ToneEngine):
        def synthesize(self, text, voice):
            return b"\x01\x00\x02"  # 1.5 samples

    s = Script("t", [Line("A", "x"), Line("B", "y")])
    ep = record(s, Odd(), {"A": "a", "B": "b"})
    with wave.open(io.BytesIO(ep.wav)) as w:
        data = w.readframes(w.getnframes())
    assert len(data) % 2 == 0
