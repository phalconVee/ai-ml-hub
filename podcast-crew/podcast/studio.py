"""The studio: voice every line, then stitch one episode WAV.

Because every engine hands back the same raw format (24 kHz mono 16-bit
PCM), "mixing" is just this, in order:

    lead-in silence, line 1, gap, line 2, gap, … line N, lead-out silence

A gap is a run of zero bytes. It's a little longer when the speaker
changes, which is what makes two synthetic voices sound like they're
taking turns instead of reading one list. The standard-library `wave`
module then writes a proper WAV header around the bytes.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field
from typing import Callable

from .config import SETTINGS
from .script import Script, clean_text

BYTES_PER_SAMPLE = 2  # 16-bit


def silence(ms: int) -> bytes:
    """`ms` milliseconds of digital silence."""
    samples = SETTINGS.sample_rate * ms // 1000
    return b"\x00" * (samples * BYTES_PER_SAMPLE)


def seconds(pcm: bytes) -> float:
    return len(pcm) / BYTES_PER_SAMPLE / SETTINGS.sample_rate


def to_wav(pcm: bytes) -> bytes:
    """Wrap raw PCM in a WAV header (in memory: Streamlit plays bytes directly)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(BYTES_PER_SAMPLE)
        w.setframerate(SETTINGS.sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


@dataclass
class Episode:
    wav: bytes                                          # the finished file
    seconds: float
    engine: str
    line_seconds: list[float] = field(default_factory=list)  # per line, for the UI


def record(script: Script, engine, voices: dict[str, str],
           on_line: Callable[[int, int], None] = lambda i, n: None) -> Episode:
    """Voice each line with its host's voice and stitch the episode.

    voices  : {"A": voice id, "B": voice id} for the chosen engine
    on_line : progress callback (line number, total) for the UI's progress bar
    """
    parts = [silence(SETTINGS.lead_in_ms)]
    line_secs: list[float] = []
    lines = [l for l in script.lines if clean_text(l.text)]  # never voice an empty line
    for i, line in enumerate(lines):
        if i > 0:  # pause before every line but the first
            same = line.speaker == lines[i - 1].speaker
            parts.append(silence(SETTINGS.gap_same_speaker_ms if same else SETTINGS.gap_turn_ms))
        pcm = engine.synthesize(clean_text(line.text), voices[line.speaker])
        if len(pcm) % 2:          # a stray odd byte would shift every later sample
            pcm = pcm[:-1]
        parts.append(pcm)
        line_secs.append(seconds(pcm))
        on_line(i + 1, len(lines))
    parts.append(silence(SETTINGS.lead_in_ms))
    pcm = b"".join(parts)
    return Episode(to_wav(pcm), seconds(pcm), getattr(engine, "name", "custom"), line_secs)
