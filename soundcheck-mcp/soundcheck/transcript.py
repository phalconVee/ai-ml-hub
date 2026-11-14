"""The one data structure every tool shares: a timestamped transcript.

    Transcript
      audio_id        short hash of the audio bytes (stable across runs)
      source_path     where the audio lives (diarization re-reads it)
      segments[]      Segment(start, end, text, speaker)

Every tool on the server works on an `audio_id`, never on raw audio: the first
call (`transcribe_audio`) does the expensive speech-to-text once and saves the
result here as JSON; everything afterwards (search, sentiment, speakers,
summary) just loads that file. That is what makes the toolkit cheap to chat
with: ten questions about a recording cost one transcription.
"""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from soundcheck.config import settings


class Segment(BaseModel):
    """One stretch of speech, roughly a sentence or a breath group."""

    start: float  # seconds from the beginning of the file
    end: float
    text: str
    speaker: str | None = None  # filled in later by identify_speakers (or AssemblyAI)


class Transcript(BaseModel):
    audio_id: str
    source_name: str  # original filename, for display
    source_path: str  # absolute path to the audio on disk
    duration: float  # seconds
    language: str | None = None
    engine: str  # which speech-to-text produced it, e.g. "whisper:base"
    segments: list[Segment] = Field(default_factory=list)
    # How the speaker labels were produced: None, "heuristic" or "assemblyai".
    speaker_method: str | None = None

    @property
    def text(self) -> str:
        """Plain text, no timestamps: what the NLP helpers chew on."""
        return " ".join(s.text.strip() for s in self.segments)

    def window(self, start: float = 0.0, end: float | None = None) -> list[Segment]:
        """Segments that overlap [start, end). Used to page through long recordings."""
        end = self.duration if end is None else end
        return [s for s in self.segments if s.end > start and s.start < end]

    def render(self, segments: list[Segment] | None = None) -> str:
        """Human/LLM-friendly lines: `[01:23] Speaker 1: text`.

        Timestamps go first on every line so a model quoting the transcript
        naturally quotes *where* something was said, too.
        """
        lines = []
        for s in self.segments if segments is None else segments:
            who = f"{s.speaker}: " if s.speaker else ""
            lines.append(f"[{fmt_ts(s.start)}] {who}{s.text.strip()}")
        return "\n".join(lines)


def fmt_ts(seconds: float) -> str:
    """75.4 -> '01:15'; 3725 -> '1:02:05'. Hours only appear when needed."""
    seconds = int(round(max(0.0, seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def audio_id_for(path: Path) -> str:
    """Hash the file's bytes, not its name: the same recording uploaded twice
    (even under different names) maps to the same cached transcript."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MB at a time: big files never sit in RAM
            digest.update(chunk)
    return digest.hexdigest()[:12]


# --- The store: one JSON file per transcript ---------------------------------
# Deliberately boring. A database would add nothing for a single-user tool, and
# JSON files are easy to inspect while learning (`cat data/transcripts/*.json`).


def _path(audio_id: str) -> Path:
    # audio_id comes from the client, so never let it pick an arbitrary path:
    # only hex characters survive, which also blocks "../" tricks.
    safe = "".join(c for c in audio_id if c in "0123456789abcdef")
    if not safe:
        raise KeyError(audio_id)
    return settings.transcripts_dir / f"{safe}.json"


def save(t: Transcript) -> None:
    settings.transcripts_dir.mkdir(parents=True, exist_ok=True)
    _path(t.audio_id).write_text(t.model_dump_json(indent=1))


def load(audio_id: str) -> Transcript:
    """Raises KeyError with a helpful message if we've never seen this id."""
    p = _path(audio_id)
    if not p.exists():
        raise KeyError(f"No transcript with audio_id {audio_id!r}. Call transcribe_audio first, "
                       f"or list_recordings to see what exists.")
    return Transcript.model_validate_json(p.read_text())


def exists(audio_id: str) -> bool:
    try:
        return _path(audio_id).exists()
    except KeyError:
        return False


def list_all() -> list[Transcript]:
    if not settings.transcripts_dir.exists():
        return []
    items = []
    for p in sorted(settings.transcripts_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            items.append(Transcript.model_validate_json(p.read_text()))
        except (ValueError, json.JSONDecodeError):
            continue  # a half-written or hand-edited file shouldn't break the listing
    return items
