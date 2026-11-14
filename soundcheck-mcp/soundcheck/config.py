"""Every knob in one place.

Read this first. It answers three questions:
  * which speech-to-text engine runs, and with which Whisper model?
  * do the "smart" tools (summary, topics) use Claude or the local fallback?
  * where do transcripts and uploads live on disk?

Everything can be overridden with an environment variable (see `.env.example`).
The defaults need **no API key at all** for transcription, speakers, sentiment
and search; only the Claude chat agent (and, optionally, Claude-quality
summaries) needs ANTHROPIC_API_KEY.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Pull variables from a local `.env` into os.environ. Real environment
# variables win over the file, so `FOO=1 uv run ...` still works.
load_dotenv()

# The project folder (the one holding pyproject.toml). Data paths default to
# folders inside it, so they don't depend on the directory you launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    """Read an env var, treating unset *and* blank as "use the default"."""
    return os.environ.get(name) or default


@dataclass
class Settings:
    # --- Speech to text --------------------------------------------------
    # "auto"       -> local Whisper (free). AssemblyAI is never picked silently.
    # "local"      -> faster-whisper on this machine, no key, no network after
    #                 the one-time model download.
    # "assemblyai" -> AssemblyAI's paid API (needs ASSEMBLYAI_API_KEY). Its one
    #                 advantage here is real, model-based speaker diarization.
    engine: str = field(default_factory=lambda: _env("SOUNDCHECK_ENGINE", "auto"))

    # Whisper size: tiny | base | small | medium | large-v3 | turbo ...
    # "base" (~145 MB) is a good laptop default: fast on CPU, decent accuracy.
    # "small" is noticeably better on accents/jargon at ~3x the time.
    whisper_model: str = field(default_factory=lambda: _env("SOUNDCHECK_WHISPER_MODEL", "base"))
    # "auto" lets CTranslate2 choose (CUDA if present, else CPU). Apple Silicon
    # runs on CPU; int8 keeps it quick and light there.
    whisper_device: str = field(default_factory=lambda: _env("SOUNDCHECK_WHISPER_DEVICE", "auto"))
    whisper_compute_type: str = field(default_factory=lambda: _env("SOUNDCHECK_WHISPER_COMPUTE", "int8"))

    assemblyai_key: str = field(default_factory=lambda: _env("ASSEMBLYAI_API_KEY", ""))

    # --- Analysis --------------------------------------------------------
    # "auto"   -> use Claude for summary/topics when an Anthropic credential is
    #             in the environment, otherwise the local (keyless) fallback.
    # "claude" -> always try Claude (use this if you authenticate with
    #             `ant auth login` rather than an env var).
    # "local"  -> never call an LLM from the server.
    analysis_mode: str = field(default_factory=lambda: _env("SOUNDCHECK_ANALYSIS", "auto"))

    # --- Claude ------------------------------------------------------------
    # The chat agent in the Streamlit app, and the server's summary/topic tools.
    agent_model: str = field(default_factory=lambda: _env("SOUNDCHECK_AGENT_MODEL", "claude-opus-5-5"))
    analysis_model: str = field(default_factory=lambda: _env("SOUNDCHECK_ANALYSIS_MODEL", "claude-opus-5-5"))
    # Chatting about a transcript is mostly lookup + light reasoning: medium.
    # Summaries and topic maps are one-shot extraction: low is plenty.
    agent_effort: str = field(default_factory=lambda: _env("SOUNDCHECK_AGENT_EFFORT", "medium"))
    analysis_effort: str = field(default_factory=lambda: _env("SOUNDCHECK_ANALYSIS_EFFORT", "low"))
    # Hard cap on think -> call tool -> read result rounds per chat message.
    max_agent_turns: int = field(default_factory=lambda: int(_env("SOUNDCHECK_MAX_TURNS", "8")))
    # Tool output is clipped to this before going back to Claude.
    max_tool_chars: int = field(default_factory=lambda: int(_env("SOUNDCHECK_MAX_TOOL_CHARS", "20000")))
    # If a safety classifier declines, the API retries on a fallback model in
    # the same call instead of just stopping.
    fallback_beta: str = "server-side-fallback-2026-07-01"

    # --- Storage -----------------------------------------------------------
    # Transcripts are cached as JSON here, keyed by a hash of the audio bytes,
    # so transcribing the same file twice is free.
    data_dir: Path = field(default_factory=lambda: Path(_env("SOUNDCHECK_DATA_DIR", str(PROJECT_ROOT / "data"))))
    # Refuse anything larger than this (MB): Whisper on CPU takes roughly
    # real-time/5 with "base", so a 2-hour file is a long wait.
    max_upload_mb: int = field(default_factory=lambda: int(_env("SOUNDCHECK_MAX_UPLOAD_MB", "200")))

    @property
    def transcripts_dir(self) -> Path:
        return self.data_dir / "transcripts"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    def claude_available(self) -> bool:
        """Should the server's analysis tools call Claude?"""
        if self.analysis_mode == "local":
            return False
        if self.analysis_mode == "claude":
            return True
        # auto: only when a credential is visibly configured.
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


# File types we accept. Decoding is done by PyAV (bundled FFmpeg), so anything
# FFmpeg reads works; this list just keeps out obvious non-audio.
AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus",
    ".webm", ".aiff", ".aif", ".wma", ".mp4", ".mov", ".mkv",
}

settings = Settings()
