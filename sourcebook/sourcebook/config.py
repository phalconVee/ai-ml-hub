"""Every knob in one place.

Values come from environment variables (a `.env` file is loaded if present),
so the Streamlit app, the tests and a Python shell all read the same settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Read `.env` from the current folder (or a parent). Real environment
# variables win over the file, so a shell export can override it.
load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


@dataclass(frozen=True)
class Settings:
    # --- The LLM --------------------------------------------------------
    # One model for everything: answers, follow-up rewriting, memory
    # summaries and the podcast script. Opus 5.5 is the current default.
    model: str = field(default_factory=lambda: _env("SOURCEBOOK_MODEL", "claude-opus-5-5"))
    # Thinking depth for *answers*. On Opus 5.5 thinking is always on and
    # effort is the dial (API default "medium"; we set it explicitly).
    effort: str = field(default_factory=lambda: _env("SOURCEBOOK_EFFORT", "medium"))
    # The small helper calls (rewrite a follow-up, summarise old turns) are
    # simple, so they run at low effort to stay fast and cheap.
    helper_effort: str = "low"
    max_tokens: int = 16_000

    # --- Chunking and retrieval -----------------------------------------
    # Target passage size in words. Small enough that a citation points at
    # one idea, big enough to carry its context.
    passage_words: int = 120
    # Sentences repeated at the start of the next passage, so a thought cut
    # at a boundary still appears whole in one of them.
    overlap_sentences: int = 1
    # How many passages go to Claude with each question.
    top_k: int = 8
    # Reciprocal Rank Fusion constant (the "60" from the original RRF paper).
    rrf_k: int = 60
    # Local embedding model for the optional dense retriever (fastembed).
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # --- Memory -----------------------------------------------------------
    # The last N question/answer pairs are replayed verbatim; older ones are
    # folded into a running summary so long chats don't grow without bound.
    memory_turns: int = 6

    # --- Audio overview ---------------------------------------------------
    # Words of source text given to the script writer, per source and total.
    podcast_words_per_source: int = 2_500
    podcast_words_total: int = 12_000
    # "auto" picks ElevenLabs if a key is set, else a local system voice.
    tts: str = field(default_factory=lambda: _env("SOURCEBOOK_TTS", "auto"))

    # --- Storage ------------------------------------------------------------
    # Notebooks (sources, chat memory, audio) are saved under this folder.
    home: Path = field(default_factory=lambda: Path(_env("SOURCEBOOK_HOME", ".sourcebook")))


SETTINGS = Settings()
