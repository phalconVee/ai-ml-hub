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
    # --- The LLM ------------------------------------------------------------
    # Opus 5.5 is the current default model. Thinking is always on for it;
    # `effort` is the dial (its API default is "medium"; we set it explicitly).
    model: str = field(default_factory=lambda: _env("VAULT_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: _env("VAULT_EFFORT", "medium"))
    max_tokens: int = 16_000
    # Safety valve for the tool loop: after this many model calls in one turn
    # we stop and return whatever the agent has, instead of looping forever.
    max_tool_rounds: int = 8

    # --- Memory kind 1: the document knowledge base ---------------------------
    # Chunk size and overlap in words. ~180 words is roughly one paragraph or
    # two: big enough to carry context, small enough to stay on one topic.
    chunk_words: int = 180
    chunk_overlap: int = 40
    # How many chunks one `search_knowledge_base` call returns.
    kb_top_k: int = 5

    # --- Memory kind 2: conversation history ----------------------------------
    # How many past messages of the current thread are replayed to Claude
    # (user + assistant each count as one). Older ones stay in the database
    # and in the UI, they just stop being sent.
    history_messages: int = 20

    # --- Memory kind 3: agent-written notes -----------------------------------
    # How many notes one `recall` call returns.
    recall_top_k: int = 5
    # Cosine *distance* under which a new note counts as a duplicate of an
    # existing one (0 = identical direction). Stops "remember" spam. Kept
    # tight on purpose: with bge-small, a paraphrase sits near 0.03 but
    # "deadline is March 3" vs "March 10" is only 0.07 apart, and an updated
    # fact must not be swallowed as a duplicate. (Outdated notes are removed
    # with the `forget` tool instead.)
    duplicate_distance: float = 0.04

    # --- Embeddings ---------------------------------------------------------
    # "fastembed": a real local embedding model (ONNX, no key, ~70 MB once).
    # "hash": a tiny keyword-hashing embedder; no download, used by the tests.
    embedder: str = field(default_factory=lambda: _env("VAULT_EMBEDDER", "fastembed"))
    embed_model: str = field(default_factory=lambda: _env("VAULT_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))

    # --- Storage --------------------------------------------------------------
    # One SQLite file holds all three kinds of memory.
    db_path: Path = field(default_factory=lambda: Path(_env("VAULT_DB", ".vault/vault.db")))


SETTINGS = Settings()
