"""Every knob in one place.

Values come from environment variables (a `.env` file is loaded if present),
so the Streamlit app, the tests and a notebook all read the same settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Read `.env` from the current folder (or any parent). Real environment
# variables win over the file, so CI or a shell export can override it.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    # --- The LLM --------------------------------------------------------
    # The agent's brain. Claude Opus 5.5 is the current default Opus model.
    model: str = field(default_factory=lambda: os.getenv("GROUNDTRUTH_MODEL", "claude-opus-5-5"))
    # Model used to *describe figures* while parsing (a vision call per figure).
    figure_model: str = field(default_factory=lambda: os.getenv("GROUNDTRUTH_FIGURE_MODEL", "claude-opus-5-5"))
    # How hard Claude thinks. On Opus 5.5 thinking is always on and `effort`
    # is the dial; the API default there is "medium", we set it explicitly.
    effort: str = field(default_factory=lambda: os.getenv("GROUNDTRUTH_EFFORT", "medium"))
    # Ceiling on one response's length. Generous so answers never get cut off.
    max_tokens: int = 16_000
    # Safety valve: the agent may call the search tool at most this many rounds.
    max_tool_rounds: int = 6

    # --- Parsing --------------------------------------------------------
    # Render figures at this resolution before sending them to the vision model.
    figure_dpi: int = 144
    # Ignore images/drawings smaller than this (in PDF points²): logos, bullets, rules.
    min_figure_area: float = 5_000.0
    # Hard cap on vision calls per document, so a 300-page scan can't surprise your bill.
    max_figures_per_doc: int = 25

    # --- Chunking and retrieval -----------------------------------------
    # Target size of a text chunk, in words. Tables and figures stay whole.
    chunk_words: int = 180
    # Words repeated between neighbouring chunks, so a sentence cut at a
    # boundary still appears whole in one of them.
    chunk_overlap: int = 40
    # How many passages one search returns to Claude.
    top_k: int = 6


SETTINGS = Settings()
