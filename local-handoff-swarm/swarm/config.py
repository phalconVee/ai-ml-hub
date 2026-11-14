"""Every knob in one place, read from environment variables (or a .env file).

Nothing else in the package hard-codes a model name, a URL or a loop limit.
The defaults are chosen so the app runs with **no key at all**: Ollama on
your own machine and keyless DuckDuckGo search.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Pull KEY=VALUE pairs from ./.env into os.environ (does nothing if there is no
# .env). Variables already set in the shell win over the file.
load_dotenv()


def _env(name: str, default: str) -> str:
    """os.environ.get, but an empty value (`FOO=`) also falls back to the default."""
    return os.environ.get(name) or default


@dataclass(frozen=True)  # frozen: settings are read, never changed at runtime
class Settings:
    # ── Which brain? ─────────────────────────────────────────────────────────
    # "ollama" (default, local, free) or "claude" (needs ANTHROPIC_API_KEY).
    backend: str = _env("SWARM_BACKEND", "ollama")

    # ── Ollama (local) ───────────────────────────────────────────────────────
    # Any Ollama model that supports *tool calling* works. qwen3:4b is a good
    # size/quality trade-off on a laptop; llama3.2:3b and qwen3:8b also work.
    ollama_model: str = _env("OLLAMA_MODEL", "qwen3:4b")
    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")
    # qwen3-style models "think" before answering. That is slow on a laptop and
    # adds nothing to routing decisions, so it is off unless you turn it on.
    ollama_think: bool = _env("OLLAMA_THINK", "false").lower() == "true"

    # ── Claude (optional) ────────────────────────────────────────────────────
    # (Prefixed SWARM_ so they can't collide with variables other tools set,
    #  e.g. Claude Code exports CLAUDE_EFFORT into every shell it starts.)
    claude_model: str = _env("SWARM_CLAUDE_MODEL", "claude-opus-5-5")
    # Opus 5.5 always thinks; `effort` is the dial. Routing and short writing
    # tasks don't need deep thought, so "low" keeps each hop fast and cheap.
    claude_effort: str = _env("SWARM_CLAUDE_EFFORT", "low")
    claude_max_tokens: int = int(_env("SWARM_CLAUDE_MAX_TOKENS", "8000"))

    # ── The orchestration loop (loop.py) ─────────────────────────────────────
    # Hard cap on model calls per user message. Small local models can get
    # stuck handing a task back and forth; this guarantees the loop ends.
    max_steps: int = int(_env("SWARM_MAX_STEPS", "12"))

    # ── Web search (tools.py) ────────────────────────────────────────────────
    search_results: int = int(_env("SEARCH_RESULTS", "5"))


SETTINGS = Settings()
