"""Every knob in one place.

Change a number here and the whole app follows. Nothing else in the package
hard-codes a model name, a chunk size or a loop limit.
"""

from dataclasses import dataclass


@dataclass(frozen=True)  # frozen: settings are read, never mutated at runtime
class Settings:
    # ── The model ────────────────────────────────────────────────────────────
    # Claude Opus 5.5 is the current default Opus. It always thinks before it
    # answers; `effort` is the only dial for how hard (low → max).
    model: str = "claude-opus-5-5"
    # Opus 5.5 defaults to "medium" already, but we say it out loud so the
    # reader knows the dial exists. Raise to "high" for gnarly documents.
    effort: str = "medium"
    max_tokens: int = 16000  # room for thinking + answer on non-streaming calls

    # Server-side refusal fallback: if a safety classifier declines a request,
    # the API re-runs it on a fallback model inside the same call. "default"
    # lets Anthropic pick the fallback by refusal category.
    fallback_beta: str = "server-side-fallback-2026-07-01"

    # ── Chunking (documents.py) ──────────────────────────────────────────────
    chunk_words: int = 180    # ~ one meaty paragraph per chunk
    overlap_words: int = 40   # so a sentence cut at a boundary survives in the next chunk

    # ── Retrieval (bm25.py, librarian.py) ────────────────────────────────────
    top_k: int = 5            # passages returned per search
    max_librarian_turns: int = 4   # tool rounds before the librarian must decide

    # ── Web fallback (scout.py) ──────────────────────────────────────────────
    max_web_searches: int = 5      # cap on searches the scout may run per question
    max_pause_resumes: int = 3     # cap on `pause_turn` continuations


SETTINGS = Settings()
