"""Every knob in one place.

Change a value here and the whole crew follows. No other module hard-codes a
model name, a chunk size or a loop limit.
"""

from dataclasses import dataclass


@dataclass(frozen=True)  # frozen: settings are read at runtime, never changed
class Settings:
    # ── The model ────────────────────────────────────────────────────────────
    # Claude Opus 5.5 is the current default Opus. Thinking is always on;
    # `effort` is the dial for how hard it thinks. Its default is "medium",
    # but we say so explicitly so the reader knows the dial exists.
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    max_tokens: int = 16000  # room for thinking + answer on a non-streaming call

    # If a safety classifier declines a request, the API can re-run it on a
    # fallback model inside the same call. "default" lets Anthropic choose
    # the fallback by refusal category. Needs this beta header.
    fallback_beta: str = "server-side-fallback-2026-07-01"

    # ── Ingest (ingest.py) ───────────────────────────────────────────────────
    max_clause_words: int = 220   # a clause longer than this is split into windows
    window_overlap: int = 40      # words shared by neighbouring windows

    # ── Retrieval (search.py, clause_finder.py) ──────────────────────────────
    top_k: int = 5                # passages per search
    max_finder_turns: int = 5     # tool rounds before the Clause Finder must report

    # ── Web research (researcher.py) ─────────────────────────────────────────
    max_web_searches: int = 4     # cap on searches per question (each one is billed)
    max_pause_resumes: int = 3    # cap on `pause_turn` continuations


SETTINGS = Settings()


# The notice is a constant, added by *our code* to every answer. We never
# rely on the model to remember it: a disclaimer that is there 99% of the
# time is not a disclaimer.
NOT_LEGAL_ADVICE = (
    "**Not legal advice.** counsel-copilot is an AI research assistant, not a "
    "lawyer. It can misread documents and the law differs by jurisdiction. "
    "Check the cited clauses yourself and talk to a qualified lawyer before "
    "you act on anything here."
)
