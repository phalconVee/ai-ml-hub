"""Every knob in one place.

Memory is mostly a handful of numbers: how many memories may enter the prompt,
how much a recent memory beats an old one, how sure we must be that a memory
is relevant before we show it to the model. They all live here.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)  # frozen: read everywhere, changed nowhere at runtime
class Settings:
    # ── The model ────────────────────────────────────────────────────────────
    model: str = "claude-opus-5-5"   # current default Opus
    # Opus 5.5 always thinks; `effort` is the dial. The two memory calls
    # (recall planning and fact extraction) are small structured tasks, so
    # they run at "low". The reply the user reads gets "medium".
    effort_memory: str = "low"
    effort_answer: str = "medium"
    max_tokens: int = 16000          # non-streaming ceiling: thinking + output
    # Server-side refusal fallback (beta): if a safety classifier declines,
    # the API re-runs the request on a fallback model inside the same call.
    fallback_beta: str = "server-side-fallback-2026-07-01"

    # ── Where memory lives ───────────────────────────────────────────────────
    # One SQLite file holds every profile, session, message and fact.
    db_path: Path = Path(".recall") / "memory.db"

    # ── Short-term memory: the current conversation ──────────────────────────
    # The last N messages of *this* session go to the model verbatim.
    # Everything older, and everything from other sessions, reaches the model
    # only through long-term memory (retrieved facts).
    recent_messages: int = 12

    # ── Long-term memory: retrieval (retriever.py) ───────────────────────────
    top_k: int = 8                   # most facts that may enter one prompt
    min_score: float = 0.12          # below this a fact is "not relevant"
    # The final score is a weighted sum of four signals, each in [0, 1].
    w_text: float = 0.55             # BM25 keyword match against the query
    w_entity: float = 0.25           # fact mentions an entity named in the query
    w_importance: float = 0.12       # how much the extractor said it matters
    w_recency: float = 0.08          # recently learned or confirmed facts win ties
    recency_half_life_days: float = 30.0
    # Facts at or above this importance are "core" (your name, a strong
    # dietary rule, ...) and are offered on every turn, matched or not.
    core_importance: int = 5

    # ── Writing memory (extractor.py) ────────────────────────────────────────
    # How many existing facts we show the extractor so it can spot a
    # contradiction ("I moved to Lisbon" vs. "lives in Berlin").
    related_for_extraction: int = 15


SETTINGS = Settings()
