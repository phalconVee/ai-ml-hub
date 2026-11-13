"""Every knob in one place.

Context engineering is mostly a set of budgets and weights: how much text may
enter the window, how much each source is trusted, how similar two passages
may be before one is thrown away. They all live here so you can tune the
behaviour without hunting through the code.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)  # frozen: read everywhere, changed nowhere at runtime
class Settings:
    # ── The model ────────────────────────────────────────────────────────────
    model: str = "claude-opus-5-5"   # current default Opus
    # Opus 5.5 always thinks; `effort` is the dial. The planner and the judge
    # are small structured tasks, so they run at "low". The final answer is
    # where quality matters, so it gets "medium".
    effort_small: str = "low"
    effort_answer: str = "medium"
    max_tokens: int = 16000          # non-streaming ceiling: thinking + output
    # Server-side refusal fallback (beta): if a safety classifier declines,
    # the API re-runs the request on a fallback model inside the same call.
    fallback_beta: str = "server-side-fallback-2026-07-01"

    # ── Gathering: how much each source may bring back ───────────────────────
    doc_chunk_words: int = 160       # chunk size for uploaded documents
    doc_overlap_words: int = 30      # overlap so a sentence cut in half survives
    per_source_candidates: int = 8   # most candidates any one source may offer
    max_web_searches: int = 3        # cap on server-side searches per question
    arxiv_results: int = 6           # papers fetched from the arXiv API
    memory_turns_kept: int = 50      # oldest memory turns are forgotten after this

    # ── Ranking: what earns a place in the window (ranker.py) ────────────────
    # Prior trust per source, multiplied into the score. Your own documents
    # are what you asked about, so they start ahead; memory is helpful but
    # second-hand (it's what *we* said before), so it starts behind.
    source_weights: dict = field(default_factory=lambda: {
        "doc": 1.0, "paper": 0.9, "web": 0.85, "memory": 0.7,
    })
    min_score: float = 0.15          # below this a candidate is "off-topic"
    dedupe_threshold: float = 0.6    # Jaccard overlap above this = near-duplicate
    max_per_source: int = 4          # diversity: no single source may hog the window
    token_budget: int = 2500         # the context window we allow ourselves
    judge_weight: float = 0.5        # how much the optional LLM judge moves a score


SETTINGS = Settings()
