"""Stage 3, the Ranker: the heart of context engineering.

Gathering is easy; the hard part is deciding what goes into a limited window.
Every candidate passes the same gauntlet, and every decision is written into
`item.reason` so the UI can show it:

  1. Relevance   one BM25 index over *all* candidates, so a web quote and a
                 PDF chunk are scored on the same scale; normalised to 0..1.
  2. Judge       (optional) an LLM grades each candidate 0-10; blended in.
                 BM25 matches words, the judge matches meaning.
  3. Trust       multiply by a per-source weight (docs > papers > web > memory).
  4. Threshold   drop anything below `min_score`: off-topic.
  5. Dedupe      drop near-copies of something already kept (Jaccard).
  6. Diversity   cap each source so one source can't fill the window.
  7. Budget      greedily pack by score until the token budget is spent;
                 a big item that doesn't fit is skipped and smaller ones may
                 still fill the gap.

Survivors get citation ids S1, S2, … in score order.
"""

from __future__ import annotations

from typing import Any

from .config import SETTINGS
from .items import ContextItem
from .llm import ask, json_of
from .textsearch import BM25, jaccard

# ── Optional LLM judge ─────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You grade how useful each numbered passage is for answering the \
question: 10 = directly answers it, 5 = related background, 0 = irrelevant. Judge \
usefulness for *this* question only."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "grades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"i": {"type": "integer"}, "score": {"type": "integer"}},
                "required": ["i", "score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["grades"],
    "additionalProperties": False,
}


def llm_judge(question: str, items: list[ContextItem], client: Any) -> None:
    """Fill `item.judge` (0..1) in place. One call grades every candidate."""
    listing = "\n\n".join(f"[{i}] ({it.source}) {it.title}\n{it.text[:600]}"
                          for i, it in enumerate(items))
    resp = ask(client, system=JUDGE_SYSTEM, effort=SETTINGS.effort_small, schema=JUDGE_SCHEMA,
               messages=[{"role": "user", "content": f"Question: {question}\n\nPassages:\n\n{listing}"}])
    for g in json_of(resp).get("grades", []):
        if 0 <= g["i"] < len(items):
            items[g["i"]].judge = max(0, min(10, g["score"])) / 10


# ── The gauntlet ───────────────────────────────────────────────────────────────

def score_items(question: str, items: list[ContextItem]) -> None:
    """Steps 1-3: relevance, judge blend, source trust. Writes item.score."""
    raw = BM25([it.text for it in items]).scores(question)
    top = max(raw, default=0.0) or 1.0              # avoid dividing by zero
    w = SETTINGS.judge_weight
    for it, r in zip(items, raw):
        it.relevance = r / top
        base = it.relevance if it.judge is None else (1 - w) * it.relevance + w * it.judge
        it.score = base * SETTINGS.source_weights.get(it.source, 0.5)


def select(items: list[ContextItem], budget: int | None = None) -> list[ContextItem]:
    """Steps 4-7. Marks every item kept/dropped with a reason; returns the kept ones."""
    budget = budget or SETTINGS.token_budget
    used = 0
    kept: list[ContextItem] = []
    per_source: dict[str, int] = {}

    for it in sorted(items, key=lambda x: x.score, reverse=True):
        if it.score < SETTINGS.min_score:
            it.reason = f"off-topic: score {it.score:.2f} < {SETTINGS.min_score}"
            continue
        twin = next((k for k in kept if jaccard(it.text, k.text) > SETTINGS.dedupe_threshold), None)
        if twin:
            it.reason = f"near-duplicate of {twin.id} ({twin.title[:40]})"
            continue
        if per_source.get(it.source, 0) >= SETTINGS.max_per_source:
            it.reason = f"source cap: already {SETTINGS.max_per_source} {it.source} items"
            continue
        if used + it.tokens > budget:
            it.reason = f"over budget: needs {it.tokens} tok, {budget - used} left"
            continue
        # It's in.
        used += it.tokens
        per_source[it.source] = per_source.get(it.source, 0) + 1
        it.kept = True
        it.id = f"S{len(kept) + 1}"
        judge = "" if it.judge is None else f", judge {it.judge:.1f}"
        it.reason = (f"kept #{len(kept) + 1}: relevance {it.relevance:.2f}{judge} × "
                     f"{it.source} weight {SETTINGS.source_weights.get(it.source, 0.5)} "
                     f"= {it.score:.2f}; {it.tokens} tok ({used}/{budget} used)")
        kept.append(it)
    return kept


def rank(question: str, items: list[ContextItem], *, client: Any = None,
         use_judge: bool = False, budget: int | None = None) -> list[ContextItem]:
    """Run the whole gauntlet. Returns the kept items in citation order."""
    if not items:
        return []
    if use_judge and client is not None:
        try:
            llm_judge(question, items, client)
        except Exception:
            pass  # the judge is an enhancement; BM25 alone still ranks
    score_items(question, items)
    return select(items, budget)
