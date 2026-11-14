"""Reading memory: which stored facts deserve a place in this prompt?

Showing the model *every* fact would work for ten memories and fail for a
thousand (cost, noise, and the model latching onto irrelevant details). So
each fact gets a score from four signals, each in [0, 1]:

    text        BM25 keyword match between the query and the fact
    entity      the fact is about a person/place/thing named in the query
    importance  what the extractor said when it stored the fact (1..5)
    recency     how recently the fact was learned or re-confirmed

    score = w_text·text + w_entity·entity + w_importance·importance + w_recency·recency

A fact must actually *match* (text or entity > 0) and clear `min_score`,
except "core" facts (importance 5) which are always offered: a peanut
allergy is relevant to "what should I cook?" even though no word matches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from .config import SETTINGS
from .store import Fact, MemoryStore
from .textsearch import BM25


@dataclass
class Recalled:
    """A fact plus its score and the per-signal breakdown (shown in the UI)."""
    fact: Fact
    score: float
    why: dict = field(default_factory=dict)


def search_text(f: Fact) -> str:
    """What BM25 sees for a fact: the sentence plus the edge in words.

    Adding "lives in Lisbon" next to "User moved to Lisbon last spring"
    means a query using either phrasing finds it.
    """
    return f"{f.statement} {f.subject} {f.predicate.replace('_', ' ')} {f.object}"


def mentioned_entities(store: MemoryStore, profile_id: int, text: str,
                       hints: list[str] = ()) -> set[int]:
    """Ids of entities named in `text` (whole-word match) or in `hints`.

    The user node is skipped: it is the subject of most facts, so matching it
    would make every fact "entity-relevant" and the signal would mean nothing.
    """
    lowered = text.lower()
    hint_keys = {" ".join(h.lower().split()) for h in hints}
    found = set()
    for e in store.entities(profile_id):
        if e["kind"] == "user":
            continue
        if e["key"] in hint_keys or re.search(rf"\b{re.escape(e['key'])}\b", lowered):
            found.add(e["id"])
    return found


def _age_days(iso: str, now: datetime) -> float:
    return max(0.0, (now - datetime.fromisoformat(iso)).total_seconds() / 86400)


def retrieve(store: MemoryStore, profile_id: int, query: str, *,
             entity_hints: list[str] = (), include_history: bool = False,
             top_k: int = SETTINGS.top_k, min_score: float = SETTINGS.min_score) -> list[Recalled]:
    """Rank this profile's facts against `query`; return the best `top_k`.

    `include_history=True` also searches superseded facts, for questions
    like "where did I live before?". They are scored a little lower so a
    current fact beats its own outdated version.
    """
    facts = store.facts(profile_id, include_superseded=include_history)
    if not facts:
        return []
    s = SETTINGS
    now = store.clock()
    text_scores = BM25([search_text(f) for f in facts]).normalized(query)
    hit_entities = mentioned_entities(store, profile_id, query, entity_hints)

    out: list[Recalled] = []
    for f, text in zip(facts, text_scores):
        entity = 1.0 if (f.subject_id in hit_entities or f.object_id in hit_entities) else 0.0
        importance = (f.importance - 1) / 4                       # 1..5 → 0..1
        # Exponential decay: a fact confirmed `half_life` days ago counts half.
        recency = 0.5 ** (_age_days(f.last_confirmed_at, now) / s.recency_half_life_days)
        score = (s.w_text * text + s.w_entity * entity
                 + s.w_importance * importance + s.w_recency * recency)
        if not f.active:
            score *= 0.8                                          # history ranks below the present
        core = f.importance >= s.core_importance and f.active
        matched = text > 0 or entity > 0
        if core or (matched and score >= min_score):
            out.append(Recalled(f, round(score, 4), {
                "text": round(text, 3), "entity": entity, "importance": round(importance, 2),
                "recency": round(recency, 3), "core": core,
            }))
    out.sort(key=lambda r: r.score, reverse=True)
    return out[:top_k]
