"""Before reading memory: turn the user's message into a good search.

The raw message is often a poor query. "What should I make for dinner
tonight?" shares no words with "User is vegetarian" or "User's partner Sam
hates mushrooms". A small, cheap Claude call rewrites the message into
keywords, names the entities involved, and says whether the user is asking
about the past (so superseded facts should be searched too).

If the call fails or is switched off, we fall back to the raw message:
memory still works, just with keyword matching only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import SETTINGS
from .llm import ask, json_of

SCHEMA = {
    "type": "object",
    "properties": {
        "keywords": {"type": "string",
                     "description": "Space-separated search words, including synonyms and related topics"},
        "entities": {"type": "array", "items": {"type": "string"},
                     "description": "Names of people, places, pets, orgs or things the message is about"},
        "about_the_past": {"type": "boolean",
                           "description": "True if the user asks how things used to be or what changed"},
    },
    "required": ["keywords", "entities", "about_the_past"],
    "additionalProperties": False,
}

SYSTEM = """You write search queries for a personal memory store. The store holds short facts
about one user: preferences, people in their life, events, goals, biography.

Given the latest message (and a little conversation context), output:
- keywords: words that would appear in facts useful for answering. Expand the topic:
  "dinner ideas" -> "food diet cooking allergy vegetarian dislike restaurant".
  Resolve pronouns from context ("she" -> the person's name).
- entities: names mentioned or clearly referred to.
- about_the_past: true only if the user asks about earlier states ("before", "used to", "what changed")."""


@dataclass
class RecallPlan:
    query: str                       # what BM25 searches with
    entities: list[str] = field(default_factory=list)
    include_history: bool = False
    by_llm: bool = False             # False = the plain fallback was used


def plain_plan(message: str) -> RecallPlan:
    """No LLM: search with the message itself."""
    return RecallPlan(query=message)


def plan_recall(client: Any, message: str, recent: list[dict]) -> RecallPlan:
    """One low-effort structured call. Never raises: falls back to `plain_plan`."""
    context = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in recent[-4:])
    prompt = f"Recent conversation:\n{context or '(none)'}\n\nLatest user message:\n{message}"
    try:
        data = json_of(ask(client, system=SYSTEM, effort=SETTINGS.effort_memory,
                           messages=[{"role": "user", "content": prompt}], schema=SCHEMA))
    except Exception:  # network/API trouble must not break the chat
        return plain_plan(message)
    if not data:
        return plain_plan(message)
    return RecallPlan(
        # Keep the original message in the query: the keywords add recall,
        # the message keeps precision.
        query=f"{message} {data.get('keywords', '')}",
        entities=[e for e in data.get("entities", []) if isinstance(e, str)],
        include_history=bool(data.get("about_the_past")),
        by_llm=True,
    )
