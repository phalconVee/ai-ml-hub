"""Writing memory: after each exchange, pull out the facts worth keeping.

People don't remember conversations word for word; they remember what they
learned: "Ana is vegetarian", "her sister Maya lives in Porto", "she's
training for a marathon in April". This step does the same. One structured
Claude call reads the latest exchange and returns facts as graph edges:

    (subject) —predicate→ (object)    + a sentence, category, importance,
                                        exclusive?, when?, replaces which ids?

To decide whether something *changed*, the model must see what we already
believe. So before the call we retrieve the existing facts most related to
the exchange and show them with their ids. The model can then say "this new
fact replaces fact 12". The store (store.add_fact) does the bookkeeping.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .config import SETTINGS
from .llm import ask, json_of
from .retriever import retrieve
from .store import MemoryStore, NewFact, WriteResult

CATEGORIES = ["preference", "person", "event", "goal", "biographical", "health", "work", "other"]
KINDS = ["user", "person", "place", "organization", "pet", "thing", "activity"]

_nullable_str = {"anyOf": [{"type": "string"}, {"type": "null"}]}

SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "subject_kind": {"type": "string", "enum": KINDS},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "object_is_entity": {"type": "boolean"},
                    "object_kind": {"type": "string", "enum": KINDS},
                    "statement": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "importance": {"type": "integer"},
                    "exclusive": {"type": "boolean"},
                    "occurred_at": _nullable_str,
                    "replaces": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["subject", "subject_kind", "predicate", "object", "object_is_entity",
                             "object_kind", "statement", "category", "importance", "exclusive",
                             "occurred_at", "replaces"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["facts"],
    "additionalProperties": False,
}

SYSTEM = """You are the memory of a personal assistant. Read the latest exchange and extract
durable facts about the USER and their world that will still be useful in future conversations.

Extract: preferences and dislikes, people in their life and their relationships, places, jobs,
health constraints (allergies, diet), goals and plans, notable events with dates, pets, routines.
Do NOT extract: small talk, the assistant's own suggestions or opinions, one-off requests
("translate this"), or anything the user did not actually state or clearly imply.
If nothing is worth remembering, return an empty list. Fewer, solid facts beat many weak ones.

For each fact:
- subject: "user" for the user, otherwise the entity's name ("Maya", "Acme Corp").
  Reuse names from KNOWN ENTITIES when they refer to the same thing.
- predicate: short snake_case relation (lives_in, works_at, likes, dislikes, allergic_to,
  sister_of, has_pet, goal, plans_to_visit, birthday_is ...).
- object: the value ("Lisbon", "peanuts", "run a marathon"). object_is_entity=true when the
  object is itself a person/place/org/pet that facts could hang off.
- statement: one self-contained sentence in the third person, e.g. "The user's sister Maya lives in Porto."
- importance 1-5: 5 = must never be forgotten (allergies, name, hard constraints),
  4 = core identity/relationships, 3 = solid preference, 2 = minor, 1 = trivia.
- exclusive: true if the subject can have only ONE value of this predicate at a time
  (lives_in, works_at, job_title, age, relationship_status). false for likes, has_pet, sibling ...
- occurred_at: for events/plans, an ISO date (YYYY-MM-DD) resolved against TODAY, else null.
- replaces: ids from CURRENT MEMORIES that this fact makes untrue (the user moved, changed
  their mind, quit, broke up, a plan was cancelled). Empty list if nothing is contradicted.
  If the user just repeats a known fact, output it unchanged with replaces=[]."""


def related_facts(store: MemoryStore, profile_id: int, text: str) -> list:
    """Current memories the extractor should see, so it can spot contradictions.

    Keyword search alone is not enough here: "I moved to Lisbon" shares no
    word with "The user lives in Berlin", yet it contradicts it. So we take
    the keyword matches first, then top up to the cap with the remaining
    active facts, exclusive ones first (they are the ones that get replaced),
    then the most important. For a small memory that is simply "everything".
    """
    cap = SETTINGS.related_for_extraction
    picked = [r.fact for r in retrieve(store, profile_id, text, top_k=cap, min_score=0.0)]
    seen = {f.id for f in picked}
    rest = sorted((f for f in store.facts(profile_id) if f.id not in seen),
                  key=lambda f: (f.exclusive, f.importance, f.last_confirmed_at), reverse=True)
    return (picked + rest)[:cap]


def _related_block(store: MemoryStore, profile_id: int, text: str) -> str:
    facts = related_facts(store, profile_id, text)
    if not facts:
        return "(none yet)"
    return "\n".join(f"[{f.id}] {f.statement}  "
                     f"(subject={f.subject}, predicate={f.predicate}, object={f.object})"
                     for f in facts)


def build_prompt(store: MemoryStore, profile_id: int, user_text: str, reply: str,
                 today: datetime) -> str:
    names = [e["name"] for e in store.entities(profile_id) if e["kind"] != "user"]
    return (f"TODAY: {today.date().isoformat()}\n\n"
            f"KNOWN ENTITIES: {', '.join(names) or '(none yet)'}\n\n"
            f"CURRENT MEMORIES (id, statement):\n{_related_block(store, profile_id, user_text)}\n\n"
            f"LATEST EXCHANGE\nUser: {user_text}\nAssistant: {reply}")


def parse_facts(data: dict) -> list[NewFact]:
    """Turn the JSON into NewFact objects, skipping anything malformed."""
    out = []
    for f in data.get("facts", []) if isinstance(data, dict) else []:
        try:
            if not (f["subject"].strip() and f["object"].strip() and f["statement"].strip()):
                continue
            out.append(NewFact(
                subject=f["subject"], predicate=f["predicate"], object=f["object"],
                statement=f["statement"], category=f.get("category", "other"),
                importance=int(f.get("importance", 3)), exclusive=bool(f.get("exclusive")),
                object_is_entity=bool(f.get("object_is_entity")),
                subject_kind=f.get("subject_kind", "thing"), object_kind=f.get("object_kind", "thing"),
                occurred_at=f.get("occurred_at"),
                replaces=[int(i) for i in f.get("replaces", []) if isinstance(i, int)],
            ))
        except (KeyError, TypeError, ValueError, AttributeError):
            continue  # one bad item shouldn't lose the others
    return out


def extract_and_store(store: MemoryStore, client: Any, profile_id: int, *, user_text: str,
                      reply: str, session_id: int | None = None,
                      message_id: int | None = None) -> list[WriteResult]:
    """The memory step: ask Claude for facts, then write each one to the store.

    Returns what happened per fact (added / confirmed, and what it superseded)
    so the UI can show "learned this turn". Errors are swallowed on purpose:
    a failed memory write must never break the conversation.
    """
    prompt = build_prompt(store, profile_id, user_text, reply, store.clock())
    try:
        data = json_of(ask(client, system=SYSTEM, effort=SETTINGS.effort_memory,
                           messages=[{"role": "user", "content": prompt}], schema=SCHEMA))
    except Exception:
        return []
    return [store.add_fact(profile_id, nf, session_id=session_id, message_id=message_id)
            for nf in parse_facts(data)]
