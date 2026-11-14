"""Turning one exchange into graph facts: Claude reads, code merges.

After every turn we show Claude the user's message, the assistant's reply,
and the names already in memory, and ask for {entities, relations} as JSON.
Structured outputs guarantee the shape, so nothing here parses free text.

The split of work is deliberate:
  Claude  understands language: who "she" is, that "I just moved to Berlin"
          means lives_in Berlin *and* that it replaces the old city (exclusive),
          that "I left Acme" means the works_at edge is over (negated).
  code    owns identity and history (graph.merge): de-duplication, aliases,
          weights, retiring facts. An LLM is bad at bookkeeping across a long
          memory and great at reading one paragraph, so each does its job.
"""

from __future__ import annotations

from typing import Any

from .config import SETTINGS
from .llm import ask_json

ENTITY_TYPES = ["person", "place", "organization", "project", "topic", "skill",
                "interest", "event", "object", "other"]

# JSON Schema for structured outputs. Every field is required and nothing
# extra is allowed, so the merge code can index without .get() surprises.
SCHEMA: dict = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": ENTITY_TYPES},
                    "description": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "type", "description", "aliases"],
                "additionalProperties": False,
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "predicate": {"type": "string"},
                    "target": {"type": "string"},
                    "exclusive": {"type": "boolean"},
                    "negated": {"type": "boolean"},
                },
                "required": ["source", "predicate", "target", "exclusive", "negated"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "relations"],
    "additionalProperties": False,
}

# Static, so it's byte-identical every call (a stable prefix the API can cache).
SYSTEM = f"""You maintain the long-term memory of a personal assistant as a knowledge graph.
You will see one exchange (the user's message and the assistant's reply) plus the names
already in memory. Return the entities and relations worth remembering from THIS exchange.

Who is who:
- The person talking is always the entity named exactly "{SETTINGS.user_node}". Never create
  a separate node for "me", "I" or the user's own name. If the user tells you their name,
  add it as an alias of "{SETTINGS.user_node}" (entity name "{SETTINGS.user_node}", aliases ["<their name>"]).
- If a thing is already in memory, use its name EXACTLY as listed (same spelling and case),
  even if the message says it differently ("my sister" → the listed "Ada" if that is her).

What to remember:
- Facts about the user: who they are, people in their life, where they live and work,
  projects, goals, preferences, plans, opinions, recent events.
- The topics the user cares about or asks about (e.g. User -interested_in-> Rust), and at
  most a couple of key facts about a topic when the reply establishes them clearly.
- Only what the user said or plainly confirmed. Not the assistant's guesses, suggestions or
  questions. Not small talk ("hi", "thanks"). An empty result is fine and often correct.

How to write it:
- entity: short canonical name ("Berlin", not "the city of Berlin"); type from the list;
  description = one short line of what we know about it now (may be "").
- predicate: a short snake_case verb phrase from the source's point of view:
  lives_in, works_at, sister_of, likes, dislikes, building, learning, plans_to_visit,
  interested_in, owns, studies, part_of, uses.
- exclusive = true when the source can hold this predicate for only ONE target at a time
  (lives_in, works_at, married_to, current_role); the new target replaces the old one.
- negated = true when the user says the fact is no longer / not true ("I left Acme",
  "I don't like jazz anymore"); give the relation exactly as it was stored.
- Keep it small: at most 8 entities and 10 relations per exchange."""


def extract(client: Any, *, user_text: str, reply_text: str, known: list[str]) -> dict:
    """One Claude call: an exchange in, {entities: [...], relations: [...]} out."""
    # Variable content goes in the user message, never the system prompt.
    known_list = ", ".join(known) if known else "(memory is empty)"
    user = (f"Already in memory: {known_list}\n\n"
            f"<user_message>\n{user_text}\n</user_message>\n\n"
            f"<assistant_reply>\n{reply_text}\n</assistant_reply>")
    return ask_json(client, system=SYSTEM, user=user,
                    effort=SETTINGS.extract_effort, schema=SCHEMA)
