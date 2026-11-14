"""Agent 2, the Scriptwriter: outline in, a two-host conversation out.

The writer only sees the researcher's outline (not the raw page), the two
hosts' names and roles, and a word budget. It answers in JSON: a list of
{"speaker": "A" | "B", "text": "..."} lines. The schema's `enum` means a
third speaker is impossible, so every line has a voice waiting for it.

Writing *for the ear* is different from writing for the page, so most of the
system prompt is about how speech sounds: short sentences, contractions,
reactions, no bullet points read aloud, no stage directions (a TTS engine
would literally say "laughs").
"""

from __future__ import annotations

from typing import Any

from .config import SETTINGS
from .llm import ask_json
from .researcher import Outline
from .script import LINES_SCHEMA, Script

SYSTEM = """You write scripts for a two-host explainer podcast. The script \
will be read aloud by text-to-speech voices, so write for the ear.

Host A is the guide: knows the material, explains with examples.
Host B is the curious co-host: asks what listeners would ask, reacts, \
summarises in plain words, sometimes pushes back.

Rules:
- Use ONLY facts from the outline. Never invent numbers, names or quotes.
- Open with the hook in the first two lines; close with the takeaway and a short sign-off.
- Alternate speakers naturally; short back-and-forth beats long speeches. \
Keep every line under 60 words.
- Sound like people talking: contractions, simple words, short sentences, the \
occasional "right", "wait", "so". Hosts address each other by name now and then.
- No stage directions, sound effects, emojis, markdown, lists or URLs: \
everything you write will be spoken exactly as written.
- Spell out anything a voice could stumble on (e.g. "about 40 percent", "the year 2024")."""

SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "lines": LINES_SCHEMA},
    "required": ["title", "lines"],
    "additionalProperties": False,
}


def target_words(minutes: float) -> int:
    return round(minutes * SETTINGS.words_per_minute)


def write_script(outline: Outline, client: Any, *, names: dict[str, str],
                 minutes: float, style: str = "") -> Script:
    words = target_words(minutes)
    user = (f"Host A is {names['A']}. Host B is {names['B']}.\n"
            f"Length: about {words} words in total (~{minutes:g} minutes spoken).\n"
            + (f"Style note from the producer: {style}\n" if style else "")
            + f"\nOUTLINE:\n{outline.brief()}")
    data = ask_json(client, who="scriptwriter", system=SYSTEM, user=user,
                    effort=SETTINGS.writer_effort, schema=SCHEMA)
    return Script.from_json(data["title"], data["lines"])
