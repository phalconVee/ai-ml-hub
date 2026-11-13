"""Agent 2, the Writer: notes + ledger in, a sourced news article out.

The writer has no tools: it cannot search, so it cannot wander off and find
"facts" of its own. Everything it may say is in front of it — the reporter's
notes and the numbered source ledger — and every factual sentence must carry
a marker like [S2] pointing back into that ledger.

The answer comes back as JSON (structured outputs), so the headline, the
standfirst ("dek") and the body arrive as separate fields we can render and
check one by one, with no parsing of prose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import SETTINGS
from .llm import ask_claude, check_refusal, text_of

SYSTEM = """You are the writer on a small news desk. You turn a reporter's \
notes into a clear, neutral news article.

Rules:
- Use ONLY facts found in the notes and the source ledger. No outside knowledge.
- After every sentence that states a fact, cite its source(s) by ledger id in \
square brackets, e.g. "Rates were held at 4.25% [S1]." or "[S1][S3]".
- Cite only ids that exist in the ledger. Prefer sources with quotes.
- Inverted pyramid: the newest, most important fact first; context after.
- Attribute claims ("according to…", "X said…"). Say what is unconfirmed.
- Plain language, short paragraphs, no hype, no opinion, no invented quotes.
- The body is Markdown paragraphs; you may use one or two ## subheads."""

ARTICLE_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "dek": {"type": "string", "description": "One-sentence standfirst under the headline"},
        "body": {"type": "string", "description": "Markdown body with [S#] citations"},
    },
    "required": ["headline", "dek", "body"],
    "additionalProperties": False,
}


@dataclass
class Draft:
    headline: str
    dek: str
    body: str
    revision: int = 0                                   # 0 = first draft
    notes_from_editor: list[str] = field(default_factory=list)  # what this draft responded to

    def markdown(self) -> str:
        return f"# {self.headline}\n\n*{self.dek}*\n\n{self.body}"


def assignment(topic: str, today: str, words: int, notes: str, ledger: str,
               previous: Draft | None = None, feedback: list[str] | None = None) -> str:
    """Everything the writer sees, in one user message."""
    parts = [
        f"Today is {today}. Write a news article of about {words} words on: {topic}",
        f"REPORTER'S NOTES:\n{notes or '(none)'}",
        f"SOURCE LEDGER (cite these ids):\n{ledger}",
    ]
    if previous is not None:
        # Revision round: show the old draft and exactly what the editor wants changed.
        parts.append(f"YOUR PREVIOUS DRAFT:\n{previous.markdown()}")
        parts.append("EDITOR'S NOTES (fix every one):\n" + "\n".join(f"- {n}" for n in feedback or []))
    return "\n\n".join(parts)


def write(topic: str, notes: str, ledger: str, client: Any, *, today: str,
          words: int = 500, previous: Draft | None = None,
          feedback: list[str] | None = None) -> Draft:
    # Each draft is a fresh, single-turn request. A revision doesn't need the
    # first conversation's history: the old draft + the notes are enough.
    content = assignment(topic, today, words, notes, ledger, previous, feedback)
    response = ask_claude(client, system=SYSTEM, effort=SETTINGS.writer_effort,
                          messages=[{"role": "user", "content": content}],
                          schema=ARTICLE_SCHEMA)
    check_refusal(response, "writer")
    data = json.loads(text_of(response))  # guaranteed to match ARTICLE_SCHEMA
    revision = 0 if previous is None else previous.revision + 1
    return Draft(data["headline"], data["dek"], data["body"], revision, list(feedback or []))
