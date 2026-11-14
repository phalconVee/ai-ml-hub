"""Agent 1, the Researcher: source text in, an episode outline out.

The researcher reads the whole source and decides what the episode is
*about*: the one-line hook, the 3-6 points worth talking through (each with
the concrete detail that makes it land), and the things a curious listener
would ask. The scriptwriter never sees the raw page, only this outline. That
split does two jobs:
  • the writer gets a short, digested brief instead of 15k tokens of page,
    so it spends its effort on dialogue, not on reading;
  • the outline is a checkpoint you can read in the UI: if the episode gets
    something wrong, you can see whether the reading or the writing slipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import SETTINGS
from .llm import ask_json
from .source import Source

SYSTEM = """You are the researcher for a two-host explainer podcast. You read \
a source document and produce an outline the scriptwriter will turn into a \
conversation.

Rules:
- Use ONLY what the source says. No outside facts, no guesses presented as facts.
- Keep numbers, names and dates exactly as written in the source.
- Pick the points a smart non-expert would find most interesting, in an order \
that builds understanding (context first, implications last).
- For each point give the concrete detail (an example, number or quote) that \
makes it stick.
- Questions are what a curious listener would ask; answer them only from the source.
- If the source is opinion, say whose opinion it is."""

OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "A short, catchy episode title"},
        "hook": {"type": "string", "description": "One sentence: why should anyone listen?"},
        "summary": {"type": "string", "description": "2-3 sentence plain-language summary"},
        "key_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "string"},
                    "detail": {"type": "string", "description": "Example, number or quote from the source"},
                },
                "required": ["point", "detail"],
                "additionalProperties": False,
            },
        },
        "listener_questions": {"type": "array", "items": {"type": "string"}},
        "takeaway": {"type": "string", "description": "The one thing to remember"},
    },
    "required": ["title", "hook", "summary", "key_points", "listener_questions", "takeaway"],
    "additionalProperties": False,
}


@dataclass
class Outline:
    title: str
    hook: str
    summary: str
    key_points: list[dict] = field(default_factory=list)   # [{"point", "detail"}]
    listener_questions: list[str] = field(default_factory=list)
    takeaway: str = ""

    def brief(self) -> str:
        """The outline as plain text: what the scriptwriter reads."""
        lines = [f"TITLE: {self.title}", f"HOOK: {self.hook}", f"SUMMARY: {self.summary}",
                 "KEY POINTS:"]
        lines += [f"{i}. {kp['point']}\n   Detail: {kp['detail']}"
                  for i, kp in enumerate(self.key_points, 1)]
        lines.append("LISTENER QUESTIONS:")
        lines += [f"- {q}" for q in self.listener_questions]
        lines.append(f"TAKEAWAY: {self.takeaway}")
        return "\n".join(lines)


def research(source: Source, client: Any) -> Outline:
    origin = source.url or "pasted by the user"
    note = "\n(The source was long and has been cut off at the end.)" if source.truncated else ""
    user = (f"SOURCE TITLE: {source.title}\nSOURCE: {origin}{note}\n\n"
            f"<source>\n{source.text}\n</source>\n\n"
            "Outline an episode on this. Aim for 3 to 6 key points.")
    data = ask_json(client, who="researcher", system=SYSTEM, user=user,
                    effort=SETTINGS.researcher_effort, schema=OUTLINE_SCHEMA)
    return Outline(**data)  # keys match the dataclass fields exactly (schema enforced)
