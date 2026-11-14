"""Step 4: conversation memory.

A notebook chat has to remember what was said, but resending every past turn
forever gets slow and expensive. So memory has three parts:

1. **Recent turns, verbatim.** The last `memory_turns` question/answer pairs
   are replayed as normal user/assistant messages. (Only the answer *text* is
   replayed, not the passages it was built from, which keeps history small.)
2. **A running summary.** When the window overflows, the oldest turns are
   folded into a short summary by Claude and dropped from the replay. The
   summary rides along in the system prompt.
3. **Follow-up rewriting.** "What about the second one?" is a terrible search
   query. Before retrieving, Claude rewrites a follow-up into a standalone
   query using the recent turns ("second risk factor in the 2024 report").

Everything here is plain data, saved with the notebook, so memory survives an
app restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import llm
from .config import SETTINGS

REWRITE_SYSTEM = (
    "You turn a user's latest message in a research chat into one standalone search query "
    "for a keyword + semantic search over their documents. Resolve pronouns and references "
    "('it', 'the second one', 'that paper') using the conversation. Keep names, numbers and "
    "technical terms exactly. If the message is already standalone, return it unchanged."
)
REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}

SUMMARY_SYSTEM = (
    "You maintain a compact memory of a research conversation. Merge the earlier summary "
    "with the new turns into at most 150 words: what the user is investigating, questions "
    "asked, key findings and open threads. Plain prose, no citations."
)
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


@dataclass
class Memory:
    turns: list[Any] = field(default_factory=list)   # Answer objects, oldest first
    summary: str = ""          # running summary of turns[:summarized]
    summarized: int = 0        # how many of the oldest turns the summary covers

    @property
    def recent(self) -> list[Any]:
        return self.turns[self.summarized:]

    def as_messages(self) -> list[dict]:
        """Recent turns as alternating user/assistant messages for the API."""
        msgs = []
        for turn in self.recent:
            msgs.append({"role": "user", "content": turn.question})
            msgs.append({"role": "assistant", "content": turn.plain_text() or "(no answer)"})
        return msgs

    def transcript(self, turns: list[Any]) -> str:
        return "\n\n".join(f"User: {t.question}\nAssistant: {t.plain_text()}" for t in turns)

    def standalone_query(self, question: str, client) -> str:
        """Rewrite a follow-up into a self-contained search query (no call on turn 1)."""
        if not self.recent:
            return question
        prompt = (
            (f"Earlier summary: {self.summary}\n\n" if self.summary else "")
            + f"Recent conversation:\n{self.transcript(self.recent[-3:])}\n\n"
            + f"Latest message: {question}"
        )
        try:
            query = llm.create_json(client, system=REWRITE_SYSTEM, prompt=prompt,
                                    schema=REWRITE_SCHEMA)["query"].strip()
            return query or question
        except Exception:
            return question   # rewriting is an optimisation; never fail the turn on it

    def add(self, turn: Any, client=None) -> None:
        """Store a finished turn, then compress if the window overflowed."""
        self.turns.append(turn)
        overflow = len(self.recent) - SETTINGS.memory_turns
        if overflow > 0 and client is not None:
            old = self.recent[:overflow]
            prompt = f"Earlier summary: {self.summary or '(none)'}\n\nNew turns:\n{self.transcript(old)}"
            try:
                self.summary = llm.create_json(client, system=SUMMARY_SYSTEM, prompt=prompt,
                                               schema=SUMMARY_SCHEMA)["summary"].strip()
                self.summarized += overflow
            except Exception:
                pass   # keep replaying verbatim; we'll try again next turn

    def clear(self) -> None:
        self.turns, self.summary, self.summarized = [], "", 0
