"""The reply pipeline: recall → answer → remember.

    user message
        │
        ├─ recall()   (retrieve.py, no LLM)   relevant facts from the graph
        ├─ reply()    (Claude, effort medium) answer using those facts + recent chat
        └─ remember() (Claude, effort low)    extract facts from the exchange,
                                              graph.merge() them (code)

`reply` and `remember` are separate on purpose: the app shows and speaks the
answer first, then updates memory while the avatar is already talking. And
if remembering fails (network blip, odd JSON), the user still has their
answer; only that turn's memory is lost, and the UI says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import SETTINGS
from .extract import extract
from .graph import Delta, KnowledgeGraph
from .llm import ask_text
from .retrieve import Recall, recall

PERSONA = "Mira"

# Static system prompt (cacheable). Everything that changes per turn, the
# recalled memory included, goes into the user message instead.
SYSTEM = f"""You are {PERSONA}, a warm, curious personal companion with a long-term memory.
Your replies are SPOKEN ALOUD by an animated avatar, so write for the ear:
- 1 to 4 short sentences, conversational, like talking to a friend.
- No markdown, no bullet points, no headings, no emoji, no URLs, no code blocks.

Memory:
- Each message comes with a <memory> block: facts remembered from earlier conversations,
  as "subject predicate object" lines. "{SETTINGS.user_node}" is the person you're talking to.
- Use memory naturally, the way a friend would ("How's the Rust project going?"), only when
  it helps. Never recite the list or mention graphs, databases or "my memory block".
- Facts marked NO LONGER TRUE are history: you may refer to them in the past tense.
- If you don't know something about the user, say so or ask. Never invent personal facts.
- When the user shares something new, respond to it; it will be remembered automatically."""


@dataclass
class TurnResult:
    reply: str
    recall: Recall
    turn: int = 0
    delta: Delta = field(default_factory=Delta)
    memory_error: str | None = None


def speakable(text: str) -> str:
    """Belt and braces: strip markdown a voice would read out ('asterisk asterisk')."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)        # code blocks
    text = re.sub(r"[*_`#>]+", "", text)                       # emphasis, headings, quotes
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)       # [label](url) → label
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)        # bullet markers
    return re.sub(r"\s+", " ", text).strip()


class Companion:
    def __init__(self, client: Any, graph: KnowledgeGraph):
        self.client = client
        self.graph = graph

    def reply(self, user_text: str, history: list[dict]) -> TurnResult:
        """Recall, then answer. Doesn't touch memory yet."""
        found = recall(self.graph, user_text)
        # Only the latest turn carries the memory block; earlier turns are
        # sent as they were said, so the prefix stays stable turn to turn.
        recent = history[-SETTINGS.history_turns * 2:]
        latest = f"<memory>\n{found.as_context()}\n</memory>\n\n{user_text}"
        messages = [*recent, {"role": "user", "content": latest}]
        text = ask_text(self.client, system=SYSTEM, messages=messages,
                        effort=SETTINGS.reply_effort)
        return TurnResult(reply=speakable(text), recall=found)

    def remember(self, user_text: str, result: TurnResult) -> TurnResult:
        """Log the turn, extract facts from it, merge them into the graph."""
        result.turn = self.graph.add_turn(user_text, result.reply)
        try:
            found = extract(self.client, user_text=user_text, reply_text=result.reply,
                            known=self.graph.names(SETTINGS.known_entities_hint))
            result.delta = self.graph.merge(found, result.turn)
        except Exception as e:  # a memory failure must never cost the user their answer
            result.memory_error = f"{type(e).__name__}: {e}"
        return result

    def turn(self, user_text: str, history: list[dict]) -> TurnResult:
        """Both steps, for callers that don't need to speak in between (tests, CLI)."""
        return self.remember(user_text, self.reply(user_text, history))
