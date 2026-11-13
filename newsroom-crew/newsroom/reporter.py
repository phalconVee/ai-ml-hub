"""Agent 1, the Reporter: go out on the web and come back with notes.

The reporter gets two *server tools*. Anthropic runs them inside the API call,
so we write no search function and need no search-API key:

  web_search  → finds pages (title, url, page age)
  web_fetch   → opens a page it found and reads the full text

We only declare the tools, then read what happened out of the response
blocks. The reporter finishes with plain-text field notes. The API attaches
*citations* to that text: each one says "this sentence came from this url,
and here is the exact text quoted". Those citations, not the model's say-so,
become the evidence the writer works from (see ledger.py).

Two quirks of server tools:
  • `pause_turn`: a long search loop can pause mid-turn. We send the paused
    turn back unchanged (no "continue" message) and the server carries on.
  • Tool errors don't raise. They arrive as a result block whose content is
    an error object instead of results; ledger.py skips those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import SETTINGS
from .llm import ask_claude, check_refusal, text_of

SYSTEM = """You are the reporter on a small news desk. Your job is to gather \
facts, not to write the story.

Search the web for the most recent, reliable coverage of the topic. Prefer \
primary sources (official statements, filings, papers, data) and established \
outlets. Open (fetch) the few pages that matter most and read them.

Then write field notes for the writer:
- Lead with what is new and when it happened (give dates).
- One fact per line, stated plainly, with numbers and names exactly as sourced.
- Note where sources disagree, and what is still unconfirmed.
- Never add facts you did not find in a source."""

# Server tools: declared, not implemented. `_20260209` versions filter results
# with code before they reach the context (fewer wasted tokens).
TOOLS = [
    {"type": "web_search_20260209", "name": "web_search",
     "max_uses": SETTINGS.max_searches},
    {"type": "web_fetch_20260209", "name": "web_fetch",
     "max_uses": SETTINGS.max_fetches,
     "max_content_tokens": SETTINGS.max_fetch_tokens},
]


@dataclass
class Notes:
    text: str                                        # the reporter's field notes
    blocks: list = field(default_factory=list)       # every response block, for the ledger
    queries: list[str] = field(default_factory=list) # what it searched for
    fetched: list[str] = field(default_factory=list) # urls it opened


def brief(topic: str, today: str, angle: str) -> str:
    """The assignment. The date goes here, not in SYSTEM, so the system prompt
    stays byte-identical across runs (good for prompt caching)."""
    lines = [f"Today is {today}.", f"Topic: {topic}"]
    if angle:
        lines.append(f"Angle the editor wants: {angle}")
    lines.append("Find what happened recently and why it matters.")
    return "\n".join(lines)


def report(topic: str, client: Any, *, today: str, angle: str = "") -> Notes:
    messages: list = [{"role": "user", "content": brief(topic, today, angle)}]
    blocks: list = []

    # One request usually finishes the job: the server runs the search/fetch
    # loop itself. We only loop to resume a `pause_turn`.
    for _ in range(SETTINGS.max_pause_resumes + 1):
        response = ask_claude(client, system=SYSTEM, messages=messages,
                              effort=SETTINGS.reporter_effort, tools=TOOLS)
        blocks.extend(response.content)
        if response.stop_reason != "pause_turn":
            break
        # Resume: append the paused assistant turn as-is and ask again.
        messages.append({"role": "assistant", "content": response.content})

    check_refusal(response, "reporter")

    # Pull the trail out of the blocks, for the UI's "how we got here" panel.
    queries, fetched = [], []
    for b in blocks:
        if getattr(b, "type", "") == "server_tool_use":
            if b.name == "web_search":
                queries.append(b.input.get("query", ""))
            elif b.name == "web_fetch":
                fetched.append(b.input.get("url", ""))

    # Only the *last* response's text is the finished notes; earlier paused
    # turns may hold half-sentences like "Let me search for…".
    return Notes(text_of(response).strip(), blocks, queries, fetched)
