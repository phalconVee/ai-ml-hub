"""Stage 2, the Scout: answer from the live web when the documents can't.

Web search here is a *server tool*: Anthropic runs the searches inside the API
call. We don't implement a search function or need a separate search API key;
we declare the tool and read the results out of the response.

Two quirks to handle:
  • `pause_turn`: a long server-side search loop can pause mid-turn. We send
    the paused turn back unchanged and the server carries on.
  • Search errors don't raise: they come back as a result block whose content
    is an error object instead of a list of results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import SETTINGS
from .llm import ask_claude, text_of

SYSTEM = """You answer questions using web search. The user's own documents \
were checked first and did not contain the answer, so search the web. Prefer \
primary and recent sources. Answer concisely and say plainly if sources disagree \
or nothing reliable was found."""

WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",  # current version, with built-in result filtering
    "name": "web_search",
    "max_uses": SETTINGS.max_web_searches,
}


@dataclass
class ScoutResult:
    answer: str
    sources: list[dict] = field(default_factory=list)  # [{"title", "url"}], de-duplicated
    queries: list[str] = field(default_factory=list)   # what it searched for


def collect_sources(content: list) -> tuple[list[dict], list[str]]:
    """Pull search queries and source links out of a response's blocks."""
    sources: dict[str, dict] = {}   # url → {"title", "url"}; dict keeps first-seen order
    queries: list[str] = []
    for block in content:
        kind = getattr(block, "type", "")
        if kind == "server_tool_use" and block.name == "web_search":
            queries.append(block.input.get("query", ""))
        elif kind == "web_search_tool_result":
            # Success → a list of results. Error → a single object. Check first.
            if isinstance(block.content, list):
                for r in block.content:
                    sources.setdefault(r.url, {"title": r.title, "url": r.url})
        elif kind == "text":
            # Citations on the answer text are the sources actually relied on.
            for c in getattr(block, "citations", None) or []:
                url = getattr(c, "url", None)
                if url:
                    sources.setdefault(url, {"title": getattr(c, "title", url), "url": url})
    return list(sources.values()), queries


def scout(question: str, client: Any) -> ScoutResult:
    messages: list = [{"role": "user", "content": question}]
    all_blocks: list = []
    for _ in range(SETTINGS.max_pause_resumes + 1):
        response = ask_claude(client, system=SYSTEM, messages=messages, tools=[WEB_SEARCH_TOOL])
        all_blocks.extend(response.content)
        if response.stop_reason != "pause_turn":
            break
        # Resume: append the paused assistant turn as-is. No "continue" message.
        messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "refusal":
        return ScoutResult("The model declined to answer this question.")
    sources, queries = collect_sources(all_blocks)
    return ScoutResult(text_of(response).strip(), sources, queries)
