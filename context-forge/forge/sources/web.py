"""Source 3: the live web, via Claude's server-side web search tool.

Anthropic runs the searches inside the API call, so there is no search API
key. The catch: a `web_search_result` block carries a title and URL but its
page text is encrypted (only Claude can read it). To get *readable* passages
we ask Claude for short research notes with citations turned on; every
citation carries the exact `cited_text` it quoted from a page. Those quotes
become our web ContextItems: verbatim text plus its URL, not a paraphrase.

Two API quirks handled here:
  • `pause_turn`: a long server-side search loop may pause; we send the
    paused turn back unchanged and it carries on.
  • search errors don't raise: they arrive as a result block whose content is
    an error object instead of a list.
"""

from __future__ import annotations

from typing import Any

from ..config import SETTINGS
from ..items import ContextItem
from ..llm import ask

SYSTEM = """You are a research scout. Search the web for the query and write \
brief factual notes, one short paragraph per distinct source. Quote or closely \
paraphrase the sources; do not add claims of your own. Prefer primary and recent \
sources."""

TOOL = {"type": "web_search_20260209", "name": "web_search",
        "max_uses": SETTINGS.max_web_searches}

MAX_RESUMES = 3   # how many `pause_turn` continuations we allow


def harvest(blocks: list) -> tuple[list[ContextItem], list[str]]:
    """Turn response blocks into ContextItems (one per URL) plus the queries run."""
    by_url: dict[str, ContextItem] = {}   # dict keeps first-seen order
    queries: list[str] = []
    for b in blocks:
        kind = getattr(b, "type", "")
        if kind == "server_tool_use" and getattr(b, "name", "") == "web_search":
            queries.append(b.input.get("query", ""))
        elif kind == "text":
            for c in getattr(b, "citations", None) or []:
                if getattr(c, "type", "") != "web_search_result_location":
                    continue
                item = by_url.get(c.url)
                if item is None:
                    item = by_url[c.url] = ContextItem(source="web", title=c.title or c.url,
                                                       text="", url=c.url)
                quote = " ".join(c.cited_text.split())
                if quote and quote not in item.text:          # several quotes, one page
                    item.text = f"{item.text} … {quote}" if item.text else quote
    return [i for i in by_url.values() if i.text], queries


def search_web(query: str, client: Any) -> tuple[list[ContextItem], list[str]]:
    messages: list = [{"role": "user", "content": query}]
    blocks: list = []
    for _ in range(MAX_RESUMES + 1):
        resp = ask(client, system=SYSTEM, messages=messages,
                   effort=SETTINGS.effort_small, tools=[TOOL])
        blocks += resp.content
        if resp.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": resp.content})  # resume as-is
    items, queries = harvest(blocks)
    return items[: SETTINGS.per_source_candidates], queries
