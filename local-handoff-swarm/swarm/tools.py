"""The "doing" tools: plain functions, no keys.

Each function's docstring is what the model reads to decide when to call it
(see schema.py), so they're written as instructions to the model.

A tool that takes a `context` parameter gets the swarm's shared dict injected
by the loop. `web_search` uses it to remember every source it found, which the
UI shows and the Writer can cite.
"""

from __future__ import annotations

import time

from .config import SETTINGS


def _ddgs_text(query: str, max_results: int) -> list[dict]:
    """Thin wrapper so tests can replace the network call (monkeypatch this)."""
    from ddgs import DDGS  # keyless DuckDuckGo (and friends) metasearch

    return DDGS().text(query, max_results=max_results)


def web_search(query: str, context: dict) -> str:
    """Search the web for up-to-date information. Use a short keyword query.
    Returns numbered results with title, URL and a snippet."""
    # Keyless search is flaky: engines throttle bursts and ddgs then reports
    # "No results found". One short retry fixes most of those.
    for attempt in range(2):
        try:
            hits = _ddgs_text(query, SETTINGS.search_results)
            break
        except Exception as exc:
            if attempt == 1:
                # Still failing: tell the model, so it can say so instead of crashing.
                return f"Search failed ({exc.__class__.__name__}). Answer from what you know, and say so."
            time.sleep(1.5)
    if not hits:
        return "No results. Try different keywords."

    sources = context.setdefault("sources", [])   # shared across agents and turns
    lines = []
    for hit in hits:
        url = hit.get("href", "")
        if url and url not in [s["url"] for s in sources]:
            sources.append({"title": hit.get("title", url), "url": url})
        n = [s["url"] for s in sources].index(url) + 1 if url else "?"
        lines.append(f"[{n}] {hit.get('title', '')}\n    {url}\n    {hit.get('body', '')}")
    return "\n".join(lines)


def word_count(text: str) -> str:
    """Count the words in a piece of text. Use it to check a summary or draft
    against a requested length."""
    return f"{len(text.split())} words"
