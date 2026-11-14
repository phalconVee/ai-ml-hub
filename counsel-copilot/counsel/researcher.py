"""Agent 2, the Legal Researcher: fetch outside legal context from the web.

It only runs when the Clause Finder says the documents aren't enough (coverage
"partial"/"none", or it listed questions about the law). Many contract
questions are really two questions: "what does my lease say?" and "is that
allowed where I live?". The documents answer the first; this agent the second.

Web search is a *server tool*: Anthropic runs the searches inside the API call.
We declare it and read results out of the response. No search API key.

Its output is turned into `search_result` blocks (one per web page, holding
the passages Claude quoted from it), so the Drafter can cite web pages with
exactly the same native citation mechanism it uses for clauses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .config import SETTINGS
from .llm import call_claude

SYSTEM = """You are a legal research assistant. Search the web for the legal \
background needed to answer the questions below. Prefer primary and official \
sources (statutes, government and court sites, bar associations) and reputable \
legal publishers over blogs and forums. If a jurisdiction is given, research \
that jurisdiction; if not, describe the general position and say that it \
varies. Report what the sources say, concisely, and note where they disagree \
or where the law has recently changed. Do not give advice."""

WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",     # current version (with built-in result filtering)
    "name": "web_search",
    "max_uses": SETTINGS.max_web_searches,
}


@dataclass
class WebSource:
    url: str
    title: str
    quotes: list[str] = field(default_factory=list)   # passages Claude cited from it


@dataclass
class Research:
    notes: str                                         # the researcher's write-up
    sources: list[WebSource] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)

    def as_search_results(self) -> list[dict]:
        """Web sources that were actually quoted, as citable search_result blocks."""
        return [{
            "type": "search_result",
            "source": s.url,
            "title": s.title,
            "content": [{"type": "text", "text": q} for q in s.quotes],
            "citations": {"enabled": True},
        } for s in self.sources if s.quotes]


def collect(blocks: list) -> tuple[list[WebSource], list[str]]:
    """Pull the queries run and the pages cited out of a response's blocks."""
    by_url: dict[str, WebSource] = {}
    queries: list[str] = []
    for b in blocks:
        kind = getattr(b, "type", "")
        if kind == "server_tool_use" and b.name == "web_search":
            queries.append(b.input.get("query", ""))
        elif kind == "web_search_tool_result":
            # Success → a list of results. Error → one error object. Check first.
            if isinstance(b.content, list):
                for r in b.content:
                    by_url.setdefault(r.url, WebSource(r.url, r.title))
        elif kind == "text":
            # Citations on the write-up are the passages it actually relied on.
            for c in getattr(b, "citations", None) or []:
                url = getattr(c, "url", None)
                if not url:
                    continue
                src = by_url.setdefault(url, WebSource(url, getattr(c, "title", None) or url))
                quote = getattr(c, "cited_text", "")
                if quote and quote not in src.quotes:
                    src.quotes.append(quote)
    return list(by_url.values()), queries


def research_law(question: str, legal_questions: list[str], jurisdiction: str,
                 clause_summary: str, client: Any,
                 on_event: Callable[[str], None] = lambda _m: None) -> Research:
    asks = "\n".join(f"- {q}" for q in legal_questions) or f"- {question}"
    prompt = (f"Jurisdiction: {jurisdiction or 'not specified'}\n\n"
              f"The user's question: {question}\n\n"
              f"What their documents say (summary): {clause_summary or 'no relevant clauses'}\n\n"
              f"Research these legal questions:\n{asks}")
    messages: list = [{"role": "user", "content": prompt}]
    on_event("Legal Researcher searching the web")

    blocks: list = []
    for _ in range(SETTINGS.max_pause_resumes + 1):
        response = call_claude(client, system=SYSTEM, messages=messages, tools=[WEB_SEARCH_TOOL])
        blocks.extend(response.content)
        if response.stop_reason != "pause_turn":
            break
        # A long server-side search loop can pause. Send the turn back as-is
        # (no extra "continue" message) and the server picks up where it left off.
        messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "refusal":
        return Research("The model declined this research request.")
    sources, queries = collect(blocks)
    # Text from every turn (a paused turn may already hold part of the write-up).
    notes = "".join(b.text for b in blocks if getattr(b, "type", "") == "text")
    return Research(notes.strip(), sources, queries)
