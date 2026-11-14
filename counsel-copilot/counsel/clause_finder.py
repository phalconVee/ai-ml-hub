"""Agent 1, the Clause Finder: locate the clauses that bear on the question.

Think of a paralegal with a highlighter. It does not answer the question; it
finds the relevant clauses, says how well they cover it, and lists what
outside legal context would be needed. That hand-off is a *tool call*:

    search_documents(query)      ← as many times as it needs (keyword search)
    file_report(clause_ids, coverage, summary, legal_questions)   ← exactly once, to finish

Why end with a tool instead of free text? A tool with `strict: true` makes the
API guarantee the arguments match the schema, so the hand-off to the next
agent is structured data, not prose we have to parse.

Then *our code* verifies the report: a clause id counts only if the finder was
actually shown that clause by a search. An id it invents is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .config import SETTINGS
from .ingest import Chunk
from .llm import call_claude
from .search import ClauseIndex

SYSTEM = """You are a meticulous paralegal. Your job is to find the clauses in \
the user's legal documents that bear on their question. You do NOT answer the \
question yourself and you never rely on memory of what such documents usually say.

Work like this:
1. Read the passages already retrieved for the question.
2. Search again with search_documents using the words a drafter would use \
("terminate", "notice", "indemnify", "governing law", defined terms in capitals). \
Search each part of a multi-part question separately. Check definitions \
of capitalised terms the clauses rely on.
3. When you have what the documents offer, call file_report once:
   - clause_ids: every relevant clause id you saw, most important first.
   - coverage: "full" if the clauses alone answer the question, "partial" if \
they answer part of it or the answer depends on the law, "none" if nothing relevant.
   - legal_questions: 0-3 short questions about the LAW (not the document) that \
an answer needs, e.g. "Are late fees above 5% enforceable in California residential \
leases?". Leave empty if the documents settle it.
   - summary: two or three sentences on what the clauses say."""

SEARCH_TOOL = {
    "name": "search_documents",
    "description": "Keyword (BM25) search over the user's uploaded legal documents. "
    "Returns the best-matching clauses with their ids, page and heading. Exact wording "
    "matters: try the drafter's vocabulary and defined terms.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Keywords to search for"}},
        "required": ["query"],
        "additionalProperties": False,
    },
}

REPORT_TOOL = {
    "name": "file_report",
    "description": "Hand your findings to the drafting attorney. Call exactly once, last.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "clause_ids": {"type": "array", "items": {"type": "string"}},
            "coverage": {"type": "string", "enum": ["full", "partial", "none"]},
            "legal_questions": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
        },
        "required": ["clause_ids", "coverage", "legal_questions", "summary"],
        "additionalProperties": False,
    },
}


@dataclass
class ClauseReport:
    clauses: list[Chunk]                 # verified, most important first
    coverage: str                        # "full" | "partial" | "none"
    legal_questions: list[str]
    summary: str
    queries: list[str] = field(default_factory=list)   # every search run, for the trace
    dropped_ids: list[str] = field(default_factory=list)  # ids it cited but never saw


def format_hits(hits: list[tuple[Chunk, float]]) -> str:
    """How the finder sees search results: id, where, score, then the clause text."""
    if not hits:
        return "No clauses matched these keywords. Try different wording."
    return "\n\n".join(f"[{c.id}] {c.title} (score {s:.2f})\n{c.text}" for c, s in hits)


def verify(raw: dict, seen: dict[str, Chunk], queries: list[str]) -> ClauseReport:
    """Keep only clause ids the finder was really shown; downgrade coverage if
    nothing survives."""
    ids = list(dict.fromkeys(raw.get("clause_ids", [])))     # de-duplicate, keep order
    kept = [seen[i] for i in ids if i in seen]
    dropped = [i for i in ids if i not in seen]
    coverage = raw.get("coverage", "none")
    if not kept:
        coverage = "none"       # "full coverage" with zero real clauses is not a thing
    return ClauseReport(kept, coverage, list(raw.get("legal_questions", []))[:3],
                        raw.get("summary", ""), queries, dropped)


def find_clauses(question: str, index: ClauseIndex, client: Any,
                 on_event: Callable[[str], None] = lambda _m: None) -> ClauseReport:
    seen: dict[str, Chunk] = {}     # every clause shown to the model, by id
    queries = [question]

    # Seed: search with the raw question before the model starts, so it always
    # has something to read even if it never calls the tool.
    first = index.search(question, SETTINGS.top_k)
    seen.update({c.id: c for c, _ in first})
    messages: list = [{"role": "user", "content": (
        f"Documents available: {', '.join(index.docs)}\n\nQuestion: {question}\n\n"
        f"Passages retrieved for the question:\n\n{format_hits(first)}")}]

    for turn in range(SETTINGS.max_finder_turns):
        response = call_claude(client, system=SYSTEM, messages=messages,
                               tools=[SEARCH_TOOL, REPORT_TOOL])
        # Append the WHOLE content (thinking blocks included): the API needs
        # them back unchanged for the conversation to stay valid.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            return ClauseReport([], "none", [], "The model declined to review these documents.", queries)

        calls = [b for b in response.content if getattr(b, "type", "") == "tool_use"]
        report = next((b for b in calls if b.name == "file_report"), None)
        if report is not None:
            on_event(f"Clause Finder filed its report ({len(report.input.get('clause_ids', []))} clauses)")
            return verify(report.input, seen, queries)

        if not calls:
            break   # it stopped talking without reporting: handled below

        # Run every search it asked for; all results go back in ONE user message.
        results = []
        for call in calls:
            query = call.input["query"]
            on_event(f"Clause Finder searching: {query!r}")
            queries.append(query)
            hits = index.search(query, SETTINGS.top_k)
            seen.update({c.id: c for c, _ in hits})
            results.append({"type": "tool_result", "tool_use_id": call.id,
                            "content": format_hits(hits)})
        messages.append({"role": "user", "content": results})

    # Out of turns (or it never filed): fall back to the seed hits, honestly labelled.
    on_event("Clause Finder did not file a report; using the top search hits")
    return ClauseReport([c for c, _ in first], "partial" if first else "none", [],
                        "The finder did not finish; these are the top keyword matches.", queries)
