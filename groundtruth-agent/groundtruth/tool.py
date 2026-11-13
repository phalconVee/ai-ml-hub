"""Brick 4: wrap the index as a tool Claude can call.

Two halves:
1. `SEARCH_TOOL` - the JSON schema Claude sees: a name, a description that
   tells it *when* to use the tool, and the arguments it may pass.
2. `run_search` - what actually happens when Claude calls it.

The key trick is the *return format*. Instead of pasting raw text back, we
return `search_result` content blocks with `citations` enabled. Claude then
cites them natively: every claim in its answer comes back attached to the
exact passage (source + title + quoted text) it relied on. No prompt-hacking
of "[1]" markers, no regex parsing of the answer.
"""

from __future__ import annotations

from typing import Any

from .config import SETTINGS
from .index import BM25Index

SEARCH_TOOL = {
    "name": "search_documents",
    "description": (
        "Keyword search over the user's uploaded documents (text, tables and figure "
        "descriptions, parsed from PDFs). Returns the best-matching passages with their "
        "page numbers. Use specific keywords: names, metrics, years, table headings. "
        "If results look off-target, search again with different wording or synonyms; "
        "for multi-part questions, search each part separately."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keywords to look for, e.g. 'operating margin 2023 table'.",
            },
            "document": {
                "type": "string",
                "description": "Optional: restrict the search to one file name from the document list.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def run_search(index: BM25Index, tool_input: dict[str, Any]) -> tuple[list[dict], int]:
    """Execute one tool call. Returns (tool_result content blocks, number of hits)."""
    # Claude's tool input is already parsed JSON; still validate before trusting it.
    query = tool_input.get("query")
    if not isinstance(query, str) or not query.strip():
        return [{"type": "text", "text": "Error: 'query' must be a non-empty string."}], 0

    doc = tool_input.get("document") or None
    if doc and doc not in index.docs:
        # Tell Claude what exists rather than failing: it can correct itself.
        return [{"type": "text", "text": f"No document named {doc!r}. Available: {index.docs}"}], 0

    hits = index.search(query, k=SETTINGS.top_k, doc=doc)
    if not hits:
        # Per the docs: an empty search is a normal text result, not an error.
        return [{"type": "text", "text": f"No passages matched {query!r}. Try other keywords."}], 0

    blocks = []
    for chunk, _score in hits:
        blocks.append({
            "type": "search_result",
            "source": chunk.source,        # e.g. "report.pdf#page=4"
            "title": chunk.title,          # e.g. "report.pdf, page 4 (table)"
            # One text block per piece: citations can point at individual
            # paragraphs or table rows instead of the whole chunk.
            "content": [{"type": "text", "text": piece} for piece in chunk.pieces],
            "citations": {"enabled": True},
        })
    return blocks, len(blocks)
