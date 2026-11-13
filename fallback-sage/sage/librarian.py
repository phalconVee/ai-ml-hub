"""Stage 1, the Librarian: answer from the user's documents, or admit it can't.

The librarian is a small agent. It gets:
  • the question, plus a first round of search results we fetch for it (so it
    always sees the documents, even if it never calls the tool), and
  • one tool, `search_documents`, to search again with better wording.

It loops (think → maybe search → think …) and must finish with a JSON verdict:
    {"verdict": "answered" | "not_found", "answer": ..., "cited_chunks": [...], "reason": ...}

Then *our code* double-checks the verdict (see `verify`). An LLM saying
"answered" is a claim; citing passages it actually retrieved is evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .bm25 import BM25Index, Hit
from .config import SETTINGS
from .llm import ask_claude, text_of

SYSTEM = """You are a careful research librarian. You answer ONLY from the user's \
uploaded documents, never from your own knowledge.

You are given passages retrieved for the question. Each starts with an id in \
square brackets. If they don't settle it, call search_documents with different \
keywords (synonyms, names, numbers): the search is keyword-based, so wording matters.

Finish with your verdict:
- "answered": the passages contain the answer. Write it concisely, cite the ids \
you relied on inline like [file.pdf#p2-c1], and list them in cited_chunks.
- "not_found": the documents do not contain the answer (or only part of it). Say \
in `reason` what is missing. Do not guess."""

# The one tool the librarian can call. `strict` guarantees the input matches.
SEARCH_TOOL = {
    "name": "search_documents",
    "description": "Keyword (BM25) search over the user's uploaded documents. "
    "Returns the best-matching passages with their ids.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Keywords to search for"}},
        "required": ["query"],
        "additionalProperties": False,
    },
}

# Shape of the final answer, enforced by structured outputs.
VERDICT_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["answered", "not_found"]},
            "answer": {"type": "string"},
            "cited_chunks": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "answer", "cited_chunks", "reason"],
        "additionalProperties": False,
    },
}


@dataclass
class LibrarianResult:
    found: bool                    # True only if the verdict survived verification
    answer: str
    reason: str
    cited: list[Hit] = field(default_factory=list)   # the passages it leaned on
    queries: list[str] = field(default_factory=list)  # every search, for the trace panel


def format_hits(hits: list[Hit]) -> str:
    """Render passages the way the model sees them: [id] (score) text."""
    if not hits:
        return "No passages matched these keywords."
    return "\n\n".join(f"[{h.chunk.id}] (score {h.score:.2f})\n{h.chunk.text}" for h in hits)


def verify(raw: dict, seen: dict[str, Hit]) -> tuple[bool, list[Hit], str]:
    """Trust, then verify. An 'answered' verdict counts only if it cites at
    least one passage the librarian was actually shown."""
    if raw.get("verdict") != "answered":
        return False, [], raw.get("reason", "")
    cited = [seen[c] for c in raw.get("cited_chunks", []) if c in seen]
    if not cited:
        return False, [], "Claimed an answer but cited no retrieved passage, so it was rejected."
    return True, cited, raw.get("reason", "")


def consult(question: str, index: BM25Index, client: Any) -> LibrarianResult:
    seen: dict[str, Hit] = {}   # every passage shown to the model, by id
    queries = [question]

    # 1. Seed retrieval: search with the raw question before the model even starts.
    first = index.search(question, SETTINGS.top_k)
    seen.update({h.chunk.id: h for h in first})
    messages: list = [{
        "role": "user",
        "content": f"Question: {question}\n\nRetrieved passages:\n\n{format_hits(first)}",
    }]

    # 2. The agent loop. Each turn: Claude either calls search_documents
    #    (stop_reason "tool_use") or writes its JSON verdict (anything else).
    for _ in range(SETTINGS.max_librarian_turns):
        response = ask_claude(client, system=SYSTEM, messages=messages,
                              tools=[SEARCH_TOOL], output_format=VERDICT_SCHEMA)
        # Always append the FULL content (thinking + tool_use blocks), not just text.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break  # it answered (or stopped) — go read the verdict

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            query = block.input["query"]
            queries.append(query)
            hits = index.search(query, SETTINGS.top_k)
            seen.update({h.chunk.id: h for h in hits})
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": format_hits(hits)})
        # All results for this turn go back in ONE user message.
        messages.append({"role": "user", "content": results})
    else:
        # Loop ran out of turns while still searching: treat as "not found".
        return LibrarianResult(False, "", "Ran out of search turns.", [], queries)

    if response.stop_reason == "refusal":
        return LibrarianResult(False, "", "The model declined this question.", [], queries)

    # 3. Parse the verdict, then verify it against what was really retrieved.
    try:
        raw = json.loads(text_of(response))
    except json.JSONDecodeError:
        return LibrarianResult(False, "", "Verdict was not valid JSON.", [], queries)
    found, cited, reason = verify(raw, seen)
    return LibrarianResult(found, raw.get("answer", "") if found else "", reason, cited, queries)
