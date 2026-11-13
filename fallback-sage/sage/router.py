"""The conductor: documents first, web second.

    question
       │
       ├─ no documents loaded? ─────────────────────────► Scout (web)
       │
       ▼
    Librarian (search docs, verdict, verified)
       │ found ──► answer from docs
       │ not found
       ▼
    web fallback allowed? ── no ──► "not in your documents"
       │ yes
       ▼
    Scout (web) ──► answer from web

The routing decision lives in plain Python, not in a prompt. The LLM can't
"forget" to try the documents first, because it never gets the web tool
until the librarian has said no.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .bm25 import BM25Index
from .librarian import consult
from .scout import scout


@dataclass
class Answer:
    route: Literal["docs", "web", "none"]   # where the answer came from
    text: str
    sources: list[dict] = field(default_factory=list)   # docs: {id, page, snippet}; web: {title, url}
    trace: list[str] = field(default_factory=list)      # human-readable steps, for the UI


def answer(question: str, index: BM25Index | None, client: Any,
           allow_web: bool = True) -> Answer:
    trace: list[str] = []

    if index is not None and len(index) > 0:
        lib = consult(question, index, client)
        trace.append(f"📚 Searched documents for: {', '.join(repr(q) for q in lib.queries)}")
        if lib.found:
            trace.append(f"✅ Answer found in {len(lib.cited)} passage(s)")
            sources = [{"id": h.chunk.id, "page": h.chunk.page, "snippet": h.chunk.text[:300]}
                       for h in lib.cited]
            return Answer("docs", lib.answer, sources, trace)
        trace.append(f"❌ Not in documents: {lib.reason}")
    else:
        trace.append("📭 No documents loaded, skipping straight to the web")

    if not allow_web:
        return Answer("none", "I couldn't find this in your documents (web fallback is off).",
                      [], trace)

    web = scout(question, client)
    trace.append(f"🌐 Web searches: {', '.join(repr(q) for q in web.queries) or 'none'}")
    return Answer("web", web.answer, web.sources, trace)
