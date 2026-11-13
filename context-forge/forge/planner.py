"""Stage 1, the Planner: decide which sources to ask and what to ask them.

A user question is rarely the best search query. "What did my notes say about
the new attention trick, and is there research backing it?" should become a
keyword query for the documents, a web query, and an arXiv query, and maybe
the web isn't needed at all. One small structured-output call makes those
decisions, and its rationale is shown in the UI.

If the call fails (no key, network, refusal) we fall back to "ask every
enabled source the raw question", so the pipeline still runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import SETTINGS
from .llm import ask, json_of

SYSTEM = """You plan retrieval for a research assistant. Given a question, \
write a focused keyword query for each source and decide whether the web and \
academic papers are worth searching. Documents and memory are local and cheap: \
always query them. Use the web for recent events, products, or facts likely to \
be online; use papers for scientific or technical research questions. Keep \
queries short (3-8 words) and keyword-style."""

SCHEMA = {
    "type": "object",
    "properties": {
        "doc_query": {"type": "string"},
        "web_query": {"type": "string"},
        "paper_query": {"type": "string"},
        "use_web": {"type": "boolean"},
        "use_papers": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["doc_query", "web_query", "paper_query", "use_web", "use_papers", "rationale"],
    "additionalProperties": False,
}


@dataclass
class Plan:
    doc_query: str
    web_query: str
    paper_query: str
    use_web: bool
    use_papers: bool
    rationale: str
    by_llm: bool = True   # False = the fallback plan was used


def fallback_plan(question: str, why: str) -> Plan:
    return Plan(question, question, question, True, True, why, by_llm=False)


def make_plan(question: str, client: Any) -> Plan:
    try:
        resp = ask(client, system=SYSTEM, effort=SETTINGS.effort_small, schema=SCHEMA,
                   messages=[{"role": "user", "content": question}])
        data = json_of(resp)
        if not data:
            return fallback_plan(question, "Planner returned nothing usable; querying every source.")
        return Plan(**data)
    except Exception as e:  # planning is an optimisation, never a blocker
        return fallback_plan(question, f"Planner unavailable ({type(e).__name__}); querying every source.")
