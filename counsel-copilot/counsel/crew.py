"""The crew: wire the three agents into one pipeline.

    question ─▶ Clause Finder ──report──▶ (needs the law?) ──yes──▶ Legal Researcher
                     │                          │ no                      │
                     └──────────────────────────┴───────────▶ Drafter ◀──┘
                                                                 │
                                         + NOT-LEGAL-ADVICE notice (our code)

The routing is plain Python, not an LLM decision: researching costs money and
time, so it runs only when the finder's report says the documents are not
enough (coverage != "full") or that the answer depends on the law.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .clause_finder import ClauseReport, find_clauses
from .drafter import Draft, draft_answer
from .researcher import Research, research_law
from .search import ClauseIndex


@dataclass
class CrewResult:
    draft: Draft
    report: ClauseReport
    research: Research | None
    route: list[str] = field(default_factory=list)   # which agents ran, in order

    def markdown(self) -> str:
        return self.draft.markdown()


def needs_research(report: ClauseReport) -> bool:
    """The routing rule, in one line so it's easy to test and to change."""
    return report.coverage != "full" or bool(report.legal_questions)


def run_crew(question: str, index: ClauseIndex, client: Any, *, jurisdiction: str = "",
             allow_web: bool = True,
             on_event: Callable[[str], None] = lambda _m: None) -> CrewResult:
    route: list[str] = []

    # 1. Clause Finder (skipped when nothing is uploaded: there is nothing to find).
    if len(index):
        report = find_clauses(question, index, client, on_event)
        route.append("Clause Finder")
    else:
        report = ClauseReport([], "none", [], "No documents were uploaded.")

    # 2. Legal Researcher, only when the report says the documents aren't enough.
    research = None
    if allow_web and needs_research(report):
        research = research_law(question, report.legal_questions, jurisdiction,
                                report.summary, client, on_event)
        route.append("Legal Researcher")

    # 3. Drafter writes the answer from the evidence gathered.
    on_event("Drafter writing the answer")
    draft = draft_answer(question, report, research, jurisdiction, client)
    route.append("Drafter")
    return CrewResult(draft, report, research, route)
