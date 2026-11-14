"""Agent 3, the Drafter: write the plain-English, cited answer.

It sees only evidence, never the raw documents:
  • the clauses the Clause Finder verified, and
  • the web passages the Legal Researcher quoted,
both as `search_result` blocks with citations enabled. Claude then attaches
citations natively: each sentence of the answer comes back with the exact
passage it relied on (clause id or URL, title, quoted text). No "[1]" markers
written by the model, no regex over the answer.

We don't give the Drafter any tools. Its whole job is to read and write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .clause_finder import ClauseReport
from .config import NOT_LEGAL_ADVICE
from .llm import call_claude
from .researcher import Research

SYSTEM = """You are an experienced attorney explaining documents to a client \
who is not a lawyer. Answer the client's question using ONLY the evidence \
provided: clauses from their documents and passages from web research.

Write in plain English:
- Lead with the direct answer in one or two sentences.
- Then explain what the relevant clauses say and what they mean in practice. \
Quote short key phrases (deadlines, amounts, "shall"/"may") where wording matters.
- Keep what the DOCUMENT says separate from what the LAW says; never present \
web research as if it were part of the contract.
- Point out ambiguities, missing terms, and anything that depends on facts \
you don't have.
- If the evidence does not answer the question, say so plainly. Do not fill \
gaps from general knowledge.
- Finish with a short "Questions to ask a lawyer" list (2-3 items).
Do not add a legal-advice disclaimer; the application adds one."""


@dataclass
class Citation:
    number: int       # the [n] shown in the answer
    kind: str         # "clause" or "web"
    source: str       # clause id ("lease.pdf#p3-c2") or URL
    title: str        # "lease.pdf, p.3, 4.2 Late Payment" or page title
    quote: str        # the exact passage the claim rests on


@dataclass
class Draft:
    segments: list[tuple[str, list[int]]] = field(default_factory=list)  # (text, [citation numbers])
    citations: list[Citation] = field(default_factory=list)
    stop_reason: str = ""

    def body_markdown(self) -> str:
        """The answer with [n] after each cited span."""
        return "".join(text + "".join(f" [{n}]" for n in nums)
                       for text, nums in self.segments).strip()

    def markdown(self) -> str:
        """Notice first, then the answer. The notice is ours, not the model's."""
        return f"> {NOT_LEGAL_ADVICE}\n\n{self.body_markdown()}"


def clause_blocks(report: ClauseReport) -> list[dict]:
    """Each verified clause becomes one citable search_result block."""
    return [{
        "type": "search_result",
        "source": c.id,
        "title": c.title,
        "content": [{"type": "text", "text": c.text}],
        "citations": {"enabled": True},
    } for c in report.clauses]


def build_content(question: str, report: ClauseReport, research: Research | None,
                  jurisdiction: str) -> list[dict]:
    """The Drafter's one user message: evidence first, then the question."""
    content: list[dict] = []
    content += clause_blocks(report)
    if research is not None:
        content += research.as_search_results()
    lines = [f"Client's question: {question}",
             f"Jurisdiction: {jurisdiction or 'not specified'}",
             f"Paralegal's note on the documents (coverage: {report.coverage}): {report.summary}"]
    if not report.clauses:
        lines.append("No relevant clauses were found in the client's documents.")
    if research is not None and research.notes:
        lines.append(f"Research memo (context only; cite the web passages above, not this memo):\n{research.notes}")
    content.append({"type": "text", "text": "\n\n".join(lines)})
    return content


def collect_citations(blocks: list, draft: Draft) -> None:
    """Walk the answer blocks and give each distinct cited passage a number."""
    numbers: dict[tuple[str, str], int] = {}   # (source, quote) → number
    for b in blocks:
        if getattr(b, "type", "") != "text":
            continue    # thinking blocks, fallback markers, etc.
        nums: list[int] = []
        for c in getattr(b, "citations", None) or []:
            if getattr(c, "type", "") != "search_result_location":
                continue
            key = (c.source, c.cited_text)
            if key not in numbers:
                numbers[key] = len(numbers) + 1
                kind = "web" if c.source.startswith(("http://", "https://")) else "clause"
                draft.citations.append(Citation(numbers[key], kind, c.source,
                                                c.title or c.source, c.cited_text))
            if numbers[key] not in nums:
                nums.append(numbers[key])
        draft.segments.append((b.text, nums))


def draft_answer(question: str, report: ClauseReport, research: Research | None,
                 jurisdiction: str, client: Any) -> Draft:
    messages = [{"role": "user", "content": build_content(question, report, research, jurisdiction)}]
    response = call_claude(client, system=SYSTEM, messages=messages)
    draft = Draft(stop_reason=response.stop_reason or "")
    if response.stop_reason == "refusal":
        draft.segments.append(("The model declined to answer this question.", []))
        return draft
    collect_citations(response.content, draft)
    if response.stop_reason == "max_tokens":
        draft.segments.append(("\n\n_(Answer cut off at the length limit.)_", []))
    return draft
