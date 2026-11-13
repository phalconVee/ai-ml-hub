"""Wire the crew together: plan -> scouts in parallel -> synthesize.

This is the one function the UI (and the CLI, and the tests) call.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from scout.agents import EventFn, _noop, plan, run_scout, synthesize
from scout.llm import LLM
from scout.schemas import ResearchPlan, ScoutReport
from scout.toolbox import open_toolbox


@dataclass
class ResearchResult:
    question: str
    plan: ResearchPlan
    reports: list[ScoutReport]
    report_md: str
    sources: list[dict] = field(default_factory=list)


async def research(
    question: str,
    *,
    llm: LLM | None = None,
    mcp_server: Any = None,           # None = launch the real MCP server subprocess
    on_event: EventFn = _noop,
    on_text: Callable[[str], None] | None = None,
) -> ResearchResult:
    llm = llm or LLM()

    # 1. Plan (no tools needed yet).
    research_plan = await plan(llm, question)
    on_event("plan", {"plan": research_plan})

    # One MCP connection, shared by every scout. MCP requests carry IDs, so
    # several scouts can have calls in flight on the same connection at once.
    async with open_toolbox(mcp_server) as toolbox:
        # 2. All scouts at once. `return_exceptions=True` means one crashed
        #    scout (say, a refusal) doesn't throw away the others' work.
        outcomes = await asyncio.gather(
            *(run_scout(llm, toolbox, t, research_plan.restated_question, on_event) for t in research_plan.tasks),
            return_exceptions=True,
        )

    reports: list[ScoutReport] = []
    for task, out in zip(research_plan.tasks, outcomes):
        if isinstance(out, BaseException):
            on_event("scout_failed", {"platform": task.platform, "error": str(out)})
            reports.append(ScoutReport(platform=task.platform, summary=f"Scout failed: {out}", findings=[], gaps=[task.brief]))
        else:
            reports.append(out)

    # 3. Merge everything into one cited report.
    on_event("synthesizing", {})
    report_md, sources = await synthesize(llm, research_plan.restated_question, reports, on_text)
    return ResearchResult(question, research_plan, reports, report_md, sources)
