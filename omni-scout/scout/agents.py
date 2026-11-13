"""The three kinds of agent: planner, scouts, synthesizer.

    planner      one call   question -> which platforms, what to look for there
    scout  (xN)  a loop     search its platform via MCP tools until it has enough,
                            then hand in a structured ScoutReport
    synthesizer  one call   merge every scout's findings into one cited report

An "agent" here is just: a system prompt + a set of tools + a loop (or not).
No framework; every step is visible below.
"""

import asyncio
import json
from typing import Any, Callable

from pydantic import ValidationError

from scout.config import PLATFORM_HINTS, PLATFORM_TOOLS, SHARED_TOOLS, settings
from scout.llm import LLM
from scout.schemas import PlatformTask, ResearchPlan, ScoutReport
from scout.toolbox import Toolbox

# Progress callback: on_event(kind, data). The UI uses it to draw live status.
EventFn = Callable[[str, dict], None]


def _noop(kind: str, data: dict) -> None:
    pass


# =============================================================================
# 1. Planner
# =============================================================================

PLANNER_SYSTEM = """You plan research across several online platforms.
Given a question, decide which platforms are worth searching and what each platform's
scout should look for. Different platforms answer different parts of a question:
official facts on the web, recency in news, lived experience on Reddit, technical
reaction on Hacker News, demos and talks on video, background on Wikipedia.
Pick only platforms that add something distinct. Write queries in the style each
platform's search rewards (short keywords for Hacker News, natural phrasing for Reddit)."""


async def plan(llm: LLM, question: str) -> ResearchPlan:
    menu = "\n".join(f"- {name}: {hint}" for name, hint in PLATFORM_HINTS.items())
    prompt = (
        f"Question: {question}\n\n"
        f"Available platforms:\n{menu}\n\n"
        f"Choose at most {settings.max_platforms} platforms."
    )
    result = await llm.structured(
        model=settings.planner_model, effort=settings.planner_effort,
        system=PLANNER_SYSTEM, prompt=prompt, schema=ResearchPlan,
    )
    # Trust, but verify: the schema guarantees the shape, not that the planner
    # only named real platforms or respected the cap. Fix both in code.
    seen: set[str] = set()
    tasks = []
    for t in result.tasks:
        if t.platform in PLATFORM_TOOLS and t.platform not in seen:
            seen.add(t.platform)
            tasks.append(t)
    result.tasks = tasks[: settings.max_platforms]
    if not result.tasks:  # a planner that picked nothing usable still gets a web search
        result.tasks = [PlatformTask(platform="web", brief=question, queries=[question])]
    return result


# =============================================================================
# 2. Scout (the agentic loop)
# =============================================================================

SCOUT_SYSTEM = """You are the {platform} scout in a research crew. Other scouts cover other
platforms; stay on yours. Use your search tools to answer the brief, open the one or two
most important results with read_page if snippets aren't enough, then call
submit_findings exactly once.

Rules:
- Every finding must cite a URL that one of your tools actually returned. Never invent URLs.
- Separate facts from opinions (set `kind`). Reddit and Hacker News are mostly opinions.
- Keep dates when tools provide them; recency matters when sources disagree.
- Prefer 4-10 strong findings over many weak ones. Record what you couldn't find in `gaps`."""

# The scout's "I'm done" button: a client-side tool (not on the MCP server).
# Its input schema IS the ScoutReport schema, so when Claude calls it, the
# arguments are the structured report. The loop ends when it's called with
# arguments that validate.
def _submit_schema() -> dict:
    # The scout doesn't need to tell us its own platform; we fill that in.
    schema = ScoutReport.model_json_schema()
    schema["properties"].pop("platform")
    schema["required"].remove("platform")
    return schema


SUBMIT_TOOL = {
    "name": "submit_findings",
    "description": "Hand in your final report. Call exactly once, when you have enough evidence.",
    "input_schema": _submit_schema(),
}


async def run_scout(
    llm: LLM, toolbox: Toolbox, task: PlatformTask, question: str, on_event: EventFn = _noop,
) -> ScoutReport:
    platform = task.platform
    # Tool scoping: this scout sees only its platform's tools + read_page + submit.
    tools = toolbox.anthropic_tools(PLATFORM_TOOLS[platform] + SHARED_TOOLS) + [SUBMIT_TOOL]
    system = SCOUT_SYSTEM.format(platform=platform)
    messages: list[dict[str, Any]] = [{
        "role": "user",
        "content": (
            f"Overall research question: {question}\n\n"
            f"Your brief on {platform}: {task.brief}\n"
            f"Suggested starting queries: {json.dumps(task.queries)}"
        ),
    }]

    for turn in range(settings.max_scout_turns):
        response = await llm.step(
            model=settings.scout_model, effort=settings.scout_effort,
            system=system, messages=messages, tools=tools,
        )
        # Append the assistant turn *as returned* (thinking blocks included):
        # the API needs them back unchanged on the next request.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "pause_turn":  # server paused a long turn; resend to continue
            continue

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break  # Claude stopped without submitting; handled after the loop

        # Run every tool call from this turn concurrently. Claude often asks for
        # several searches at once; there's no reason to do them one by one.
        async def handle(block: Any) -> dict:
            if block.name == "submit_findings":
                try:
                    report = ScoutReport.model_validate({**block.input, "platform": platform})
                    return {"type": "tool_result", "tool_use_id": block.id, "content": "Received.", "_report": report}
                except ValidationError as exc:
                    # Invalid report: tell Claude exactly what's wrong and let it retry.
                    return {"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                            "content": f"Report invalid, fix and resubmit:\n{exc}"}
            on_event("tool", {"platform": platform, "tool": block.name, "args": block.input})
            text, is_error = await toolbox.call(block.name, block.input)
            return {"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": is_error}

        results = await asyncio.gather(*(handle(b) for b in tool_uses))

        # Did a valid submission arrive? Then this scout is finished.
        for r in results:
            if "_report" in r:
                on_event("scout_done", {"platform": platform, "report": r["_report"]})
                return r["_report"]

        # Otherwise, send every result back in ONE user message (splitting them
        # across messages teaches Claude to stop making parallel calls).
        content: list[dict] = [r for r in results]
        if turn == settings.max_scout_turns - 2:
            # One turn left: nudge it to wrap up rather than search again.
            content.append({"type": "text", "text": "Budget nearly spent: call submit_findings on your next turn."})
        messages.append({"role": "user", "content": content})

    # Out of turns (or stopped talking) without a valid submission. Return an
    # honest empty report rather than failing the whole crew.
    report = ScoutReport(
        platform=platform,
        summary="This scout ran out of turns before submitting findings.",
        findings=[],
        gaps=[task.brief],
    )
    on_event("scout_done", {"platform": platform, "report": report})
    return report


# =============================================================================
# 3. Synthesizer
# =============================================================================

SYNTH_SYSTEM = """You write the final research report by merging findings that several scouts
gathered on different platforms. Your value is the cross-checking:
- Lead with a direct answer to the question.
- Say where platforms agree (stronger evidence) and where they disagree, and why they
  might (official claims vs. user experience, old vs. new information).
- Distinguish facts from opinions. Never state an opinion as fact.
- Cite every claim with the numeric source IDs given, like [3] or [2][5]. Use only those IDs.
- Do not write a sources list at the end; it is appended automatically.
Use markdown headings: ## Answer, ## Where sources agree, ## Where they differ,
## By platform, ## Open questions."""


def number_sources(reports: list[ScoutReport]) -> tuple[list[dict], dict[str, int]]:
    """Give every distinct URL one global number, merging duplicates across scouts.

    If the web scout and the news scout both found the same article, it becomes
    ONE source cited by both: that overlap is itself evidence of importance.
    """
    sources: list[dict] = []
    ids: dict[str, int] = {}
    for rep in reports:
        for f in rep.findings:
            key = f.source_url.rstrip("/")
            if key not in ids:
                ids[key] = len(sources) + 1
                sources.append({"id": ids[key], "url": f.source_url, "title": f.source_title, "platforms": [rep.platform]})
            elif rep.platform not in sources[ids[key] - 1]["platforms"]:
                sources[ids[key] - 1]["platforms"].append(rep.platform)
    return sources, ids


async def synthesize(
    llm: LLM, question: str, reports: list[ScoutReport], on_text: Callable[[str], None] | None = None,
) -> tuple[str, list[dict]]:
    sources, ids = number_sources(reports)

    # Give Claude a compact, numbered digest instead of raw JSON.
    blocks = []
    for rep in reports:
        lines = [f"### {rep.platform}", f"Scout summary: {rep.summary}"]
        for f in rep.findings:
            date = f" ({f.published})" if f.published else ""
            lines.append(f"- [{ids[f.source_url.rstrip('/')]}] ({f.kind}){date} {f.claim}")
        if rep.gaps:
            lines.append("Gaps: " + "; ".join(rep.gaps))
        blocks.append("\n".join(lines))
    prompt = f"Question: {question}\n\nScout findings:\n\n" + "\n\n".join(blocks)

    body = await llm.stream_text(
        model=settings.synth_model, effort=settings.synth_effort,
        system=SYNTH_SYSTEM, prompt=prompt, on_text=on_text,
    )
    # The sources list is built by code from tool-returned URLs, never written
    # by the model, so it can't contain a hallucinated link.
    refs = "\n".join(f"{s['id']}. [{s['title'] or s['url']}]({s['url']}) · {', '.join(s['platforms'])}" for s in sources)
    return f"{body.strip()}\n\n## Sources\n\n{refs}\n", sources
