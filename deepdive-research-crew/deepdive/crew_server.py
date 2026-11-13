"""MCP in the other direction: the whole crew, offered as ONE MCP tool.

`tools_server.py` gives tools *to* our agents. This file turns the finished
crew into a tool *for other agents*: add it to Claude Desktop or Claude Code
and they can call `deep_research("...")` and get a cited report back.

    uv run python -m deepdive.crew_server

Inside, a call to `deep_research` starts its own tools server as a
subprocess, so this is MCP all the way down: client -> crew_server -> crew
-> tools_server. A run takes minutes, so we report progress as we go.
"""

from __future__ import annotations

import asyncio

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from deepdive.agent import Event
from deepdive.config import DEPTH_PRESETS
from deepdive.crew import run_research

server = MCPServer(
    name="deepdive-crew",
    instructions="Multi-agent deep web research. Slow (minutes) but thorough, with citations.",
)

# Stages in the order the crew reports them, used as a progress bar.
_STAGES = ["Planning", "Sending", "Cross-checking", "Writing", "Done"]


@server.tool()
async def deep_research(question: str, ctx: Context, depth: str = "standard") -> str:
    """Research a question on the web with a team of agents and return a cited Markdown report.

    depth: "quick" (2 angles), "standard" (3 angles) or "deep" (5 angles, more reading).
    Takes from about one to several minutes.
    """
    if depth not in DEPTH_PRESETS:
        raise ToolError(f"depth must be one of {sorted(DEPTH_PRESETS)}")

    pending: set[asyncio.Task] = set()

    def on_event(e: Event) -> None:
        # The crew calls this synchronously, but sending an MCP progress
        # notification is async, so we schedule it as a background task
        # (and keep a reference so it isn't garbage-collected mid-flight).
        if e.kind != "stage":
            return
        step = next((i for i, s in enumerate(_STAGES) if e.text.startswith(s)), 0)
        task = asyncio.create_task(ctx.report_progress(step + 1, len(_STAGES), e.text))
        pending.add(task)
        task.add_done_callback(pending.discard)

    try:
        report = await run_research(question, DEPTH_PRESETS[depth], on_event=on_event)
    except Exception as exc:
        raise ToolError(f"research failed: {type(exc).__name__}: {exc}") from exc
    return report.report


if __name__ == "__main__":
    server.run("stdio")
