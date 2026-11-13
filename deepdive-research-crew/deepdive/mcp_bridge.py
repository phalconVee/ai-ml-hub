"""The bridge between MCP and the Claude API.

MCP and the Claude Messages API both describe tools as "name + description +
JSON Schema for the input", but with slightly different field names. This
module does the two small translations the crew needs:

    MCP Tool            ->  Claude tool definition      (so Claude can see it)
    Claude tool_use     ->  MCP call_tool               (so the call runs)
    MCP CallToolResult  ->  Claude tool_result content  (so Claude sees the answer)

It also starts the tools server as a subprocess and keeps one connection
open for the whole research run. All agents share that one connection: MCP
is JSON-RPC, every request carries an id, so concurrent calls don't mix.

(The Anthropic SDK ships a helper, `anthropic.lib.tools.mcp.async_mcp_tool`,
that does the first translation for its beta tool runner. We do it by hand
here because it is five lines and worth seeing.)
"""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Environment variables the tools server needs. The stdio client starts the
# subprocess with a minimal, safe environment (PATH, HOME, ...), not a copy of
# ours, so anything the server needs must be passed through explicitly.
_PASS_THROUGH_ENV = ("DEEPDIVE_SEARCH_BACKEND", "TAVILY_API_KEY", "DEEPDIVE_PAGE_CHAR_LIMIT", "DEEPDIVE_SEARCH_RESULTS")


@dataclass
class ToolCallRecord:
    """One tool call, kept for the UI's activity log and the sources list."""

    agent: str
    name: str
    arguments: dict
    is_error: bool
    preview: str


@dataclass
class ToolBox:
    """A live MCP connection plus the tools it offers, in Claude's format."""

    client: Client
    tools: list[dict]
    calls: list[ToolCallRecord] = field(default_factory=list)

    async def call(self, agent: str, name: str, arguments: dict) -> tuple[str, bool]:
        """Run one tool over MCP. Returns (text_for_claude, is_error).

        Never raises for tool failures: a failed search is information the
        model can act on ("try another query"), so it goes back as an error
        result instead of crashing the run.
        """
        try:
            result = await self.client.call_tool(name, arguments)
            # An MCP result is a list of content blocks (text, image, ...).
            # Our server only returns text, so join the text blocks.
            text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
            is_error = bool(result.is_error)
        except Exception as exc:  # transport failure, timeout, bad arguments
            text, is_error = f"{type(exc).__name__}: {exc}", True

        self.calls.append(ToolCallRecord(agent, name, arguments, is_error, text[:300]))
        return text, is_error


def mcp_tool_to_claude(tool) -> dict:
    """Translate one MCP `Tool` into a Claude API tool definition."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.input_schema,  # both sides speak JSON Schema
    }


def tools_server_params() -> StdioServerParameters:
    """How to launch our tools server: the same Python, as a module."""
    env = {k: os.environ[k] for k in _PASS_THROUGH_ENV if k in os.environ}
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "deepdive.tools_server"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )


@asynccontextmanager
async def open_toolbox(server: Any = None) -> AsyncIterator[ToolBox]:
    """Start the MCP server, discover its tools, and hand back a ToolBox.

    `server` is anything `mcp.Client` accepts: launch parameters for a
    subprocess (the default), a URL of a remote MCP server, or - in tests -
    an `MCPServer` object to talk to in-process.

    Usage:
        async with open_toolbox() as box:
            text, err = await box.call("scout-1", "web_search", {"query": "..."})

    Leaving the `async with` closes the connection and stops the subprocess.
    """
    async with Client(server or tools_server_params()) as client:
        listing = await client.list_tools()
        yield ToolBox(client=client, tools=[mcp_tool_to_claude(t) for t in listing.tools])
