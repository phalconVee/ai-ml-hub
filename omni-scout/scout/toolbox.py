"""The MCP client side: connect to the server, hand its tools to Claude, run calls.

Claude doesn't speak MCP here; it speaks the Messages API's tool-use format.
The toolbox translates between the two:

    MCP tool  (name, description, input_schema)  -->  Anthropic tool dict
    Claude's tool_use block (name, input)        -->  MCP call_tool request
    MCP CallToolResult (content, is_error)       -->  tool_result block for Claude
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from mcp import Client, StdioServerParameters

from scout.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Toolbox:
    """A live MCP connection plus the tool definitions it advertises."""

    def __init__(self, client: Client, tools: list[Any]):
        self.client = client
        # Keep the raw MCP tool objects keyed by name.
        self._tools = {t.name: t for t in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def anthropic_tools(self, allowed: list[str]) -> list[dict]:
        """Convert the allowed subset of MCP tools into Messages API tool definitions.

        This is the whole "tool scoping" trick behind the multi-agent design:
        each scout gets *only* its platform's tools, so the Reddit scout can't
        wander off and do a web search, and its prompt stays short.
        """
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.input_schema,  # MCP already gives us JSON Schema
            }
            for name, t in self._tools.items()
            if name in allowed
        ]

    async def call(self, name: str, args: dict) -> tuple[str, bool]:
        """Run one tool on the MCP server. Returns (text for the model, is_error).

        Errors are *returned*, not raised: a failed search is information the
        scout can react to ("try another query"), not a reason to crash the crew.
        """
        if name not in self._tools:
            return f"Unknown tool {name!r}. Available: {self.names}", True
        try:
            result = await self.client.call_tool(name, args)
        except Exception as exc:  # transport-level failure (server died, timeout...)
            return f"Tool {name} failed: {exc}", True
        # A result is a list of content blocks; ours are always text.
        text = "\n".join(getattr(block, "text", "") for block in result.content)
        # Long tool output eats context and money; clip it.
        if len(text) > settings.max_tool_chars:
            text = text[: settings.max_tool_chars] + "\n...[truncated]"
        return text, bool(result.is_error)


@asynccontextmanager
async def open_toolbox(server: Any = None) -> AsyncIterator[Toolbox]:
    """Connect to the MCP server for the duration of a `with` block.

    By default this launches `python -m scout.mcp_server` as a subprocess and
    talks to it over stdio, exactly as Claude Desktop would. Tests pass an
    in-process server object instead (no subprocess, no network).
    """
    if server is None:
        # sys.executable = the same Python (and virtualenv) we are running in.
        # cwd = the project root, so `scout` is importable whatever folder
        # Streamlit was launched from.
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "scout.mcp_server"],
            cwd=str(PROJECT_ROOT),
        )
    async with Client(server) as client:
        listing = await client.list_tools()
        yield Toolbox(client, listing.tools)
