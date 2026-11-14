"""The MCP client: start the server, list its tools, call them.

This is the bridge between two vocabularies:

    MCP tool (name, description, input_schema)  ->  Claude tool definition (same three fields)
    Claude tool_use block (name, input)         ->  MCP call_tool(name, arguments)
    MCP CallToolResult (content, is_error)      ->  Claude tool_result block

Claude never speaks MCP here, and the server never knows Claude is involved.
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp import Client, StdioServerParameters

from soundcheck.config import PROJECT_ROOT, settings

# Environment variable prefixes passed through to the server process.
FORWARDED_ENV = ("SOUNDCHECK_", "ANTHROPIC_", "ASSEMBLYAI_", "HF_")


class Toolbox:
    """A live MCP session plus the tool list the server advertised."""

    def __init__(self, client: Client, tools: list[Any], instructions: str | None):
        self.client = client
        self.tools = {t.name: t for t in tools}
        # The server's own usage notes (from MCPServer(instructions=...)).
        self.instructions = instructions or ""

    def claude_tools(self) -> list[dict]:
        """MCP already describes tools in JSON Schema, which is exactly what the
        Messages API wants, so conversion is a rename of one key."""
        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in self.tools.values()
        ]

    async def call(self, name: str, args: dict) -> tuple[str, bool]:
        """Run a tool. Returns (text, is_error); never raises.

        A failed tool call is information for the model ("file not found",
        "call transcribe_audio first"), so it goes back as an error result
        rather than crashing the conversation.
        """
        if name not in self.tools:
            return f"Unknown tool {name!r}. Available: {list(self.tools)}", True
        try:
            result = await self.client.call_tool(name, args)
        except Exception as exc:  # transport-level: server crashed, pipe closed, timeout
            return f"Tool {name} failed to run: {exc}", True
        text = "\n".join(getattr(block, "text", "") for block in result.content)
        if len(text) > settings.max_tool_chars:
            text = text[: settings.max_tool_chars] + "\n...[truncated]"
        return text, bool(result.is_error)


@asynccontextmanager
async def open_toolbox(server: Any = None) -> AsyncIterator[Toolbox]:
    """Connect for the duration of an `async with` block.

    Default: launch `python -m soundcheck.server` as a child process and talk
    over its stdin/stdout, exactly like Claude Desktop does. Tests pass the
    server *object* instead, which the SDK connects to in-memory.
    """
    if server is None:
        server = StdioServerParameters(
            command=sys.executable,      # same interpreter + virtualenv as us
            args=["-m", "soundcheck.server"],
            cwd=str(PROJECT_ROOT),       # so `soundcheck` imports and .env loads wherever Streamlit was started
            # The SDK deliberately gives a child server only a minimal
            # environment (PATH, HOME...), not ours. Forward just the settings
            # this server reads, so `export SOUNDCHECK_WHISPER_MODEL=small`
            # in your shell reaches it too (.env is read by the server itself).
            env={k: v for k, v in os.environ.items() if k.startswith(FORWARDED_ENV)},
        )
    async with Client(server) as client:
        listing = await client.list_tools()
        yield Toolbox(client, listing.tools, client.instructions)
