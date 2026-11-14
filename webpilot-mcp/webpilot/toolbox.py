"""The MCP client side: launch the browser server, list its tools, call them.

Claude speaks the Messages API's tool-use format, not MCP. The toolbox translates:

    MCP tool (name, description, input_schema)  ->  Anthropic tool definition
    Claude's tool_use (name, input)             ->  MCP call_tool
    MCP result (text + image content, is_error) ->  ToolOutput -> tool_result blocks
"""

import base64
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from mcp import Client, StdioServerParameters

from webpilot.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ToolOutput:
    text: str
    images: list[tuple[str, str]] = field(default_factory=list)  # (mime type, base64 data)
    is_error: bool = False

    def to_blocks(self) -> list[dict]:
        """Content for a Messages API tool_result: text first, then any images."""
        blocks: list[dict] = [{"type": "text", "text": self.text or "(no output)"}]
        for mime, data in self.images:
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
        return blocks


class Toolbox:
    """A live MCP connection plus the tool definitions it advertises."""

    def __init__(self, client: Client, tools: list[Any]):
        self.client = client
        self._tools = {t.name: t for t in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def anthropic_tools(self) -> list[dict]:
        """Every tool the server registered, in Messages API format.

        No filtering needed for read-only mode: the server simply doesn't
        register write tools then, so they never reach this list.
        """
        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in self._tools.values()
        ]

    async def call(self, name: str, args: dict) -> ToolOutput:
        """Run one tool. Errors are returned (is_error=True), never raised:
        a blocked domain or a stale element id is information for Claude."""
        if name not in self._tools:
            return ToolOutput(f"Unknown tool {name!r}. Available: {self.names}", is_error=True)
        try:
            result = await self.client.call_tool(name, args)
        except Exception as exc:  # transport failure: server died, timeout...
            return ToolOutput(f"Tool {name} failed: {exc}", is_error=True)
        texts, images = [], []
        for block in result.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "image":
                images.append((block.mime_type, block.data))  # data is already base64
        return ToolOutput("\n".join(texts), images, bool(result.is_error))

    async def screenshot_bytes(self) -> bytes | None:
        """A screenshot for the UI (not sent to Claude). None if it fails."""
        out = await self.call("screenshot", {})
        if out.is_error or not out.images:
            return None
        return base64.b64decode(out.images[0][1])


@asynccontextmanager
async def open_toolbox(settings: Settings, server: Any = None) -> AsyncIterator[Toolbox]:
    """Connect to the browser MCP server for the duration of a `with` block.

    By default this launches `python -m webpilot.mcp_server` as a subprocess
    over stdio, passing the guard rails as env vars. Tests pass an in-process
    server (from build_server) instead.
    """
    if server is None:
        server = StdioServerParameters(
            command=sys.executable,  # same Python + virtualenv as us
            args=["-m", "webpilot.mcp_server"],
            cwd=str(PROJECT_ROOT),
            # The subprocess gets our env (PATH, Playwright cache...) plus the policy.
            env={**os.environ, **settings.server_env()},
        )
    async with Client(server) as client:
        listing = await client.list_tools()
        yield Toolbox(client, listing.tools)
