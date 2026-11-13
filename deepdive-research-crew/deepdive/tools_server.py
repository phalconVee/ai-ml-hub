"""The MCP tools server: web search and page reading, spoken over MCP.

Run it on its own and any MCP client (Claude Desktop, Claude Code, the
MCP Inspector, or our own crew) can use it:

    uv run python -m deepdive.tools_server

Why put the tools behind MCP at all, when the crew could just import
`web_tools`? Because MCP makes the *tools* a separate, swappable process:

  - the crew discovers tools at runtime (`list_tools`), so adding a tool here
    makes it available to every agent with no change to the agent code;
  - any other MCP-speaking app can reuse the same server;
  - you could swap this server for someone else's (a Brave Search MCP, a
    company-intranet MCP) and the crew would not notice.

`MCPServer` is mcp 2.x's name for what mcp 1.x called `FastMCP`.
"""

from __future__ import annotations

import json

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from deepdive import web_tools

server = MCPServer(
    name="deepdive-web",
    instructions="Web research tools: search the web, then read the most promising pages.",
)


# The decorator turns each function into an MCP tool. Its *name* is the
# function name, its *description* is the docstring, and its *input schema*
# is generated from the type hints. Those three things are exactly what the
# model sees, so the docstrings are written for the model, not for us.


@server.tool()
async def web_search(query: str, max_results: int = 6) -> str:
    """Search the web. Returns a JSON list of results, each with title, url and snippet.

    Use short, specific queries (like you would type into a search engine).
    Snippets are previews only: open promising results with read_page before
    relying on them.
    """
    max_results = max(1, min(max_results, 10))
    # The search libraries are blocking; run them in a worker thread so the
    # server can serve several agents' calls at the same time.
    try:
        hits = await anyio.to_thread.run_sync(web_tools.search, query, max_results)
    except Exception as exc:
        # ToolError = "a failure I expected": its message reaches the model as
        # an is_error result. Any other exception is treated as a crash and the
        # model only sees a generic "Error executing tool", which it can't act on.
        raise ToolError(f"search failed: {exc}") from exc
    return json.dumps([h.to_dict() for h in hits], ensure_ascii=False)


@server.tool()
async def read_page(url: str) -> str:
    """Fetch a web page and return its readable text as JSON: url, title, text, truncated.

    Only http(s) URLs on the public internet. Long pages are cut off; if
    `truncated` is true the page continues beyond the text you got.
    """
    try:
        page = await anyio.to_thread.run_sync(web_tools.read_page, url)
    except Exception as exc:
        raise ToolError(f"could not read {url}: {exc}") from exc
    return json.dumps(page, ensure_ascii=False)


if __name__ == "__main__":
    # stdio transport: the client launches us as a subprocess and we talk
    # JSON-RPC over stdin/stdout. (Never print() to stdout in here: it would
    # corrupt the protocol stream. Logs go to stderr.)
    server.run("stdio")
