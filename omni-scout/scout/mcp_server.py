"""The MCP server: every way the crew can touch the internet, behind one protocol.

Why MCP and not plain Python functions? The agents never import `sources.py`.
They only see a list of tools the server advertises (name, description, JSON
schema) and send `call_tool` requests over a pipe. So:
  * the same server can be plugged into Claude Desktop, Claude Code or any
    other MCP client unchanged;
  * it runs in its own process, so a hung HTTP request can't freeze the UI;
  * swapping a free source for a paid one (Bright Data, SerpAPI, the YouTube
    API...) is a server-side change; the agents don't notice.

Run it standalone (it speaks MCP over stdin/stdout):

    uv run python -m scout.mcp_server
"""

import json
from typing import Any, Awaitable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from scout import sources

# The name shows up in MCP clients' tool pickers.
server = MCPServer("omni-scout-sources")


async def _run(pending: Awaitable[Any]) -> str:
    """Await a source call and turn its result into text for the model.

    Failures are re-raised as `ToolError`. MCP treats that as an *expected*
    failure and passes our message to the client, so the scout reads e.g.
    "HTTPStatusError: 429 Too Many Requests" and can adapt. Any other exception
    counts as a crash: MCP hides its text and the scout only sees
    "Error executing tool".
    """
    try:
        result = await pending
    except Exception as exc:
        raise ToolError(f"{type(exc).__name__}: {exc}") from exc
    if isinstance(result, str):  # read_page returns plain text
        return result or "Page had no readable text."
    if not result:
        return "No results. Try a broader or differently-worded query."
    # Search tools return lists of hits: compact JSON keeps titles/URLs unambiguous.
    return json.dumps(result, ensure_ascii=False, indent=1)


# Each tool below is: a typed signature (becomes the JSON schema the model sees),
# a docstring (becomes the description the model reads when deciding what to call),
# and one line of work. The docstrings are prompts: they tell the model what each
# platform is good for and how to phrase queries for it.
#
# We call `sources.web(...)`, not `web(...)` imported directly, so tests can
# swap `sources.web` for a fake without touching the network.


@server.tool()
async def search_web(query: str, max_results: int = 8) -> str:
    """General web search. Good for official pages, documentation, blogs and vendor sites.
    Use plain keyword queries, e.g. 'MCP specification transports'."""
    return await _run(sources.web(query, max_results))


@server.tool()
async def search_news(query: str, max_results: int = 8) -> str:
    """News search with publication dates and outlet names. Good for recent events,
    announcements and how coverage has changed over time."""
    return await _run(sources.news(query, max_results))


@server.tool()
async def search_videos(query: str, max_results: int = 8) -> str:
    """Video search (mostly YouTube): title, description, uploader, duration, views, date.
    Good for talks, tutorials, demos and reviews. You cannot watch the video; judge it
    by its metadata and description."""
    return await _run(sources.videos(query, max_results))


@server.tool()
async def search_reddit(query: str, max_results: int = 8) -> str:
    """Find Reddit threads about a topic. Good for practitioners' first-hand experience,
    complaints and recommendations. Open a promising thread with read_page to read comments."""
    return await _run(sources.reddit(query, max_results))


@server.tool()
async def search_hackernews(query: str, max_results: int = 8) -> str:
    """Search Hacker News stories, with points and comment counts as a signal of interest.
    Good for technical community reaction to launches, papers and tools. Use short queries."""
    return await _run(sources.hackernews(query, max_results))


@server.tool()
async def search_wikipedia(query: str, max_results: int = 5) -> str:
    """Search Wikipedia article titles and excerpts. Good for definitions, history and
    settled background. Open an article with read_page for the full text."""
    return await _run(sources.wikipedia(query, max_results))


@server.tool()
async def read_page(url: str) -> str:
    """Open a URL returned by another tool and return its readable text (truncated).
    Use sparingly: only for the 1-2 results that look most important."""
    return await _run(sources.page_text(url))


if __name__ == "__main__":
    # Default transport is stdio: the parent process (our toolbox) writes JSON-RPC
    # requests to our stdin and reads responses from our stdout.
    server.run()
