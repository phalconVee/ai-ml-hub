"""The browser MCP server: BrowserSession's methods, exposed as MCP tools.

Why write our own instead of using an existing browser MCP server?
  * Learning: the whole server is this file plus browser.py, about 300 lines.
  * Guard rails belong in the server. The domain policy and read-only mode are
    enforced *here*, below the model, so no prompt can talk its way past them.
  * No paid browser cloud: it drives a local Chromium via Playwright.

Any MCP client can use it (Claude Desktop, Claude Code, the MCP Inspector):

    uv run python -m webpilot.mcp_server

It reads its guard rails from env vars at startup (see config.Settings.server_env),
so the client that launches it decides the policy, not the model calling it.
"""

from contextlib import asynccontextmanager

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image

from webpilot.browser import BrowserActionError, BrowserSession
from webpilot.config import Settings
from webpilot.guards import DomainPolicy


def build_server(settings: Settings) -> MCPServer:
    """Create a server with its own browser and the given guard rails.

    A factory (rather than one module-level server) lets tests build servers
    with different policies, in-process, side by side.
    """
    policy = DomainPolicy(allow=settings.allow, deny=settings.deny)
    browser = BrowserSession(policy, headless=settings.headless, nav_timeout_ms=settings.nav_timeout_ms)

    @asynccontextmanager
    async def lifespan(_server):
        # Chromium starts lazily on the first tool call; we only need to make
        # sure it is shut down when the client disconnects.
        try:
            yield {}
        finally:
            await browser.close()

    server = MCPServer(
        "webpilot-browser",
        instructions=(
            "Browse with goto/read_page/click. Element ids come from the latest read_page. "
            + policy.describe()
            + (" Read-only mode: typing and form submission are disabled." if settings.read_only else "")
        ),
        lifespan=lifespan,
    )

    async def guarded(action):
        """Await a browser action; turn expected failures into ToolErrors.

        MCP passes a ToolError's message through to the client, so Claude reads
        e.g. "Blocked by domain policy: ..." and can adapt. Any other exception
        is treated as a crash and its text is hidden, so unexpected Playwright
        errors are also converted, with just their type and first line.
        """
        try:
            return await action
        except BrowserActionError as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
            raise ToolError(f"{type(exc).__name__}: {first_line[:300]}") from exc

    # Each tool: typed signature -> the JSON schema Claude sees; docstring -> the
    # description Claude reads when deciding what to call. Docstrings are prompts.

    @server.tool()
    async def goto(url: str) -> str:
        """Open a URL in the browser tab. Prefer direct URLs (e.g. a site's search
        results page with the query in the URL) over clicking through menus."""
        return await guarded(browser.goto(url))

    @server.tool()
    async def read_page() -> str:
        """Read the current page: its URL, title, a numbered list of clickable or
        fillable elements, and the visible text. Call after every navigation;
        element ids are only valid until the page changes."""
        return await guarded(browser.read_page(settings.max_page_chars, settings.max_elements))

    @server.tool()
    async def click(element_id: int) -> str:
        """Click a link or button by its id from the latest read_page. Cannot click
        buttons that submit a form (use submit_form for that)."""
        return await guarded(browser.click(element_id))

    @server.tool()
    async def scroll(direction: str = "down") -> str:
        """Scroll the page 'down' or 'up'. Only needed to change what a screenshot
        shows: read_page already returns the full page text."""
        return await guarded(browser.scroll(direction))

    @server.tool()
    async def go_back() -> str:
        """Go back to the previous page in this tab's history."""
        return await guarded(browser.go_back())

    @server.tool()
    async def screenshot() -> Image:
        """Take a screenshot of the visible part of the page. Use when layout or
        images matter; read_page is cheaper for text."""
        data = await guarded(browser.screenshot())
        return Image(data=data, format="jpeg")

    @server.tool()
    async def describe_form(element_id: int) -> dict:
        """Show the form that an element belongs to: where it is sent (action,
        method) and the current field values (passwords masked). Read-only."""
        return await guarded(browser.describe_form(element_id))

    # Write tools exist only when read-only mode is off. Not registering them
    # (rather than registering and refusing) means Claude never even sees them.
    if not settings.read_only:

        @server.tool()
        async def type_text(element_id: int, text: str) -> str:
            """Type text into an input or textarea (replacing what's there). This
            does NOT submit anything; use submit_form for that."""
            return await guarded(browser.type_text(element_id, text))

        @server.tool()
        async def submit_form(element_id: int) -> str:
            """Submit the form that contains this element (a field or its submit
            button). The user is asked to approve every submission first; if
            they decline, do not try to submit it another way."""
            return await guarded(browser.submit_form(element_id))

    return server


def main() -> None:
    # Settings() reads WEBPILOT_* env vars, which the launching client set.
    build_server(Settings()).run()  # stdio transport by default


if __name__ == "__main__":
    main()
