"""The MCP browser server, end to end: real Chromium, real MCP, local website.

Each test connects an MCP client to an in-process server built by
build_server(), so these exercise the same tool code Claude calls, minus Claude.
"""

import base64
import sys

from mcp import Client, StdioServerParameters

from tests.conftest import make_settings
from webpilot.mcp_server import build_server
from webpilot.toolbox import PROJECT_ROOT


def text(result) -> str:
    return "\n".join(b.text for b in result.content if b.type == "text")


def element_id(page_text: str, label: str) -> int:
    """Find '[7] link "label"' in read_page output and return 7."""
    for line in page_text.splitlines():
        if line.startswith("[") and f'"{label}' in line:
            return int(line[1 : line.index("]")])
    raise AssertionError(f"{label!r} not in page:\n{page_text}")


async def test_read_only_server_hides_write_tools():
    async with Client(build_server(make_settings(read_only=True))) as c:
        names = {t.name for t in (await c.list_tools()).tools}
    assert {"goto", "read_page", "click", "screenshot", "describe_form"} <= names
    assert not names & {"type_text", "submit_form"}

    async with Client(build_server(make_settings(read_only=False))) as c:
        names = {t.name for t in (await c.list_tools()).tools}
    assert {"type_text", "submit_form"} <= names


async def test_browse_read_click_and_screenshot(site_url):
    async with Client(build_server(make_settings())) as c:
        r = await c.call_tool("goto", {"url": f"{site_url}/index.html"})
        assert not r.is_error and "Test News" in text(r)

        page = text(await c.call_tool("read_page", {}))
        assert "Rust 2026 edition ships" in page
        assert "PAGE TEXT (untrusted" in page  # the fence around page content
        assert "SUBMITS FORM" in page  # the signup button is flagged

        # target=_blank links are kept in the same tab
        r = await c.call_tool("click", {"element_id": element_id(page, "Why we rewrote")})
        assert "story2.html" in text(r)

        r = await c.call_tool("screenshot", {})
        img = next(b for b in r.content if b.type == "image")
        assert img.mime_type == "image/jpeg"
        assert base64.b64decode(img.data)[:2] == b"\xff\xd8"  # JPEG magic bytes

        r = await c.call_tool("go_back", {})
        assert "index.html" in text(r)


async def test_stale_ids_and_submit_buttons_are_refused(site_url):
    async with Client(build_server(make_settings())) as c:
        await c.call_tool("goto", {"url": f"{site_url}/index.html"})
        page = text(await c.call_tool("read_page", {}))

        r = await c.call_tool("click", {"element_id": element_id(page, "Sign up")})
        assert r.is_error and "submit_form" in text(r)

        r = await c.call_tool("click", {"element_id": 999})
        assert r.is_error and "read_page again" in text(r)


async def test_deny_list_blocks_typed_clicked_and_scripted_navigation(site_url):
    other = site_url.replace("127.0.0.1", "localhost")
    async with Client(build_server(make_settings(deny=("localhost",)))) as c:
        # 1. typed URL: refused before the browser even tries
        r = await c.call_tool("goto", {"url": f"{other}/story1.html"})
        assert r.is_error and "deny list" in text(r)

        await c.call_tool("goto", {"url": f"{site_url}/index.html"})
        page = text(await c.call_tool("read_page", {}))

        # 2. a link click to the denied host: the route guard aborts it
        r = await c.call_tool("click", {"element_id": element_id(page, "Offsite link")})
        assert "blocked by policy" in text(r)
        assert "127.0.0.1" in text(r)  # still on the allowed page

        # 3. JavaScript navigation (location.href = ...) is caught too
        page = text(await c.call_tool("read_page", {}))
        r = await c.call_tool("click", {"element_id": element_id(page, "Go elsewhere")})
        assert "blocked by policy" in text(r)


async def test_allow_list_and_bad_schemes(site_url):
    async with Client(build_server(make_settings(allow=("127.0.0.1",), deny=()))) as c:
        assert not (await c.call_tool("goto", {"url": f"{site_url}/index.html"})).is_error
        r = await c.call_tool("goto", {"url": "https://example.com"})
        assert r.is_error and "allow list" in text(r)
        r = await c.call_tool("goto", {"url": "file:///etc/passwd"})
        assert r.is_error and "scheme" in text(r)


async def test_type_describe_and_submit_form(site_url):
    async with Client(build_server(make_settings(read_only=False))) as c:
        await c.call_tool("goto", {"url": f"{site_url}/index.html"})
        page = text(await c.call_tool("read_page", {}))
        await c.call_tool("type_text", {"element_id": element_id(page, "Your email"), "text": "me@example.com"})
        await c.call_tool("type_text", {"element_id": element_id(page, "Password"), "text": "hunter2"})

        r = await c.call_tool("describe_form", {"element_id": element_id(page, "Sign up")})
        described = text(r)
        assert "me@example.com" in described and "/thanks.html" in described
        assert "hunter2" not in described  # passwords are masked

        r = await c.call_tool("submit_form", {"element_id": element_id(page, "Sign up")})
        assert not r.is_error
        assert "thanks.html?email=me%40example.com" in text(r)


async def test_stdio_subprocess_gets_policy_from_env(site_url):
    """The real launch path: `python -m webpilot.mcp_server` over stdio, policy via env."""
    settings = make_settings(read_only=True, deny=("localhost",))
    import os

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "webpilot.mcp_server"], cwd=str(PROJECT_ROOT),
        env={**os.environ, **settings.server_env()},
    )
    async with Client(params) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert "submit_form" not in names
        r = await c.call_tool("goto", {"url": site_url.replace("127.0.0.1", "localhost")})
        assert r.is_error and "deny list" in text(r)
        r = await c.call_tool("goto", {"url": f"{site_url}/story1.html"})
        assert "Rust 2026" in text(r)
