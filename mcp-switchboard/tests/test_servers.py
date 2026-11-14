"""The example servers, called through a real MCP client (in-process: no subprocess, no network).

`Client(server_object)` runs the full MCP request/response path in memory, so
these tests exercise argument validation, ToolError handling and annotations
exactly as a remote client would see them.
"""

from pathlib import Path

import pytest
from mcp import Client

from example_servers import calc_clock, sandbox_files, web_reader


def text(result) -> str:
    return "\n".join(c.text for c in result.content)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "hello.txt").write_text("hi there", encoding="utf-8")
    monkeypatch.setattr(sandbox_files, "ROOT", root.resolve())
    return root


async def test_sandbox_read_write_and_annotations(sandbox):
    async with Client(sandbox_files.server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
        assert tools["read_file"].annotations.read_only_hint is True
        assert tools["write_file"].annotations.read_only_hint is False
        assert tools["delete_path"].annotations.destructive_hint is True

        await c.call_tool("write_file", {"path": "notes/a.md", "content": "# A"})
        assert (sandbox / "notes" / "a.md").read_text() == "# A"
        assert text(await c.call_tool("read_file", {"path": "/notes/a.md"})) == "# A"  # leading slash tolerated
        assert "hello.txt" in text(await c.call_tool("list_dir", {}))
        assert "hello.txt" in text(await c.call_tool("search_files", {"query": "THERE"}))


@pytest.mark.parametrize("path", ["../outside.txt", "/../../etc/passwd", "notes/../../x"])
async def test_sandbox_refuses_escapes(sandbox, path):
    async with Client(sandbox_files.server) as c:
        r = await c.call_tool("write_file", {"path": path, "content": "x"})
        assert r.is_error and "outside the sandbox" in text(r)  # ToolError text reaches the client
    assert not (sandbox.parent / "outside.txt").exists()


async def test_sandbox_refuses_symlink_tunnel(sandbox, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    (sandbox / "link").symlink_to(tmp_path)  # a door from inside the sandbox to outside it
    async with Client(sandbox_files.server) as c:
        r = await c.call_tool("read_file", {"path": "link/secret.txt"})
        assert r.is_error and "top secret" not in text(r)


async def test_sandbox_delete_rules(sandbox):
    async with Client(sandbox_files.server) as c:
        assert (await c.call_tool("delete_path", {"path": "."})).is_error  # never the root
        (sandbox / "d").mkdir()
        (sandbox / "d" / "f").write_text("x")
        assert (await c.call_tool("delete_path", {"path": "d"})).is_error  # not empty
        assert not (await c.call_tool("delete_path", {"path": "d/f"})).is_error
        assert not (sandbox / "d" / "f").exists()


async def test_calculator_is_safe_and_correct():
    async with Client(calc_clock.server) as c:
        assert text(await c.call_tool("calculate", {"expression": "2^10 + sqrt(16)"})).endswith("= 1028.0")
        for evil in ["__import__('os').system('echo pwned')", "(1).__class__", "9**9**9", "factorial(100000)"]:
            r = await c.call_tool("calculate", {"expression": evil})
            assert r.is_error, evil


async def test_time_tools():
    async with Client(calc_clock.server) as c:
        r = text(await c.call_tool("convert_time", {
            "time": "2026-11-13 09:00", "from_timezone": "Africa/Lagos", "to_timezone": "Asia/Tokyo"}))
        assert "17:00 Asia/Tokyo" in r  # Lagos UTC+1, Tokyo UTC+9
        assert "42 days" in text(await c.call_tool("days_between", {"start": "2026-11-13", "end": "2026-12-25"}))
        bad = await c.call_tool("now", {"timezone": "Lagos"})
        assert bad.is_error and "Africa/Lagos" in text(bad)  # helpful suggestion


def test_html_to_text():
    p = web_reader._TextExtractor()
    p.feed("<html><head><title>T</title><style>x{}</style></head><body><nav>menu</nav>"
           "<h1>Head</h1><p>Para one.</p><script>evil()</script><p>Two</p></body></html>")
    assert p.title == "T"
    out = p.text()
    assert "## Head" in out and "Para one." in out and "Two" in out
    assert "menu" not in out and "evil" not in out


@pytest.mark.parametrize("url", ["http://127.0.0.1:8080/admin", "http://localhost/", "file:///etc/passwd",
                                 "http://10.0.0.1/"])
async def test_read_page_blocks_private_addresses(url):
    async with Client(web_reader.server) as c:
        r = await c.call_tool("read_page", {"url": url})
        assert r.is_error


def test_project_ships_example_config():
    from switchboard.config import load_config
    specs = load_config(Path(__file__).resolve().parents[1] / "servers.json")
    assert {s.name for s in specs} >= {"files", "calc", "web", "web-http"}
    assert next(s for s in specs if s.name == "web-http").transport == "http"
