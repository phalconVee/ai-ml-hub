"""The MCP server and toolbox, tested in-process with fake sources (no network)."""

import json

from scout import sources
from scout.config import PLATFORM_TOOLS, SHARED_TOOLS
from scout.mcp_server import server
from scout.toolbox import open_toolbox

FAKE = [{"title": "Fake HN story", "url": "https://news.ycombinator.com/item?id=1",
         "snippet": "s", "published": "2026-01-01", "meta": "10 points, 2 comments"}]


async def test_server_advertises_every_configured_tool():
    async with open_toolbox(server) as tb:
        expected = {t for tools in PLATFORM_TOOLS.values() for t in tools} | set(SHARED_TOOLS)
        assert expected <= set(tb.names)


async def test_tools_convert_to_anthropic_format_and_scope():
    async with open_toolbox(server) as tb:
        defs = tb.anthropic_tools(["search_hackernews", "read_page"])
        assert {d["name"] for d in defs} == {"search_hackernews", "read_page"}  # nothing else leaks in
        hn = next(d for d in defs if d["name"] == "search_hackernews")
        assert hn["input_schema"]["properties"]["query"]["type"] == "string"
        assert "Hacker News" in hn["description"]


async def test_call_runs_source_and_returns_json(monkeypatch):
    async def fake_hn(query, n=8):
        assert query == "mcp"
        return FAKE
    monkeypatch.setattr(sources, "hackernews", fake_hn)
    async with open_toolbox(server) as tb:
        text, is_error = await tb.call("search_hackernews", {"query": "mcp"})
    assert not is_error
    assert json.loads(text)[0]["url"] == FAKE[0]["url"]


async def test_source_failure_becomes_error_result_not_crash(monkeypatch):
    async def broken(query, n=8):
        raise RuntimeError("rate limited")
    monkeypatch.setattr(sources, "web", broken)
    async with open_toolbox(server) as tb:
        text, is_error = await tb.call("search_web", {"query": "x"})
        unknown_text, unknown_err = await tb.call("search_tiktok", {"query": "x"})
    assert is_error and "rate limited" in text
    assert unknown_err and "Unknown tool" in unknown_text
