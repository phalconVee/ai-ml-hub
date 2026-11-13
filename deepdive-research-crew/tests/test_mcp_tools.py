"""The MCP tools server, tested over a REAL stdio subprocess (offline backend).

This is the same path the crew uses at runtime: spawn `deepdive.tools_server`,
handshake, list tools, call them.
"""

import json

import pytest

from deepdive.mcp_bridge import open_toolbox


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("DEEPDIVE_SEARCH_BACKEND", "fake")


async def test_tools_are_discovered_in_claude_format():
    async with open_toolbox() as box:
        names = {t["name"] for t in box.tools}
        assert names == {"web_search", "read_page"}
        for t in box.tools:
            # Exactly the three fields the Claude API expects for a tool.
            assert set(t) == {"name", "description", "input_schema"}
            assert t["input_schema"]["type"] == "object"


async def test_search_then_read_over_mcp():
    async with open_toolbox() as box:
        text, err = await box.call("test", "web_search", {"query": "adoption", "max_results": 2})
        assert not err
        hits = json.loads(text)
        page_text, err = await box.call("test", "read_page", {"url": hits[0]["url"]})
        assert not err and "42%" in json.loads(page_text)["text"]
        assert [c.name for c in box.calls] == ["web_search", "read_page"]


async def test_tool_failure_reaches_the_model_as_a_readable_error():
    async with open_toolbox() as box:
        text, err = await box.call("test", "read_page", {"url": "http://localhost:8501"})
        assert err
        # The *reason* survives the trip (ToolError), so the model can adapt.
        assert "non-public" in text
