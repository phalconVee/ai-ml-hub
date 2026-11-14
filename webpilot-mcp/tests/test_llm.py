"""What actually goes over the wire to Claude: real SDK, fake HTTP.

The Anthropic SDK (2.x) is built on httpx2, so a MockTransport lets us capture
the exact request without a network or an API key.
"""

import json

import httpx2
import pytest
from anthropic import AsyncAnthropic

from tests.conftest import make_settings
from webpilot.llm import LLM, RefusalError


def fake_client(captured: list, stop_reason: str = "end_turn") -> AsyncAnthropic:
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": "hi"}], "stop_reason": stop_reason,
            "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
        })

    return AsyncAnthropic(api_key="test", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))


async def test_request_shape():
    captured = []
    llm = LLM(make_settings(effort="low"), client=fake_client(captured))
    tools = [{"name": "goto", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
    resp = await llm.step(system="sys", messages=[{"role": "user", "content": "go"}], tools=tools,
                          tool_choice={"type": "none"})
    assert resp.content[0].text == "hi"

    req = captured[0]
    body = json.loads(req.content)
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"] == {"effort": "low"}
    assert body["fallbacks"] == "default"
    assert body["tool_choice"] == {"type": "none"}
    assert body["tools"][0]["name"] == "goto"
    assert "server-side-fallback-2026-07-01" in req.headers["anthropic-beta"]


async def test_no_tool_choice_by_default_and_refusal_raises():
    captured = []
    llm = LLM(make_settings(), client=fake_client(captured, stop_reason="refusal"))
    with pytest.raises(RefusalError):
        await llm.step(system="s", messages=[{"role": "user", "content": "x"}], tools=[])
    assert "tool_choice" not in json.loads(captured[0].content)
