"""The real Anthropic SDK, with the network swapped for a mock transport.

`test_crew.py` fakes the client object itself. This test keeps the real
`AsyncAnthropic` and only fakes HTTP, which proves two things the fake can't:
  1. the SDK accepts our request parameters and puts them on the wire the way
     the API expects (body fields + the beta header);
  2. real SDK response objects (tool_use blocks, thinking blocks) can be
     appended to the history and sent straight back.
"""

import json

import httpx2 as httpx  # the anthropic SDK 2.x runs on httpx2 (same API as httpx)
from anthropic import AsyncAnthropic

from deepdive.agent import Agent
from deepdive.config import FALLBACK_BETA, CrewSettings
from deepdive.mcp_bridge import open_toolbox
from deepdive.tools_server import server as tools_server


def _message(content, stop_reason):
    return {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 50, "output_tokens": 5},
    }


async def test_request_wire_format_and_round_trip(monkeypatch):
    monkeypatch.setenv("DEEPDIVE_SEARCH_BACKEND", "fake")
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if len(sent) == 1:  # first call: Claude thinks, then asks for a search
            body = _message([
                {"type": "thinking", "thinking": "", "signature": "sig-abc"},
                {"type": "tool_use", "id": "toolu_1", "name": "web_search", "input": {"query": "mcp"}},
            ], "tool_use")
        else:
            body = _message([{"type": "text", "text": "## Findings\n- done"}], "end_turn")
        return httpx.Response(200, json=body)

    client = AsyncAnthropic(api_key="test-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async with open_toolbox(tools_server) as box:
        agent = Agent("scout-1", "scout", "You are a scout.", client, CrewSettings(), toolbox=box, max_turns=3)
        result = await agent.run("Research MCP")

    assert result.text == "## Findings\n- done"
    first, second = (json.loads(r.content) for r in sent)

    # Request 1: the shape we intended, as the API will receive it.
    assert sent[0].url.path == "/v1/messages"
    assert FALLBACK_BETA in sent[0].headers["anthropic-beta"]
    assert first["model"] == "claude-opus-5-5"
    assert first["thinking"] == {"type": "adaptive"}
    assert first["output_config"] == {"effort": "low"}
    assert first["fallbacks"] == "default"
    assert {t["name"] for t in first["tools"]} == {"web_search", "read_page"}

    # Request 2: the assistant turn went back unchanged (thinking signature
    # intact), followed by the tool result for the matching id.
    assistant, tool_turn = second["messages"][1], second["messages"][2]
    assert assistant["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    assert tool_turn["content"][0]["tool_use_id"] == "toolu_1"
    assert "Alpha Institute" in tool_turn["content"][0]["content"]
