"""Run the tool loop through the *real* Anthropic SDK with the network faked.

The fake client can't tell us whether the SDK accepts our request (strict
tools, fallbacks, effort, beta header) or whether echoing the SDK's own
response objects back as the next assistant turn serialises correctly
(thinking signature intact, tool_use id kept). This test can.
"""

import json

import anthropic
import httpx2

from vault import run_turn

TOOL_ID = "toolu_wire_1"


def _message(content, stop_reason):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "stop_reason": stop_reason, "stop_sequence": None, "content": content,
            "usage": {"input_tokens": 10, "output_tokens": 10}}


def test_tool_round_trip_through_the_sdk(vault):
    vault.ingest("station.md", b"The satellite phone is tested every Friday at noon in the radio room.")
    thread = vault.store.create_thread("wire")
    seen = []

    def handler(request):
        seen.append((json.loads(request.content), request.headers.get("anthropic-beta", "")))
        if len(seen) == 1:
            return httpx2.Response(200, json=_message([
                {"type": "thinking", "thinking": "", "signature": "sig-abc"},
                {"type": "tool_use", "id": TOOL_ID, "name": "search_knowledge_base",
                 "input": {"query": "satellite phone test"}},
            ], "tool_use"))
        return httpx2.Response(200, json=_message(
            [{"type": "text", "text": "Every Friday at noon (station.md, §1)."}], "end_turn"))

    client = anthropic.Anthropic(
        api_key="test-key",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    result = run_turn(vault, thread.id, "When is the sat phone tested?", client)

    first, beta = seen[0]
    assert "server-side-fallback-2026-07-01" in beta
    assert first["model"] == "claude-opus-5-5" and first["fallbacks"] == "default"
    assert first["output_config"] == {"effort": "medium"}
    assert all(t["strict"] is True for t in first["tools"])

    second, _ = seen[1]
    assistant, user = second["messages"][1], second["messages"][2]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    assert assistant["content"][1]["id"] == TOOL_ID
    assert user["content"][0]["tool_use_id"] == TOOL_ID
    assert "satellite phone" in user["content"][0]["content"]

    assert result.text == "Every Friday at noon (station.md, §1)."
    assert vault.store.get_messages(thread.id)[-1].trace[0]["chunks"][0]["doc"] == "station.md"
