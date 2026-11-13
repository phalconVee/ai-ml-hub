"""The real `anthropic` SDK, with only the network swapped out.

The other tests use a hand-made fake client. This one uses the genuine
`anthropic.Anthropic` and plugs a mock HTTP transport underneath it, so we
check two things the fakes can't:
  • the SDK accepts every parameter we send, and the JSON on the wire is
    what the API docs describe (tool types, fallbacks, output_config…);
  • real response JSON parses into the SDK's own block types, and our
    ledger code reads those types correctly.
"""

import json

import anthropic
import httpx2 as httpx  # anthropic 2.x is built on httpx2 (same API as httpx)

from newsroom.desk import run_desk

URL = "https://agency.example/story"


def reply(content, stop="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10}}


REPORTER = reply([
    {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "topic"}},
    {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
        {"type": "web_search_result", "url": URL, "title": "Agency story",
         "encrypted_content": "x", "page_age": "3 hours ago"}]},
    {"type": "text", "text": "- Something happened today.", "citations": [
        {"type": "web_search_result_location", "url": URL, "title": "Agency story",
         "cited_text": "Something happened today.", "encrypted_index": "y"}]},
])
WRITER = reply([{"type": "text", "text": json.dumps(
    {"headline": "Something happened", "dek": "Today.", "body": "Something happened today [S1]."})}])
EDITOR = reply([{"type": "text", "text": json.dumps({"verdict": "publish", "notes": []})}])


def test_real_sdk_sends_documented_shapes_and_parses_real_blocks():
    bodies, headers, canned = [], [], [REPORTER, WRITER, EDITOR]

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        headers.append(request.headers)
        return httpx.Response(200, json=canned.pop(0))

    client = anthropic.Anthropic(api_key="test-key", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    story = run_desk("topic", client, today="Friday, 13 November 2026")

    # What went over the wire.
    reporter, writer, editor = bodies
    assert reporter["model"] == "claude-opus-5-5"
    assert reporter["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in headers[0]["anthropic-beta"]
    assert [t["type"] for t in reporter["tools"]] == ["web_search_20260209", "web_fetch_20260209"]
    assert reporter["output_config"] == {"effort": "medium"}
    assert writer["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in writer and "thinking" not in writer

    # Real SDK types flowed through our ledger and desk.
    assert story.status == "published"
    [s] = story.sources
    assert (s.id, s.url, s.age, s.quotes) == ("S1", URL, "3 hours ago", ["Something happened today."])
