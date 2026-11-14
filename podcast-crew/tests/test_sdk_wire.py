"""The real `anthropic` SDK, with only the network swapped out.

The fakes elsewhere are hand-made. Here the genuine `anthropic.Anthropic`
client runs over a mock HTTP transport, so we check what fakes can't: the
SDK accepts every parameter we send, the JSON on the wire matches the API
docs, and real response JSON parses into blocks our code reads correctly.
"""

import json

import anthropic
import httpx2 as httpx  # anthropic 2.x is built on httpx2 (same API as httpx)

from podcast.crew import produce

from .fakes import ARTICLE_HTML, OUTLINE, lines


def message(obj):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": json.dumps(obj)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10}}


def test_real_sdk_wire_format():
    canned = [message(OUTLINE),
              message({"title": "Bees", "lines": lines(8)}),
              message({"title": "Bees", "lines": lines(8), "changes": ["Cut filler."]})]
    bodies, headers = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        headers.append(request.headers)
        return httpx.Response(200, json=canned.pop(0))

    client = anthropic.Anthropic(api_key="test-key", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    prod = produce("https://blog.example/bees", client, fetch=lambda u: ARTICLE_HTML, minutes=1)

    assert len(bodies) == 3
    for body, h in zip(bodies, headers):
        assert body["model"] == "claude-opus-5-5"
        assert body["fallbacks"] == "default"
        assert "server-side-fallback-2026-07-01" in h["anthropic-beta"]
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert "thinking" not in body and "temperature" not in body
    # The speaker enum reached the wire: the model can only write lines for A or B.
    speaker = bodies[1]["output_config"]["format"]["schema"]["properties"]["lines"]["items"]["properties"]["speaker"]
    assert speaker == {"type": "string", "enum": ["A", "B"]}

    assert prod.edit.changes == ["Cut filler."] and len(prod.script.lines) == 8
