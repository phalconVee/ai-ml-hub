"""The real `anthropic` SDK, with only the network swapped out.

The fakes elsewhere are hand-made. Here the genuine client runs over a mock
HTTP transport, so we check what fakes can't: the SDK accepts every
parameter we send, the JSON on the wire is right, and real response JSON
(thinking block first) parses into what our code reads.
"""

import json

import anthropic
import httpx2 as httpx  # anthropic 2.x is built on httpx2 (same API as httpx)

from memoir.companion import Companion
from memoir.graph import KnowledgeGraph

from .fakes import TURN1


def message(text):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "thinking", "thinking": "hmm", "signature": "sig"},
                        {"type": "text", "text": text}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10}}


def test_real_sdk_wire_format():
    canned = [message("Hi Henry, Lagos sounds great."), message(json.dumps(TURN1))]
    bodies, headers = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        headers.append(request.headers)
        return httpx.Response(200, json=canned.pop(0))

    client = anthropic.Anthropic(api_key="test-key", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    kg = KnowledgeGraph(":memory:")
    r = Companion(client, kg).turn("I'm Henry, I live in Lagos.", [])

    assert r.reply == "Hi Henry, Lagos sounds great." and r.memory_error is None
    assert kg.find("Henry")["name"] == "User" and kg.stats()["facts"] == 2
    reply_body, extract_body = bodies
    for body, h in zip(bodies, headers):
        assert body["model"] == "claude-opus-5-5" and body["fallbacks"] == "default"
        assert "server-side-fallback-2026-07-01" in h["anthropic-beta"]
        assert "thinking" not in body and "temperature" not in body
    assert reply_body["output_config"] == {"effort": "medium"}
    fmt = extract_body["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False
