"""The *real* Anthropic SDK with only the network faked.

The other tests fake the client object. Here `anthropic.Anthropic` is real
and only its HTTP transport is swapped, which proves the SDK accepts our
parameters (structured outputs, effort, fallbacks) and puts them on the wire
as the API expects, and that our code reads the SDK's real response types.
"""

import json

import httpx2 as httpx  # anthropic 2.x runs on httpx2 (same API as httpx)
from anthropic import Anthropic

from recall.config import SETTINGS
from recall.extractor import extract_and_store
from tests.fakes import fact


def message(content, stop="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": SETTINGS.model,
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}}


def test_extractor_request_and_response_on_the_wire(store):
    sent = []
    reply = message([{"type": "text", "text": json.dumps({"facts": [
        fact("user", "lives_in", "Porto", "The user lives in Porto.", exclusive=True)]})}])

    def handler(req):
        sent.append(req)
        return httpx.Response(200, json=reply)

    client = Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    pid = store.get_or_create_profile("Ana")
    writes = extract_and_store(store, client, pid, user_text="I live in Porto.", reply="Nice!")

    assert [w.fact.statement for w in writes] == ["The user lives in Porto."]
    body = json.loads(sent[0].content)
    assert SETTINGS.fallback_beta in sent[0].headers["anthropic-beta"]
    assert body["model"] == "claude-opus-5-5" and body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "low"
    schema = body["output_config"]["format"]
    assert schema["type"] == "json_schema" and "facts" in schema["schema"]["properties"]
