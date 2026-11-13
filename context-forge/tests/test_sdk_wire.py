"""The *real* Anthropic SDK with only the network faked.

The other tests fake the client object. This one keeps `anthropic.Anthropic`
and swaps its HTTP transport, which proves the SDK accepts our parameters
(structured outputs, server web-search tool, fallbacks) and puts them on the
wire as the API expects, and that our code can read real SDK response types
(citations on text blocks).
"""

import json

import httpx2 as httpx  # anthropic 2.x runs on httpx2 (same API as httpx)
from anthropic import Anthropic

from forge.config import SETTINGS
from forge.planner import make_plan
from forge.sources.web import search_web


def message(content, stop="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}}


def client_with(reply, sent):
    def handler(req):
        sent.append(req)
        return httpx.Response(200, json=reply)
    return Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_planner_request_on_the_wire():
    sent = []
    plan_json = {"doc_query": "d", "web_query": "w", "paper_query": "p",
                 "use_web": False, "use_papers": True, "rationale": "r"}
    client = client_with(message([{"type": "text", "text": json.dumps(plan_json)}]), sent)
    plan = make_plan("question?", client)

    assert plan.by_llm and plan.paper_query == "p" and plan.use_web is False
    body = json.loads(sent[0].content)
    assert SETTINGS.fallback_beta in sent[0].headers["anthropic-beta"]
    assert body["fallbacks"] == "default" and body["model"] == "claude-opus-5-5"
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"


def test_web_search_tool_and_real_citation_types():
    sent = []
    reply = message([
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
         "input": {"query": "context engineering"}},
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
            {"type": "web_search_result", "url": "https://ex.com", "title": "Ex",
             "encrypted_content": "enc", "page_age": None}]},
        {"type": "text", "text": "It curates the window.", "citations": [
            {"type": "web_search_result_location", "url": "https://ex.com", "title": "Ex",
             "cited_text": "Context engineering curates what the model sees.",
             "encrypted_index": "idx"}]},
    ])
    items, queries = search_web("context engineering", client_with(reply, sent))

    tool = json.loads(sent[0].content)["tools"][0]
    assert tool["type"] == "web_search_20260209" and tool["max_uses"] == SETTINGS.max_web_searches
    assert queries == ["context engineering"]
    assert items[0].url == "https://ex.com" and items[0].text.startswith("Context engineering")
