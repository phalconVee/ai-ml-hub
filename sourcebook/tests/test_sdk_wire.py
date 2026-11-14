"""Run the pipeline through the *real* Anthropic SDK with the network faked.

The fake client in fakes.py can't tell us whether the SDK will serialise our
request (top-level search_result blocks, fallbacks, effort, beta header) or
whether it parses citations into the attributes answer.py reads. This can.
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
import httpx2

from sourcebook import Notebook, ask, generate_script, load_text

SAMPLES = Path(__file__).parent.parent / "sample_sources"


def _message(content: list[dict], stop_reason: str = "end_turn") -> dict:
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "stop_reason": stop_reason, "stop_sequence": None, "content": content,
            "usage": {"input_tokens": 10, "output_tokens": 10}}


def _client(handler) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="test-key",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )


def test_cited_answer_round_trip(tmp_path):
    nb = Notebook("wire", home=tmp_path)
    nb.add_source(load_text((SAMPLES / "waggle_dance.md").read_text(), "waggle_dance.md"))
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((body, request.headers.get("anthropic-beta", "")))
        first = body["messages"][-1]["content"][0]
        return httpx2.Response(200, json=_message([
            {"type": "text", "text": "Radar tracking showed recruits fly toward the danced location.",
             "citations": [{"type": "search_result_location", "source": first["source"],
                            "title": first["title"], "cited_text": first["content"][0]["text"],
                            "search_result_index": 0, "start_block_index": 0, "end_block_index": 1}]},
        ]))

    ans = ask(nb, "What did radar tracking show?", _client(handler))

    body, beta = seen[0]
    assert "server-side-fallback-2026-07-01" in beta
    assert body["fallbacks"] == "default" and body["model"] == "claude-opus-5-5"
    assert body["messages"][-1]["content"][0]["type"] == "search_result"
    assert ans.citations[0].start == 0 and ans.citations[0].end == 1
    assert ans.markdown().endswith("location.[1]")


def test_structured_output_script_round_trip(tmp_path):
    nb = Notebook("wire", home=tmp_path)
    nb.add_source(load_text((SAMPLES / "bee_navigation.md").read_text(), "bee_navigation.md"))
    script_json = {"title": "Bees", "turns": [{"speaker": "A", "text": "Hi."}, {"speaker": "B", "text": "Hello."}]}

    def handler(request):
        body = json.loads(request.content)
        assert body["output_config"]["format"]["type"] == "json_schema"
        return httpx2.Response(200, json=_message([{"type": "text", "text": json.dumps(script_json)}]))

    script = generate_script(nb, _client(handler))
    assert script.turns == [("A", "Hi."), ("B", "Hello.")]
