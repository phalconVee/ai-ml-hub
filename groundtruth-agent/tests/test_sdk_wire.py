"""Run the agent through the *real* Anthropic SDK, with the network swapped for a fake server.

This checks what the fakes in test_pipeline.py can't: that our request body is
something the SDK will serialise (search_result blocks, fallbacks, effort, beta
header), and that the SDK parses citation objects into the attributes the agent reads.
"""

from __future__ import annotations

import json

import anthropic
import httpx2

from groundtruth import BM25Index, ask, ingest_pdf
from sample_docs.make_sample import build_sample_pdf

NAME = "northwind_report.pdf"


def _message(stop_reason: str, content: list[dict]) -> dict:
    return {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "stop_reason": stop_reason, "stop_sequence": None, "content": content,
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }


def test_real_sdk_round_trip():
    seen: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        seen.append({"body": body, "beta": request.headers.get("anthropic-beta", "")})
        if len(seen) == 1:
            return httpx2.Response(200, json=_message("tool_use", [
                {"type": "tool_use", "id": "toolu_1", "name": "search_documents",
                 "input": {"query": "bikes shipped 2024"}},
            ]))
        return httpx2.Response(200, json=_message("end_turn", [
            {"type": "text", "text": "Northwind shipped 12,400 bikes in 2024.", "citations": [{
                "type": "search_result_location", "source": f"{NAME}#page=1",
                "title": f"{NAME}, page 1 (table)", "cited_text": "| Bikes shipped | 9,850 | 12,400 |",
                "search_result_index": 0, "start_block_index": 3, "end_block_index": 4,
            }]},
        ]))

    client = anthropic.Anthropic(
        api_key="test-key",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    index = BM25Index()
    ingest_pdf(index, build_sample_pdf(), NAME)

    answer = ask("How many bikes shipped in 2024?", index, client=client)

    first = seen[0]
    assert "server-side-fallback-2026-07-01" in first["beta"]
    assert first["body"]["fallbacks"] == "default"
    assert first["body"]["output_config"] == {"effort": "medium"}

    # Second request carries the tool result as citable search_result blocks.
    tool_result = seen[1]["body"]["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["content"][0]["type"] == "search_result"
    assert tool_result["content"][0]["citations"] == {"enabled": True}

    assert answer.markdown() == "Northwind shipped 12,400 bikes in 2024. [1]"
    assert answer.citations[0].quote == "| Bikes shipped | 9,850 | 12,400 |"
