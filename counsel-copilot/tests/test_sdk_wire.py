"""Run the whole crew through the *real* anthropic SDK, with the network
swapped for an in-process fake server (httpx2.MockTransport).

The fakes in test_crew.py prove the logic. This proves the plumbing: the SDK
accepts our request bodies (strict tools, search_result blocks, web search
tool, fallbacks, beta header) and parses the JSON responses into the objects
and attributes our code reads (tool_use.input, citations, web results).
"""

from __future__ import annotations

import json

import anthropic
import httpx2

from counsel import ClauseIndex, chunk_document, run_crew
from sample_docs.make_sample import build_sample_pdf

LATE = "sample_lease.pdf#p1-c4"
URL = "https://courts.example.gov/late-fees"


def message(stop_reason: str, content: list[dict]) -> dict:
    return {"id": "msg_x", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "stop_reason": stop_reason, "stop_sequence": None, "content": content,
            "usage": {"input_tokens": 1, "output_tokens": 1}}


RESPONSES = [
    # 1. Clause Finder: one search…
    message("tool_use", [{"type": "tool_use", "id": "toolu_1", "name": "search_documents",
                          "input": {"query": "late fee"}}]),
    # 2. …then the report, flagging a question about the law.
    message("tool_use", [{"type": "tool_use", "id": "toolu_2", "name": "file_report", "input": {
        "clause_ids": [LATE], "coverage": "partial",
        "legal_questions": ["Are 10% late fees enforceable in California?"],
        "summary": "Clause 3 charges 10% after five days."}}]),
    # 3. Legal Researcher: server-side web search + cited write-up.
    message("end_turn", [
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
         "input": {"query": "California residential late fee enforceable"}},
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
            {"type": "web_search_result", "url": URL, "title": "Late fees | Courts",
             "encrypted_content": "abc", "page_age": None}]},
        {"type": "text", "text": "A late fee must be a reasonable estimate of damages.", "citations": [
            {"type": "web_search_result_location", "url": URL, "title": "Late fees | Courts",
             "encrypted_index": "x", "cited_text": "a reasonable estimate of damages"}]},
    ]),
    # 4. Drafter: answer citing one clause and one web passage.
    message("end_turn", [
        {"type": "text", "text": "Your lease charges 10% after five days.", "citations": [
            {"type": "search_result_location", "source": LATE, "title": "sample_lease.pdf, p.1, 3. Late Payment",
             "cited_text": "a late fee of ten percent (10%)", "search_result_index": 0,
             "start_block_index": 0, "end_block_index": 1}]},
        {"type": "text", "text": " California courts ask whether that is reasonable.", "citations": [
            {"type": "search_result_location", "source": URL, "title": "Late fees | Courts",
             "cited_text": "a reasonable estimate of damages", "search_result_index": 1,
             "start_block_index": 0, "end_block_index": 1}]},
    ]),
]


def test_full_crew_through_real_sdk():
    seen: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append({"body": json.loads(request.content),
                     "beta": request.headers.get("anthropic-beta", "")})
        return httpx2.Response(200, json=RESPONSES[len(seen) - 1])

    client = anthropic.Anthropic(
        api_key="test-key",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)))
    index = ClauseIndex()
    index.add(chunk_document(build_sample_pdf(), "sample_lease.pdf"))

    r = run_crew("Is my late fee legal?", index, client, jurisdiction="California")

    assert len(seen) == 4
    assert all("server-side-fallback-2026-07-01" in s["beta"] for s in seen)
    finder_tools = {t["name"]: t for t in seen[0]["body"]["tools"]}
    assert finder_tools["file_report"]["strict"] is True
    assert seen[2]["body"]["tools"][0]["type"] == "web_search_20260209"
    drafter_blocks = seen[3]["body"]["messages"][0]["content"]
    assert [b["source"] for b in drafter_blocks if b["type"] == "search_result"] == [LATE, URL]

    assert r.route == ["Clause Finder", "Legal Researcher", "Drafter"]
    assert r.draft.body_markdown() == ("Your lease charges 10% after five days. [1] "
                                      "California courts ask whether that is reasonable. [2]")
    assert [(c.kind, c.source) for c in r.draft.citations] == [("clause", LATE), ("web", URL)]
