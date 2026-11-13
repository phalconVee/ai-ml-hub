"""The agent loop and the docs→web routing, with Claude replaced by a script.

`FakeClient` mimics `client.beta.messages.create(...)`: each call pops the next
canned response and records the request, so we can assert both what the
agents *did* and what they *sent*.
"""

import json
from types import SimpleNamespace as NS

from sage.bm25 import BM25Index
from sage.documents import build_chunks
from sage.router import answer

DOCS = [("handbook.md", b"Employees get 25 days of paid leave per year. "
                         b"Leave requests go to your manager in the HR portal.")]


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        # Snapshot `messages`: the agent keeps appending to the same list object.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def msg(stop_reason, *blocks):
    return NS(stop_reason=stop_reason, content=list(blocks))


def text(t, citations=None):
    return NS(type="text", text=t, citations=citations)


def verdict(v, answer="", cited=(), reason=""):
    return text(json.dumps({"verdict": v, "answer": answer,
                            "cited_chunks": list(cited), "reason": reason}))


def tool_use(query, id="tu_1"):
    return NS(type="tool_use", id=id, name="search_documents", input={"query": query})


def web_answer(t="Paris is the capital of France.", url="https://example.org/paris"):
    return msg("end_turn",
               NS(type="server_tool_use", name="web_search", input={"query": "capital of France"}),
               NS(type="web_search_tool_result",
                  content=[NS(url=url, title="Paris - Example")]),
               text(t, citations=[NS(url=url, title="Paris - Example")]))


def index():
    return BM25Index(build_chunks(DOCS))


def test_answer_from_docs_after_one_extra_search():
    client = FakeClient([
        msg("tool_use", tool_use("paid leave days")),                       # turn 1: search again
        msg("end_turn", verdict("answered", "25 days [handbook.md#p1-c1]",  # turn 2: verdict
                                ["handbook.md#p1-c1"])),
    ])
    a = answer("How much paid leave do I get?", index(), client)

    assert a.route == "docs" and "25 days" in a.text
    assert a.sources[0]["id"] == "handbook.md#p1-c1"
    assert len(client.requests) == 2                  # never reached the web
    # Turn 2 must carry the tool result back, tied to the tool_use id.
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "tu_1"
    assert "25 days" in tool_result["content"]
    # The librarian never gets the web tool.
    assert all(t["name"] == "search_documents" for t in client.requests[0]["tools"])


def test_not_found_falls_back_to_web():
    client = FakeClient([
        msg("end_turn", verdict("not_found", reason="Handbook says nothing about France")),
        web_answer(),
    ])
    a = answer("What is the capital of France?", index(), client)

    assert a.route == "web" and "Paris" in a.text
    assert a.sources == [{"title": "Paris - Example", "url": "https://example.org/paris"}]
    assert client.requests[1]["tools"][0]["type"].startswith("web_search")


def test_answer_citing_unseen_passage_is_rejected():
    # Claims "answered" but cites an id it was never shown → verification fails → web.
    client = FakeClient([
        msg("end_turn", verdict("answered", "42", ["made-up.pdf#p9-c9"])),
        web_answer(),
    ])
    a = answer("Anything", index(), client)
    assert a.route == "web"
    assert any("rejected" in step for step in a.trace)


def test_web_fallback_can_be_switched_off():
    client = FakeClient([msg("end_turn", verdict("not_found", reason="nope"))])
    a = answer("What is the capital of France?", index(), client, allow_web=False)
    assert a.route == "none" and len(client.requests) == 1


def test_no_documents_goes_straight_to_web():
    client = FakeClient([web_answer()])
    a = answer("What is the capital of France?", None, client)
    assert a.route == "web" and len(client.requests) == 1


def test_pause_turn_is_resumed_without_extra_user_message():
    paused = msg("pause_turn", NS(type="server_tool_use", name="web_search",
                                  input={"query": "q"}))
    client = FakeClient([paused, web_answer()])
    a = answer("Q?", None, client)
    assert a.route == "web"
    resumed = client.requests[1]["messages"]
    assert [m["role"] for m in resumed] == ["user", "assistant"]  # paused turn sent back as-is


def test_every_request_uses_configured_model_and_fallbacks():
    client = FakeClient([msg("end_turn", verdict("not_found")), web_answer()])
    answer("Q?", index(), client)
    for r in client.requests:
        assert r["model"] == "claude-opus-5-5"
        assert r["fallbacks"] == "default" and r["output_config"]["effort"] == "medium"
