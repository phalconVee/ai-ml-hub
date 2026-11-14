"""The reply pipeline offline: what Claude is sent, and what memory does with the answer."""

import pytest

from memoir.companion import SYSTEM, Companion, speakable
from memoir.extract import SCHEMA
from memoir.graph import KnowledgeGraph
from memoir.llm import RefusedError

from .fakes import TURN1, TURN2, FakeClient, as_json, text


def test_full_turn_recall_reply_extract_merge():
    kg = KnowledgeGraph(":memory:")
    client = FakeClient([text("Nice to meet you, Henry! Lagos is lovely."), as_json(TURN1)])
    r = Companion(client, kg).turn("Hi, I'm Henry. I live in Lagos with my sister Ada.", [])

    assert r.reply == "Nice to meet you, Henry! Lagos is lovely."    # thinking block skipped
    assert r.turn == 1 and "User lives in Lagos" in r.delta.new_facts
    assert kg.find("Henry")["name"] == "User"

    reply_req, extract_req = client.requests
    assert reply_req["system"] == SYSTEM                             # static → cacheable
    assert reply_req["output_config"] == {"effort": "medium"}        # plain text, no schema
    assert "(nothing relevant in memory yet)" in reply_req["messages"][-1]["content"]
    assert extract_req["output_config"]["effort"] == "low"
    assert extract_req["output_config"]["format"]["schema"] is SCHEMA
    assert "Already in memory: User" in extract_req["messages"][0]["content"]
    for req in client.requests:
        assert req["model"] == "claude-opus-5-5" and req["fallbacks"] == "default"


def test_second_turn_answer_is_grounded_in_memory_and_history_is_sent():
    kg = KnowledgeGraph(":memory:")
    kg.merge(TURN1, turn=1)
    kg.merge(TURN2, turn=2)
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]
    client = FakeClient([text("Maybe a pottery class for Ada?"), as_json({"entities": [], "relations": []})])
    r = Companion(client, kg).turn("Gift ideas for my sister?", history)

    msgs = client.requests[0]["messages"]
    assert msgs[:2] == history and msgs[-1]["role"] == "user"
    assert "Ada likes Pottery" in msgs[-1]["content"] and msgs[-1]["content"].endswith("Gift ideas for my sister?")
    assert any(f.dst == "Pottery" for f in r.recall.facts)
    assert r.delta.empty()
    # Known names are offered to the extractor so it reuses "Ada" instead of "my sister".
    assert "Ada" in client.requests[1]["messages"][0]["content"]


def test_memory_failure_keeps_the_answer():
    kg = KnowledgeGraph(":memory:")
    client = FakeClient([text("Hello!"), RuntimeError("network blip")])
    r = Companion(client, kg).turn("hi", [])
    assert r.reply == "Hello!" and r.memory_error == "RuntimeError: network blip"
    assert kg.turn_count() == 1                                      # the turn is still logged


def test_refusal_and_cutoff_are_reported():
    kg = KnowledgeGraph(":memory:")
    with pytest.raises(RefusedError):
        Companion(FakeClient([text("", stop="refusal")]), kg).reply("x", [])
    with pytest.raises(RuntimeError, match="max_tokens"):
        Companion(FakeClient([text("cut", stop="max_tokens")]), kg).reply("x", [])


def test_history_is_trimmed():
    history = [{"role": r, "content": str(i)} for i in range(40) for r in ("user", "assistant")]
    client = FakeClient([text("ok")])
    Companion(client, KnowledgeGraph(":memory:")).reply("now", history)
    assert len(client.requests[0]["messages"]) == 6 * 2 + 1


def test_speakable_strips_what_a_voice_would_read_out():
    md = "**Sure!** Here's [the docs](https://x.y).\n- one\n- two\n```py\nprint()\n```\n# Done"
    assert speakable(md) == "Sure! Here's the docs. one two Done"
