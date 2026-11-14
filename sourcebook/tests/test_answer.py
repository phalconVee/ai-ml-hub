"""The core pipeline with a scripted fake Claude: citations, memory, podcast."""

from __future__ import annotations

from pathlib import Path

from fakes import FakeClient, cite, text_block

from sourcebook import Notebook, ask, generate_script, load_text
from sourcebook.answer import Answer, text_fragment_link
from sourcebook.config import SETTINGS
from sourcebook.sources import Source

SAMPLES = Path(__file__).parent.parent / "sample_sources"


def bee_notebook(tmp_path) -> Notebook:
    nb = Notebook("bees", home=tmp_path)
    for name in ["bee_navigation.md", "waggle_dance.md"]:
        nb.add_source(load_text((SAMPLES / name).read_text(), name))
    return nb


def test_answer_maps_citations_to_exact_sentences(tmp_path):
    nb = bee_notebook(tmp_path)
    # Find which passage BM25 will send first, so the fake can cite it.
    first = nb.retriever.search("How do bees use optic flow to measure distance?")[0]
    idx = next(i for i, s in enumerate(first.sentences) if "optic flow" in s)
    client = FakeClient(answers=[[
        text_block("Bees judge distance by "),
        text_block("optic flow, the visual motion they see", [cite(0, idx, idx + 1, first.sentences[idx])]),
        text_block("."),
    ]])

    ans = ask(nb, "How do bees use optic flow to measure distance?", client)

    # The request: search_result blocks first, then the question, citations on.
    req = client.requests[0]
    content = req["messages"][-1]["content"]
    assert content[0]["type"] == "search_result" and content[0]["citations"] == {"enabled": True}
    assert [b["text"] for b in content[0]["content"]] == first.sentences   # one block per sentence
    assert content[-1] == {"type": "text", "text": "How do bees use optic flow to measure distance?"}
    assert req["fallbacks"] == "default" and req["output_config"] == {"effort": SETTINGS.effort}

    # The answer: [1] sits after the cited segment and points at that sentence.
    assert ans.markdown() == "Bees judge distance by optic flow, the visual motion they see[1]."
    c = ans.citations[0]
    assert (c.passage_id, c.start, c.end) == (first.id, idx, idx + 1)
    assert c.source_title == "bee_navigation.md" and c.kind == "text"
    assert ans.search_query == ans.question          # first turn: no rewrite call
    assert len(client.requests) == 1


def test_same_passage_cited_twice_gets_one_number(tmp_path):
    nb = bee_notebook(tmp_path)
    p = nb.retriever.search("waggle run duration distance")[0]
    client = FakeClient(answers=[[
        text_block("A", [cite(0, 0, 1, p.sentences[0])]),
        text_block("B", [cite(0, 0, 1, p.sentences[0]), cite(99, 0, 1, "bogus index")]),
    ]])
    ans = ask(nb, "waggle run duration distance", client)
    assert [c.number for c in ans.citations] == [1]
    assert ans.segments == [("A", [1]), ("B", [1])]       # out-of-range index ignored


def test_follow_up_is_rewritten_and_history_replayed(tmp_path):
    nb = bee_notebook(tmp_path)
    client = FakeClient(
        answers=[[text_block("Through the waggle dance.")], [text_block("Radar tracking.")]],
        json_replies=[{"query": "evidence that recruits follow the waggle dance radar"}],
    )
    ask(nb, "How do bees tell others where food is?", client)
    ans = ask(nb, "What evidence supports that?", client)

    assert ans.search_query == "evidence that recruits follow the waggle dance radar"
    msgs = client.answer_requests[1]["messages"]
    assert msgs[0] == {"role": "user", "content": "How do bees tell others where food is?"}
    assert msgs[1] == {"role": "assistant", "content": "Through the waggle dance."}
    # The original wording is what Claude answers, not the rewrite.
    assert msgs[2]["content"][-1]["text"] == "What evidence supports that?"


def test_memory_window_folds_old_turns_into_summary(tmp_path):
    object.__setattr__(SETTINGS, "memory_turns", 2)   # SETTINGS is frozen; tests may bend it
    try:
        nb = bee_notebook(tmp_path)
        client = FakeClient(
            answers=[[text_block(f"answer {i}")] for i in range(3)],
            json_replies=[{"query": "q2"}, {"query": "q3"}, {"summary": "User asked about bees."}],
        )
        for q in ["q1", "q2", "q3"]:
            ask(nb, q, client)
        assert nb.memory.summarized == 1 and nb.memory.summary == "User asked about bees."
        assert [m["content"] for m in nb.memory.as_messages()][::2] == ["q2", "q3"]
        nb.save()
        again = Notebook.load("bees", home=tmp_path)
        assert again.memory.summary == "User asked about bees." and len(again.memory.turns) == 3
        # Next answer carries the summary in its system prompt.
        client2 = FakeClient(answers=[[text_block("ok")]], json_replies=[{"query": "q4"}, {"summary": "s"}])
        ask(again, "q4", client2)
        assert "User asked about bees." in client2.answer_requests[0]["system"]
    finally:
        object.__setattr__(SETTINGS, "memory_turns", 6)


def test_disabled_sources_are_not_searched_or_listed(tmp_path):
    nb = bee_notebook(tmp_path)
    nav = next(s for s in nb.sources.values() if s.title == "bee_navigation.md")
    nav.enabled = False
    client = FakeClient(answers=[[text_block("x")]])
    ask(nb, "optic flow distance sun compass", client)
    req = client.requests[0]
    assert "bee_navigation.md" not in req["system"]
    titles = {b["title"] for b in req["messages"][-1]["content"] if b["type"] == "search_result"}
    assert titles and all(t.startswith("waggle_dance.md") for t in titles)


def test_no_sources_means_no_api_call(tmp_path):
    client = FakeClient()
    ans = ask(Notebook("empty", home=tmp_path), "anything?", client)
    assert client.requests == [] and "Add a source" in ans.plain_text()


def test_answer_serialises(tmp_path):
    nb = bee_notebook(tmp_path)
    p = nb.retriever.search("sun compass")[0]
    ans = ask(nb, "sun compass", FakeClient(answers=[[text_block("Sun.", [cite(0, 0, 1, p.sentences[0])])]]))
    assert Answer.from_dict(ans.to_dict()) == ans


def test_web_citations_get_text_fragment_links(tmp_path):
    nb = Notebook("web", home=tmp_path)
    nb.add_source(Source("w1", "Bee blog", "web", "https://example.org/bees#top",
                         ["Bees can see polarised light in blue sky. " * 3]))
    p = nb.retriever.search("polarised light")[0]
    ans = ask(nb, "polarised light", FakeClient(answers=[[text_block("Yes.", [cite(0, 0, 1, p.sentences[0])])]]))
    assert ans.citations[0].url == "https://example.org/bees#:~:text=Bees%20can%20see%20polarised%20light%20in%20blue%20sky."
    long = text_fragment_link("https://x.org", "one two three four five six seven eight nine ten eleven twelve")
    assert long == "https://x.org#:~:text=one%20two%20three%20four%20five,eight%20nine%20ten%20eleven%20twelve"
    assert "%2D" in text_fragment_link("https://x.org", "self-driving")   # '-' must be escaped


def test_podcast_script_uses_digest_and_schema(tmp_path):
    nb = bee_notebook(tmp_path)
    client = FakeClient(json_replies=[{"title": "Bees, decoded", "turns": [
        {"speaker": "A", "text": "How do bees navigate?"},
        {"speaker": "B", "text": "With a sun compass and optic flow."},
        {"speaker": "A", "text": "   "},
    ]}])
    script = generate_script(nb, client, focus="navigation")
    assert script.title == "Bees, decoded" and len(script.turns) == 2   # blank turn dropped
    req = client.requests[0]
    assert req["output_config"]["format"]["type"] == "json_schema"
    prompt = req["messages"][0]["content"]
    assert "### Source 1: bee_navigation.md" in prompt and "### Source 2: waggle_dance.md" in prompt
    assert "Focus the episode on: navigation" in prompt
    script.save(nb.script_path)
    from sourcebook.podcast import Script
    assert Script.load(nb.script_path) == script
