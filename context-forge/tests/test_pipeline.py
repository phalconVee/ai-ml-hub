"""End-to-end pipeline tests with a fake Claude and fake arXiv. No network, no key."""

from pathlib import Path

from forge.items import ContextItem
from forge.pipeline import run
from forge.planner import make_plan
from forge.sources.documents import DocumentStore
from forge.sources.memory import MemoryStore
from forge.sources.papers import parse_feed
from forge.sources.web import harvest, search_web
from forge.synthesizer import check_citations
from tests.fakes import FakeClient, response, text_block, web_citation

SAMPLE = Path(__file__).parents[1] / "sample_docs" / "context-notes.md"
Q = "What is context engineering and why does the context window budget matter?"

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2307.03172v3</id>
    <published>2023-07-06T17:54:11Z</published>
    <title>Lost in the Middle: How Language Models
      Use Long Contexts</title>
    <summary>  Language models use information in the middle of a long context
      window less reliably than at the start or end.</summary>
    <author><name>Jane Researcher</name></author>
  </entry>
</feed>"""


def fake_arxiv(query):
    return parse_feed(ATOM)


def stores(tmp_path):
    docs = DocumentStore()
    docs.add(SAMPLE.name, SAMPLE.read_text())
    return docs, MemoryStore(tmp_path / "mem.json")


# ── individual stages ───────────────────────────────────────────────────────

def test_arxiv_feed_parsing():
    [p] = parse_feed(ATOM)
    assert p.title == "Lost in the Middle: How Language Models Use Long Contexts"  # newlines squashed
    assert p.url.endswith("2307.03172v3") and p.meta["published"] == "2023-07-06"
    assert p.text.startswith("Language models use")


def test_web_harvest_groups_quotes_by_url_and_skips_errors():
    blocks = [
        # A failed search: content is an error object, not a list. Must not crash.
        response([]).content,
        [text_block("a", [web_citation("https://x.io", "X", "first quote"),
                          web_citation("https://x.io", "X", "second quote")]),
         text_block("b", [web_citation("https://y.io", None, "y quote")])],
    ]
    items, _ = harvest([b for group in blocks for b in group])
    assert [i.url for i in items] == ["https://x.io", "https://y.io"]
    assert items[0].text == "first quote … second quote"
    assert items[1].title == "https://y.io"        # missing title falls back to URL


def test_web_search_resumes_after_pause_turn():
    client = FakeClient()
    paused = response([text_block("searching…")], stop="pause_turn")
    finished = client._create  # keep the normal canned reply for the 2nd call
    replies = iter([paused])
    client.beta.messages.create = lambda **kw: next(replies, None) or finished(**kw)
    items, queries = search_web("context engineering", client)
    assert items and queries == ["context engineering"]


def test_planner_falls_back_when_claude_is_down():
    plan = make_plan(Q, FakeClient(fail_on=("plan",)))
    assert not plan.by_llm and plan.doc_query == Q and plan.use_web


def test_citation_check_strips_made_up_ids():
    a = check_citations("True [S1]. Invented [S9]. Also [S2][S1].", {"S1", "S2"})
    assert a.cited == ["S1", "S2"] and a.invalid == ["S9"]
    assert "[S9]" not in a.text


def test_memory_persists_and_is_retrieved(tmp_path):
    m = MemoryStore(tmp_path / "m.json")
    m.remember("What is RoPE?", "Rotary position embeddings encode position by rotation.")
    again = MemoryStore(tmp_path / "m.json")          # simulates an app restart
    [hit] = again.search("rotary position embeddings")
    assert hit.source == "memory" and "Rotary" in hit.text


# ── the full flow ──────────────────────────────────────────────────────────

def test_full_run_uses_all_four_sources(tmp_path):
    docs, memory = stores(tmp_path)
    memory.remember("What does lost in the middle mean?",
                    "Models under-use context in the middle of a long window.")
    client = FakeClient()
    r = run(Q, client=client, docs=docs, memory=memory, arxiv=fake_arxiv)

    assert {c.source for c in r.candidates} == {"doc", "web", "paper", "memory"}
    assert r.kept and all(c.kept for c in r.kept)
    assert all(c.reason for c in r.candidates)          # every decision explained
    assert "[S1]" in r.context                          # the window the model saw
    assert r.answer.cited == ["S1"] and not r.errors
    assert len(memory.turns) == 2                       # this turn was remembered

    # The planner and the answer both went through the one `ask` door.
    plan_call, answer_call = client.calls[0], client.calls[-1]
    assert plan_call["output_config"]["format"]["type"] == "json_schema"
    assert answer_call["model"] == "claude-opus-5-5" and answer_call["fallbacks"] == "default"
    assert "<context>" in answer_call["messages"][0]["content"]


def test_planner_can_skip_web_and_papers(tmp_path):
    docs, memory = stores(tmp_path)
    plan = {"doc_query": "context window", "web_query": "", "paper_query": "",
            "use_web": False, "use_papers": False, "rationale": "local notes suffice"}
    called = []
    r = run(Q, client=FakeClient(plan=plan), docs=docs, memory=memory,
            arxiv=lambda q: called.append(q) or [])
    assert not called and {c.source for c in r.candidates} <= {"doc", "memory"}


def test_a_failing_source_does_not_sink_the_answer(tmp_path):
    docs, memory = stores(tmp_path)

    def broken_arxiv(q):
        raise TimeoutError("arXiv timed out")

    r = run(Q, client=FakeClient(fail_on=("web",)), docs=docs, memory=memory, arxiv=broken_arxiv)
    assert set(r.errors) == {"web", "paper"}
    assert r.kept and r.answer.cited                     # docs alone still answered


def test_judge_scores_change_the_ranking(tmp_path):
    docs = DocumentStore()
    docs.add("a.md", "context window budget basics and context window budget tips")
    docs.add("b.md", "context window budget, explained with the key insight")
    # BM25 likes a.md (more repeats); the judge says b.md is the useful one.
    client = FakeClient(judge=[{"i": 0, "score": 0}, {"i": 1, "score": 10}])
    r = run("context window budget", client=client, docs=docs,
            memory=MemoryStore(tmp_path / "m.json"), enabled={"doc"}, use_judge=True)
    assert r.kept[0].title.startswith("b.md") and r.kept[0].judge == 1.0


def test_empty_sources_give_an_honest_answer(tmp_path):
    r = run(Q, client=FakeClient(), docs=DocumentStore(),
            memory=MemoryStore(tmp_path / "m.json"), enabled={"doc", "memory"})
    assert not r.kept and "couldn't find" in r.answer.text
    assert not ContextItem("doc", "t", "x").kept
