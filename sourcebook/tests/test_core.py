"""Worked examples for the bricks: sources, chunking, retrieval, persistence."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from sourcebook import Notebook, load_pdf, load_text, load_url
from sourcebook.chunk import chunk_source, split_sentences
from sourcebook.retrieve import BM25, Retriever, rrf, tokenize
from sourcebook.sources import html_to_text

SAMPLES = Path(__file__).parent.parent / "sample_sources"


def sample(name):
    return load_text((SAMPLES / name).read_text(), name)


# --- sources ------------------------------------------------------------------

def test_pdf_keeps_pages_separate():
    doc = pymupdf.open()
    for text in ["Page one talks about pollen.", "Page two talks about nectar."]:
        doc.new_page().insert_text((72, 72), text)
    src = load_pdf(doc.tobytes(), "bees.pdf")
    assert src.kind == "pdf" and len(src.pages) == 2
    assert "nectar" in src.pages[1]


def test_empty_pdf_is_rejected():
    doc = pymupdf.open()
    doc.new_page()
    with pytest.raises(ValueError, match="no text layer"):
        load_pdf(doc.tobytes(), "scan.pdf")


def test_html_prefers_article_and_drops_chrome():
    html = """<html><head><title>Bee facts</title><script>var x=1;</script></head><body>
      <nav>Home | About | Login</nav>
      <article><h1>Bees</h1><p>""" + "Bees dance to share where food is. " * 12 + """</p></article>
      <footer>Copyright 2026</footer></body></html>"""
    title, text = html_to_text(html)
    assert title == "Bee facts"
    assert text.startswith("# Bees")
    assert "Login" not in text and "Copyright" not in text and "var x" not in text


def test_load_url_uses_injected_fetch():
    page = "<html><title>T</title><body><main><p>" + "word " * 60 + "</p></main></body></html>"
    src = load_url("https://example.org/a", fetch=lambda u: ("text/html", page))
    assert src.kind == "web" and src.url == "https://example.org/a" and src.title == "T"
    with pytest.raises(ValueError, match="readable"):
        load_url("https://example.org/b", fetch=lambda u: ("text/html", "<html><body>hi</body></html>"))


# --- chunking -------------------------------------------------------------------

def test_sentence_split():
    assert split_sentences("Bees fly far. They return home! Do they get lost? Rarely.") == [
        "Bees fly far.", "They return home!", "Do they get lost?", "Rarely."]
    assert split_sentences("## A heading") == ["## A heading"]


def test_chunks_respect_size_overlap_and_pages():
    src = load_text("\n\n".join(f"Sentence number {i} is about bees and flowers." for i in range(60)), "s.txt")
    passages = chunk_source(src, words=40, overlap=1)
    assert len(passages) > 5
    assert all(len(p.text.split()) <= 48 for p in passages)          # ~size, never far over
    for a, b in zip(passages, passages[1:]):
        assert a.sentences[-1] == b.sentences[0]                      # one-sentence overlap
    doc = pymupdf.open()
    for t in ["Alpha page text.", "Beta page text."]:
        doc.new_page().insert_text((72, 72), t)
    pdf_passages = chunk_source(load_pdf(doc.tobytes(), "x.pdf"))
    assert [p.page for p in pdf_passages] == [1, 2]                  # never crosses a page


# --- retrieval --------------------------------------------------------------------

def test_tokenize_folds_plurals_and_drops_stopwords():
    assert tokenize("The bees and their dances") == ["bee", "dance"]


def test_bm25_ranks_the_right_passage_first():
    nb = Notebook("t", home=Path("/nonexistent"))
    nb.add_source(sample("bee_navigation.md"))
    nb.add_source(sample("waggle_dance.md"))
    top = nb.retriever.search("optic flow tunnel distance", k=3)
    assert "optic flow" in top[0].text
    top = nb.retriever.search("radar tracking recruits", k=3)
    assert "radar" in top[0].text


def test_source_filter():
    nb = Notebook("t", home=Path("/nonexistent"))
    a, b = sample("bee_navigation.md"), sample("waggle_dance.md")
    nb.add_source(a)
    nb.add_source(b)
    hits = nb.retriever.search("bees dance sun", k=8, allowed_sources={b.id})
    assert hits and all(p.source_id == b.id for p in hits)


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [2, 1, 4]])
    assert {fused[0][0], fused[1][0]} == {1, 2}      # in both lists -> top two
    assert fused[-1][0] in (3, 4)


class KeywordEmbedder:
    """A toy 'embedding': counts of two concept groups. Enough to test fusion."""
    def embed(self, texts):
        return [[t.lower().count("polar"),
                 t.lower().count("smell") + t.lower().count("odour")] for t in texts]


def test_hybrid_mode_uses_both_lists():
    passages = chunk_source(sample("waggle_dance.md")) + chunk_source(sample("bee_navigation.md"))
    r = Retriever(passages, KeywordEmbedder())
    assert r.mode.startswith("hybrid")
    # "polarisation" is not a word in the text, so BM25 alone finds nothing,
    # but the toy embedder maps it onto the sun/polar dimension.
    assert not r.bm25.rank("polarisation")
    hits = r.search("polarisation", k=2)
    assert any("polarised" in p.text for p in hits)


# --- persistence ----------------------------------------------------------------------

def test_notebook_round_trip(tmp_path):
    nb = Notebook("Bee research", home=tmp_path)
    src = sample("waggle_dance.md")
    assert nb.add_source(src) is True
    assert nb.add_source(sample("waggle_dance.md")) is False      # duplicate detected
    src.enabled = False
    nb.save()
    again = Notebook.load("Bee research", home=tmp_path)
    assert list(again.sources) == [src.id]
    assert again.sources[src.id].enabled is False
    from sourcebook import list_notebooks
    assert list_notebooks(tmp_path) == ["Bee research"]
