"""Chunking and BM25: pure Python, no API needed."""

import pytest

from sage.bm25 import BM25Index, tokenize
from sage.documents import build_chunks, chunk_text


def test_chunks_overlap_and_cover_everything():
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_text(" ".join(words), size=40, overlap=10)
    # Windows start at 0, 30, 60 (step 30). The one at 60 reaches w99, so it's the last.
    assert [c.split()[0] for c in chunks] == ["w0", "w30", "w60"]
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]  # 10 shared words
    assert chunks[-1].split()[-1] == "w99"                     # nothing lost at the end


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("a b c", size=5, overlap=5)


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the GPU's VRAM?") == ["gpu", "s", "vram"]


def test_bm25_ranks_the_relevant_chunk_first():
    files = [
        ("pets.txt", b"Cats sleep most of the day. Dogs need daily walks."),
        ("db.md", b"Postgres stores vectors with the pgvector extension."),
        ("misc.txt", b"The weather was mild and the day was long."),
    ]
    index = BM25Index(build_chunks(files))
    hits = index.search("which extension stores vectors in postgres?")
    assert hits[0].chunk.source == "db.md"
    assert all(h.score > 0 for h in hits)
    assert index.search("quantum chromodynamics") == []  # no shared words → nothing


def test_chunk_ids_are_citable():
    (chunk,) = build_chunks([("notes.md", b"hello world")])
    assert chunk.id == "notes.md#p1-c1" and chunk.page == 1
