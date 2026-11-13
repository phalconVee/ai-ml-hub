"""The ranker is pure Python, so these tests need no fakes at all."""

from forge.items import ContextItem
from forge.ranker import rank
from forge.textsearch import BM25, chunk_words, jaccard, tokenize

Q = "how does a token budget shape the context window"


def item(source, text, title="t"):
    return ContextItem(source=source, title=title, text=text)


def test_tokenize_and_chunk():
    assert tokenize("How does the RoPE scale?") == ["rope", "scale"]
    chunks = chunk_words(" ".join(str(i) for i in range(10)), size=4, overlap=1)
    assert chunks == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]   # overlap of one word, no empty tail


def test_bm25_prefers_matching_passage():
    idx = BM25(["cats and dogs", "token budget for the context window", "weather today"])
    assert idx.top("context window budget", 3)[0][0] == 1


def test_off_topic_items_are_dropped_with_reason():
    items = [item("doc", "The token budget limits the context window size."),
             item("web", "A recipe for banana bread with walnuts.")]
    kept = rank(Q, items)
    assert [k.text for k in kept] == [items[0].text]
    assert items[1].reason.startswith("off-topic")


def test_near_duplicates_are_removed():
    a = "The token budget decides how much text enters the context window."
    b = "The token budget decides how much text enters the context window!"   # same words
    items = [item("doc", a), item("web", b)]
    assert jaccard(a, b) == 1.0
    kept = rank(Q, items)
    assert len(kept) == 1
    assert "near-duplicate of S1" in items[1].reason


def test_source_weight_breaks_ties_in_favour_of_documents():
    text = "token budget context window"
    items = [item("memory", text + " memory"), item("doc", text + " doc")]
    kept = rank(Q, items)
    assert kept[0].source == "doc" and kept[0].id == "S1"


def test_budget_is_respected_and_small_items_fill_the_gap():
    # Distinct filler words: Jaccard works on word *sets*, so "filler " * 400
    # would collapse to one word and look like a near-copy of `small`.
    big = item("doc", "token budget context window " + " ".join(f"w{i}" for i in range(400)))
    small = item("paper", "context window token budget tradeoffs.")       # ~10 tok
    kept = rank(Q, [big, small], budget=100)
    assert kept == [small]
    assert big.reason.startswith("over budget")
    assert sum(k.tokens for k in kept) <= 100


def test_per_source_cap_keeps_the_window_diverse():
    docs = [item("doc", f"token budget context window variant {w}") for w in
            ["alpha one", "beta two", "gamma three", "delta four", "epsilon five", "zeta six"]]
    kept = rank(Q, docs)
    assert len(kept) == 4    # SETTINGS.max_per_source
    assert any("source cap" in d.reason for d in docs)


def test_every_item_gets_a_reason_and_kept_ids_are_sequential():
    items = [item("doc", "context window budget"), item("web", "token budget"),
             item("paper", "unrelated astronomy text")]
    kept = rank(Q, items)
    assert all(i.reason for i in items)
    assert [k.id for k in kept] == [f"S{n}" for n in range(1, len(kept) + 1)]
