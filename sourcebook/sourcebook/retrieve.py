"""Step 3: find the passages that answer a question.

Two retrievers, fused:

1. **BM25** (always on, written from scratch below). Keyword search: exact on
   names, numbers and jargon, needs no model, instant to build.
2. **Dense embeddings** (optional, `uv sync --extra dense`). A small local
   ONNX model (fastembed, bge-small) turns passages and the query into vectors;
   cosine similarity finds paraphrases BM25 misses ("car" vs "automobile").

The two produce scores on different scales, so we don't add scores. We fuse
*ranks* with Reciprocal Rank Fusion: each passage gets sum(1 / (60 + rank))
over the lists it appears in. A passage ranked well by both wins; one ranked
first by either still makes the cut. No tuning, no score normalisation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Optional, Protocol

from .chunk import Passage
from .config import SETTINGS

_STOPWORDS = set("""
a an and are as at be been but by can did do does for from had has have how i if in into
is it its of on or our so than that the their them then there these they this to was we
were what when where which who why will with you your about after also not no more most
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without stopwords. A crude plural fold ("models"
    -> "model") lets singular and plural match without a stemmer library."""
    out = []
    for tok in re.findall(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*", text.lower()):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


class BM25:
    """Okapi BM25 over a fixed list of passages.

    score(q, d) = sum over query terms t of
        idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

    - idf rewards rare terms ("transformer" beats "model");
    - the tf fraction saturates, so repeating a word 50 times doesn't win;
    - b normalises for length, so long passages don't win by being long.
    """

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(p.text)) for p in passages]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        # The "+1" inside the log keeps idf positive even for very common terms.
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def rank(self, query: str) -> list[tuple[int, float]]:
        """(passage index, score) for every passage with score > 0, best first."""
        terms = tokenize(query)
        scores = []
        for i, doc in enumerate(self.docs):
            s = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1))
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            if s > 0:
                scores.append((i, s))
        return sorted(scores, key=lambda x: -x[1])


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    """Local embeddings via fastembed (ONNX). Only built if the extra is installed."""

    def __init__(self, model: str = SETTINGS.embed_model):
        from fastembed import TextEmbedding  # optional dependency
        self._model = TextEmbedding(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]


def try_dense_embedder() -> Optional[Embedder]:
    """Return a FastEmbedder if fastembed is installed and loads, else None."""
    try:
        return FastEmbedder()
    except Exception:   # not installed, or the model download failed
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def rrf(rankings: Iterable[list[int]], k: int = SETTINGS.rrf_k) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion of several best-first lists of passage indices."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


class Retriever:
    """Holds every passage of a notebook and searches them."""

    def __init__(self, passages: list[Passage], embedder: Optional[Embedder] = None):
        self.passages = passages
        self.bm25 = BM25(passages)
        self.embedder = embedder
        # Embed every passage once, up front; queries are embedded per search.
        self.vectors = embedder.embed([p.text for p in passages]) if (embedder and passages) else []

    @property
    def mode(self) -> str:
        return "hybrid (BM25 + dense, RRF)" if self.vectors else "BM25"

    def search(self, query: str, k: int = SETTINGS.top_k,
               allowed_sources: Optional[set[str]] = None) -> list[Passage]:
        def ok(i: int) -> bool:   # respect the "include this source" checkboxes
            return allowed_sources is None or self.passages[i].source_id in allowed_sources

        # Take a deeper list from each retriever than we need, then fuse.
        depth = k * 4
        lists = [[i for i, _ in self.bm25.rank(query) if ok(i)][:depth]]
        if self.vectors:
            q = self.embedder.embed([query])[0]
            sims = sorted(((i, _cosine(q, v)) for i, v in enumerate(self.vectors) if ok(i)),
                          key=lambda x: -x[1])
            lists.append([i for i, _ in sims[:depth]])
        return [self.passages[i] for i, _ in rrf(lists)[:k]]
