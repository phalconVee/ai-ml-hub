"""Tokenising, chunking and BM25: the lexical machinery shared by every stage.

BM25 scores how well a passage d matches a query q:

    score(d, q) = Σ_t  IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))

  IDF(t)       rare words count more than common ones
  tf           repeats help, with diminishing returns (k1)
  |d|/avgdl    long passages are nudged down so length alone doesn't win (b)

We use it twice: inside the document and memory sources to *retrieve*, and in
the ranker to put candidates from every source on one relevance scale.
No embeddings, so no extra API key, and every number is inspectable.
"""

from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its "
    "of on or our so than that the their them then there these this to was we were "
    "what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """'How does RoPE scale?' → ['rope', 'scale']  (lower-case, no stop-words)."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into windows of `size` words that overlap by `overlap` words."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break  # the last window already reached the end
    return chunks


def jaccard(a: str, b: str) -> float:
    """Word-set overlap in [0, 1]. Used to spot near-duplicate passages."""
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class BM25:
    """A tiny in-memory BM25 index over a list of strings."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(t)) for t in texts]          # term counts per passage
        self.lens = [sum(tf.values()) for tf in self.tfs]          # |d|
        self.avgdl = (sum(self.lens) / len(texts)) if texts else 0.0
        df: Counter[str] = Counter()                               # passages containing t
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(texts)
        # "+1" inside the log keeps IDF positive for very common words.
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        """One score per passage, in the order the passages were given."""
        terms = tokenize(query)
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t in terms:
                if t in tf:
                    norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                    s += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
            out.append(s)
        return out

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        """(index, score) of the k best passages with a non-zero score."""
        ranked = sorted(enumerate(self.scores(query)), key=lambda p: p[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0]
