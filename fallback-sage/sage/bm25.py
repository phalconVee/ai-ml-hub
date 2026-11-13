"""BM25 keyword search, written from scratch (no vector DB, no embeddings API).

BM25 is the ranking formula behind classic search engines. For a query q and a
chunk d it adds up, over every query word t:

    score(d, q) = Σ  IDF(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )

  IDF(t)  rare words matter more ("pgvector" > "the")
  tf      a word appearing more often in the chunk helps, but with diminishing
          returns (k1 caps how much)
  |d|/avgdl  long chunks are penalised a little, so they don't win just by
          containing more words (b sets how much)

Why not embeddings? For "does my document mention X?" questions, exact-term
matching is strong, needs no extra API key, and every number can be printed
and checked by hand. The LLM covers the synonym gap by rewriting queries.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .documents import Chunk

# A tiny stop-word list: words so common they carry no signal.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or "
    "that the this to was were what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case, keep letters/digits, drop stop-words: 'The GPU's VRAM' → ['gpu', 's', 'vram']."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


@dataclass
class Hit:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        # Pre-compute everything that doesn't depend on the query.
        self.term_freqs = [Counter(tokenize(c.text)) for c in chunks]  # tf per chunk
        self.lengths = [sum(tf.values()) for tf in self.term_freqs]    # |d|
        self.avgdl = (sum(self.lengths) / len(chunks)) if chunks else 0.0
        # Document frequency: in how many chunks does each word appear?
        df: Counter[str] = Counter()
        for tf in self.term_freqs:
            df.update(tf.keys())
        n = len(chunks)
        # The "+1" inside the log keeps IDF positive even for words in >half the chunks.
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def __len__(self) -> int:
        return len(self.chunks)

    def score(self, query_terms: list[str], i: int) -> float:
        tf, dl = self.term_freqs[i], self.lengths[i]
        s = 0.0
        for t in query_terms:
            if t not in tf:
                continue  # word absent → contributes nothing
            f = tf[t]
            norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return s

    def search(self, query: str, k: int = 5) -> list[Hit]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []
        scored = [Hit(c, self.score(terms, i)) for i, c in enumerate(self.chunks)]
        scored = [h for h in scored if h.score > 0]  # zero = shares no word with the query
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]
