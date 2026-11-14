"""Step 2: a keyword search index over clauses (Okapi BM25, written by hand).

Why keywords and not embeddings? Legal text is unusually literal. The words
that matter ("indemnify", "terminate", "30 days", "Landlord") are the exact
words in the clause, and a lawyer's question usually reuses them. BM25 needs
no extra API key, runs in microseconds, and every score can be printed and
explained. The Clause Finder agent covers the synonym gap by re-phrasing its
searches ("end the lease" → "termination notice").

For one query word t and one clause d, BM25 adds:

    idf(t) · tf·(k1 + 1) / (tf + k1·(1 − b + b·len(d)/avg_len))

    idf  rare words count more ("indemnify" beats "party")
    tf   repeats help, with diminishing returns (k1 sets the ceiling)
    len  long clauses are discounted so they don't win by sheer size (b)
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .ingest import Chunk

# Words too common to carry signal. Note what is NOT here: "not", "no",
# "shall", "may", "must". In a contract those words *are* the meaning.
STOPWORDS = frozenset(
    "a an and are as at be been by for from has have in into is it its of on or "
    "that the their them this these those to was were which with".split()
)


def tokenize(text: str) -> list[str]:
    """'The Tenant's deposit (USD 2,000)' → ['tenant', 's', 'deposit', 'usd', '2', '000']."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class ClauseIndex:
    def __init__(self, k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self.by_id: dict[str, Chunk] = {}

    # ── building ────────────────────────────────────────────────────────────
    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks and recompute the statistics (cheap at this scale)."""
        self.chunks.extend(chunks)
        self.by_id.update({c.id: c for c in chunks})
        # Index the heading as well as the body: "Termination" in a heading
        # is strong evidence the clause is about termination.
        self.tfs = [Counter(tokenize(c.text)) for c in self.chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lens) / len(self.lens) if self.lens else 0.0
        df = Counter(t for tf in self.tfs for t in tf)          # docs containing t
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @property
    def docs(self) -> list[str]:
        return sorted({c.doc for c in self.chunks})

    def __len__(self) -> int:
        return len(self.chunks)

    # ── querying ────────────────────────────────────────────────────────────
    def _score(self, terms: list[str], i: int) -> float:
        tf, length = self.tfs[i], self.lens[i]
        norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
        return sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                   for t in terms if t in tf)

    def search(self, query: str, k: int = 5, doc: str | None = None) -> list[tuple[Chunk, float]]:
        """Top-k (chunk, score), best first. `doc` limits the search to one file."""
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []
        scored = [(c, self._score(terms, i)) for i, c in enumerate(self.chunks)
                  if doc is None or c.doc == doc]
        scored = [(c, s) for c, s in scored if s > 0]   # 0 = no shared word
        scored.sort(key=lambda cs: cs[1], reverse=True)
        return scored[:k]
