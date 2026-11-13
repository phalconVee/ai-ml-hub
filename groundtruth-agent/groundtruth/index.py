"""Brick 3: a search engine in ~60 lines (BM25, written from scratch).

Why BM25 and not embeddings? It needs no extra API key or model download,
it is exact about numbers, codes and names ("Q3 2024", "EBITDA", "ISO-27001")
where embeddings get fuzzy, and the agent compensates for its weak spot
(synonyms) by rephrasing and searching again. That retry is the "agentic" in
agentic RAG.

BM25 in one paragraph: a passage scores high for a query word if the word is
*frequent in the passage* (term frequency, with diminishing returns set by k1)
and *rare across all passages* (inverse document frequency). Long passages
are penalised a little (b), so they can't win just by containing everything.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .chunk import Chunk

# A small stop-word list: words so common they carry no signal for search.
STOP = set("""a an and are as at be by for from has have in is it its of on or
that the this to was were will with what which who how when where why does did
do than then there these those not no can""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")  # keeps 3.5 and 1,200 as one token


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words/numbers, drop stop words, crude plural folding."""
    tokens = []
    for tok in TOKEN_RE.findall(text.lower()):
        if tok in STOP:
            continue
        # "revenues" -> "revenue", "tables" -> "table"; leave "analysis", "gas" alone.
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") and not tok.isdigit():
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self.term_freqs: list[Counter] = []  # per chunk: word -> count
        self.doc_freq: Counter = Counter()  # word -> how many chunks contain it
        self.lengths: list[int] = []

    def add(self, chunks: list[Chunk]) -> None:
        """Index more chunks (call once per uploaded document)."""
        for chunk in chunks:
            # Prefix the label so "table" or a page number is searchable too.
            tokens = tokenize(f"{chunk.kind} {chunk.text}")
            tf = Counter(tokens)
            self.chunks.append(chunk)
            self.term_freqs.append(tf)
            self.lengths.append(len(tokens))
            self.doc_freq.update(tf.keys())  # each word counted once per chunk

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def docs(self) -> list[str]:
        return sorted({c.doc for c in self.chunks})

    def _idf(self, term: str) -> float:
        # The "+1 inside the log" variant: always positive, even for common words.
        n, df = len(self.chunks), self.doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 6, doc: str | None = None) -> list[tuple[Chunk, float]]:
        """Return the top-k (chunk, score) pairs, best first. Optionally filter to one doc."""
        if not self.chunks:
            return []
        terms = tokenize(query)
        avg_len = sum(self.lengths) / len(self.lengths)
        scored = []
        for i, chunk in enumerate(self.chunks):
            if doc and chunk.doc != doc:
                continue
            tf, length = self.term_freqs[i], self.lengths[i]
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                # The BM25 formula: idf * saturating term frequency with length normalisation.
                norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / avg_len))
                score += self._idf(term) * norm
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:k]
