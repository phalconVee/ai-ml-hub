"""Keyword search written from scratch: a tokenizer and Okapi BM25.

Why BM25 and not embeddings? Memories are short sentences like
"User is allergic to peanuts". Keyword overlap works well on text that short,
needs no model download and no extra key, and every score can be explained
by hand. The recall planner (recall_planner.py) bridges the vocabulary gap
by turning "what should I cook?" into keywords like "food diet allergy".
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Words so common they say nothing about what a memory is *about*.
STOPWORDS = frozenset("""
a an and are as at be been but by for from has have he her his i if in into is it
its me my of on or our she so that the their them they this to was we were what
when where which who will with you your do does did not no yes can could would
should about just than then there these those am im i'm
""".split())

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase words, stopwords removed, with a very light plural fold.

    A real stemmer is overkill here, so we only strip a trailing "s" from
    words longer than 3 letters: "peanuts" → "peanut", "dogs" → "dog",
    while "bus", "class" and "is" stay put. Crude, predictable, enough.
    """
    words = _WORD.findall(text.lower())
    out = []
    for w in words:
        if w in STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


class BM25:
    """Okapi BM25 over a small list of documents (here: one per memory).

    score(q, d) = Σ over query terms t of
        idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))

    - idf rewards rare words: "peanut" says more than "like".
    - tf saturates (k1): saying "coffee" three times isn't 3× as relevant.
    - b normalises for length: a long memory shouldn't win just by being long.
    """

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.tfs = [Counter(d) for d in self.docs]
        n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / n) if n else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))  # document frequency: in how many docs a term appears
        # The "+1" inside the log keeps idf positive even for very common terms.
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        """Raw BM25 score of `query` against every document, in order."""
        q = tokenize(query)
        out = []
        for doc, tf in zip(self.docs, self.tfs):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                norm = 1 - self.b + self.b * (len(doc) / self.avgdl if self.avgdl else 1)
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * norm)
            out.append(s)
        return out

    def normalized(self, query: str) -> list[float]:
        """Scores divided by the best score, so they sit in [0, 1].

        That makes BM25 combinable with the other retrieval signals, which
        are already in [0, 1]. If nothing matches at all, everything is 0.
        """
        raw = self.scores(query)
        top = max(raw, default=0.0)
        return [s / top if top > 0 else 0.0 for s in raw]
