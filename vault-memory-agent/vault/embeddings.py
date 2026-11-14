"""Turning text into vectors, locally.

Two embedders share one tiny interface (`name`, `dim`, `embed_documents`,
`embed_query`), so the store never knows which one it is talking to:

- `FastEmbedder`: a real sentence-embedding model (bge-small, 384 numbers per
  text) run on the CPU through ONNX by `fastembed`. Free, no key; the model
  downloads once (~70 MB).
- `HashEmbedder`: words hashed into buckets. It only matches shared words, not
  meaning, but needs no download, so the tests and a quick demo use it.

Every vector is L2-normalised, so cosine similarity = dot product, and the
database's cosine distance (1 - similarity) is meaningful for both.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .config import SETTINGS


class Embedder(Protocol):
    name: str   # stored in the database, so a DB is never queried with a different model
    dim: int    # vector length; the vector tables are created with this size

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class FastEmbedder:
    """bge-small-en-v1.5 via fastembed (local ONNX runtime)."""

    def __init__(self, model_name: str = SETTINGS.embed_model):
        # Imported here so the hash embedder (and the tests) never load ONNX.
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.name = f"fastembed:{model_name}"
        # Ask the model for one vector to learn its size, rather than hard-coding 384.
        self.dim = len(next(iter(self._model.embed(["probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # bge models are trained with different prefixes for passages and
        # queries; fastembed's passage_embed / query_embed add them for us.
        return [_normalise(v.tolist()) for v in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return _normalise(next(iter(self._model.query_embed(text))).tolist())


_WORD = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Bag-of-words hashed into `dim` buckets (the "hashing trick").

    Each word adds +1 or -1 to one bucket chosen by its hash. Texts sharing
    words point in similar directions. No semantics ("car" ≠ "automobile"),
    but deterministic and instant, which is exactly what a test wants.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash:{dim}"

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return _normalise(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def make_embedder(kind: str = SETTINGS.embedder) -> Embedder:
    if kind == "hash":
        return HashEmbedder()
    if kind == "fastembed":
        return FastEmbedder()
    raise ValueError(f"Unknown VAULT_EMBEDDER {kind!r}; use 'fastembed' or 'hash'.")
