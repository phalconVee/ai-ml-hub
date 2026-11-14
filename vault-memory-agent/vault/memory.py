"""`Vault`: a store plus an embedder, speaking in text instead of vectors.

The store only understands vectors; callers only have text. This small layer
sits between them: it chunks and embeds documents on the way in, embeds
queries on the way out, and nothing else. The tools and the UI use it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from .config import SETTINGS
from .embeddings import Embedder, make_embedder
from .ingest import chunk_file
from .sqlite_store import SqliteVecStore
from .store import ChunkHit, Document, MemoryStore, Note


class Vault:
    def __init__(self, store: MemoryStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    @classmethod
    def open(cls, path: str | Path = SETTINGS.db_path, embedder: Optional[Embedder] = None) -> "Vault":
        """The default wiring: a local SQLite file and the configured embedder.
        A hosted store would be constructed here instead."""
        embedder = embedder or make_embedder()
        return cls(SqliteVecStore(path, embedder.name, embedder.dim), embedder)

    # --- Memory 1: knowledge base ----------------------------------------------

    def ingest(self, filename: str, data: bytes) -> tuple[Document, bool]:
        """File bytes -> chunks -> vectors -> stored. Returns (doc, created)."""
        chunks = chunk_file(filename, data)
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        content_hash = hashlib.sha256(data).hexdigest()
        return self.store.add_document(filename, content_hash,
                                       [(c.location, c.text) for c in chunks], vectors)

    def search(self, query: str, k: int = SETTINGS.kb_top_k) -> list[ChunkHit]:
        return self.store.search_chunks(self.embedder.embed_query(query), k)

    # --- Memory 3: notes -------------------------------------------------------
    # (Memory 2, conversation history, needs no embedding: the agent reads the
    # store's thread methods directly.)

    def remember(self, content: str, kind: str = "fact", thread_id: Optional[str] = None) -> tuple[Note, bool]:
        # Notes are embedded as *documents* (what gets searched) and recall
        # queries as *queries*, matching how bge-style models were trained.
        vector = self.embedder.embed_documents([content])[0]
        return self.store.add_note(content, kind, vector, thread_id, SETTINGS.duplicate_distance)

    def recall(self, query: str, k: int = SETTINGS.recall_top_k) -> list[Note]:
        return self.store.search_notes(self.embedder.embed_query(query), k)
