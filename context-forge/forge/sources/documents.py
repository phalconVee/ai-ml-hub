"""Source 1: the user's own documents (PDF, Markdown, plain text).

Load → chunk → BM25 index, all in memory. `search` returns the best chunks as
ContextItems. Nothing leaves the machine at this stage.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from ..config import SETTINGS
from ..items import ContextItem
from ..textsearch import BM25, chunk_words


def read_file(name: str, data: bytes) -> str:
    """Bytes of an uploaded file → plain text."""
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader  # lazy: only needed when a PDF shows up

        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="replace")  # .md / .txt


@dataclass
class Chunk:
    doc: str      # file name
    n: int        # chunk number inside that file (1-based, shown to the user)
    text: str


class DocumentStore:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self._index: BM25 | None = None   # rebuilt lazily after each add

    def add(self, name: str, text: str) -> int:
        """Chunk one document into the store; returns how many chunks it made."""
        pieces = chunk_words(text, SETTINGS.doc_chunk_words, SETTINGS.doc_overlap_words)
        self.chunks += [Chunk(name, i + 1, p) for i, p in enumerate(pieces)]
        self._index = None  # stale now
        return len(pieces)

    @property
    def names(self) -> list[str]:
        return sorted({c.doc for c in self.chunks})

    def search(self, query: str, k: int | None = None) -> list[ContextItem]:
        if not self.chunks:
            return []
        if self._index is None:
            self._index = BM25([c.text for c in self.chunks])
        k = k or SETTINGS.per_source_candidates
        return [
            ContextItem(
                source="doc",
                title=f"{self.chunks[i].doc} · chunk {self.chunks[i].n}",
                text=self.chunks[i].text,
                meta={"bm25": round(score, 2)},
            )
            for i, score in self._index.top(query, k)
        ]
