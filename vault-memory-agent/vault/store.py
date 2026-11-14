"""The storage interface: what the agent needs from a database, and nothing more.

The agent, the tools and the UI only ever talk to a `MemoryStore`. The one
implementation here is SQLite + sqlite-vec (`sqlite_store.py`), a single
local file. Swapping in a hosted database (MongoDB Atlas with $vectorSearch,
Postgres + pgvector, ...) means writing one more class with these methods;
nothing else changes. The README maps each method to its Atlas equivalent.

Three kinds of memory, three groups of methods:

1. Knowledge base: documents split into chunks, searched by vector.
2. Conversation history: threads, each an ordered list of messages.
3. Notes: short facts the agent itself chose to remember, searched by vector.

Vectors are passed in, never computed here: the store is plain storage, the
caller owns the embedding model (see `embeddings.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


# --- Records the store hands back --------------------------------------------
# Plain dataclasses, so the UI and tests never touch database rows directly.

@dataclass
class Document:
    id: int
    name: str
    n_chunks: int
    added_at: str


@dataclass
class ChunkHit:
    """One knowledge-base search result."""
    chunk_id: int
    doc_id: int
    doc_name: str
    location: str
    text: str
    score: float        # cosine similarity, 1.0 = same direction


@dataclass
class Thread:
    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass
class Message:
    id: int
    thread_id: str
    role: str           # "user" or "assistant"
    content: str        # the visible text only; tool calls live in `trace`
    trace: list = field(default_factory=list)   # what the agent did to produce it
    created_at: str = ""


@dataclass
class Note:
    id: int
    content: str
    kind: str           # fact | preference | task | other
    thread_id: Optional[str]
    created_at: str
    score: Optional[float] = None   # set only when returned by a search


# --- The interface -------------------------------------------------------------

class MemoryStore(Protocol):
    # 1. Knowledge base ---------------------------------------------------------
    def add_document(self, name: str, content_hash: str, chunks: list[tuple[str, str]],
                     vectors: list[list[float]]) -> tuple[Document, bool]:
        """Store a document's (location, text) chunks with their vectors.
        Returns (document, created); created is False if the same bytes were
        already stored (dedupe by content hash)."""
    def list_documents(self) -> list[Document]: ...
    def delete_document(self, doc_id: int) -> None: ...
    def search_chunks(self, vector: list[float], k: int) -> list[ChunkHit]: ...

    # 2. Conversation history --------------------------------------------------
    def create_thread(self, title: str) -> Thread: ...
    def list_threads(self) -> list[Thread]: ...
    def rename_thread(self, thread_id: str, title: str) -> None: ...
    def delete_thread(self, thread_id: str) -> None: ...
    def add_message(self, thread_id: str, role: str, content: str,
                    trace: Optional[list] = None) -> Message: ...
    def get_messages(self, thread_id: str, limit: Optional[int] = None) -> list[Message]:
        """Oldest first. With `limit`, only the most recent `limit` messages."""

    # 3. Notes -------------------------------------------------------------------
    def add_note(self, content: str, kind: str, vector: list[float],
                 thread_id: Optional[str], duplicate_distance: float) -> tuple[Note, bool]:
        """Returns (note, created). If a near-identical note exists, returns
        that one with created=False instead of storing a duplicate."""
    def search_notes(self, vector: list[float], k: int) -> list[Note]: ...
    def list_notes(self) -> list[Note]: ...
    def delete_note(self, note_id: int) -> bool:
        """True if a note was deleted, False if there was no such note."""

    def stats(self) -> dict[str, int]: ...
