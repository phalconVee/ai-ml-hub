"""`MemoryStore` on SQLite + sqlite-vec: all three memories in one local file.

Why this pairing: SQLite is in Python's standard library and needs no server;
sqlite-vec is a small extension that adds `vec0` virtual tables, which store
float vectors and answer "k nearest neighbours" queries. So relational data
(threads, messages, document names) and vectors live side by side, in one
transaction, in one file you can copy, back up or delete.

Layout (ordinary tables on the left, vector tables on the right):

    documents ──< chunks ─────── chunk_vectors (vec0: chunk_id, embedding)
    threads   ──< messages
    notes ────────────────────── note_vectors  (vec0: note_id, embedding)
    meta  (which embedding model filled the vectors, and its size)

Each vector row shares its primary key with the row it describes, so a KNN
query returns ids that we join back to the text.

sqlite-vec's search is exact (it compares the query with every vector).
That's fine into the hundreds of thousands of chunks; beyond that you'd move
to a store with an approximate index, which is what the interface is for.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sqlite_vec

from .store import ChunkHit, Document, Message, Note, Thread


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

-- 1. Knowledge base -------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,      -- same bytes uploaded twice = one document
    n_chunks     INTEGER NOT NULL,
    added_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id       INTEGER PRIMARY KEY,           -- also the key of its row in chunk_vectors
    doc_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,              -- order within the document
    location TEXT NOT NULL,                 -- "p. 3", "§2", ...
    text     TEXT NOT NULL
);

-- 2. Conversation history --------------------------------------------------
CREATE TABLE IF NOT EXISTS threads (
    id         TEXT PRIMARY KEY,            -- a uuid, safe to put in a URL
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY,         -- autoincrementing, so it orders the thread
    thread_id  TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    trace      TEXT NOT NULL DEFAULT '[]',  -- JSON: the tool calls behind an answer
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_thread ON messages(thread_id, id);

-- 3. Notes the agent wrote ----------------------------------------------------
CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY,         -- also the key of its row in note_vectors
    content    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    thread_id  TEXT,                        -- where it was learned (kept if the thread is deleted)
    created_at TEXT NOT NULL
);
"""


class SqliteVecStore:
    def __init__(self, path: str | Path, embedder_name: str, dim: int):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: Streamlit may call us from different worker
        # threads across reruns. The lock below serialises access instead.
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._lock = threading.RLock()

        # Load the sqlite-vec extension into this connection, then switch
        # extension loading back off so nothing else can load one.
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)

        self.db.execute("PRAGMA foreign_keys = ON")   # make ON DELETE CASCADE work
        self.db.execute("PRAGMA journal_mode = WAL")  # readers don't block the writer
        with self.db:
            self.db.executescript(SCHEMA)
            self._check_embedder(embedder_name, dim)
            # vec0 tables need the vector size up front, so they're created
            # only once we know which embedder this database belongs to.
            # distance_metric=cosine: `distance` = 1 - cosine similarity.
            self.db.execute(f"""CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id INTEGER PRIMARY KEY, embedding float[{dim}] distance_metric=cosine)""")
            self.db.execute(f"""CREATE VIRTUAL TABLE IF NOT EXISTS note_vectors USING vec0(
                note_id INTEGER PRIMARY KEY, embedding float[{dim}] distance_metric=cosine)""")

    def _check_embedder(self, name: str, dim: int) -> None:
        """Pin the database to one embedding model.

        Vectors from different models live in different spaces; comparing a
        bge query with hash-embedded chunks would return nonsense silently.
        So the first embedder to open a database owns it.
        """
        row = self.db.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
        if row is None:
            self.db.execute("INSERT INTO meta VALUES ('embedder', ?)", (json.dumps([name, dim]),))
        elif json.loads(row["value"]) != [name, dim]:
            stored = json.loads(row["value"])[0]
            raise RuntimeError(
                f"This database was built with embedder {stored!r}, not {name!r}. "
                "Use the same VAULT_EMBEDDER, or point VAULT_DB at a new file.")

    # ------------------------------------------------------------------------
    # 1. Knowledge base
    # ------------------------------------------------------------------------

    def add_document(self, name, content_hash, chunks, vectors):
        assert len(chunks) == len(vectors), "one vector per chunk"
        with self._lock, self.db:   # `with self.db` = one transaction: all or nothing
            existing = self.db.execute(
                "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)).fetchone()
            if existing:
                return self._document(existing), False
            cur = self.db.execute(
                "INSERT INTO documents (name, content_hash, n_chunks, added_at) VALUES (?, ?, ?, ?)",
                (name, content_hash, len(chunks), _now()))
            doc_id = cur.lastrowid
            for position, ((location, text), vector) in enumerate(zip(chunks, vectors)):
                chunk_id = self.db.execute(
                    "INSERT INTO chunks (doc_id, position, location, text) VALUES (?, ?, ?, ?)",
                    (doc_id, position, location, text)).lastrowid
                # serialize_float32 packs the list into the compact binary
                # format vec0 stores (4 bytes per number).
                self.db.execute(
                    "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, sqlite_vec.serialize_float32(vector)))
            row = self.db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            return self._document(row), True

    def list_documents(self):
        with self._lock:
            rows = self.db.execute("SELECT * FROM documents ORDER BY id").fetchall()
        return [self._document(r) for r in rows]

    def delete_document(self, doc_id):
        with self._lock, self.db:
            # Virtual tables don't take part in foreign keys, so the vectors
            # are deleted by hand before the cascade removes the chunks.
            self.db.execute(
                "DELETE FROM chunk_vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)",
                (doc_id,))
            self.db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def search_chunks(self, vector, k):
        with self._lock:
            # Step 1, the vector search: `embedding MATCH ? AND k = ?` is
            # sqlite-vec's KNN syntax. It runs inside the CTE, so `k` limits
            # the neighbours before we join anything.
            # Step 2, an ordinary join from ids back to text and doc name.
            rows = self.db.execute("""
                WITH nearest AS (
                    SELECT chunk_id, distance FROM chunk_vectors
                    WHERE embedding MATCH ? AND k = ?
                )
                SELECT c.id, c.doc_id, d.name, c.location, c.text, n.distance
                FROM nearest n
                JOIN chunks c ON c.id = n.chunk_id
                JOIN documents d ON d.id = c.doc_id
                ORDER BY n.distance
            """, (sqlite_vec.serialize_float32(vector), k)).fetchall()
        return [ChunkHit(chunk_id=r[0], doc_id=r[1], doc_name=r[2], location=r[3], text=r[4],
                         score=round(1.0 - r[5], 4)) for r in rows]

    @staticmethod
    def _document(row) -> Document:
        return Document(id=row["id"], name=row["name"], n_chunks=row["n_chunks"], added_at=row["added_at"])

    # ------------------------------------------------------------------------
    # 2. Conversation history
    # ------------------------------------------------------------------------

    def create_thread(self, title):
        thread = Thread(id=uuid.uuid4().hex, title=title, created_at=_now(), updated_at=_now())
        with self._lock, self.db:
            self.db.execute("INSERT INTO threads VALUES (?, ?, ?, ?)",
                            (thread.id, thread.title, thread.created_at, thread.updated_at))
        return thread

    def list_threads(self):
        # Most recently active first: that's the order a thread picker wants.
        with self._lock:
            rows = self.db.execute("SELECT * FROM threads ORDER BY updated_at DESC, rowid DESC").fetchall()
        return [Thread(**dict(r)) for r in rows]

    def rename_thread(self, thread_id, title):
        with self._lock, self.db:
            self.db.execute("UPDATE threads SET title = ? WHERE id = ?", (title, thread_id))

    def delete_thread(self, thread_id):
        with self._lock, self.db:
            self.db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))   # messages cascade

    def add_message(self, thread_id, role, content, trace=None):
        created = _now()
        with self._lock, self.db:
            msg_id = self.db.execute(
                "INSERT INTO messages (thread_id, role, content, trace, created_at) VALUES (?, ?, ?, ?, ?)",
                (thread_id, role, content, json.dumps(trace or []), created)).lastrowid
            self.db.execute("UPDATE threads SET updated_at = ? WHERE id = ?", (created, thread_id))
        return Message(id=msg_id, thread_id=thread_id, role=role, content=content,
                       trace=trace or [], created_at=created)

    def get_messages(self, thread_id, limit=None):
        with self._lock:
            if limit is None:
                rows = self.db.execute(
                    "SELECT * FROM messages WHERE thread_id = ? ORDER BY id", (thread_id,)).fetchall()
            else:
                # Take the newest `limit` (DESC), then flip back to oldest-first.
                rows = self.db.execute(
                    "SELECT * FROM messages WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
                    (thread_id, limit)).fetchall()[::-1]
        return [Message(id=r["id"], thread_id=r["thread_id"], role=r["role"], content=r["content"],
                        trace=json.loads(r["trace"]), created_at=r["created_at"]) for r in rows]

    # ------------------------------------------------------------------------
    # 3. Notes
    # ------------------------------------------------------------------------

    def add_note(self, content, kind, vector, thread_id, duplicate_distance):
        blob = sqlite_vec.serialize_float32(vector)
        with self._lock, self.db:
            # Before writing, look for the single nearest existing note. If it
            # is (almost) the same direction, the agent is repeating itself.
            nearest = self.db.execute(
                "SELECT note_id, distance FROM note_vectors WHERE embedding MATCH ? AND k = 1",
                (blob,)).fetchone()
            if nearest and nearest["distance"] <= duplicate_distance:
                return self._note(self.db.execute(
                    "SELECT * FROM notes WHERE id = ?", (nearest["note_id"],)).fetchone()), False
            note_id = self.db.execute(
                "INSERT INTO notes (content, kind, thread_id, created_at) VALUES (?, ?, ?, ?)",
                (content, kind, thread_id, _now())).lastrowid
            self.db.execute("INSERT INTO note_vectors (note_id, embedding) VALUES (?, ?)", (note_id, blob))
            return self._note(self.db.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()), True

    def search_notes(self, vector, k):
        with self._lock:
            rows = self.db.execute("""
                WITH nearest AS (
                    SELECT note_id, distance FROM note_vectors WHERE embedding MATCH ? AND k = ?
                )
                SELECT n.*, nearest.distance FROM nearest JOIN notes n ON n.id = nearest.note_id
                ORDER BY nearest.distance
            """, (sqlite_vec.serialize_float32(vector), k)).fetchall()
        return [self._note(r, score=round(1.0 - r["distance"], 4)) for r in rows]

    def list_notes(self):
        with self._lock:
            rows = self.db.execute("SELECT * FROM notes ORDER BY id DESC").fetchall()
        return [self._note(r) for r in rows]

    def delete_note(self, note_id):
        with self._lock, self.db:
            self.db.execute("DELETE FROM note_vectors WHERE note_id = ?", (note_id,))
            return self.db.execute("DELETE FROM notes WHERE id = ?", (note_id,)).rowcount > 0

    @staticmethod
    def _note(row, score: Optional[float] = None) -> Note:
        return Note(id=row["id"], content=row["content"], kind=row["kind"], thread_id=row["thread_id"],
                    created_at=row["created_at"], score=score)

    # ------------------------------------------------------------------------

    def stats(self):
        with self._lock:
            count = lambda table: self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: E731
            return {t: count(t) for t in ("documents", "chunks", "threads", "messages", "notes")}

    def close(self):
        self.db.close()
