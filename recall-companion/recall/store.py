"""The temporal knowledge store: one SQLite file, five tables.

This is the "long-term memory" of the assistant, written by hand so the whole
mechanism is visible. The shape is a tiny knowledge graph with time on it:

    entities   the things memories are about: the user, people, places, ...
    facts      edges between them: (subject) —predicate→ (object), plus a
               plain-English `statement`, and a validity window in time
    recalls    which facts were shown to the model for which answer

and the conversation itself:

    profiles   one per person using the app; memories never cross profiles
    sessions   separate chats of one profile; memories DO cross sessions
    messages   every user/assistant message, per session

The time model has two clocks, the same idea used by temporal databases:

    valid_from / valid_to   when the fact was TRUE in the user's world
    recorded_at             when WE learned it

A fact is never overwritten when the world changes. "Lives in Berlin" gets a
`valid_to` and a `superseded_by` pointer when "lives in Lisbon" arrives, so
the store can still answer "where did I live before?" and the inspector can
show the history. Only an explicit user delete removes a row.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# Words the extractor (and people) use for "the person I'm talking to".
# All of them resolve to the profile's own user entity.
USER_ALIASES = frozenset({"user", "the user", "i", "me", "myself", "you"})

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY,
    profile_id  INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY,
    profile_id  INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,               -- as first written: "Maya"
    key         TEXT NOT NULL,               -- normalised for matching: "maya"
    kind        TEXT NOT NULL,               -- user/person/place/org/pet/thing/...
    created_at  TEXT NOT NULL,
    UNIQUE (profile_id, key)                 -- one "Maya" per profile
);
CREATE TABLE IF NOT EXISTS facts (
    id                INTEGER PRIMARY KEY,
    profile_id        INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    subject_id        INTEGER NOT NULL REFERENCES entities(id),
    predicate         TEXT NOT NULL,         -- snake_case relation: lives_in
    object            TEXT NOT NULL,         -- the value, as text: "Lisbon"
    object_id         INTEGER REFERENCES entities(id),  -- set when the value is an entity
    statement         TEXT NOT NULL,         -- one readable sentence; what we search
    category          TEXT NOT NULL,         -- preference/person/event/goal/biographical/...
    importance        INTEGER NOT NULL,      -- 1 (trivia) .. 5 (always relevant)
    exclusive         INTEGER NOT NULL,      -- 1 = only one value at a time (lives_in)
    occurred_at       TEXT,                  -- for events: when it happens/happened
    valid_from        TEXT NOT NULL,         -- became true (world time)
    valid_to          TEXT,                  -- stopped being true; NULL = still true
    recorded_at       TEXT NOT NULL,         -- when we learned it (system time)
    last_confirmed_at TEXT NOT NULL,         -- last time the user said it again
    superseded_by     INTEGER REFERENCES facts(id),
    status            TEXT NOT NULL DEFAULT 'active',  -- active | superseded
    origin            TEXT NOT NULL,         -- conversation | user edit
    source_session_id INTEGER,
    source_message_id INTEGER
);
CREATE TABLE IF NOT EXISTS recalls (
    message_id  INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    fact_id     INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    rank        INTEGER NOT NULL,
    score       REAL NOT NULL,
    PRIMARY KEY (message_id, fact_id)
);
CREATE INDEX IF NOT EXISTS facts_by_subject ON facts(profile_id, subject_id, predicate, status);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def norm_key(name: str) -> str:
    """Entity matching key: lowercase, single spaces. "  Maya " → "maya"."""
    return " ".join(name.lower().split())


def norm_predicate(p: str) -> str:
    """Relations are snake_case so "Lives in" and "lives_in" are the same edge."""
    return "_".join(p.lower().replace("-", " ").split()) or "related_to"


@dataclass
class NewFact:
    """A fact as the extractor (or the user) proposes it, before it is stored."""
    subject: str                      # entity name, or "user"
    predicate: str
    object: str
    statement: str
    category: str = "biographical"
    importance: int = 3
    exclusive: bool = False
    object_is_entity: bool = False    # is the object itself a person/place/...?
    subject_kind: str = "thing"
    object_kind: str = "thing"
    occurred_at: str | None = None
    replaces: list[int] = field(default_factory=list)  # fact ids it contradicts


@dataclass
class Fact:
    """A stored fact, joined with its entity names for display and search."""
    id: int
    subject_id: int
    subject: str
    predicate: str
    object: str
    object_id: int | None
    statement: str
    category: str
    importance: int
    exclusive: bool
    occurred_at: str | None
    valid_from: str
    valid_to: str | None
    recorded_at: str
    last_confirmed_at: str
    superseded_by: int | None
    status: str
    origin: str
    source_session_id: int | None
    source_message_id: int | None

    @property
    def active(self) -> bool:
        return self.status == "active"


@dataclass
class WriteResult:
    """What one memory write did, so the UI can say it out loud."""
    action: str                        # added | confirmed | edited
    fact: Fact
    superseded: list[Fact] = field(default_factory=list)


class MemoryStore:
    """All reads and writes of memory go through here.

    `clock` is injectable so tests can move time forward ("a week later")
    without sleeping. `check_same_thread=False` because Streamlit may serve
    reruns from different threads; we only ever use one connection per store.
    """

    def __init__(self, path: str | Path = ":memory:", clock: Callable[[], datetime] = utcnow):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self.clock = clock

    def now(self) -> str:
        return self.clock().isoformat()

    # ── Profiles & sessions ──────────────────────────────────────────────────
    def get_or_create_profile(self, name: str) -> int:
        name = name.strip()
        row = self.db.execute("SELECT id FROM profiles WHERE name = ?", (name,)).fetchone()
        if row:
            return row["id"]
        with self.db:
            pid = self.db.execute("INSERT INTO profiles (name, created_at) VALUES (?, ?)",
                                  (name, self.now())).lastrowid
            # Every profile starts with one entity: the user themself. Facts
            # like "is vegetarian" hang off this node.
            self.db.execute("INSERT INTO entities (profile_id, name, key, kind, created_at) "
                            "VALUES (?, ?, ?, 'user', ?)", (pid, name, "user", self.now()))
        return pid

    def profiles(self) -> list[tuple[int, str]]:
        return [(r["id"], r["name"]) for r in
                self.db.execute("SELECT id, name FROM profiles ORDER BY name")]

    def new_session(self, profile_id: int, title: str | None = None) -> int:
        n = self.db.execute("SELECT COUNT(*) FROM sessions WHERE profile_id = ?",
                            (profile_id,)).fetchone()[0]
        with self.db:
            return self.db.execute(
                "INSERT INTO sessions (profile_id, title, created_at) VALUES (?, ?, ?)",
                (profile_id, title or f"Session {n + 1}", self.now())).lastrowid

    def sessions(self, profile_id: int) -> list[tuple[int, str, str]]:
        """(id, title, created_at), newest first."""
        return [(r["id"], r["title"], r["created_at"]) for r in self.db.execute(
            "SELECT id, title, created_at FROM sessions WHERE profile_id = ? ORDER BY id DESC",
            (profile_id,))]

    # ── Messages (short-term memory) ─────────────────────────────────────────
    def add_message(self, session_id: int, role: str, content: str) -> int:
        with self.db:
            return self.db.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, self.now())).lastrowid

    def messages(self, session_id: int, last: int | None = None) -> list[dict]:
        """Messages of one session in order; `last=N` keeps only the newest N."""
        rows = self.db.execute(
            "SELECT id, role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,)).fetchall()
        rows = rows[-last:] if last else rows
        return [dict(r) for r in rows]

    # ── Entities (the nodes) ─────────────────────────────────────────────────
    def user_entity(self, profile_id: int) -> int:
        return self.db.execute("SELECT id FROM entities WHERE profile_id = ? AND key = 'user'",
                               (profile_id,)).fetchone()["id"]

    def resolve_entity(self, profile_id: int, name: str, kind: str = "thing") -> int:
        """Find an entity by name or create it. "I"/"me"/"user" → the user node.

        This is the whole of our entity resolution: exact match on a
        normalised key. Real systems also merge "Mom" and "my mother"; we let
        the extractor do that by showing it the names that already exist.
        """
        key = norm_key(name)
        if key in USER_ALIASES:
            return self.user_entity(profile_id)
        row = self.db.execute("SELECT id FROM entities WHERE profile_id = ? AND key = ?",
                              (profile_id, key)).fetchone()
        if row:
            return row["id"]
        with self.db:
            return self.db.execute(
                "INSERT INTO entities (profile_id, name, key, kind, created_at) VALUES (?, ?, ?, ?, ?)",
                (profile_id, name.strip(), key, kind or "thing", self.now())).lastrowid

    def entities(self, profile_id: int) -> list[dict]:
        """Every entity with how many still-true facts touch it."""
        return [dict(r) for r in self.db.execute("""
            SELECT e.id, e.name, e.kind, e.key,
                   (SELECT COUNT(*) FROM facts f WHERE f.status = 'active'
                      AND (f.subject_id = e.id OR f.object_id = e.id)) AS active_facts
            FROM entities e WHERE e.profile_id = ? ORDER BY e.kind = 'user' DESC, e.name
        """, (profile_id,))]

    # ── Facts (the edges) ────────────────────────────────────────────────────
    _FACT_SELECT = """
        SELECT f.*, s.name AS subject FROM facts f JOIN entities s ON s.id = f.subject_id
    """

    def _fact(self, row: sqlite3.Row) -> Fact:
        d = dict(row)
        d.pop("profile_id")
        d["exclusive"] = bool(d["exclusive"])
        return Fact(**d)

    def get_fact(self, fact_id: int) -> Fact | None:
        row = self.db.execute(self._FACT_SELECT + " WHERE f.id = ?", (fact_id,)).fetchone()
        return self._fact(row) if row else None

    def facts(self, profile_id: int, *, include_superseded: bool = False) -> list[Fact]:
        """All facts of a profile, newest first. By default only the true-now ones."""
        where = "f.profile_id = ?" + ("" if include_superseded else " AND f.status = 'active'")
        return [self._fact(r) for r in self.db.execute(
            self._FACT_SELECT + f" WHERE {where} ORDER BY f.id DESC", (profile_id,))]

    def add_fact(self, profile_id: int, new: NewFact, *, origin: str = "conversation",
                 session_id: int | None = None, message_id: int | None = None) -> WriteResult:
        """Write one fact, applying the two rules that make memory *temporal*.

        Rule 1, confirmation: if the same (subject, predicate, object) is
        already true, don't store a duplicate; just note that the user said
        it again (`last_confirmed_at`), which keeps it fresh for recency.

        Rule 2, supersession: the new fact closes the validity window of
          a) every fact the extractor explicitly said it `replaces`, and
          b) if the predicate is `exclusive` (one value at a time, like
             lives_in or job_title), every other active fact with the same
             subject and predicate.
        Path (a) catches contradictions a rule can't see ("I quit coffee" vs.
        "drinks coffee every morning"); path (b) is a deterministic safety
        net for the common case, so it doesn't depend on the LLM noticing.
        """
        now = self.now()
        subject_id = self.resolve_entity(profile_id, new.subject, new.subject_kind)
        object_id = (self.resolve_entity(profile_id, new.object, new.object_kind)
                     if new.object_is_entity else None)
        predicate = norm_predicate(new.predicate)

        # Rule 1: is this exact fact already known and true?
        dup = self.db.execute("""
            SELECT id FROM facts WHERE profile_id = ? AND status = 'active'
              AND subject_id = ? AND predicate = ? AND lower(object) = lower(?)
        """, (profile_id, subject_id, predicate, new.object.strip())).fetchone()
        if dup:
            with self.db:
                self.db.execute("UPDATE facts SET last_confirmed_at = ? WHERE id = ?",
                                (now, dup["id"]))
            return WriteResult("confirmed", self.get_fact(dup["id"]))

        # Rule 2: collect what the new fact makes untrue.
        targets: set[int] = set()
        for fid in new.replaces:
            row = self.db.execute(  # only active facts of THIS profile may be replaced
                "SELECT id FROM facts WHERE id = ? AND profile_id = ? AND status = 'active'",
                (fid, profile_id)).fetchone()
            if row:
                targets.add(row["id"])
        if new.exclusive:
            targets.update(r["id"] for r in self.db.execute("""
                SELECT id FROM facts WHERE profile_id = ? AND status = 'active'
                  AND subject_id = ? AND predicate = ?
            """, (profile_id, subject_id, predicate)))

        with self.db:  # one transaction: insert + supersede succeed or fail together
            fid = self.db.execute("""
                INSERT INTO facts (profile_id, subject_id, predicate, object, object_id,
                    statement, category, importance, exclusive, occurred_at,
                    valid_from, valid_to, recorded_at, last_confirmed_at,
                    superseded_by, status, origin, source_session_id, source_message_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, 'active', ?, ?, ?)
            """, (profile_id, subject_id, predicate, new.object.strip(), object_id,
                  new.statement.strip(), new.category, max(1, min(5, int(new.importance))),
                  int(new.exclusive), new.occurred_at, now, now, now,
                  origin, session_id, message_id)).lastrowid
            for old in targets:
                # The old fact stops being true at the moment the new one starts.
                self.db.execute("""
                    UPDATE facts SET status = 'superseded', valid_to = ?, superseded_by = ?
                    WHERE id = ?
                """, (now, fid, old))
        return WriteResult("added", self.get_fact(fid), [self.get_fact(t) for t in sorted(targets)])

    def history(self, fact_id: int) -> list[Fact]:
        """The version chain a fact belongs to, oldest first.

        Walk backwards (who did this fact supersede?) then forwards
        (who superseded it?). Gives "Berlin → Lisbon → Porto".
        """
        chain = [self.get_fact(fact_id)]
        while True:
            prev = self.db.execute("SELECT id FROM facts WHERE superseded_by = ? LIMIT 1",
                                   (chain[0].id,)).fetchone()
            if not prev:
                break
            chain.insert(0, self.get_fact(prev["id"]))
        while chain[-1].superseded_by:
            chain.append(self.get_fact(chain[-1].superseded_by))
        return chain

    # ── User control: edit and forget ────────────────────────────────────────
    def edit_fact(self, fact_id: int, statement: str, obj: str | None = None,
                  importance: int | None = None) -> WriteResult:
        """A user correction. It becomes a new version that supersedes the old
        one (origin = "user edit"), so the history shows what was changed."""
        old = self.get_fact(fact_id)
        profile_id = self.db.execute("SELECT profile_id FROM facts WHERE id = ?",
                                     (fact_id,)).fetchone()[0]
        new_obj = (obj if obj is not None else old.object).strip()
        if new_obj.lower() == old.object.lower() and old.status == "active":
            # Same value: rule 1 would just "confirm", so rewrite the wording
            # directly (fixing a typo shouldn't create a new version).
            with self.db:
                self.db.execute("UPDATE facts SET statement = ?, importance = ? WHERE id = ?",
                                (statement.strip(), importance or old.importance, fact_id))
            return WriteResult("edited", self.get_fact(fact_id))
        return self.add_fact(profile_id, NewFact(
            subject=old.subject if old.subject_id != self.user_entity(profile_id) else "user",
            predicate=old.predicate, object=new_obj, statement=statement,
            category=old.category, importance=importance or old.importance,
            exclusive=old.exclusive, occurred_at=old.occurred_at,
            object_is_entity=old.object_id is not None, replaces=[fact_id],
        ), origin="user edit")

    def delete_fact(self, fact_id: int) -> None:
        """Forget for real: remove the row. Links that pointed at it are
        cleared so the rest of the history stays consistent."""
        with self.db:
            self.db.execute("UPDATE facts SET superseded_by = NULL WHERE superseded_by = ?",
                            (fact_id,))
            self.db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))

    # ── Recall log: which memories shaped which answer ───────────────────────
    def record_recall(self, message_id: int, scored: list[tuple[int, float]]) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO recalls (message_id, fact_id, rank, score) VALUES (?, ?, ?, ?)",
                [(message_id, fid, i + 1, round(s, 4)) for i, (fid, s) in enumerate(scored)])

    def recalled_for(self, message_id: int) -> list[tuple[Fact, float]]:
        rows = self.db.execute(
            "SELECT fact_id, score FROM recalls WHERE message_id = ? ORDER BY rank",
            (message_id,)).fetchall()
        return [(self.get_fact(r["fact_id"]), r["score"]) for r in rows]

    def times_recalled(self, fact_id: int) -> int:
        return self.db.execute("SELECT COUNT(*) FROM recalls WHERE fact_id = ?",
                               (fact_id,)).fetchone()[0]
