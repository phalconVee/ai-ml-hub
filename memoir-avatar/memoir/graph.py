"""The long-term memory: a knowledge graph stored in one SQLite file.

Two representations of the same memory, each used for what it's good at:

  SQLite (on disk)        durable, transactional, survives restarts; the
                          source of truth. Merging new facts happens here.
  networkx (in memory)    built from SQLite on demand; graph algorithms
                          (neighbourhoods, degree, layout) happen here.

Vocabulary used throughout:

  entity    a node: a person, place, organisation, project, interest... One
            row per real-world thing, found again by its normalised `key`
            or by any of its aliases ("Henry" → the User node).
  relation  a directed, labelled edge: (User) -works_at-> (Acme).
            Saying the same fact again doesn't duplicate it; it raises the
            edge's `weight` and `last_turn` (how often / how recently).
  retired   a relation that *used* to be true. We never delete it: it stays
            with active=0 and the turn it ended, so the assistant can say
            "you used to live in Lagos" instead of forgetting or contradicting.
  turn      one user message + one reply. Every fact records the turn it was
            first and last mentioned in, which is our clock.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from .config import SETTINGS

# Words that always mean "the person I'm talking to". The extractor is told to
# use "User", but if it slips and writes "me" or "I", we still land on one node.
USER_WORDS = {"user", "me", "i", "myself", "the user"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY,
    key         TEXT UNIQUE NOT NULL,      -- normalised name, the identity
    name        TEXT NOT NULL,             -- display name, as first written
    type        TEXT NOT NULL DEFAULT 'other',
    description TEXT NOT NULL DEFAULT '',
    mentions    INTEGER NOT NULL DEFAULT 1,
    first_turn  INTEGER NOT NULL,
    last_turn   INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS aliases (
    alias_key   TEXT PRIMARY KEY,          -- normalised alternative name
    entity_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS relations (
    id          INTEGER PRIMARY KEY,
    src         INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate   TEXT NOT NULL,             -- snake_case verb phrase: works_at, likes
    dst         INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    weight      INTEGER NOT NULL DEFAULT 1,-- times this fact was stated
    first_turn  INTEGER NOT NULL,
    last_turn   INTEGER NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1,-- 0 = retired ("used to")
    ended_turn  INTEGER,                   -- when it was retired
    UNIQUE (src, predicate, dst)
);
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY,       -- 1, 2, 3...: the memory's clock
    at          TEXT NOT NULL,
    user_text   TEXT NOT NULL,
    reply_text  TEXT NOT NULL
);
"""


# ── Normalisation: how two spellings become one node ────────────────────────

def norm_key(name: str) -> str:
    """'  The  Eiffel Tower! ' → 'eiffel tower'. The identity of an entity."""
    s = re.sub(r"[^\w\s'-]", " ", name.lower())      # drop punctuation (keep o'neil, x-ray)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(the|a|an) ", "", s)               # "the Louvre" == "Louvre"
    return "user" if s in USER_WORDS else s


def norm_predicate(p: str) -> str:
    """'Works At' / 'works-at' / 'works at' → 'works_at'."""
    return re.sub(r"[^a-z0-9]+", "_", p.lower()).strip("_") or "related_to"


# ── Plain data that leaves this module ──────────────────────────────────────

@dataclass
class Fact:
    """One relation, with both endpoints' display names: what retrieval returns."""
    src: str
    predicate: str
    dst: str
    weight: int = 1
    first_turn: int = 0
    last_turn: int = 0
    active: bool = True
    ended_turn: int | None = None

    def sentence(self) -> str:
        """A line Claude (and a human) can read: 'User works at Acme'."""
        verb = self.predicate.replace("_", " ")
        line = f"{self.src} {verb} {self.dst}"
        if not self.active:
            line += f" (NO LONGER TRUE since turn {self.ended_turn})"
        return line + f" [turns {self.first_turn}-{self.last_turn}, said {self.weight}x]"


@dataclass
class Delta:
    """What one merge changed. Shown in the UI as 'memory updated: ...'."""
    new_entities: list[str] = field(default_factory=list)
    new_facts: list[str] = field(default_factory=list)
    reinforced: list[str] = field(default_factory=list)   # facts said again
    retired: list[str] = field(default_factory=list)      # facts no longer true

    def empty(self) -> bool:
        return not (self.new_entities or self.new_facts or self.reinforced or self.retired)


# ── The graph ───────────────────────────────────────────────────────────────

class KnowledgeGraph:
    def __init__(self, path: str | Path = SETTINGS.db_path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # Streamlit runs each rerun on a worker thread, so allow cross-thread
        # use and serialise writes with our own lock.
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")    # make ON DELETE CASCADE work
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()
        self._ensure_user()

    def _ensure_user(self) -> None:
        """The User node exists from the start: it's the hub of every personal fact."""
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO entities (key, name, type, description, mentions,"
                " first_turn, last_turn) VALUES ('user', ?, 'person',"
                " 'The person talking to the assistant.', 0, 0, 0)",
                (SETTINGS.user_node,))

    # ── Turns (the clock) ───────────────────────────────────────────────────

    def add_turn(self, user_text: str, reply_text: str) -> int:
        """Log one exchange and return its number. Facts are stamped with it."""
        with self.lock, self.db:
            cur = self.db.execute(
                "INSERT INTO turns (at, user_text, reply_text) VALUES (?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), user_text, reply_text))
            return cur.lastrowid

    def turn_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM turns").fetchone()[0]

    # ── Finding entities ────────────────────────────────────────────────────

    def find(self, name: str) -> sqlite3.Row | None:
        """Entity by name or alias, or None. 'me', 'Henry' (an alias) → User."""
        key = norm_key(name)
        row = self.db.execute("SELECT * FROM entities WHERE key = ?", (key,)).fetchone()
        if row is None:
            row = self.db.execute(
                "SELECT e.* FROM aliases a JOIN entities e ON e.id = a.entity_id"
                " WHERE a.alias_key = ?", (key,)).fetchone()
        return row

    def names(self, limit: int | None = None) -> list[str]:
        """Display names, most-mentioned first: the extractor's 'reuse these' list."""
        sql = "SELECT name FROM entities ORDER BY mentions DESC, last_turn DESC"
        rows = self.db.execute(sql + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
        return [r["name"] for r in rows]

    def aliases_of(self, entity_id: int) -> list[str]:
        rows = self.db.execute("SELECT alias_key FROM aliases WHERE entity_id = ?", (entity_id,))
        return [r[0] for r in rows]

    # ── Merging new knowledge (the heart of "memory") ───────────────────────

    def merge(self, extraction: dict, turn: int) -> Delta:
        """Fold one extraction ({entities: [...], relations: [...]}) into the graph.

        Upsert, never duplicate:
          * an entity seen before (by name or alias) gets +1 mention, fresher
            last_turn, and its description/type filled in if they were blank;
          * a relation seen before gets +1 weight (it's being confirmed);
          * an `exclusive` relation (lives_in, works_at...) retires the other
            active edges with the same source and predicate: you live in one
            city at a time, so the new one replaces the old;
          * a `negated` relation ("I quit Acme") retires that exact edge.
        All of it in one transaction: a crash mid-merge leaves the memory as it was.
        """
        delta = Delta()
        with self.lock, self.db:
            for ent in extraction.get("entities", []):
                self._upsert_entity(ent, turn, delta)
            for rel in extraction.get("relations", []):
                self._apply_relation(rel, turn, delta)
        return delta

    def _upsert_entity(self, ent: dict, turn: int, delta: Delta) -> int | None:
        name = (ent.get("name") or "").strip()
        if not norm_key(name):
            return None                                 # blank / punctuation-only: skip
        etype = (ent.get("type") or "other").strip().lower()
        desc = (ent.get("description") or "").strip()
        row = self.find(name)
        if row is None:
            cur = self.db.execute(
                "INSERT INTO entities (key, name, type, description, mentions, first_turn,"
                " last_turn) VALUES (?, ?, ?, ?, 1, ?, ?)",
                (norm_key(name), name, etype, desc, turn, turn))
            eid = cur.lastrowid
            delta.new_entities.append(name)
        else:
            eid = row["id"]
            # A newer description wins (people change); an empty one never erases.
            # A specific type beats 'other'.
            self.db.execute(
                "UPDATE entities SET mentions = mentions + 1, last_turn = ?,"
                " description = CASE WHEN ? != '' THEN ? ELSE description END,"
                " type = CASE WHEN type = 'other' THEN ? ELSE type END WHERE id = ?",
                (turn, desc, desc, etype, eid))
        for alias in ent.get("aliases") or []:
            akey = norm_key(alias)
            # An alias may not shadow a different entity's own name.
            clash = self.db.execute("SELECT id FROM entities WHERE key = ?", (akey,)).fetchone()
            if akey and (clash is None or clash["id"] == eid):
                self.db.execute("INSERT OR IGNORE INTO aliases VALUES (?, ?)", (akey, eid))
        return eid

    def _entity_id(self, name: str, turn: int, delta: Delta) -> int | None:
        """A relation may name an entity the extractor forgot to list: create it."""
        row = self.find(name)
        return row["id"] if row else self._upsert_entity({"name": name}, turn, delta)

    def _apply_relation(self, rel: dict, turn: int, delta: Delta) -> None:
        src = self._entity_id(rel.get("source", ""), turn, delta)
        dst = self._entity_id(rel.get("target", ""), turn, delta)
        if src is None or dst is None or src == dst:
            return                                      # self-loops carry no information
        pred = norm_predicate(rel.get("predicate", ""))
        label = f"{self._name(src)} {pred.replace('_', ' ')} {self._name(dst)}"
        existing = self.db.execute(
            "SELECT * FROM relations WHERE src = ? AND predicate = ? AND dst = ?",
            (src, pred, dst)).fetchone()

        if rel.get("negated"):
            # "I don't work at Acme any more": retire it if we had it; if we
            # never knew it, there's nothing to remember about it.
            if existing and existing["active"]:
                self._retire(existing["id"], turn)
                delta.retired.append(label)
            return

        if rel.get("exclusive"):
            # Retire every *other* active edge with this source + predicate.
            others = self.db.execute(
                "SELECT r.id, e.name FROM relations r JOIN entities e ON e.id = r.dst"
                " WHERE r.src = ? AND r.predicate = ? AND r.dst != ? AND r.active = 1",
                (src, pred, dst)).fetchall()
            for o in others:
                self._retire(o["id"], turn)
                delta.retired.append(f"{self._name(src)} {pred.replace('_', ' ')} {o['name']}")

        if existing is None:
            self.db.execute(
                "INSERT INTO relations (src, predicate, dst, first_turn, last_turn)"
                " VALUES (?, ?, ?, ?, ?)", (src, pred, dst, turn, turn))
            delta.new_facts.append(label)
        else:
            # Said again: stronger, fresher, and active again if it had been retired.
            self.db.execute(
                "UPDATE relations SET weight = weight + 1, last_turn = ?, active = 1,"
                " ended_turn = NULL WHERE id = ?", (turn, existing["id"]))
            delta.reinforced.append(label)

    def _retire(self, rel_id: int, turn: int) -> None:
        self.db.execute("UPDATE relations SET active = 0, ended_turn = ? WHERE id = ?",
                        (turn, rel_id))

    def _name(self, eid: int) -> str:
        return self.db.execute("SELECT name FROM entities WHERE id = ?", (eid,)).fetchone()[0]

    # ── Reading the graph ───────────────────────────────────────────────────

    def facts(self, include_retired: bool = True) -> list[Fact]:
        """Every relation as a Fact (endpoints by display name)."""
        sql = ("SELECT s.name AS src, r.predicate, d.name AS dst, r.weight, r.first_turn,"
               " r.last_turn, r.active, r.ended_turn FROM relations r"
               " JOIN entities s ON s.id = r.src JOIN entities d ON d.id = r.dst")
        if not include_retired:
            sql += " WHERE r.active = 1"
        return [Fact(**{**dict(r), "active": bool(r["active"])})
                for r in self.db.execute(sql + " ORDER BY r.last_turn DESC")]

    def to_networkx(self) -> nx.MultiDiGraph:
        """The whole memory as a networkx graph, nodes keyed by display name.

        MultiDiGraph because two things can be linked by more than one
        relation (User works_at Acme *and* User founded Acme).
        """
        g = nx.MultiDiGraph()
        for r in self.db.execute("SELECT * FROM entities"):
            g.add_node(r["name"], type=r["type"], description=r["description"],
                       mentions=r["mentions"], first_turn=r["first_turn"],
                       last_turn=r["last_turn"], aliases=self.aliases_of(r["id"]))
        for f in self.facts():
            g.add_edge(f.src, f.dst, key=f.predicate, predicate=f.predicate, fact=f,
                       active=f.active, weight=f.weight, last_turn=f.last_turn)
        return g

    def stats(self) -> dict:
        q = lambda sql: self.db.execute(sql).fetchone()[0]  # noqa: E731
        return {"entities": q("SELECT COUNT(*) FROM entities"),
                "facts": q("SELECT COUNT(*) FROM relations WHERE active = 1"),
                "retired": q("SELECT COUNT(*) FROM relations WHERE active = 0"),
                "turns": q("SELECT COUNT(*) FROM turns")}

    # ── The user stays in control of their memory ───────────────────────────

    def forget(self, name: str) -> bool:
        """Delete one entity and every fact touching it (CASCADE). Not the User."""
        row = self.find(name)
        if row is None or row["key"] == "user":
            return False
        with self.lock, self.db:
            self.db.execute("DELETE FROM entities WHERE id = ?", (row["id"],))
        return True

    def reset(self) -> None:
        """Wipe everything and start over with just the User node."""
        with self.lock, self.db:
            for table in ("relations", "aliases", "entities", "turns"):
                self.db.execute(f"DELETE FROM {table}")
        self._ensure_user()
