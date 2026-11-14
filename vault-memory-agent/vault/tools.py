"""The agent's tools, and the code that runs them against the vault.

search_knowledge_base reads memory 1; recall, remember and forget read and
write memory 3. (Memory 2, the thread history, is replayed by the agent loop
itself, so it needs no tool.)

Claude never touches the database. It asks for a tool by name with JSON
arguments; `ToolBox.run` executes the request and returns a `tool_result`
block. Each call is also appended to `trace`, which is saved next to the
answer so the UI can show exactly what memory an answer used.

Tool definitions use `strict: true`, so the arguments are guaranteed to match
the schema (no missing `query`, no invented `kind`).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .memory import Vault

NOTE_KINDS = ["fact", "preference", "task", "other"]

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Semantic search over the documents the user uploaded. Returns the most "
            "relevant passages with their document name, location and a similarity "
            "score (1.0 = closest). Use it for any question that the uploaded "
            "documents might answer. Search again with different wording if the "
            "first results miss."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, phrased as a statement or question."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "recall",
        "description": (
            "Search your long-term notes: facts, preferences and tasks you saved in "
            "earlier conversations (any thread). Use it when the user refers to "
            "something from before, or when knowing their preferences would change "
            "your answer."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you are trying to remember."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "remember",
        "description": (
            "Save one durable note to long-term memory, readable from every future "
            "conversation. Use it for facts about the user, their preferences, "
            "decisions and open tasks: things worth knowing next week. Write the "
            "note so it makes sense on its own (\"User's team deploys on Fridays\", "
            "not \"they deploy then\"). Don't save document contents (the knowledge "
            "base already has them) or small talk."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The note, one self-contained sentence."},
                "kind": {"type": "string", "enum": NOTE_KINDS},
            },
            "required": ["content", "kind"],
            "additionalProperties": False,
        },
    },
    {
        "name": "forget",
        "description": (
            "Delete one note by its id (ids come from recall). Use it when a note "
            "is wrong or out of date, for example after the user changes a "
            "preference or a deadline moves: forget the old note, then remember "
            "the new fact."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "note_id": {"type": "integer", "description": "The id of the note to delete."},
            },
            "required": ["note_id"],
            "additionalProperties": False,
        },
    },
]


class ToolBox:
    """Runs tool calls for one turn of one thread, and records what happened."""

    def __init__(self, vault: Vault, thread_id: Optional[str]):
        self.vault = vault
        self.thread_id = thread_id
        self.trace: list[dict] = []   # one entry per tool call, saved with the answer

    def run(self, block) -> dict:
        """Execute one `tool_use` block; always return a `tool_result` block.

        Errors are returned to Claude as `is_error` results rather than raised:
        the model can read "no such tool" or a database error and adapt.
        """
        try:
            handler = getattr(self, f"_{block.name}", None)
            if handler is None:
                raise ValueError(f"Unknown tool {block.name!r}")
            output, record = handler(**block.input)
            self.trace.append({"tool": block.name, "input": dict(block.input), **record})
            return {"type": "tool_result", "tool_use_id": block.id, "content": output}
        except Exception as exc:   # noqa: BLE001 - every failure goes back to the model
            self.trace.append({"tool": block.name, "input": dict(block.input), "error": str(exc)})
            return {"type": "tool_result", "tool_use_id": block.id, "content": f"Error: {exc}",
                    "is_error": True}

    # Each handler returns (text for Claude, dict for the trace).

    def _search_knowledge_base(self, query: str):
        hits = self.vault.search(query)
        if not hits:
            return "The knowledge base is empty: no documents have been uploaded.", {"chunks": []}
        chunks = [{"doc": h.doc_name, "location": h.location, "score": h.score,
                   "text": h.text, "chunk_id": h.chunk_id} for h in hits]
        # Claude reads a compact JSON list; the same data goes into the trace.
        payload = [{k: c[k] for k in ("doc", "location", "score", "text")} for c in chunks]
        return json.dumps(payload, ensure_ascii=False), {"chunks": chunks}

    def _recall(self, query: str):
        notes = self.vault.recall(query)
        if not notes:
            return "No notes saved yet.", {"notes": []}
        found = [{"id": n.id, "kind": n.kind, "content": n.content, "score": n.score,
                  "saved": n.created_at[:10]} for n in notes]
        return json.dumps(found, ensure_ascii=False), {"notes": found}

    def _remember(self, content: str, kind: str):
        note, created = self.vault.remember(content.strip(), kind, self.thread_id)
        if created:
            return f"Saved note #{note.id}.", {"saved": {"id": note.id, "kind": kind, "content": note.content}}
        # Telling the model it's a duplicate stops it retrying with new wording.
        return (f"Already known as note #{note.id}: {note.content!r}. Nothing new saved.",
                {"duplicate_of": {"id": note.id, "content": note.content}})

    def _forget(self, note_id: int):
        if not self.vault.store.delete_note(note_id):
            # Raising turns this into an is_error result the model can read.
            raise ValueError(f"No note with id {note_id}. Use recall to find the right id.")
        return f"Deleted note #{note_id}.", {"forgot": note_id}
