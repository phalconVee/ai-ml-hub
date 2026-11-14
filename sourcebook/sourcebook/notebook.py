"""Step 6: the notebook - sources + index + memory, saved to disk.

A notebook is a folder:

    .sourcebook/notebooks/<slug>/
        notebook.json      sources (their extracted text), chat turns, memory summary
        overview.json      the latest audio-overview script
        overview.wav       ...and its audio, if a voice backend was available

The search index is *not* saved: rebuilding BM25 over a few thousand passages
takes milliseconds, and it means the saved file is just readable JSON.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .answer import Answer
from .chunk import Passage, chunk_source
from .config import SETTINGS
from .memory import Memory
from .retrieve import Embedder, Retriever
from .sources import Source


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "notebook"


class Notebook:
    def __init__(self, name: str, home: Optional[Path] = None, embedder: Optional[Embedder] = None):
        self.name = name
        self.home = Path(home or SETTINGS.home)
        self.sources: dict[str, Source] = {}
        self.memory = Memory()
        self.embedder = embedder        # None -> BM25 only
        self._retriever: Optional[Retriever] = None   # built lazily, reset on change

    # --- paths ----------------------------------------------------------------
    @property
    def folder(self) -> Path:
        return self.home / "notebooks" / slugify(self.name)

    @property
    def audio_path(self) -> Path:
        return self.folder / "overview.wav"

    @property
    def script_path(self) -> Path:
        return self.folder / "overview.json"

    # --- sources --------------------------------------------------------------
    def add_source(self, source: Source) -> bool:
        """Add a source; False if the same content is already in the notebook."""
        if source.id in self.sources:
            return False
        self.sources[source.id] = source
        self._retriever = None
        return True

    def remove_source(self, source_id: str) -> None:
        self.sources.pop(source_id, None)
        self._retriever = None

    def set_embedder(self, embedder: Optional[Embedder]) -> None:
        self.embedder = embedder
        self._retriever = None

    @property
    def passages(self) -> list[Passage]:
        return self.retriever.passages

    @property
    def retriever(self) -> Retriever:
        # All sources are indexed; the enabled/disabled filter is applied at
        # search time, so ticking a box doesn't trigger a re-index.
        if self._retriever is None:
            passages = [p for s in self.sources.values() for p in chunk_source(s)]
            self._retriever = Retriever(passages, self.embedder)
        return self._retriever

    # --- persistence ----------------------------------------------------------
    def save(self) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name,
            "sources": [s.to_dict() for s in self.sources.values()],
            "turns": [t.to_dict() for t in self.memory.turns],
            "summary": self.memory.summary,
            "summarized": self.memory.summarized,
        }
        path = self.folder / "notebook.json"
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, name: str, home: Optional[Path] = None,
             embedder: Optional[Embedder] = None) -> "Notebook":
        nb = cls(name, home, embedder)
        path = nb.folder / "notebook.json"
        if path.exists():
            data = json.loads(path.read_text())
            nb.name = data.get("name", name)
            nb.sources = {d["id"]: Source.from_dict(d) for d in data.get("sources", [])}
            nb.memory = Memory(
                turns=[Answer.from_dict(t) for t in data.get("turns", [])],
                summary=data.get("summary", ""),
                summarized=data.get("summarized", 0),
            )
        return nb


def list_notebooks(home: Optional[Path] = None) -> list[str]:
    """Names of saved notebooks, most recently changed first."""
    root = Path(home or SETTINGS.home) / "notebooks"
    files = sorted(root.glob("*/notebook.json"), key=lambda p: -p.stat().st_mtime) if root.exists() else []
    names = []
    for f in files:
        try:
            names.append(json.loads(f.read_text()).get("name", f.parent.name))
        except (OSError, json.JSONDecodeError):
            continue
    return names
