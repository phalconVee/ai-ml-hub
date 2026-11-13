"""Source 4: conversation memory, a local JSON file of past questions and answers.

Every finished turn is saved. On a new question we BM25-search old turns and
offer the relevant ones as candidates, so the assistant can build on what it
already worked out ("as we found earlier…") without replaying the whole chat.

That is the key context-engineering idea for memory: don't send *all* history,
retrieve the *relevant* history and let it compete for space like any other
source. Stored on disk, so it survives a restart. No external service.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import SETTINGS
from ..items import ContextItem
from ..textsearch import BM25


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.turns: list[dict] = []
        if self.path.exists():
            self.turns = json.loads(self.path.read_text())

    def remember(self, question: str, answer: str) -> None:
        self.turns.append({"q": question, "a": answer, "t": time.time()})
        self.turns = self.turns[-SETTINGS.memory_turns_kept:]   # forget the oldest
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.turns, indent=1))

    def clear(self) -> None:
        self.turns = []
        if self.path.exists():
            self.path.unlink()

    def search(self, query: str, k: int | None = None) -> list[ContextItem]:
        if not self.turns:
            return []
        texts = [f"Q: {t['q']}\nA: {t['a']}" for t in self.turns]
        hits = BM25(texts).top(query, k or SETTINGS.per_source_candidates)
        return [
            ContextItem(source="memory", title=f"Earlier: {self.turns[i]['q'][:70]}",
                        text=texts[i], meta={"saved": time.strftime(
                            "%Y-%m-%d %H:%M", time.localtime(self.turns[i]["t"]))})
            for i, _ in hits
        ]
