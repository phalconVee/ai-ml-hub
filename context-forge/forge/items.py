"""The one data type every stage passes around: a candidate piece of context.

Each source (documents, web, papers, memory) produces `ContextItem`s. The
ranker scores them and writes down *why* each was kept or dropped, and the
UI shows exactly that. Keeping a single shape is what lets four very
different sources compete for the same window on equal terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: ~4 characters per token for English text.

    Exact counts need an API call (`messages.count_tokens`); for budgeting a
    few thousand tokens a rule of thumb is close enough and costs nothing.
    """
    return max(1, len(text) // 4)


@dataclass
class ContextItem:
    source: str            # "doc" | "web" | "paper" | "memory"
    title: str             # human label: file name, page title, paper title...
    text: str              # the passage itself: what would enter the window
    url: str = ""          # link for web/paper items, empty otherwise
    meta: dict = field(default_factory=dict)   # anything source-specific (authors, date…)

    # Filled in by the ranker ──────────────────────────────────────────────
    id: str = ""           # citation label, e.g. "S3"; given only to kept items
    relevance: float = 0.0  # 0..1 lexical match to the question (BM25, normalised)
    judge: float | None = None  # 0..1 LLM-judge relevance, if the judge ran
    score: float = 0.0     # final score after weights
    kept: bool = False
    reason: str = ""       # one line: why kept, or why dropped

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)
