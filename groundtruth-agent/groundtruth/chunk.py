"""Brick 2: cut blocks into passages small enough to search and cite.

Rules:
- Prose is merged across neighbouring paragraphs on the same page, then cut
  into windows of ~`chunk_words` words with `chunk_overlap` words of overlap.
- Tables and figures are never split: half a table is worse than useless.
- A chunk never crosses a page, so every citation points at exactly one page.

Each chunk also keeps its text as a list of *pieces* (paragraphs, or table
rows). Claude's citations point at piece ranges, so finer pieces mean
tighter, more precise quotes in the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

from .config import SETTINGS
from .parse import Block


@dataclass
class Chunk:
    id: int            # position in the index, handy for debugging
    doc: str
    page: int
    kind: str          # "text" | "table" | "figure"
    pieces: list[str]  # citation-sized sub-parts, in order

    @property
    def text(self) -> str:
        """The whole chunk as one string (what the search engine scores)."""
        return "\n".join(self.pieces)

    @property
    def source(self) -> str:
        """A stable identifier Claude echoes back in citations."""
        return f"{self.doc}#page={self.page}"

    @property
    def title(self) -> str:
        """Human-readable label shown next to a citation."""
        return f"{self.doc}, page {self.page} ({self.kind})"


def _split_table(md: str) -> list[str]:
    """Keep the header with every piece, so a cited row still says what its columns mean.

    A Markdown table is: header line, separator line (|---|---|), then rows.
    Piece 0 is the header + separator; each row becomes its own piece.
    """
    lines = [l for l in md.splitlines() if l.strip()]
    if len(lines) <= 2:
        return [md]
    return ["\n".join(lines[:2])] + lines[2:]


def _window_prose(paragraphs: list[str], size: int, overlap: int) -> list[list[str]]:
    """Group paragraphs into windows of about `size` words with some overlap.

    Very long paragraphs are first cut into sentence-ish pieces of <= size words.
    """
    # 1. Make sure no single piece is bigger than a whole window.
    pieces: list[str] = []
    for para in paragraphs:
        words = para.split()
        for start in range(0, len(words), size):
            pieces.append(" ".join(words[start:start + size]))

    # 2. Greedily pack pieces into windows.
    windows: list[list[str]] = []
    current: list[str] = []
    count = 0
    for piece in pieces:
        n = len(piece.split())
        if current and count + n > size:
            windows.append(current)
            # Overlap: carry trailing pieces forward until we have ~overlap words.
            carry, carried = [], 0
            for prev in reversed(current):
                if carried >= overlap:
                    break
                carry.insert(0, prev)
                carried += len(prev.split())
            # Never carry the whole previous window (that would loop forever).
            current = carry if carried < count else []
            count = sum(len(p.split()) for p in current)
        current.append(piece)
        count += n
    if current:
        windows.append(current)
    return windows


def chunk_blocks(blocks: list[Block], start_id: int = 0) -> list[Chunk]:
    """Turn parsed Blocks into searchable Chunks."""
    chunks: list[Chunk] = []

    def add(doc: str, page: int, kind: str, pieces: list[str]) -> None:
        chunks.append(Chunk(start_id + len(chunks), doc, page, kind, pieces))

    # Walk the document page by page.
    for (doc, page), page_blocks in groupby(blocks, key=lambda b: (b.doc, b.page)):
        prose: list[str] = []

        def flush() -> None:
            # Emit buffered prose as windowed chunks.
            for window in _window_prose(prose, SETTINGS.chunk_words, SETTINGS.chunk_overlap):
                add(doc, page, "text", window)
            prose.clear()

        for block in page_blocks:
            if block.kind == "text":
                prose.append(block.text)
            else:
                # A table or figure: flush the prose before it, keep it whole.
                flush()
                pieces = _split_table(block.text) if block.kind == "table" else block.text.split("\n")
                add(doc, page, block.kind, [p for p in pieces if p.strip()])
        flush()

    return chunks
