"""Turn uploaded files into small, citable chunks.

Pipeline:  raw bytes ──► pages of text ──► overlapping word windows (Chunks)

Why chunk at all? A search engine scores *pieces* of text against a query.
Score a whole 80-page PDF and every query matches it a little; score a
paragraph and only the right paragraphs match a lot.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from pypdf import PdfReader

from .config import SETTINGS


@dataclass(frozen=True)
class Chunk:
    id: str        # stable, human-readable handle the LLM cites, e.g. "handbook.pdf#p3-c1"
    source: str    # file name
    page: int      # 1-based page number (text files count as a single page 1)
    text: str


def read_pages(name: str, data: bytes) -> list[str]:
    """Return the text of each page. PDFs page by page; .txt/.md as one page."""
    if name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        # extract_text() can return None on scanned (image-only) pages → "".
        return [(page.extract_text() or "") for page in reader.pages]
    # Anything else is treated as UTF-8 text. errors="replace" means one bad
    # byte gives a "�" instead of crashing the whole upload.
    return [data.decode("utf-8", errors="replace")]


def chunk_text(
    text: str,
    size: int = SETTINGS.chunk_words,
    overlap: int = SETTINGS.overlap_words,
) -> list[str]:
    """Slide a window of `size` words across the text, stepping `size - overlap`.

    words:  w0 w1 w2 ... w179 | w140 ... w319 | w280 ...
            └──── chunk 0 ───┘
                        └──── chunk 1 ────┘      (40 words shared)
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    words = text.split()            # split() also collapses PDF whitespace junk
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):  # this window already reached the end
            break
    return chunks


def build_chunks(files: list[tuple[str, bytes]]) -> list[Chunk]:
    """files = [(file_name, raw_bytes), ...] → every chunk from every file."""
    out: list[Chunk] = []
    for name, data in files:
        for page_no, page_text in enumerate(read_pages(name, data), start=1):
            for i, piece in enumerate(chunk_text(page_text), start=1):
                out.append(Chunk(f"{name}#p{page_no}-c{i}", name, page_no, piece))
    return out
