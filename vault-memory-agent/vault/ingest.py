"""Uploaded file -> text -> overlapping chunks.

A chunk is the unit the knowledge base stores, embeds and returns. Each one
remembers where it came from (`location`, e.g. "p. 4"), so the agent can say
which page an answer rests on.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from .config import SETTINGS


@dataclass
class Chunk:
    text: str
    location: str   # human-readable position: "p. 3" for PDFs, "§2" for text


def load_pages(filename: str, data: bytes) -> list[tuple[str, str]]:
    """Return [(location, text)] for one uploaded file.

    PDFs are read page by page so a chunk never straddles two pages (its
    location stays exact). Text and Markdown are one "page".
    """
    lower = filename.lower()
    if lower.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = [(f"p. {i}", page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
        if not any(text.strip() for _, text in pages):
            # A scanned PDF has images, not text; we'd store nothing useful.
            raise ValueError(f"{filename}: no text layer found (scanned PDF?). Run OCR first.")
        return pages
    if lower.endswith((".txt", ".md", ".markdown", ".rst", ".csv")):
        return [("", data.decode("utf-8", errors="replace"))]
    raise ValueError(f"{filename}: unsupported type. Use PDF, TXT or Markdown.")


def chunk_text(text: str, location: str = "", words: int = SETTINGS.chunk_words,
               overlap: int = SETTINGS.chunk_overlap) -> list[Chunk]:
    """Slide a window of `words` words over the text, stepping `words - overlap`.

    The overlap means a sentence cut by one boundary still appears whole in
    the neighbouring chunk, so a search can always find it intact.
    """
    tokens = re.sub(r"\s+", " ", text).strip().split(" ")
    tokens = [t for t in tokens if t]
    if not tokens:
        return []
    step = max(1, words - overlap)
    chunks = []
    for n, start in enumerate(range(0, len(tokens), step), 1):
        window = tokens[start:start + words]
        # Text files get "§n" so every chunk still has a findable label.
        label = location or f"§{n}"
        chunks.append(Chunk(" ".join(window), label))
        if start + words >= len(tokens):   # this window reached the end
            break
    return chunks


def chunk_file(filename: str, data: bytes) -> list[Chunk]:
    chunks: list[Chunk] = []
    for location, text in load_pages(filename, data):
        chunks.extend(chunk_text(text, location))
    if not chunks:
        raise ValueError(f"{filename}: the file is empty.")
    return chunks
