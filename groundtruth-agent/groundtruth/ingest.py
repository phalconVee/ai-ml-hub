"""Glue: PDF bytes -> parsed blocks -> chunks -> added to the index.

Kept separate so the Streamlit app and a plain Python script share one path.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from .chunk import chunk_blocks
from .index import BM25Index
from .parse import FigureDescriber, parse_pdf


def ingest_pdf(
    index: BM25Index,
    data: bytes,
    name: str,
    describe_figure: Optional[FigureDescriber] = None,
) -> dict:
    """Parse one PDF into `index`. Returns counts for the UI, e.g. {"text": 40, "table": 3, "figure": 2}."""
    blocks = parse_pdf(data, name, describe_figure)
    chunks = chunk_blocks(blocks, start_id=len(index))   # ids continue across documents
    index.add(chunks)
    stats = Counter(c.kind for c in chunks)
    stats["pages"] = max((b.page for b in blocks), default=0)
    return dict(stats)
