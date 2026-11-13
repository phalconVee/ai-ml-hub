"""Two vision jobs, both done by Claude looking at a PNG.

1. `ocr_page`: a scanned page has no text layer, so Claude transcribes it.
   This replaces a Tesseract install: no system binary, and it reads
   equations, tables and two-column layouts better than classic OCR.
2. `read_figure`: Claude looks at one figure, together with its caption and
   the paragraph(s) that mention it, and returns a structured FIGURE_NOTE.

Each call is independent, so the pipeline runs them in parallel threads.
"""

from __future__ import annotations

import re
from typing import Any

from . import schemas
from .llm import ask_claude, check_stop, image_block, json_of, text_of
from .pdf_extract import Figure, Paper

OCR_SYSTEM = """You transcribe scanned pages of scientific papers.
Return the page's text in reading order (for two columns: left column, then right).
Keep numbers, units, symbols and equation content exactly as printed.
Render tables as Markdown tables. Where a figure appears, write one line:
[Figure: <caption if visible, else a one-sentence description>].
Output only the transcription, no commentary."""

FIGURE_SYSTEM = """You are a scientist reading one figure from a research paper.
Describe what is actually plotted or pictured before interpreting it.
Read numbers off the axes, labels and legends only when they are legible; never invent values.
The caption and the passages that cite the figure are given as context. If the image and
the text disagree, say so in `caveats`."""


# Crude sentence splitter: break after . ! ? when the next word starts with a
# capital letter. Requiring the capital keeps "Fig. 2" and "p. 3" whole; the
# lookbehinds keep "Fig. S2" and "et al. Smith" whole too.
SENTENCE_SPLIT = re.compile(r"(?<!Fig\.)(?<!Figs\.)(?<!al\.)(?<=[.!?])\s+(?=[A-Z])")


def ocr_page(client: Any, png: bytes, page_number: int) -> str:
    """Transcribe one scanned page. Returns plain text."""
    response = ask_claude(
        client,
        system=OCR_SYSTEM,
        messages=[{"role": "user", "content": [
            image_block(png),
            {"type": "text", "text": f"Transcribe page {page_number}."},
        ]}],
    )
    check_stop(response)
    return text_of(response).strip()


def read_figure(client: Any, fig: Figure, paper: Paper) -> dict:
    """Analyse one figure image in the context of its caption and citing text."""
    context = mentions_of(fig, paper)
    prompt = (
        f"Figure id: {fig.fig_id} (page {fig.page}, printed as: {fig.label or 'unlabelled'})\n"
        f"Caption: {fig.caption or '(no caption found)'}\n\n"
        f"Passages that refer to it:\n{context or '(none found)'}"
    )
    response = ask_claude(
        client,
        system=FIGURE_SYSTEM,
        messages=[{"role": "user", "content": [image_block(fig.image_png),
                                               {"type": "text", "text": prompt}]}],
        schema=schemas.FIGURE_NOTE,
    )
    return json_of(response)


def mentions_of(fig: Figure, paper: Paper, max_passages: int = 3) -> str:
    """Sentences anywhere in the paper that cite this figure ("Fig. 2", "Figure 2b").

    Giving Claude the authors' own words about a figure stops it from guessing
    what the figure is meant to prove.
    """
    if not fig.label:
        return ""
    number = fig.label.split()[-1].rstrip("abcdefgh")  # "Figure 2b" → "2"
    # Match "Fig. 2", "Figure 2", "Figs. 2", "Fig 2a" but not "Fig. 21".
    pattern = re.compile(rf"\bfig(?:ure)?s?\.?\s*{number}(?![0-9])", re.IGNORECASE)
    hits = []
    for page in paper.pages:
        for sentence in SENTENCE_SPLIT.split(" ".join(page.text.split())):
            if pattern.search(sentence) and sentence.strip() not in fig.caption:  # skip the caption itself
                hits.append(f"(p.{page.number}) {sentence.strip()}")
                if len(hits) >= max_passages:
                    return "\n".join(hits)
    return "\n".join(hits)
