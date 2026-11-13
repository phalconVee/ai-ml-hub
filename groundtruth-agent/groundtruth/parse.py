"""Brick 1: turn a messy PDF into clean, typed blocks.

A naive `pdf -> text` dump destroys exactly the parts that hold the answers in
real documents: tables become a soup of numbers, and figures vanish entirely.
So instead of one big string we produce a list of `Block`s, each tagged with
its page and its *kind*:

    text    a paragraph of running prose
    table   a table, rebuilt as Markdown so rows and columns survive
    figure  an image or chart, turned into words (caption + a vision description)

Everything downstream (chunking, search, citations) keeps that page + kind,
which is what lets the final answer say "page 4, table" instead of "somewhere".
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Callable, Optional

import pymupdf  # PyMuPDF: fast, pure-wheel PDF engine with table detection built in

from .config import SETTINGS

# A figure describer takes PNG bytes (+ nearby caption) and returns a description.
# Kept as a plain callable so tests can pass a fake and skip the API entirely.
FigureDescriber = Callable[[bytes, str], str]

# Captions usually start like "Figure 3:", "Fig. 2.", "Chart 1 -", "Exhibit 4".
CAPTION_RE = re.compile(r"^\s*(fig(ure)?\.?|chart|exhibit|diagram|graph)\s*\d+", re.IGNORECASE)


@dataclass
class Block:
    """One typed piece of a document."""

    doc: str     # file name, e.g. "annual_report.pdf"
    page: int    # 1-indexed page number (what humans and citations use)
    kind: str    # "text" | "table" | "figure"
    text: str    # the content as plain text / Markdown


# ---------------------------------------------------------------------------
# Small geometry helpers. PyMuPDF gives every element a bounding box (x0, y0, x1, y1).
# ---------------------------------------------------------------------------

def _overlap_ratio(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    """Fraction of `a`'s area that lies inside `b` (0.0 .. 1.0)."""
    inter = pymupdf.Rect(a) & b          # `&` is rectangle intersection
    if inter.is_empty or a.is_empty:
        return 0.0
    return inter.get_area() / a.get_area()


def _inside_any(rect: pymupdf.Rect, regions: list[pymupdf.Rect], threshold: float = 0.5) -> bool:
    """True if at least `threshold` of `rect` sits inside one of `regions`."""
    return any(_overlap_ratio(rect, r) >= threshold for r in regions)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _extract_tables(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, str]]:
    """Find tables on the page and return (bounding box, Markdown) pairs.

    `page.find_tables()` looks at ruling lines and text alignment to find the
    grid; `to_markdown()` turns it into `| a | b |` rows that an LLM reads well.
    """
    found = []
    for table in page.find_tables().tables:
        # Skip degenerate "tables" (a single cell is usually a framed paragraph).
        if table.row_count < 2 or table.col_count < 2:
            continue
        # The detector gives us the grid: rows, each a list of cell rectangles
        # (None where cells are merged). We read each cell's text by clipping the
        # page to that rectangle, then build the Markdown ourselves. (The built-in
        # `to_markdown()` may borrow a heading *above* the table as its header row.)
        rows = [
            [page.get_text("text", clip=cell) if cell else "" for cell in row.cells]
            for row in table.rows
        ]
        md = _rows_to_markdown(rows)
        if md:
            found.append((pymupdf.Rect(table.bbox), md))
    return found


def _rows_to_markdown(rows: list[list]) -> str:
    """[[a, b], [1, 2]] -> '| a | b |\n|---|---|\n| 1 | 2 |'. First row is the header."""
    clean = [[" ".join((cell or "").split()).replace("|", "/") for cell in row] for row in rows]
    clean = [row for row in clean if any(row)]          # drop fully empty rows
    if len(clean) < 2:
        return ""
    lines = ["| " + " | ".join(clean[0]) + " |", "|" + "---|" * len(clean[0])]
    lines += ["| " + " | ".join(row) + " |" for row in clean[1:]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _figure_regions(page: pymupdf.Page, table_boxes: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """Locate figures: embedded raster images *and* vector-drawn charts.

    Many charts in reports are not images at all, they are hundreds of drawn
    lines and rectangles. `cluster_drawings()` groups nearby drawings into one
    box per chart. Table borders are drawings too, so we drop clusters that
    sit on a table we already extracted.
    """
    boxes: list[pymupdf.Rect] = []

    # 1. Raster images (photos, screenshots, exported charts).
    for info in page.get_image_info():
        boxes.append(pymupdf.Rect(info["bbox"]))

    # 2. Vector graphics (charts drawn with PDF path operators).
    if hasattr(page, "cluster_drawings"):  # present in PyMuPDF >= 1.24
        try:
            for rect in page.cluster_drawings():
                if not _inside_any(rect, table_boxes, 0.3):
                    boxes.append(pymupdf.Rect(rect))
        except Exception:
            pass  # unusual drawing data should never kill parsing

    # Keep only boxes big enough to matter, then de-duplicate overlapping ones
    # (an image inside a drawn frame would otherwise be counted twice).
    boxes = [b & page.rect for b in boxes]   # clip to the visible page
    boxes = [b for b in boxes if b.get_area() >= SETTINGS.min_figure_area]
    unique: list[pymupdf.Rect] = []
    for box in sorted(boxes, key=lambda r: -r.get_area()):   # biggest first
        if not _inside_any(box, unique, 0.6):
            unique.append(box)
    return unique


def _find_caption(fig: pymupdf.Rect, text_blocks: list[tuple[pymupdf.Rect, str]]) -> str:
    """Pick the caption: the closest 'Figure N ...' text block just above or below."""
    best, best_gap = "", 80.0  # look at most ~80pt (about 1 inch) away
    for rect, text in text_blocks:
        if not CAPTION_RE.match(text):
            continue
        gap = min(abs(rect.y0 - fig.y1), abs(fig.y0 - rect.y1))  # below or above
        horizontally_aligned = rect.x1 > fig.x0 and rect.x0 < fig.x1
        if horizontally_aligned and gap < best_gap:
            best, best_gap = text, gap
    return best


def _render_png(page: pymupdf.Page, clip: pymupdf.Rect) -> bytes:
    """Rasterise just the figure's rectangle to PNG bytes."""
    return page.get_pixmap(clip=clip, dpi=SETTINGS.figure_dpi).tobytes("png")


# ---------------------------------------------------------------------------
# The vision describer (the only part of parsing that calls an LLM)
# ---------------------------------------------------------------------------

FIGURE_PROMPT = """You are helping index a document for question answering.
Describe this figure so that someone who cannot see it could answer questions about it.
- Say what kind of figure it is (bar chart, line chart, diagram, photo, ...).
- Transcribe titles, axis labels, legend entries and every readable number or label.
- State the main trend or takeaway in one sentence.
Plain prose, no preamble. If the image is decorative (logo, divider), reply exactly: DECORATIVE
Caption found near the figure (may be empty): {caption}"""


def make_claude_describer(client=None, model: Optional[str] = None) -> FigureDescriber:
    """Build a FigureDescriber backed by Claude's vision.

    `client` defaults to `anthropic.Anthropic()`, which reads ANTHROPIC_API_KEY.
    """
    import anthropic  # imported lazily so parsing without vision needs no key

    client = client or anthropic.Anthropic()
    model = model or SETTINGS.figure_model

    def describe(png: bytes, caption: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=2_000,
            output_config={"effort": "low"},  # describing a chart is easy work
            messages=[{
                "role": "user",
                "content": [
                    # Image first, then the instruction: Claude reads better that way.
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.standard_b64encode(png).decode(),
                    }},
                    {"type": "text", "text": FIGURE_PROMPT.format(caption=caption or "(none)")},
                ],
            }],
        )
        if response.stop_reason == "refusal":
            return ""
        # Responses can hold thinking blocks too; keep only the visible text.
        return "".join(b.text for b in response.content if b.type == "text").strip()

    return describe


# ---------------------------------------------------------------------------
# The main entry point
# ---------------------------------------------------------------------------

def parse_pdf(
    data: bytes,
    name: str,
    describe_figure: Optional[FigureDescriber] = None,
) -> list[Block]:
    """Parse PDF bytes into typed Blocks, page by page, in reading order.

    `describe_figure=None` keeps figures as caption-only blocks: no API call.
    """
    blocks: list[Block] = []
    figures_described = 0

    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for page_index, page in enumerate(pdf):
            page_no = page_index + 1

            # --- Tables first, so we know which text to skip later ------
            tables = _extract_tables(page)
            table_boxes = [box for box, _ in tables]

            # --- Raw text blocks -------------------------------------------
            # get_text("blocks") -> (x0, y0, x1, y1, text, block_no, block_type);
            # block_type 0 is text, 1 is an image placeholder.
            text_blocks = [
                (pymupdf.Rect(b[:4]), " ".join(b[4].split()))  # collapse whitespace/newlines
                for b in page.get_text("blocks", sort=True)
                if b[6] == 0 and b[4].strip()
            ]

            # --- Figures -----------------------------------------------------
            figure_boxes = _figure_regions(page, table_boxes)
            # Axis ticks and legends often sit just *outside* the drawn chart box,
            # so for dropping stray labels we grow each figure by a small margin.
            figure_halos = [pymupdf.Rect(b) + (-15, -15, 15, 15) for b in figure_boxes]

            # Collect every element with its top y-coordinate, then sort, so the
            # output follows the page from top to bottom like a human reads it.
            items: list[tuple[float, Block]] = []

            for box, md in tables:
                items.append((box.y0, Block(name, page_no, "table", md)))

            captions_used: set[str] = set()
            for box in figure_boxes:
                caption = _find_caption(box, text_blocks)
                description = ""
                if describe_figure and figures_described < SETTINGS.max_figures_per_doc:
                    description = describe_figure(_render_png(page, box), caption)
                    figures_described += 1
                    if description.strip().upper() == "DECORATIVE":
                        continue  # logos and dividers add nothing to search
                if not (caption or description):
                    continue  # nothing we can say about this figure in words
                captions_used.add(caption)
                body = "\n".join(p for p in (caption, description) if p)
                items.append((box.y0, Block(name, page_no, "figure", body)))

            for box, text in text_blocks:
                # Drop text that lives inside a table (already captured as
                # Markdown) or inside a figure (axis ticks, legend labels),
                # and captions we already attached to a figure.
                if _inside_any(box, table_boxes) or _inside_any(box, figure_halos, 0.8):
                    continue
                if text in captions_used:
                    continue
                items.append((box.y0, Block(name, page_no, "text", text)))

            items.sort(key=lambda pair: pair[0])
            blocks.extend(block for _, block in items)

    return blocks
