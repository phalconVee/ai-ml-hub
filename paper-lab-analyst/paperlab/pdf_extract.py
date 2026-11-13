"""PDF → pages of text + figure images. No LLM in this file.

What we pull out of the PDF:

* **Text, page by page.** PyMuPDF reads the text layer in reading order.
* **Scanned pages.** If a page has almost no selectable text, it's probably a
  photo/scan of paper. We render it to a PNG and mark it `scanned`; later,
  `vision.py` asks Claude to transcribe it (that's our OCR).
* **Figures.** Papers draw figures in two ways, and we catch both:
    - raster images (photos, microscopy, exported PNG plots) → `get_image_info`
    - vector drawings (most matplotlib/R charts) → `cluster_drawings`
  Each candidate region is paired with the nearest "Figure N" / "Fig. N"
  caption, panels sharing one caption are merged, and the region is rendered
  to a PNG so Claude can look at exactly what a reader sees.

Everything is plain data (dataclasses) so the rest of the app, and the tests,
never touch PyMuPDF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

from .config import SETTINGS, Settings

# "Figure 3:", "Fig. 3b.", "FIGURE 12 |", "Fig. 2 Phosphatase ..." at the start
# of a text block. Group 1 is the number ("3", "9.1" in books, plus an optional
# panel letter) so we can name the figure. The lookahead after the number
# demands caption punctuation or a capitalised word, so a body paragraph that
# starts "Figure 2 shows the engine..." is not mistaken for a caption.
CAPTION_RE = re.compile(
    # (?i:...) makes only the word "fig/figure" case-insensitive; the capital
    # letter test after the number must stay case-sensitive.
    r"^\s*(?i:fig(?:ure)?\.?)\s*(\d+(?:\.\d+)?[a-z]?)\b(?=\s*[:.|\-–—]|\s+[A-Z(\[])"
)


@dataclass
class Page:
    number: int                 # 1-based, like the page numbers a human uses
    text: str                   # selectable text, or the OCR transcript once vision.py runs
    scanned: bool = False       # True → text layer was (nearly) empty
    image_png: bytes | None = None  # rendered page, kept only for scanned pages
    ocr_done: bool = False      # set by the pipeline after Claude transcribes it


@dataclass
class Figure:
    fig_id: str                 # our own stable id: "F1", "F2", ... (citable in the report)
    page: int
    image_png: bytes
    caption: str = ""           # full caption text, "" if none was found nearby
    label: str = ""             # "Figure 2" as printed in the paper, if the caption says so
    kind: str = "raster"        # "raster" | "vector" | "mixed"


@dataclass
class Paper:
    title: str
    pages: list[Page]
    figures: list[Figure]
    total_pages: int            # pages in the PDF (may exceed len(pages) if truncated)
    warnings: list[str] = field(default_factory=list)

    def full_text(self, max_chars: int | None = None) -> str:
        """All page text with `=== Page N ===` markers, so the LLM can cite pages."""
        parts = [f"=== Page {p.number} ===\n{p.text.strip()}" for p in self.pages]
        text = "\n\n".join(parts)
        if max_chars is not None and len(text) > max_chars:
            # Never cut silently: the marker tells the model (and the reader) why it stops.
            text = text[:max_chars] + "\n\n[... text truncated for length ...]"
        return text

    @property
    def scanned_pages(self) -> list[Page]:
        return [p for p in self.pages if p.scanned]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_paper(pdf_bytes: bytes, settings: Settings = SETTINGS) -> Paper:
    """Open a PDF from bytes (what Streamlit's uploader gives us) and extract it."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    warnings: list[str] = []
    pages: list[Page] = []
    raw_figures: list[tuple[int, float, Figure]] = []  # (page, y, fig) for sorting

    if doc.page_count > settings.max_pages:
        warnings.append(f"PDF has {doc.page_count} pages; only the first "
                        f"{settings.max_pages} are analysed.")

    for index in range(min(doc.page_count, settings.max_pages)):
        page = doc[index]
        # sort=True → top-to-bottom, left-to-right reading order, which keeps
        # two-column layouts from being interleaved line by line.
        text = page.get_text("text", sort=True)

        if len(text.strip()) < settings.scanned_text_chars:
            # Too little text to be a normal page: render it for OCR. We don't
            # hunt for figures here; the OCR prompt describes them instead.
            png = page.get_pixmap(dpi=settings.render_dpi).tobytes("png")
            pages.append(Page(number=index + 1, text=text, scanned=True, image_png=png))
            continue

        pages.append(Page(number=index + 1, text=text))
        for y, fig in _find_figures(page, index + 1, settings):
            raw_figures.append((index + 1, y, fig))

    # Number figures in reading order (page, then top-to-bottom): F1, F2, ...
    raw_figures.sort(key=lambda t: (t[0], t[1]))
    figures = []
    for n, (_, _, fig) in enumerate(raw_figures, start=1):
        fig.fig_id = f"F{n}"
        figures.append(fig)

    title = _guess_title(doc, pages)
    total = doc.page_count
    doc.close()
    return Paper(title=title, pages=pages, figures=figures, total_pages=total, warnings=warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _find_figures(page: pymupdf.Page, page_no: int, s: Settings) -> list[tuple[float, Figure]]:
    """Find figure regions on one page, attach captions, render them."""
    page_area = page.rect.width * page.rect.height

    def big_enough(r: pymupdf.Rect) -> bool:
        return r.width >= s.min_figure_pts and r.height >= s.min_figure_pts

    def not_noise(r: pymupdf.Rect) -> bool:
        # Size is judged on the *finished* figure, not on pieces: a figure is
        # often several small images (e.g. five phone screenshots in a row).
        # Here we only drop specks and hairlines (rules, bullets, underlines).
        return r.width >= s.min_piece_pts and r.height >= s.min_piece_pts

    # 1. Candidate pieces, tagged by how they were drawn.
    regions: list[tuple[pymupdf.Rect, str]] = []
    for info in page.get_image_info():                 # raster images placed on the page
        r = pymupdf.Rect(info["bbox"]) & page.rect     # clip to page (images can overhang)
        if not_noise(r):
            regions.append((r, "raster"))
    for r in page.cluster_drawings():                  # groups of nearby vector paths
        r = pymupdf.Rect(r) & page.rect
        if not_noise(r) and r.get_area() < 0.9 * page_area:  # skip page frames/backgrounds
            regions.append((r, "vector"))

    # 2. Overlapping pieces are one figure (e.g. a plot image + its vector axes).
    regions = _merge_overlapping(regions)

    # 3. Caption blocks. "blocks" gives (x0, y0, x1, y1, text, block_no, type).
    captions = [(pymupdf.Rect(b[:4]), " ".join(b[4].split()))
                for b in page.get_text("blocks") if b[6] == 0 and CAPTION_RE.match(b[4])]

    # 4. Pair each piece with its caption. Pieces sharing a caption are panels
    #    of one figure → union them into one rectangle.
    by_caption: dict[int, tuple[pymupdf.Rect, set[str]]] = {}
    uncaptioned: list[tuple[pymupdf.Rect, str]] = []
    for rect, kind in regions:
        idx = _nearest_caption(rect, captions, s.caption_gap_pts)
        if idx is None:
            uncaptioned.append((rect, kind))
        elif idx in by_caption:
            by_caption[idx][0].include_rect(rect)
            by_caption[idx][1].add(kind)
        else:
            by_caption[idx] = (pymupdf.Rect(rect), {kind})

    # 5. Grow figures: a top panel can sit far above the caption, and a side
    #    panel can stick out past the caption's width, but each one is close
    #    to a panel that *is* paired. Keep absorbing orphan pieces lying near
    #    an already-captioned figure (in any direction) until none are left.
    grew = True
    while grew:
        grew = False
        for piece in list(uncaptioned):
            rect, kind = piece
            for fig_rect, kinds in by_caption.values():
                x_gap = max(fig_rect.x0 - rect.x1, rect.x0 - fig_rect.x1, 0)
                y_gap = max(fig_rect.y0 - rect.y1, rect.y0 - fig_rect.y1, 0)
                if max(x_gap, y_gap) <= s.caption_gap_pts / 2:
                    fig_rect.include_rect(rect)
                    kinds.add(kind)
                    uncaptioned.remove(piece)
                    grew = True
                    break

    out: list[tuple[float, Figure]] = []
    for idx, (rect, kinds) in by_caption.items():
        if not big_enough(rect):
            continue                                   # e.g. a tiny icon that sits near a caption
        cap_text = captions[idx][1]
        number = CAPTION_RE.match(cap_text).group(1)
        out.append((rect.y0, _render(page, page_no, rect, kinds, cap_text, f"Figure {number}", s)))
    for rect, kind in _merge_overlapping(uncaptioned):
        # No caption → maybe a logo, an equation, a table rule. Keep it only if
        # it's large (≥15% of the page): that's almost always a real figure.
        if big_enough(rect) and rect.get_area() >= 0.15 * page_area:
            out.append((rect.y0, _render(page, page_no, rect, {kind}, "", "", s)))
    return out


def _render(page, page_no, rect, kinds, caption, label, s: Settings) -> Figure:
    clip = rect + (-4, -4, 4, 4)  # small margin so axis labels at the edge aren't cut
    png = page.get_pixmap(clip=clip & page.rect, dpi=s.render_dpi).tobytes("png")
    kind = kinds.pop() if len(kinds) == 1 else "mixed"
    return Figure(fig_id="", page=page_no, image_png=png, caption=caption, label=label, kind=kind)


def _merge_overlapping(regions: list[tuple[pymupdf.Rect, str]]) -> list[tuple[pymupdf.Rect, str]]:
    """Union any regions that intersect, repeating until nothing changes."""
    items = [(pymupdf.Rect(r), k) for r, k in regions]
    changed = True
    while changed:
        changed = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (a, ka), (b, kb) = items[i], items[j]
                if a.intersects(b):
                    a.include_rect(b)
                    items[i] = (a, ka if ka == kb else "mixed")
                    del items[j]
                    changed = True
                    break
            if changed:
                break
    return items


def _nearest_caption(rect: pymupdf.Rect, captions, max_gap: float) -> int | None:
    """Index of the caption for `rect`, or None.

    Figure captions sit *below* the figure, so a caption below always wins.
    Only if there is none do we accept one above. Without this rule, a figure
    stacked between two captions grabs the previous figure's caption.
    """
    below, above = [], []
    for i, (cap, _) in enumerate(captions):
        if cap.x1 < rect.x0 or cap.x0 > rect.x1:      # no horizontal overlap → other column
            continue
        if cap.y0 >= rect.y1 - 2:                       # caption below the figure (the norm)
            below.append((cap.y0 - rect.y1, i))
        elif cap.y1 <= rect.y0 + 2:                     # caption above (rare for figures)
            above.append((rect.y0 - cap.y1, i))
    for candidates in (below, above):
        close = [(gap, i) for gap, i in candidates if gap <= max_gap]
        if close:
            return min(close)[1]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────

def _guess_title(doc: pymupdf.Document, pages: list[Page]) -> str:
    """PDF metadata title if it looks real, else the first line of text."""
    meta = (doc.metadata or {}).get("title", "").strip()
    # Word/LaTeX exports often leave junk titles like "Microsoft Word - draft3.docx".
    if meta and not re.search(r"\.(docx?|tex|pdf)$|^untitled$|microsoft word", meta, re.I):
        return meta
    for p in pages:
        for line in p.text.splitlines():
            if len(line.strip()) > 8:
                return line.strip()[:200]
    return "Untitled paper"
