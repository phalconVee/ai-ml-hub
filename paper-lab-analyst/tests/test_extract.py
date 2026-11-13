"""PDF extraction on the synthetic 3-page paper from conftest.py."""

from paperlab.config import Settings
from paperlab.pdf_extract import extract_paper


def test_pages_text_and_scanned_detection(sample_pdf):
    paper = extract_paper(sample_pdf)
    assert paper.total_pages == 3
    assert [p.scanned for p in paper.pages] == [False, False, True]
    assert "phosphatase" in paper.pages[0].text
    assert paper.pages[2].image_png.startswith(b"\x89PNG")   # rendered for OCR
    assert paper.pages[0].image_png is None                   # normal pages aren't rendered


def test_finds_raster_and_vector_figures_with_captions(sample_pdf):
    paper = extract_paper(sample_pdf)
    assert [(f.fig_id, f.page, f.label, f.kind) for f in paper.figures] == [
        ("F1", 1, "Figure 1", "raster"),
        ("F2", 2, "Figure 2", "vector"),   # bars drawn as vector paths, merged into one figure
    ]
    assert paper.figures[1].caption.startswith("Fig. 2")
    assert all(f.image_png.startswith(b"\x89PNG") for f in paper.figures)


def test_title_from_first_line(sample_pdf):
    assert extract_paper(sample_pdf).title == "Soil Enzymes Under Drought: A Field Study"


def test_page_limit_warns_instead_of_silently_dropping(sample_pdf):
    paper = extract_paper(sample_pdf, Settings(max_pages=1))
    assert len(paper.pages) == 1 and paper.total_pages == 3
    assert "only the first 1" in paper.warnings[0]


def test_full_text_has_page_markers_and_marks_truncation(sample_pdf):
    paper = extract_paper(sample_pdf)
    assert "=== Page 2 ===" in paper.full_text()
    assert paper.full_text(max_chars=50).endswith("[... text truncated for length ...]")


# ── Regressions found by running real papers ─────────────────────────────────

import pymupdf
import pytest

from paperlab.pdf_extract import CAPTION_RE


@pytest.mark.parametrize("text,number", [
    ("Figure 2: Taobao search", "2"), ("Fig. 2 Phosphatase", "2"), ("FIGURE 12 | Map", "12"),
    ("Figure 9.1. A chapter figure", "9.1"), ("fig. 3b. Panels", "3b"),
    ("Figure 2 shows the engine", None), ("Figure 5 illustrates how", None),  # body text
])
def test_caption_regex(text, number):
    m = CAPTION_RE.match(text)
    assert (m.group(1) if m else None) == number


def _bars(page, top):
    """A 100pt-tall vector panel starting at y=top."""
    shape = page.new_shape()
    for i in range(4):
        shape.draw_rect(pymupdf.Rect(150 + i * 70, top + 20 * i, 190 + i * 70, top + 100))
    shape.finish(fill=(0.3, 0.3, 0.8))
    shape.commit()


def test_stacked_panels_and_figure_between_two_captions():
    doc = pymupdf.open()
    page = doc.new_page()
    # Enough body text (at the bottom) that the page is not treated as scanned.
    page.insert_textbox(pymupdf.Rect(72, 560, 523, 800), "Results text. " * 40, fontsize=9)
    # Figure 1: two panels stacked; the top one is ~140pt above the caption.
    _bars(page, 70)
    _bars(page, 190)
    page.insert_textbox(pymupdf.Rect(120, 300, 480, 320), "Figure 1: Two stacked panels.", fontsize=9)
    # Figure 2 sits 15pt below Figure 1's caption and 20pt above its own:
    # it must take the caption *below* it.
    _bars(page, 335)
    page.insert_textbox(pymupdf.Rect(120, 455, 480, 475), "Figure 2: Single panel.", fontsize=9)
    paper = extract_paper(doc.tobytes())
    assert [f.label for f in paper.figures] == ["Figure 1", "Figure 2"]
    fig1 = pymupdf.open(stream=paper.figures[0].image_png)[0].rect
    fig2 = pymupdf.open(stream=paper.figures[1].image_png)[0].rect
    assert fig1.height > 1.8 * fig2.height   # both panels in Figure 1, one in Figure 2
