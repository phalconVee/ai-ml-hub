"""Generate a small but realistic test PDF: prose, a ruled table, a vector chart and a raster image.

Run:  uv run python sample_docs/make_sample.py
Writes sample_docs/northwind_report.pdf (a made-up company, numbers are invented).
The tests import `build_sample_pdf()` to get the same PDF as bytes.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

BLACK, BLUE, GREY = (0, 0, 0), (0.2, 0.4, 0.8), (0.6, 0.6, 0.6)

INTRO = (
    "Northwind Bicycles is a fictional maker of electric cargo bikes. This report "
    "summarises fiscal year 2024. Revenue grew because the Hauler model entered three "
    "new European markets, while gross margin fell after battery costs rose sharply in "
    "the second half of the year."
)
OUTLOOK = (
    "For 2025 management expects revenue between 61 and 64 million euros. The main risk "
    "is the supply of battery cells; a second supplier in Poland was signed in November."
)
TABLE = [
    ["Metric", "2023", "2024"],
    ["Revenue (EUR m)", "41.2", "55.8"],
    ["Gross margin (%)", "38.5", "34.1"],
    ["Bikes shipped", "9,850", "12,400"],
]
CHART = [("Q1", 11.0), ("Q2", 13.5), ("Q3", 14.1), ("Q4", 17.2)]  # revenue per quarter


def _table(page: pymupdf.Page, x0: float, y0: float) -> None:
    """Draw a fully ruled table so PyMuPDF's line-based detector can find it."""
    col_w, row_h = [150, 80, 80], 22
    width, height = sum(col_w), row_h * len(TABLE)
    for r in range(len(TABLE) + 1):  # horizontal rules
        page.draw_line((x0, y0 + r * row_h), (x0 + width, y0 + r * row_h), color=BLACK)
    x = x0
    for c in range(len(col_w) + 1):  # vertical rules
        page.draw_line((x, y0), (x, y0 + height), color=BLACK)
        x += col_w[c] if c < len(col_w) else 0
    for r, row in enumerate(TABLE):
        x = x0
        for c, cell in enumerate(row):
            page.insert_text((x + 5, y0 + r * row_h + 15), cell, fontsize=10)
            x += col_w[c]


def _bar_chart(page: pymupdf.Page, x0: float, y0: float) -> None:
    """A vector bar chart: drawn rectangles + labels (no image inside)."""
    w, h = 300, 140
    page.draw_rect((x0, y0, x0 + w, y0 + h), color=GREY)  # frame
    bar_w, gap = 50, 20
    for i, (label, value) in enumerate(CHART):
        bx = x0 + gap + i * (bar_w + gap)
        top = y0 + h - value * 7
        page.draw_rect((bx, top, bx + bar_w, y0 + h), color=BLUE, fill=BLUE)
        page.insert_text((bx + 15, y0 + h + 12), label, fontsize=9)
        page.insert_text((bx + 12, top - 4), f"{value}", fontsize=8)
    page.insert_text((x0, y0 + h + 32), "Figure 1: Quarterly revenue 2024 (EUR m)", fontsize=10)


def build_sample_pdf() -> bytes:
    doc = pymupdf.open()

    # Page 1: title, prose, table.
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Northwind Bicycles - Annual Report 2024", fontsize=16)
    p1.insert_textbox(pymupdf.Rect(72, 95, 523, 200), INTRO, fontsize=11)
    p1.insert_text((72, 225), "Key figures", fontsize=12)
    _table(p1, 72, 240)

    # Page 2: chart with caption, raster image, outlook prose.
    p2 = doc.new_page()
    _bar_chart(p2, 72, 80)
    # A raster image: a tiny generated gradient, scaled up on the page.
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 64, 64), False)
    pix.set_rect(pix.irect, (200, 120, 40))
    p2.insert_image(pymupdf.Rect(72, 290, 232, 410), pixmap=pix)
    p2.insert_text((72, 425), "Figure 2: The Hauler cargo bike", fontsize=10)
    p2.insert_textbox(pymupdf.Rect(72, 450, 523, 560), OUTLOOK, fontsize=11)

    data = doc.tobytes()
    doc.close()
    return data


if __name__ == "__main__":
    out = Path(__file__).with_name("northwind_report.pdf")
    out.write_bytes(build_sample_pdf())
    print(f"wrote {out}")
