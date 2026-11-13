"""Every knob in one place.

Change a number here and the whole app follows. Nothing else in the package
hard-codes a model name, a page limit or a loop cap.
"""

from dataclasses import dataclass


@dataclass(frozen=True)  # frozen: settings are read, never changed at runtime
class Settings:
    # ── The model ────────────────────────────────────────────────────────────
    # Claude Opus 5.5: the current default Opus. It reads images natively
    # (that's our OCR and figure reader) and always thinks before answering;
    # `effort` is the only dial for how hard (low → max).
    model: str = "claude-opus-5-5"
    effort: str = "medium"        # Opus 5.5's default, said out loud. "high" for dense papers.
    max_tokens: int = 16000       # room for thinking + answer on a non-streaming call

    # Server-side refusal fallback: if a safety classifier declines a request
    # (e.g. a biology paper tripping the `bio` classifier), the API re-runs it
    # on a fallback model inside the same call. "default" lets Anthropic pick.
    fallback_beta: str = "server-side-fallback-2026-07-01"

    # ── PDF extraction (pdf_extract.py) ──────────────────────────────────────
    max_pages: int = 60           # longer PDFs are cut here (supplements, theses)
    scanned_text_chars: int = 200  # a page with less selectable text than this is "scanned"
    render_dpi: int = 110         # DPI for page/figure images sent to Claude
    min_figure_pts: float = 90.0  # a finished figure narrower or shorter than this is dropped (logos, icons)
    min_piece_pts: float = 8.0    # pieces smaller than this are noise (rules, bullets, specks)
    caption_gap_pts: float = 80.0  # how far (in PDF points) a caption may sit from its figure

    # ── Vision budget (vision.py) ────────────────────────────────────────────
    # Every figure and every scanned page is one vision call, so these caps
    # are the main cost control.
    max_figures: int = 8
    max_ocr_pages: int = 10

    # ── Analysis (analysts.py, reviewer.py) ──────────────────────────────────
    # Opus 5.5 has a 1M-token context, so a whole paper fits in one prompt.
    # We still cap it so a 300-page PDF can't produce a surprise bill.
    max_paper_chars: int = 250_000
    max_reviewer_turns: int = 6   # tool rounds before the reviewer must deliver
    search_hits: int = 4          # passages returned by the reviewer's search tool

    # ── Grounding check (grounding.py) ───────────────────────────────────────
    # Share of a quote's words that must appear, in order-insensitive windows,
    # in the paper for it to count as "found". 1.0 = exact words only.
    quote_match_threshold: float = 0.8


SETTINGS = Settings()
