"""The whole workflow, start to finish. Read this file to see the big picture.

    PDF bytes
      │ 1. extract      (pdf_extract.py, no LLM)    text per page, figure PNGs, scanned pages
      │ 2. OCR          (vision.py, parallel)       Claude transcribes scanned pages
      │ 3. figures      (vision.py, parallel)       Claude reads each figure → FIGURE_NOTE
      │ 4. analysts     (analysts.py, parallel)     methods draft ‖ findings draft
      │ 5. reviewer     (reviewer.py, agent loop)   verifies with tools → final REPORT
      │ 6. grounding    (grounding.py, no LLM)      every quote looked up in the PDF
      ▼
    AnalysisResult

The order is fixed in code; only step 5 lets the model choose what to do.
That's deliberate: the steps where order matters (OCR before analysis,
analysis before review) aren't left to a prompt.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from . import analysts, vision
from .config import SETTINGS
from .grounding import ground_report
from .pdf_extract import Paper, extract_paper
from .reviewer import review

# Parallel Claude calls. The SDK client is thread-safe; 4 keeps us well under
# typical rate limits while cutting wall-clock time for 8 figures roughly 4×.
WORKERS = 4


@dataclass
class AnalysisResult:
    paper: Paper
    figure_notes: dict[str, dict]       # fig_id → FIGURE_NOTE
    methods_draft: dict
    findings_draft: dict
    report: dict                        # final REPORT, annotated with `grounding`
    grounding: dict                     # summary counts from ground_report
    trace: list[str] = field(default_factory=list)  # human-readable log for the UI


def analyse_pdf(pdf_bytes: bytes, client: Any,
                on_event: Callable[[str], None] = lambda _msg: None) -> AnalysisResult:
    trace: list[str] = []

    def log(msg: str) -> None:
        trace.append(msg)
        on_event(msg)

    # ── 1. Extract ───────────────────────────────────────────────────────────
    paper = extract_paper(pdf_bytes)
    log(f"📄 Extracted {len(paper.pages)} pages, {len(paper.figures)} figures, "
        f"{len(paper.scanned_pages)} scanned page(s).")
    for w in paper.warnings:
        log(f"⚠️ {w}")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        # ── 2. OCR scanned pages ─────────────────────────────────────────────
        to_ocr = paper.scanned_pages[:SETTINGS.max_ocr_pages]
        if len(paper.scanned_pages) > len(to_ocr):
            log(f"⚠️ Only the first {len(to_ocr)} scanned pages are OCR'd (max_ocr_pages).")
        if to_ocr:
            log(f"👁️ OCR: transcribing {len(to_ocr)} scanned page(s) with Claude vision…")
            texts = pool.map(lambda p: vision.ocr_page(client, p.image_png, p.number), to_ocr)
            for page, text in zip(to_ocr, texts):
                page.text, page.ocr_done = text, True

        # ── 3. Figures ───────────────────────────────────────────────────────
        # After OCR, so `mentions_of` can find "Fig. 2" on scanned pages too.
        figs = paper.figures[:SETTINGS.max_figures]
        if len(paper.figures) > len(figs):
            log(f"⚠️ Only the first {len(figs)} of {len(paper.figures)} figures are read (max_figures).")
        figure_notes: dict[str, dict] = {}
        if figs:
            log(f"🖼️ Reading {len(figs)} figure(s) with Claude vision…")
            for fig, note in zip(figs, pool.map(lambda f: vision.read_figure(client, f, paper), figs)):
                figure_notes[fig.fig_id] = note

        # ── 4. Analysts, in parallel ─────────────────────────────────────────
        log("🧪 Methods and findings analysts drafting in parallel…")
        context = analysts.paper_context(paper, figure_notes)
        methods_future = pool.submit(analysts.analyse_methods, client, context)
        findings_future = pool.submit(analysts.analyse_findings, client, context)
        methods, findings = methods_future.result(), findings_future.result()
        log(f"   drafts in: {len(findings.get('findings', []))} findings, "
            f"{len(methods.get('evidence', []))} methods quotes.")

    # ── 5. Reviewer agent ────────────────────────────────────────────────────
    log("🧑‍⚖️ Reviewer verifying drafts against the paper…")
    report = review(client, paper, context, methods, findings, on_event=log)

    # ── 6. Grounding check ───────────────────────────────────────────────────
    grounding = ground_report(report, paper)
    log(f"✅ Grounding: {grounding['verified']} verified, {grounding['wrong_page']} wrong page, "
        f"{grounding['not_found']} not found.")

    return AnalysisResult(paper, figure_notes, methods, findings, report, grounding, trace)
