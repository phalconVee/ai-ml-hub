"""Shared test fixtures: a tiny synthetic "paper" built with PyMuPDF.

Building the PDF in code (instead of shipping a binary file) means the test
shows exactly what's on each page:

  page 1: title + abstract text, a raster image with a "Figure 1." caption below
  page 2: methods/results text, a vector bar chart with a "Fig. 2" caption below
  page 3: no text at all, just a full-page image → should be detected as scanned
"""

import pymupdf
import pytest

FILLER = ("We measured enzyme activity across twelve soil samples and report the "
          "mean response with standard error. ") * 4


def _solid_png(w: int, h: int, rgb=(40, 120, 200)) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, w, h), False)
    pix.set_rect(pix.irect, rgb)
    return pix.tobytes("png")


def build_sample_pdf() -> bytes:
    doc = pymupdf.open()

    # Page 1 — text + raster figure + caption below it.
    p1 = doc.new_page()  # A4-ish 595 x 842 pt
    p1.insert_text((72, 72), "Soil Enzymes Under Drought: A Field Study", fontsize=16)
    p1.insert_textbox(pymupdf.Rect(72, 90, 523, 250),
                      "Abstract. Drought reduced phosphatase activity by 31% (p < 0.01). " + FILLER,
                      fontsize=10)
    p1.insert_image(pymupdf.Rect(150, 300, 450, 500), stream=_solid_png(300, 200))
    p1.insert_textbox(pymupdf.Rect(150, 510, 450, 560),
                      "Figure 1. Map of the twelve sampling sites.", fontsize=9)

    # Page 2 — text + vector "bar chart" + caption.
    p2 = doc.new_page()
    p2.insert_textbox(pymupdf.Rect(72, 60, 523, 250),
                      "Methods. Samples were incubated at 25 C for 48 h. Results. " + FILLER,
                      fontsize=10)
    shape = p2.new_shape()
    shape.draw_line((120, 560), (480, 560))   # x axis
    shape.draw_line((120, 560), (120, 340))   # y axis
    for i, height in enumerate([150, 90, 180, 60]):
        x = 150 + i * 80
        shape.draw_rect(pymupdf.Rect(x, 560 - height, x + 40, 560))
    shape.finish(color=(0, 0, 0), fill=(0.8, 0.3, 0.3))
    shape.commit()
    p2.insert_textbox(pymupdf.Rect(120, 575, 480, 620),
                      "Fig. 2 Phosphatase activity by treatment.", fontsize=9)

    # Page 3 — a "scan": an image covering the page, no text layer.
    p3 = doc.new_page()
    p3.insert_image(p3.rect, stream=_solid_png(400, 560, (230, 230, 220)))

    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(scope="session")
def sample_pdf() -> bytes:
    return build_sample_pdf()


# ─────────────────────────────────────────────────────────────────────────────
# A fake Claude client
# ─────────────────────────────────────────────────────────────────────────────
#
# It mimics `client.beta.messages.create(...)` and answers based on WHICH agent
# is calling (recognised from the system prompt). Routing by role instead of
# by call order matters because the pipeline makes calls from parallel threads.

import json
import threading
from types import SimpleNamespace


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use(id_, name, args):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=args)


def reply(*blocks, stop="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop, stop_details=None)


FIGURE_NOTE = {"figure_type": "bar chart", "what_it_shows": "Four bars by treatment.",
               "key_takeaway": "Treatment C is highest.", "quantitative_details": ["n = 12"],
               "caveats": ["No error bars shown."]}
METHODS_NOTE = {"research_question": "Does drought change enzyme activity?",
                "study_type": "field study", "design_summary": "12 sites, 2 treatments.",
                "data_and_samples": "12 soil samples", "techniques": ["incubation"],
                "statistics": "t-test", "reproducibility_notes": "No code link.",
                "evidence": [{"quote": "Samples were incubated at 25 C for 48 h.", "page": 2}]}
FINDINGS_NOTE = {"findings": [{"claim": "Drought cut phosphatase activity by 31%.",
                               "evidence": {"quote": "Drought reduced phosphatase activity by 31%", "page": 1},
                               "figure_ids": ["F2"], "strength": "moderate"}],
                 "stated_limitations": []}
REPORT = {
    "title": "Soil Enzymes Under Drought", "field": "Soil ecology",
    "one_line_summary": "Drought lowers phosphatase activity.",
    "executive_summary": "A small field study.", "research_question": "Does drought matter?",
    "methods": {"study_design": "Field study", "data": "12 samples",
                "techniques": ["incubation"], "statistics": "t-test"},
    "key_findings": [
        # real quote, right page → verified
        {"claim": "Phosphatase down 31%.", "confidence": "medium", "figure_ids": ["F2", "F9"],
         "evidence": {"quote": "Drought reduced phosphatase activity by 31% (p < 0.01).", "page": 1}},
        # real quote, wrong page (it's on page 2) → wrong_page
        {"claim": "Incubated 48 h.", "confidence": "high", "figure_ids": [],
         "evidence": {"quote": "Samples were incubated at 25 C for 48 h.", "page": 1}},
    ],
    "figure_insights": [{"figure_id": "F2", "insight": "Bars differ."},
                        {"figure_id": "F7", "insight": "Does not exist."}],
    "limitations": [
        # invented quote → not_found
        {"limitation": "Only one season sampled.", "source": "reviewer_inferred",
         "evidence": {"quote": "the authors sampled during a single exceptionally wet winter season", "page": 2}},
    ],
    "open_questions": ["Other enzymes?"], "practical_implications": ["Irrigation timing."],
    "reviewer_notes": ["Downgraded finding 1: no error bars."],
}


class FakeClient:
    """Records every request; replies per agent role. The reviewer is scripted:
    turn 1 asks for two tools, turn 2 returns the report (unless `reviewer_script`
    is overridden)."""

    def __init__(self, reviewer_script=None):
        self.calls = []
        self._lock = threading.Lock()
        self.reviewer_script = reviewer_script or [
            reply(text_block("Let me check."),
                  tool_use("tu_1", "search_paper", {"query": "phosphatase drought"}),
                  tool_use("tu_2", "view_figure", {"figure_id": "F2"}), stop="tool_use"),
            reply(text_block(json.dumps(REPORT))),
        ]
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        with self._lock:
            # Deep-ish copy of messages: the reviewer mutates its list later.
            self.calls.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
            system = kwargs["system"]
            if "transcribe" in system:
                return reply(text_block("OCR TEXT: Table 3. Site rainfall in mm."))
            if "reading one figure" in system:
                return reply(text_block(json.dumps(FIGURE_NOTE)))
            if "methods analyst" in system:
                return reply(text_block(json.dumps(METHODS_NOTE)))
            if "findings analyst" in system:
                return reply(text_block(json.dumps(FINDINGS_NOTE)))
            if "senior reviewer" in system:
                n = sum(1 for c in self.calls if "senior reviewer" in c["system"])
                return self.reviewer_script[min(n, len(self.reviewer_script)) - 1]
            raise AssertionError(f"unexpected system prompt: {system[:60]}")

    def calls_for(self, needle):
        return [c for c in self.calls if needle in c["system"]]


@pytest.fixture
def fake_client():
    return FakeClient()
