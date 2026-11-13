"""Offline tests: no API key, no network. Claude is replaced by small fakes.

They double as worked examples of what each brick produces.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from groundtruth import BM25Index, ask, ingest_pdf
from groundtruth.chunk import chunk_blocks
from groundtruth.config import SETTINGS
from groundtruth.parse import parse_pdf
from groundtruth.tool import run_search
from sample_docs.make_sample import build_sample_pdf

NAME = "northwind_report.pdf"


@pytest.fixture(scope="module")
def pdf() -> bytes:
    return build_sample_pdf()


def fake_describer(png: bytes, caption: str) -> str:
    assert png.startswith(b"\x89PNG")  # the figure really was rendered to an image
    if "Quarterly revenue" in caption:
        return "Bar chart of quarterly revenue: Q1 11.0, Q2 13.5, Q3 14.1, Q4 17.2. Revenue rose every quarter."
    return "Photo of an orange cargo bike."


# --- Brick 1: parsing -------------------------------------------------------

def test_parser_finds_text_table_and_figures(pdf):
    blocks = parse_pdf(pdf, NAME, fake_describer)
    kinds = [(b.page, b.kind) for b in blocks]
    assert (1, "table") in kinds
    assert kinds.count((2, "figure")) == 2

    table = next(b for b in blocks if b.kind == "table")
    assert table.text.splitlines()[0] == "| Metric | 2023 | 2024 |"
    assert "| Gross margin (%) | 38.5 | 34.1 |" in table.text

    chart = next(b for b in blocks if b.kind == "figure")
    assert chart.text.startswith("Figure 1: Quarterly revenue")
    assert "Q4 17.2" in chart.text                  # vision description attached

    texts = [b.text for b in blocks if b.kind == "text"]
    assert not any(t in {"Q1", "Q2", "Q3", "Q4"} for t in texts)  # axis labels dropped
    assert not any("41.2" in t for t in texts)                    # table cells not duplicated as prose


def test_parser_without_vision_keeps_captions(pdf):
    figures = [b for b in parse_pdf(pdf, NAME) if b.kind == "figure"]
    assert [f.text for f in figures] == ["Figure 1: Quarterly revenue 2024 (EUR m)", "Figure 2: The Hauler cargo bike"]


# --- Brick 2: chunking ------------------------------------------------------

def test_tables_stay_whole_and_keep_their_header(pdf):
    chunks = chunk_blocks(parse_pdf(pdf, NAME))
    table = next(c for c in chunks if c.kind == "table")
    assert table.pieces[0].startswith("| Metric")       # header piece first
    assert len(table.pieces) == 1 + 3                   # header + 3 data rows
    assert table.source == f"{NAME}#page=1"
    assert all(len({c.page}) == 1 for c in chunks)


def test_long_prose_is_windowed_with_overlap():
    from groundtruth.parse import Block
    paras = [f"Paragraph {i} " + "word " * 60 for i in range(10)]   # ~620 words
    chunks = chunk_blocks([Block("x.pdf", 1, "text", p) for p in paras])
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= SETTINGS.chunk_words + 62 for c in chunks)
    # Overlap: the last piece of one window starts the next one.
    assert chunks[0].pieces[-1] == chunks[1].pieces[0]


# --- Brick 3: retrieval -----------------------------------------------------

def test_bm25_ranks_the_right_passage_first(pdf):
    index = BM25Index()
    ingest_pdf(index, pdf, NAME, fake_describer)
    assert index.search("gross margin 2024")[0][0].kind == "table"
    assert index.search("quarterly revenue Q4")[0][0].kind == "figure"
    assert "battery" in index.search("battery cell supplier Poland")[0][0].text
    assert index.search("zebra giraffe") == []


# --- Brick 4: the tool ------------------------------------------------------

def test_tool_returns_citable_search_results(pdf):
    index = BM25Index()
    ingest_pdf(index, pdf, NAME)
    blocks, hits = run_search(index, {"query": "bikes shipped"})
    assert hits >= 1
    first = blocks[0]
    assert first["type"] == "search_result" and first["citations"] == {"enabled": True}
    assert first["title"] == f"{NAME}, page 1 (table)"
    assert all(b["type"] == "text" and b["text"] for b in first["content"])

    # Bad inputs come back as text Claude can read and recover from, not exceptions.
    assert run_search(index, {"query": ""})[1] == 0
    assert "Available" in run_search(index, {"query": "x", "document": "nope.pdf"})[0][0]["text"]


# --- Brick 5: the agent loop, with a scripted fake Claude ---------------------

class FakeClaude:
    """Mimics client.beta.messages.create: first asks for a search, then answers with citations."""

    def __init__(self):
        self.calls = []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kwargs):
        # Snapshot the message list: the agent keeps appending to the same list.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if len(self.calls) == 1:
            return NS(stop_reason="tool_use", content=[
                NS(type="thinking", thinking="", signature="sig"),
                NS(type="tool_use", id="tu_1", name="search_documents", input={"query": "gross margin 2024"}),
            ])
        cite = NS(type="search_result_location", source=f"{NAME}#page=1",
                  title=f"{NAME}, page 1 (table)", cited_text="| Gross margin (%) | 38.5 | 34.1 |",
                  search_result_index=0, start_block_index=2, end_block_index=3)
        return NS(stop_reason="end_turn", content=[
            NS(type="text", text="Gross margin fell from 38.5% to 34.1%", citations=[cite]),
            NS(type="text", text=", mainly because battery costs rose.", citations=[cite]),
        ])


def test_agent_searches_then_answers_with_citations(pdf):
    index = BM25Index()
    ingest_pdf(index, pdf, NAME)
    fake = FakeClaude()
    events = []

    answer = ask("What happened to gross margin?", index, client=fake, on_event=events.append)

    # Round 1 request: our tool and settings were sent.
    first = fake.calls[0]
    assert first["model"] == SETTINGS.model
    assert first["tools"][0]["name"] == "search_documents"
    assert first["tool_choice"] == {"type": "auto"}

    # Round 2 request: assistant turn echoed unchanged, then tool results as search_result blocks.
    second = fake.calls[1]["messages"]
    assert second[1]["role"] == "assistant" and second[1]["content"][0].type == "thinking"
    tool_result = second[2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "tu_1"
    assert tool_result["content"][0]["type"] == "search_result"

    # The answer: one distinct passage -> one citation number, reused by both segments.
    assert answer.markdown() == "Gross margin fell from 38.5% to 34.1% [1], mainly because battery costs rose. [1]"
    assert [c.number for c in answer.citations] == [1]
    assert answer.citations[0].title.endswith("(table)")
    assert answer.searches == [{"query": "gross margin 2024", "document": None, "hits": answer.searches[0]["hits"]}]
    assert events == ["Searching: 'gross margin 2024'"]
    assert answer.stop_reason == "end_turn"


def test_agent_stops_searching_after_the_round_cap(pdf):
    index = BM25Index()
    ingest_pdf(index, pdf, NAME)

    class Loopy(FakeClaude):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["tool_choice"] == {"type": "none"}:
                return NS(stop_reason="end_turn", content=[NS(type="text", text="Best effort.", citations=None)])
            return NS(stop_reason="tool_use", content=[
                NS(type="tool_use", id=f"tu_{len(self.calls)}", name="search_documents", input={"query": "revenue"})])

    fake = Loopy()
    answer = ask("anything", index, client=fake)
    assert len(fake.calls) == SETTINGS.max_tool_rounds + 1
    assert answer.markdown() == "Best effort."


def test_agent_handles_refusal(pdf):
    index = BM25Index()
    ingest_pdf(index, pdf, NAME)
    client = NS(beta=NS(messages=NS(create=lambda **_: NS(stop_reason="refusal", content=[]))))
    assert "declined" in ask("q", index, client=client).markdown()
