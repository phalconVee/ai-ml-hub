"""The reviewer's tools, run directly (no LLM)."""

from paperlab.paper_tools import PaperIndex, run_tool
from paperlab.pdf_extract import extract_paper
from paperlab.vision import mentions_of


def test_search_ranks_the_page_with_rare_words(sample_pdf):
    paper = extract_paper(sample_pdf)
    hits = PaperIndex(paper).search("incubated 48")
    assert hits and hits[0].page == 2


def test_tools_return_text_or_image(sample_pdf):
    paper = extract_paper(sample_pdf)
    index = PaperIndex(paper)
    assert "[page 1]" in run_tool("search_paper", {"query": "drought phosphatase"}, paper, index)
    assert run_tool("read_page", {"page": 2}, paper, index).startswith("=== Page 2 ===")
    assert "Valid pages" in run_tool("read_page", {"page": 99}, paper, index)
    image, caption = run_tool("view_figure", {"figure_id": "F2"}, paper, index)
    assert image["type"] == "image" and image["source"]["media_type"] == "image/png"
    assert "Fig. 2" in caption["text"]
    assert "Valid ids: F1, F2" in run_tool("view_figure", {"figure_id": "F5"}, paper, index)


def test_mentions_of_finds_citing_sentences_not_fig_21():
    from paperlab.pdf_extract import Figure, Page, Paper
    fig = Figure("F1", 1, b"", caption="Figure 2. Bars.", label="Figure 2")
    paper = Paper("t", [Page(1, "As Fig. 2 shows, yield rose. See Fig. 21 for soil. Figure 2. Bars.")],
                  [fig], total_pages=1)
    assert mentions_of(fig, paper) == "(p.1) As Fig. 2 shows, yield rose."
