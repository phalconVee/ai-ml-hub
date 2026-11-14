"""Ingest and search: no LLM involved, so these are plain unit tests."""

from counsel.ingest import chunk_document, split_clauses, windows
from counsel.search import ClauseIndex, tokenize
from sample_docs.make_sample import build_sample_pdf

NAME = "sample_lease.pdf"


def lease_chunks():
    return chunk_document(build_sample_pdf(), NAME)


def test_pdf_splits_into_one_chunk_per_clause_with_pages():
    chunks = lease_chunks()
    headings = [c.heading for c in chunks]
    assert "3. Late Payment" in headings and "10. Governing Law" in headings
    early = next(c for c in chunks if c.heading == "5. Early Termination")
    assert early.page == 2                       # page numbers survive pypdf
    assert "sixty (60) days" in early.text
    assert early.id == f"{NAME}#p2-c6"
    assert early.title == f"{NAME}, p.2, 5. Early Termination"


def test_heading_regex_ignores_numbers_mid_sentence():
    text = "4.2 Late Payment\nRent is due on day 1.\n30 days after notice the fee applies.\nARTICLE IX\nMisc."
    assert [h for h, _ in split_clauses(text)] == ["4.2 Late Payment", "ARTICLE IX"]


def test_clause_continuing_on_next_page_keeps_its_heading():
    data = "7. Entry by Landlord\nLandlord may enter\fafter 24 hours notice.\n8. Default\nText".encode()
    chunks = chunk_document(data, "x.txt")
    assert [(c.page, c.heading) for c in chunks] == [
        (1, "7. Entry by Landlord"), (2, "7. Entry by Landlord (cont.)"), (2, "8. Default")]


def test_long_clause_becomes_overlapping_windows():
    words = [str(i) for i in range(500)]
    w = windows(words, 220, 40)
    assert w[0][-40:] == w[1][:40]               # overlap carries context across the cut
    assert w[-1][-1] == "499"                    # nothing lost at the end


def test_tokenize_keeps_legal_modal_words():
    assert tokenize("Tenant shall NOT pay the fee") == ["tenant", "shall", "not", "pay", "fee"]


def test_bm25_ranks_the_right_clause_first():
    index = ClauseIndex()
    index.add(lease_chunks())
    top, score = index.search("late fee overdue rent", k=3)[0]
    assert top.heading == "3. Late Payment" and score > 0
    assert index.search("terminate early notice")[0][0].heading == "5. Early Termination"
    assert index.search("zebra quantum") == []   # no shared word → no hits
    assert index.search("rent", doc="other.pdf") == []
