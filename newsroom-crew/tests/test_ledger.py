"""The ledger is plain code, so it gets plain, exact tests."""

from newsroom.ledger import build_ledger, render_ledger

from .fakes import BANK, BLOG, WIRE, cite, fetch, reporter_turns, search, search_error, text


def all_blocks():
    return [b for m in reporter_turns() for b in m.content]


def test_quoted_sources_rank_first_and_get_ids():
    sources = build_ledger(all_blocks())
    # BANK: read + quoted; WIRE: quoted; BLOG: only a search hit → last.
    assert [s.url for s in sources] == [BANK, WIRE, BLOG]
    assert [s.id for s in sources] == ["S1", "S2", "S3"]
    assert sources[0].read and not sources[1].read
    assert sources[0].quotes == ["The key rate remains at 4.25%."]
    assert sources[1].age == "1 day ago"


def test_same_page_different_spelling_is_one_source():
    blocks = [*search("q", (WIRE + "/", "Bank holds", None)),
              text("x", citations=[cite(WIRE + "#para2", "two of nine")]),
              text("y", citations=[cite(WIRE, "two of nine")])]   # duplicate quote
    sources = build_ledger(blocks)
    assert len(sources) == 1
    assert sources[0].quotes == ["two of nine"]                  # kept once


def test_search_errors_are_skipped_not_crashed_on():
    assert build_ledger(search_error("anything")) == []


def test_fetch_title_upgrades_a_bare_url():
    blocks = [text("x", citations=[cite(BANK, "quote", title=None)]), *fetch(BANK, "Policy statement")]
    [s] = build_ledger(blocks)
    assert s.title == "Policy statement" and s.read


def test_render_shows_ids_outlets_and_quotes():
    out = render_ledger(build_ledger(all_blocks()))
    assert "[S1] Monetary policy statement (centralbank.example, read in full)" in out
    assert '• "two of nine members voted to cut"' in out
    assert render_ledger([]) == "(no sources were found)"
