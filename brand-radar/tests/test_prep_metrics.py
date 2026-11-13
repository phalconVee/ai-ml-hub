"""The plain-Python parts: filtering, de-duplication, the numbers, citation checks.

These are the pieces that keep the LLM honest, so they get worked examples.
"""

from types import SimpleNamespace as NS

from radar.agents import check_verdicts, mentions_from_search
from radar.collectors import clean, ddg_timelimit
from radar.metrics import check_citations, compute, net_sentiment
from radar.prep import mentions_brand, prepare
from radar.schemas import AnalystOutput, BrandQuery, Mention, SourceReport, Verdict

Q = BrandQuery("Arc", context="web browser", aliases=["Arc Browser"], exclude=["arc welding"])


def m(source, text, url="", title=""):
    return Mention(source, title, url or f"https://x.test/{abs(hash(text))}", text)


def v(id, sentiment="neutral", relevant=True, action="none", urgency=0, themes=()):
    return Verdict(id=id, relevant=relevant, sentiment=sentiment, themes=list(themes),
                   action=action, urgency=urgency, summary="s")


def test_brand_must_appear_as_a_whole_word():
    assert mentions_brand(m("web", "I switched to Arc last week"), Q)
    assert not mentions_brand(m("web", "searching the archive"), Q)    # 'arc' inside words
    assert not mentions_brand(m("web", "Arc welding for beginners"), Q)  # excluded phrase


def test_prepare_filters_dedupes_and_labels():
    raw = {
        "news": [m("news", "Arc ships a new release for Windows users today", "https://a.test/1?utm=x"),
                 m("news", "Nothing relevant here at all", "https://a.test/2")],
        "web": [m("web", "copy", "https://www.a.test/1/"),     # same URL as news-1, different spelling
                m("web", "Arc ships a new release for Windows users today and more", "https://b.test"),  # syndicated copy
                m("web", "Arc is slow on my laptop", "https://c.test")],
    }
    clean_, dropped = prepare(raw, Q)
    assert [x.id for x in clean_["news"]] == ["news-1"]
    assert [x.text for x in clean_["web"]] == ["Arc is slow on my laptop"]
    assert clean_["web"][0].id == "web-1"           # ids are per source and contiguous
    assert dropped == {"news": 1, "web": 2}


def test_query_strings_that_identify_pages_survive_dedupe():
    raw = {"hackernews": [m("hackernews", "Arc one", "https://news.ycombinator.com/item?id=1"),
                          m("hackernews", "Arc two, other words", "https://news.ycombinator.com/item?id=2"),
                          m("hackernews", "Arc again", "https://news.ycombinator.com/item?id=1&utm_source=x")]}
    clean_, _ = prepare(raw, Q)
    assert [x.url for x in clean_["hackernews"]] == ["https://news.ycombinator.com/item?id=1",
                                                       "https://news.ycombinator.com/item?id=2"]


def test_check_verdicts_drops_unknown_ids_and_reports_missing():
    batch = [m("bluesky", "Arc a"), m("bluesky", "Arc b")]
    batch[0].id, batch[1].id = "bluesky-1", "bluesky-2"
    out = AnalystOutput(verdicts=[v("bluesky-1", urgency=9), v("bluesky-99")], source_summary="x")
    kept, problems = check_verdicts(batch, out)
    assert list(kept) == ["bluesky-1"]
    assert kept["bluesky-1"].urgency == 3            # clamped into 0..3
    assert any("bluesky-99" in p for p in problems) and any("bluesky-2" in p for p in problems)


def test_net_sentiment_and_compute():
    assert net_sentiment([]) is None
    assert net_sentiment(["positive", "negative", "neutral", "positive"]) == 25
    a = [m("news", "Arc"), m("news", "Arc")]
    a[0].id, a[1].id = "news-1", "news-2"
    reports = {"news": SourceReport("news", a, {
        "news-1": v("news-1", "negative", action="fix", themes=["Crashes"]),
        "news-2": v("news-2", "positive", relevant=False),  # namesake: must not count
    })}
    stats = compute(reports)
    assert stats["relevant"] == 1 and stats["net_sentiment"] == -100
    assert stats["top_themes"] == [{"theme": "crashes", "count": 1, "sentiment": {"negative": 1}}]
    assert stats["actions"] == {"fix": 1}


def test_check_citations():
    assert check_citations("Do X [news-1][reddit-2] and [claude_sweep-1]", {"news-1", "reddit-2", "claude_sweep-1"}) == []
    assert check_citations("See [news-9].", {"news-1"}) == ["news-9"]


def test_sweep_citations_become_mentions():
    cite = lambda url, t: NS(url=url, title="T", cited_text=t)
    blocks = [NS(type="server_tool_use"),
              NS(type="text", citations=[cite("https://r.test", "Arc crashed again"), cite("https://r.test", "fix it")]),
              NS(type="text", citations=None)]
    found = mentions_from_search(blocks)
    assert len(found) == 1 and found[0].text == "Arc crashed again ... fix it"


def test_small_helpers():
    assert clean("<p>Arc &amp; <b>friends</b>&#x27;</p>") == "Arc & friends '"
    assert [ddg_timelimit(d) for d in (1, 7, 14, 90)] == ["d", "w", "m", "y"]
