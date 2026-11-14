"""Step 0: turning a link or pasted text into clean source text (no network)."""

import pytest

from podcast.source import SourceError, html_to_text, load_source, looks_like_url

from .fakes import ARTICLE_HTML


def test_url_detection():
    assert looks_like_url("https://example.com/a?b=1")
    assert looks_like_url("  http://x.org  ")
    assert not looks_like_url("see https://example.com for more")  # prose containing a link
    assert not looks_like_url("example.com")


def test_html_extraction_keeps_the_article_and_drops_the_chrome():
    title, text = html_to_text(ARTICLE_HTML)
    assert title == "The Bee Dance | Nature Blog"
    assert "waggle dance & other bees" in text            # entities decoded
    assert "one kilometre of flight" in text               # list items kept
    for junk in ("tracking", "color:red", "About us", "newsletter", "Copyright", "Share"):
        assert junk not in text                            # script/style/nav/header/footer/tiny
    assert text.count("angle of the waggle run") == 1      # duplicate paragraph removed


def test_load_source_from_url_uses_the_fetcher():
    seen = []
    src = load_source("https://blog.example/bees", fetch=lambda u: seen.append(u) or ARTICLE_HTML)
    assert seen == ["https://blog.example/bees"]
    assert src.url == "https://blog.example/bees" and src.title.startswith("The Bee Dance")


def test_pasted_text_is_used_as_is_with_first_line_as_title():
    text = "Bee dances\n" + "Bees dance to share where the flowers are. " * 10
    src = load_source(text)
    assert src.url is None and src.title == "Bee dances" and src.text == text.strip()


def test_errors_are_friendly():
    with pytest.raises(SourceError, match="Paste a link"):
        load_source("   ")
    with pytest.raises(SourceError, match="too little text"):
        load_source("Just one short sentence.")
    with pytest.raises(SourceError, match="Couldn't load"):
        load_source("https://down.example", fetch=lambda u: (_ for _ in ()).throw(OSError("timed out")))


def test_long_sources_are_cut_and_flagged(monkeypatch):
    monkeypatch.setattr("podcast.source.SETTINGS",
                        type("S", (), {"min_source_chars": 10, "max_source_chars": 100})())
    src = load_source("word " * 100)
    assert src.truncated and len(src.text) == 100
