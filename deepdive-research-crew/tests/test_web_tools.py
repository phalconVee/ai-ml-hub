"""The raw web abilities, tested offline. Run with: uv run pytest"""

from deepdive.web_tools import FAKE_PAGES, _is_public_http_url, html_to_text, read_page, search


def test_html_to_text_keeps_article_drops_clutter():
    html = """<html><head><title> Demo page </title><style>body{}</style></head>
    <body><nav>Home | About</nav><h1>Big news</h1><p>First   paragraph.</p>
    <script>alert('x')</script><p>Second paragraph.</p><footer>(c) 2026</footer></body></html>"""
    title, text = html_to_text(html)
    assert title == "Demo page"
    assert "Big news" in text and "First paragraph." in text and "Second paragraph." in text
    # Menus, scripts, styles and footers are not article text.
    for junk in ("Home | About", "alert", "body{}", "(c) 2026"):
        assert junk not in text
    # Paragraphs stay separate lines instead of running together.
    assert "First paragraph.\n" in text


def test_only_public_web_urls_are_fetchable():
    assert _is_public_http_url("https://en.wikipedia.org/wiki/MCP")
    for bad in ("http://localhost:8501", "http://127.0.0.1/", "http://192.168.1.1/admin",
                "file:///etc/passwd", "ftp://example.com", "http://router.local/"):
        assert not _is_public_http_url(bad), bad


def test_fake_backend_is_deterministic_and_readable():
    hits = search("anything", max_results=2, backend="fake")
    assert len(hits) == 2
    page = read_page(hits[0].url)
    assert page["text"] == FAKE_PAGES[hits[0].url][1]
