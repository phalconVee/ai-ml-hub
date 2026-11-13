"""The raw web abilities: search the web, read a page.

These are plain Python functions with no MCP and no LLM in sight, so they can
be tested on their own. `tools_server.py` wraps them as MCP tools; the agents
only ever reach them through MCP.

Two jobs:
  1. `search(query)`   -> a short list of {title, url, snippet}
  2. `read_page(url)`  -> the readable text of one page (scripts, menus and
                          other clutter stripped), truncated to a sane size
"""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx

from deepdive.config import PAGE_CHAR_LIMIT, SEARCH_BACKEND, SEARCH_RESULTS

# A browser-like User-Agent: many sites return 403 to the default httpx one.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36 deepdive-research-crew/0.1",
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
}


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 1. Search
# ---------------------------------------------------------------------------


def search(query: str, max_results: int = SEARCH_RESULTS, backend: str | None = None) -> list[SearchHit]:
    """Run one web search on the configured backend and normalise the results."""
    backend = (backend or os.getenv("DEEPDIVE_SEARCH_BACKEND") or SEARCH_BACKEND).lower()
    if backend == "fake":
        return _search_fake(query, max_results)
    if backend == "tavily":
        return _search_tavily(query, max_results)
    if backend == "duckduckgo":
        return _search_duckduckgo(query, max_results)
    raise ValueError(f"Unknown search backend: {backend!r} (use duckduckgo, tavily or fake)")


def _search_duckduckgo(query: str, max_results: int) -> list[SearchHit]:
    # `ddgs` (formerly duckduckgo_search) scrapes public search engines: no key,
    # but it can be rate-limited if you hammer it.
    # It is flaky by nature (engines rate-limit or change their HTML), and a
    # second try a moment later often succeeds, so retry twice before failing.
    import time

    from ddgs import DDGS

    for attempt in range(3):
        try:
            rows = DDGS().text(query, max_results=max_results) or []
            return [SearchHit(r.get("title", ""), r.get("href", ""), r.get("body", "")) for r in rows]
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    return []  # unreachable; keeps type checkers happy


def _search_tavily(query: str, max_results: int) -> list[SearchHit]:
    # Tavily is a search API built for LLM agents. We call its REST endpoint
    # directly with httpx rather than adding another SDK dependency.
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("DEEPDIVE_SEARCH_BACKEND=tavily but TAVILY_API_KEY is not set")
    resp = httpx.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {key}"},
        json={"query": query, "max_results": max_results, "search_depth": "advanced"},
        timeout=30,
    )
    resp.raise_for_status()
    rows = resp.json().get("results", [])
    return [SearchHit(r.get("title", ""), r.get("url", ""), r.get("content", "")) for r in rows]


# Canned pages for offline runs. URLs use the reserved `.test` TLD so they can
# never collide with a real site.
FAKE_PAGES: dict[str, tuple[str, str]] = {
    "https://alpha.example.test/report": (
        "Alpha Institute annual report",
        "The Alpha Institute measured a 42% rise in adoption between 2024 and 2026. "
        "The authors caution that the sample covers only large companies.",
    ),
    "https://beta.example.test/blog": (
        "Beta Blog: a sceptical take",
        "Beta's survey of small firms found adoption grew by only 12% over the same period, "
        "and costs were the main barrier.",
    ),
    "https://gamma.example.test/news": (
        "Gamma News explainer",
        "Analysts say the gap between large and small companies is widening. "
        "Regulation is expected in 2027.",
    ),
}


def _search_fake(query: str, max_results: int) -> list[SearchHit]:
    hits = [SearchHit(title, url, body[:120]) for url, (title, body) in FAKE_PAGES.items()]
    return hits[:max_results]


# ---------------------------------------------------------------------------
# 2. Read a page
# ---------------------------------------------------------------------------

# Tags whose contents are never article text.
_SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe", "aside"}
# Tags that end a "line" of text, so paragraphs don't run together.
_BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "pre"}


class _TextExtractor(HTMLParser):
    """A tiny HTML -> text converter built on the standard library.

    It walks the tag stream, ignores anything inside `_SKIP_TAGS`, and inserts
    newlines at block boundaries. Not as clever as a readability library, but
    dependency-free and easy to read.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0  # >0 while we are inside a skipped tag
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)  # squash runs of spaces
        raw = re.sub(r"\n\s*\n+", "\n\n", raw)  # at most one blank line
        return raw.strip()


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, readable_text) for an HTML document."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.title.strip(), parser.text()


def _is_public_http_url(url: str) -> bool:
    """Refuse non-http(s) schemes and obvious local/private addresses.

    The agents choose which URLs to open, and their choices come from web
    content we don't control. This stops a malicious page from steering the
    crew into reading `http://localhost:...` or your home router.
    """
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    host = parts.hostname.lower()
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True  # a domain name; fine
    return ip.is_global


def read_page(url: str, max_chars: int = PAGE_CHAR_LIMIT) -> dict:
    """Fetch a URL and return {url, title, text, truncated}.

    Errors are raised as exceptions; the MCP layer turns them into
    `is_error` tool results that the model can read and react to.
    """
    if url in FAKE_PAGES:  # offline mode
        title, body = FAKE_PAGES[url]
        return {"url": url, "title": title, "text": body, "truncated": False}

    if not _is_public_http_url(url):
        raise ValueError(f"Refusing to fetch non-public URL: {url}")

    resp = httpx.get(url, headers=_HEADERS, timeout=20, follow_redirects=True)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    if "html" in ctype:
        title, text = html_to_text(resp.text)
    elif ctype.startswith("text/") or "json" in ctype:
        title, text = "", resp.text
    else:
        raise ValueError(f"Unsupported content type {ctype!r} (PDFs and binaries are not handled)")

    truncated = len(text) > max_chars
    return {"url": str(resp.url), "title": title, "text": text[:max_chars], "truncated": truncated}
