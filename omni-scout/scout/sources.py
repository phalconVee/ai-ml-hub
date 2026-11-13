"""Raw access to each platform, with no AI involved.

Each function takes a query and returns a list of `Hit` dicts with the same keys,
whatever the platform. Normalising here means the MCP server, the agents and the
UI never have to know that Hacker News calls it `objectID` and DuckDuckGo calls
it `href`.

Every source here is free and needs no API key:
  * DuckDuckGo (via the `ddgs` package) for web, news, video and Reddit
  * the Hacker News search API run by Algolia
  * Wikipedia's REST search
"""

import asyncio
import re
from typing import TypedDict

import httpx
from ddgs import DDGS
from ddgs.exceptions import DDGSException

# Be a polite client: identify ourselves and never hang forever.
HEADERS = {"User-Agent": "omni-scout/0.1 (research demo)"}
TIMEOUT = httpx.Timeout(15.0)


class Hit(TypedDict):
    title: str
    url: str
    snippet: str
    published: str  # ISO-ish date string, or "" when the platform gives none
    meta: str       # platform-specific extras, e.g. "312 points, 98 comments"


def _hit(title: str = "", url: str = "", snippet: str = "", published: str = "", meta: str = "") -> Hit:
    """Build a Hit, squashing whitespace so snippets stay compact."""
    clean = lambda s: re.sub(r"\s+", " ", str(s or "")).strip()
    return Hit(title=clean(title), url=clean(url), snippet=clean(snippet)[:500], published=clean(published), meta=clean(meta))


# --- DuckDuckGo-backed sources ------------------------------------------------
# `ddgs` is a synchronous library. Calling it directly inside async code would
# freeze the event loop (and every other scout with it), so each call is pushed
# onto a worker thread with `asyncio.to_thread`.

async def web(query: str, n: int = 8) -> list[Hit]:
    rows = await asyncio.to_thread(lambda: DDGS().text(query, max_results=n))
    return [_hit(r.get("title"), r.get("href"), r.get("body")) for r in rows]


async def news(query: str, n: int = 8) -> list[Hit]:
    rows = await asyncio.to_thread(lambda: DDGS().news(query, max_results=n))
    return [_hit(r.get("title"), r.get("url"), r.get("body"), r.get("date"), r.get("source")) for r in rows]


async def videos(query: str, n: int = 8) -> list[Hit]:
    try:
        rows = await asyncio.to_thread(lambda: DDGS().videos(query, max_results=n))
    except DDGSException:
        # The keyless video endpoint gets throttled after a burst of calls. Fall
        # back to ordinary web results restricted to YouTube: less metadata
        # (no views or duration), but the scout still gets real video pages.
        rows = await asyncio.to_thread(lambda: DDGS().text(f"{query} site:youtube.com", max_results=n))
        return [_hit(r.get("title"), r.get("href"), r.get("body"), meta="YouTube (web result)") for r in rows]
    out = []
    for r in rows:
        views = (r.get("statistics") or {}).get("viewCount")
        meta = " | ".join(x for x in [r.get("publisher"), r.get("uploader"), r.get("duration"), f"{views} views" if views else ""] if x)
        # For videos, ddgs puts the watch URL in `content`.
        out.append(_hit(r.get("title"), r.get("content"), r.get("description"), r.get("published"), meta))
    return out


async def reddit(query: str, n: int = 8) -> list[Hit]:
    # Reddit's own JSON API now blocks most anonymous traffic, so we ask the
    # search engine for Reddit pages instead. Thread pages can then be opened
    # with `read_page` for the actual discussion.
    rows = await asyncio.to_thread(lambda: DDGS().text(f"{query} site:reddit.com", max_results=n))
    return [_hit(r.get("title"), r.get("href"), r.get("body")) for r in rows if "reddit.com" in (r.get("href") or "")]


# --- HTTP JSON APIs -----------------------------------------------------------

async def hackernews(query: str, n: int = 8) -> list[Hit]:
    url = "https://hn.algolia.com/api/v1/search"
    async with httpx.AsyncClient(headers=HEADERS, timeout=TIMEOUT) as http:
        r = await http.get(url, params={"query": query, "tags": "story", "hitsPerPage": n})
        r.raise_for_status()
    out = []
    for h in r.json().get("hits", []):
        discussion = f"https://news.ycombinator.com/item?id={h['objectID']}"
        out.append(_hit(
            h.get("title"),
            discussion,
            # Link posts have no text; point at the article they link to instead.
            h.get("story_text") or f"Links to: {h.get('url') or 'n/a'}",
            h.get("created_at"),
            f"{h.get('points', 0)} points, {h.get('num_comments', 0)} comments",
        ))
    return out


async def wikipedia(query: str, n: int = 5) -> list[Hit]:
    url = "https://en.wikipedia.org/w/rest.php/v1/search/page"
    async with httpx.AsyncClient(headers=HEADERS, timeout=TIMEOUT) as http:
        r = await http.get(url, params={"q": query, "limit": n})
        r.raise_for_status()
    out = []
    for p in r.json().get("pages", []):
        snippet = re.sub(r"<[^>]+>", "", p.get("excerpt") or "")  # strip <span class="searchmatch">
        out.append(_hit(p.get("title"), f"https://en.wikipedia.org/wiki/{p['key']}", snippet, meta=p.get("description") or ""))
    return out


# --- Reading a page -----------------------------------------------------------

async def page_text(url: str, max_chars: int = 12000) -> str:
    """Fetch a URL and return readable text: tags, scripts and styles removed.

    Deliberately crude (regexes, not a full HTML parser): good enough for an
    agent to skim an article, and nothing to install.
    """
    async with httpx.AsyncClient(headers=HEADERS, timeout=TIMEOUT, follow_redirects=True) as http:
        r = await http.get(url)
        r.raise_for_status()
    html = r.text
    html = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]
