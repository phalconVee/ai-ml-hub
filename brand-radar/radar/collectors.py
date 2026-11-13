"""Collectors: fetch raw brand mentions from each source. No AI in this file.

Every collector has the same signature,

    async def collect_x(q: BrandQuery, n: int) -> list[Mention]

so the pipeline can run them all side by side and never needs to know that
Bluesky calls a post's body `record.text` while Hacker News calls it
`comment_text`. All sources are free and need no API key.

Recency: each collector asks its source for the last `q.days` days where the
source supports a time filter (DuckDuckGo only knows day / week / month / year,
so we round up to the nearest bucket).
"""

import asyncio
import html
import re
from datetime import datetime, timedelta, timezone

import httpx
from ddgs import DDGS
from ddgs.exceptions import DDGSException

from radar.schemas import BrandQuery, Mention

# A polite client: say who we are and never hang forever.
HEADERS = {"User-Agent": "brand-radar/0.1 (portfolio demo)"}
TIMEOUT = httpx.Timeout(15.0)


def clean(s: object, limit: int = 2000) -> str:
    """Strip HTML tags and squash whitespace. Snippets arrive in many shapes."""
    s = re.sub(r"(?s)<[^>]+>", " ", str(s or ""))
    s = html.unescape(s)              # &#x27; -> ', &gt; -> >, &amp; -> &
    return re.sub(r"\s+", " ", s).strip()[:limit]


def search_phrase(q: BrandQuery) -> str:
    """The query string sent to search engines.

    The brand is quoted so the engine treats it as an exact word. We do NOT
    append the context ("notes and docs app"): search engines AND those words
    together and return almost nothing. Context is used later, by the analysts,
    to throw out namesakes.
    """
    return f'"{q.brand}"'


def ddg_timelimit(days: int) -> str:
    """DuckDuckGo's time filter only has buckets: d, w, m, y. Round up."""
    return "d" if days <= 1 else "w" if days <= 7 else "m" if days <= 31 else "y"


def since(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


# --- DuckDuckGo-backed sources --------------------------------------------------
# `ddgs` is synchronous. Calling it directly would freeze the event loop (and
# every other collector with it), so each call runs on a worker thread.
# It also raises DDGSException for "no results", which for us just means [].

async def _ddg(kind: str, query: str, n: int, timelimit: str) -> list[dict]:
    def call() -> list[dict]:
        try:
            return getattr(DDGS(), kind)(query, max_results=n, timelimit=timelimit)
        except DDGSException:
            return []
    return await asyncio.to_thread(call)


async def collect_news(q: BrandQuery, n: int) -> list[Mention]:
    rows = await _ddg("news", search_phrase(q), n, ddg_timelimit(q.days))
    return [Mention("news", clean(r.get("title")), r.get("url", ""), clean(r.get("body")),
                    author=clean(r.get("source")), published=r.get("date", "")) for r in rows]


async def collect_web(q: BrandQuery, n: int) -> list[Mention]:
    rows = await _ddg("text", search_phrase(q), n, ddg_timelimit(q.days))
    return [Mention("web", clean(r.get("title")), r.get("href", ""), clean(r.get("body"))) for r in rows]


async def collect_reddit(q: BrandQuery, n: int) -> list[Mention]:
    # Reddit's own JSON API blocks anonymous clients (HTTP 403), so we ask the
    # search engine for reddit.com pages instead. Week-old Reddit pages are
    # often not indexed yet, so this source looks back at least a month.
    rows = await _ddg("text", f"{search_phrase(q)} site:reddit.com", n, ddg_timelimit(max(q.days, 30)))
    return [Mention("reddit", clean(r.get("title")), r.get("href", ""), clean(r.get("body")))
            for r in rows if "reddit.com" in (r.get("href") or "")]


async def collect_youtube(q: BrandQuery, n: int) -> list[Mention]:
    rows = await _ddg("videos", search_phrase(q), n, ddg_timelimit(q.days))
    if not rows:
        # The keyless video endpoint is often throttled and returns nothing.
        # Fall back to ordinary web results restricted to YouTube: no view
        # counts, but still real, recent video pages with descriptions.
        rows = await _ddg("text", f"{search_phrase(q)} site:youtube.com", n, ddg_timelimit(q.days))
        return [Mention("youtube", clean(r.get("title")), r.get("href", ""), clean(r.get("body")))
                for r in rows if "youtube.com" in (r.get("href") or "")]
    out = []
    for r in rows:
        views = (r.get("statistics") or {}).get("viewCount") or 0
        out.append(Mention("youtube", clean(r.get("title")), r.get("content", ""),  # ddgs puts the watch URL in `content`
                           clean(r.get("description")), author=clean(r.get("uploader")),
                           published=r.get("published", ""), engagement=int(views)))
    return out


# --- JSON APIs ------------------------------------------------------------------

async def collect_hackernews(q: BrandQuery, n: int) -> list[Mention]:
    """Stories AND comments: complaints usually live in the comments."""
    url = "https://hn.algolia.com/api/v1/search_by_date"
    params = {
        "query": q.brand,
        "tags": "(story,comment)",          # either kind
        "numericFilters": f"created_at_i>{int(since(q.days).timestamp())}",
        "hitsPerPage": n,
        # Algolia forgives typos by default, so "notion" also matched
        # "national". Brand names need exact matching.
        "typoTolerance": "false",
    }
    async with httpx.AsyncClient(headers=HEADERS, timeout=TIMEOUT) as http:
        r = await http.get(url, params=params)
        r.raise_for_status()
    out = []
    for h in r.json().get("hits", []):
        is_comment = bool(h.get("comment_text"))
        out.append(Mention(
            "hackernews",
            clean(h.get("title") or h.get("story_title") or "HN comment"),
            f"https://news.ycombinator.com/item?id={h['objectID']}",
            clean(h.get("comment_text") or h.get("story_text") or h.get("title")),
            author=h.get("author", ""),
            published=h.get("created_at", ""),
            # Comments have no points in the search API; stories do.
            engagement=0 if is_comment else int(h.get("points") or 0) + int(h.get("num_comments") or 0),
        ))
    return out


async def collect_bluesky(q: BrandQuery, n: int) -> list[Mention]:
    """Bluesky's public AppView allows anonymous search (api.bsky.app).

    Note: the `public.api.bsky.app` host answers 403 to search; this one works.
    """
    url = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
    params = {"q": q.brand, "sort": "latest", "limit": min(n, 100),
              "since": since(q.days).strftime("%Y-%m-%dT%H:%M:%SZ")}
    async with httpx.AsyncClient(headers=HEADERS, timeout=TIMEOUT) as http:
        r = await http.get(url, params=params)
        r.raise_for_status()
    out = []
    for p in r.json().get("posts", []):
        handle = p.get("author", {}).get("handle", "")
        post_id = p.get("uri", "").rsplit("/", 1)[-1]           # at://did/.../post/<rkey>
        text = clean(p.get("record", {}).get("text"))
        out.append(Mention(
            "bluesky", text[:80], f"https://bsky.app/profile/{handle}/post/{post_id}", text,
            author=handle, published=p.get("record", {}).get("createdAt", ""),
            engagement=sum(int(p.get(k) or 0) for k in ("likeCount", "repostCount", "replyCount", "quoteCount")),
        ))
    return out


async def collect_mastodon(q: BrandQuery, n: int) -> list[Mention]:
    """Mastodon full-text search needs a login, but hashtag timelines are public.

    So this source sees posts tagged #brand (e.g. #notion), which is narrower
    than all mentions but a real, keyless social signal.
    """
    tag = re.sub(r"[^0-9a-zA-Z_]", "", q.brand)
    if not tag:
        return []
    url = f"https://mastodon.social/api/v1/timelines/tag/{tag}"
    async with httpx.AsyncClient(headers=HEADERS, timeout=TIMEOUT) as http:
        r = await http.get(url, params={"limit": min(n, 40)})
        r.raise_for_status()
    cutoff = since(q.days)
    out = []
    for s in r.json():
        created = datetime.fromisoformat(s["created_at"].replace("Z", "+00:00"))
        if created < cutoff:
            continue                       # the timeline has no date filter: do it here
        text = clean(s.get("content"))
        out.append(Mention(
            "mastodon", text[:80], s.get("url") or s.get("uri", ""), text,
            author=s.get("account", {}).get("acct", ""), published=s["created_at"],
            engagement=sum(int(s.get(k) or 0) for k in ("favourites_count", "reblogs_count", "replies_count")),
        ))
    return out


# Name -> function, so the pipeline can loop over whatever the user ticked.
COLLECTORS = {
    "news": collect_news,
    "web": collect_web,
    "reddit": collect_reddit,
    "hackernews": collect_hackernews,
    "bluesky": collect_bluesky,
    "mastodon": collect_mastodon,
    "youtube": collect_youtube,
}


async def collect_all(q: BrandQuery, sources: list[str], n: int) -> dict[str, list[Mention] | Exception]:
    """Run the chosen collectors concurrently. A failing source returns its
    exception instead of sinking the whole run (`return_exceptions=True`)."""
    results = await asyncio.gather(*(COLLECTORS[s](q, n) for s in sources), return_exceptions=True)
    return dict(zip(sources, results))
