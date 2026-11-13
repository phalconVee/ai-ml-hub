"""The source ledger: the only facts the writer is allowed to use.

The reporter's response is a pile of blocks. This module turns it into a
numbered list the writer and editor can point at:

    [S1] Reuters: "Central bank holds rates"  (https://…, 2 days ago)
         • "The bank kept its key rate at 4.25%, citing…"
         • "Three of nine members voted for a cut."
    [S2] …

Where each piece comes from:
  web_search_tool_result  → a source exists (title, url, page age)
  web_fetch_tool_result   → the reporter actually opened it (marked "read")
  citations on the notes  → *quotes*: exact text the API says the notes rest on

Built by plain code, not by an LLM. The model cannot add a source the API
never returned, and every quote is text the API itself extracted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .config import SETTINGS


@dataclass
class Source:
    id: str = ""                  # "S1", "S2"… assigned after ranking
    url: str = ""
    title: str = ""
    age: str = ""                 # "3 days ago" etc., from web search, when known
    read: bool = False            # True if the reporter fetched the full page
    quotes: list[str] = field(default_factory=list)  # cited text from this page

    @property
    def outlet(self) -> str:
        """'www.reuters.com' → 'reuters.com' — a short label for the UI."""
        host = urlsplit(self.url).netloc
        return host.removeprefix("www.")


def normalize(url: str) -> str:
    """Same page, different spelling → same key. Drops #fragments and a trailing /."""
    return url.split("#")[0].rstrip("/")


def build_ledger(blocks: list) -> list[Source]:
    by_url: dict[str, Source] = {}  # dict keeps first-seen order

    def get(url: str, title: str = "") -> Source:
        key = normalize(url)
        if key not in by_url:
            by_url[key] = Source(url=url, title=title or url)
        elif title and by_url[key].title == by_url[key].url:
            by_url[key].title = title  # upgrade a bare url to a real title
        return by_url[key]

    for b in blocks:
        kind = getattr(b, "type", "")
        if kind == "web_search_tool_result":
            # Success → a list of results. Error → one object. Check first.
            if isinstance(b.content, list):
                for r in b.content:
                    s = get(r.url, r.title)
                    s.age = s.age or (getattr(r, "page_age", None) or "")
        elif kind == "web_fetch_tool_result":
            # Success → type "web_fetch_result" with the page as a document.
            if getattr(b.content, "type", "") == "web_fetch_result":
                doc_title = getattr(b.content.content, "title", None) or ""
                get(b.content.url, doc_title).read = True
        elif kind == "text":
            for c in getattr(b, "citations", None) or []:
                url = getattr(c, "url", None)
                if not url:
                    continue  # only web citations carry a url
                s = get(url, getattr(c, "title", None) or "")
                quote = " ".join(c.cited_text.split())  # squash whitespace
                if quote and quote not in s.quotes:
                    s.quotes.append(quote)

    sources = list(by_url.values())
    # Rank: quoted sources first (they back actual notes), then pages the
    # reporter read, then the rest. sort() is stable, so ties keep search order.
    sources.sort(key=lambda s: (not s.quotes, not s.read))
    sources = sources[: SETTINGS.max_sources]
    for i, s in enumerate(sources, start=1):
        s.id = f"S{i}"
        s.quotes = s.quotes[: SETTINGS.max_quotes_per_source]
    return sources


def render_ledger(sources: list[Source]) -> str:
    """The ledger as the writer and editor see it."""
    if not sources:
        return "(no sources were found)"
    out = []
    for s in sources:
        meta = ", ".join(x for x in (s.outlet, s.age, "read in full" if s.read else "") if x)
        out.append(f"[{s.id}] {s.title} ({meta})\n    {s.url}")
        out.extend(f'    • "{q}"' for q in s.quotes)
    return "\n".join(out)
