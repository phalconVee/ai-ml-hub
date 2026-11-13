"""Source 2: academic papers from the arXiv API. Free and keyless.

arXiv answers a plain HTTP GET with an Atom (XML) feed. We parse it with the
standard library, so this source has no dependencies and no key. Each paper's
abstract becomes one ContextItem: abstracts are dense, self-contained
summaries, which is exactly what a context window wants.

arXiv asks clients to keep to about one request every three seconds; one
search per question is well within that.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from ..config import SETTINGS
from ..items import ContextItem

API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"   # XML namespace prefix used by every tag


def build_url(query: str, n: int) -> str:
    # `all:` searches title, abstract and authors. Relevance sort, not date.
    params = {"search_query": f"all:{query}", "start": 0,
              "max_results": n, "sortBy": "relevance"}
    return f"{API}?{urllib.parse.urlencode(params)}"


def parse_feed(xml_text: str) -> list[ContextItem]:
    """Atom XML → ContextItems. Kept separate from the HTTP call so tests can
    feed it a saved response without touching the network."""
    root = ET.fromstring(xml_text)
    items = []
    for entry in root.findall(f"{ATOM}entry"):
        def get(tag: str) -> str:
            el = entry.find(f"{ATOM}{tag}")
            return " ".join((el.text or "").split()) if el is not None else ""  # squash newlines

        authors = [" ".join((a.findtext(f"{ATOM}name") or "").split())
                   for a in entry.findall(f"{ATOM}author")]
        items.append(ContextItem(
            source="paper",
            title=get("title"),
            text=get("summary"),        # the abstract
            url=get("id"),              # e.g. http://arxiv.org/abs/2401.01234v1
            meta={"authors": authors[:4], "published": get("published")[:10]},
        ))
    return items


def search_arxiv(query: str, n: int | None = None, timeout: float = 20.0) -> list[ContextItem]:
    url = build_url(query, n or SETTINGS.arxiv_results)
    req = urllib.request.Request(url, headers={"User-Agent": "context-forge/0.1 (research demo)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return parse_feed(resp.read().decode("utf-8"))
