"""Step 1: turn whatever the user adds into a `Source` - plain text, split by page.

Three loaders, one output shape:

    PDF bytes      -> one "page" per PDF page          (PyMuPDF)
    text/markdown  -> one "page" for the whole file
    web URL        -> one "page" of readable article text (stdlib HTML parser)

Keeping pages separate matters later: a passage never crosses a page, so every
citation can say "page 4" and mean it.

No paid scraping service is used. Web pages are fetched with the standard
library and cleaned with a ~60-line HTML-to-text parser. That handles articles,
docs and blogs well; it can't run JavaScript, so app-like pages that render
their text in the browser come back nearly empty (we raise a clear error).
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Callable


@dataclass
class Source:
    id: str                 # short stable hash of the content, e.g. "a1b2c3d4"
    title: str              # what the UI shows: file name or page <title>
    kind: str               # "pdf" | "text" | "web"
    origin: str             # file name or URL
    pages: list[str] = field(default_factory=list)   # pages[0] is page 1
    enabled: bool = True    # the UI checkbox: include this source in answers?

    @property
    def url(self) -> str | None:
        return self.origin if self.kind == "web" else None

    @property
    def word_count(self) -> int:
        return sum(len(p.split()) for p in self.pages)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "Source":
        return cls(**d)


def _make_id(*parts: str) -> str:
    """Same content -> same id, so adding a file twice is detected."""
    return hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:8]


def _tidy(text: str) -> str:
    """Normalise whitespace but keep paragraph breaks (blank lines)."""
    text = text.replace("\r\n", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)          # runs of spaces -> one
    text = re.sub(r" *\n *", "\n", text)          # trim around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)        # at most one blank line
    return text.strip()


# --- PDF --------------------------------------------------------------------

def load_pdf(data: bytes, name: str) -> Source:
    import pymupdf  # imported here so text-only use doesn't need it loaded

    pages = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            # sort=True reads blocks top-to-bottom, left-to-right, which fixes
            # most multi-column and header/footer ordering surprises.
            pages.append(_tidy(page.get_text("text", sort=True)))
    if not any(pages):
        raise ValueError(f"{name}: no text layer found (a scanned PDF needs OCR first).")
    return Source(_make_id(name, *pages), name, "pdf", name, pages)


# --- Plain text / Markdown ----------------------------------------------------

def load_text(text: str, name: str) -> Source:
    text = _tidy(text)
    if not text:
        raise ValueError(f"{name}: the file is empty.")
    return Source(_make_id(name, text), name, "text", name, [text])


# --- Web pages ----------------------------------------------------------------

class _ReadableText(HTMLParser):
    """Collect the visible, readable text of an HTML page.

    Rules of thumb that get most articles right:
    - skip non-content elements (scripts, nav bars, footers, forms...);
    - start a new line at block elements so paragraphs stay paragraphs;
    - if the page has <article> or <main>, keep only the text inside it.
    """

    SKIP = {"script", "style", "noscript", "svg", "nav", "header", "footer",
            "aside", "form", "button", "iframe", "template"}
    BLOCK = {"p", "div", "section", "article", "main", "li", "ul", "ol", "br",
             "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "blockquote",
             "pre", "figcaption", "dd", "dt"}
    MAIN = {"article", "main"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._skip_depth = 0     # >0 while inside a SKIP element
        self._main_depth = 0     # >0 while inside <article>/<main>
        self.all_text: list[str] = []
        self.main_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self._skip_depth += 1
        if tag in self.MAIN:
            self._main_depth += 1
        if tag in self.BLOCK:
            self._emit("\n\n" if tag in {"p", "h1", "h2", "h3", "h4", "li", "pre"} else "\n")
        # Keep headings recognisable as Markdown, which also helps chunking.
        if tag in {"h1", "h2", "h3"}:
            self._emit("#" * int(tag[1]) + " ")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag in self.MAIN and self._main_depth:
            self._main_depth -= 1
        if tag in self.BLOCK:
            self._emit("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        else:
            self._emit(data)

    def _emit(self, s: str) -> None:
        if self._skip_depth:
            return
        self.all_text.append(s)
        if self._main_depth:
            self.main_text.append(s)

    def result(self) -> tuple[str, str]:
        main = _tidy("".join(self.main_text))
        body = main if len(main.split()) >= 50 else _tidy("".join(self.all_text))
        return _tidy(self.title), body


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, readable text) for an HTML document."""
    parser = _ReadableText()
    parser.feed(html)
    parser.close()
    return parser.result()


def _http_get(url: str) -> tuple[str, str]:
    """Fetch a URL; returns (content-type, decoded body)."""
    req = urllib.request.Request(url, headers={"User-Agent": "sourcebook/0.1 (research notebook)"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        ctype = resp.headers.get("Content-Type", "")
        charset = resp.headers.get_content_charset() or "utf-8"
        return ctype, resp.read().decode(charset, errors="replace")


def load_url(url: str, fetch: Callable[[str], tuple[str, str]] = _http_get) -> Source:
    """Fetch a web page and keep its readable text. `fetch` is swappable for tests."""
    if not re.match(r"^https?://", url):
        raise ValueError("URL must start with http:// or https://")
    ctype, body = fetch(url)
    if "html" in ctype or body.lstrip().startswith("<"):
        title, text = html_to_text(body)
    else:  # plain text, markdown, etc.
        title, text = "", _tidy(body)
    if len(text.split()) < 30:
        raise ValueError(f"{url}: almost no readable text (the page may need JavaScript).")
    return Source(_make_id(url, text), title or url, "web", url, [text])
