"""Step 0, plain code: turn "a URL or some pasted text" into clean article text.

No LLM and no scraping API here. A web page is fetched with the standard
library and its readable text is pulled out with a ~40-line HTML parser:
keep headings, paragraphs and list items; drop scripts, styles, menus,
headers and footers. That's crude next to a full "reader mode", but it is
free, keyless, deterministic, and good enough for articles and blog posts,
which is what people turn into podcasts.

Why not let Claude fetch the page (the web_fetch server tool)? It would
work, but doing it in code means the research agent always sees exactly the
text we show the user, the step is testable offline, and a page that fails
to load fails *before* we spend a single token.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser

from .config import SETTINGS


@dataclass
class Source:
    title: str
    text: str                 # the readable body, paragraphs separated by blank lines
    url: str | None = None    # None when the user pasted text
    truncated: bool = False   # True when we had to cut it to max_source_chars


class SourceError(ValueError):
    """The input can't become an episode (bad URL, empty page, too short)."""


def looks_like_url(s: str) -> bool:
    """One line, starts with http(s)://, no spaces: treat it as a link."""
    s = s.strip()
    return bool(re.fullmatch(r"https?://\S+", s))


# ── HTML → text ─────────────────────────────────────────────────────────────

# Tags whose contents we never want: code, styling, navigation chrome.
SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form",
        "svg", "button", "iframe", "template"}
# Tags that hold the readable text. Each one becomes its own paragraph.
BLOCKS = {"p", "h1", "h2", "h3", "h4", "li", "blockquote", "pre", "figcaption", "td"}


class _Extractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)  # turns &amp; etc. into characters
        self.skip_depth = 0      # >0 while we're inside a SKIP tag (they can nest)
        self.block_depth = 0     # >0 while we're inside a BLOCKS tag
        self.in_title = False
        self.title = ""
        self.buf: list[str] = []         # text of the block being read
        self.paragraphs: list[str] = []  # finished blocks

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip_depth += 1
        elif tag == "title":
            self.in_title = True
        elif tag in BLOCKS:
            self.block_depth += 1
        elif tag == "br":
            self.buf.append(" ")

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag == "title":
            self.in_title = False
        elif tag in BLOCKS and self.block_depth:
            self.block_depth -= 1
            if self.block_depth == 0:     # outermost block closed → flush a paragraph
                self._flush()

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif self.skip_depth == 0 and self.block_depth > 0:
            self.buf.append(data)

    def _flush(self):
        text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        self.buf = []
        if len(text.split()) >= 4:        # skip "Share", "Read more", lone dates…
            self.paragraphs.append(text)


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, body text) from an HTML page."""
    p = _Extractor()
    p.feed(html)
    p.close()
    # De-duplicate repeated blocks (cookie notices, repeated pull quotes)
    # while keeping the original order.
    seen, paras = set(), []
    for para in p.paragraphs:
        if para not in seen:
            seen.add(para)
            paras.append(para)
    return re.sub(r"\s+", " ", p.title).strip(), "\n\n".join(paras)


def fetch_html(url: str) -> str:
    """Download a page. A browser-like User-Agent avoids many naive bot blocks."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (podcast-crew; +local study project)",
        "Accept": "text/html,application/xhtml+xml",
    })
    with urllib.request.urlopen(req, timeout=SETTINGS.fetch_timeout_s) as resp:
        ctype = resp.headers.get("Content-Type", "")
        if "html" not in ctype and "text" not in ctype:
            raise SourceError(f"That link is not a web page ({ctype or 'unknown type'}).")
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


# ── The one function the crew calls ─────────────────────────────────────────

def load_source(user_input: str, *, fetch=None) -> Source:
    """URL → fetched + extracted page. Anything else → the pasted text itself.

    `fetch` is a parameter so tests can hand in a fake page without a network.
    It's looked up at call time (not as a default argument, which Python
    would freeze at import), so patching `fetch_html` also works.
    """
    fetch = fetch or fetch_html
    raw = user_input.strip()
    if not raw:
        raise SourceError("Paste a link or some text first.")

    if looks_like_url(raw):
        try:
            title, text = html_to_text(fetch(raw))
        except SourceError:
            raise
        except Exception as e:  # DNS, 404, timeout, TLS… one friendly message
            raise SourceError(f"Couldn't load that page: {e}") from e
        url = raw
    else:
        # Pasted text: first short line doubles as a title, if it looks like one.
        first = raw.splitlines()[0].strip()
        title = first if len(first) <= 120 and len(raw.splitlines()) > 1 else ""
        text, url = raw, None

    if len(text) < SETTINGS.min_source_chars:
        raise SourceError("There's too little text there to talk about "
                          f"(need at least {SETTINGS.min_source_chars} characters). "
                          "If it's a JavaScript-heavy page, paste the article text instead.")

    truncated = len(text) > SETTINGS.max_source_chars
    if truncated:
        text = text[: SETTINGS.max_source_chars]
    return Source(title=title or "Untitled", text=text, url=url, truncated=truncated)
