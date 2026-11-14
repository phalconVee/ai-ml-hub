"""A keyless web MCP server: search the web, search Wikipedia, read a page.

Runs over either transport, which is the point of shipping it:

    uv run python -m example_servers.web_reader                 # stdio (the client launches it)
    uv run python -m example_servers.web_reader --http          # Streamable HTTP on 127.0.0.1:8765/mcp

With stdio the client owns the server's lifetime: it starts the process and
kills it on disconnect. With Streamable HTTP the server is a long-running web
service and any number of clients connect to it by URL. Same tools, same code;
only the last line of this file differs.

No API keys: web search uses DuckDuckGo through the `ddgs` package, Wikipedia
has a free JSON API, and page reading is plain HTTP + a small HTML-to-text pass.
"""

import argparse
import asyncio
import ipaddress
import os
import re
import socket
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx
from ddgs import DDGS
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

server = MCPServer(
    "web-reader",
    instructions=(
        "search_web for the open web, search_wikipedia for background facts, "
        "then read_page on the one or two best URLs. Cite the URLs you used."
    ),
)
# open_world_hint: these tools reach outside the machine. Still read-only.
READ_WORLD = ToolAnnotations(read_only_hint=True, open_world_hint=True)

# Wikipedia rejects (HTTP 403) user agents without a URL or e-mail to contact the
# operator. Set WEB_READER_CONTACT in .env to your own; the default points at the
# etiquette page so requests still go through.
CONTACT = os.environ.get("WEB_READER_CONTACT") or "contact not configured; https://www.mediawiki.org/wiki/API:Etiquette"
USER_AGENT = f"mcp-switchboard/0.1 (open-source MCP demo; {CONTACT})"
TIMEOUT = httpx.Timeout(15.0)


# --- HTML to text ------------------------------------------------------------------


class _TextExtractor(HTMLParser):
    """Collect visible text, skipping script/style/nav chrome. Crude but dependency-free."""

    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n")
        if tag in ("h1", "h2", "h3"):
            self.parts.append("## ")  # keep headings visible to the model

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)  # squeeze runs of spaces
        raw = re.sub(r"\n\s*\n+", "\n\n", raw)  # and runs of blank lines
        return raw.strip()


# --- safety: don't let the model probe your private network --------------------------


def _check_public(url: str) -> None:
    """Refuse non-http(s) URLs and hosts that resolve to private/loopback addresses.

    A page the model reads could contain "now fetch http://192.168.1.1/admin".
    A fetch tool that obeys is a way into your LAN (server-side request forgery).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ToolError("Only http(s) URLs are allowed.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ToolError(f"Can't resolve host {parsed.hostname!r}.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ToolError(f"Refusing to fetch {parsed.hostname!r}: it resolves to a non-public address.")


# --- tools -------------------------------------------------------------------------


@server.tool(annotations=READ_WORLD)
async def read_page(url: str, max_chars: int = 12000) -> str:
    """Fetch a web page and return its readable text (title first). Good after a search."""
    _check_public(url)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=True,
                                     headers={"User-Agent": USER_AGENT}) as http:
            resp = await http.get(url)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ToolError(f"Fetch failed: {type(exc).__name__}: {exc}") from exc
    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype:
        body = resp.text if ctype.startswith("text/") or "json" in ctype else f"(binary content: {ctype})"
        return body[:max_chars]
    parser = _TextExtractor()
    parser.feed(resp.text)
    text = parser.text()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return f"# {parser.title.strip() or url}\nURL: {resp.url}\n\n{text}"


@server.tool(annotations=READ_WORLD)
async def search_web(query: str, max_results: int = 6) -> str:
    """Keyless web search (DuckDuckGo). Returns title, URL and snippet per hit."""
    def _search() -> list[dict]:
        return DDGS().text(query, max_results=max_results)

    try:
        # ddgs is synchronous; run it in a thread so the server's event loop stays free.
        hits = await asyncio.to_thread(_search)
    except Exception as exc:  # ddgs raises its own exception types when rate-limited
        raise ToolError(f"Web search failed ({type(exc).__name__}: {exc}). "
                        "Try search_wikipedia or a different query.") from exc
    if not hits:
        return "No results. Try different words."
    return "\n\n".join(f"{h.get('title')}\n{h.get('href')}\n{h.get('body', '')}" for h in hits)


@server.tool(annotations=READ_WORLD)
async def search_wikipedia(query: str, max_results: int = 5, language: str = "en") -> str:
    """Search Wikipedia and return each article's title, URL and opening summary."""
    api = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query", "format": "json", "generator": "search", "gsrsearch": query,
        "gsrlimit": max_results, "prop": "extracts|info", "exintro": 1, "explaintext": 1,
        "exsentences": 3, "inprop": "url",
    }
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}) as http:
            resp = await http.get(api, params=params)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ToolError(f"Wikipedia request failed: {exc}") from exc
    pages = sorted(resp.json().get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    if not pages:
        return "No Wikipedia articles matched."
    return "\n\n".join(f"{p['title']}\n{p.get('fullurl', '')}\n{p.get('extract', '').strip()}" for p in pages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyless web MCP server")
    parser.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    cli = parser.parse_args()
    if cli.http:
        # Endpoint: http://127.0.0.1:8765/mcp. Bound to localhost only by default.
        server.run("streamable-http", host=cli.host, port=cli.port)
    else:
        server.run()
