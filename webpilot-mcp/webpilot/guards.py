"""Guard rail #1: which URLs the browser may load. Pure Python, no browser.

The rule, in order:
  1. Only http(s) pages (plus about:blank). No file://, javascript:, chrome://...
  2. Deny list wins: a host on it is always blocked.
  3. If an allow list is set, the host must be on it. An empty allow list
     means "anything not denied".

"On the list" means the host IS the domain or is a subdomain of it, so
'wikipedia.org' covers 'en.wikipedia.org' but NOT 'notwikipedia.org'.

This file is used in two places (defence in depth):
  * the MCP server checks every *top-level navigation* the browser makes,
    including redirects and link clicks, via Playwright request routing;
  * the same check runs before `goto` so the model gets a clear message.
"""

from dataclasses import dataclass
from urllib.parse import urlsplit

ALWAYS_OK = {"about:blank"}


def host_matches(host: str, domain: str) -> bool:
    """True if `host` is `domain` or one of its subdomains."""
    host, domain = host.lower().rstrip("."), domain.lower().rstrip(".")
    return host == domain or host.endswith("." + domain)


@dataclass(frozen=True)
class DomainPolicy:
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()

    def check(self, url: str) -> str | None:
        """Return None if `url` may be loaded, else a human-readable reason."""
        if url in ALWAYS_OK:
            return None
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"}:
            return f"scheme '{parts.scheme}:' is not allowed (only http/https)"
        host = parts.hostname or ""
        if not host:
            return "URL has no host"
        for d in self.deny:
            if host_matches(host, d):
                return f"'{host}' is on the deny list ({d})"
        if self.allow and not any(host_matches(host, d) for d in self.allow):
            return f"'{host}' is not on the allow list ({', '.join(self.allow)})"
        return None

    def describe(self) -> str:
        """One line for the system prompt, so Claude doesn't waste steps on blocked sites."""
        allow = ", ".join(self.allow) if self.allow else "any domain"
        deny = ", ".join(self.deny) if self.deny else "none"
        return f"Allowed domains: {allow}. Denied domains: {deny}."
