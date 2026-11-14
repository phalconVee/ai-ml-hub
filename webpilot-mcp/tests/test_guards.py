"""Domain policy: pure logic, no browser."""

from webpilot.config import parse_domains
from webpilot.guards import DomainPolicy


def test_empty_policy_allows_web_but_not_other_schemes():
    p = DomainPolicy()
    assert p.check("https://news.ycombinator.com/") is None
    assert p.check("about:blank") is None
    assert "scheme" in p.check("file:///etc/passwd")
    assert "scheme" in p.check("javascript:alert(1)")


def test_allow_list_covers_subdomains_but_not_lookalikes():
    p = DomainPolicy(allow=("wikipedia.org",))
    assert p.check("https://en.wikipedia.org/wiki/Rust") is None
    assert p.check("https://wikipedia.org/") is None
    assert "allow list" in p.check("https://notwikipedia.org/")
    assert "allow list" in p.check("https://wikipedia.org.evil.com/")


def test_deny_beats_allow():
    p = DomainPolicy(allow=("example.com",), deny=("admin.example.com",))
    assert p.check("https://www.example.com/") is None
    assert "deny list" in p.check("https://admin.example.com/login")


def test_parse_domains_is_forgiving():
    assert parse_domains(" Example.com, *.wikipedia.org\n.news.ycombinator.com ") == (
        "example.com", "wikipedia.org", "news.ycombinator.com",
    )
