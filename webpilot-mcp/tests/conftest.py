"""Shared fixtures: a local static website, and settings that point at it.

The site (tests/site/) is served on 127.0.0.1 by Python's built-in HTTP
server, so the browser tests run real Chromium against real HTTP, offline.
"""

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from webpilot.config import Settings

SITE = Path(__file__).parent / "site"


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture(scope="session")
def site_url():
    handler = functools.partial(_QuietHandler, directory=str(SITE))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)  # port 0 = any free port
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def make_settings(**overrides) -> Settings:
    """Safe defaults for tests: headless, and 'localhost' denied so the test
    page's offsite link (same server, other hostname) counts as a blocked site."""
    base = dict(read_only=True, headless=True, allow=(), deny=("localhost",), max_steps=6)
    base.update(overrides)
    s = Settings()
    for k, v in base.items():
        setattr(s, k, v)
    return s
