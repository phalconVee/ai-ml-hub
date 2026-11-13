"""Every knob in one place. Read this file first.

It answers three questions:
  1. Which sources can the radar sweep? (SOURCES)
  2. Which Claude model and effort does each agent use?
  3. How much does one run fetch and spend? (limits)

Anything here can be overridden with an environment variable (see `.env.example`).
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Pull variables from a local `.env` into os.environ, if the file exists.
# Real environment variables win over the file (override=False is the default).
load_dotenv()


# --- Sources -----------------------------------------------------------------
# Every source is FREE and KEYLESS. The value is a short label for the UI and
# the analyst prompt, so the analyst knows what kind of text it is reading
# (a news headline reads very differently from a forum comment).
SOURCES: dict[str, str] = {
    "news": "News articles (DuckDuckGo News)",
    "web": "Web pages and blogs (DuckDuckGo)",
    "reddit": "Reddit threads (found via DuckDuckGo)",
    "hackernews": "Hacker News stories and comments (Algolia API)",
    "bluesky": "Bluesky posts (public AppView search)",
    "mastodon": "Mastodon posts under the brand's hashtag (mastodon.social)",
    "youtube": "YouTube videos (DuckDuckGo Videos)",
}

# A seventh, optional source: Claude's own server-side web search. It costs
# a little per search (billed on the Anthropic key), so it is off by default.
CLAUDE_SWEEP = "claude_sweep"


def _env(name: str, default: str) -> str:
    """Read an env var, falling back to `default` when unset or blank."""
    return os.environ.get(name) or default


@dataclass
class Settings:
    # --- Models ---------------------------------------------------------
    # Analysts do many small, similar classification jobs (one per source),
    # so they use Sonnet at low effort: fast, cheap, plenty for "is this
    # positive or negative?". The strategist reads everything and writes the
    # recommendations, which is where judgment matters: Opus at high effort.
    analyst_model: str = field(default_factory=lambda: _env("BRAND_RADAR_ANALYST_MODEL", "claude-sonnet-5-5"))
    analyst_effort: str = field(default_factory=lambda: _env("BRAND_RADAR_ANALYST_EFFORT", "low"))
    strategist_model: str = field(default_factory=lambda: _env("BRAND_RADAR_STRATEGIST_MODEL", "claude-opus-5-5"))
    strategist_effort: str = field(default_factory=lambda: _env("BRAND_RADAR_STRATEGIST_EFFORT", "high"))
    sweep_model: str = field(default_factory=lambda: _env("BRAND_RADAR_SWEEP_MODEL", "claude-sonnet-5-5"))

    # --- Limits ---------------------------------------------------------
    # Raw results asked from each source. More = better coverage, more tokens.
    per_source: int = field(default_factory=lambda: int(_env("BRAND_RADAR_PER_SOURCE", "15")))
    # Mentions per analyst request. Each batch is one Claude call; batches of
    # one source run in parallel.
    batch_size: int = field(default_factory=lambda: int(_env("BRAND_RADAR_BATCH_SIZE", "15")))
    # At most this many Claude calls in flight at once (be kind to rate limits).
    max_parallel_llm: int = field(default_factory=lambda: int(_env("BRAND_RADAR_MAX_PARALLEL", "4")))
    # Server-side web searches the optional Claude sweep may run.
    sweep_max_searches: int = field(default_factory=lambda: int(_env("BRAND_RADAR_SWEEP_SEARCHES", "5")))
    # Text of each mention is clipped to this many characters before analysis.
    max_text_chars: int = 600

    # --- Server-side refusal fallback ----------------------------------
    # If a safety classifier declines a request, the API retries it on a
    # fallback model within the same call instead of just stopping.
    fallback_beta: str = "server-side-fallback-2026-07-01"


settings = Settings()
