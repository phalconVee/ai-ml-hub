"""Every knob in one place.

Read this file first: it names the platforms the crew can search, which MCP tools
belong to each platform, and which Claude model and effort level each agent uses.
Anything here can be overridden with an environment variable (see `.env.example`).
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Pull variables from a local `.env` file into `os.environ`, if one exists.
# Real environment variables win over the file (override=False is the default).
load_dotenv()


# --- Platforms -------------------------------------------------------------
# A "platform" is one corner of the internet a scout agent is responsible for.
# Each maps to the MCP tools that scout is allowed to use. Every scout also gets
# `read_page`, so it can open a promising link and read the full text.
#
# The planner may only pick platforms from this dict, so adding a platform is:
#   1. add a tool to `scout/mcp_server.py`
#   2. add a line here
PLATFORM_TOOLS: dict[str, list[str]] = {
    "web": ["search_web"],               # general search engine results
    "news": ["search_news"],             # dated news articles
    "video": ["search_videos"],          # YouTube & co. (titles, descriptions, uploader)
    "reddit": ["search_reddit"],         # community discussion threads
    "hackernews": ["search_hackernews"], # tech community stories + points/comments
    "wikipedia": ["search_wikipedia"],   # encyclopedic background
}
SHARED_TOOLS: list[str] = ["read_page"]

# One-line descriptions shown to the planner so it can pick platforms sensibly.
PLATFORM_HINTS: dict[str, str] = {
    "web": "official sites, docs, blogs, vendor pages",
    "news": "recent events, announcements, press coverage with dates",
    "video": "talks, tutorials, demos, reviews (YouTube etc.)",
    "reddit": "practitioner opinions, complaints, real-world experience",
    "hackernews": "technical community reaction, launches, critiques",
    "wikipedia": "definitions, history, settled background facts",
}


def _env(name: str, default: str) -> str:
    """Read an env var, falling back to `default` when unset or blank."""
    return os.environ.get(name) or default


@dataclass
class Settings:
    # --- Models ---------------------------------------------------------
    # Default to Claude Opus 5.5 for every role. Scouts do lots of small,
    # mechanical tool calls, so they run at low effort; the synthesizer does
    # the hard thinking (cross-checking sources), so it runs at high effort.
    planner_model: str = field(default_factory=lambda: _env("OMNI_SCOUT_PLANNER_MODEL", "claude-opus-5-5"))
    scout_model: str = field(default_factory=lambda: _env("OMNI_SCOUT_SCOUT_MODEL", "claude-opus-5-5"))
    synth_model: str = field(default_factory=lambda: _env("OMNI_SCOUT_SYNTH_MODEL", "claude-opus-5-5"))

    planner_effort: str = field(default_factory=lambda: _env("OMNI_SCOUT_PLANNER_EFFORT", "medium"))
    scout_effort: str = field(default_factory=lambda: _env("OMNI_SCOUT_SCOUT_EFFORT", "low"))
    synth_effort: str = field(default_factory=lambda: _env("OMNI_SCOUT_SYNTH_EFFORT", "high"))

    # --- Budgets --------------------------------------------------------
    # Hard cap on think -> call tools -> read results rounds per scout.
    # Stops a confused scout from looping forever (and spending forever).
    max_scout_turns: int = field(default_factory=lambda: int(_env("OMNI_SCOUT_MAX_TURNS", "6")))
    # How many platforms the planner may assign at most.
    max_platforms: int = field(default_factory=lambda: int(_env("OMNI_SCOUT_MAX_PLATFORMS", "4")))
    # Tool results are truncated to this many characters before going back
    # to the model: a full web page can be 100k+ chars of noise.
    max_tool_chars: int = field(default_factory=lambda: int(_env("OMNI_SCOUT_MAX_TOOL_CHARS", "6000")))

    # --- Server-side refusal fallback ----------------------------------
    # If a safety classifier declines a request, the API re-runs it on a
    # fallback model inside the same call. "default" lets the API choose.
    fallback_beta: str = "server-side-fallback-2026-07-01"


settings = Settings()
