"""Every knob in one place.

Read this file first: it tells you what the crew is made of (which model,
how hard each agent thinks, how many research angles, how many tool calls
each agent may spend) before you see any of the machinery.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Load a local .env file if there is one, so `ANTHROPIC_API_KEY` etc. can live
# next to the code instead of in your shell profile. Missing file = no-op.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at runtime
    pass


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
# One model for the whole crew. Different agents get different *effort*
# levels instead of different models: lower effort = fewer, more consolidated
# tool calls and shorter thinking; higher effort = more careful reasoning.
# Override with DEEPDIVE_MODEL if you want to try another Claude model.
MODEL = os.getenv("DEEPDIVE_MODEL", "claude-opus-5-5")

# Server-side refusal fallback. If a safety classifier declines a request,
# the API re-runs it on Anthropic's recommended fallback model inside the
# same call instead of handing us an empty "refusal" response.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


# ---------------------------------------------------------------------------
# Search backend used by the MCP tools server
# ---------------------------------------------------------------------------
#   "duckduckgo" - free, no key (default). Good enough to learn with.
#   "tavily"     - better results for research; needs TAVILY_API_KEY.
#   "fake"       - canned offline results, used by the tests and dry runs.
SEARCH_BACKEND = os.getenv("DEEPDIVE_SEARCH_BACKEND", "duckduckgo")


@dataclass
class CrewSettings:
    """How big and how careful one research run is."""

    model: str = MODEL

    # How many independent research angles the planner splits a question into.
    # Each angle gets its own scout, and scouts run in parallel.
    num_angles: int = 3

    # How many scouts may run at the same time (caps API + search rate).
    max_parallel_scouts: int = 3

    # Tool-use budget: the most model turns an agent may spend calling tools
    # before it is told to stop and write up what it has.
    scout_max_turns: int = 8
    analyst_max_turns: int = 4

    # Effort per role ("low" | "medium" | "high" | "xhigh" | "max").
    # Planner and writer shape the whole report, so they think harder.
    # Scouts do many cheap search/read cycles, so they stay light.
    effort: dict[str, str] = field(
        default_factory=lambda: {
            "planner": "high",
            "scout": "low",
            "analyst": "medium",
            "writer": "high",
        }
    )

    # Per-response output cap. 16k is safe for non-streaming requests.
    max_tokens: int = 16000


# Depth presets the UI exposes as a single slider.
DEPTH_PRESETS: dict[str, CrewSettings] = {
    "quick": CrewSettings(num_angles=2, scout_max_turns=5, analyst_max_turns=2),
    "standard": CrewSettings(),
    "deep": CrewSettings(num_angles=5, max_parallel_scouts=3, scout_max_turns=12, analyst_max_turns=6),
}


# How much of a fetched page we hand back to the model. Pages can be huge;
# the first ~12k characters of readable text almost always hold the substance.
PAGE_CHAR_LIMIT = int(os.getenv("DEEPDIVE_PAGE_CHAR_LIMIT", "12000"))

# Results per web search.
SEARCH_RESULTS = int(os.getenv("DEEPDIVE_SEARCH_RESULTS", "6"))


# Rough price per million tokens for the cost estimate shown in the UI
# (Claude Opus 5.5 list price; check https://claude.com/pricing if you change MODEL).
PRICE_PER_MTOK = {"input": 4.00, "output": 20.00}


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Dollars, ignoring prompt-caching discounts. An upper-bound-ish estimate."""
    return (input_tokens * PRICE_PER_MTOK["input"] + output_tokens * PRICE_PER_MTOK["output"]) / 1_000_000
