"""Every knob in one place.

Change a number here and the whole app follows. No other file hard-codes a
model name, an effort level, a search limit or a scoring weight.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)  # frozen: settings are read, never changed at runtime
class Settings:
    # ── The model ────────────────────────────────────────────────────────────
    # Claude Opus 5.5 is the current default Opus. It always thinks before it
    # answers; `effort` is the dial for how hard. Each agent picks its own
    # effort below, because their jobs are very different in difficulty.
    model: str = "claude-opus-5-5"
    max_tokens: int = 16000  # room for thinking + answer on non-streaming calls

    # Turning a sentence into fields is easy work → low effort (fast, cheap).
    parser_effort: str = "low"
    # Hunting fares on the web and reading pages takes judgement → medium.
    scout_effort: str = "medium"
    # Pulling numbers out of the scout's notes into JSON is mechanical → low.
    extract_effort: str = "low"
    # Weighing trade-offs and explaining them well → medium.
    advisor_effort: str = "medium"

    # Server-side refusal fallback: if a safety classifier declines a
    # request, the API re-runs it on a fallback model inside the same call.
    fallback_beta: str = "server-side-fallback-2026-07-01"

    # ── Keyless search: Claude web search (sources.py) ───────────────────────
    max_web_searches: int = 6      # cap on searches per leg
    max_pause_resumes: int = 3     # cap on `pause_turn` continuations

    # ── SerpApi (sources.py), only used when SERPAPI_API_KEY is set ──────────
    serpapi_url: str = "https://serpapi.com/search.json"
    http_timeout_s: float = 30.0

    # ── Comparison (analyst.py) ──────────────────────────────────────────────
    max_options_per_leg: int = 12  # keep tables (and the advisor's prompt) small
    # How much price / duration / stops matter, per traveller priority.
    # Each row sums to 1. Lower weighted score = better option.
    weights: dict = field(default_factory=lambda: {
        "cheapest": {"price": 0.70, "duration": 0.20, "stops": 0.10},
        "fastest":  {"price": 0.20, "duration": 0.60, "stops": 0.20},
        "balanced": {"price": 0.45, "duration": 0.35, "stops": 0.20},
    })


SETTINGS = Settings()
