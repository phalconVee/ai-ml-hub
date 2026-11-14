"""Every knob in one place.

Change a value here and the whole app follows. No other file hard-codes a
model name, an effort level, a search radius or a scoring weight.
"""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)  # frozen: settings are read, never changed at runtime
class Settings:
    # ── Brain #1: Claude (cloud, default) ────────────────────────────────────
    # Claude Opus 5.5 always thinks before answering; `effort` is the dial for
    # how hard. Each agent picks its own effort, because their jobs differ.
    claude_model: str = "claude-opus-5-5"
    claude_max_tokens: int = 16000      # room for thinking + answer (non-streaming)
    parser_effort: str = "low"          # sentence → fields: easy
    scout_effort: str = "medium"        # searching + reading hotel pages: judgement
    extract_effort: str = "low"         # notes → JSON rows: mechanical
    advisor_effort: str = "medium"      # weighing trade-offs and explaining them

    # ── Brain #2: Ollama (100% local) ────────────────────────────────────────
    # Any chat model you've pulled works. qwen3:4b is a small *reasoning* model
    # (~2.6 GB) that handles structured JSON well. Override with OLLAMA_MODEL.
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_think: bool = True           # let a reasoning model think first
    ollama_num_ctx: int = 8192          # the advisor's table needs a few k tokens
    ollama_timeout_s: float = 300.0     # local models on a laptop can be slow
    json_retries: int = 1               # small models sometimes emit broken JSON

    # ── Hotel data: Claude web search (keyless beyond ANTHROPIC_API_KEY) ─────
    web_search_tool: str = "web_search_20260318"
    max_web_searches: int = 6
    max_pause_resumes: int = 3          # cap on `pause_turn` continuations

    # ── Hotel data: OpenStreetMap (free, no key) ─────────────────────────────
    # Nominatim (geocoding) asks for a descriptive User-Agent and ≤1 request/s.
    nominatim_url: str = "https://nominatim.openstreetmap.org/search"
    # Overpass is volunteer-run and often busy (HTTP 429/502/504), so we try
    # public mirrors in order until one answers.
    overpass_urls: tuple = ("https://overpass-api.de/api/interpreter",
                            "https://overpass.private.coffee/api/interpreter",
                            "https://overpass.kumi.systems/api/interpreter")
    user_agent: str = "stayfinder-crew/0.1 (open-source portfolio demo)"
    geocode_pause_s: float = 1.1        # respect Nominatim's 1 req/s policy
    osm_radius_m: tuple = (2000, 5000)  # try a tight circle, widen if too few hits
    osm_min_results: int = 6
    http_timeout_s: float = 40.0
    # Geocode web-search hotels too (so they get a distance + a map pin).
    geocode_web_hotels: bool = True

    # ── Comparison (analyst.py) ──────────────────────────────────────────────
    max_options: int = 12               # keeps the table and the advisor prompt small
    # How much each factor matters, per traveller priority. Rows sum to 1.
    # Higher fit score = better.
    weights: dict = field(default_factory=lambda: {
        "value":    {"price": 0.45, "location": 0.20, "quality": 0.20, "amenities": 0.15},
        "location": {"price": 0.20, "location": 0.50, "quality": 0.15, "amenities": 0.15},
        "quality":  {"price": 0.15, "location": 0.20, "quality": 0.50, "amenities": 0.15},
        "balanced": {"price": 0.30, "location": 0.25, "quality": 0.25, "amenities": 0.20},
    })
    # Distance at which the location score has dropped to 0.5.
    half_score_km: float = 2.0


SETTINGS = Settings()
