"""Agent 2, the Fare Scout: where flight options come from.

Three interchangeable sources, all with the same shape:

    source.search(leg, trip) -> SearchOutcome(options, sources, notes)

  SerpApiSource   real Google Flights results. Needs SERPAPI_API_KEY (free plan
                  exists). Plain HTTP + JSON: no LLM involved at all.
  WebFareSource   no extra key. Claude searches the web (server-side web search
                  tool), then a second, cheap call turns its notes into rows.
                  Prices are *indicative* (what a page said), so every option
                  is marked `estimated=True`, and options whose link Claude
                  never actually saw in a search result are thrown out.
  DemoSource      no keys, no network: deterministic made-up flights, clearly
                  labelled. For trying the UI and for tests.

`pick_source()` chooses: SerpApi when its key is set, otherwise web search.
"""

from __future__ import annotations

import json
import os
import random
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .config import SETTINGS
from .llm import ask_claude, json_of, text_of
from .models import FlightOption, Leg, TripRequest


@dataclass
class SearchOutcome:
    options: list[FlightOption]
    sources: list[dict] = field(default_factory=list)  # [{"title", "url"}] for the UI
    notes: list[str] = field(default_factory=list)      # what happened, for the trace


def number_options(options: list[FlightOption], leg: Leg) -> list[FlightOption]:
    """Give each option a short, stable id the advisor can cite: OUT-1, RET-2…"""
    prefix = "OUT" if leg.name == "outbound" else "RET"
    for i, opt in enumerate(options, start=1):
        opt.id = f"{prefix}-{i}"
    return options


# ═════════════════════════════════ SerpApi ══════════════════════════════════
# SerpApi wraps Google Flights. Codes from its docs:
CLASS_CODE = {"economy": 1, "premium_economy": 2, "business": 3, "first": 4}
# Their `stops` filter: 0 = any, 1 = nonstop only, 2 = ≤1 stop, 3 = ≤2 stops.
def stops_code(max_stops: int | None) -> int:
    if max_stops is None:
        return 0
    return min(max_stops, 2) + 1


class SerpApiSource:
    name = "serpapi"
    label = "Google Flights via SerpApi (live prices)"

    def __init__(self, api_key: str, fetch=None):
        self.api_key = api_key
        # `fetch(url) -> dict` is injectable so tests never touch the network.
        self.fetch = fetch or _http_get_json

    def search(self, leg: Leg, trip: TripRequest) -> SearchOutcome:
        params = {
            "engine": "google_flights",
            "type": 2,                       # one-way: each leg is its own search
            "departure_id": leg.origin,
            "arrival_id": leg.destination,
            "outbound_date": leg.date,
            "adults": trip.adults,
            "travel_class": CLASS_CODE[trip.cabin],
            "stops": stops_code(trip.max_stops),
            "currency": trip.currency,
            "hl": "en",
            "api_key": self.api_key,
        }
        data = self.fetch(f"{SETTINGS.serpapi_url}?{urllib.parse.urlencode(params)}")
        if "error" in data:  # SerpApi reports problems inside a 200 JSON body
            return SearchOutcome([], notes=[f"SerpApi: {data['error']}"])
        options = parse_serpapi(data, leg, trip)
        link = data.get("search_metadata", {}).get("google_flights_url", "")
        sources = [{"title": "Google Flights", "url": link}] if link else []
        return SearchOutcome(options, sources,
                             [f"SerpApi returned {len(options)} options for {leg.name}"])


def parse_serpapi(data: dict, leg: Leg, trip: TripRequest) -> list[FlightOption]:
    """Turn SerpApi's `best_flights` + `other_flights` into FlightOptions."""
    link = data.get("search_metadata", {}).get("google_flights_url", "")
    options = []
    for item in data.get("best_flights", []) + data.get("other_flights", []):
        segments = item.get("flights", [])
        if not segments:
            continue
        first, last = segments[0], segments[-1]
        layovers = [f"{l.get('id', '?')} {_hm(l.get('duration'))}"
                    for l in item.get("layovers", [])]
        options.append(FlightOption(
            id="",
            leg=leg.name,
            # "Air France + KLM" when segments are on different carriers.
            airline=" + ".join(dict.fromkeys(s.get("airline", "?") for s in segments)),
            flight_numbers=[s.get("flight_number", "") for s in segments],
            depart_time=first.get("departure_airport", {}).get("time", ""),
            arrive_time=last.get("arrival_airport", {}).get("time", ""),
            duration_minutes=item.get("total_duration"),
            stops=len(segments) - 1,
            price=item.get("price"),
            currency=trip.currency,
            source="serpapi",
            source_url=link,
            layovers=layovers,
        ))
    return number_options(options[: SETTINGS.max_options_per_leg], leg)


def _hm(minutes: int | None) -> str:
    if minutes is None:
        return ""
    return f"{minutes // 60}h {minutes % 60:02d}m"


def _http_get_json(url: str) -> dict:
    """Stdlib GET → JSON. No extra HTTP dependency for one request."""
    with urllib.request.urlopen(url, timeout=SETTINGS.http_timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ═════════════════════════ Claude web search (keyless) ══════════════════════
WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",   # current version, with built-in result filtering
    "name": "web_search",
    "max_uses": SETTINGS.max_web_searches,
}

SCOUT_SYSTEM = """You are a fare scout. Use web search to find concrete flight
options for ONE one-way leg. Look for pages that list specific flights with
prices: airline sites, Google Flights / Kayak / Skyscanner style result pages,
route pages ("cheap flights from X to Y"), and schedule pages for durations
and stops.

Write plain notes, one option per line, with whatever is known:
airline, flight numbers, departure and arrival times, total duration, number
of stops and where, price and currency, and the exact URL where you saw it.
Only report what a page actually said; write "unknown" for anything missing.
Prices on the web are often "from" prices or for other dates: say so when
that's the case. Aim for 4-10 distinct options. This is information only:
never try to book anything."""

EXTRACT_SYSTEM = """Convert flight-search notes into JSON rows. Copy values
from the notes; never invent a flight, price or URL. Use null for unknown
numbers and "" for unknown text. Times as "YYYY-MM-DD HH:MM" when known.
duration_minutes is the total journey time. source_url must be a URL that
appears in the notes."""

_OPTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "airline": {"type": "string"},
        "flight_numbers": {"type": "array", "items": {"type": "string"}},
        "depart_time": {"type": "string"},
        "arrive_time": {"type": "string"},
        "duration_minutes": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "stops": {"type": "integer"},
        "layovers": {"type": "array", "items": {"type": "string"}},
        "price": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "currency": {"type": "string"},
        "source_url": {"type": "string"},
        "caveat": {"type": "string", "description": "e.g. 'from-price, date not confirmed'"},
    },
    "required": ["airline", "flight_numbers", "depart_time", "arrive_time",
                 "duration_minutes", "stops", "layovers", "price", "currency",
                 "source_url", "caveat"],
}
EXTRACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"options": {"type": "array", "items": _OPTION_SCHEMA}},
    "required": ["options"],
}


class WebFareSource:
    name = "web"
    label = "Claude web search (indicative prices, no extra key)"

    def __init__(self, client: Any):
        self.client = client

    def search(self, leg: Leg, trip: TripRequest) -> SearchOutcome:
        # Step 1: research. Claude searches and writes notes with URLs.
        notes_text, seen = self._research(leg, trip)
        # Step 2: extraction. A separate, tool-free call with structured outputs.
        # Why two calls? Web search answers carry citations, and keeping the
        # JSON step separate keeps each call simple and each failure visible.
        data = json_of(ask_claude(
            self.client,
            system=EXTRACT_SYSTEM,
            messages=[{"role": "user", "content": f"Notes for {leg.origin} → "
                       f"{leg.destination} on {leg.date}:\n\n{notes_text}"}],
            effort=SETTINGS.extract_effort,
            schema=EXTRACT_SCHEMA,
        ))
        options, dropped = [], 0
        for row in data["options"]:
            # Trust, but verify: keep a row only if its link is one the search
            # actually returned. This filters out invented or mangled fares.
            if _norm(row["source_url"]) not in seen:
                dropped += 1
                continue
            options.append(FlightOption(
                id="", leg=leg.name, airline=row["airline"] or "unknown",
                flight_numbers=row["flight_numbers"], depart_time=row["depart_time"],
                arrive_time=row["arrive_time"], duration_minutes=row["duration_minutes"],
                stops=max(row["stops"], 0), price=row["price"],
                currency=row["currency"] or trip.currency, source="web",
                source_url=row["source_url"], layovers=row["layovers"], estimated=True,
            ))
        options = number_options(options[: SETTINGS.max_options_per_leg], leg)
        trace = [f"web search saw {len(seen)} pages, extracted {len(data['options'])} "
                 f"rows, kept {len(options)}"]
        if dropped:
            trace.append(f"dropped {dropped} row(s) whose link was never in a search result")
        return SearchOutcome(options, list(seen.values()), trace)

    def _research(self, leg: Leg, trip: TripRequest) -> tuple[str, dict[str, dict]]:
        """Run the web-search turn (resuming `pause_turn`) and collect URLs seen."""
        ask = (f"One-way flights {leg.origin} → {leg.destination} on {leg.date}, "
               f"{trip.adults} adult(s), {trip.cabin.replace('_', ' ')}, prices in "
               f"{trip.currency}. Max stops: "
               f"{'any' if trip.max_stops is None else trip.max_stops}. "
               f"Traveller notes: {trip.notes or 'none'}")
        messages: list = [{"role": "user", "content": ask}]
        seen: dict[str, dict] = {}     # normalised url → {"title", "url"}
        notes = []
        for _ in range(SETTINGS.max_pause_resumes + 1):
            response = ask_claude(self.client, system=SCOUT_SYSTEM, messages=messages,
                                  effort=SETTINGS.scout_effort, tools=[WEB_SEARCH_TOOL])
            _collect_urls(response, seen)
            notes.append(text_of(response))
            if response.stop_reason != "pause_turn":
                break
            # The server-side search loop hit its iteration cap mid-turn. Send
            # the partial turn back as-is and the API resumes where it stopped
            # (no extra "continue" message needed).
            messages = [messages[0], {"role": "assistant", "content": response.content}]
        return "\n".join(n for n in notes if n), seen


def _collect_urls(response: Any, seen: dict[str, dict]) -> None:
    """URLs from search results and from citations on the text blocks."""
    for block in response.content:
        kind = getattr(block, "type", "")
        if kind == "web_search_tool_result":
            # Success: a list of results. Error: a single object (no exception!).
            if isinstance(block.content, list):
                for r in block.content:
                    seen.setdefault(_norm(r.url), {"title": r.title, "url": r.url})
        elif kind == "text":
            for c in getattr(block, "citations", None) or []:
                url = getattr(c, "url", None)
                if url:
                    seen.setdefault(_norm(url), {"title": getattr(c, "title", url), "url": url})


def _norm(url: str) -> str:
    """Compare URLs loosely: ignore scheme case, fragment and trailing slash."""
    parts = urllib.parse.urlsplit(url.strip())
    return f"{parts.netloc.lower()}{parts.path.rstrip('/')}?{parts.query}"


# ═══════════════════════════════ Demo (no keys) ═════════════════════════════
DEMO_AIRLINES = ["Demo Air", "Sample Skies", "Placeholder Airways", "Mockingbird Jet"]


class DemoSource:
    name = "demo"
    label = "Demo data (made-up flights, no network)"

    def search(self, leg: Leg, trip: TripRequest) -> SearchOutcome:
        # Seeded by the leg, so the same search always gives the same table.
        rng = random.Random(f"{leg.origin}{leg.destination}{leg.date}{trip.cabin}")
        cabin_mult = {"economy": 1, "premium_economy": 1.8, "business": 3.5, "first": 6}[trip.cabin]
        base_minutes = rng.randint(90, 600)
        options = []
        for i in range(8):
            stops = rng.choice([0, 0, 1, 1, 1, 2])
            if trip.max_stops is not None and stops > trip.max_stops:
                stops = trip.max_stops
            duration = base_minutes + stops * rng.randint(60, 240)
            dep_h, dep_m = rng.randint(5, 22), rng.choice([0, 15, 30, 45])
            arr_total = dep_h * 60 + dep_m + duration
            # Fewer stops tend to cost more, like real fares.
            price = round((120 + base_minutes * 0.6) * cabin_mult * (1.35 - 0.15 * stops)
                          * rng.uniform(0.8, 1.3))
            airline = rng.choice(DEMO_AIRLINES)
            options.append(FlightOption(
                id="", leg=leg.name, airline=airline,
                flight_numbers=[f"DM{rng.randint(100, 999)}" for _ in range(stops + 1)],
                depart_time=f"{leg.date} {dep_h:02d}:{dep_m:02d}",
                arrive_time=f"+{arr_total // 1440}d {arr_total % 1440 // 60:02d}:{arr_total % 60:02d}",
                duration_minutes=duration, stops=stops, price=float(price),
                currency=trip.currency, source="demo",
                layovers=[f"HUB{j + 1} {_hm(rng.randint(50, 200))}" for j in range(stops)],
            ))
        return SearchOutcome(number_options(options, leg),
                             notes=[f"demo source made {len(options)} fake options"])


# ═══════════════════════════════ Choosing one ═══════════════════════════════
def pick_source(choice: str, client: Any):
    """'auto' → SerpApi if its key is set, else Claude web search."""
    key = os.environ.get("SERPAPI_API_KEY", "").strip()
    if choice == "demo":
        return DemoSource()
    if choice == "serpapi" or (choice == "auto" and key):
        if not key:
            raise ValueError("SERPAPI_API_KEY is not set")
        return SerpApiSource(key)
    return WebFareSource(client)
