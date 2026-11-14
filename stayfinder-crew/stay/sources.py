"""Agent 2, the Hotel Scout: where hotel options come from.

Three interchangeable sources with the same shape:

    source.search(request, target_point) -> SearchOutcome(options, sources, notes)

  WebSource    Claude only. Claude searches the web (server-side web_search
               tool) and writes notes; a second, tool-free call turns the notes
               into rows. Prices and ratings are *indicative* (what a page
               said), so every option is `estimated=True`, and any row whose
               link never appeared in a search result is thrown out.
  OsmSource    Any brain, no key, no LLM at all: real hotels from
               OpenStreetMap around the target point, with stars and the
               amenities mappers tagged. OSM has no prices or review scores,
               so those stay unknown (and the UI says so). This is the source
               that makes a 100%-local run (Ollama + OSM) possible.
  DemoSource   No network: deterministic made-up hotels, clearly labelled.

`pick_source()` chooses: web search when the brain can search, else OSM.
"""

from __future__ import annotations

import json
import random
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import SETTINGS
from .geo import NoGeocoder, http_post_form_json, km
from .llm import text_of
from .models import AMENITIES, HotelOption, StayRequest


@dataclass
class SearchOutcome:
    options: list[HotelOption]
    sources: list[dict] = field(default_factory=list)  # [{"title", "url"}] for the UI
    notes: list[str] = field(default_factory=list)      # what happened, for the trace


def finish(options: list[HotelOption], point) -> list[HotelOption]:
    """Shared last step: distance from the target, de-dupe, cap, number H1…Hn."""
    seen, unique = set(), []
    for o in options:
        key = o.name.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(o)
    for o in unique:
        if point and o.lat is not None and o.lon is not None:
            o.distance_km = round(km(point, (o.lat, o.lon)), 2)
    unique = unique[: SETTINGS.max_options]
    for i, o in enumerate(unique, start=1):
        o.id = f"H{i}"  # short ids the advisor must cite (and verify() checks)
    return unique


# ═══════════════════════════ OpenStreetMap (keyless) ════════════════════════
# OSM tag → our amenity key. A tag value of "no" is just as useful as "yes":
# it lets the analyst rule a hotel OUT on a must-have instead of guessing.
OSM_YES = {"yes", "wlan", "wifi", "free", "customers", "terminal", "wired"}


def osm_amenities(tags: dict) -> tuple[list[str], list[str]]:
    """(has, lacks) from an OSM element's tags. Untagged = unknown = neither."""
    checks = {
        "wifi": tags.get("internet_access"),
        "wheelchair_accessible": tags.get("wheelchair"),
        "air_conditioning": tags.get("air_conditioning"),
        "breakfast": tags.get("breakfast"),
        "pool": tags.get("swimming_pool"),
        "pet_friendly": tags.get("pets") or tags.get("dog"),
        "parking": tags.get("parking"),
        "non_smoking": "yes" if tags.get("smoking") == "no" else
                       ("no" if tags.get("smoking") in {"yes", "isolated"} else None),
        "restaurant": tags.get("restaurant"),
        "gym": tags.get("fitness_centre") or tags.get("gym"),
    }
    has, lacks = [], []
    for key, value in checks.items():
        if value is None:
            continue
        value = value.lower()
        if value in OSM_YES or value.isdigit() or value.startswith("yes"):
            has.append(key)
        elif value == "no":
            lacks.append(key)
    return has, lacks


def overpass_query(lat: float, lon: float, radius_m: int) -> str:
    # nwr = nodes, ways and relations (a hotel can be a point or a building
    # outline); `out center` gives a single point for outlines too.
    return (f"[out:json][timeout:30];"
            f'nwr["tourism"~"^(hotel|guest_house|hostel|motel|apartment)$"]["name"]'
            f"(around:{radius_m},{lat},{lon});out center tags 80;")


def parse_overpass(data: dict) -> list[HotelOption]:
    options = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lon = el.get("lon", el.get("center", {}).get("lon"))
        has, lacks = osm_amenities(tags)
        stars = tags.get("stars", "")
        street = " ".join(p for p in (tags.get("addr:housenumber"), tags.get("addr:street")) if p)
        options.append(HotelOption(
            id="", name=tags["name"], source="osm", kind=tags.get("tourism", "hotel"),
            address=street, lat=lat, lon=lon,
            stars=int(stars[0]) if stars[:1].isdigit() else None,  # "4", "4S" → 4
            amenities=has, lacks=lacks,
            source_url=tags.get("website") or tags.get("contact:website")
            or f"https://www.openstreetmap.org/{el['type']}/{el['id']}",
            caveat="from OpenStreetMap: no price or reviews",
        ))
    return options


class OsmSource:
    name = "osm"
    label = "OpenStreetMap (real hotels, no prices, no key)"

    def __init__(self, post: Callable = http_post_form_json):
        self.post = post  # injectable so tests never hit the network

    def search(self, request: StayRequest, point) -> SearchOutcome:
        if point is None:
            return SearchOutcome([], notes=[f"couldn't geocode {request.target!r}"])
        options: list[HotelOption] = []
        notes = []
        for radius in SETTINGS.osm_radius_m:  # widen the circle if too few hits
            data = self._query(overpass_query(*point, radius), notes)
            if data is None:
                break  # every mirror failed; keep whatever the last radius found
            options = parse_overpass(data)
            notes.append(f"{len(options)} places within {radius / 1000:g} km")
            if len(options) >= SETTINGS.osm_min_results:
                break
        # Real hotels before hostels/apartments, then nearest first, so the
        # cap in finish() keeps the most relevant rows.
        def rank(o: HotelOption):
            far = km(point, (o.lat, o.lon)) if o.lat is not None else 1e9
            return (o.kind != "hotel", far)
        options = finish(sorted(options, key=rank), point)
        return SearchOutcome(options, [{"title": "OpenStreetMap contributors",
                                        "url": "https://www.openstreetmap.org/copyright"}], notes)

    def _query(self, query: str, notes: list[str]) -> dict | None:
        """POST the query to each mirror until one answers with JSON."""
        for url in SETTINGS.overpass_urls:
            try:
                return self.post(url, {"data": query})
            except (OSError, ValueError) as err:  # HTTPError/timeouts ⊂ OSError
                notes.append(f"{urllib.parse.urlsplit(url).netloc}: {err}")
        return None


# ═════════════════════════ Claude web search (keyless) ══════════════════════
SCOUT_SYSTEM = """You are a hotel scout. Use web search to find concrete hotel
options for the stay described. Good pages: hotel booking/listing pages,
review sites, "best hotels in <area>" guides, the hotels' own sites.

Write plain notes, one hotel per line, with whatever is known: name,
neighbourhood or address, nightly price and currency (say if it's a total for
the stay or a "from" price, and for which dates), review score and its scale
(e.g. 8.7/10, 4.5/5) and review count, hotel class (stars), amenities
(wifi, breakfast, parking, pool, gym, AC, pets…), and the exact URL where you
saw it. Only report what a page actually said; write "unknown" for anything
missing. Aim for 6-12 distinct hotels that fit the budget and must-haves where
possible. Information only: never try to book or pay."""

EXTRACT_SYSTEM = f"""Convert hotel-search notes into JSON rows. Copy values
from the notes; never invent a hotel, price, rating or URL. Use null for
unknown numbers and "" for unknown text. price_basis says whether price is per
night or for the whole stay. rating_scale is 10 or 5 (or null). amenities and
lacks use only these keys: {", ".join(AMENITIES)}. source_url must be a URL
that appears in the notes."""

_ROW = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "address": {"type": "string"},
        "price": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "price_basis": {"type": "string", "enum": ["per_night", "total_stay", "unknown"]},
        "currency": {"type": "string"},
        "rating": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "rating_scale": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "review_count": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "stars": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "amenities": {"type": "array", "items": {"type": "string", "enum": AMENITIES}},
        "lacks": {"type": "array", "items": {"type": "string", "enum": AMENITIES}},
        "source_url": {"type": "string"},
        "caveat": {"type": "string", "description": "e.g. 'from-price, dates not confirmed'"},
    },
    "required": ["name", "address", "price", "price_basis", "currency", "rating",
                 "rating_scale", "review_count", "stars", "amenities", "lacks",
                 "source_url", "caveat"],
}
EXTRACT_SCHEMA = {"type": "object", "additionalProperties": False,
                  "properties": {"hotels": {"type": "array", "items": _ROW}},
                  "required": ["hotels"]}


def nightly(price, basis: str, nights: int) -> float | None:
    """Normalise to price per night: '€600 total for 4 nights' → 150."""
    if price is None or price <= 0:
        return None
    return round(price / nights, 2) if basis == "total_stay" and nights > 0 else float(price)


def rating10(rating, scale) -> float | None:
    """Normalise review scores to 0-10: 4.5/5 → 9.0."""
    if rating is None:
        return None
    if scale == 5 or (scale is None and rating <= 5):
        rating *= 2  # a bare "4.6" is almost always out of 5
    return round(min(rating, 10.0), 1)


class WebSource:
    name = "web"
    label = "Claude web search (indicative prices, no extra key)"

    def __init__(self, brain, geocoder=None):
        self.brain = brain  # must be a ClaudeBrain: uses brain.create with tools
        self.geocoder = geocoder or NoGeocoder()

    def search(self, request: StayRequest, point) -> SearchOutcome:
        # Step 1: research. Claude searches and writes notes with URLs.
        notes_text, seen = self._research(request)
        # Step 2: extraction, a separate tool-free call with structured outputs.
        # Keeping JSON out of the search call keeps each call simple and each
        # failure visible in the trace.
        data = self.brain.json(
            system=EXTRACT_SYSTEM,
            user=f"Stay: {request.target}, {request.check_in} → {request.check_out} "
                 f"({request.nights} nights). Notes:\n\n{notes_text}",
            schema=EXTRACT_SCHEMA, effort=SETTINGS.extract_effort)
        options, dropped = [], 0
        for row in data["hotels"]:
            # Trust, but verify: keep a row only if its link is one the search
            # actually returned. This filters out invented or mangled hotels.
            if _norm(row["source_url"]) not in seen:
                dropped += 1
                continue
            options.append(HotelOption(
                id="", name=row["name"], source="web", address=row["address"],
                price_per_night=nightly(row["price"], row["price_basis"], request.nights),
                currency=(row["currency"] or request.currency).upper()[:3],
                rating=rating10(row["rating"], row["rating_scale"]),
                review_count=row["review_count"],
                stars=row["stars"] if row["stars"] and 1 <= row["stars"] <= 5 else None,
                amenities=[a for a in row["amenities"] if a in AMENITIES],
                lacks=[a for a in row["lacks"] if a in AMENITIES],
                source_url=row["source_url"], estimated=True, caveat=row["caveat"],
            ))
        options = options[: SETTINGS.max_options]
        located = 0
        if SETTINGS.geocode_web_hotels:
            # Pin each hotel on the map so we can measure distance to the target.
            for o in options:
                hit = self.geocoder.locate(f"{o.name}, {o.address or request.city}")
                if hit:
                    o.lat, o.lon = hit
                    located += 1
        trace = [f"web search saw {len(seen)} pages, extracted {len(data['hotels'])} "
                 f"rows, kept {len(options)}, geocoded {located}"]
        if dropped:
            trace.append(f"dropped {dropped} row(s) whose link was never in a search result")
        return SearchOutcome(finish(options, point), list(seen.values()), trace)

    def _research(self, request: StayRequest) -> tuple[str, dict[str, dict]]:
        """Run the web-search turn (resuming `pause_turn`) and collect URLs seen."""
        ask = json.dumps({
            "city": request.city, "near": request.area or "city centre",
            "check_in": request.check_in, "check_out": request.check_out,
            "guests": request.guests, "rooms": request.rooms,
            "budget_per_night": request.budget_per_night, "currency": request.currency,
            "must_haves": request.must_haves, "min_rating_10": request.min_rating,
            "notes": request.notes,
        })
        tool = {"type": SETTINGS.web_search_tool, "name": "web_search",
                "max_uses": SETTINGS.max_web_searches}
        messages: list = [{"role": "user", "content": f"Find hotels for: {ask}"}]
        seen: dict[str, dict] = {}  # normalised url → {"title", "url"}
        notes = []
        for _ in range(SETTINGS.max_pause_resumes + 1):
            response = self.brain.create(system=SCOUT_SYSTEM, messages=messages,
                                         effort=SETTINGS.scout_effort, tools=[tool])
            _collect_urls(response, seen)
            notes.append(text_of(response))
            if response.stop_reason != "pause_turn":
                break
            # The server-side search loop hit its iteration cap mid-turn. Send
            # the partial turn back as-is and the API resumes where it stopped.
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
    """Compare URLs loosely: ignore scheme, host case, fragment, trailing slash."""
    parts = urllib.parse.urlsplit(url.strip())
    return f"{parts.netloc.lower()}{parts.path.rstrip('/')}?{parts.query}"


# ═══════════════════════════════ Demo (no keys) ═════════════════════════════
DEMO_NAMES = ["Demo Grand", "Sample Suites", "Placeholder Inn", "Mockingbird Hotel",
              "Example House", "Test Tower Hotel", "Fixture Lodge", "Stub & Co. Rooms"]


class DemoSource:
    name = "demo"
    label = "Demo data (made-up hotels, no network)"

    def search(self, request: StayRequest, point) -> SearchOutcome:
        # Seeded by the request, so the same request always gives the same table.
        rng = random.Random(f"{request.city}|{request.check_in}|{request.area}")
        base_lat, base_lon = point or (0.0, 0.0)
        budget = request.budget_per_night or 150
        options = []
        for name in DEMO_NAMES:
            stars = rng.randint(2, 5)
            has = rng.sample(AMENITIES, rng.randint(3, 8))
            options.append(HotelOption(
                id="", name=f"{name} (demo)", source="demo",
                address=f"{rng.randint(1, 200)} Example St, {request.city}",
                lat=base_lat + rng.uniform(-0.03, 0.03) if point else None,
                lon=base_lon + rng.uniform(-0.03, 0.03) if point else None,
                distance_km=None if point else round(rng.uniform(0.2, 6.0), 1),
                price_per_night=round(budget * rng.uniform(0.5, 1.6) * (0.7 + stars / 10)),
                currency=request.currency,
                rating=round(rng.uniform(6.2, 9.6), 1),
                review_count=rng.randint(40, 4000),
                stars=stars, amenities=has,
                lacks=[a for a in rng.sample(AMENITIES, 2) if a not in has],
                source_url="https://example.com/demo", estimated=True,
                caveat="made-up demo data",
            ))
        return SearchOutcome(finish(options, point), notes=["demo data, not real hotels"])


def pick_source(choice: str, brain, geocoder=None):
    """'auto' | 'web' | 'osm' | 'demo' → a source object."""
    if choice == "auto":
        choice = "web" if brain.can_search_web else "osm"
    if choice == "web":
        if not brain.can_search_web:
            raise ValueError("Web search needs the Claude brain. With Ollama, use "
                             "OpenStreetMap (real hotels) or Demo data.")
        return WebSource(brain, geocoder)
    if choice == "osm":
        return OsmSource()
    if choice == "demo":
        return DemoSource()
    raise ValueError(f"unknown source {choice!r}")
