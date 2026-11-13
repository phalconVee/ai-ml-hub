"""The data that flows between the agents.

Three shapes carry the whole app:

    TripRequest   what the traveller wants   (made by the parser agent)
    FlightOption  one flight you could take  (made by a flight source)
    Leg           "outbound" or "return": one origin → destination search

Plain dataclasses, no framework: you can print any of them and see
everything. The JSON schemas the LLM must fill sit next to the parser and
extractor that use them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CABINS = ("economy", "premium_economy", "business", "first")
PRIORITIES = ("cheapest", "fastest", "balanced")


@dataclass
class TripRequest:
    origin: str                  # IATA airport/city code, e.g. "LOS"
    destination: str             # e.g. "LHR"
    depart_date: str             # ISO "YYYY-MM-DD"
    return_date: str | None      # None = one-way
    adults: int = 1
    cabin: str = "economy"
    max_stops: int | None = None  # None = any number of stops
    priority: str = "balanced"    # which trade-off the traveller cares about
    budget: float | None = None   # total per adult, in `currency`
    currency: str = "USD"
    origin_name: str = ""         # human names, for display only
    destination_name: str = ""
    notes: str = ""               # anything else ("prefers morning flights")

    def legs(self) -> list[Leg]:
        """A round trip is searched as two one-way legs (see README: why)."""
        legs = [Leg("outbound", self.origin, self.destination, self.depart_date)]
        if self.return_date:
            legs.append(Leg("return", self.destination, self.origin, self.return_date))
        return legs


@dataclass(frozen=True)
class Leg:
    name: str          # "outbound" | "return"
    origin: str
    destination: str
    date: str


@dataclass
class FlightOption:
    id: str                       # "OUT-1", "RET-3": stable handle the advisor must cite
    leg: str                      # "outbound" | "return"
    airline: str
    flight_numbers: list[str]
    depart_time: str              # "YYYY-MM-DD HH:MM" local time, or "" if unknown
    arrive_time: str
    duration_minutes: int | None  # door-to-door, including layovers
    stops: int
    price: float | None           # per adult; None when the source didn't say
    currency: str
    source: str                   # "serpapi" | "web" | "demo"
    source_url: str = ""          # where a human can check this fare
    layovers: list[str] = field(default_factory=list)  # e.g. ["CDG 1h 20m"]
    estimated: bool = False       # True = a price seen on the web, not a live quote

    # Filled in by analyst.py, never by a source:
    score: float | None = None         # 0 (best) … 1 (worst) under the trip's priority
    badges: list[str] = field(default_factory=list)  # "cheapest", "fastest", …
    over_budget: bool = False
