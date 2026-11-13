"""Agent 1, the Trip Parser: plain language in, a `TripRequest` out.

"Lagos to London around Dec 18, back after New Year, 2 of us, cheap but no
red-eyes" has to become airport codes, ISO dates, a passenger count and a
priority. That's exactly what LLMs are good at, so Claude does it, in ONE
call, with structured outputs (the reply must match `TRIP_SCHEMA`).

Then plain Python checks the result (`validate`), because a well-formed JSON
object can still be wrong: a date in the past, a return before departure,
"London" instead of a 3-letter code. If the request is too vague to search
(no destination at all, say), Claude answers `needs_clarification` with one
question for the traveller instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from .config import SETTINGS
from .llm import ask_claude, json_of
from .models import CABINS, PRIORITIES, TripRequest

# A JSON field that may be a value or null. Structured outputs need every
# property listed in `required`, so "optional" is spelled "nullable" here.
def _nullable(kind: str) -> dict:
    return {"anyOf": [{"type": kind}, {"type": "null"}]}


TRIP_SCHEMA = {
    "type": "object",
    "additionalProperties": False,  # required by structured outputs on every object
    "properties": {
        "status": {"type": "string", "enum": ["ok", "needs_clarification"]},
        "clarification_question": {"type": "string"},
        "origin": {"type": "string", "description": "3-letter IATA airport or city code"},
        "origin_name": {"type": "string"},
        "destination": {"type": "string", "description": "3-letter IATA airport or city code"},
        "destination_name": {"type": "string"},
        "depart_date": {"type": "string", "format": "date"},
        "return_date": _nullable("string"),
        "adults": {"type": "integer"},
        "cabin": {"type": "string", "enum": list(CABINS)},
        "max_stops": _nullable("integer"),
        "priority": {"type": "string", "enum": list(PRIORITIES)},
        "budget": _nullable("number"),
        "currency": {"type": "string", "description": "ISO 4217 code, e.g. USD"},
        "notes": {"type": "string"},
    },
    "required": ["status", "clarification_question", "origin", "origin_name",
                 "destination", "destination_name", "depart_date", "return_date",
                 "adults", "cabin", "max_stops", "priority", "budget", "currency", "notes"],
}

SYSTEM = """You turn a traveller's description of a flight into search parameters.

Rules:
- Airports: use the 3-letter IATA code. For a city with several airports, use
  the metropolitan city code when one exists (London → LON, New York → NYC,
  Lagos → LOS). Put the human-readable place in origin_name/destination_name.
- Dates: resolve relative dates ("next Friday", "mid-December", "after New
  Year") against TODAY, which the user message gives you. Pick a concrete
  date; mention the assumption in `notes`. Dates must not be in the past.
- One-way unless a return is mentioned: then return_date is null.
- adults defaults to 1, cabin to economy, currency to USD unless the text
  implies another (e.g. a budget in GBP).
- max_stops: 0 for "direct/nonstop", 1 for "at most one stop", null if not said.
- priority: "cheapest" if price is what matters, "fastest" if time is,
  otherwise "balanced".
- budget: the per-person ceiling if one is stated, else null.
- notes: other preferences (times of day, airlines to avoid, bags) and any
  assumption you made.
- If origin, destination or a rough travel date cannot be worked out at all,
  set status to needs_clarification and ask ONE short question in
  clarification_question; fill the other fields with your best guesses.
  Otherwise status is ok and clarification_question is "".
"""

IATA = re.compile(r"^[A-Z]{3}$")


@dataclass
class ParseResult:
    trip: TripRequest | None       # set when the request is searchable
    question: str = ""             # set when we must ask the traveller first
    problems: list[str] | None = None  # validation failures, shown in the UI


def parse_trip(client: Any, text: str, today: date) -> ParseResult:
    """Ask Claude to fill TRIP_SCHEMA, then validate it in plain Python."""
    response = ask_claude(
        client,
        system=SYSTEM,
        # TODAY goes in the user message, not the system prompt, so the system
        # prompt stays byte-identical across days (better for prompt caching).
        messages=[{"role": "user", "content": f"TODAY: {today.isoformat()}\n\nTrip: {text}"}],
        effort=SETTINGS.parser_effort,
        schema=TRIP_SCHEMA,
    )
    data = json_of(response)
    if data["status"] == "needs_clarification":
        return ParseResult(trip=None, question=data["clarification_question"])

    trip = TripRequest(
        origin=data["origin"].strip().upper(),
        destination=data["destination"].strip().upper(),
        depart_date=data["depart_date"],
        return_date=data["return_date"] or None,
        adults=data["adults"],
        cabin=data["cabin"],
        max_stops=data["max_stops"],
        priority=data["priority"],
        budget=data["budget"],
        currency=(data["currency"] or "USD").strip().upper(),
        origin_name=data["origin_name"],
        destination_name=data["destination_name"],
        notes=data["notes"],
    )
    problems = validate(trip, today)
    if problems:
        return ParseResult(trip=None, problems=problems,
                           question="I couldn't turn that into a valid search: "
                                    + "; ".join(problems) + ". Could you rephrase?")
    return ParseResult(trip=trip)


def validate(trip: TripRequest, today: date) -> list[str]:
    """Checks a JSON schema can't express. Returns a list of problems (empty = OK)."""
    problems = []
    for label, code in (("origin", trip.origin), ("destination", trip.destination)):
        if not IATA.match(code):
            problems.append(f"{label} '{code}' is not a 3-letter airport code")
    if trip.origin == trip.destination:
        problems.append("origin and destination are the same")

    try:
        depart = date.fromisoformat(trip.depart_date)
        if depart < today:
            problems.append(f"departure {trip.depart_date} is in the past")
        if trip.return_date:
            if date.fromisoformat(trip.return_date) < depart:
                problems.append("return date is before the departure date")
    except ValueError as err:  # malformed date string
        problems.append(f"bad date ({err})")

    if not 1 <= trip.adults <= 9:   # 9 = the usual online booking limit
        problems.append(f"{trip.adults} adults is outside 1–9")
    if trip.max_stops is not None and trip.max_stops < 0:
        problems.append("max_stops cannot be negative")
    if trip.budget is not None and trip.budget <= 0:
        problems.append("budget must be positive")
    return problems
