"""Agent 4, the Advisor: turn a scored table into a recommendation a person
can act on, in words.

The analyst already did the maths. The advisor's job is judgement and
explanation: reading the traveller's notes ("no red-eyes", "hates
connections in Paris"), weighing a $40 saving against 5 extra hours, and
saying why. It answers in JSON (structured outputs) and must name its picks
by option id.

Then `verify()` checks those ids against the table. An LLM can write a
convincing paragraph about a flight that doesn't exist; an id that isn't in
the table is caught here and dropped, and any leg left without a pick gets
the analyst's top-scored option instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .analyst import TripAnalysis
from .config import SETTINGS
from .llm import ask_claude, json_of
from .models import TripRequest

ROLES = ["best_overall", "cheapest", "fastest", "most_comfortable", "alternative"]

ADVICE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "headline": {"type": "string", "description": "one-sentence bottom line"},
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "option_id": {"type": "string"},
                    "role": {"type": "string", "enum": ROLES},
                    "reason": {"type": "string"},
                },
                "required": ["option_id", "role", "reason"],
            },
        },
        "tradeoffs": {"type": "string"},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "picks", "tradeoffs", "caveats"],
}

SYSTEM = """You are a flight advisor. You get a trip request and, per leg, a
table of options that has already been scored (score 0 = best under the
traveller's priority; badges mark cheapest / fastest / fewest stops / best
match; "efficient" means no other option is better on price, duration and
stops at once).

Recommend flights for EACH leg: one best_overall pick per leg, plus up to two
other picks per leg when they offer a real trade-off (e.g. cheapest, fastest).
Prefer "efficient" options. Respect the traveller's notes (times of day,
airlines, layover limits) even when the score doesn't capture them.

Reasons must use the table's numbers (price, duration, stops, times). Only use
option_ids that appear in the table. If prices are marked estimated, say they
must be re-checked before booking. In tradeoffs, compare the picks in two or
three sentences. Caveats: anything the traveller should verify (baggage,
self-transfer, overnight layovers, budget). You give information only: never
offer to book or pay."""


@dataclass
class Advice:
    headline: str
    picks: list[dict]                    # [{"option_id", "role", "reason", "leg"}]
    tradeoffs: str
    caveats: list[str]
    corrections: list[str] = field(default_factory=list)  # what verify() changed


def _table(analysis: TripAnalysis) -> list[dict]:
    """The compact view the advisor sees: only fields that matter for a choice."""
    rows = []
    for la in analysis.legs:
        for o in la.options:
            rows.append({
                "id": o.id, "leg": o.leg, "airline": o.airline,
                "depart": o.depart_time, "arrive": o.arrive_time,
                "duration_min": o.duration_minutes, "stops": o.stops,
                "layovers": o.layovers, "price": o.price, "currency": o.currency,
                "score": o.score, "badges": o.badges,
                "over_budget": o.over_budget, "estimated": o.estimated,
            })
    return rows


def advise(client: Any, trip: TripRequest, analysis: TripAnalysis) -> Advice:
    payload = {
        "trip": {
            "route": f"{trip.origin_name or trip.origin} ({trip.origin}) → "
                     f"{trip.destination_name or trip.destination} ({trip.destination})",
            "depart_date": trip.depart_date, "return_date": trip.return_date,
            "adults": trip.adults, "cabin": trip.cabin, "priority": trip.priority,
            "max_stops": trip.max_stops, "budget_per_adult": trip.budget,
            "currency": trip.currency, "notes": trip.notes,
        },
        "cheapest_total_per_adult": analysis.cheapest_total,
        "within_budget": analysis.within_budget,
        "options": _table(analysis),
    }
    data = json_of(ask_claude(
        client,
        system=SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload, indent=1)}],
        effort=SETTINGS.advisor_effort,
        schema=ADVICE_SCHEMA,
    ))
    advice = Advice(data["headline"], data["picks"], data["tradeoffs"], data["caveats"])
    return verify(advice, analysis)


def verify(advice: Advice, analysis: TripAnalysis) -> Advice:
    """Keep only picks whose id is in the table; make sure every leg has one."""
    leg_of = {o.id: la.leg for la in analysis.legs for o in la.options}
    kept, seen = [], set()
    for pick in advice.picks:
        oid = pick["option_id"]
        if oid not in leg_of:
            advice.corrections.append(f"dropped pick {oid!r}: not in the options table")
            continue
        if oid in seen:
            continue  # the same flight recommended twice: keep the first role
        seen.add(oid)
        kept.append({**pick, "leg": leg_of[oid]})

    for la in analysis.legs:
        if la.options and not any(p["leg"] == la.leg for p in kept):
            top = la.options[0]  # analyst's best match (list is sorted by score)
            kept.append({"option_id": top.id, "role": "best_overall", "leg": la.leg,
                         "reason": "Top-scored option for your priority (added "
                                   "automatically: the advisor gave no valid pick here)."})
            advice.corrections.append(f"added {top.id} for the {la.leg} leg")
    advice.picks = kept
    return advice
