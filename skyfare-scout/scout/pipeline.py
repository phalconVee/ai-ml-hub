"""The crew, in order: parse → search → compare → advise.

This file is the whole "multi-agent" orchestration, and it's deliberately
plain Python. Each agent has one job and hands a typed object to the next:

    text ──parser──► TripRequest ──source──► FlightOptions per leg
         ──analyst──► TripAnalysis ──advisor──► Advice

The order never changes, so it lives in code rather than in a prompt. LLMs
are used where language or judgement is needed (parser, web scout, advisor)
and plain code where exactness is (validation, scoring, id checks).
Every step appends a line to `trace`, which the UI shows, so you can see
what each agent did and how long it took.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .advisor_agent import Advice, advise
from .analyst import TripAnalysis, analyse
from .parser_agent import parse_trip
from .sources import SearchOutcome
from .models import TripRequest


@dataclass
class ScoutResult:
    question: str = ""                       # set → ask the traveller, nothing searched
    trip: TripRequest | None = None
    source_label: str = ""
    searches: dict[str, SearchOutcome] = field(default_factory=dict)
    analysis: TripAnalysis | None = None
    advice: Advice | None = None
    trace: list[str] = field(default_factory=list)


def run(text: str, *, client: Any, source, today: date | None = None,
        on_step=lambda msg: None) -> ScoutResult:
    """Run the four agents. `on_step(msg)` lets the UI show live progress."""
    today = today or date.today()
    result = ScoutResult(source_label=source.label)

    def step(msg: str, started: float) -> None:
        line = f"{msg}  ({time.perf_counter() - started:.1f}s)"
        result.trace.append(line)
        on_step(line)

    # 1 ── Parser ─────────────────────────────────────────────────────────────
    t = time.perf_counter()
    parsed = parse_trip(client, text, today)
    if parsed.trip is None:
        result.question = parsed.question
        step(f"parser: needs clarification → {parsed.question}", t)
        return result
    trip = result.trip = parsed.trip
    step(f"parser: {trip.origin}→{trip.destination} {trip.depart_date}"
         f"{' ↩ ' + trip.return_date if trip.return_date else ' one-way'}, "
         f"{trip.adults} adult(s), {trip.cabin}, priority={trip.priority}", t)

    # 2 ── Fare scout, once per leg ───────────────────────────────────────────
    for leg in trip.legs():
        t = time.perf_counter()
        outcome = source.search(leg, trip)
        result.searches[leg.name] = outcome
        step(f"scout [{source.name}] {leg.name} {leg.origin}→{leg.destination} "
             f"{leg.date}: {len(outcome.options)} option(s); " + "; ".join(outcome.notes), t)

    # 3 ── Analyst (pure Python) ──────────────────────────────────────────────
    t = time.perf_counter()
    analysis = result.analysis = analyse(
        trip, {name: s.options for name, s in result.searches.items()})
    for la in analysis.legs:
        step(f"analyst {la.leg}: {len(la.options)} ranked"
             + (f"; {'; '.join(la.notes)}" if la.notes else ""), t)

    if not any(la.options for la in analysis.legs):
        result.trace.append("advisor skipped: no options to compare")
        return result

    # 4 ── Advisor ────────────────────────────────────────────────────────────
    t = time.perf_counter()
    result.advice = advise(client, trip, analysis)
    step(f"advisor: {len(result.advice.picks)} pick(s)"
         + (f"; corrections: {'; '.join(result.advice.corrections)}"
            if result.advice.corrections else ""), t)
    return result
