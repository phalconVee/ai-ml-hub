"""The crew, in order: parse → locate → scout → compare → advise.

This file is the whole "multi-agent" orchestration, and it's deliberately
plain Python. Each agent has one job and hands a typed object to the next:

    text ─parser─► StayRequest ─geocoder─► target point
         ─scout─► [HotelOption] ─analyst─► Analysis ─advisor─► Advice

The order never changes, so it lives in code rather than in a prompt. LLMs
are used where language or judgement is needed (parser, web scout, advisor);
plain code where exactness is (dates, geocoding, distances, scoring, id
checks). Every step appends a timed line to `trace`, which the UI shows live.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

from .advisor_agent import Advice, advise
from .analyst import Analysis, analyse
from .geo import NoGeocoder
from .models import StayRequest
from .parser_agent import parse_stay
from .sources import SearchOutcome


@dataclass
class StayResult:
    question: str = ""                  # set → ask the traveller; nothing searched
    request: StayRequest | None = None
    brain_label: str = ""
    source_label: str = ""
    target_point: tuple | None = None
    search: SearchOutcome | None = None
    analysis: Analysis | None = None
    advice: Advice | None = None
    trace: list[str] = field(default_factory=list)


def run(text: str, *, brain, source, geocoder=None, today: date | None = None,
        on_step=lambda msg: None) -> StayResult:
    """Run the crew. `on_step(msg)` lets the UI stream progress lines."""
    today = today or date.today()
    geocoder = geocoder or NoGeocoder()
    result = StayResult(brain_label=brain.label, source_label=source.label)

    def step(msg: str, started: float) -> None:
        line = f"{msg}  ({time.perf_counter() - started:.1f}s)"
        result.trace.append(line)
        on_step(line)

    # 1 ── Parser (LLM) ───────────────────────────────────────────────────────
    t = time.perf_counter()
    parsed = parse_stay(brain, text, today)
    if parsed.request is None:
        result.question = parsed.question
        step(f"parser: needs clarification → {parsed.question}", t)
        return result
    req = result.request = parsed.request
    step(f"parser: {req.target}, {req.check_in}→{req.check_out} ({req.nights} nights), "
         f"{req.guests} guest(s), budget {req.budget_per_night or 'any'} {req.currency}, "
         f"must-haves {req.must_haves or 'none'}, priority={req.priority}", t)

    # 2 ── Locate the target (code + OSM) ─────────────────────────────────────
    t = time.perf_counter()
    point = result.target_point = geocoder.locate(req.target)
    if point is None and req.area:
        point = result.target_point = geocoder.locate(req.city)  # area unknown: use city
    step(f"geocoder: {req.target} → {point or 'not located (distances unknown)'}", t)

    # 3 ── Hotel scout ────────────────────────────────────────────────────────
    t = time.perf_counter()
    outcome = result.search = source.search(req, point)
    step(f"scout [{source.name}]: {len(outcome.options)} hotel(s); "
         + "; ".join(outcome.notes), t)

    # 4 ── Analyst (pure Python) ──────────────────────────────────────────────
    t = time.perf_counter()
    analysis = result.analysis = analyse(req, outcome.options)
    step(f"analyst: {len(analysis.ranked)} ranked"
         + (f"; {'; '.join(analysis.notes)}" if analysis.notes else ""), t)
    if not analysis.ranked:
        result.trace.append("advisor skipped: no hotels to compare")
        return result

    # 5 ── Advisor (LLM, with verify + fallback) ──────────────────────────────
    t = time.perf_counter()
    result.advice = advise(brain, req, analysis)
    step(f"advisor: {len(result.advice.picks)} pick(s)"
         + (f"; corrections: {'; '.join(result.advice.corrections)}"
            if result.advice.corrections else ""), t)
    return result
