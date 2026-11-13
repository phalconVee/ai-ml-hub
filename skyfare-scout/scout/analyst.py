"""Agent 3, the Analyst: compare options with arithmetic, not with an LLM.

Comparing price, duration and stops is maths. Doing it in Python makes it
exact, free, instant and testable, and it gives the advisor (agent 4) solid
facts to explain instead of numbers to guess at. Per leg it:

  1. filters out options with more stops than the traveller allows,
  2. scores every option 0 (best) … 1 (worst) with priority-dependent weights,
  3. finds the Pareto-efficient options: those no other option beats on
     price, duration AND stops at once. Anything else is strictly worse than
     something on the list, so it can never be the "right" pick,
  4. hands out badges (cheapest, fastest, fewest stops, best match, efficient)
     and flags options over budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import SETTINGS
from .models import FlightOption, TripRequest

INF = math.inf


@dataclass
class LegAnalysis:
    leg: str                              # "outbound" | "return"
    options: list[FlightOption]           # sorted best score first
    removed: int = 0                      # dropped by the max-stops filter
    notes: list[str] = field(default_factory=list)


@dataclass
class TripAnalysis:
    legs: list[LegAnalysis]
    cheapest_total: float | None          # cheapest outbound + cheapest return, per adult
    within_budget: bool | None            # None when there's no budget or no prices


def comparable_price(opt: FlightOption, trip: TripRequest) -> float:
    """A price we can rank on, or +inf. Unknown prices and prices in another
    currency can't be compared fairly, so they rank last on price."""
    if opt.price is None or opt.currency.upper() != trip.currency:
        return INF
    return float(opt.price)


def _metrics(opt: FlightOption, trip: TripRequest) -> tuple[float, float, float]:
    """(price, duration, stops) with unknowns as +inf: 'unknown' never wins."""
    duration = INF if opt.duration_minutes is None else float(opt.duration_minutes)
    return comparable_price(opt, trip), duration, float(opt.stops)


def _normalise(values: list[float]) -> list[float]:
    """Min-max scale to 0…1 over the known values; unknown (+inf) → 1 (worst).
    If every known value is equal, they all get 0: that metric can't separate them."""
    known = [v for v in values if v != INF]
    if not known:
        return [1.0] * len(values)
    lo, hi = min(known), max(known)
    span = hi - lo
    return [1.0 if v == INF else (0.0 if span == 0 else (v - lo) / span) for v in values]


def score(options: list[FlightOption], trip: TripRequest) -> None:
    """Weighted sum of the normalised metrics. Writes `opt.score` in place."""
    w = SETTINGS.weights[trip.priority]
    metrics = [_metrics(o, trip) for o in options]
    prices = _normalise([m[0] for m in metrics])
    durations = _normalise([m[1] for m in metrics])
    stops = _normalise([m[2] for m in metrics])
    for opt, p, d, s in zip(options, prices, durations, stops):
        opt.score = round(w["price"] * p + w["duration"] * d + w["stops"] * s, 4)


def dominates(a: tuple, b: tuple) -> bool:
    """a dominates b: no worse on every metric and strictly better on one."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_ids(options: list[FlightOption], trip: TripRequest) -> set[str]:
    """Ids of options that nothing dominates. O(n²), fine for ≤ a few dozen."""
    m = {o.id: _metrics(o, trip) for o in options}
    return {o.id for o in options
            if not any(dominates(m[other.id], m[o.id]) for other in options if other is not o)}


def analyse_leg(leg: str, options: list[FlightOption], trip: TripRequest) -> LegAnalysis:
    kept = [o for o in options if trip.max_stops is None or o.stops <= trip.max_stops]
    result = LegAnalysis(leg=leg, options=kept, removed=len(options) - len(kept))
    if result.removed:
        result.notes.append(f"{result.removed} option(s) removed: more than "
                            f"{trip.max_stops} stop(s)")
    if not kept:
        return result

    score(kept, trip)
    efficient = pareto_ids(kept, trip)
    for o in kept:
        o.badges = ["efficient"] if o.id in efficient else []
        if trip.budget is not None and comparable_price(o, trip) != INF:
            o.over_budget = o.price > trip.budget

    def award(badge: str, key) -> None:
        """Give `badge` to every option tied for the best known value of `key`."""
        values = [key(o) for o in kept]
        known = [v for v in values if v != INF]
        if not known or (badge == "fewest stops" and len(set(known)) == 1):
            return  # nothing to award (unknown, or everyone has the same stops)
        best = min(known)
        for o, v in zip(kept, values):
            if v == best:
                o.badges.insert(0, badge)

    award("cheapest", lambda o: comparable_price(o, trip))
    award("fastest", lambda o: INF if o.duration_minutes is None else o.duration_minutes)
    award("fewest stops", lambda o: o.stops)
    kept.sort(key=lambda o: (o.score, o.id))  # id breaks ties → stable order
    kept[0].badges.insert(0, "best match")

    mixed = {o.currency.upper() for o in kept} - {trip.currency}
    if mixed:
        result.notes.append(f"prices in {', '.join(sorted(mixed))} ranked last on price "
                            f"(not converted to {trip.currency})")
    return result


def analyse(trip: TripRequest, per_leg: dict[str, list[FlightOption]]) -> TripAnalysis:
    legs = [analyse_leg(name, opts, trip) for name, opts in per_leg.items()]

    # Cheapest whole trip = cheapest option on each leg, added up.
    cheapest = []
    for la in legs:
        prices = [comparable_price(o, trip) for o in la.options]
        cheapest.append(min(prices, default=INF))
    total = None if not cheapest or INF in cheapest else round(sum(cheapest), 2)
    within = None if (trip.budget is None or total is None) else total <= trip.budget
    return TripAnalysis(legs=legs, cheapest_total=total, within_budget=within)
