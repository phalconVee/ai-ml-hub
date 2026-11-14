"""Agent 3, the Analyst: pure Python, no LLM.

Comparing twelve hotels on four factors is arithmetic, and arithmetic is
something code does exactly and a language model only does approximately. So
this agent is deterministic: same table in, same ranking out.

For each hotel it:
  1. rules it OUT if a source says it lacks a must-have, or its known review
     score is under the traveller's minimum (unknown never rules anything out);
  2. scores four factors on 0-1: price, location, quality, amenities;
  3. blends them with the weights for the traveller's priority (config.py)
     into a 0-100 fit score and sorts best-first;
  4. adds badges (cheapest, closest, top rated, best match).

Missing data scores a neutral 0.5 rather than 0 or 1. An OSM hotel with no
price isn't punished as if it were expensive, nor rewarded as if it were
free, and the UI lists which facts were unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import SETTINGS
from .models import HotelOption, StayRequest

NEUTRAL = 0.5


@dataclass
class Analysis:
    ranked: list[HotelOption]                          # best first
    excluded: list[tuple[HotelOption, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    within_budget: int = 0
    known: dict = field(default_factory=dict)          # how many have price/rating/distance


def analyse(req: StayRequest, options: list[HotelOption]) -> Analysis:
    kept, excluded = [], []
    for o in options:
        reason = rule_out(req, o)
        (excluded.append((o, reason)) if reason else kept.append(o))

    weights = SETTINGS.weights[req.priority]
    price_part = price_scores(req, kept)  # needs the whole list (min/max)
    for o in kept:
        o.must_have_status = {a: status(o, a) for a in req.must_haves}
        o.parts = {
            "price": price_part[o.id],
            "location": location_score(o),
            "quality": quality_score(o),
            "amenities": amenity_score(req, o),
        }
        o.score = round(100 * sum(weights[k] * v for k, v in o.parts.items()), 1)

    # Sort by score; on a tie prefer the hotel we know more about.
    kept.sort(key=lambda o: (-o.score, unknown_count(o)))
    add_badges(req, kept)

    a = Analysis(kept, excluded)
    a.within_budget = sum(1 for o in kept if o.price_per_night is not None and not o.over_budget)
    a.known = {
        "price": sum(comparable_price(req, o) is not None for o in kept),
        "rating": sum(o.rating is not None for o in kept),
        "distance": sum(o.distance_km is not None for o in kept),
    }
    if kept and a.known["price"] == 0:
        a.notes.append(f"no prices known in {req.currency}: price scored neutral for every hotel")
    if req.budget_per_night and a.known["price"] and a.within_budget == 0:
        a.notes.append("nothing with a known price is within budget")
    if excluded:
        a.notes.append(f"ruled out {len(excluded)}")
    return a


def rule_out(req: StayRequest, o: HotelOption) -> str:
    """A reason to drop this hotel, or "" to keep it. Only KNOWN facts count."""
    missing = [a for a in req.must_haves if a in o.lacks]
    if missing:
        return f"lacks must-have: {', '.join(missing)}"
    if req.min_rating is not None and o.rating is not None and o.rating < req.min_rating:
        return f"rated {o.rating}/10, below your minimum {req.min_rating}"
    return ""


def status(o: HotelOption, amenity: str) -> str:
    return "yes" if amenity in o.amenities else "no" if amenity in o.lacks else "unknown"


def comparable_price(req: StayRequest, o: HotelOption) -> float | None:
    """Only compare prices in the traveller's currency (no FX guessing)."""
    if o.price_per_night is None or (o.currency and o.currency != req.currency):
        return None
    return o.price_per_night


def price_scores(req: StayRequest, options: list[HotelOption]) -> dict[str, float]:
    """Cheapest known = 1, dearest known = 0; over budget is penalised further."""
    prices = {o.id: comparable_price(req, o) for o in options}
    known = [p for p in prices.values() if p is not None]
    lo, hi = (min(known), max(known)) if known else (0, 0)
    scores = {}
    for o in options:
        p = prices[o.id]
        if p is None:
            scores[o.id] = NEUTRAL
            continue
        s = 1.0 if hi == lo else 1 - (p - lo) / (hi - lo)
        if req.budget_per_night and p > req.budget_per_night:
            o.over_budget = True
            over = (p - req.budget_per_night) / req.budget_per_night  # 0.2 = 20% over
            s *= max(0.0, 1 - 2 * over)  # 20% over halves the score, 50% over zeroes it
        scores[o.id] = round(s, 3)
    return scores


def location_score(o: HotelOption) -> float:
    """1 at the target, 0.5 at `half_score_km`, falling smoothly after that."""
    if o.distance_km is None:
        return NEUTRAL
    return round(1 / (1 + o.distance_km / SETTINGS.half_score_km), 3)


def quality_score(o: HotelOption) -> float:
    """Guest reviews matter most; the official star class is a weaker signal."""
    if o.rating is not None and o.stars is not None:
        return round(0.8 * o.rating / 10 + 0.2 * o.stars / 5, 3)
    if o.rating is not None:
        return round(o.rating / 10, 3)
    if o.stars is not None:
        return round(o.stars / 5, 3)
    return NEUTRAL


def amenity_score(req: StayRequest, o: HotelOption) -> float:
    """Share of wanted amenities confirmed. Must-haves count double."""
    wanted = [(a, 2) for a in req.must_haves] + [(a, 1) for a in req.nice_to_haves]
    if not wanted:
        return 1.0  # nothing asked for: every hotel equally fine
    value = {"yes": 1.0, "unknown": NEUTRAL, "no": 0.0}
    total = sum(w for _, w in wanted)
    return round(sum(w * value[status(o, a)] for a, w in wanted) / total, 3)


def unknown_count(o: HotelOption) -> int:
    return sum(v is None for v in (o.price_per_night, o.rating, o.distance_km))


def add_badges(req: StayRequest, ranked: list[HotelOption]) -> None:
    if not ranked:
        return
    ranked[0].badges.append("best match")

    def best(key, label):
        candidates = [o for o in ranked if key(o) is not None]
        if candidates:
            min(candidates, key=key).badges.append(label)

    best(lambda o: comparable_price(req, o), "cheapest")
    best(lambda o: o.distance_km, "closest")
    best(lambda o: -o.rating if o.rating is not None else None, "top rated")
    if req.must_haves:
        for o in ranked:
            if all(v == "yes" for v in o.must_have_status.values()):
                o.badges.append("all must-haves confirmed")
