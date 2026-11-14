"""Agent 4, the Advisor: turn a scored table into advice a person can act on.

The analyst did the maths. The advisor adds judgement and explanation:
reading the traveller's notes ("quiet room", "walkable to the old town"),
weighing €30 a night against 2 km, and saying why, citing hotels by id.

Two guard rails, because the same code runs on Claude and on a 4B local model:

  verify()    An LLM can write a convincing paragraph about a hotel that isn't
              in the table. Every pick's id is checked against the table;
              unknown ids are dropped, and if nothing valid is left the
              analyst's top hotel is added as best_overall.
  fallback    If the brain fails outright (offline, broken JSON twice), the
              advisor still answers with a plain template built from the
              analyst's ranking, and says it did. The run never dead-ends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .analyst import Analysis
from .config import SETTINGS
from .llm import BrainError
from .models import StayRequest

ROLES = ["best_overall", "best_value", "best_location", "top_rated", "alternative"]

SCHEMA = {
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
                    "hotel_id": {"type": "string"},
                    "role": {"type": "string", "enum": ROLES},
                    "reason": {"type": "string"},
                },
                "required": ["hotel_id", "role", "reason"],
            },
        },
        "tradeoffs": {"type": "string"},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "picks", "tradeoffs", "caveats"],
}

SYSTEM = """You are a hotel advisor. You get a stay request and a table of
hotels that has already been scored (fit score 0-100, higher is better under
the traveller's priority; null means unknown, NOT zero). Badges mark the
cheapest, closest, top rated and best match.

Recommend 2-4 hotels: exactly one best_overall, plus picks for best_value,
best_location, top_rated or alternative when they offer a real trade-off.
Respect the traveller's notes even when the score doesn't capture them.

Each reason: 1-2 sentences using the table's facts (price per night, distance
in km, rating, must-haves confirmed or unknown). Use ONLY hotel ids from the
table. In tradeoffs, compare the picks in two or three sentences. Caveats:
what to verify before booking (estimated prices, unknown must-haves, taxes and
fees, cancellation terms). Information only: never offer to book or pay."""


@dataclass
class Advice:
    headline: str
    picks: list[dict]               # [{"hotel_id", "role", "reason"}]
    tradeoffs: str
    caveats: list[str]
    corrections: list[str] = field(default_factory=list)  # what verify()/fallback changed


def table(req: StayRequest, analysis: Analysis) -> list[dict]:
    """The compact view the advisor sees: only what matters for a choice."""
    return [{
        "id": o.id, "name": o.name, "kind": o.kind, "address": o.address,
        "price_per_night": o.price_per_night, "currency": o.currency or None,
        "over_budget": o.over_budget, "distance_km": o.distance_km,
        "rating_10": o.rating, "reviews": o.review_count, "stars": o.stars,
        "must_haves": o.must_have_status, "amenities": o.amenities,
        "fit_score": o.score, "badges": o.badges, "estimated": o.estimated,
        "caveat": o.caveat,
    } for o in analysis.ranked]


def advise(brain, req: StayRequest, analysis: Analysis) -> Advice:
    payload = {
        "stay": {"city": req.city, "near": req.area or "centre",
                 "dates": f"{req.check_in} → {req.check_out} ({req.nights} nights)",
                 "guests": req.guests, "rooms": req.rooms,
                 "budget_per_night": req.budget_per_night, "currency": req.currency,
                 "must_haves": req.must_haves, "nice_to_haves": req.nice_to_haves,
                 "min_rating_10": req.min_rating, "priority": req.priority,
                 "notes": req.notes},
        "hotels": table(req, analysis),
    }
    try:
        data = brain.json(system=SYSTEM, user=json.dumps(payload, indent=1),
                          schema=SCHEMA, effort=SETTINGS.advisor_effort)
        advice = Advice(data["headline"], data["picks"], data["tradeoffs"],
                        [str(c) for c in data["caveats"]])
    except (BrainError, KeyError, TypeError, ValueError) as err:  # ValueError ⊇ bad JSON
        advice = fallback(analysis)
        advice.corrections.append(f"advisor model failed ({err}); used the analyst's ranking")
    return verify(advice, analysis)


def verify(advice: Advice, analysis: Analysis) -> Advice:
    """Keep only picks whose id is in the table; guarantee one best_overall."""
    ids = {o.id for o in analysis.ranked}
    kept, seen = [], set()
    for pick in advice.picks:
        hid = str(pick.get("hotel_id", "")).strip()
        if hid not in ids:
            advice.corrections.append(f"dropped pick {hid!r}: not in the hotel table")
            continue
        if hid in seen:
            continue  # same hotel recommended twice: keep its first role
        seen.add(hid)
        role = pick.get("role") if pick.get("role") in ROLES else "alternative"
        kept.append({"hotel_id": hid, "role": role, "reason": pick.get("reason", "")})
    if analysis.ranked and not any(p["role"] == "best_overall" for p in kept):
        top = analysis.ranked[0]
        already = next((p for p in kept if p["hotel_id"] == top.id), None)
        if already:  # picked under another role: promote it, keep its reason
            already["role"] = "best_overall"
            kept.remove(already)
            kept.insert(0, already)
        else:
            kept.insert(0, {"hotel_id": top.id, "role": "best_overall",
                            "reason": "Highest fit score for your priority (added "
                                      "automatically: the advisor gave no valid best pick)."})
        advice.corrections.append(f"{top.id} set as best_overall")
    advice.picks = kept
    return advice


def fallback(analysis: Analysis) -> Advice:
    """No-LLM advice from the ranking alone: top hotel + the badge holders."""
    picks, used = [], set()
    for role, badge in [("best_overall", "best match"), ("best_value", "cheapest"),
                        ("best_location", "closest"), ("top_rated", "top rated")]:
        o = next((o for o in analysis.ranked if badge in o.badges), None)
        if o and o.id not in used:
            used.add(o.id)
            picks.append({"hotel_id": o.id, "role": role,
                          "reason": f"{o.name}: fit score {o.score}, badge '{badge}'."})
    return Advice(headline="Ranked by the analyst's fit score (no model commentary).",
                  picks=picks, tradeoffs="", caveats=["Check prices, fees and "
                  "cancellation terms on the hotel's own site before booking."])
