"""Agent 1, the Parser: plain words → a validated StayRequest.

"Lisbon, 3 nights from Dec 12, two of us, under €150, near Alfama, need wifi
and AC" becomes typed fields. The LLM does the language part (relative dates,
"two of us", currency symbols, synonyms for amenities); plain Python then
checks the result, because a model can return a check-out before the
check-in just as fluently as a correct one.

If something essential is missing (no city, no dates), the parser asks one
clarifying question instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .config import SETTINGS
from .models import AMENITIES, PRIORITIES, StayRequest

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["ok", "needs_clarification"]},
        "clarification_question": {"type": "string"},
        "city": {"type": "string"},
        "area": {"type": "string"},
        "check_in": {"type": "string", "description": "YYYY-MM-DD"},
        "check_out": {"type": "string", "description": "YYYY-MM-DD"},
        "guests": {"type": "integer"},
        "rooms": {"type": "integer"},
        "budget_per_night": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "currency": {"type": "string", "description": "ISO 4217, e.g. USD, EUR"},
        "must_haves": {"type": "array", "items": {"type": "string", "enum": AMENITIES}},
        "nice_to_haves": {"type": "array", "items": {"type": "string", "enum": AMENITIES}},
        "min_rating": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "priority": {"type": "string", "enum": PRIORITIES},
        "notes": {"type": "string"},
    },
    "required": ["status", "clarification_question", "city", "area", "check_in",
                 "check_out", "guests", "rooms", "budget_per_night", "currency",
                 "must_haves", "nice_to_haves", "min_rating", "priority", "notes"],
}

SYSTEM = f"""You turn a traveller's hotel request into JSON fields.

Rules:
- Resolve relative dates ("next Friday", "3 nights from Dec 12") against the
  date given as TODAY. Dates are YYYY-MM-DD; check_out is the departure day.
  If only a length of stay is given with a start date, compute check_out.
- If no year is given, use the next occurrence of that date on or after TODAY.
- area: a neighbourhood or landmark they want to be near; "" if none.
- budget_per_night: per room per night. If they give a total budget for the
  stay, divide by the number of nights. null if no budget.
- currency: from symbols or words (€ → EUR, £ → GBP, ₦ → NGN); default USD.
- must_haves: amenities they REQUIRE ("need", "must", "has to have").
  nice_to_haves: amenities they'd LIKE. Map to these keys only:
  {", ".join(AMENITIES)}.
- min_rating on a 0-10 scale ("4+ stars on reviews" → 8, "rated 8+" → 8);
  null if not said. Hotel class ("a 4-star hotel") goes in notes, not here.
- priority: "value" (cheap / good deal), "location" (close to something),
  "quality" (best rated / luxury), else "balanced".
- notes: anything else that matters (quiet room, hotel class, vibe), in
  their words.
- guests default 2, rooms default 1.
- If the city or the dates are missing or impossible to infer, set status to
  "needs_clarification" and ask ONE short question; fill other fields with
  your best guess or empty values. Otherwise status "ok" and
  clarification_question "".
"""


@dataclass
class ParseOutcome:
    request: StayRequest | None = None
    question: str = ""  # set when we need to ask the traveller something


def parse_stay(brain, text: str, today: date) -> ParseOutcome:
    data = brain.json(system=SYSTEM, user=f"TODAY: {today.isoformat()}\n\n{text}",
                      schema=SCHEMA, effort=SETTINGS.parser_effort)
    if data.get("status") == "needs_clarification":
        return ParseOutcome(question=data.get("clarification_question")
                            or "Which city and dates?")
    return validate(data, today)


def validate(data: dict, today: date) -> ParseOutcome:
    """Turn the model's dict into a StayRequest, or a question if it's unusable.

    Everything here is deterministic and cheap, and it's the one place where a
    plausible-but-wrong model answer gets caught before any search runs.
    """
    city = (data.get("city") or "").strip()
    if not city:
        return ParseOutcome(question="Which city are you staying in?")
    try:
        check_in = date.fromisoformat(data["check_in"])
        check_out = date.fromisoformat(data["check_out"])
    except (KeyError, TypeError, ValueError):
        return ParseOutcome(question="What are your check-in and check-out dates?")
    if check_out <= check_in:
        return ParseOutcome(question=f"Check-out ({check_out}) must be after check-in "
                                     f"({check_in}). Which dates did you mean?")
    if check_in < today:
        return ParseOutcome(question=f"{check_in} is in the past. Which dates did you mean?")

    def keys(values) -> list[str]:
        # Keep only known amenity keys, de-duplicated, order preserved. Local
        # models usually obey the enum, but this makes it a guarantee.
        return list(dict.fromkeys(v for v in values or [] if v in AMENITIES))

    must = keys(data.get("must_haves"))
    budget = data.get("budget_per_night")
    rating = data.get("min_rating")
    return ParseOutcome(StayRequest(
        city=city,
        area=(data.get("area") or "").strip(),
        check_in=check_in.isoformat(),
        check_out=check_out.isoformat(),
        guests=max(int(data.get("guests") or 2), 1),
        rooms=max(int(data.get("rooms") or 1), 1),
        budget_per_night=float(budget) if budget and budget > 0 else None,
        currency=(data.get("currency") or "USD").upper()[:3],
        must_haves=must,
        # Something can't be both required and merely nice.
        nice_to_haves=[k for k in keys(data.get("nice_to_haves")) if k not in must],
        min_rating=min(float(rating), 10.0) if rating else None,
        priority=data.get("priority") if data.get("priority") in PRIORITIES else "balanced",
        notes=(data.get("notes") or "").strip(),
    ))
