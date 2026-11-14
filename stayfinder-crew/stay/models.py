"""The typed objects the agents hand to each other.

    text ─parser─► StayRequest ─scout─► [HotelOption] ─analyst─► Analysis ─advisor─► Advice

Plain dataclasses, no framework. Each agent takes one of these and returns
the next, which is what makes every step testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

# A fixed amenity vocabulary. The parser maps "free wifi", "WiFi", "internet"
# → "wifi", and every source reports amenities with the same keys, so
# "does this hotel have what I asked for?" becomes a set comparison.
AMENITIES = [
    "wifi", "breakfast", "parking", "pool", "gym", "spa", "air_conditioning",
    "pet_friendly", "wheelchair_accessible", "restaurant", "kitchen",
    "airport_shuttle", "family_rooms", "non_smoking", "workspace",
]

PRIORITIES = ["value", "location", "quality", "balanced"]


@dataclass
class StayRequest:
    city: str
    check_in: str                    # ISO date "YYYY-MM-DD"
    check_out: str
    guests: int = 2
    rooms: int = 1
    area: str = ""                   # neighbourhood / landmark to be near ("" = centre)
    budget_per_night: float | None = None
    currency: str = "USD"
    must_haves: list[str] = field(default_factory=list)      # AMENITIES keys
    nice_to_haves: list[str] = field(default_factory=list)   # AMENITIES keys
    min_rating: float | None = None  # on a 0-10 scale
    priority: str = "balanced"       # one of PRIORITIES
    notes: str = ""                  # anything else, in the traveller's words

    @property
    def nights(self) -> int:
        return (date.fromisoformat(self.check_out) - date.fromisoformat(self.check_in)).days

    @property
    def target(self) -> str:
        """The place distances are measured from."""
        return f"{self.area}, {self.city}" if self.area else self.city


@dataclass
class HotelOption:
    id: str                          # "H1", "H2"… set by the source; cited by the advisor
    name: str
    source: str                      # "web" | "osm" | "demo"
    kind: str = "hotel"              # hotel | guest_house | hostel | motel | apartment
    address: str = ""
    lat: float | None = None
    lon: float | None = None
    distance_km: float | None = None # from StayRequest.target
    price_per_night: float | None = None
    currency: str = ""
    rating: float | None = None      # normalised to 0-10
    review_count: int | None = None
    stars: int | None = None         # official class, 1-5
    amenities: list[str] = field(default_factory=list)  # known YES
    lacks: list[str] = field(default_factory=list)      # known NO
    source_url: str = ""
    estimated: bool = False          # price came from a web page, not a live quote
    caveat: str = ""
    # ── filled in by the analyst ──
    score: float | None = None       # 0-100 fit score, higher = better
    parts: dict = field(default_factory=dict)            # per-factor scores 0-1
    badges: list[str] = field(default_factory=list)
    must_have_status: dict = field(default_factory=dict) # amenity → "yes"|"no"|"unknown"
    over_budget: bool = False
