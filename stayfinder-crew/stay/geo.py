"""Maps, for free: geocoding and distances via OpenStreetMap.

  Geocoder.locate("Alfama, Lisbon") -> (lat, lon) | None    (Nominatim)
  overpass(query) -> dict                                    (Overpass API)
  km(a, b) -> great-circle distance in kilometres            (pure maths)

Both OSM services are free and keyless, with fair-use rules we follow:
a descriptive User-Agent, at most one Nominatim request per second, and a
small in-memory cache so the same place is never looked up twice.
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from typing import Callable

from .config import SETTINGS


def km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Haversine distance. Accurate to ~0.5%, which is plenty for 'how far'."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def http_get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": SETTINGS.user_agent})
    with urllib.request.urlopen(req, timeout=SETTINGS.http_timeout_s) as resp:
        return json.loads(resp.read())


def http_post_form_json(url: str, form: dict) -> object:
    req = urllib.request.Request(url, data=urllib.parse.urlencode(form).encode(),
                                 headers={"User-Agent": SETTINGS.user_agent})
    with urllib.request.urlopen(req, timeout=SETTINGS.http_timeout_s) as resp:
        return json.loads(resp.read())


class Geocoder:
    """Nominatim lookups with a cache and a polite pause between requests.

    `get` is injectable (tests pass a dict-backed fake) and `sleep` too, so
    tests don't wait a second per lookup.
    """

    def __init__(self, get: Callable = http_get_json, sleep: Callable = time.sleep):
        self.get, self.sleep = get, sleep
        self.cache: dict[str, tuple[float, float] | None] = {}
        self._last = 0.0

    def locate(self, place: str) -> tuple[float, float] | None:
        key = place.strip().lower()
        if key in self.cache:
            return self.cache[key]
        wait = SETTINGS.geocode_pause_s - (time.monotonic() - self._last)
        if wait > 0:
            self.sleep(wait)
        url = SETTINGS.nominatim_url + "?" + urllib.parse.urlencode(
            {"q": place, "format": "jsonv2", "limit": 1})
        try:
            hits = self.get(url)
        except OSError:
            hits = []  # offline or rate-limited: no coordinates, not a crash
        self._last = time.monotonic()
        point = (float(hits[0]["lat"]), float(hits[0]["lon"])) if hits else None
        self.cache[key] = point
        return point


class NoGeocoder:
    """Stand-in that never looks anything up (demo mode, offline tests)."""

    def locate(self, place: str) -> None:
        return None
