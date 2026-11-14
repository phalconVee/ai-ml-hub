"""Scripted stand-ins for Claude and a voice.

`FakeClient` mimics `client.beta.messages.create(...)`: each call pops the
next canned response and records the request, so tests check both what the
pipeline *did* and what it *sent*. `ToneEngine` returns real 24 kHz PCM
(a beep whose length depends on the text), so lip-sync is tested on audio.
"""

import json
import math
import struct
from types import SimpleNamespace as NS

from memoir.config import SETTINGS


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def text(t, stop="end_turn"):
    """A reply whose content starts with a thinking block, like the real model's."""
    return NS(stop_reason=stop, content=[NS(type="thinking", thinking="..."), NS(type="text", text=t)])


def as_json(obj, stop="end_turn"):
    return text(json.dumps(obj), stop)


def ent(name, type_="other", description="", aliases=()):
    return {"name": name, "type": type_, "description": description, "aliases": list(aliases)}


def rel(source, predicate, target, exclusive=False, negated=False):
    return {"source": source, "predicate": predicate, "target": target,
            "exclusive": exclusive, "negated": negated}


# A small life story told over three turns.
TURN1 = {"entities": [ent("User", "person", "Software engineer", ["Henry"]),
                      ent("Lagos", "place", "City in Nigeria"),
                      ent("Ada", "person", "The user's sister")],
         "relations": [rel("User", "lives_in", "Lagos", exclusive=True),
                       rel("User", "sister", "Ada")]}
TURN2 = {"entities": [ent("Pottery", "interest", "")],
         "relations": [rel("Ada", "likes", "Pottery")]}
TURN3 = {"entities": [ent("Berlin", "place", "Capital of Germany")],
         "relations": [rel("User", "lives_in", "Berlin", exclusive=True)]}


class ToneEngine:
    """A pretend TTS: 20 ms of 440 Hz per character, then 200 ms of silence."""

    name = "tone"

    def __init__(self):
        self.calls = []

    def synthesize(self, text, voice):
        self.calls.append((text, voice))
        rate = SETTINGS.sample_rate
        n = int(rate * 0.02 * len(text))
        tone = [int(12000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(n)]
        silence = [0] * int(rate * 0.2)
        return struct.pack(f"<{n + len(silence)}h", *tone, *silence)
