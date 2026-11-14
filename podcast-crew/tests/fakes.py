"""Scripted stand-ins for Claude, a web page and a voice engine.

`FakeClient` mimics `client.beta.messages.create(...)`: each call pops the
next canned response and records the request, so tests can check both what
the crew *did* and what it *sent*. `ToneEngine` is a voice engine that
returns a short beep whose length depends on the text: real audio bytes in
the real format, with no TTS installed.
"""

import json
import math
import struct
from types import SimpleNamespace as NS

from podcast.config import SETTINGS


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def reply(obj, stop="end_turn"):
    return NS(stop_reason=stop, content=[NS(type="text", text=json.dumps(obj))])


OUTLINE = {
    "title": "Why Bees Dance",
    "hook": "Honeybees give each other directions with a dance.",
    "summary": "Forager bees use a waggle dance to tell the hive where food is.",
    "key_points": [
        {"point": "The waggle run points the way", "detail": "Its angle to vertical matches the angle to the sun."},
        {"point": "Duration encodes distance", "detail": "About one second of waggling per kilometre."},
        {"point": "Not every bee follows", "detail": "Recruits often need several dances."},
    ],
    "listener_questions": ["What happens on a cloudy day?"],
    "takeaway": "A tiny brain can run a map language.",
}


def lines(n=8, words=12):
    """n alternating lines, each `words` words long."""
    return [{"speaker": "AB"[i % 2], "text": " ".join(["bee"] * (words - 1)) + f" {i}."}
            for i in range(n)]


def draft(n=8, words=12, **extra):
    return reply({"title": "Why Bees Dance", "lines": lines(n, words), **extra})


def edited(n=8, words=12, changes=("Tightened the opening.",)):
    return reply({"title": "Why Bees Dance", "lines": lines(n, words), "changes": list(changes)})


ARTICLE_HTML = """<html><head><title>The Bee Dance | Nature Blog</title>
<script>var tracking = "do not read me";</script><style>p{color:red}</style></head>
<body><nav><a>Home</a> <a>About us and our long menu</a></nav>
<header><p>Subscribe to our newsletter for weekly updates today</p></header>
<article><h1>How honeybees give directions</h1>
<p>Forager bees perform a waggle dance &amp; other bees watch closely in the dark hive.</p>
<p>The angle of the waggle run matches the angle between the food and the sun.</p>
<p>Share</p>
<ul><li>Each second of waggling means roughly one kilometre of flight.</li></ul>
<p>The angle of the waggle run matches the angle between the food and the sun.</p>
""" + "<p>" + "More detail about bees and flowers and the hive. " * 6 + "</p>" + """
</article><footer><p>Copyright 2026 the nature blog all rights reserved</p></footer></body></html>"""


class ToneEngine:
    """A pretend TTS: 10 ms of a 440 Hz tone per character."""

    name = "tone"

    def __init__(self):
        self.calls = []

    def voices(self):
        return ["low", "high"]

    def synthesize(self, text, voice):
        self.calls.append((text, voice))
        n = SETTINGS.sample_rate // 100 * len(text)
        return b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / SETTINGS.sample_rate)))
                        for i in range(n))
