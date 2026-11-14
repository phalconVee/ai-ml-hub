"""The thin Jev client: send one request, get typed answers back.

Three parts, each small enough to read in one go:

1. Answer types: plain dataclasses, one per Jev primitive.
2. parse_response(): raw JSON -> {question_id: Answer}. Pure, easy to test.
3. JevClient: a Protocol (any object with an `ask()` method works), and
   HttpJevClient, the real one: multipart POST with httpx + simple retries.

Why plain httpx instead of the official `typesafe-sdk`? The SDK is great
(typed questions, built-in retries, `Image(...)` objects). Here, though, the
point is to SEE the wire format: the multipart upload, the images header and
the raw answers. Swapping in the SDK is a ~15-line change inside
HttpJevClient.ask(); nothing else in the app would notice.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from sight import config
from sight.questions import IMAGE_PART


# ── 1. Answer types ─────────────────────────────────────────────────────────
@dataclass
class ChoiceAnswer:
    choice: str                      # the winning label
    probabilities: dict[str, float]  # one probability per label (sums to ~1)
    confidence: float                # 1 = all mass on one label, 0 = evenly spread

    @property
    def top_probability(self) -> float:
        """The probability of the label we'd act on. The docs recommend
        thresholding THIS for a Choice: 'act above 0.9' then reads as
        'act when the chance of the wrong label is under 10%'."""
        return self.probabilities.get(self.choice, 0.0)


@dataclass
class ScoreAnswer:
    score: float                     # position on the levels, 0..n-1 (expected level)
    probabilities: dict[int, float]  # one per level index
    confidence: float
    legend: dict[int, str] = field(default_factory=dict)


@dataclass
class NoulAnswer:
    noul: float                      # probability the statement is TRUE (0..1)
    # Nouls carry no `confidence`: the probability already says how sure it is.


@dataclass
class BoxAnswer:
    top: float                       # all four edges are fractions of the image,
    bottom: float                    # measured from the TOP-LEFT corner:
    left: float                      # left/right of the width,
    right: float                     # top/bottom of the height.
    confidence: float | None         # can be null (the API reference shows one)


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer | BoxAnswer


@dataclass
class JevResult:
    model: str                       # the VERSIONED model that answered (e.g. jev-1.14.0)
    answers: dict[str, Answer]
    input_tokens: int = 0
    output_tokens: int = 0


# ── 2. Parsing ──────────────────────────────────────────────────────────────
def parse_answer(raw: dict) -> Answer:
    """One raw answer dict -> the matching dataclass, chosen by its `type`."""
    kind = raw.get("type")
    if kind == "choice":
        return ChoiceAnswer(raw["choice"], dict(raw["probabilities"]), raw["confidence"])
    if kind == "score":
        # JSON object keys are strings ("0", "1", ...); levels are ints.
        probs = {int(k): v for k, v in raw.get("probabilities", {}).items()}
        legend = {int(k): v for k, v in raw.get("legend", {}).items()}
        return ScoreAnswer(raw["score"], probs, raw["confidence"], legend)
    if kind == "noul":
        return NoulAnswer(raw["noul"])
    if kind == "box":
        b = raw["box"]
        return BoxAnswer(b["top"], b["bottom"], b["left"], b["right"], raw.get("confidence"))
    raise ValueError(f"Unknown answer type: {kind!r}")


def parse_response(data: dict) -> JevResult:
    usage = data.get("usage") or {}
    return JevResult(
        model=data.get("model", "?"),
        answers={qid: parse_answer(a) for qid, a in data["answers"].items()},
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )


# ── 3. Clients ──────────────────────────────────────────────────────────────
class JevClient(Protocol):
    """Anything with this method can answer questions: the HTTP client, the
    demo client, or a five-line fake in a test."""

    def ask(self, request: dict, image_jpeg: bytes) -> JevResult: ...


class JevError(RuntimeError):
    """A non-retryable API error, with the API's own explanation attached."""


class HttpJevClient:
    def __init__(self, api_key: str | None = None, *,
                 http: httpx.Client | None = None, sleep=time.sleep):
        self.api_key = api_key or os.environ.get(config.API_KEY_ENV, "")
        if not self.api_key:
            raise JevError(f"Set {config.API_KEY_ENV} in .env (or turn on demo mode).")
        # `http` and `sleep` can be swapped in tests (httpx.MockTransport, no waiting).
        self.http = http or httpx.Client(timeout=config.TIMEOUT_SECONDS)
        self.sleep = sleep

    def ask(self, request: dict, image_jpeg: bytes) -> JevResult:
        headers = {"Authorization": f"Bearer {self.api_key}", **config.IMAGES_HEADER}
        # Multipart, as the docs recommend for images: the JSON body goes FIRST,
        # in a part named `request`; the JPEG follows as its own file part, whose
        # name matches `assets[ref].part`. (Base64-in-JSON also works but makes
        # the upload ~33% bigger.) httpx sends `files` in insertion order.
        files = {
            "request": (None, json.dumps(request), "application/json"),
            IMAGE_PART: ("frame.jpg", image_jpeg, "image/jpeg"),
        }
        for attempt in range(1, config.MAX_ATTEMPTS + 1):
            resp = self.http.post(config.JEV_URL, headers=headers, files=files)
            if resp.status_code in config.RETRY_STATUSES and attempt < config.MAX_ATTEMPTS:
                # Honour retry-after when present, else back off 1s, 2s, 4s...
                wait = float(resp.headers.get("retry-after", 2 ** (attempt - 1)))
                self.sleep(wait)
                continue
            if resp.status_code != 200:
                raise JevError(f"Jev returned {resp.status_code}: {resp.text[:500]}")
            return parse_response(resp.json())
        raise JevError("unreachable")  # the loop always returns or raises
