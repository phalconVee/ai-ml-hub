"""Demo mode: a stand-in Jev that needs no key and no network.

It reads the REAL request the app built and answers each question with a
canned, plausible value that matches the synthetic sample desk in frames.py
(a mug, a laptop and a book). The values are chosen so every branch of the
decision layer shows up: confident accepts, a "not sure" Noul, a
low-confidence Box, and an absent object whose box gets skipped.

It does not look at the pixels. On a live webcam frame in demo mode the
answers are still the canned ones, and the UI says so.
"""

from sight.frames import SAMPLE_OBJECTS
from sight.jev import JevResult, parse_response

# How sure the fake model is that each object is present, and its box confidence.
PRESENCE = {"mug": 0.93, "laptop": 0.97, "book": 0.81, "phone": 0.06, "keys": 0.04}
BOX_CONFIDENCE = {"mug": 0.88, "laptop": 0.94, "book": 0.48}  # book: low -> review
MAIN_OBJECT = "laptop"


def _choice(criteria: dict) -> dict:
    labels = list(criteria)
    winner = next((l for l in labels if MAIN_OBJECT in l.lower()), labels[0])
    rest = [l for l in labels if l != winner]
    probs = {winner: 0.86, **{l: round(0.14 / len(rest), 3) for l in rest}}
    n, top = len(labels), probs[winner]
    # Jev's documented Choice confidence: (n·p_top − 1) / (n − 1).
    return {"type": "choice", "choice": winner, "probabilities": probs,
            "confidence": round((n * top - 1) / (n - 1), 3)}


def _match(text: str) -> str | None:
    """Which sample object (if any) a question is about."""
    text = text.lower()
    return next((name for name in PRESENCE if name in text), None)


def _noul(instructions: str) -> dict:
    text = instructions.lower()
    if "person" in text or "people" in text or "face" in text:
        return {"type": "noul", "noul": 0.03}   # the sample desk is empty
    name = _match(text)
    # Unknown objects / free-form checks land in the "not sure" band on purpose.
    return {"type": "noul", "noul": PRESENCE[name] if name else 0.55}


def _box(instructions: str) -> dict:
    name = _match(instructions)
    box = SAMPLE_OBJECTS.get(name, {"left": 0.42, "top": 0.35, "right": 0.58, "bottom": 0.6})
    return {"type": "box", "box": dict(box), "confidence": BOX_CONFIDENCE.get(name, 0.31)}


def _score(criteria: list) -> dict:
    n = len(criteria)
    probs = {str(i): 0.0 for i in range(n)}
    probs[str(n - 1)], probs[str(n - 2)] = 0.8, 0.2       # mostly "clear and well lit"
    score = sum(int(k) * v for k, v in probs.items())     # expected level, like Jev
    return {"type": "score", "score": round(score, 2), "probabilities": probs,
            "legend": {str(i): c for i, c in enumerate(criteria)}, "confidence": 0.7}


class DemoJevClient:
    """Same interface as HttpJevClient: ask(request, image_jpeg) -> JevResult."""

    def __init__(self) -> None:
        self.calls: list[dict] = []   # every request seen (handy in tests)

    def ask(self, request: dict, image_jpeg: bytes) -> JevResult:
        self.calls.append(request)
        answers = {}
        for qid, q in request["questions"].items():
            kind = q["type"]
            if kind == "choice":
                answers[qid] = _choice(q["criteria"])
            elif kind == "noul":
                answers[qid] = _noul(q["instructions"])
            elif kind == "box":
                answers[qid] = _box(q["instructions"])
            elif kind == "score":
                answers[qid] = _score(q["criteria"])
        # Go through the same parser as real responses, so demo mode exercises it.
        return parse_response({"model": "demo (canned answers)", "answers": answers,
                               "usage": {"input_tokens": 0, "output_tokens": 0}})
