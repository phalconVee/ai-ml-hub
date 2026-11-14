"""Build the typed questions and the request body. Pure functions, no network.

One camera frame becomes ONE Jev request: the frame goes into `state` once,
and every question (classify, find each object, yes/no checks, image quality)
refers to it by name. Jev reads the state once and answers all questions
against it in parallel, so adding a question costs only that question's
tokens, not another copy of the image.

The request we build looks like this (image sent as a multipart file part):

    {
      "model": "jev-latest",
      "state":  {"frame": "<uuid>"},
      "assets": {"<uuid>": {"type": "image", "part": "frame"}},
      "questions": {
        "classify":      {"type": "choice", "instructions": "...`frame`...", "criteria": {...}},
        "present_mug":   {"type": "noul",   "instructions": "Is there a mug in `frame`?"},
        "box_mug":       {"type": "box",    "instructions": "the mug in `frame`"},
        "check_person":  {"type": "noul",   "instructions": "..."},
        "quality":       {"type": "score",  "instructions": "...", "criteria": [...]}
      }
    }
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Literal

from sight import config

Kind = Literal["choice", "score", "noul", "box"]
# What each question is FOR in this app. The decision layer routes on it.
Role = Literal["classify", "presence", "locate", "check", "quality"]

# The image sits in the state under this key; questions say `frame` to point at it.
STATE_KEY = "frame"
# The multipart part name that carries the JPEG bytes.
IMAGE_PART = "frame"

QUALITY_LEVELS = [
    "Too dark, blurry or blocked to tell what is in it",
    "Usable, but dim, blurry or partly blocked",
    "Clear and well lit",
]


@dataclass(frozen=True)
class Question:
    """One typed question, plus the bookkeeping our app needs around it."""
    id: str        # the key in the request's `questions` map (answers come back under it)
    kind: Kind     # the Jev primitive
    role: Role     # what the app does with the answer
    subject: str   # human-readable: "mug", "Is a person present?", "label set"
    body: dict     # exactly what is sent to the API for this question


@dataclass
class Plan:
    """What the user wants to know about each frame (filled from the sidebar)."""
    labels: dict[str, str] = field(default_factory=dict)  # label -> description
    classify_prompt: str = "What is the main object in `frame`?"
    objects: list[str] = field(default_factory=list)      # things to locate
    checks: list[str] = field(default_factory=list)       # free-form yes/no questions
    quality: bool = True                                   # add the image-quality Score


def slug(text: str) -> str:
    """'Coffee mug!' -> 'coffee_mug'. Question ids must be stable and readable."""
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s or "item"


def parse_labels(text: str) -> dict[str, str]:
    """Sidebar text -> Choice criteria.

    Accepts 'mug, phone, nothing' or one 'label: description' per line.
    A label with no description gets a plain one, so Jev knows what it means.
    """
    labels: dict[str, str] = {}
    for part in re.split(r"[\n,]", text):
        if not part.strip():
            continue
        name, _, desc = part.partition(":")
        name = name.strip()
        if name.lower() in {"nothing", "none"} and not desc.strip():
            desc = "No clear object: an empty scene, or nothing that fits the other options."
        labels[name] = desc.strip() or f"The main object in `{STATE_KEY}` is a {name}."
    return labels


def build_questions(plan: Plan) -> list[Question]:
    """Turn a Plan into a flat list of typed questions (order = display order)."""
    qs: list[Question] = []

    # Classify: ONE Choice over the user's labels. Needs at least two options.
    if len(plan.labels) >= 2:
        qs.append(Question("classify", "choice", "classify", "label set", {
            "type": "choice",
            "instructions": plan.classify_prompt,
            "criteria": dict(plan.labels),
        }))

    # Locate: for each object, a Noul ("is it there?") AND a Box ("where?").
    # A Box always returns a box, even for something absent (docs: there is
    # no "not found" answer), so the Noul is what lets code ignore phantom boxes.
    for obj in plan.objects:
        s = slug(obj)
        qs.append(Question(f"present_{s}", "noul", "presence", obj, {
            "type": "noul",
            "instructions": f"Is there a {obj} visible in `{STATE_KEY}`?",
        }))
        qs.append(Question(f"box_{s}", "box", "locate", obj, {
            "type": "box",
            "instructions": f"the {obj} in `{STATE_KEY}`",
        }))

    # Checks: free-form yes/no statements about the frame.
    for i, check in enumerate(plan.checks):
        qs.append(Question(f"check_{i}_{slug(check)[:40]}", "noul", "check", check, {
            "type": "noul",
            # Point the question at the image unless the user already did.
            "instructions": check if f"`{STATE_KEY}`" in check else f"In `{STATE_KEY}`: {check}",
        }))

    # Quality: an ordered Score. A poor frame makes every other answer shakier,
    # so the decision layer uses this to send everything to review.
    if plan.quality:
        qs.append(Question("quality", "score", "quality", "image quality", {
            "type": "score",
            "instructions": f"How clearly can you see what is in `{STATE_KEY}`?",
            "criteria": list(QUALITY_LEVELS),
        }))
    return qs


def build_request(questions: list[Question], model: str = config.MODEL,
                  image_ref: str | None = None) -> dict:
    """The JSON body (the `request` part of the multipart upload).

    The image itself is NOT in here: `assets` says "the picture for this
    reference is in the multipart part named `frame`". A fresh UUID per
    request is the docs' advice: references are matched as plain text
    anywhere in the request, so they must not occur by accident.
    """
    if not questions:
        raise ValueError("Ask at least one question.")
    ref = image_ref or str(uuid.uuid4())
    return {
        "model": model,
        "state": {STATE_KEY: ref},
        "assets": {ref: {"type": "image", "part": IMAGE_PART}},
        "questions": {q.id: q.body for q in questions},
    }
