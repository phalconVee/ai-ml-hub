"""A fake Claude client: same call shape as `client.beta.messages.create`,
canned answers chosen by which system prompt is calling.

- recall planner  → keywords from `planner_keywords` (or the message itself)
- extractor       → facts produced by `extract(user_text, prompt)` (a test hook)
- answerer        → echoes which memory ids it was shown, so tests can assert
                    that a memory really reached the prompt
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

from recall import companion, extractor, recall_planner


def response(text: str, stop: str = "end_turn"):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop)


def fact(subject="user", predicate="likes", obj="tea", statement=None, *, importance=3,
         exclusive=False, category="preference", object_is_entity=False, kind="thing",
         replaces=(), occurred_at=None):
    """A fact dict exactly as the extractor's JSON schema describes it."""
    return {"subject": subject, "subject_kind": "user" if subject == "user" else "person",
            "predicate": predicate, "object": obj, "object_is_entity": object_is_entity,
            "object_kind": kind, "statement": statement or f"The user {predicate} {obj}.",
            "category": category, "importance": importance, "exclusive": exclusive,
            "occurred_at": occurred_at, "replaces": list(replaces)}


class FakeMessages:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kw):
        o = self.owner
        o.calls.append(kw)
        if o.fail:
            raise RuntimeError("network down")
        system = kw["system"]
        user = kw["messages"][-1]["content"]
        if system == recall_planner.SYSTEM:
            msg = user.split("Latest user message:\n", 1)[-1]
            return response(json.dumps({"keywords": o.planner_keywords.get(msg, ""),
                                        "entities": o.planner_entities.get(msg, []),
                                        "about_the_past": "before" in msg.lower()}))
        if system == extractor.SYSTEM:
            text = user.split("User: ", 1)[-1].split("\nAssistant: ", 1)[0]
            return response(json.dumps({"facts": o.extract(text, user)}))
        if system.startswith(companion.SYSTEM[:40]):
            if o.refuse:
                return SimpleNamespace(content=[], stop_reason="refusal")
            ids = re.findall(r"\[M(\d+)\]", system)
            return response(f"(answer using memories {','.join(ids) or 'none'})")
        raise AssertionError("unknown caller")


class FakeClient:
    def __init__(self, extract=None, planner_keywords=None, planner_entities=None):
        self.calls: list[dict] = []
        self.extract = extract or (lambda text, prompt: [])
        self.planner_keywords = planner_keywords or {}
        self.planner_entities = planner_entities or {}
        self.fail = False
        self.refuse = False
        self.beta = SimpleNamespace(messages=FakeMessages(self))

    def systems(self):
        return [c["system"] for c in self.calls]
