"""One door to Claude. All three agents call `ask_json`, so the model, the
refusal fallback and the request shape are set in exactly one place.

`client` is always passed in, never created here. In the app it's a real
`anthropic.Anthropic()`; in the tests it's a fake that returns canned
responses, so the whole crew runs without an API key.
"""

from __future__ import annotations

import json
from typing import Any

from .config import SETTINGS


class RefusedError(RuntimeError):
    """The model (and its server-side fallback) declined the request."""


def make_client():
    """Real client. Reads ANTHROPIC_API_KEY (or an `ant auth login` profile)."""
    import anthropic  # lazy: importing this module never needs credentials

    return anthropic.Anthropic()


def ask_json(client: Any, *, who: str, system: str, user: str, effort: str,
             schema: dict) -> dict:
    """Send one single-turn request and return the reply as a dict.

    Every agent in this crew answers in JSON, so we always use structured
    outputs: `output_config.format` makes the API guarantee the final text
    matches `schema`, and json.loads() can't fail on a well-formed reply.
    """
    response = client.beta.messages.create(  # beta namespace: needed for `fallbacks`
        model=SETTINGS.model,
        max_tokens=SETTINGS.max_tokens,
        system=system,                                   # identical on every run → cacheable
        messages=[{"role": "user", "content": user}],    # everything that varies goes here
        output_config={"effort": effort,
                       "format": {"type": "json_schema", "schema": schema}},
        betas=[SETTINGS.fallback_beta],
        fallbacks="default",
    )
    # Always look at stop_reason before trusting the content.
    if response.stop_reason == "refusal":
        raise RefusedError(f"The {who} declined this content.")
    if response.stop_reason == "max_tokens":
        # A JSON reply cut off mid-way would not parse. Say so plainly.
        raise RuntimeError(f"The {who} ran out of room (max_tokens). Try a shorter episode.")
    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    return json.loads(text)
