"""One door to Claude. The reply step and the extraction step both go through
here, so the model, the refusal fallback and the request shape live in
exactly one place.

`client` is always passed in, never created here. In the app it's a real
`anthropic.Anthropic()`; in the tests it's a fake that returns canned
responses, so the whole pipeline runs without an API key or network.
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


def _create(client: Any, *, system: str, messages: list[dict], effort: str,
            schema: dict | None = None):
    """The single API call both helpers below share."""
    output_config: dict = {"effort": effort}
    if schema is not None:
        # Structured outputs: the API guarantees the final text matches `schema`.
        output_config["format"] = {"type": "json_schema", "schema": schema}
    response = client.beta.messages.create(   # beta namespace: needed for `fallbacks`
        model=SETTINGS.model,
        max_tokens=SETTINGS.max_tokens,
        system=system,                         # identical every turn → cache-friendly prefix
        messages=messages,                     # everything that varies goes here
        output_config=output_config,
        betas=[SETTINGS.fallback_beta],
        fallbacks="default",
    )
    # Always read stop_reason before trusting the content.
    if response.stop_reason == "refusal":
        raise RefusedError("Claude declined to answer that.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Claude ran out of room (max_tokens) before finishing.")
    # Thinking blocks come first; only the text blocks are the answer.
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")


def ask_text(client: Any, *, system: str, messages: list[dict], effort: str) -> str:
    """A normal multi-turn chat call. Returns the reply text."""
    return _create(client, system=system, messages=messages, effort=effort).strip()


def ask_json(client: Any, *, system: str, user: str, effort: str, schema: dict) -> dict:
    """A single-turn call whose answer is guaranteed to be JSON matching `schema`."""
    text = _create(client, system=system, messages=[{"role": "user", "content": user}],
                   effort=effort, schema=schema)
    return json.loads(text)
