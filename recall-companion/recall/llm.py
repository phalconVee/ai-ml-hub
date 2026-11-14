"""The single door to Claude. Model, effort and refusal fallback are set once.

The client is passed in, never created inside the memory code. The app passes
a real `anthropic.Anthropic()`; the tests pass a fake that returns canned
responses, so every step can be tested with no key and no network.
"""

from __future__ import annotations

import json
from typing import Any

from .config import SETTINGS


def make_client():
    """Real client. Reads ANTHROPIC_API_KEY (or an `ant auth login` profile)."""
    import anthropic  # lazy: the offline tests never import or configure it

    return anthropic.Anthropic()


def ask(client: Any, *, system: str, messages: list, effort: str,
        schema: dict | None = None) -> Any:
    """One Messages API request; returns the raw response object."""
    output_config: dict = {"effort": effort}
    if schema is not None:
        # Structured outputs: the reply is guaranteed to be JSON matching this
        # schema, so the memory step never has to "parse JSON out of prose".
        output_config["format"] = {"type": "json_schema", "schema": schema}
    return client.beta.messages.create(
        model=SETTINGS.model,
        max_tokens=SETTINGS.max_tokens,
        system=system,
        messages=messages,
        output_config=output_config,
        betas=[SETTINGS.fallback_beta],   # the beta namespace is needed for `fallbacks`
        fallbacks="default",              # on a refusal, retry on a fallback model
    )


def text_of(response: Any) -> str:
    """All text blocks joined (thinking blocks are skipped)."""
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")


def json_of(response: Any) -> dict:
    """Parse a structured-output reply. A refusal carries no JSON, so → {}."""
    if getattr(response, "stop_reason", "") == "refusal":
        return {}
    try:
        return json.loads(text_of(response))
    except json.JSONDecodeError:
        return {}
