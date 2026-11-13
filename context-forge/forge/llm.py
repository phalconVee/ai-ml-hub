"""The single door to Claude. Model, effort and refusal fallback are set here once.

The client is passed in, never created inside the stages. The app passes a
real `anthropic.Anthropic()`; the tests pass a fake that returns canned
responses, so every stage can be tested with no key and no network.
"""

from __future__ import annotations

import json
from typing import Any

from .config import SETTINGS


def make_client():
    """Real client. Reads ANTHROPIC_API_KEY (or an `ant auth login` profile)."""
    import anthropic  # lazy: the tests never import or configure it

    return anthropic.Anthropic()


def ask(client: Any, *, system: str, messages: list, effort: str,
        tools: list | None = None, schema: dict | None = None) -> Any:
    """One Messages API request; returns the raw response object."""
    output_config: dict = {"effort": effort}
    if schema is not None:
        # Structured outputs: the final text is guaranteed to be JSON matching
        # this schema, so we never have to "parse JSON out of prose".
        output_config["format"] = {"type": "json_schema", "schema": schema}
    kwargs: dict = dict(
        model=SETTINGS.model,
        max_tokens=SETTINGS.max_tokens,
        system=system,
        messages=messages,
        output_config=output_config,
        betas=[SETTINGS.fallback_beta],   # beta namespace is needed for `fallbacks`
        fallbacks="default",              # on a refusal, retry on a fallback model
    )
    if tools:
        kwargs["tools"] = tools
    return client.beta.messages.create(**kwargs)


def text_of(response: Any) -> str:
    """All text blocks joined (thinking and tool blocks are skipped)."""
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")


def json_of(response: Any) -> dict:
    """Parse a structured-output response. A refusal has no JSON, so → {}."""
    if getattr(response, "stop_reason", "") == "refusal":
        return {}
    try:
        return json.loads(text_of(response))
    except json.JSONDecodeError:
        return {}
