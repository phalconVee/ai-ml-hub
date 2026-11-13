"""One door to Claude. Every agent calls `ask_claude`, so the model, the
refusal fallback and the refusal check live in exactly one place.

`client` is always passed in rather than created here. In the app it's a real
`anthropic.Anthropic()`; in the tests it's a fake that returns canned
responses, so the agent logic is checked without an API key.
"""

from __future__ import annotations

import json
from typing import Any

from .config import SETTINGS


class ClaudeRefused(RuntimeError):
    """The request was declined by a safety classifier (stop_reason='refusal')."""


def make_client():
    """Real client. Reads ANTHROPIC_API_KEY (or an `ant auth login` profile)."""
    import anthropic  # imported lazily so the tests never need it configured

    return anthropic.Anthropic()


def ask_claude(client: Any, *, system: str, messages: list, effort: str,
               tools: list | None = None, schema: dict | None = None) -> Any:
    """Send one Messages API request and return the raw response.

    `schema`, when given, switches on structured outputs: the final text block
    is guaranteed to be JSON matching it, so parsing can't break on prose.
    """
    output_config: dict = {"effort": effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    kwargs: dict = dict(
        model=SETTINGS.model,
        max_tokens=SETTINGS.max_tokens,
        system=system,
        messages=messages,
        output_config=output_config,
        betas=[SETTINGS.fallback_beta],  # beta namespace: needed for `fallbacks`
        fallbacks="default",  # on a safety refusal, re-run on a fallback model
    )
    if tools:
        kwargs["tools"] = tools
    response = client.beta.messages.create(**kwargs)
    # Check stop_reason before trusting content: a refusal still returns HTTP 200.
    if response.stop_reason == "refusal":
        raise ClaudeRefused("Claude declined this request.")
    return response


def text_of(response: Any) -> str:
    """Join every text block in a response (thinking/tool blocks are skipped)."""
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")


def json_of(response: Any) -> dict:
    """The JSON object a structured-output response carries."""
    return json.loads(text_of(response))
