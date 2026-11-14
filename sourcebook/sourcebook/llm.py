"""The single place that talks to Claude.

Every call in the app goes through `create()`, so model, effort and the
refusal-fallback opt-in are set once. `create_json()` adds structured outputs
(the response is guaranteed to match a JSON schema) for the helper calls.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .config import SETTINGS

# Opt in to server-side refusal fallbacks: if a safety classifier declines on
# the main model, the API re-runs the request on a fallback model inside the
# same call instead of handing us an empty refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def make_client():
    import anthropic
    return anthropic.Anthropic()   # reads ANTHROPIC_API_KEY (or an `ant auth` profile)


def create(client, *, system: str, messages: list[dict], effort: str,
           output_format: Optional[dict] = None, max_tokens: int = SETTINGS.max_tokens):
    output_config: dict[str, Any] = {"effort": effort}
    if output_format:
        output_config["format"] = output_format
    return client.beta.messages.create(
        model=SETTINGS.model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        output_config=output_config,
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )


def text_of(response) -> str:
    """Join the text blocks of a response (skipping thinking blocks etc.)."""
    return "".join(b.text for b in response.content if b.type == "text")


def create_json(client, *, system: str, prompt: str, schema: dict,
                effort: str = SETTINGS.helper_effort) -> dict:
    """One structured-output call: the reply is JSON matching `schema`."""
    response = create(
        client, system=system, effort=effort,
        messages=[{"role": "user", "content": prompt}],
        output_format={"type": "json_schema", "schema": schema},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined this request.")
    return json.loads(text_of(response))
