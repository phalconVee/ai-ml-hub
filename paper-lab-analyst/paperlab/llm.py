"""One door to Claude. Every agent calls `ask_claude`, so the model, effort and
refusal fallback are set in exactly one place.

`client` is always passed in, never created deep inside an agent. In the app
it's a real `anthropic.Anthropic()`; in the tests it's a fake that returns
canned responses, so the whole workflow runs without an API key.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from .config import SETTINGS


class AnalysisError(RuntimeError):
    """A Claude call ended in a way we can't use (refusal, cut off, bad JSON)."""


def make_client():
    """Real client. Reads ANTHROPIC_API_KEY (or an `ant auth login` profile)."""
    import anthropic  # imported lazily so the tests never need it configured

    return anthropic.Anthropic()


def ask_claude(client: Any, *, system: str, messages: list, tools: list | None = None,
               schema: dict | None = None, tool_choice: dict | None = None) -> Any:
    """Send one Messages API request and return the raw response."""
    output_config: dict = {"effort": SETTINGS.effort}
    if schema is not None:
        # Structured outputs: the final text block is guaranteed to be JSON that
        # matches this schema. Tool calls (if any) still happen before it.
        output_config["format"] = {"type": "json_schema", "schema": schema}

    kwargs: dict = {}
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    return client.beta.messages.create(  # beta namespace: needed for `fallbacks`
        model=SETTINGS.model,
        max_tokens=SETTINGS.max_tokens,
        system=system,
        messages=messages,
        output_config=output_config,
        betas=[SETTINGS.fallback_beta],
        fallbacks="default",  # on a safety refusal, re-run on a fallback model
        **kwargs,
    )


def text_of(response: Any) -> str:
    """Join every text block (thinking and tool blocks are skipped)."""
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")


def check_stop(response: Any) -> None:
    """Raise a readable error for stop reasons that mean "no usable answer"."""
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        raise AnalysisError(f"Claude declined this request (category: {category or 'unknown'}).")
    if response.stop_reason == "max_tokens":
        raise AnalysisError("Claude's answer was cut off (max_tokens). Raise it in config.py.")


def json_of(response: Any) -> dict:
    """Parse a structured-output response into a dict."""
    check_stop(response)
    try:
        return json.loads(text_of(response))
    except json.JSONDecodeError as exc:  # shouldn't happen with structured outputs
        raise AnalysisError(f"Expected JSON from Claude, got: {text_of(response)[:200]}") from exc


def image_block(png: bytes) -> dict:
    """A PNG as a Messages API image content block (base64, no newlines)."""
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png",
                   "data": base64.standard_b64encode(png).decode("ascii")},
    }
