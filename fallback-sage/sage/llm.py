"""One door to Claude. Both agents call `ask_claude`, so model, effort and the
refusal fallback are set in exactly one place.

`client` is passed in rather than created here. In the app it's a real
`anthropic.Anthropic()`; in the tests it's a fake that returns canned
responses, so the agent logic can be checked without an API key.
"""

from __future__ import annotations

from typing import Any

from .config import SETTINGS


def make_client():
    """Real client. Reads ANTHROPIC_API_KEY (or an `ant auth login` profile)."""
    import anthropic  # imported lazily so the tests never need it configured

    return anthropic.Anthropic()


def ask_claude(client: Any, *, system: str, messages: list, tools: list,
               output_format: dict | None = None) -> Any:
    """Send one Messages API request and return the raw response."""
    output_config: dict = {"effort": SETTINGS.effort}
    if output_format is not None:
        # Structured outputs: the final text block is guaranteed to be JSON
        # matching this schema. Tool calls still work normally before it.
        output_config["format"] = output_format
    return client.beta.messages.create(  # beta namespace: needed for `fallbacks`
        model=SETTINGS.model,
        max_tokens=SETTINGS.max_tokens,
        system=system,
        messages=messages,
        tools=tools,
        output_config=output_config,
        betas=[SETTINGS.fallback_beta],
        fallbacks="default",  # on a safety refusal, re-run on a fallback model
    )


def text_of(response: Any) -> str:
    """Join every text block in a response (thinking/tool blocks are skipped)."""
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
