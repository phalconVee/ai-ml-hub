"""The one door to Claude. All three agents go through `call_claude`, so the
model, effort and refusal fallback live in exactly one place.

The client is always passed in. In the app it's a real `anthropic.Anthropic()`;
in the tests it's a fake that replays canned responses, so every agent can be
tested without an API key or network.
"""

from __future__ import annotations

from typing import Any

from .config import SETTINGS


def make_client():
    """A real client. Reads ANTHROPIC_API_KEY (or an `ant auth login` profile)."""
    import anthropic  # lazy import: the offline tests never construct one

    return anthropic.Anthropic()


def call_claude(client: Any, *, system: str, messages: list, tools: list | None = None,
                tool_choice: dict | None = None) -> Any:
    """Send one Messages API request and return the raw response object."""
    kwargs: dict = {}
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        # Only "auto" or "none" here: Opus 5.5 rejects forced tool choice.
        kwargs["tool_choice"] = tool_choice
    return client.beta.messages.create(   # beta namespace: needed for `fallbacks`
        model=SETTINGS.model,
        max_tokens=SETTINGS.max_tokens,
        system=system,
        messages=messages,
        output_config={"effort": SETTINGS.effort},
        betas=[SETTINGS.fallback_beta],
        fallbacks="default",               # on a safety refusal, retry on a fallback model
        **kwargs,
    )


def text_of(response: Any) -> str:
    """Join the text blocks of a response (thinking and tool blocks skipped)."""
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
