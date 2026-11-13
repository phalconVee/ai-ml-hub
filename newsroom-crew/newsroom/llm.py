"""One door to Claude. All three agents call `ask_claude`, so the model, the
refusal fallback and the request shape are set in exactly one place.

`client` is passed in, never created here. In the app it's a real
`anthropic.Anthropic()`; in the tests it's a fake that returns canned
responses, so the whole desk can be checked without an API key.
"""

from __future__ import annotations

from typing import Any

from .config import SETTINGS


class RefusedError(RuntimeError):
    """The model (and its fallback) declined the request."""


def make_client():
    """Real client. Reads ANTHROPIC_API_KEY (or an `ant auth login` profile)."""
    import anthropic  # imported lazily so the tests never need it configured

    return anthropic.Anthropic()


def ask_claude(client: Any, *, system: str, messages: list, effort: str,
               tools: list | None = None, schema: dict | None = None) -> Any:
    """Send one Messages API request and return the raw response.

    tools  : server tools (web search / fetch) for the reporter, none for the others
    schema : a JSON schema; when given, the final text is guaranteed to match it
             (structured outputs), so we can json.loads() it without guesswork
    """
    output_config: dict = {"effort": effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    kwargs: dict = {}
    if tools:
        kwargs["tools"] = tools
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
    """Join every text block in a response (thinking/tool blocks are skipped)."""
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")


def check_refusal(response: Any, who: str) -> None:
    """Always look at stop_reason before trusting content."""
    if response.stop_reason == "refusal":
        raise RefusedError(f"The {who} declined this topic.")
