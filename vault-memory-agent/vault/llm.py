"""The single place that calls Claude.

Model, effort, the tool list and the refusal-fallback opt-in are set here
once, so the agent loop reads as pure control flow.
"""

from __future__ import annotations

from .config import SETTINGS

# Opt in to server-side refusal fallbacks: if a safety classifier declines on
# the main model, the API re-runs the request on a fallback model inside the
# same call instead of handing back an empty refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def make_client():
    import anthropic
    return anthropic.Anthropic()   # reads ANTHROPIC_API_KEY (or an `ant auth` profile)


def create(client, *, system: str, messages: list, tools: list[dict]):
    """One model call. Thinking is always on for Opus 5.5; effort sets its depth."""
    return client.beta.messages.create(
        model=SETTINGS.model,
        max_tokens=SETTINGS.max_tokens,
        system=system,
        messages=messages,
        tools=tools,
        output_config={"effort": SETTINGS.effort},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )


def text_of(response) -> str:
    """Join the visible text blocks (skipping thinking, tool_use, fallback...)."""
    return "".join(b.text for b in response.content if b.type == "text").strip()
