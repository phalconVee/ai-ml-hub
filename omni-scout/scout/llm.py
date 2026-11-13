"""A thin wrapper around the Anthropic SDK: the only file that talks to Claude.

Why wrap it at all? Two reasons:
  1. Every call shares the same settings (refusal fallback, effort, refusal
     check), written once here instead of in every agent.
  2. Tests swap this class for a scripted fake, so the whole crew can be
     exercised offline without an API key (see tests/test_pipeline.py).

Three kinds of call are all the agents need:
  structured()  ->  one request, answer validated against a Pydantic model (planner)
  step()        ->  one request with tools, returns the raw response (scout loop)
  stream_text() ->  one streamed request, text arrives piece by piece (synthesizer)
"""

from typing import Any, Awaitable, Callable, TypeVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from scout.config import settings

T = TypeVar("T", bound=BaseModel)


class RefusalError(RuntimeError):
    """The model (and its fallback) declined the request."""


def _check(response: Any) -> Any:
    # A refusal is HTTP 200 with stop_reason "refusal", not an exception, so it
    # must be checked explicitly before reading the content.
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise RefusalError(f"Request declined: {getattr(details, 'category', None) or 'no category'}")
    return response


class LLM:
    def __init__(self, client: AsyncAnthropic | None = None):
        # With no arguments, the SDK reads ANTHROPIC_API_KEY (or an `ant auth login` profile).
        self.client = client or AsyncAnthropic()

    def _common(self, model: str, effort: str) -> dict:
        """Arguments every request shares."""
        return {
            "model": model,
            # Effort sets how hard Claude thinks (and how many tokens it spends).
            # Thinking itself is adaptive by default on Opus 5.5: no `thinking` arg needed.
            "output_config": {"effort": effort},
            # If a safety classifier declines, the API retries on a fallback
            # model inside the same call instead of just stopping.
            "betas": [settings.fallback_beta],
            "fallbacks": "default",
        }

    async def structured(self, *, model: str, effort: str, system: str, prompt: str, schema: type[T]) -> T:
        """Ask once; get back a validated instance of `schema`."""
        response = await self.client.beta.messages.parse(
            **self._common(model, effort),
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,  # the API constrains the JSON; the SDK parses it
        )
        return _check(response).parsed_output

    async def step(self, *, model: str, effort: str, system: str, messages: list, tools: list[dict]) -> Any:
        """One turn of a tool-using conversation. The caller runs the loop."""
        response = await self.client.beta.messages.create(
            **self._common(model, effort),
            max_tokens=16000,
            system=system,
            messages=messages,
            tools=tools,
        )
        return _check(response)

    async def stream_text(
        self, *, model: str, effort: str, system: str, prompt: str,
        on_text: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> str:
        """Stream a long answer, calling `on_text(chunk)` as text arrives (for the UI)."""
        async with self.client.beta.messages.stream(
            **self._common(model, effort),
            max_tokens=64000,  # streaming, so a big ceiling won't hit HTTP timeouts
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for chunk in stream.text_stream:
                if on_text:
                    maybe = on_text(chunk)
                    if maybe is not None:
                        await maybe
            final = await stream.get_final_message()
        _check(final)
        return "".join(b.text for b in final.content if b.type == "text")
