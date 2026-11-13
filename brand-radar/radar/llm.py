"""The only file that talks to Claude.

Why a wrapper?
  1. Every call shares the same settings (effort, refusal fallback, refusal
     check), written once here instead of in every agent.
  2. Tests replace this class with a scripted fake, so the whole pipeline runs
     offline with no API key (see tests/test_pipeline.py).

The agents need exactly three kinds of call:
  structured()   one request, reply validated against a Pydantic model   (analysts)
  stream_text()  one streamed request, text arrives chunk by chunk       (strategist)
  web_search()   one request with Claude's server-side web search tool   (optional sweep)
"""

from typing import Any, Callable, TypeVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from radar.config import settings

T = TypeVar("T", bound=BaseModel)


class RefusalError(RuntimeError):
    """The model (and its fallback) declined the request."""


def _check(response: Any) -> Any:
    # A refusal arrives as HTTP 200 with stop_reason "refusal", not as an
    # exception, so it has to be checked before reading the content.
    if response.stop_reason == "refusal":
        raise RefusalError("Claude declined this request.")
    return response


class LLM:
    def __init__(self, client: AsyncAnthropic | None = None):
        # No arguments: the SDK reads ANTHROPIC_API_KEY from the environment.
        self.client = client or AsyncAnthropic()

    @staticmethod
    def _common(model: str, effort: str) -> dict:
        return {
            "model": model,
            # Effort = how hard Claude thinks (and how many tokens it spends).
            # Thinking is adaptive by default on current models: no `thinking` arg.
            "output_config": {"effort": effort},
            # If a safety classifier declines, retry on a fallback model in the
            # same call rather than failing the run.
            "betas": [settings.fallback_beta],
            "fallbacks": "default",
        }

    async def structured(self, *, model: str, effort: str, system: str, prompt: str, schema: type[T]) -> T:
        """Ask once; get back a validated instance of `schema`.

        `output_format=schema` makes the API constrain Claude's output to the
        model's JSON schema, and the SDK parses it into `parsed_output`.
        """
        response = await self.client.beta.messages.parse(
            **self._common(model, effort),
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        return _check(response).parsed_output

    async def stream_text(self, *, model: str, effort: str, system: str, prompt: str,
                          on_text: Callable[[str], None] | None = None) -> str:
        """Stream a long answer, calling on_text(chunk) as text arrives (the UI types it live)."""
        async with self.client.beta.messages.stream(
            **self._common(model, effort),
            max_tokens=32000,   # streaming, so a big ceiling won't trip HTTP timeouts
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for chunk in stream.text_stream:
                if on_text:
                    on_text(chunk)
            final = await stream.get_final_message()
        _check(final)
        return "".join(b.text for b in final.content if b.type == "text")

    async def web_search(self, *, model: str, effort: str, system: str, prompt: str,
                         max_searches: int, max_resumes: int = 3) -> list:
        """Run Claude with its server-side web search tool; return every content block.

        The searches happen INSIDE the API call (Anthropic runs them), so there
        is no search key and no tool loop on our side. One quirk: a long search
        session can stop with `pause_turn`. We resume by sending the paused
        assistant turn back unchanged, and the server carries on.
        """
        tool = {"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches}
        messages: list = [{"role": "user", "content": prompt}]
        blocks: list = []
        for _ in range(max_resumes + 1):
            response = await self.client.beta.messages.create(
                **self._common(model, effort), max_tokens=16000,
                system=system, messages=messages, tools=[tool],
            )
            blocks.extend(response.content)
            if response.stop_reason != "pause_turn":
                break
            messages.append({"role": "assistant", "content": response.content})
        _check(response)
        return blocks
