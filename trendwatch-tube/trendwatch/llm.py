"""The only file that talks to Claude.

Every agent goes through `ClaudeLLM`, so model choice, refusal handling and
fallbacks live in one place, and tests can swap in a fake with the same two
methods (`structured` and `stream_text`) without touching the network.

Three API features doing the heavy lifting:

* **Structured outputs** (`messages.parse(output_format=PydanticModel)`): the
  reply is constrained to the schema and handed back as a validated object.
* **Effort** (`output_config={"effort": ...}`): Opus 5.5 always thinks; effort
  sets how hard. Cheap per-video reading gets "low", the cross-video synthesis
  gets "high".
* **Server-side refusal fallbacks** (`fallbacks="default"`): if a safety
  classifier declines (rare, but transcripts are arbitrary internet text),
  the API retries on a fallback model inside the same call.
"""

from collections.abc import Iterator
from typing import TypeVar

import anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept server-side fallbacks. Haiku 5.5 does not, so we only
# send the parameter when the chosen model supports it.
MODELS_WITH_FALLBACKS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1"}


class AgentError(RuntimeError):
    """Claude finished but didn't give us something usable."""


class ClaudeLLM:
    def __init__(self, client: anthropic.Anthropic | None = None):
        # Anthropic() reads ANTHROPIC_API_KEY (or an `ant auth login` profile).
        # max_retries covers 429s and 5xx with backoff, which matters when four
        # analysts hit the API at once.
        self.client = client or anthropic.Anthropic(max_retries=4)

    def _extra(self, model: str) -> dict:
        if model in MODELS_WITH_FALLBACKS:
            return {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        return {}

    @staticmethod
    def _check(stop_reason: str | None, stop_details, who: str) -> None:
        # Always look at stop_reason before trusting content.
        if stop_reason == "refusal":
            category = getattr(stop_details, "category", None)
            raise AgentError(f"{who}: Claude declined (category={category}).")
        if stop_reason == "max_tokens":
            raise AgentError(f"{who}: ran out of output tokens before finishing.")

    def structured(self, *, model: str, system: str, prompt: str, schema: type[T],
                   effort: str = "medium", max_tokens: int = 16_000, who: str = "agent") -> T:
        """Ask once, get back a validated instance of `schema`."""
        resp = self.client.beta.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            output_config={"effort": effort},
            **self._extra(model),
        )
        self._check(resp.stop_reason, resp.stop_details, who)
        if resp.parsed_output is None:
            raise AgentError(f"{who}: reply did not match the {schema.__name__} schema.")
        return resp.parsed_output

    def stream_text(self, *, model: str, system: str, prompt: str,
                    effort: str = "medium", max_tokens: int = 32_000, who: str = "agent") -> Iterator[str]:
        """Yield prose as it is written, so the UI can show it live."""
        with self.client.beta.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort},
            **self._extra(model),
        ) as stream:
            yield from stream.text_stream          # only text deltas; thinking is skipped
            final = stream.get_final_message()
        self._check(final.stop_reason, final.stop_details, who)
