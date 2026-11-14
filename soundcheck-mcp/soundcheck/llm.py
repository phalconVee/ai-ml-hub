"""The only file that talks to Claude.

Two kinds of call cover everything:

    structured()  one request; the answer is JSON validated against a Pydantic
                  model. Used by the server's summarize / extract_topics tools.
    step()        one turn of a tool-using conversation; the caller owns the
                  loop. Used by the chat agent in agent.py.

Keeping them here means one place sets the shared options (model, effort,
refusal fallback) and one place tests replace with a scripted fake.
"""

from typing import Any, TypeVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from soundcheck.config import settings

T = TypeVar("T", bound=BaseModel)


# --- Output shapes for the server's Claude-backed tools -----------------------
# Field descriptions are part of the schema Claude sees, so they double as
# instructions ("timestamp as mm:ss copied from the transcript").


class KeyPoint(BaseModel):
    point: str = Field(description="One self-contained takeaway, one sentence.")
    timestamp: str = Field(description="mm:ss (or h:mm:ss) where it is said, copied from the transcript line.")


class Summary(BaseModel):
    headline: str = Field(description="One line: what this recording is.")
    summary: str = Field(description="A short paragraph (3-6 sentences) covering the whole recording.")
    key_points: list[KeyPoint] = Field(description="The 3-8 most important points, in time order.")
    action_items: list[str] = Field(description="Explicit commitments or next steps said aloud. Empty if none.")


class Topic(BaseModel):
    name: str = Field(description="2-5 word topic label.")
    description: str = Field(description="One sentence on what is said about it.")
    start: str = Field(description="mm:ss where discussion of it begins.")
    end: str = Field(description="mm:ss where it ends.")


class Entity(BaseModel):
    name: str
    type: str = Field(description="person | organization | place | product | event | other")
    context: str = Field(description="Why it is mentioned, a few words.")


class TopicMap(BaseModel):
    topics: list[Topic] = Field(description="The recording's sections, in time order.")
    entities: list[Entity] = Field(description="Named things mentioned. Do not invent any.")
    keywords: list[str] = Field(description="5-15 keywords a search index should use.")


# --- The client -----------------------------------------------------------------


class RefusalError(RuntimeError):
    """Claude (and its server-side fallback) declined the request."""


def _check(response: Any) -> Any:
    # A refusal arrives as HTTP 200 with stop_reason "refusal", not as an
    # exception, so it must be checked before reading the content.
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise RefusalError(f"Request declined ({getattr(details, 'category', None) or 'no category'})")
    return response


class LLM:
    def __init__(self, client: AsyncAnthropic | None = None):
        # No arguments: the SDK finds ANTHROPIC_API_KEY or an `ant auth login` profile itself.
        self.client = client or AsyncAnthropic()

    @staticmethod
    def _common(model: str, effort: str) -> dict:
        return {
            "model": model,
            # Effort = how hard Claude thinks. Opus 5.5 thinks adaptively by
            # default; effort is the dial (its default would be "medium").
            "output_config": {"effort": effort},
            # Opt into server-side refusal fallback: if a safety classifier
            # declines, the API re-runs the request on a fallback model.
            "betas": [settings.fallback_beta],
            "fallbacks": "default",
        }

    async def structured(self, *, system: str, prompt: str, schema: type[T]) -> T:
        response = await self.client.beta.messages.parse(
            **self._common(settings.analysis_model, settings.analysis_effort),
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,  # the API constrains output to this JSON schema; the SDK parses it
        )
        return _check(response).parsed_output

    async def step(self, *, system: str, messages: list, tools: list[dict]) -> Any:
        response = await self.client.beta.messages.create(
            **self._common(settings.agent_model, settings.agent_effort),
            max_tokens=16000,
            system=system,
            messages=messages,
            tools=tools,
        )
        return _check(response)
