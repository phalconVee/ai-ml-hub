"""The only file that talks to Claude. Tests replace it with a scripted fake.

One method is all the agent needs: `step()`, a single Messages API call with
tools. The agent loop (agent.py) decides what to do with the answer.
"""

from typing import Any

from anthropic import AsyncAnthropic

from switchboard.settings import MAX_TOKENS


class LLM:
    def __init__(self, client: AsyncAnthropic | None = None):
        # No arguments: the SDK reads ANTHROPIC_API_KEY from the environment (.env is loaded in settings.py).
        self.client = client or AsyncAnthropic()

    async def step(self, *, model: str, effort: str, system: str, messages: list[dict], tools: list[dict]) -> Any:
        return await self.client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            tools=tools,
            # Effort trades depth of thinking for speed and cost. Current Claude
            # models think adaptively by default, so no `thinking` argument.
            output_config={"effort": effort},
            # Prompt caching: tools + system prompt + earlier turns are the same
            # on every step of the loop, so cache them and pay ~10% for the re-reads.
            cache_control={"type": "ephemeral"},
        )


def block_to_dict(block: Any) -> dict:
    """SDK content block -> plain dict, so the conversation can live in Streamlit's
    session state and be shown, saved or replayed. Thinking blocks keep their
    signature: the API needs them sent back unchanged on the next step."""
    if isinstance(block, dict):
        return block
    return block.model_dump(mode="json", exclude_none=True)
