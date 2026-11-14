"""The only file that talks to Claude.

One kind of call is all the agent needs: `step()`, one turn of a tool-using
conversation. The agent (agent.py) owns the loop. Tests replace this class with
a scripted fake, so the whole agent runs offline.
"""

from typing import Any

from anthropic import AsyncAnthropic

from webpilot.config import Settings


class RefusalError(RuntimeError):
    """The model (and its server-side fallback) declined the request."""


class LLM:
    def __init__(self, settings: Settings, client: AsyncAnthropic | None = None):
        self.settings = settings
        # No arguments: the SDK reads ANTHROPIC_API_KEY (or an `ant auth login` profile).
        self.client = client or AsyncAnthropic()

    async def step(self, *, system: str, messages: list, tools: list[dict], tool_choice: dict | None = None) -> Any:
        """Send the conversation so far; return Claude's next message."""
        response = await self.client.beta.messages.create(
            model=self.settings.model,
            max_tokens=16000,
            system=system,
            messages=messages,
            tools=tools,
            # Thinking is adaptive by default on Opus 5.5; effort sets how much.
            output_config={"effort": self.settings.effort},
            # If a safety classifier declines, retry on a fallback model inside
            # the same request instead of just stopping.
            betas=[self.settings.fallback_beta],
            fallbacks="default",
            **({"tool_choice": tool_choice} if tool_choice else {}),
        )
        # A refusal is HTTP 200 with stop_reason "refusal", so check explicitly.
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise RefusalError(f"Request declined ({getattr(details, 'category', None) or 'no category'})")
        return response
