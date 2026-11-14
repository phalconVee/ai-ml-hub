"""What an agent *is* in this project: a name, instructions and some functions.

That's the whole idea. There is no Agent base class to subclass and no hidden
state. An agent is data; the loop in loop.py is what brings it to life.

The one trick that makes handoffs work
--------------------------------------
Every tool is a plain Python function. When the loop runs a tool it looks at
what the function *returned*:

    returns a str (or anything else)  → that's a tool result; send it back to the model
    returns an Agent                  → that's a HANDOFF; make that agent the active one

So "hand off to the writer" is literally a function that returns the writer:

    def transfer_to_writer():
        "Hand the conversation to the Writer."
        return writer

The model sees `transfer_to_writer` as just another tool. It never needs to
know that handoffs are special; only the loop does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Agent:
    name: str                       # shown in the UI and in the transcript, e.g. "Researcher"
    instructions: str               # becomes the system prompt while this agent is active
    functions: list[Callable] = field(default_factory=list)  # tools AND handoffs
    emoji: str = "🤖"               # purely for the UI

    def tool(self, name: str) -> Callable | None:
        """Find one of this agent's functions by name (what the model asks for)."""
        return next((f for f in self.functions if f.__name__ == name), None)

    # A readable repr keeps test failures and debug prints short.
    def __repr__(self) -> str:
        return f"Agent({self.name!r})"
