"""The agent loop: one user message in, one answer out, memory read and written.

    load thread history ─► call Claude ─► tool_use? ─yes─► run tools ─► call Claude again ...
     (memory 2)              │                              (memory 1 and 3)
                             └─ no: final answer ─► save both messages + trace to the thread

The loop is written by hand (no framework, no SDK tool runner) so every step
is visible: what goes to the model, what comes back, and when we stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from . import llm
from .config import SETTINGS
from .memory import Vault
from .tools import TOOLS, ToolBox

SYSTEM = """\
You are a helpful assistant with a long-term memory stored in a database. You \
reach it only through tools:

- search_knowledge_base: passages from documents the user uploaded.
- recall: notes you saved in earlier conversations (facts, preferences, tasks).
- remember: save a new note for future conversations.
- forget: delete a note that is wrong or out of date.

The current conversation's recent messages are already in front of you. \
Anything older, or from other conversations, is only available through recall.

How to work:
- If the question could be answered from the uploaded documents, search the \
knowledge base before answering, and cite what you use as (document, location), \
for example (handbook.pdf, p. 4). If the passages don't contain the answer, say so.
- If the user refers to earlier conversations, or their preferences would shape \
the answer, recall first.
- When the user tells you something durable about themselves, their work, a \
preference or a decision, save it with remember. One fact per note. Don't save \
document contents or passing remarks, and don't ask permission to remember.
- If something the user says contradicts a recalled note, forget the old note \
and remember the new fact.
- Answer in plain prose. Be concise."""


@dataclass
class TurnResult:
    text: str
    trace: list[dict]        # every tool call made this turn, in order
    rounds: int              # how many model calls the turn took
    stop_reason: str


def build_messages(vault: Vault, thread_id: str, user_text: str) -> list[dict]:
    """Memory 2: replay the thread's recent messages, then the new question.

    Only the visible text of past turns is stored and replayed. The tool calls
    behind old answers are already summarised by those answers, and dropping
    them keeps every request small. (The trace is still kept for the UI.)
    """
    history = vault.store.get_messages(thread_id, limit=SETTINGS.history_messages)
    messages = [{"role": m.role, "content": m.content} for m in history]
    # The window may start with an assistant message; the API wants a user first.
    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    messages.append({"role": "user", "content": user_text})
    return messages


def run_turn(vault: Vault, thread_id: str, user_text: str, client=None,
             on_tool: Optional[Callable[[dict], None]] = None) -> TurnResult:
    """Answer `user_text` in `thread_id`, using and updating all three memories.

    `on_tool` is called with each trace entry as it happens, so a UI can show
    live progress ("searching the knowledge base...").
    """
    client = client or llm.make_client()
    messages = build_messages(vault, thread_id, user_text)
    # Save the question first: if the API call fails, the thread still shows it.
    vault.store.add_message(thread_id, "user", user_text)

    toolbox = ToolBox(vault, thread_id)
    response = None
    for rounds in range(1, SETTINGS.max_tool_rounds + 1):
        response = llm.create(client, system=SYSTEM, messages=messages, tools=TOOLS)

        # Append the assistant turn *exactly as returned*: thinking blocks,
        # text and tool_use blocks must go back unchanged on the next call.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break   # end_turn, max_tokens or refusal: the model is done

        # Run every tool call in this turn (Claude may ask for several at
        # once) and return all results in ONE user message.
        results = []
        for block in response.content:
            if block.type == "tool_use":
                results.append(toolbox.run(block))
                if on_tool:
                    on_tool(toolbox.trace[-1])
        messages.append({"role": "user", "content": results})
    else:
        # The loop ran out of rounds while Claude still wanted tools.
        rounds = SETTINGS.max_tool_rounds

    text = llm.text_of(response)
    if response.stop_reason == "refusal":
        text = text or "I can't help with that request."
    elif response.stop_reason == "tool_use":
        text = (text + "\n\n" if text else "") + "_(Stopped: too many tool calls in one turn.)_"
    elif not text:
        text = "_(No answer was produced.)_"

    vault.store.add_message(thread_id, "assistant", text, trace=toolbox.trace)
    return TurnResult(text=text, trace=toolbox.trace, rounds=rounds, stop_reason=response.stop_reason)
