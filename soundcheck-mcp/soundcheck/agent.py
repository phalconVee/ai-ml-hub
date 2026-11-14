"""The chat agent: Claude + the MCP toolbox, in one hand-written loop.

One user message triggers this loop:

    send history + tool list to Claude
      -> Claude answers with text            : done, return it
      -> Claude asks for tools (tool_use)    : run them all on the MCP server,
                                               append the results, go round again

That's the whole "agent". No framework: about fifty lines you can step
through in a debugger, which is the point of this repo.
"""

import asyncio
from typing import Any, Callable

from soundcheck.config import settings
from soundcheck.llm import LLM
from soundcheck.toolbox import Toolbox

# The UI passes a callback to watch the agent work: ("tool_call", {...}),
# ("tool_result", {...}). Plain function, so tests can just append to a list.
EventFn = Callable[[str, dict], None]

SYSTEM = """You are Soundcheck, an assistant that answers questions about audio recordings.
You can't hear audio yourself: you work only through the soundcheck tools, which transcribe and analyse it.

How to work:
- The user's recording is already transcribed; its audio_id is below. Don't transcribe it again.
- Prefer targeted tools over reading everything: search_transcript to find where something was said,
  get_transcript with a time range to read a passage, sentiment_timeline for mood, identify_speakers
  for who said what, summarize_recording / extract_topics for overviews.
- For a short recording (under ~10 minutes), reading the whole transcript once is fine.
- Cite timestamps as [mm:ss] for every claim about what was said, so the user can jump there.
- Speaker labels from the local heuristic are approximate; say so if a conclusion depends on them.
- If the transcript doesn't contain the answer, say that plainly rather than guessing.

Server notes: {instructions}

{recording}"""


def system_prompt(toolbox: Toolbox, recording: dict | None) -> str:
    if recording:
        rec = (f"Active recording: audio_id={recording['audio_id']}, file={recording['file']}, "
               f"duration={recording['duration']}, language={recording.get('language')}.")
    else:
        rec = "No recording is loaded yet. Ask the user to upload one."
    return SYSTEM.format(instructions=toolbox.instructions, recording=rec)


async def run_turn(
    messages: list,
    toolbox: Toolbox,
    llm: LLM,
    recording: dict | None = None,
    on_event: EventFn | None = None,
) -> str:
    """Advance the conversation by one user message (already appended to `messages`).

    `messages` is mutated in place: assistant turns and tool results are
    appended exactly as the API returned them (thinking blocks included).
    Keeping history append-only matters: newer Claude models check that
    replayed thinking blocks belong to an unedited conversation.
    """
    emit = on_event or (lambda kind, data: None)
    system = system_prompt(toolbox, recording)
    tools = toolbox.claude_tools()

    for _ in range(settings.max_agent_turns):
        response = await llm.step(system=system, messages=messages, tools=tools)
        # Append the *whole* content list, not just the text: tool_use blocks
        # must be answered by id, and thinking blocks must be replayed as-is.
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            return "".join(b.text for b in response.content if b.type == "text")

        async def run_one(block: Any) -> dict:
            emit("tool_call", {"name": block.name, "input": block.input})
            text, is_error = await toolbox.call(block.name, block.input)
            emit("tool_result", {"name": block.name, "text": text, "is_error": is_error})
            return {"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": is_error}

        # Claude may ask for several tools at once (e.g. sentiment + speakers).
        # Run them concurrently, and return ALL results in ONE user message:
        # splitting them up teaches the model to stop making parallel calls.
        results = await asyncio.gather(*(run_one(b) for b in tool_uses))
        messages.append({"role": "user", "content": list(results)})

    # Budget spent. Ask for a final answer. (Consecutive user messages are
    # merged by the API, so this text joins the last batch of tool results.)
    # Tools stay declared: the history contains tool_use blocks that refer to them.
    messages.append({"role": "user", "content": "Tool budget reached. Answer now with what you have; no more tools."})
    response = await llm.step(system=system, messages=messages, tools=tools)
    messages.append({"role": "assistant", "content": response.content})
    return "".join(b.text for b in response.content if b.type == "text")
