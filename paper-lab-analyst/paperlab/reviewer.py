"""The reviewer: the one truly agentic step.

The analysts each read the paper once and wrote drafts. The reviewer gets
those drafts plus the figure notes, and is told to be sceptical. It decides
for itself what to double-check, using three tools (paper_tools.py):
search the text, read a page, or look at a figure. When it's satisfied, it
writes the final report as JSON matching `schemas.REPORT`.

The loop below is the standard tool-use loop, written out by hand so you can
see every step:

    ask Claude ─► stop_reason == "tool_use"? ─ yes ─► run tools ─► send results ─┐
        ▲                                     no                                 │
        │                                      ▼                                 │
        └──────────────────────────────── parse final JSON ◄─────────────────────┘
"""

from __future__ import annotations

import json
from typing import Any, Callable

from . import schemas
from .config import SETTINGS
from .llm import AnalysisError, ask_claude, json_of
from .paper_tools import TOOLS, PaperIndex, run_tool
from .pdf_extract import Paper

REVIEWER_SYSTEM = """You are the senior reviewer on a paper-analysis team. Two analysts wrote
drafts (methods, findings) and a figure reader described each figure. Your job is to produce
the final intelligence report, and you are accountable for its accuracy.

Before writing, verify. Use the tools to:
- check the key numbers and claims in the findings draft against the paper text;
- look at a figure yourself when a finding leans on it or its note seems off;
- search for limitations the drafts missed (sample size, controls, confounders,
  generalisability, missing statistics, conflicts of interest, preprint status).
A handful of targeted checks is better than reading everything.

Then write the report:
- Drop or downgrade any claim you could not confirm, and record that in `reviewer_notes`.
- `confidence` reflects the evidence, not the authors' tone.
- Every `evidence.quote` must be copied verbatim from the paper (8-30 words), with its page.
- Mark each limitation `stated_by_authors` or `reviewer_inferred`; never present your
  inference as the authors' admission.
- Use figure ids exactly as given ("F1", "F2", ...)."""


def review(client: Any, paper: Paper, context: str, methods: dict, findings: dict,
           on_event: Callable[[str], None] = lambda _msg: None) -> dict:
    """Run the reviewer agent and return the final report dict."""
    index = PaperIndex(paper)
    drafts = json.dumps({"methods_draft": methods, "findings_draft": findings}, indent=1)
    messages: list = [{"role": "user", "content": f"{context}\n\nANALYST DRAFTS:\n{drafts}"}]

    for turn in range(SETTINGS.max_reviewer_turns + 1):
        out_of_turns = turn == SETTINGS.max_reviewer_turns
        if out_of_turns:
            # Budget spent: say so, and forbid more tool calls for this last request.
            # (Opus 5.5 doesn't allow *forcing* a tool, but `none` is fine.)
            _add_user_text(messages, "Verification budget used up. Write the final report now.")
        response = ask_claude(
            client,
            system=REVIEWER_SYSTEM,
            messages=messages,
            tools=TOOLS,
            schema=schemas.REPORT,
            tool_choice={"type": "none"} if out_of_turns else None,
        )
        # Always append the *whole* assistant content (thinking + tool_use
        # blocks included). The API needs them back unchanged on the next call.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "pause_turn":
            continue  # the server paused a long turn; sending it back resumes it

        if response.stop_reason != "tool_use":
            return json_of(response)  # final answer (json_of raises on refusal/cut-off)

        # Run every tool call in this turn and return all results in ONE user
        # message, each tied to its call by `tool_use_id`.
        results = []
        for block in response.content:
            if getattr(block, "type", "") != "tool_use":
                continue
            on_event(f"🔎 reviewer → {block.name}({_short(block.input)})")
            output = run_tool(block.name, block.input, paper, index)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})

    raise AnalysisError("Reviewer did not produce a report.")  # unreachable in practice


def _add_user_text(messages: list, text: str) -> None:
    """Add a text note to the conversation's last user turn (or start one).

    Normally the last message is our tool results, so the note rides along in
    that same user turn instead of becoming a second user message in a row.
    """
    last = messages[-1]
    if last["role"] != "user":
        messages.append({"role": "user", "content": text})
        return
    if isinstance(last["content"], str):
        last["content"] = [{"type": "text", "text": last["content"]}]
    last["content"].append({"type": "text", "text": text})


def _short(args: dict) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return text if len(text) < 80 else text[:77] + "..."
