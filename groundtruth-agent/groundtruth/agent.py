"""Brick 5: the agent loop. Think -> search -> read -> (search again) -> answer.

The loop is written by hand (no framework) so every step is visible:

    1. Send the conversation + the tool definition to Claude.
    2. If Claude asks for a tool, run the search, append the results, go to 1.
    3. If Claude answers, collect its text and the citations attached to it.

Claude decides how many searches it needs. A simple lookup is one call; a
question like "compare 2022 and 2023 margins and explain the change" becomes
three or four targeted searches. `max_tool_rounds` caps the bill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .config import SETTINGS
from .index import BM25Index
from .tool import SEARCH_TOOL, run_search

SYSTEM_PROMPT = """You answer questions using only the user's documents, which you can reach through the search_documents tool.

How to work:
- Always search before answering. Search again with different keywords when the first results don't contain the answer; split multi-part questions into separate searches.
- Tables arrive as Markdown and figures as written descriptions. Read numbers carefully and keep their units.
- Answer only from what the search results say. If they don't contain the answer, say so plainly and say what you did find. Never fill gaps from general knowledge.
- Be concise: lead with the direct answer, then the supporting detail."""

# Opt in to server-side refusal fallbacks: if a safety classifier declines the
# request on the main model, the API re-runs it on a fallback model inside the
# same call instead of returning an empty refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Citation:
    number: int      # the [n] shown in the answer
    title: str       # "report.pdf, page 4 (table)"
    source: str      # "report.pdf#page=4"
    quote: str       # the exact passage text Claude relied on


@dataclass
class Answer:
    # The answer as (text, [citation numbers]) segments, so a UI can render
    # superscripts exactly where Claude attached them.
    segments: list[tuple[str, list[int]]] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    searches: list[dict] = field(default_factory=list)   # {"query", "document", "hits"}
    stop_reason: str = ""
    # Full conversation including this turn, to pass back for follow-up questions.
    messages: list[dict] = field(default_factory=list)

    def markdown(self) -> str:
        """The answer with [n] markers inline, ready for st.markdown or print."""
        out = []
        for text, nums in self.segments:
            out.append(text + "".join(f" [{n}]" for n in nums))
        return "".join(out).strip()


def _collect_answer(content: list[Any], answer: Answer) -> None:
    """Walk Claude's final content blocks and number each distinct cited passage."""
    seen: dict[tuple, int] = {}   # (source, quote) -> citation number
    for block in content:
        if block.type != "text":
            continue   # skip thinking blocks, fallback markers, etc.
        nums: list[int] = []
        for cite in block.citations or []:
            if cite.type != "search_result_location":
                continue
            key = (cite.source, cite.cited_text)
            if key not in seen:
                seen[key] = len(seen) + 1
                answer.citations.append(Citation(seen[key], cite.title or cite.source, cite.source, cite.cited_text))
            if seen[key] not in nums:
                nums.append(seen[key])
        answer.segments.append((block.text, nums))


def ask(
    question: str,
    index: BM25Index,
    client=None,
    history: Optional[list[dict]] = None,
    on_event: Callable[[str], None] = lambda _msg: None,
) -> Answer:
    """Answer one question with grounded citations.

    `history` is the `messages` list from the previous Answer (for follow-ups).
    `on_event` receives short progress strings, e.g. for a Streamlit status box.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY

    # Tell Claude which files exist, so it can scope searches with `document`.
    system = SYSTEM_PROMPT + "\n\nDocuments available: " + ", ".join(index.docs)

    messages = list(history or []) + [{"role": "user", "content": question}]
    answer = Answer()

    for round_no in range(SETTINGS.max_tool_rounds + 1):
        last_round = round_no == SETTINGS.max_tool_rounds
        response = client.beta.messages.create(
            model=SETTINGS.model,
            max_tokens=SETTINGS.max_tokens,
            system=system,
            tools=[SEARCH_TOOL],
            # On the final allowed round, forbid more tool calls: answer with what you have.
            tool_choice={"type": "none"} if last_round else {"type": "auto"},
            output_config={"effort": SETTINGS.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            messages=messages,
        )

        # Always append the *whole* content back (thinking blocks included):
        # the API needs them unchanged to keep the conversation valid.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            answer.segments.append(("The model declined to answer this request.", []))
            break

        if response.stop_reason == "pause_turn":
            continue   # server paused a long turn; resend to let it continue

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_calls:
            _collect_answer(response.content, answer)   # final answer reached
            if response.stop_reason == "max_tokens":
                answer.segments.append(("\n\n(Answer cut off at the length limit.)", []))
            break

        # Run every requested search. Claude may ask for several in parallel;
        # all results go back in ONE user message, as the API expects.
        results = []
        for call in tool_calls:
            on_event(f"Searching: {call.input.get('query', '')!r}")
            content, hits = run_search(index, call.input)
            answer.searches.append({
                "query": call.input.get("query", ""),
                "document": call.input.get("document"),
                "hits": hits,
            })
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": content})
        messages.append({"role": "user", "content": results})

    answer.stop_reason = response.stop_reason or ""
    answer.messages = messages
    return answer
