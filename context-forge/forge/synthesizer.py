"""Stage 4, the Synthesizer: write the answer from the selected context only.

The kept items are laid out as numbered blocks [S1], [S2], … and Claude must
cite the block behind every claim. Afterwards we *check* the citations: any
[S#] that doesn't exist in the window is a fabricated reference, so it is
stripped from the text and reported. Citations you can trust are the whole
point of grounding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import SETTINGS
from .items import ContextItem
from .llm import ask, text_of

SYSTEM = """You are a careful research assistant. Answer the question using ONLY \
the numbered context blocks provided. After every sentence that uses a block, cite \
it like [S2] (several: [S1][S3]). Do not cite anything that is not in the context. \
If the context does not contain enough to answer, say what is missing rather than \
guessing. Blocks marked (memory) are earlier answers of yours: treat them as \
helpful notes, and prefer primary sources when they disagree. Be concise: a few \
short paragraphs or a tight bullet list."""

CITE = re.compile(r"\[S(\d+)\]")


@dataclass
class Answer:
    text: str
    cited: list[str] = field(default_factory=list)    # valid ids, in order of first use
    invalid: list[str] = field(default_factory=list)  # ids the model made up


def render_context(items: list[ContextItem]) -> str:
    """The exact string that enters the model's window (also shown in the UI)."""
    blocks = []
    for it in items:
        head = f"[{it.id}] ({it.source}) {it.title}" + (f" <{it.url}>" if it.url else "")
        blocks.append(f"{head}\n{it.text}")
    return "\n\n".join(blocks)


def check_citations(text: str, valid_ids: set[str]) -> Answer:
    cited, invalid = [], []
    for n in CITE.findall(text):
        sid = f"S{n}"
        bucket = cited if sid in valid_ids else invalid
        if sid not in bucket:
            bucket.append(sid)
    # Remove the made-up markers so the reader never sees a dead reference.
    clean = CITE.sub(lambda m: m.group(0) if f"S{m.group(1)}" in valid_ids else "", text)
    return Answer(clean.strip(), cited, invalid)


def synthesize(question: str, items: list[ContextItem], client: Any) -> Answer:
    if not items:
        return Answer("I couldn't find any relevant context for this question in the "
                      "enabled sources. Try enabling more sources or rephrasing.")
    prompt = f"<context>\n{render_context(items)}\n</context>\n\nQuestion: {question}"
    resp = ask(client, system=SYSTEM, effort=SETTINGS.effort_answer,
               messages=[{"role": "user", "content": prompt}])
    if resp.stop_reason == "refusal":
        return Answer("The model declined to answer this question.")
    return check_citations(text_of(resp), {it.id for it in items})
