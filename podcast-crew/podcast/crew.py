"""The producer: runs the crew in a fixed order, in code.

    input ─► source.py ─► Researcher ─► Scriptwriter ─► lint + Editor ─► tidy ─► script
             (code)       (Agent 1)     (Agent 2)       (code + Agent 3) (code)

There's no "manager agent" deciding who goes next: the order never changes,
so plain Python is cheaper, faster and easier to test than asking a model.
Each agent is a single request with a JSON answer, so a full script costs
exactly three Claude calls.

Audio is a separate step (studio.record) on purpose: the UI shows the
script first, and you can re-voice the same script with another engine or
other voices without paying for the writing again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .editor import Edit, edit_script
from .researcher import Outline, research
from .script import Script, tidy
from .scriptwriter import target_words, write_script
from .source import Source, load_source


@dataclass
class Production:
    source: Source
    outline: Outline
    draft: Script
    edit: Edit
    trace: list[str] = field(default_factory=list)

    @property
    def script(self) -> Script:
        """The final script: edited (or kept draft), then cleaned for the voice."""
        return tidy(self.edit.script)


def produce(user_input: str, client: Any, *, names: dict[str, str] | None = None,
            minutes: float = 4, style: str = "", fetch=None,
            on_step: Callable[[str], None] = lambda s: None) -> Production:
    names = names or {"A": "Maya", "B": "Theo"}
    trace: list[str] = []

    def step(msg: str) -> None:  # log for the UI *and* keep it on the result
        trace.append(msg)
        on_step(msg)

    source = load_source(user_input, fetch=fetch)
    step(f"Source: “{source.title}” ({len(source.text):,} characters"
         f"{', cut to fit' if source.truncated else ''}).")

    outline = research(source, client)
    step(f"Researcher: outlined {len(outline.key_points)} key points for “{outline.title}”.")

    words = target_words(minutes)
    draft = write_script(outline, client, names=names, minutes=minutes, style=style)
    step(f"Scriptwriter: drafted {len(draft.lines)} lines, {draft.words()} words.")

    edit = edit_script(draft, outline, client, names=names, target_words=words)
    if edit.kept_draft:
        step("Editor: the edit was unusable, so the draft was kept.")
    else:
        step(f"Editor: {len(edit.issues_before)} issue(s) flagged, {len(edit.changes)} change(s); "
             f"final script {len(edit.script.lines)} lines, {edit.script.words()} words.")

    return Production(source, outline, draft, edit, trace)
