"""Agent 3, the Editor: draft script in, a tighter final script out.

Editing happens in two layers:

  1. lint() (plain code, script.py) lists what's mechanically wrong:
     stage directions, over-long lines, monologues, lopsided hosts, length.
  2. The LLM editor gets the draft, the outline and that to-do list, and
     returns the full revised script plus a short list of what it changed.

Then code gets the last word again: if the edit is broken (too short, or it
lost a host entirely) we keep the draft rather than ship something worse.
The editor improves a script; it should never be able to destroy one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import SETTINGS
from .llm import ask_json
from .researcher import Outline
from .script import LINES_SCHEMA, SPEAKERS, Script, lint

SYSTEM = """You are the editor of a two-host explainer podcast. You get a \
draft script, the research outline it was written from, and a list of issues \
found by an automatic check. Return the complete edited script.

Edit for:
- Accuracy: remove or fix any claim that is not supported by the outline.
- Flow: a strong hook in the first two lines, one idea at a time, a clear takeaway at the end.
- Tightness: cut filler, repeated points and "great question" padding.
- Speakability: short sentences, no stage directions, markdown, lists or URLs; \
numbers written the way they are said.
- Fix every issue in the automatic check.
Keep the hosts' voices and names. Keep what already works; this is an edit, not a rewrite.
In "changes", list your edits in a few short bullet-style sentences."""

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "lines": LINES_SCHEMA,
        "changes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "lines", "changes"],
    "additionalProperties": False,
}


@dataclass
class Edit:
    script: Script                                    # what goes to the studio
    changes: list[str] = field(default_factory=list)  # the editor's own notes
    issues_before: list[str] = field(default_factory=list)  # lint on the draft
    issues_after: list[str] = field(default_factory=list)   # lint on the result
    kept_draft: bool = False                          # True if the edit was rejected


def usable(script: Script) -> bool:
    """The minimum a script must be to replace the draft."""
    return (len(script.lines) >= SETTINGS.min_lines
            and all(any(l.speaker == s for l in script.lines) for s in SPEAKERS))


def edit_script(draft: Script, outline: Outline, client: Any, *, names: dict[str, str],
                target_words: int) -> Edit:
    issues = lint(draft, target_words)
    user = (f"Host A is {names['A']}. Host B is {names['B']}. "
            f"Target length: about {target_words} words.\n\n"
            f"OUTLINE:\n{outline.brief()}\n\n"
            f"DRAFT SCRIPT:\n{draft.as_text(names)}\n\n"
            "AUTOMATIC CHECK:\n" + ("\n".join(f"- {i}" for i in issues) or "- No issues found."))
    data = ask_json(client, who="editor", system=SYSTEM, user=user,
                    effort=SETTINGS.editor_effort, schema=SCHEMA)
    edited = Script.from_json(data["title"], data["lines"])

    if not usable(edited):
        # Safety net: never replace a working draft with a broken edit.
        return Edit(draft, ["Edit rejected (too short or lost a host); kept the draft."],
                    issues, issues, kept_draft=True)
    return Edit(edited, list(data["changes"]), issues, lint(edited, target_words))
