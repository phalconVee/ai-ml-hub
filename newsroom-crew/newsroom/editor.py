"""Agent 3, the Editor — plus a mechanical check that runs before it.

Two layers of review, cheapest first:

  1. audit()  — plain Python. Finds citation ids that don't exist in the
                ledger and paragraphs that cite nothing. These are facts,
                not opinions, so no LLM is needed to spot them.
  2. edit()   — Claude as editor. Reads the draft next to the ledger's quotes
                and judges what code can't: is each claim actually supported
                by the source it cites? Is anything overstated or missing?

The editor returns a JSON verdict, "publish" or "revise", with concrete notes.
If the audit found broken citations, desk.py sends the draft back no matter
what the editor said: a hard rule beats a soft judgement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .config import SETTINGS
from .llm import ask_claude, check_refusal, text_of
from .writer import Draft

CITE = re.compile(r"\[(S\d+)\]")  # matches [S1], [S12] … one id per bracket


@dataclass
class Audit:
    cited: list[str] = field(default_factory=list)       # ids used, in first-use order
    unknown: list[str] = field(default_factory=list)     # ids not in the ledger
    uncited: list[str] = field(default_factory=list)     # paragraphs with no citation

    @property
    def ok(self) -> bool:
        return not self.unknown and not self.uncited

    def problems(self) -> list[str]:
        """The audit as editor notes the writer can act on."""
        notes = [f"Citation [{i}] does not exist in the ledger; remove it or use a real id."
                 for i in self.unknown]
        notes += [f'This paragraph has no citation; source it or cut it: "{p[:80]}…"'
                  for p in self.uncited]
        return notes


def paragraphs(body: str) -> list[str]:
    """Factual paragraphs: blank-line separated, skipping ## subheads."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body)]
    return [p for p in paras if p and not p.startswith("#")]


def audit(draft: Draft, ledger_ids: set[str]) -> Audit:
    a = Audit()
    for i in CITE.findall(draft.body):
        if i not in a.cited:
            a.cited.append(i)
    a.unknown = [i for i in a.cited if i not in ledger_ids]
    a.uncited = [p for p in paragraphs(draft.body) if not CITE.search(p)]
    return a


SYSTEM = """You are the editor on a small news desk. You check a draft before \
it is published. You do not rewrite it yourself.

Check, against the source ledger and its quotes:
- Is every factual claim supported by the source it cites? Flag numbers, \
dates, names or quotes that don't match.
- Is anything overstated, speculative, or written as opinion?
- Is the most important news in the first paragraph?
- Are there claims in the reporter's notes that the draft gets wrong?

Verdict "publish" if it is accurate and fair, even if not perfect. Verdict \
"revise" only for real problems, and then list each one as a concrete, \
fixable note (quote the sentence, say what is wrong, say what the source says)."""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["publish", "revise"]},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["verdict", "notes"],
    "additionalProperties": False,
}


@dataclass
class Review:
    verdict: str              # "publish" | "revise"
    notes: list[str]          # editor notes, audit problems first
    audit: Audit


def edit(draft: Draft, notes: str, ledger: str, ledger_ids: set[str], client: Any) -> Review:
    a = audit(draft, ledger_ids)
    content = "\n\n".join([
        f"DRAFT:\n{draft.markdown()}",
        f"REPORTER'S NOTES:\n{notes or '(none)'}",
        f"SOURCE LEDGER:\n{ledger}",
        # Tell the editor what the mechanical check already found, so it
        # spends its attention on what code can't check.
        "AUTOMATED CITATION CHECK:\n" + ("\n".join(a.problems()) or "No problems found."),
    ])
    response = ask_claude(client, system=SYSTEM, effort=SETTINGS.editor_effort,
                          messages=[{"role": "user", "content": content}],
                          schema=VERDICT_SCHEMA)
    check_refusal(response, "editor")
    data = json.loads(text_of(response))

    # Hard rule: broken citations always mean "revise", whatever the LLM said.
    verdict = data["verdict"] if a.ok else "revise"
    all_notes = a.problems() + [n for n in data["notes"] if n.strip()]
    return Review(verdict, all_notes, a)
