"""Two specialist analysts. Each reads the whole paper once and writes a draft.

Why two narrow analysts instead of one "analyse everything" prompt?
A single prompt tends to spend its attention on the abstract and skim the
methods. Giving each analyst one job (and its own schema) gets deeper notes,
and the two calls run in parallel, so it costs no extra wall-clock time.

Both analysts must back every claim with a short verbatim quote + page.
They're drafts: the reviewer agent (reviewer.py) checks and merges them.
"""

from __future__ import annotations

import json
from typing import Any

from . import schemas
from .config import SETTINGS
from .llm import ask_claude, json_of
from .pdf_extract import Paper

QUOTE_RULE = ("Every `quote` must be copied verbatim from the paper text (8-30 words), "
              "and `page` must be the page marker it appears under.")

METHODS_SYSTEM = f"""You are the methods analyst on a paper-review team.
Your only job: how was this study done, and could someone repeat it?
Be specific: sample sizes, conditions, controls, models, datasets, statistical tests.
If a detail is missing from the paper, say it's missing; don't fill gaps from general knowledge.
{QUOTE_RULE}"""

FINDINGS_SYSTEM = f"""You are the findings analyst on a paper-review team.
Your only job: what did this paper find, and how strong is the evidence for each finding?
List the main results (usually 3-7), each with its effect size or key number when one is reported.
Rate `strength` by the evidence shown (sample size, controls, statistics), not by how confidently
the authors write. Link a finding to figures via `figure_ids` (ids like "F2" from the figure notes).
Also collect the limitations the authors themselves state.
{QUOTE_RULE}"""


def paper_context(paper: Paper, figure_notes: dict[str, dict]) -> str:
    """The text block both analysts (and the reviewer) start from."""
    figures = [
        {"figure_id": fid, "page": fig.page, "label": fig.label, "caption": fig.caption,
         **figure_notes.get(fid, {})}
        for fid, fig in ((f.fig_id, f) for f in paper.figures)
    ]
    return (
        f"PAPER TITLE: {paper.title}\n\n"
        f"FIGURE NOTES (from the figure reader; ids are ours, not the paper's):\n"
        f"{json.dumps(figures, indent=1)}\n\n"
        f"FULL TEXT (page markers are authoritative for citations):\n"
        f"{paper.full_text(SETTINGS.max_paper_chars)}"
    )


def _run(client: Any, system: str, schema: dict, context: str) -> dict:
    response = ask_claude(
        client,
        system=system,
        messages=[{"role": "user", "content": context}],
        schema=schema,
    )
    return json_of(response)


def analyse_methods(client: Any, context: str) -> dict:
    return _run(client, METHODS_SYSTEM, schemas.METHODS_NOTE, context)


def analyse_findings(client: Any, context: str) -> dict:
    return _run(client, FINDINGS_SYSTEM, schemas.FINDINGS_NOTE, context)
