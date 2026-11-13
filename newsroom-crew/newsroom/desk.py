"""The news desk: who works when, decided in plain Python.

    topic
      │
      ▼
    Reporter ── web_search + web_fetch (server tools) ──► field notes + blocks
      │
      ▼
    build_ledger()  (code)  ──► numbered sources [S1..Sn] with API-extracted quotes
      │  no sources? ──► spike the story (refuse to write from nothing)
      ▼
    Writer ──► Draft (headline, dek, body with [S#])
      │
      ▼
    audit() (code) + Editor (LLM) ──► "publish" | "revise" + notes
      │ revise, and revisions left? ──► Writer again, with the notes
      ▼
    Story (final draft, sources, every draft and review, a trace)

The LLMs decide *what to search* and *how to word things*. Code decides the
order of work, what counts as a source, and when to stop.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import SETTINGS
from .editor import Review, edit
from .ledger import Source, build_ledger, render_ledger
from .reporter import Notes, report
from .writer import Draft, write


@dataclass
class Story:
    topic: str
    status: str                    # "published" | "published_with_notes" | "spiked"
    draft: Draft | None = None     # the final draft (None if spiked)
    sources: list[Source] = field(default_factory=list)
    notes: Notes | None = None
    drafts: list[Draft] = field(default_factory=list)    # every draft, oldest first
    reviews: list[Review] = field(default_factory=list)  # one review per draft
    trace: list[str] = field(default_factory=list)       # human-readable log

    def used_sources(self) -> list[Source]:
        """Only the sources the final article actually cites, in ledger order."""
        if not self.reviews:
            return []
        cited = set(self.reviews[-1].audit.cited)
        return [s for s in self.sources if s.id in cited]

    def markdown(self) -> str:
        """The finished article plus its source list, ready to download."""
        if self.draft is None:
            return ""
        refs = "\n".join(f"- **[{s.id}]** [{s.title}]({s.url}) ({s.outlet})"
                         for s in self.used_sources())
        return f"{self.draft.markdown()}\n\n---\n\n### Sources\n\n{refs}\n"


def run_desk(topic: str, client: Any, *, words: int = 500, angle: str = "",
             today: str | None = None,
             on_step: Callable[[str], None] = lambda msg: None) -> Story:
    """Produce one story. `on_step` receives progress messages (the UI shows them)."""
    today = today or dt.date.today().strftime("%A, %d %B %Y")
    story = Story(topic, status="spiked")

    def log(msg: str) -> None:
        story.trace.append(msg)
        on_step(msg)

    # 1. Reporting ────────────────────────────────────────────────────────────
    log("Reporter: searching the web…")
    notes = report(topic, client, today=today, angle=angle)
    story.notes = notes
    log(f"Reporter: {len(notes.queries)} search(es), {len(notes.fetched)} page(s) read.")

    # 2. The ledger (code, not LLM) ──────────────────────────────────────────
    story.sources = build_ledger(notes.blocks)
    ledger_text = render_ledger(story.sources)
    ledger_ids = {s.id for s in story.sources}
    quoted = sum(1 for s in story.sources if s.quotes)
    log(f"Ledger: {len(story.sources)} source(s), {quoted} with quoted evidence.")
    if not story.sources:
        log("Desk: no sources found, so the story is spiked (we don't publish unsourced news).")
        return story

    # 3. Write → review → (maybe) revise ─────────────────────────────────────
    draft, feedback = None, None
    for round_no in range(SETTINGS.max_revisions + 1):
        log("Writer: drafting…" if draft is None else f"Writer: revision {round_no}…")
        draft = write(topic, notes.text, ledger_text, client, today=today,
                      words=words, previous=draft, feedback=feedback)
        story.drafts.append(draft)

        log("Editor: checking the draft against the sources…")
        review = edit(draft, notes.text, ledger_text, ledger_ids, client)
        story.reviews.append(review)
        log(f"Editor: {review.verdict} ({len(review.notes)} note(s)).")
        if review.verdict == "publish":
            story.status = "published"
            break
        feedback = review.notes
    else:
        # Out of revision rounds and the editor still has notes: publish the
        # last draft, but say so loudly and show the notes in the UI.
        story.status = "published_with_notes"
        log("Desk: revision budget used up; publishing with the editor's notes attached.")

    story.draft = draft
    return story
