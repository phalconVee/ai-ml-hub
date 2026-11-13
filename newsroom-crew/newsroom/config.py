"""Every knob in one place.

Change a number here and the whole desk follows. No other file hard-codes a
model name, a search budget or a loop limit.
"""

from dataclasses import dataclass


@dataclass(frozen=True)  # frozen: settings are read, never changed at runtime
class Settings:
    # ── The model ────────────────────────────────────────────────────────────
    # Claude Opus 5.5 is the current default Opus. It always thinks before it
    # answers; `effort` is the dial for how hard (low → max). Its default is
    # "medium", so we set each agent's effort explicitly.
    model: str = "claude-opus-5-5"
    reporter_effort: str = "medium"  # searching + reading: breadth matters more than depth
    writer_effort: str = "high"      # turning evidence into clean prose is the hard part
    editor_effort: str = "medium"    # checking claims against quotes
    max_tokens: int = 16000          # room for thinking + output on non-streaming calls

    # Server-side refusal fallback: if a safety classifier declines a request,
    # the API re-runs it on a fallback model inside the same call. "default"
    # lets Anthropic pick the fallback model by refusal category.
    fallback_beta: str = "server-side-fallback-2026-07-01"

    # ── Reporter (reporter.py) ───────────────────────────────────────────────
    max_searches: int = 6          # web searches per story
    max_fetches: int = 4           # full pages the reporter may open and read
    max_fetch_tokens: int = 12000  # cap on how much of one page enters the context
    max_pause_resumes: int = 4     # how often we resume a `pause_turn`

    # ── Source ledger (ledger.py) ────────────────────────────────────────────
    max_sources: int = 12          # sources the writer is allowed to cite
    max_quotes_per_source: int = 4 # evidence snippets kept per source

    # ── Editing loop (desk.py) ───────────────────────────────────────────────
    max_revisions: int = 1         # editor may send the draft back this many times


SETTINGS = Settings()
