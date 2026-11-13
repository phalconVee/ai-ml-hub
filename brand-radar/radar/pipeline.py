"""The conductor: runs the whole radar, start to finish.

    collect (all sources at once, keyless)          collectors.py
        + optional Claude web-search sweep          agents.run_sweep
      -> clean: filter, de-dupe, label              prep.py
      -> analysts (one per source, in parallel)     agents.run_analyst
      -> numbers (plain Python)                     metrics.py
      -> strategist writes the report (streamed)    agents.run_strategist
      -> citation check                             metrics.check_citations

The order is fixed in code, not decided by an LLM: brand monitoring is a known
procedure, so there's nothing to gain from letting a model plan it, and a lot
to gain (cost, predictability, testability) from not doing so. The "agentic"
part is in what each specialist does with its slice of the data.

`on_event(stage, message)` lets a UI show progress without this file knowing
anything about Streamlit.
"""

import asyncio
from typing import Callable

from radar import agents, metrics
from radar.collectors import collect_all
from radar.config import CLAUDE_SWEEP, settings
from radar.llm import LLM
from radar.prep import prepare
from radar.schemas import BrandQuery, Mention, RadarRun, SourceReport

Event = Callable[[str, str], None]


def _noop(stage: str, message: str) -> None:
    pass


async def run_radar(q: BrandQuery, sources: list[str], *, use_sweep: bool = False,
                    llm: LLM | None = None, on_event: Event = _noop,
                    on_report_text: Callable[[str], None] | None = None) -> RadarRun:
    llm = llm or LLM()
    warnings: list[str] = []

    # 1. Collect. The free sources and the (optional) Claude sweep run together.
    on_event("collect", f"Searching {len(sources)} sources for '{q.brand}' (last {q.days} days)...")
    jobs = [collect_all(q, sources, settings.per_source)]
    if use_sweep:
        jobs.append(agents.run_sweep(llm, q))
    results = await asyncio.gather(*jobs, return_exceptions=True)

    collected = results[0]
    if isinstance(collected, Exception):          # only if gather itself blew up
        collected = {s: collected for s in sources}
    raw: dict[str, list[Mention]] = {}
    errors: dict[str, str] = {}
    for name, res in collected.items():
        if isinstance(res, Exception):
            errors[name] = f"{type(res).__name__}: {res}"[:200]
            raw[name] = []
        else:
            raw[name] = res
    if use_sweep:
        sweep = results[1]
        if isinstance(sweep, Exception):
            errors[CLAUDE_SWEEP] = f"{type(sweep).__name__}: {sweep}"[:200]
            raw[CLAUDE_SWEEP] = []
        else:
            raw[CLAUDE_SWEEP] = sweep
    for name, err in errors.items():
        warnings.append(f"{name}: collection failed ({err})")
        on_event("collect", f"⚠️ {name} failed: {err}")

    # 2. Clean.
    clean, dropped = prepare(raw, q, settings.max_text_chars)
    for name in raw:
        on_event("collect", f"{name}: {len(raw[name])} found, {len(clean[name])} kept after filtering")

    # 3. Analysts: one per source with something to read, all at once. The
    # semaphore is shared, so no more than `max_parallel_llm` calls run at a time.
    gate = asyncio.Semaphore(settings.max_parallel_llm)
    todo = {s: m for s, m in clean.items() if m}
    on_event("analyse", f"{len(todo)} analyst agents reading {sum(map(len, todo.values()))} mentions...")
    outcomes = await asyncio.gather(
        *(agents.run_analyst(llm, q, s, m, gate) for s, m in todo.items()), return_exceptions=True)

    reports: dict[str, SourceReport] = {}
    for name in raw:
        reports[name] = SourceReport(name, clean[name], {}, error=errors.get(name, ""))
    for name, outcome in zip(todo, outcomes):
        if isinstance(outcome, Exception):
            reports[name].error = f"analysis failed: {type(outcome).__name__}: {outcome}"[:200]
            warnings.append(f"{name}: {reports[name].error}")
            continue
        verdicts, summary, problems = outcome
        reports[name].verdicts, reports[name].summary = verdicts, summary
        warnings += [f"{name}: {p}" for p in problems]
        on_event("analyse", f"{name}: {sum(v.relevant for v in verdicts.values())}/{len(verdicts)} relevant")

    # 4. Numbers.
    stats = metrics.compute(reports)
    failed = [o for o in outcomes if isinstance(o, Exception)]
    if todo and len(failed) == len(todo):
        # Every analyst failed (missing key, no credit, network down). Say so,
        # instead of the misleading "no relevant mentions" below.
        on_event("report", "All analysts failed; no report.")
        return RadarRun(q, reports, stats, f"_Analysis failed for every source, so there is no report. "
                        f"First error: `{type(failed[0]).__name__}: {str(failed[0])[:300]}`_", warnings)
    if stats["relevant"] == 0:
        # Nothing to write about. Don't pay for a report that can only say "nothing".
        on_event("report", "No relevant mentions found; skipping the report.")
        return RadarRun(q, reports, stats, "_No relevant mentions found in this window. "
                        "Try a longer look-back, more sources, or check the brand context._", warnings)

    # 5. Strategist.
    on_event("report", "Strategist is writing the report...")
    notes = [f"{n}: {r.error}" for n, r in reports.items() if r.error] + \
            [f"{n}: 0 mentions after filtering" for n, r in reports.items() if not r.error and not r.mentions]
    summaries = {n: r.summary for n, r in reports.items() if r.summary}
    report_md = await agents.run_strategist(
        llm, q, stats, summaries, metrics.flagged_mentions(reports), notes, on_text=on_report_text)

    # 6. Trust, but verify: every [id] the report cites must be a real mention.
    known = {m.id for r in reports.values() for m in r.mentions}
    bad = metrics.check_citations(report_md, known)
    if bad:
        warnings.append(f"report cites unknown ids: {', '.join(bad)}")
    on_event("done", "Done.")
    return RadarRun(q, reports, stats, report_md, warnings)
