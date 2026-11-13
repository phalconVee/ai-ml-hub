"""Command-line entry point.

    uv run python -m radar "Notion" --context "notes and docs app"           # full run (needs ANTHROPIC_API_KEY)
    uv run python -m radar "Notion" --context "notes app" --collect-only     # no AI, no key: just see what the sources return
"""

import argparse
import asyncio
import json
from dataclasses import asdict

from radar.collectors import collect_all
from radar.config import SOURCES, settings
from radar.prep import prepare
from radar.schemas import BrandQuery


def to_json(run) -> str:
    """A RadarRun as JSON (verdicts are Pydantic models, so dump them first)."""
    return json.dumps({
        "query": asdict(run.query),
        "metrics": run.metrics,
        "warnings": run.warnings,
        "report_md": run.report_md,
        "sources": {
            name: {
                "summary": rep.summary, "error": rep.error,
                "mentions": [asdict(m) | {"verdict": rep.verdicts[m.id].model_dump() if m.id in rep.verdicts else None}
                             for m in rep.mentions],
            } for name, rep in run.sources.items()
        },
    }, indent=1, ensure_ascii=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Brand monitoring radar")
    p.add_argument("brand")
    p.add_argument("--context", default="", help="what the brand is, to rule out namesakes")
    p.add_argument("--alias", action="append", default=[])
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--sources", default=",".join(SOURCES))
    p.add_argument("--sweep", action="store_true", help="add Claude's web search as an extra source")
    p.add_argument("--collect-only", action="store_true", help="skip the AI: print what the free sources found")
    p.add_argument("--json", help="write the full run to this file")
    a = p.parse_args()

    q = BrandQuery(a.brand, a.context, a.alias, a.exclude, a.days)
    sources = [s.strip() for s in a.sources.split(",") if s.strip()]

    if a.collect_only:
        raw = asyncio.run(collect_all(q, sources, settings.per_source))
        ok = {s: r for s, r in raw.items() if not isinstance(r, Exception)}
        clean, _ = prepare(ok, q)
        for s, r in raw.items():
            if isinstance(r, Exception):
                print(f"\n## {s}: FAILED {type(r).__name__}: {r}")
                continue
            print(f"\n## {s}: {len(r)} found, {len(clean[s])} kept")
            for m in clean[s][:5]:
                print(f"  [{m.id}] {m.title[:70]!r}  {m.url[:70]}")
        return

    from radar.pipeline import run_radar   # imported here so --collect-only never needs anthropic set up
    run = asyncio.run(run_radar(q, sources, use_sweep=a.sweep,
                                on_event=lambda stage, msg: print(f"[{stage}] {msg}"),
                                on_report_text=lambda t: print(t, end="", flush=True)))
    print("\n\nWarnings:", *run.warnings or ["none"], sep="\n  ")
    if a.json:
        open(a.json, "w").write(to_json(run))


if __name__ == "__main__":
    main()
