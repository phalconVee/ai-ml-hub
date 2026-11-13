"""Command-line entry point, handy for quick runs without the UI.

    uv run python -m scout "What do developers think of MCP in production?"
"""

import asyncio
import sys

from scout.pipeline import research


def on_event(kind: str, data: dict) -> None:
    # Progress goes to stderr so stdout holds only the report (pipe-friendly).
    if kind == "plan":
        for t in data["plan"].tasks:
            print(f"[plan] {t.platform}: {t.brief}", file=sys.stderr)
    elif kind == "tool":
        print(f"[{data['platform']}] {data['tool']} {data['args']}", file=sys.stderr)
    elif kind == "scout_done":
        print(f"[{data['platform']}] done: {len(data['report'].findings)} findings", file=sys.stderr)
    elif kind == "scout_failed":
        print(f"[{data['platform']}] FAILED: {data['error']}", file=sys.stderr)
    elif kind == "synthesizing":
        print("[synth] merging findings...", file=sys.stderr)


def main() -> None:
    question = " ".join(sys.argv[1:]) or input("Research question: ")
    result = asyncio.run(research(question, on_event=on_event))
    print(result.report_md)


if __name__ == "__main__":
    main()
