"""Run one goal from the terminal (same agent and guard rails as the UI).

    uv run python -m webpilot "Find the top 3 Hacker News posts about Rust and summarise them"
    uv run python -m webpilot --write --allow example.com "..."   # allow typing + (approved) submits

Form submissions are approved or declined at a y/N prompt.
"""

import argparse
from pathlib import Path

from webpilot.config import Settings, parse_domains
from webpilot.session import PilotSession


def main() -> None:
    ap = argparse.ArgumentParser(prog="webpilot")
    ap.add_argument("goal")
    ap.add_argument("--write", action="store_true", help="turn read-only mode off")
    ap.add_argument("--allow", default=None, help="comma-separated allow list")
    ap.add_argument("--deny", default=None, help="comma-separated deny list")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--show", action="store_true", help="show the browser window")
    ap.add_argument("--shots", default=None, help="folder to save screenshots into")
    args = ap.parse_args()

    s = Settings()
    if args.write:
        s.read_only = False
    if args.allow is not None:
        s.allow = parse_domains(args.allow)
    if args.deny is not None:
        s.deny = parse_domains(args.deny)
    if args.max_steps:
        s.max_steps = args.max_steps
    if args.show:
        s.headless = False
    shots = Path(args.shots) if args.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)

    session = PilotSession(s).start()
    try:
        run = session.new_run(args.goal)
        approved = None
        n = 0
        while True:
            for ev in session.advance(run, approved):
                if ev.kind == "text":
                    print(f"\n💭 {ev.text}")
                elif ev.kind == "tool":
                    print(f"→ {ev.tool}({ev.args})")
                elif ev.kind == "result":
                    first = ev.text.splitlines()[0] if ev.text else ""
                    print(f"  {'✗' if ev.is_error else '✓'} {first[:160]}")
                elif ev.kind == "image" and shots:
                    n += 1
                    (shots / f"step-{n:02d}.jpg").write_bytes(ev.image)
                elif ev.kind == "approval":
                    print(f"\n⚠️  Claude wants to submit a form:\n{ev.text}")
                elif ev.kind in {"final", "limit", "error"}:
                    print(f"\n[{ev.kind}]\n{ev.text}")
            if not run.awaiting_approval:
                break
            approved = input("Approve this submission? [y/N] ").strip().lower() == "y"
    finally:
        session.close()


if __name__ == "__main__":
    main()
