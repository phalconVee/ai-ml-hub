"""The orchestrator: run the crew in order and pass results between them.

    collect ──► analyse (fan-out, parallel) ──► count signals ──► spot trends ──► write brief
     code            Claude x N                    code              Claude          Claude

This is a *workflow*, not a free-roaming agent: the steps are known in advance,
so code fixes the order and Claude does the judgement inside each step.
That makes runs predictable, cheap to reason about, and easy to test.

Also runnable headless:
    uv run python -m trendwatch.pipeline @fireship @ThePrimeTimeagen --days 7
"""

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .agents import run_analyst, run_editor, run_trend_spotter
from .config import Settings, get_settings
from .llm import AgentError, ClaudeLLM
from .models import TrendReport, Video, VideoDigest
from .signals import TopicSignal, topic_signals
from .youtube import YouTubeCollector

Progress = Callable[[str], None]


@dataclass
class RunResult:
    videos: list[Video]
    digests: list[VideoDigest]
    signals: list[TopicSignal]
    report: TrendReport
    warnings: list[str] = field(default_factory=list)


def analyse_videos(llm: ClaudeLLM, settings: Settings, videos: list[Video],
                   on_progress: Progress) -> tuple[list[VideoDigest], list[str]]:
    """Fan out: one analyst per video, a few at a time. One failure != a failed run."""
    digests, warnings = [], []
    with ThreadPoolExecutor(max_workers=settings.analyst_workers) as pool:
        futures = {pool.submit(run_analyst, llm, settings.worker_model, v): v for v in videos}
        for done, fut in enumerate(as_completed(futures), start=1):
            video = futures[fut]
            try:
                digests.append(fut.result())
                on_progress(f"Analysed {done}/{len(videos)}: {video.title[:60]}")
            except Exception as e:  # noqa: BLE001 - any analyst failure is reported, not fatal
                warnings.append(f"Analyst failed on '{video.title}': {e}")
    # as_completed returns in finish order; put digests back in video order for stable output.
    order = {v.video_id: i for i, v in enumerate(videos)}
    digests.sort(key=lambda d: order.get(d.video_id, 0))
    return digests, warnings


def run(channel_refs: list[str], *, settings: Settings | None = None, llm: ClaudeLLM | None = None,
        collector: YouTubeCollector | None = None, on_progress: Progress = print) -> RunResult:
    """Everything up to the trend report. The editor runs separately so the UI can stream it."""
    settings = settings or get_settings()
    llm = llm or ClaudeLLM()
    collector = collector or YouTubeCollector(settings)

    on_progress("Collecting videos and transcripts…")
    videos, warnings = collector.collect(channel_refs, on_progress)
    if not videos:
        raise AgentError("No recent videos found. Try more channels or a longer time window.\n" + "\n".join(warnings))

    on_progress(f"Analysing {len(videos)} videos…")
    digests, analyst_warnings = analyse_videos(llm, settings, videos, on_progress)
    warnings += analyst_warnings
    if not digests:
        raise AgentError("Every analyst failed.\n" + "\n".join(warnings))

    on_progress("Counting topic signals…")
    signals = topic_signals(videos, digests)

    on_progress("Spotting trends…")
    report = run_trend_spotter(llm, settings.lead_model, videos, digests, signals)

    # Keep the model honest: drop evidence ids it didn't get from us.
    known = {d.video_id for d in digests}
    for trend in report.trends:
        trend.evidence_video_ids = [vid for vid in trend.evidence_video_ids if vid in known]

    return RunResult(videos, digests, signals, report, warnings)


def write_brief(result: RunResult, audience: str, *, settings: Settings | None = None, llm: ClaudeLLM | None = None):
    """Stream the editor's Markdown brief."""
    settings = settings or get_settings()
    return run_editor(llm or ClaudeLLM(), settings.lead_model, result.report, result.videos, audience)


def main() -> None:
    p = argparse.ArgumentParser(description="Explain what's trending across some YouTube channels.")
    p.add_argument("channels", nargs="+", help="channel URLs, @handles or UC... ids")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--per-channel", type=int, default=5)
    p.add_argument("--audience", default="a tech content creator deciding what to make next")
    args = p.parse_args()

    settings = get_settings()
    settings.days_back, settings.videos_per_channel = args.days, args.per_channel
    llm = ClaudeLLM()
    result = run(args.channels, settings=settings, llm=llm)
    for w in result.warnings:
        print("warning:", w)
    print()
    for chunk in write_brief(result, args.audience, settings=settings, llm=llm):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
