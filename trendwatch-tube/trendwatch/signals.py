"""Hard numbers, computed in plain Python, between the two LLM steps.

Why bother when Claude could "just look"? Because LLMs are poor at counting
and averaging across 30 items, and because a trend claim is more honest when
it rests on a number you can check. So after the analysts tag every video,
we count the tags here and hand the table to the trend spotter as evidence.

The core idea is **velocity**: 50k views in 2 days is hotter than 200k in
a month, so we compare videos by views per day, not raw views.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .models import Video, VideoDigest


def views_per_day(video: Video, now: datetime | None = None) -> float | None:
    if video.views is None:
        return None
    now = now or datetime.now(timezone.utc)
    age_days = (now - video.published_at).total_seconds() / 86_400
    # Floor at half a day so a 2-hour-old video with 5k views isn't "60k/day".
    return video.views / max(age_days, 0.5)


@dataclass
class TopicSignal:
    topic: str
    videos: list[str] = field(default_factory=list)      # video_ids
    channels: set[str] = field(default_factory=set)      # channel titles
    views_per_day: float = 0.0                           # summed across its videos
    newest: datetime | None = None

    @property
    def buzz(self) -> float:
        """Breadth x heat. A topic on 3 channels beats one channel posting 3 times,
        and log() stops one viral video from drowning out everything else."""
        return len(self.channels) * math.log1p(self.views_per_day) + len(self.videos)

    def as_row(self) -> dict:
        return {
            "topic": self.topic,
            "videos": len(self.videos),
            "channels": len(self.channels),
            "views/day": round(self.views_per_day),
            "buzz": round(self.buzz, 1),
            "newest": self.newest.date().isoformat() if self.newest else "",
        }


def _normalise(tag: str) -> str:
    # "AI Agents " and "ai agents" are the same topic.
    return " ".join(tag.lower().split())


def topic_signals(videos: list[Video], digests: list[VideoDigest], now: datetime | None = None) -> list[TopicSignal]:
    """One row per topic tag, strongest first."""
    by_id = {v.video_id: v for v in videos}
    table: dict[str, TopicSignal] = {}

    for d in digests:
        video = by_id.get(d.video_id)
        if video is None:                   # the analyst mangled the id: ignore, don't guess
            continue
        vpd = views_per_day(video, now) or 0.0
        for tag in {_normalise(t) for t in d.topics}:   # a set: count each tag once per video
            sig = table.setdefault(tag, TopicSignal(topic=tag))
            sig.videos.append(video.video_id)
            sig.channels.add(video.channel_title)
            sig.views_per_day += vpd
            if sig.newest is None or video.published_at > sig.newest:
                sig.newest = video.published_at

    return sorted(table.values(), key=lambda s: s.buzz, reverse=True)
