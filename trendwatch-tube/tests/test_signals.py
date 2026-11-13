"""Worked examples of what the signal maths rewards."""

from datetime import datetime, timedelta, timezone

from trendwatch.models import Video, VideoDigest
from trendwatch.signals import topic_signals, views_per_day

NOW = datetime(2026, 11, 13, tzinfo=timezone.utc)


def video(vid, channel, views, days_old):
    return Video(video_id=vid, channel_id="UC" + channel, channel_title=channel, title=vid,
                 published_at=NOW - timedelta(days=days_old), views=views)


def digest(vid, *topics):
    return VideoDigest(video_id=vid, summary="", topics=list(topics), key_claims=[], named_things=[],
                       tone="neutral", novelty="new")


def test_views_per_day_is_floored_for_brand_new_videos():
    assert views_per_day(video("v", "A", 1000, days_old=10), NOW) == 100
    assert views_per_day(video("v", "A", 1000, days_old=0.01), NOW) == 2000   # floor of half a day
    assert views_per_day(Video(**{**video("v", "A", 1, 1).model_dump(), "views": None}), NOW) is None


def test_breadth_beats_one_channel_repeating_itself():
    videos = [
        video("a1", "A", 1000, 2), video("a2", "A", 1000, 2), video("a3", "A", 1000, 2),   # A says "rust" 3x
        video("b1", "B", 1000, 2), video("c1", "C", 1000, 2),                             # "agents" on A, B, C
    ]
    digests = [
        digest("a1", "Rust", "AI Agents"), digest("a2", "rust"), digest("a3", "rust"),
        digest("b1", "ai agents"), digest("c1", "ai  agents"),
    ]
    signals = topic_signals(videos, digests, NOW)
    assert signals[0].topic == "ai agents"          # 3 channels beats 3 videos on 1 channel
    assert len(signals[0].channels) == 3
    assert {s.topic for s in signals} == {"ai agents", "rust"}   # case and spacing normalised


def test_digest_with_unknown_id_is_ignored():
    assert topic_signals([video("a", "A", 1, 1)], [digest("ghost", "x")], NOW) == []
