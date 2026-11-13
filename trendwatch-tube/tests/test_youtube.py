"""The collector, tested offline: fake HTTP (httpx.MockTransport) and fake transcripts."""

from datetime import datetime, timedelta, timezone

import httpx
import pytest
from youtube_transcript_api import TranscriptsDisabled

from trendwatch.config import Settings
from trendwatch.youtube import YouTubeCollector, parse_channel_ref, parse_rss

CID = "UC" + "a" * 22
NOW = datetime.now(timezone.utc)


def rss(entries: list[tuple[str, str, datetime, int]]) -> str:
    """A minimal feed in the same shape YouTube serves."""
    body = "".join(
        f"""<entry><yt:videoId>{vid}</yt:videoId><title>{title}</title>
        <published>{when.isoformat()}</published>
        <media:group><media:description>desc of {title}</media:description>
          <media:community><media:statistics views="{views}"/></media:community></media:group></entry>"""
        for vid, title, when, views in entries
    )
    return f"""<?xml version="1.0"?>
<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015" xmlns:media="http://search.yahoo.com/mrss/"
      xmlns="http://www.w3.org/2005/Atom">
  <yt:channelId>{CID[2:]}</yt:channelId><title>Test Channel</title>{body}</feed>"""


@pytest.mark.parametrize(
    "ref, expected",
    [
        (CID, ("id", CID)),
        (f"https://www.youtube.com/channel/{CID}/videos", ("id", CID)),
        ("@fireship", ("handle", "@fireship")),
        ("https://youtube.com/@t3dotgg/videos", ("handle", "@t3dotgg")),
        ("mkbhd", ("handle", "@mkbhd")),
    ],
)
def test_parse_channel_ref(ref, expected):
    assert parse_channel_ref(ref) == expected


def test_parse_channel_ref_rejects_junk():
    with pytest.raises(ValueError):
        parse_channel_ref("not a channel at all")


def test_parse_rss_restores_uc_prefix_and_reads_views():
    channel, videos = parse_rss(rss([("v1", "Hello", NOW, 1234)]))
    assert channel.channel_id == CID and channel.title == "Test Channel"
    assert videos[0].views == 1234 and videos[0].description == "desc of Hello"


class FakeTranscripts:
    """Stands in for YouTubeTranscriptApi: returns snippets, or raises for 'v_off'."""

    class Snip:
        def __init__(self, text):
            self.text = text

    def fetch(self, video_id, languages=("en",)):
        if video_id == "v_off":
            raise TranscriptsDisabled(video_id)
        return [self.Snip("hello"), self.Snip("world " * 20)]


def make_collector(handler, **overrides) -> YouTubeCollector:
    s = Settings(youtube_api_key=None, **overrides)
    return YouTubeCollector(s, http=httpx.Client(transport=httpx.MockTransport(handler)), transcripts=FakeTranscripts())


def test_collect_rss_filters_by_date_clips_and_warns():
    feed = rss([
        ("v_new", "New", NOW - timedelta(days=1), 100),
        ("v_off", "Captions off", NOW - timedelta(days=2), 50),
        ("v_old", "Old", NOW - timedelta(days=90), 999),
    ])

    def handler(request: httpx.Request):
        if request.url.path == "/@chan":   # keyless handle resolution reads the channel page
            return httpx.Response(200, text=f'<link rel="canonical" href="https://www.youtube.com/channel/{CID}">')
        assert request.url.params["channel_id"] == CID
        return httpx.Response(200, text=feed)

    videos, warnings = make_collector(handler, max_transcript_chars=30).collect(["@chan"], on_progress=lambda _: None)

    assert [v.video_id for v in videos] == ["v_new", "v_off"]       # 90-day-old video dropped, newest first
    assert videos[0].transcript_truncated and len(videos[0].transcript) == 30
    assert videos[1].transcript is None
    assert any("No transcript" in w for w in warnings)


def test_bad_channel_is_a_warning_not_a_crash():
    videos, warnings = make_collector(lambda r: httpx.Response(404)).collect([CID], on_progress=lambda _: None)
    assert videos == [] and "Skipped" in warnings[0]


def test_collect_via_data_api():
    published = (NOW - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def handler(request: httpx.Request):
        assert request.url.params["key"] == "k"
        path = request.url.path.rsplit("/", 1)[-1]
        if path == "channels":
            assert request.url.params["forHandle"] == "@chan"
            return httpx.Response(200, json={"items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UUx"}}}]})
        if path == "playlistItems":
            return httpx.Response(200, json={"items": [{"contentDetails": {"videoId": "v1"}}]})
        return httpx.Response(200, json={"items": [{
            "id": "v1",
            "snippet": {"channelId": CID, "channelTitle": "Chan", "title": "T", "publishedAt": published},
            "statistics": {"viewCount": "10", "likeCount": "2"},   # commentCount missing = comments off
        }]})

    s = Settings(youtube_api_key="k")
    c = YouTubeCollector(s, http=httpx.Client(transport=httpx.MockTransport(handler)), transcripts=FakeTranscripts())
    videos, _ = c.collect(["@chan"], on_progress=lambda _: None)
    assert (videos[0].views, videos[0].likes, videos[0].comments) == (10, 2, None)
