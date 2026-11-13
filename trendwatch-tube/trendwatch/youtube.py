"""The collector: turn "channels I care about" into a list of recent videos with transcripts.

No LLM here on purpose. Fetching data is deterministic work, and code does it
cheaper, faster and more reliably than an agent "deciding" to call a tool.
The agents only start once the facts are on the table.

Two ways to list a channel's videos:

* **YouTube Data API v3** (needs a free `YOUTUBE_API_KEY`): official, resolves
  @handles, and returns likes and comment counts.
* **Public RSS feed** (no key): `youtube.com/feeds/videos.xml?channel_id=...`
  gives the latest 15 uploads with view counts. @handles are resolved by
  reading the channel page, which is best-effort.

Transcripts come from `youtube-transcript-api`, which reads the captions YouTube
already shows under a video. No key, but no captions means no transcript.
"""

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import httpx
from youtube_transcript_api import CouldNotRetrieveTranscript, YouTubeTranscriptApi

from .config import Settings
from .models import Channel, Video

API = "https://www.googleapis.com/youtube/v3"
RSS = "https://www.youtube.com/feeds/videos.xml"

# A channel id is "UC" + 22 URL-safe base64 characters.
CHANNEL_ID_RE = re.compile(r"(UC[\w-]{22})")
HANDLE_RE = re.compile(r"@([\w.\-]+)")

# XML namespaces used by the RSS (Atom) feed.
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}

Progress = Callable[[str], None]


# --------------------------------------------------------------------------
# Step 1: understand what the user typed
# --------------------------------------------------------------------------
def parse_channel_ref(ref: str) -> tuple[str, str]:
    """Classify one line of user input as ("id", "UC...") or ("handle", "@name").

    Accepts a bare id, a bare @handle, a bare name, or any youtube.com URL
    containing /channel/UC... or /@handle.
    """
    ref = ref.strip()
    if m := CHANNEL_ID_RE.search(ref):          # id anywhere in the string wins
        return "id", m.group(1)
    if m := HANDLE_RE.search(ref):              # "@name" or ".../@name/videos"
        return "handle", "@" + m.group(1)
    if ref and "/" not in ref and " " not in ref:
        return "handle", "@" + ref              # "mkbhd" -> "@mkbhd"
    raise ValueError(f"Can't read a channel from {ref!r}. Use a channel URL, @handle or UC... id.")


# --------------------------------------------------------------------------
# Step 2: parse the RSS feed (pure function, so it's easy to test)
# --------------------------------------------------------------------------
def parse_rss(xml_text: str) -> tuple[Channel, list[Video]]:
    root = ET.fromstring(xml_text)
    channel_id = root.findtext("yt:channelId", default="", namespaces=NS)
    if channel_id and not channel_id.startswith("UC"):
        channel_id = "UC" + channel_id          # some feeds drop the "UC" prefix
    channel = Channel(channel_id=channel_id, title=root.findtext("atom:title", default="", namespaces=NS))

    videos = []
    for entry in root.findall("atom:entry", NS):
        stats = entry.find("media:group/media:community/media:statistics", NS)
        videos.append(
            Video(
                video_id=entry.findtext("yt:videoId", namespaces=NS),
                channel_id=channel.channel_id,
                channel_title=channel.title,
                title=entry.findtext("atom:title", default="", namespaces=NS),
                description=entry.findtext("media:group/media:description", default="", namespaces=NS),
                published_at=datetime.fromisoformat(entry.findtext("atom:published", namespaces=NS)),
                views=int(stats.get("views")) if stats is not None and stats.get("views") else None,
            )
        )
    return channel, videos


def _int_or_none(value: str | None) -> int | None:
    return int(value) if value not in (None, "") else None


# --------------------------------------------------------------------------
# Step 3: the collector itself
# --------------------------------------------------------------------------
class YouTubeCollector:
    def __init__(self, settings: Settings, http: httpx.Client | None = None, transcripts=None):
        self.s = settings
        # Both dependencies can be swapped for fakes in tests (see tests/).
        self.http = http or httpx.Client(timeout=20, follow_redirects=True, headers={"User-Agent": "trendwatch-tube/0.1"})
        self.transcripts = transcripts or YouTubeTranscriptApi()

    # ---- public entry point ------------------------------------------------
    def collect(self, refs: list[str], on_progress: Progress = print) -> tuple[list[Video], list[str]]:
        """Return (videos, warnings). A bad channel becomes a warning, not a crash."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.s.days_back)
        videos: list[Video] = []
        warnings: list[str] = []

        for ref in refs:
            try:
                recent = [v for v in self._list_videos(ref) if v.published_at >= cutoff]
                recent.sort(key=lambda v: v.published_at, reverse=True)
                recent = recent[: self.s.videos_per_channel]
                on_progress(f"{ref}: {len(recent)} video(s) in the last {self.s.days_back} days")
                videos.extend(recent)
            except (httpx.HTTPError, ValueError, ET.ParseError, LookupError) as e:
                warnings.append(f"Skipped {ref!r}: {e}")

        for v in videos:
            on_progress(f"Transcript: {v.title[:60]}")
            self._attach_transcript(v)
            if v.transcript is None:
                warnings.append(f"No transcript for '{v.title}': analysed from title and description only.")
        return videos, warnings

    # ---- listing: pick the API or RSS route ---------------------------------
    def _list_videos(self, ref: str) -> list[Video]:
        kind, value = parse_channel_ref(ref)
        if self.s.youtube_api_key:
            return self._list_via_api(kind, value)
        channel_id = value if kind == "id" else self._scrape_channel_id(value)
        resp = self.http.get(RSS, params={"channel_id": channel_id})
        resp.raise_for_status()
        return parse_rss(resp.text)[1]

    def _scrape_channel_id(self, handle: str) -> str:
        """Keyless handle -> id: the channel page embeds its own id. Best-effort."""
        resp = self.http.get(f"https://www.youtube.com/{handle}")
        resp.raise_for_status()
        # Prefer the canonical link; fall back to any id-shaped string in the page.
        m = re.search(r'<link rel="canonical" href="https://www\.youtube\.com/channel/(UC[\w-]{22})"', resp.text)
        m = m or re.search(r'"externalId":"(UC[\w-]{22})"', resp.text)
        if not m:
            raise ValueError(f"couldn't find a channel id for {handle} (set YOUTUBE_API_KEY, or paste the UC... id)")
        return m.group(1)

    def _api(self, endpoint: str, **params) -> dict:
        resp = self.http.get(f"{API}/{endpoint}", params={**params, "key": self.s.youtube_api_key})
        resp.raise_for_status()
        return resp.json()

    def _list_via_api(self, kind: str, value: str) -> list[Video]:
        # 1 quota unit: find the channel and its hidden "uploads" playlist.
        lookup = {"id": value} if kind == "id" else {"forHandle": value}
        items = self._api("channels", part="contentDetails", **lookup).get("items", [])
        if not items:
            raise ValueError(f"no channel found for {value}")
        uploads = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

        # 1 unit: newest uploads. (search.list would cost 100 units; this is the cheap road.)
        page = self._api("playlistItems", part="contentDetails", playlistId=uploads, maxResults=25)
        ids = [it["contentDetails"]["videoId"] for it in page.get("items", [])]
        if not ids:
            return []

        # 1 unit: titles, descriptions and counts for up to 50 videos at once.
        details = self._api("videos", part="snippet,statistics", id=",".join(ids))
        out = []
        for it in details.get("items", []):
            sn, st = it["snippet"], it.get("statistics", {})
            out.append(
                Video(
                    video_id=it["id"],
                    channel_id=sn["channelId"],
                    channel_title=sn["channelTitle"],
                    title=sn["title"],
                    description=sn.get("description", ""),
                    published_at=datetime.fromisoformat(sn["publishedAt"].replace("Z", "+00:00")),
                    views=_int_or_none(st.get("viewCount")),
                    likes=_int_or_none(st.get("likeCount")),       # hidden on some channels
                    comments=_int_or_none(st.get("commentCount")),  # None if comments are off
                )
            )
        return out

    # ---- transcripts --------------------------------------------------------
    def _attach_transcript(self, video: Video) -> None:
        try:
            fetched = self.transcripts.fetch(video.video_id, languages=self.s.transcript_languages)
        except CouldNotRetrieveTranscript:   # base class: disabled, not found, blocked, ...
            return
        text = " ".join(snippet.text for snippet in fetched)
        limit = self.s.max_transcript_chars
        video.transcript = text[:limit]
        video.transcript_truncated = len(text) > limit
