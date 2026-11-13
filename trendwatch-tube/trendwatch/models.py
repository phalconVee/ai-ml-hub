"""The shapes of the data that flow between the agents.

Two kinds of model live here:

* Plain records we build ourselves from YouTube (`Channel`, `Video`).
* Contracts we ask Claude to fill in (`VideoDigest`, `TrendReport`). These are
  passed to the API as `output_format`, so Claude's reply is *guaranteed* to
  validate against them: no regex-parsing of free text, no "please reply in JSON".

The `Field(description=...)` strings are not just docs for you: they go into the
JSON schema the model sees, so they double as instructions.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# 1. What the collector gathers (no LLM involved)
# --------------------------------------------------------------------------
class Channel(BaseModel):
    channel_id: str          # the stable "UC..." id; handles can change, this can't
    title: str


class Video(BaseModel):
    video_id: str
    channel_id: str
    channel_title: str
    title: str
    description: str = ""
    published_at: datetime
    views: int | None = None         # None = the source didn't tell us
    likes: int | None = None
    comments: int | None = None
    transcript: str | None = None    # None = no captions available
    transcript_truncated: bool = False

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


# --------------------------------------------------------------------------
# 2. Agent 1 (the analyst) fills one of these per video
# --------------------------------------------------------------------------
class VideoDigest(BaseModel):
    video_id: str = Field(description="Copy the video_id you were given, unchanged.")
    summary: str = Field(description="Two or three sentences: what the video actually says.")
    topics: list[str] = Field(
        description=(
            "3-6 short, lowercase topic tags (1-3 words each), reusable across videos, "
            "e.g. 'ai agents', 'vibe coding', 'rust'. Prefer the general name over a pun or brand slogan."
        )
    )
    key_claims: list[str] = Field(description="Up to 4 concrete claims, predictions or how-tos from the video.")
    named_things: list[str] = Field(description="Products, tools, companies, people or papers mentioned by name.")
    tone: Literal["excited", "neutral", "skeptical", "critical", "instructional"]
    novelty: Literal["new", "update", "evergreen"] = Field(
        description="'new' = something that recently appeared; 'update' = news on a known thing; 'evergreen' = timeless."
    )


# --------------------------------------------------------------------------
# 3. Agent 2 (the trend spotter) fills one of these for the whole batch
# --------------------------------------------------------------------------
class Trend(BaseModel):
    name: str = Field(description="A short, plain name for the trend.")
    explanation: str = Field(description="What the trend is, in plain words, for someone who hasn't seen the videos.")
    why_now: str = Field(description="Why it is showing up now, based on the evidence.")
    momentum: Literal["emerging", "rising", "established", "fading"]
    evidence_video_ids: list[str] = Field(description="video_ids that support this trend. Use only ids you were given.")
    channels: list[str] = Field(description="Channel titles that cover it.")
    confidence: float = Field(ge=0, le=1, description="0-1. Low if it rests on one video or one channel.")


class TrendReport(BaseModel):
    headline: str = Field(description="One sentence: the single most important takeaway.")
    trends: list[Trend] = Field(description="3-7 trends, strongest first.")
    outliers: list[str] = Field(description="Videos or ideas that stand out but are not (yet) a trend.")
    caveats: list[str] = Field(description="Limits of this analysis: small sample, missing transcripts, etc.")
