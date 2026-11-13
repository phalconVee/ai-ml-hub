"""Every knob in one place.

Read this file first: it tells you what the system can be tuned on, before you
see how any of it works. Values come from the environment (or a `.env` file)
so you never edit code to change a key or a model.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Pull variables from trendwatch-tube/.env into os.environ (if the file exists).
# Real environment variables win over the file, which is what you want in CI.
load_dotenv()


@dataclass
class Settings:
    # --- Claude -------------------------------------------------------------
    # The lead model does the hard thinking: spotting trends across many videos.
    lead_model: str = field(default_factory=lambda: os.getenv("TRENDWATCH_LEAD_MODEL") or "claude-opus-5-5")
    # The worker model reads one video at a time. Same model by default; set it to
    # claude-haiku-5-5 in .env to make the per-video step much cheaper.
    worker_model: str = field(default_factory=lambda: os.getenv("TRENDWATCH_WORKER_MODEL") or "claude-opus-5-5")

    # --- YouTube ------------------------------------------------------------
    # Optional. With a key we use the official Data API (handles, likes and
    # comment counts). Without one we fall back to each channel's public RSS feed.
    youtube_api_key: str | None = field(default_factory=lambda: os.getenv("YOUTUBE_API_KEY") or None)

    # --- Pipeline -----------------------------------------------------------
    days_back: int = 14              # only videos published in the last N days count
    videos_per_channel: int = 5      # newest N per channel (after the date filter)
    # A one-hour talk is ~60k characters. We clip longer transcripts and *say so*
    # in the UI, rather than silently cutting them.
    max_transcript_chars: int = 60_000
    analyst_workers: int = 4         # how many videos the analysts read at once
    transcript_languages: tuple[str, ...] = ("en",)


def get_settings() -> Settings:
    """A fresh Settings each call, so a changed .env is picked up on rerun."""
    return Settings()
