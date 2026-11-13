"""The shapes of data that flow through the radar.

Read top to bottom, this file is the pipeline's story:

  BrandQuery   what the user asked us to watch
  Mention      one raw hit from one source (no AI involved yet)
  Verdict      an analyst's judgment on one mention  <- Claude fills this in
  AnalystOutput  what one analyst call returns (verdicts + a summary)
  SourceReport the finished analysis of one source
  RadarRun     everything from one run, ready for the UI or a JSON download

Pydantic models are used where Claude produces the data: the API constrains
Claude's JSON to the model's schema, and Pydantic validates it again on our side.
"""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


@dataclass
class BrandQuery:
    brand: str                       # "Notion"
    context: str = ""                # "the productivity / notes app": disambiguates the name
    aliases: list[str] = field(default_factory=list)   # other names: "Notion AI", "notion.so"
    exclude: list[str] = field(default_factory=list)   # words that mark a false hit: "notion of"
    days: int = 7                    # look-back window

    @property
    def terms(self) -> list[str]:
        """Every name a mention may use for the brand, lower-cased."""
        return [t.lower() for t in [self.brand, *self.aliases] if t.strip()]


@dataclass
class Mention:
    source: str          # key from config.SOURCES (or "claude_sweep")
    title: str
    url: str
    text: str            # the snippet / post body we actually analyse
    author: str = ""
    published: str = ""  # ISO-ish date, "" when the source doesn't say
    engagement: int = 0  # likes + reposts + points + comments, roughly "reach"
    id: str = ""         # short citable id like "news-3", assigned in prep.py


# --- What the analyst agent returns (Claude fills these in) --------------------

Sentiment = Literal["positive", "neutral", "negative", "mixed"]
# The analyst flags what (if anything) the brand should DO about a mention.
# Fixed categories make the report's "action queue" groupable and countable.
Action = Literal[
    "none",           # just noise or neutral coverage
    "respond",        # a complaint or question a human should reply to
    "fix",            # a concrete bug / outage / product problem
    "amplify",        # praise worth sharing or thanking
    "escalate",       # PR, legal, security or safety risk
    "opportunity",    # partnership, feature idea, competitor weakness, lead
]


class Verdict(BaseModel):
    id: str = Field(description="The mention id exactly as given, e.g. 'news-3'")
    relevant: bool = Field(description="True only if the mention is really about THIS brand (not a namesake)")
    sentiment: Sentiment = Field(description="Sentiment toward the brand, not toward the topic in general")
    themes: list[str] = Field(description="1-3 short lowercase themes, e.g. 'pricing', 'outage', 'ai features'")
    action: Action
    urgency: int = Field(description="0 (ignore) to 3 (act today)")
    summary: str = Field(description="One sentence: what this mention says about the brand")


class AnalystOutput(BaseModel):
    verdicts: list[Verdict]
    source_summary: str = Field(description="2-3 sentences: the overall picture on this source")


# --- Finished results (built in plain Python) --------------------------------

@dataclass
class SourceReport:
    source: str
    mentions: list[Mention]
    verdicts: dict[str, Verdict]     # mention id -> verdict (only ids that passed the check)
    summary: str = ""
    error: str = ""                  # set when collection or analysis failed


@dataclass
class RadarRun:
    query: BrandQuery
    sources: dict[str, SourceReport]
    metrics: dict
    report_md: str = ""
    warnings: list[str] = field(default_factory=list)
