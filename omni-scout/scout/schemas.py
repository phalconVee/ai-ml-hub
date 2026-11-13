"""The shapes of data that flow between agents.

Agents don't pass free text to each other: they pass these Pydantic objects.
That makes the hand-offs checkable (the API validates Claude's output against
the schema) and lets the UI render each stage without parsing prose.

    question --planner--> ResearchPlan --scouts--> [ScoutReport, ...] --synthesizer--> markdown report
"""

from typing import Literal

from pydantic import BaseModel, Field


# --- Planner output -----------------------------------------------------------

class PlatformTask(BaseModel):
    """One assignment: "go to <platform> and find out <brief>"."""

    platform: str = Field(description="One of the platform names offered to the planner.")
    brief: str = Field(description="What this scout should look for on its platform, in one or two sentences.")
    queries: list[str] = Field(description="2-4 starting search queries tuned to this platform's style.")


class ResearchPlan(BaseModel):
    """The planner's decomposition of the user's question."""

    restated_question: str = Field(description="The question restated precisely, resolving ambiguity.")
    tasks: list[PlatformTask]


# --- Scout output -------------------------------------------------------------

class Finding(BaseModel):
    """One claim a scout found, tied to where it found it."""

    claim: str = Field(description="A single factual statement or opinion, in your own words.")
    source_url: str = Field(description="URL the claim came from. Must be a URL returned by a tool.")
    source_title: str
    published: str = Field(default="", description="Publication date if the tool gave one, else empty.")
    kind: Literal["fact", "opinion", "announcement", "data"] = "fact"


class ScoutReport(BaseModel):
    """Everything one scout brings back from its platform."""

    platform: str
    summary: str = Field(description="2-4 sentences: what this platform says about the question.")
    findings: list[Finding]
    gaps: list[str] = Field(default_factory=list, description="What you looked for but could not find.")
