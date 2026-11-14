"""The team: four agents and the handoff functions that connect them.

    ┌──────────┐  transfer_to_researcher   ┌────────────┐
    │  Triage  │ ────────────────────────► │ Researcher │ ── web_search
    │ (router) │  transfer_to_summarizer   ├────────────┤
    │          │ ────────────────────────► │ Summarizer │ ── word_count
    │          │  transfer_to_writer       ├────────────┤
    │          │ ────────────────────────► │   Writer   │ ── word_count
    └──────────┘ ◄──── transfer_back_to_triage (from every specialist)

Specialists can also hand to each other mid-task: the Researcher passes its
findings to the Summarizer or the Writer when the user asked for a summary or
a written piece, so "research X and write a blog post" flows
Triage → Researcher → Writer in one turn.

Who may hand to whom is just *which functions each agent has*. Want the
Writer to never call the Researcher? Don't give it `transfer_to_researcher`.
"""

from __future__ import annotations

from .agent import Agent
from .tools import web_search, word_count

# Every agent's prompt starts with this, so each one knows it's part of a team,
# that the transcript contains other agents' work, and how handoffs behave.
# Small local models need this said plainly.
TEAM_PREAMBLE = """You are {name}, one member of a small team of AI agents that share one conversation.
Earlier messages may come from the user or from teammates; build on their work.
To pass the conversation to a teammate, call the matching transfer_to_... tool and say nothing else;
the teammate takes over immediately. Only transfer when another teammate is clearly better suited.
If you can do the task yourself, do it and reply directly to the user."""


def _prompt(name: str, role: str) -> str:
    return TEAM_PREAMBLE.format(name=name) + "\n\nYour role:\n" + role


# ── Handoff functions ───────────────────────────────────────────────────────
# Each returns an Agent. That return value is the entire handoff mechanism
# (see loop.py). They reference agents defined further down; that's fine in
# Python because the name is looked up when the function *runs*, not when
# it's defined.

def transfer_to_researcher():
    """Hand the conversation to the Researcher, who searches the web for facts,
    news, prices, dates or anything that may have changed recently."""
    return researcher


def transfer_to_summarizer():
    """Hand the conversation to the Summarizer, who condenses text the user
    pasted, or research already in the conversation, into a short summary."""
    return summarizer


def transfer_to_writer():
    """Hand the conversation to the Writer, who drafts emails, blog posts,
    social posts, essays or rewrites text in a requested tone."""
    return writer


def transfer_back_to_triage():
    """Hand the conversation back to Triage when the user's new request is
    outside your role."""
    return triage


# ── The agents ──────────────────────────────────────────────────────────────
triage = Agent(
    name="Triage",
    emoji="🧭",
    instructions=_prompt("Triage", """You are the front desk. You do not do the work yourself.
Read the user's latest request and immediately transfer to exactly one teammate:
- needs current facts or anything to look up online → transfer_to_researcher
- wants text condensed or the key points of something → transfer_to_summarizer
- wants something written, drafted or rewritten → transfer_to_writer
If the request mixes steps (e.g. research then write), transfer to the teammate for the FIRST step.
Only answer directly for greetings or to ask one short clarifying question."""),
    functions=[transfer_to_researcher, transfer_to_summarizer, transfer_to_writer],
)

researcher = Agent(
    name="Researcher",
    emoji="🔎",
    instructions=_prompt("Researcher", """You find facts on the web.
1. Call web_search with a short keyword query (search again with new keywords if results are poor).
2. If the user ALSO asked for a summary, transfer_to_summarizer; if they asked for a written piece
   (post, email, article), transfer_to_writer. Your search results stay in the conversation for them.
3. Otherwise answer the user yourself: a few clear sentences or bullets, citing sources as [n]."""),
    functions=[web_search, transfer_to_summarizer, transfer_to_writer, transfer_back_to_triage],
)

summarizer = Agent(
    name="Summarizer",
    emoji="🗜️",
    instructions=_prompt("Summarizer", """You condense text.
Summarize the text the user gave, or the research already in the conversation, into tight bullet points
(5 or fewer unless a length was requested). Keep facts and numbers exact; keep [n] citations.
If the facts you need are not in the conversation yet, transfer_to_researcher.
If the user wants the summary turned into a written piece, transfer_to_writer."""),
    functions=[word_count, transfer_to_researcher, transfer_to_writer, transfer_back_to_triage],
)

writer = Agent(
    name="Writer",
    emoji="✍️",
    instructions=_prompt("Writer", """You write. Draft exactly what the user asked for (format, tone, length),
using the facts already in the conversation and keeping any [n] citations.
Don't invent facts: if the piece needs facts that aren't in the conversation, transfer_to_researcher first.
Reply with the finished piece only."""),
    functions=[word_count, transfer_to_researcher, transfer_back_to_triage],
)

# For the UI: look an agent up by name, list them in a stable order.
AGENTS: dict[str, Agent] = {a.name: a for a in (triage, researcher, summarizer, writer)}
