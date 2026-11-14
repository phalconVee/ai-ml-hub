"""Streamlit UI: chat with the swarm and watch the handoffs happen.

Run:  uv run streamlit run app.py

The page is a thin shell around swarm.loop.run_turn(): it keeps a SwarmState
in st.session_state, feeds it the user's message, and draws every Event the
loop yields as it arrives (handoffs, tool calls, answers).
"""

from __future__ import annotations

import os

import streamlit as st

from swarm.backends import ClaudeBackend, OllamaBackend
from swarm.config import SETTINGS
from swarm.loop import Event, SwarmState, run_turn
from swarm.team import AGENTS, triage

st.set_page_config(page_title="Local Handoff Swarm", page_icon="🔀", layout="wide")


# ── Session state: survives Streamlit's rerun-on-every-click model ──────────
if "swarm" not in st.session_state:
    st.session_state.swarm = SwarmState(active=triage)
    st.session_state.log = []          # what to redraw: [("user", text) | ("turn", [Event, ...])]
state: SwarmState = st.session_state.swarm


@st.cache_resource(show_spinner=False)  # one client per (backend, model), reused across reruns
def get_backend(kind: str, model: str):
    return ClaudeBackend(model=model) if kind == "claude" else OllamaBackend(model=model)


def agent_label(name: str) -> str:
    agent = AGENTS.get(name)
    return f"{agent.emoji} {name}" if agent else name


def draw_event(ev: Event) -> None:
    """One Event → one visual element. Used live and when redrawing history."""
    if ev.kind == "handoff":
        st.markdown(f"🔀 **Handoff:** {agent_label(ev.agent)} → **{agent_label(ev.target)}**")
    elif ev.kind == "tool_call":
        args = ", ".join(f"{k}={v!r}" for k, v in ev.arguments.items())
        st.caption(f"🔧 {agent_label(ev.agent)} called `{ev.tool}({args})`")
    elif ev.kind == "tool_result":
        with st.expander(f"Result of `{ev.tool}`"):
            st.text(ev.text)
    elif ev.kind == "message":
        st.markdown(f"**{agent_label(ev.agent)}**")
        st.markdown(ev.text)
    elif ev.kind == "error":
        st.warning(ev.text)


# ── Sidebar: backend choice, the team, sources ──────────────────────────────
with st.sidebar:
    st.header("🔀 Local Handoff Swarm")
    kind = st.radio("Backend", ["ollama", "claude"],
                    index=0 if SETTINGS.backend != "claude" else 1, horizontal=True,
                    help="Ollama runs on your machine with no key. Claude needs ANTHROPIC_API_KEY.")
    default_model = SETTINGS.claude_model if kind == "claude" else SETTINGS.ollama_model
    model = st.text_input("Model", value=default_model)
    if kind == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        st.info("No ANTHROPIC_API_KEY found. Add it to .env (see .env.example).")

    always_triage = st.toggle(
        "Start every message at Triage", value=False,
        help="Off (default): a follow-up goes to whichever agent spoke last, "
             "like a real hand-off. On: Triage re-routes every message.")

    st.subheader("Team")
    # A placeholder we can overwrite mid-turn, so the highlight follows each handoff live.
    roster = st.empty()

    def draw_roster(active_name: str) -> None:
        lines = [f"**▶ {agent_label(n)}** (active)" if n == active_name else f"&nbsp;&nbsp;&nbsp;{agent_label(n)}"
                 for n in AGENTS]
        roster.markdown("  \n".join(lines))

    draw_roster(state.active.name)

    sources = state.context.get("sources", [])
    if sources:
        st.subheader("Sources found")
        for i, s in enumerate(sources, 1):
            st.markdown(f"[{i}] [{s['title']}]({s['url']})")

    if st.button("New conversation", use_container_width=True):
        del st.session_state["swarm"]
        st.rerun()


# ── Main pane: history, then the live turn ──────────────────────────────────
st.title("Local Handoff Swarm")
st.caption("A triage agent routes your message; specialists hand the conversation to each other. "
           "Try: *Research the latest Ollama release and write a three-line LinkedIn post about it.*")

for kind_, payload in st.session_state.log:
    if kind_ == "user":
        st.chat_message("user").markdown(payload)
    else:
        with st.chat_message("assistant", avatar="🔀"):
            for ev in payload:
                draw_event(ev)

if prompt := st.chat_input("Ask the team…"):
    st.chat_message("user").markdown(prompt)
    st.session_state.log.append(("user", prompt))
    if always_triage:
        state.active = triage

    events: list[Event] = []
    with st.chat_message("assistant", avatar="🔀"):
        try:
            with st.spinner(f"{agent_label(state.active.name)} is thinking…"):
                for ev in run_turn(state, get_backend(kind, model), prompt):
                    events.append(ev)
                    draw_event(ev)                 # draw each step as it happens
                    if ev.kind == "handoff":
                        draw_roster(ev.target)     # move the highlight live
        except Exception as exc:  # most often: Ollama not running or model not pulled
            hint = (f"Is Ollama running, and is the model pulled? Try `ollama pull {model}`."
                    if kind == "ollama" else "Check ANTHROPIC_API_KEY and the model name.")
            events.append(Event("error", state.active.name, text=f"{exc.__class__.__name__}: {exc}\n\n{hint}"))
            draw_event(events[-1])
    st.session_state.log.append(("turn", events))
    st.rerun()  # redraw so the sidebar shows the final active agent and new sources
