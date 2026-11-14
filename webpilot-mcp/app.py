"""WebPilot: Streamlit UI.

    uv run streamlit run app.py

Type a goal; watch Claude drive the browser step by step, with the latest
screenshot beside the log. Guard rails live in the sidebar.

Streamlit re-runs this whole file on every interaction, so anything that
must survive (the browser session, the conversation log, a run paused for
approval) lives in st.session_state.
"""

from dataclasses import replace

import streamlit as st

from webpilot.agent import AgentRun, Event
from webpilot.config import Settings, parse_domains
from webpilot.session import PilotSession

st.set_page_config(page_title="WebPilot", page_icon="🧭", layout="wide")

EXAMPLES = [
    "Find the top 3 Hacker News posts about Rust and summarise them",
    "What is the current stable Python version according to python.org?",
    "Open the Wikipedia article on the Model Context Protocol and give me 3 key facts",
]

# --- sidebar: the guard rails -----------------------------------------------------
defaults = Settings()
with st.sidebar:
    st.header("🧭 WebPilot")
    st.caption("Claude drives a real Chromium through a Playwright MCP server.")
    st.subheader("Guard rails")
    read_only = st.toggle("Read-only browsing", value=defaults.read_only,
                          help="Off = Claude may type into fields and ask to submit forms.")
    max_steps = st.slider("Max steps (Claude turns)", 3, 40, defaults.max_steps)
    allow_raw = st.text_area("Allow list (empty = any domain)", "\n".join(defaults.allow),
                             placeholder="news.ycombinator.com\nwikipedia.org", height=90)
    deny_raw = st.text_area("Deny list (always blocked)", "\n".join(defaults.deny),
                            placeholder="facebook.com\nmybank.com", height=90)
    st.caption("Form submissions always need your approval here first.")
    st.subheader("Model & browser")
    model = st.text_input("Model", defaults.model)
    effort = st.selectbox("Effort", ["low", "medium", "high", "xhigh", "max"],
                          index=["low", "medium", "high", "xhigh", "max"].index(defaults.effort))
    headless = st.toggle("Headless browser", value=defaults.headless,
                         help="Off = a visible Chromium window opens on this machine.")

settings = replace(
    defaults, read_only=read_only, max_steps=max_steps, model=model, effort=effort, headless=headless,
    allow=parse_domains(allow_raw), deny=parse_domains(deny_raw),
)


# --- the browser session (restarted only when server-side settings change) -------
def ensure_session() -> PilotSession:
    # These settings are baked into the MCP server at launch, so changing any of
    # them means a fresh server + browser. Model/effort/steps don't need that.
    key = (settings.allow, settings.deny, settings.read_only, settings.headless)
    if st.session_state.get("pilot_key") != key:
        old = st.session_state.pop("pilot", None)
        if old:
            old.close()
        st.session_state.pop("run", None)  # a paused run belonged to the old browser
        with st.spinner("Launching browser..."):
            st.session_state.pilot = PilotSession(settings).start()
        st.session_state.pilot_key = key
    pilot = st.session_state.pilot
    # Same object the agent and LLM read from, so these apply to the next turn.
    pilot.settings.model, pilot.settings.effort, pilot.settings.max_steps = model, effort, max_steps
    return pilot


st.session_state.setdefault("history", [])  # list of turns: {"goal", "events", "shot"}


# --- rendering ----------------------------------------------------------------------
def render(ev: Event, box) -> None:
    """Draw one agent event inside `box` (a status container)."""
    if ev.kind == "text":
        box.markdown(f"💭 {ev.text}")
    elif ev.kind == "tool":
        args = ", ".join(f"{k}={v!r}" for k, v in (ev.args or {}).items())
        box.markdown(f"**→ `{ev.tool}({args})`**")
    elif ev.kind == "result":
        first = (ev.text.splitlines() or [""])[0]
        if ev.is_error:
            box.error(ev.text[:500])
        elif ev.tool == "read_page":
            with box.expander(f"page: {first[:90]}"):
                st.text(ev.text[:4000])
        else:
            box.caption(first[:300])
    elif ev.kind == "approval":
        box.warning("Waiting for your approval to submit a form.")
    elif ev.kind == "limit":
        box.warning(ev.text)
    elif ev.kind == "error":
        box.error(ev.text)


def drive(turn: dict, approved: bool | None = None) -> None:
    """Advance the current run, rendering each event live and saving it to history."""
    pilot = ensure_session()
    run: AgentRun = st.session_state.run
    log_col, shot_col = st.columns([3, 2])
    with log_col:
        status = st.status("Browsing...", expanded=True)
        for old in turn["events"]:  # replay this turn's earlier steps
            if old.kind not in {"final", "image"}:
                render(old, status)
    with shot_col:
        shot_slot = st.empty()
        if turn["shot"]:
            shot_slot.image(turn["shot"], caption="Latest screenshot")
    for ev in pilot.advance(run, approved):
        turn["events"].append(ev)
        if ev.kind == "image":
            turn["shot"] = ev.image
            shot_slot.image(ev.image, caption="Latest screenshot")
        elif ev.kind == "final":
            turn["answer"] = ev.text
        else:
            render(ev, status)
    if run.awaiting_approval:
        status.update(label="Paused: approval needed", state="running")
    else:
        status.update(label=f"Done in {run.steps} steps", state="complete", expanded=False)
        st.session_state.pop("run", None)
    st.rerun()  # redraw from history (shows the answer, or the approval buttons)


def show_turn(turn: dict, live: bool) -> None:
    """Re-draw a finished (or paused) turn from history."""
    with st.chat_message("user"):
        st.markdown(turn["goal"])
    with st.chat_message("assistant"):
        log_col, shot_col = st.columns([3, 2])
        with log_col:
            n_tools = sum(e.kind == "tool" for e in turn["events"])
            with st.expander(f"Steps ({n_tools} tool calls)", expanded=live):
                for ev in turn["events"]:
                    if ev.kind not in {"final", "image"}:
                        render(ev, st)
            if turn.get("answer"):
                st.markdown(turn["answer"])
            # Failures shouldn't hide inside the collapsed step log.
            for ev in turn["events"]:
                if ev.kind == "error":
                    hint = ("  \nSet `ANTHROPIC_API_KEY` in `.env` (see `.env.example`) and restart."
                            if "authentication" in ev.text.lower() else "")
                    st.error(f"Run failed: {ev.text}{hint}")
        with shot_col:
            if turn["shot"]:
                st.image(turn["shot"], caption="Latest screenshot")


# --- main page ---------------------------------------------------------------------
st.title("WebPilot")
st.caption(
    f"{'🔒 Read-only' if settings.read_only else '✍️ Typing allowed, submits need approval'} · "
    f"max {settings.max_steps} steps · "
    f"allow: {', '.join(settings.allow) or 'any'} · deny: {', '.join(settings.deny) or 'none'}"
)

history = st.session_state.history
paused = "run" in st.session_state and st.session_state.run.awaiting_approval
for i, turn in enumerate(history):
    show_turn(turn, live=paused and i == len(history) - 1)

# The confirmation prompt: shown while a run is paused on submit_form.
if paused:
    turn = history[-1]
    ask = next(e for e in reversed(turn["events"]) if e.kind == "approval")
    with st.container(border=True):
        st.subheader("⚠️ Approve form submission?")
        st.markdown(f"Claude wants to call `{ask.tool}({ask.args})`. The form it would send:")
        st.code(ask.text or "(could not describe the form)", language="json")
        yes, no, _ = st.columns([1, 1, 4])
        if yes.button("Approve", type="primary"):
            drive(turn, approved=True)
        if no.button("Reject"):
            drive(turn, approved=False)

if not history:
    st.markdown("Try one of these:")
    for ex in EXAMPLES:
        if st.button(ex):
            st.session_state.queued_goal = ex
            st.rerun()

goal = st.chat_input("What should the browser do?", disabled=paused) or st.session_state.pop("queued_goal", None)
if goal:
    pilot = ensure_session()
    st.session_state.run = pilot.new_run(goal)
    turn = {"goal": goal, "events": [], "shot": None}
    history.append(turn)
    with st.chat_message("user"):
        st.markdown(goal)
    with st.chat_message("assistant"):
        drive(turn)
