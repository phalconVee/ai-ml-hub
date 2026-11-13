"""Streamlit UI for omni-scout.

    uv run streamlit run app.py

Layout: question box -> live plan -> one status panel per scout (showing each
MCP tool call as it happens) -> the merged report streaming in -> raw findings.
"""

import asyncio
import os

import streamlit as st

from scout.config import PLATFORM_HINTS, settings
from scout.pipeline import research

st.set_page_config(page_title="omni-scout", page_icon="🛰️", layout="wide")

# --- Sidebar: credentials and knobs ------------------------------------------
with st.sidebar:
    st.header("Settings")
    key = st.text_input(
        "Anthropic API key", type="password",
        help="Leave blank if ANTHROPIC_API_KEY is set in your environment or .env.",
    )
    if key:
        # The SDK reads this when the LLM client is created inside research().
        os.environ["ANTHROPIC_API_KEY"] = key
    settings.max_platforms = st.slider("Max platforms", 1, len(PLATFORM_HINTS), settings.max_platforms)
    settings.max_scout_turns = st.slider("Max turns per scout", 2, 10, settings.max_scout_turns)
    st.caption(f"Models: planner `{settings.planner_model}` · scouts `{settings.scout_model}` · synth `{settings.synth_model}`")
    st.divider()
    st.markdown("**Platforms the planner can pick**")
    for name, hint in PLATFORM_HINTS.items():
        st.markdown(f"- **{name}**: {hint}")

st.title("🛰️ omni-scout")
st.caption("One question, several platforms, one cited answer. Searches run through a local MCP server.")

question = st.text_area(
    "Research question",
    placeholder="e.g. How are developers actually using MCP servers in production, and what problems do they hit?",
    height=90,
)
go = st.button("Research", type="primary", disabled=not question.strip())

if go:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("No ANTHROPIC_API_KEY found; the SDK will try an `ant auth login` profile.")

    plan_box = st.container()
    st.subheader("Scouts")
    scouts_area = st.container()
    st.subheader("Report")
    report_box = st.empty()

    # Streamlit widgets are created lazily as events arrive.
    panels: dict = {}  # platform -> st.status panel
    streamed: list[str] = []

    def on_event(kind: str, data: dict) -> None:
        if kind == "plan":
            p = data["plan"]
            with plan_box:
                st.subheader("Plan")
                st.markdown(f"**Restated:** {p.restated_question}")
                st.table([{"platform": t.platform, "brief": t.brief, "queries": ", ".join(t.queries)} for t in p.tasks])
            cols = scouts_area.columns(len(p.tasks))
            for col, t in zip(cols, p.tasks):
                panels[t.platform] = col.status(f"{t.platform}: searching…", expanded=True)
        elif kind == "tool":
            args = ", ".join(f"{k}={v!r}" for k, v in data["args"].items())
            panels[data["platform"]].write(f"`{data['tool']}`({args})")
        elif kind == "scout_done":
            rep = data["report"]
            panels[data["platform"]].update(label=f"{data['platform']}: {len(rep.findings)} findings", state="complete", expanded=False)
        elif kind == "scout_failed":
            panels[data["platform"]].update(label=f"{data['platform']}: failed", state="error")
            panels[data["platform"]].write(data["error"])
        elif kind == "synthesizing":
            report_box.info("Merging findings across platforms…")

    def on_text(chunk: str) -> None:
        streamed.append(chunk)
        report_box.markdown("".join(streamed) + " ▌")

    try:
        result = asyncio.run(research(question.strip(), on_event=on_event, on_text=on_text))
    except Exception as exc:  # show the error in the page instead of a stack trace
        st.error(f"Research failed: {exc}")
        st.stop()

    report_box.markdown(result.report_md)
    st.download_button("Download report (.md)", result.report_md, file_name="omni-scout-report.md")

    with st.expander("Raw findings by platform"):
        for rep in result.reports:
            st.markdown(f"#### {rep.platform}\n{rep.summary}")
            if rep.findings:
                st.dataframe([f.model_dump() for f in rep.findings], width="stretch")
            if rep.gaps:
                st.caption("Gaps: " + "; ".join(rep.gaps))
