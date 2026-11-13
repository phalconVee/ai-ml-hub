"""Streamlit front end: type a question, watch the crew work, read the report.

    uv run streamlit run app.py

Streamlit reruns this whole script top to bottom on every interaction. So
the finished report is kept in `st.session_state`, otherwise clicking the
download button (which triggers a rerun) would wipe it from the page.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace

import anthropic
import streamlit as st

from deepdive.agent import Event
from deepdive.config import DEPTH_PRESETS, MODEL, estimate_cost
from deepdive.crew import ResearchReport, run_research

st.set_page_config(page_title="DeepDive Research Crew", page_icon="🔎", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: keys and settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    # Keys typed here live only in this browser session's memory. If the key
    # is already in the environment (.env), we never echo it back.
    env_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        placeholder="set in .env" if env_key else "sk-ant-...",
        help="Leave empty to use ANTHROPIC_API_KEY from your environment / .env.",
    )

    backend = st.selectbox(
        "Search backend",
        ["duckduckgo", "tavily", "fake"],
        index=["duckduckgo", "tavily", "fake"].index(os.getenv("DEEPDIVE_SEARCH_BACKEND", "duckduckgo")),
        help="duckduckgo: free, no key. tavily: better results, needs a key. fake: offline canned pages.",
    )
    tavily_key = ""
    if backend == "tavily":
        tavily_key = st.text_input("Tavily API key", type="password", placeholder="tvly-... (or set in .env)")

    depth = st.select_slider("Depth", options=list(DEPTH_PRESETS), value="standard")
    preset = DEPTH_PRESETS[depth]
    st.caption(
        f"{preset.num_angles} scouts · up to {preset.scout_max_turns} tool rounds each · "
        f"analyst {preset.analyst_max_turns} rounds"
    )
    model = st.text_input("Model", value=MODEL)

# ---------------------------------------------------------------------------
# Main column
# ---------------------------------------------------------------------------
st.title("🔎 DeepDive Research Crew")
st.write(
    "A planner splits your question into angles, scouts research each angle on the web "
    "through an **MCP** tools server, an analyst cross-checks them, and a writer produces a cited report."
)

question = st.text_area(
    "What should the crew research?",
    placeholder="e.g. How far has small modular reactor deployment actually got by 2026, and what is holding it back?",
    height=90,
)
go = st.button("Research", type="primary", disabled=not question.strip())


def run_crew(q: str) -> ResearchReport:
    # The tools server is a subprocess; it reads its settings from env vars
    # that `mcp_bridge` passes through. Set them before it starts.
    os.environ["DEEPDIVE_SEARCH_BACKEND"] = backend
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else anthropic.AsyncAnthropic()
    settings = replace(preset, model=model)

    with st.status("The crew is working...", expanded=True) as status:

        def on_event(e: Event) -> None:
            # Called from inside the async crew, on this same thread, so
            # writing to the page here is safe.
            if e.kind == "stage":
                status.update(label=e.text)
                st.markdown(f"**{e.text}**")
            elif e.kind == "tool":
                st.caption(f"`{e.agent}` → {e.text}")
            elif e.kind == "warning":
                st.warning(f"{e.agent}: {e.text}")
            elif e.kind == "done":
                st.caption(f"`{e.agent}` {e.text}")

        report = asyncio.run(run_research(q, settings, client=client, on_event=on_event))
        status.update(label="Research complete", state="complete", expanded=False)
    return report


if go:
    try:
        st.session_state["report"] = run_crew(question.strip())
    except anthropic.AuthenticationError:
        st.error("The Anthropic API key was rejected. Add a valid key in the sidebar or in .env.")
    except anthropic.APIConnectionError:
        st.error("Could not reach the Anthropic API. Check your connection.")
    except anthropic.RateLimitError:
        st.error("Rate limited by the Anthropic API. Wait a minute or pick a smaller depth.")
    except Exception as exc:  # show anything else rather than a blank page
        st.exception(exc)

# ---------------------------------------------------------------------------
# Results (from session_state, so they survive reruns)
# ---------------------------------------------------------------------------
report: ResearchReport | None = st.session_state.get("report")
if report:
    u = report.usage
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Angles", len(report.angles))
    c2.metric("Pages read", len(report.sources))
    c3.metric("Tool calls", len(report.tool_calls))
    c4.metric("Est. cost", f"${estimate_cost(u.input_tokens, u.output_tokens):.2f}",
              help=f"{u.input_tokens:,} input + {u.output_tokens:,} output tokens")

    tab_report, tab_analysis, tab_notes, tab_activity = st.tabs(["Report", "Analyst", "Scout notes", "Activity"])
    with tab_report:
        st.markdown(report.report)
        st.download_button("Download Markdown", report.report, file_name="deepdive-report.md", mime="text/markdown")
    with tab_analysis:
        st.markdown(report.analysis)
    with tab_notes:
        for angle in report.angles:
            with st.expander(angle.title):
                st.caption(angle.question)
                st.markdown(report.notes.get(angle.title, ""))
    with tab_activity:
        st.dataframe(
            [
                {"agent": c.agent, "tool": c.name, "arguments": str(c.arguments), "error": c.is_error, "result": c.preview}
                for c in report.tool_calls
            ],
            hide_index=True,
        )
