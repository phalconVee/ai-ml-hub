"""Streamlit front end for brand-radar.

    uv run streamlit run app.py

Layout:
  sidebar   brand, context, aliases, exclusions, look-back, sources, sweep toggle
  main      live progress -> headline metrics -> charts -> report -> per-source tabs

All the real work happens in radar/pipeline.py; this file only collects inputs,
shows progress and draws results. The last run is kept in st.session_state so
clicking around the tabs doesn't re-run (and re-pay for) the whole radar.
"""

import asyncio

import pandas as pd
import streamlit as st

from radar.__main__ import to_json
from radar.config import CLAUDE_SWEEP, SOURCES
from radar.pipeline import run_radar
from radar.schemas import BrandQuery

st.set_page_config(page_title="Brand Radar", page_icon="📡", layout="wide")
st.title("📡 Brand Radar")
st.caption("Multi-agent brand monitoring: free sources → one Claude analyst per source → a strategist's action plan.")


def split(text: str) -> list[str]:
    """'a, b,c' -> ['a', 'b', 'c']"""
    return [t.strip() for t in text.split(",") if t.strip()]


# --- Inputs ----------------------------------------------------------------------
with st.sidebar:
    brand = st.text_input("Brand", placeholder="e.g. Notion")
    context = st.text_input("What is it? (rules out namesakes)", placeholder="e.g. notes and docs app")
    aliases = st.text_input("Other names (comma-separated)", placeholder="Notion AI, notion.so")
    exclude = st.text_input("Ignore mentions containing", placeholder="notion of, the notion that")
    days = st.slider("Look back (days)", 1, 30, 7)
    picked = st.multiselect("Sources (all free, no keys)", list(SOURCES), default=list(SOURCES),
                            format_func=lambda s: SOURCES[s])
    sweep = st.toggle("Add Claude web-search sweep", value=False,
                      help="Claude searches the web itself for mentions the free sources missed. "
                           "Billed per search on your Anthropic key.")
    go = st.button("Run radar", type="primary", disabled=not brand.strip() or not (picked or sweep))

# --- Run -------------------------------------------------------------------------
if go:
    q = BrandQuery(brand.strip(), context.strip(), split(aliases), split(exclude), days)
    status = st.status("Running the radar...", expanded=True)
    report_box = st.empty()          # the strategist's report types itself in here
    streamed: list[str] = []

    def on_event(stage: str, message: str) -> None:
        status.write(message)

    def on_report_text(chunk: str) -> None:
        streamed.append(chunk)
        report_box.markdown("".join(streamed))

    try:
        st.session_state.run = asyncio.run(
            run_radar(q, picked, use_sweep=sweep, on_event=on_event, on_report_text=on_report_text))
        status.update(label="Radar complete", state="complete", expanded=False)
    except Exception as e:  # noqa: BLE001 - show any failure in the UI rather than a stack trace
        status.update(label="Radar failed", state="error")
        st.error(f"{type(e).__name__}: {e}")
        st.stop()
    report_box.empty()               # the full report is redrawn below

run = st.session_state.get("run")
if not run:
    st.info("Enter a brand in the sidebar and press **Run radar**. "
            "Tip: fill in *What is it?* for brands with common-word names.")
    st.stop()

m = run.metrics

# --- Headline numbers --------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mentions collected", m["collected"])
c2.metric("Relevant", m["relevant"])
c3.metric("Net sentiment", "n/a" if m["net_sentiment"] is None else f"{m['net_sentiment']:+d}",
          help="(positive − negative) ÷ relevant, from −100 to +100")
c4.metric("Actions flagged", sum(m["actions"].values()))

for w in run.warnings:
    st.warning(w, icon="⚠️")

# --- Charts --------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Sentiment by source")
    rows = {s: d["sentiment"] for s, d in m["per_source"].items() if d["relevant"]}
    if rows:
        df = pd.DataFrame(rows).T.fillna(0).astype(int)
        order = [c for c in ["positive", "neutral", "mixed", "negative"] if c in df.columns]
        st.bar_chart(df[order], horizontal=True)
    else:
        st.caption("No relevant mentions.")
with right:
    st.subheader("Top themes")
    if m["top_themes"]:
        st.bar_chart(pd.DataFrame(m["top_themes"]).set_index("theme")["count"], horizontal=True)
    else:
        st.caption("No themes.")

# --- Report ----------------------------------------------------------------------------
st.subheader("Report")
st.markdown(run.report_md)
st.download_button("Download report (.md)", run.report_md, f"{run.query.brand}-radar.md")
st.download_button("Download full run (.json)", to_json(run), f"{run.query.brand}-radar.json")

# --- Evidence, one tab per source ------------------------------------------------------------
st.subheader("Evidence")
names = [n for n in run.sources if run.sources[n].mentions or run.sources[n].error]
if names:
    labels = [SOURCES.get(n, "Claude web-search sweep" if n == CLAUDE_SWEEP else n).split(" (")[0] for n in names]
    for tab, name in zip(st.tabs(labels), names):
        rep = run.sources[name]
        with tab:
            if rep.error:
                st.error(rep.error)
            if rep.summary:
                st.write(rep.summary)
            table = []
            for mn in rep.mentions:
                v = rep.verdicts.get(mn.id)
                table.append({
                    "id": mn.id, "relevant": v.relevant if v else None,
                    "sentiment": v.sentiment if v else "", "action": v.action if v else "",
                    "urgency": v.urgency if v else None, "summary": v.summary if v else mn.title,
                    "themes": ", ".join(v.themes) if v else "", "reach": mn.engagement, "url": mn.url,
                })
            if table:
                st.dataframe(pd.DataFrame(table), hide_index=True,
                             column_config={"url": st.column_config.LinkColumn("link")})
