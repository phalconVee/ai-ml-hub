"""Streamlit UI: upload a paper, watch the agents work, read the report.

Layout only. All the logic lives in the `paperlab` package; this file calls
`analyse_pdf` once and draws the result in four tabs.

Run:  uv run streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from paperlab.config import SETTINGS
from paperlab.llm import AnalysisError, make_client
from paperlab.pipeline import AnalysisResult, analyse_pdf
from paperlab.report import BADGE, to_markdown

load_dotenv()  # pick up ANTHROPIC_API_KEY from .env, if present

st.set_page_config(page_title="Paper Lab Analyst", page_icon="🔬", layout="wide")


def get_client():
    """The Claude client. Tests swap this out via st.session_state["client"]."""
    if "client" not in st.session_state:
        st.session_state["client"] = make_client()
    return st.session_state["client"]


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔬 Paper Lab Analyst")
    st.caption("Upload a scientific paper (PDF). A small team of Claude agents reads the "
               "text and the figures, cross-checks each other, and writes a report where "
               "every quote is looked up in the PDF.")
    st.markdown(f"**Model:** `{SETTINGS.model}` · effort `{SETTINGS.effort}`  \n"
                f"**Vision budget:** ≤{SETTINGS.max_figures} figures, "
                f"≤{SETTINGS.max_ocr_pages} scanned pages")
    if not os.getenv("ANTHROPIC_API_KEY") and "client" not in st.session_state:
        st.info("No ANTHROPIC_API_KEY found. Add it to `.env` (or run `ant auth login`).")

uploaded = st.file_uploader("Paper PDF", type=["pdf"])

if uploaded is not None and st.button("Analyse paper", type="primary"):
    with st.status("Analysing…", expanded=True) as status:
        try:
            result = analyse_pdf(uploaded.getvalue(), get_client(), on_event=st.write)
            st.session_state["result"] = result
            status.update(label="Done", state="complete", expanded=False)
        except AnalysisError as exc:
            status.update(label="Analysis stopped", state="error")
            st.error(str(exc))

result: AnalysisResult | None = st.session_state.get("result")
if result is None:
    st.stop()

r = result.report
st.title(r["title"])
st.caption(f"{r['field']} · {result.paper.total_pages} pages · {len(result.paper.figures)} figures")
st.markdown(f"**{r['one_line_summary']}**")

g = result.grounding
c1, c2, c3 = st.columns(3)
c1.metric("✅ Quotes verified", g["verified"])
c2.metric("📄 On another page", g["wrong_page"])
c3.metric("⚠️ Not found in PDF", g["not_found"])


def show_evidence(item: dict) -> None:
    ev, gr = item["evidence"], item.get("grounding", {})
    note = ""
    if gr.get("status") == "wrong_page":
        note = f" — actually on p. {gr['found_page']}"
    elif gr.get("status") == "not_found":
        note = " — not found in the PDF, check by hand"
    st.caption(f'{BADGE.get(gr.get("status"), "")} "{ev["quote"]}" (p. {ev["page"]}){note}')


report_tab, figures_tab, drafts_tab, trace_tab = st.tabs(
    ["📋 Report", "🖼️ Figures", "📝 Analyst drafts", "🔎 Trace"])

with report_tab:
    st.subheader("Executive summary")
    st.write(r["executive_summary"])
    st.subheader("Research question")
    st.write(r["research_question"])

    st.subheader("Methods")
    m = r["methods"]
    st.markdown(f"- **Design:** {m['study_design']}\n- **Data:** {m['data']}\n"
                f"- **Techniques:** {', '.join(m['techniques'])}\n- **Statistics:** {m['statistics']}")

    st.subheader("Key findings")
    for f in r["key_findings"]:
        figs = f"  ·  {', '.join(f['figure_ids'])}" if f["figure_ids"] else ""
        st.markdown(f"**{f['claim']}**  \n`confidence: {f['confidence']}`{figs}")
        show_evidence(f)

    st.subheader("Limitations")
    for lim in r["limitations"]:
        who = "👩‍🔬 authors" if lim["source"] == "stated_by_authors" else "🧑‍⚖️ reviewer"
        st.markdown(f"**[{who}]** {lim['limitation']}")
        show_evidence(lim)

    left, right = st.columns(2)
    with left:
        st.subheader("Open questions")
        st.markdown("\n".join(f"- {q}" for q in r["open_questions"]) or "—")
    with right:
        st.subheader("Practical implications")
        st.markdown("\n".join(f"- {p}" for p in r["practical_implications"]) or "—")

    if r["reviewer_notes"]:
        with st.expander("Reviewer notes (what changed vs. the drafts)"):
            st.markdown("\n".join(f"- {n}" for n in r["reviewer_notes"]))

    st.download_button("⬇️ Download report (Markdown)", to_markdown(result),
                       file_name="paper-report.md", mime="text/markdown")

with figures_tab:
    insights = {i["figure_id"]: i["insight"] for i in r["figure_insights"]}
    if not result.paper.figures:
        st.info("No figures were detected in this PDF.")
    for fig in result.paper.figures:
        img_col, text_col = st.columns([1, 1])
        img_col.image(fig.image_png, caption=f"{fig.fig_id} · {fig.label or 'unlabelled'} · p.{fig.page}")
        with text_col:
            if fig.caption:
                st.caption(fig.caption)
            if fig.fig_id in insights:
                st.markdown(f"**Reviewer insight:** {insights[fig.fig_id]}")
            note = result.figure_notes.get(fig.fig_id)
            if note:
                st.markdown(f"**{note['figure_type']}.** {note['what_it_shows']}")
                st.markdown(f"**Takeaway:** {note['key_takeaway']}")
                for d in note["quantitative_details"]:
                    st.markdown(f"- {d}")
                for c in note["caveats"]:
                    st.markdown(f"- ⚠️ {c}")
            else:
                st.caption("Not analysed (over the max_figures budget).")
        st.divider()

with drafts_tab:
    st.caption("The analysts' first drafts, before the reviewer checked them.")
    st.subheader("Methods draft")
    st.json(result.methods_draft, expanded=False)
    st.subheader("Findings draft")
    st.json(result.findings_draft, expanded=False)

with trace_tab:
    st.code("\n".join(result.trace), language=None)
