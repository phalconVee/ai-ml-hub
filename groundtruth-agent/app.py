"""Brick 6: the Streamlit UI.

Run with:  uv run streamlit run app.py

Layout:
- Sidebar: upload PDFs, choose whether figures get a vision description, build the index.
- Main: a chat. Each answer shows inline [n] markers, the quoted passages
  behind them, and the searches the agent ran to get there.

Streamlit re-runs this whole file on every click, so anything that must
survive (the index, the chat) lives in `st.session_state`.
"""

from __future__ import annotations

import os

import streamlit as st

from groundtruth import BM25Index, ask, ingest_pdf, make_claude_describer
from groundtruth.config import SETTINGS

st.set_page_config(page_title="GroundTruth Agent", page_icon="📑", layout="wide")

# --- Session state: created once, then kept across re-runs ----------------
if "index" not in st.session_state:
    st.session_state.index = BM25Index()
    st.session_state.ingested = {}      # file name -> stats dict
    st.session_state.chat = []          # list of {"role", "content", "answer"?}
    st.session_state.history = []       # raw API messages for follow-up questions

index: BM25Index = st.session_state.index
has_key = bool(os.getenv("ANTHROPIC_API_KEY"))

# --- Sidebar: documents -----------------------------------------------------
with st.sidebar:
    st.header("Documents")
    if not has_key:
        st.warning("ANTHROPIC_API_KEY is not set. Copy `.env.example` to `.env` and add your key.")

    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    describe = st.checkbox(
        "Describe figures with Claude vision",
        value=has_key,
        disabled=not has_key,
        help=f"One vision call per figure (max {SETTINGS.max_figures_per_doc} per document). "
             "Off = figures are indexed by caption only.",
    )

    new_files = [f for f in uploads or [] if f.name not in st.session_state.ingested]
    if new_files and st.button(f"Parse and index {len(new_files)} file(s)", type="primary"):
        describer = make_claude_describer() if describe else None
        for f in new_files:
            with st.spinner(f"Parsing {f.name}…"):
                st.session_state.ingested[f.name] = ingest_pdf(index, f.getvalue(), f.name, describer)
        st.rerun()

    # Show what each document turned into: proof that tables/figures were found.
    for name, stats in st.session_state.ingested.items():
        st.markdown(
            f"**{name}**  \n{stats.get('pages', 0)} pages · {stats.get('text', 0)} text · "
            f"{stats.get('table', 0)} tables · {stats.get('figure', 0)} figures"
        )

    if st.session_state.ingested and st.button("Reset everything"):
        for key in ("index", "ingested", "chat", "history"):
            del st.session_state[key]
        st.rerun()

# --- Main: chat ---------------------------------------------------------------
st.title("📑 GroundTruth Agent")
st.caption(f"Answers come only from your PDFs, with a citation on every claim · model: `{SETTINGS.model}`")


def render_answer(answer) -> None:
    """Answer text with [n] markers, then the evidence and the agent's search trail."""
    st.markdown(answer.markdown())
    if answer.citations:
        with st.expander(f"Sources ({len(answer.citations)})", expanded=True):
            for c in answer.citations:
                st.markdown(f"**[{c.number}] {c.title}**")
                st.markdown("> " + c.quote.replace("\n", "\n> "))
    if answer.searches:
        with st.expander(f"Agent trace: {len(answer.searches)} search(es)"):
            for s in answer.searches:
                scope = f" in `{s['document']}`" if s["document"] else ""
                st.markdown(f"- `{s['query']}`{scope} → {s['hits']} passage(s)")


# Replay the conversation so far (Streamlit redraws from scratch every run).
for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        if turn.get("answer"):
            render_answer(turn["answer"])
        else:
            st.markdown(turn["content"])

ready = len(index) > 0 and has_key
prompt = st.chat_input(
    "Ask about your documents…" if ready else "Upload and index a PDF first",
    disabled=not ready,
)

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Thinking…", expanded=False) as status:
            answer = ask(
                prompt,
                index,
                history=st.session_state.history,
                on_event=lambda msg: status.update(label=msg),
            )
            status.update(label=f"Done after {len(answer.searches)} search(es)", state="complete")
        render_answer(answer)

    st.session_state.history = answer.messages
    st.session_state.chat.append({"role": "assistant", "content": answer.markdown(), "answer": answer})
