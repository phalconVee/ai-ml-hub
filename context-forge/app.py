"""context-forge, the Streamlit UI.   Run:  uv run streamlit run app.py

Layout
  sidebar  upload documents · pick sources · token budget · LLM judge · memory
  main     chat; under each answer, four tabs that open up the context engineering:
             Sources      the citations, numbered as in the answer
             Window       every item that made it in, with its score and reason
             Rejected     everything gathered but left out, and why
             Trace        planner decision, queries, timings, the raw window text

Streamlit re-runs this whole file on every click, so anything that must survive
a click (the document store, the chat history) lives in `st.session_state`.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from forge.config import SETTINGS
from forge.llm import make_client
from forge.pipeline import run
from forge.sources.documents import DocumentStore, read_file
from forge.sources.memory import MemoryStore

load_dotenv()  # picks up ANTHROPIC_API_KEY from .env if present
HERE = Path(__file__).parent
SOURCE_LABELS = {"doc": "📄 Your documents", "web": "🌐 Web search",
                 "paper": "🎓 arXiv papers", "memory": "🧠 Conversation memory"}

st.set_page_config(page_title="context-forge", page_icon="🔥", layout="wide")

# ── Session state: created once, then reused across re-runs ───────────────────
ss = st.session_state
if "docs" not in ss:
    ss.docs = DocumentStore()
    ss.loaded = set()           # file names already chunked (avoid re-adding on re-run)
    ss.history = []             # [(question, Run)]
if "memory" not in ss:
    ss.memory = MemoryStore(HERE / ".forge_memory" / "memory.json")
if "client" not in ss:
    ss.client = None            # built lazily, so the page renders without a key

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1 · Documents")
    files = st.file_uploader("PDF, Markdown or text", type=["pdf", "md", "txt"],
                             accept_multiple_files=True)
    for f in files or []:
        if f.name not in ss.loaded:
            n = ss.docs.add(f.name, read_file(f.name, f.getvalue()))
            ss.loaded.add(f.name)
            st.toast(f"{f.name}: {n} chunks")
    if st.button("Load sample notes"):
        p = HERE / "sample_docs" / "context-notes.md"
        if p.name not in ss.loaded:
            ss.docs.add(p.name, p.read_text())
            ss.loaded.add(p.name)
    st.caption(f"{len(ss.docs.chunks)} chunks from {len(ss.docs.names)} file(s)")

    st.header("2 · Sources")
    enabled = {k for k, label in SOURCE_LABELS.items() if st.checkbox(label, value=True)}

    st.header("3 · Window")
    budget = st.slider("Token budget", 500, 8000, SETTINGS.token_budget, step=250,
                       help="How many tokens of context the answer may see. "
                            "Shrink it and watch the Rejected tab grow.")
    use_judge = st.toggle("LLM relevance judge", value=False,
                          help="One extra Claude call grades every candidate 0-10 and "
                               "blends that with BM25. Better ranking, a little slower.")

    st.header("4 · Memory")
    st.caption(f"{len(ss.memory.turns)} past turn(s) stored locally")
    if st.button("Forget everything"):
        ss.memory.clear()
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🔥 context-forge")
st.caption("Gathers context from your documents, the web, arXiv and memory, "
           "ranks it into a token budget, and answers with checked citations.")


def show_run(r) -> None:
    """Render one finished run: answer + the four context-engineering tabs."""
    st.markdown(r.answer.text)
    if r.answer.invalid:
        st.warning(f"Removed citations to blocks that weren't in the window: {', '.join(r.answer.invalid)}")
    for name, err in r.errors.items():
        st.error(f"{SOURCE_LABELS.get(name, name)} failed: {err}")

    dropped = [c for c in r.candidates if not c.kept]
    used = sum(c.tokens for c in r.kept)
    tabs = st.tabs([f"Sources ({len(r.answer.cited)})", f"Window ({len(r.kept)} · {used} tok)",
                    f"Rejected ({len(dropped)})", "Trace"])

    with tabs[0]:  # citations, numbered as in the answer
        by_id = {c.id: c for c in r.kept}
        for sid in r.answer.cited:
            c = by_id[sid]
            link = f" — [{c.url}]({c.url})" if c.url else ""
            st.markdown(f"**[{sid}]** {SOURCE_LABELS[c.source]} · {c.title}{link}")

    with tabs[1]:  # what got in, and why
        for c in r.kept:
            with st.expander(f"[{c.id}] {SOURCE_LABELS[c.source]} · {c.title[:80]} · score {c.score:.2f}"):
                st.caption(c.reason)
                st.write(c.text)

    with tabs[2]:  # what was left out, and why
        if not dropped:
            st.caption("Nothing was rejected.")
        st.dataframe(
            [{"source": c.source, "title": c.title[:70], "score": round(c.score, 2),
              "tokens": c.tokens, "why dropped": c.reason} for c in dropped],
            hide_index=True, width="stretch",
        )

    with tabs[3]:  # the agent's decisions
        p = r.plan
        st.markdown(f"**Planner** ({'Claude' if p.by_llm else 'fallback'}): {p.rationale}")
        st.json({"doc_query": p.doc_query, "web_query": p.web_query if p.use_web else "(skipped)",
                 "paper_query": p.paper_query if p.use_papers else "(skipped)",
                 "web_searches_run": r.web_queries,
                 "seconds": {k: round(v, 2) for k, v in r.timings.items()}})
        st.markdown("**Exact context window sent to the model**")
        st.code(r.context or "(empty)", language="markdown")


for q, r in ss.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        show_run(r)

question = st.chat_input("Ask a research question…")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            if ss.client is None:
                ss.client = make_client()
            with st.status("Planning → gathering → ranking → answering…") as status:
                result = run(question, client=ss.client, docs=ss.docs, memory=ss.memory,
                             enabled=enabled, use_judge=use_judge, budget=budget)
                status.update(label=f"Done: {len(result.candidates)} candidates → "
                                    f"{len(result.kept)} in the window", state="complete")
            ss.history.append((question, result))
            show_run(result)
        except Exception as e:  # missing key, network, etc.: say so plainly
            st.error(f"{type(e).__name__}: {e}")
