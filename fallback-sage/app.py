"""Streamlit UI. Run with:  uv run streamlit run app.py

Streamlit re-runs this whole script top to bottom on every click or message.
Anything that must survive a re-run (the search index, the chat history) is
kept in `st.session_state`.
"""

import streamlit as st
from dotenv import load_dotenv

from sage.bm25 import BM25Index
from sage.documents import build_chunks
from sage.llm import make_client
from sage.router import answer

load_dotenv()  # picks up ANTHROPIC_API_KEY from a local .env file, if present

st.set_page_config(page_title="Fallback Sage", page_icon="🦉", layout="wide")
st.title("🦉 Fallback Sage")
st.caption("Answers from your documents first. Searches the web only when they come up empty.")

# ── Sidebar: documents and options ──────────────────────────────────────────
with st.sidebar:
    st.header("Your documents")
    uploads = st.file_uploader("PDF, TXT or Markdown", type=["pdf", "txt", "md"],
                               accept_multiple_files=True)
    allow_web = st.toggle("Fall back to web search", value=True)

    # Rebuild the index only when the set of files changes. The "signature"
    # is (name, size) per file: cheap to compute, and changes when files do.
    signature = tuple((f.name, f.size) for f in uploads or [])
    if st.session_state.get("signature") != signature:
        with st.spinner("Reading and indexing…"):
            chunks = build_chunks([(f.name, f.getvalue()) for f in uploads or []])
            st.session_state.index = BM25Index(chunks) if chunks else None
            st.session_state.signature = signature

    index = st.session_state.get("index")
    if index is not None:
        st.success(f"{len(signature)} file(s) → {len(index)} searchable chunks")
    elif uploads:
        st.warning("No text could be extracted (scanned PDFs need OCR first).")
    else:
        st.info("No documents yet: questions will go straight to the web.")

    if st.button("Clear chat"):
        st.session_state.history = []

# ── Chat ────────────────────────────────────────────────────────────────────
BADGES = {"docs": "📚 From your documents", "web": "🌐 From the web", "none": "🤷 Not found"}


def render(ans) -> None:
    """Draw one assistant answer: badge, text, sources, and the routing trace."""
    st.markdown(f"**{BADGES[ans.route]}**")
    st.markdown(ans.text)
    if ans.sources:
        with st.expander(f"Sources ({len(ans.sources)})"):
            for s in ans.sources:
                if "url" in s:  # web source
                    st.markdown(f"- [{s['title']}]({s['url']})")
                else:           # document passage
                    st.markdown(f"**{s['id']}** (page {s['page']})")
                    st.caption(s["snippet"] + "…")
    with st.expander("How I got here"):
        for step in ans.trace:
            st.write(step)


st.session_state.setdefault("history", [])  # list of (question, Answer)
for q, a in st.session_state.history:      # replay earlier turns after each re-run
    st.chat_message("user").write(q)
    with st.chat_message("assistant"):
        render(a)

if question := st.chat_input("Ask about your documents…"):
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking your documents…"):
            try:
                result = answer(question, index, make_client(), allow_web=allow_web)
            except Exception as e:  # missing key, network error, rate limit…
                st.error(f"{type(e).__name__}: {e}")
                st.stop()
        render(result)
    st.session_state.history.append((question, result))
