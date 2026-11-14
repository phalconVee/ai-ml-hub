"""Streamlit UI. Run:  uv run streamlit run app.py

Sidebar: upload PDFs (or load the sample lease), pick a jurisdiction, toggle
web research. Main pane: a chat. Each answer shows the notice, the cited
answer, the sources it cites, and a trace of what each agent did.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from counsel import NOT_LEGAL_ADVICE, ClauseIndex, chunk_document, make_client, run_crew

load_dotenv()   # picks up ANTHROPIC_API_KEY from .env, if present
SAMPLE = Path(__file__).parent / "sample_docs" / "sample_lease.pdf"

st.set_page_config(page_title="counsel-copilot", page_icon="⚖️", layout="wide")
st.title("⚖️ counsel-copilot")
st.caption("A paralegal-style crew: Clause Finder → Legal Researcher → Drafter.")
st.warning(NOT_LEGAL_ADVICE)

# ── Session state: survives Streamlit's rerun-on-every-click model ──────────
if "index" not in st.session_state:
    st.session_state.index = ClauseIndex()
    st.session_state.files = set()      # names already indexed (avoid double-adding)
    st.session_state.chat = []          # [(role, payload)]
index: ClauseIndex = st.session_state.index


def add_file(name: str, data: bytes) -> None:
    if name in st.session_state.files:
        return
    chunks = chunk_document(data, name)
    index.add(chunks)
    st.session_state.files.add(name)
    st.sidebar.success(f"{name}: {len(chunks)} clauses indexed")


# ── Sidebar: documents and options ──────────────────────────────────────────
with st.sidebar:
    st.header("Documents")
    for f in st.file_uploader("Contracts, policies, case documents",
                              type=["pdf", "txt", "md"], accept_multiple_files=True) or []:
        add_file(f.name, f.getvalue())
    if SAMPLE.exists() and st.button("Load the sample lease"):
        add_file(SAMPLE.name, SAMPLE.read_bytes())
    st.write(f"**Indexed:** {', '.join(index.docs) or 'nothing yet'}")

    st.header("Options")
    jurisdiction = st.text_input("Jurisdiction (optional)", placeholder="e.g. California, USA")
    allow_web = st.toggle("Allow web research", value=True,
                          help="Lets the Legal Researcher search the web when your documents "
                               "don't settle the question.")


def render(result) -> None:
    """Draw one crew answer: answer, sources, and the agents' trace."""
    st.markdown(result.markdown())
    if result.draft.citations:
        with st.expander(f"Sources ({len(result.draft.citations)})", expanded=True):
            for c in result.draft.citations:
                where = f"[{c.title}]({c.source})" if c.kind == "web" else f"**{c.title}**"
                st.markdown(f"**[{c.number}]** {'🌐' if c.kind == 'web' else '📄'} {where}")
                st.caption(f"“{c.quote}”")
    with st.expander("How the crew worked"):
        r = result.report
        st.markdown(f"**Route:** {' → '.join(result.route)}")
        st.markdown(f"**Clause Finder** · coverage `{r.coverage}` · searches: "
                    + ", ".join(f"`{q}`" for q in r.queries))
        if r.legal_questions:
            st.markdown("Legal questions raised: " + "; ".join(r.legal_questions))
        if r.dropped_ids:
            st.markdown(f"Dropped unverified ids: {r.dropped_ids}")
        if result.research is not None:
            st.markdown("**Legal Researcher** · web searches: "
                        + (", ".join(f"`{q}`" for q in result.research.queries) or "none"))


# ── Chat ────────────────────────────────────────────────────────────────────
for role, payload in st.session_state.chat:
    with st.chat_message(role):
        render(payload) if role == "assistant" else st.markdown(payload)

if question := st.chat_input("Ask about your documents, e.g. 'Can I end the lease early?'"):
    st.session_state.chat.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        client = st.session_state.get("client") or make_client()
        st.session_state.client = client
        result = None
        with st.status("The crew is working…", expanded=True) as status:
            try:
                result = run_crew(question, index, client, jurisdiction=jurisdiction,
                                  allow_web=allow_web, on_event=status.write)
                status.update(label="Done", state="complete", expanded=False)
            except Exception as e:  # missing key, network, rate limit: show it, don't crash
                status.update(label="The crew hit an error", state="error", expanded=True)
                st.error(f"{type(e).__name__}: {e}")
                if "authentication" in str(e).lower():
                    st.info("Set ANTHROPIC_API_KEY in a .env file next to app.py (see .env.example).")
        if result is not None:
            render(result)
            st.session_state.chat.append(("assistant", result))
