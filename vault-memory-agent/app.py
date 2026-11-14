"""Streamlit UI: chat with an agent whose memory you can see.

Layout
- Sidebar: thread picker (memory 2), document upload and list (memory 1).
- "Chat" tab: the conversation; under each answer, the memory it used.
- "Vault" tab: every saved note (memory 3), and the chunks retrieved per answer.

Run with:  uv run streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st

from vault import Vault, make_embedder, run_turn
from vault.config import SETTINGS
from vault.llm import make_client

st.set_page_config(page_title="Vault Memory Agent", page_icon="🗄️", layout="wide")


# --- Shared resources -----------------------------------------------------------
# st.cache_resource builds these once per server process, not on every rerun:
# loading the embedding model takes seconds, opening the DB a few milliseconds.

@st.cache_resource(show_spinner="Loading the embedding model...")
def get_embedder():
    return make_embedder(SETTINGS.embedder)


@st.cache_resource
def get_vault() -> Vault:
    return Vault.open(SETTINGS.db_path, get_embedder())


@st.cache_resource
def get_client():
    return make_client()


vault = get_vault()
store = vault.store


# --- Small render helpers ---------------------------------------------------------

TOOL_LABELS = {"search_knowledge_base": "🔎 searched documents", "recall": "🧠 recalled notes",
               "remember": "📝 saved a note", "forget": "🗑️ forgot a note"}


def describe(step: dict) -> str:
    """One line per tool call, e.g. 🔎 searched documents for "refund policy" → 5 chunks."""
    label = TOOL_LABELS.get(step["tool"], step["tool"])
    arg = step["input"].get("query") or step["input"].get("content") or f"#{step['input'].get('note_id')}"
    if "error" in step:
        result = f"error: {step['error']}"
    elif "chunks" in step:
        result = f"{len(step['chunks'])} chunks"
    elif "notes" in step:
        result = f"{len(step['notes'])} notes"
    elif "saved" in step:
        result = f"note #{step['saved']['id']}"
    elif "forgot" in step:
        result = "deleted"
    else:
        result = f"already known (#{step['duplicate_of']['id']})"
    return f'{label}: "{arg}" → {result}'


def render_chunks(chunks: list[dict]) -> None:
    for c in chunks:
        st.markdown(f"**{c['doc']}** · {c['location']} · score `{c['score']:.2f}`")
        st.caption(c["text"])


def render_trace(trace: list[dict]) -> None:
    if not trace:
        return
    with st.expander(f"Memory used ({len(trace)} tool call{'s' if len(trace) != 1 else ''})"):
        for step in trace:
            st.markdown(describe(step))
            if step.get("notes"):
                for n in step["notes"]:
                    st.caption(f"#{n['id']} [{n['kind']}] {n['content']} (score {n['score']:.2f})")


# --- Sidebar: threads and documents ------------------------------------------------

with st.sidebar:
    st.header("🗄️ Vault Memory Agent")
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY is not set. Uploads and the vault work; chat needs the key "
                   "(or an `ant auth login` profile).")

    st.subheader("Threads")
    threads = store.list_threads()
    if st.button("➕ New thread", width="stretch") or not threads:
        new = store.create_thread("New thread")
        st.session_state.thread_id = new.id
        threads = store.list_threads()
    ids = [t.id for t in threads]
    if st.session_state.get("thread_id") not in ids:
        st.session_state.thread_id = ids[0]
    titles = {t.id: t.title for t in threads}
    thread_id = st.selectbox("Conversation", ids, key="thread_id", format_func=lambda i: titles[i])
    if st.button("🗑️ Delete this thread", width="stretch"):
        store.delete_thread(thread_id)
        del st.session_state["thread_id"]
        st.rerun()
    st.caption("Notes the agent saved survive thread deletion; they live in the vault.")

    st.subheader("Knowledge base")
    # A changing key resets the uploader after each ingest, so files aren't re-added.
    uploads = st.file_uploader("Add documents", type=["pdf", "txt", "md"], accept_multiple_files=True,
                               key=f"uploader_{st.session_state.get('upload_round', 0)}")
    if uploads and st.button("Add to knowledge base", type="primary", width="stretch"):
        with st.spinner("Chunking and embedding..."):
            for f in uploads:
                try:
                    doc, created = vault.ingest(f.name, f.getvalue())
                    st.toast(f"{doc.name}: {doc.n_chunks} chunks" + ("" if created else " (already stored)"))
                except ValueError as exc:
                    st.error(str(exc))
        st.session_state.upload_round = st.session_state.get("upload_round", 0) + 1
        st.rerun()
    for doc in store.list_documents():
        left, right = st.columns([5, 1])
        left.markdown(f"📄 {doc.name}  \n<small>{doc.n_chunks} chunks</small>", unsafe_allow_html=True)
        if right.button("✕", key=f"del_doc_{doc.id}", help="Remove from the knowledge base"):
            store.delete_document(doc.id)
            st.rerun()


# --- Main area -------------------------------------------------------------------------

chat_tab, vault_tab = st.tabs(["💬 Chat", "🗄️ Vault"])

with chat_tab:
    messages = store.get_messages(thread_id)
    for m in messages:
        with st.chat_message(m.role):
            st.markdown(m.content)
            render_trace(m.trace)

    # st.chat_input pins itself to the bottom of the page.
    if prompt := st.chat_input("Ask about your documents, or tell me something to remember"):
        if not messages:   # name the thread after its first question
            store.rename_thread(thread_id, prompt[:60] + ("…" if len(prompt) > 60 else ""))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            status = st.status("Thinking...", expanded=False)
            try:
                result = run_turn(vault, thread_id, prompt, get_client(),
                                  on_tool=lambda step: status.write(describe(step)))
                status.update(label=f"Done in {result.rounds} model call(s)", state="complete")
                st.rerun()   # redraw from the database, the single source of truth
            except Exception as exc:  # noqa: BLE001 - show any API/config error in the UI
                status.update(label="Failed", state="error")
                st.error(f"{type(exc).__name__}: {exc}")

with vault_tab:
    stats = store.stats()
    cols = st.columns(5)
    for col, (name, value) in zip(cols, stats.items()):
        col.metric(name.capitalize(), value)

    st.subheader("📝 Notes the agent saved")
    notes = store.list_notes()
    if not notes:
        st.caption("None yet. Tell the agent something about yourself and it may remember it.")
    for note in notes:
        left, right = st.columns([8, 1])
        left.markdown(f"**#{note.id}** `{note.kind}` {note.content}  \n"
                      f"<small>{note.created_at[:16].replace('T', ' ')}</small>", unsafe_allow_html=True)
        if right.button("Forget", key=f"del_note_{note.id}"):
            store.delete_note(note.id)
            st.rerun()

    st.subheader("🔎 Retrieved chunks per answer (this thread)")
    answers = [m for m in store.get_messages(thread_id) if m.role == "assistant"]
    with_chunks = [m for m in answers if any(s.get("chunks") for s in m.trace)]
    if not with_chunks:
        st.caption("No answer in this thread has searched the knowledge base yet.")
    for m in reversed(with_chunks):   # newest first
        with st.expander(m.content[:90] + ("…" if len(m.content) > 90 else "")):
            for step in m.trace:
                if step.get("chunks"):
                    st.markdown(f'Query: _"{step["input"]["query"]}"_')
                    render_chunks(step["chunks"])

    st.subheader("Try a search yourself")
    q = st.text_input("Query the knowledge base and notes directly (no LLM call)")
    if q:
        left, right = st.columns(2)
        with left:
            st.markdown("**Chunks**")
            render_chunks([{"doc": h.doc_name, "location": h.location, "score": h.score, "text": h.text}
                           for h in vault.search(q)])
        with right:
            st.markdown("**Notes**")
            for n in vault.recall(q):
                st.caption(f"#{n.id} [{n.kind}] {n.content} (score {n.score:.2f})")
