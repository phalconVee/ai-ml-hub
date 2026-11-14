"""recall-companion: Streamlit UI.

Left: the chat. Right: the memory inspector, which shows what the assistant
remembers, when it learned it, what changed, and which memories it used for
the latest answer. Every memory can be corrected or forgotten from there.

Run:  uv run streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from recall.companion import answer, remember
from recall.config import SETTINGS
from recall.llm import make_client
from recall.store import MemoryStore

load_dotenv()
st.set_page_config(page_title="recall-companion", page_icon="🧠", layout="wide")

# ── Long-lived objects ───────────────────────────────────────────────────────
# Streamlit re-runs this whole file on every click, so anything that must
# survive (the DB connection, the API client, the latest turn) lives in
# st.session_state. The tests pre-fill "store" and "client" with fakes.
ss = st.session_state
if "store" not in ss:
    ss.store = MemoryStore(SETTINGS.db_path)
store: MemoryStore = ss.store
ss.setdefault("last_turn", None)


def get_client():
    """Create the Claude client on first use, so the UI loads without a key."""
    if "client" not in ss:
        ss.client = make_client()
    return ss.client


# ── Sidebar: who is talking, and in which session ────────────────────────────
with st.sidebar:
    st.title("🧠 recall-companion")
    st.caption("A chat assistant that remembers you across sessions.")

    st.subheader("Profile")
    names = [n for _, n in store.profiles()]
    new_name = st.text_input("Create a profile", placeholder="e.g. Ana")
    if st.button("Add profile", disabled=not new_name.strip()):
        ss.profile_id = store.get_or_create_profile(new_name)
        ss.session_id = store.new_session(ss.profile_id)
        ss.last_turn = None
        st.rerun()
    if not names:
        st.info("Create a profile to start. Each profile has its own memory.")
        st.stop()

    ids = dict((n, i) for i, n in store.profiles())
    current = next((n for n, i in ids.items() if i == ss.get("profile_id")), names[0])
    chosen = st.selectbox("Talking as", names, index=names.index(current))
    if ids[chosen] != ss.get("profile_id"):
        ss.profile_id, ss.last_turn = ids[chosen], None
        ss.pop("session_id", None)
    profile_id: int = ss.profile_id

    st.subheader("Session")
    sessions = store.sessions(profile_id)
    if not sessions:
        ss.session_id = store.new_session(profile_id)
        sessions = store.sessions(profile_id)
    if st.button("➕ New session"):
        ss.session_id, ss.last_turn = store.new_session(profile_id), None
        st.rerun()
    labels = {sid: f"{title} · {created[:16].replace('T', ' ')}" for sid, title, created in sessions}
    sid_list = list(labels)
    ss.setdefault("session_id", sid_list[0])
    if ss.session_id not in labels:
        ss.session_id = sid_list[0]
    picked = st.selectbox("Open session", sid_list, index=sid_list.index(ss.session_id),
                          format_func=labels.get)
    if picked != ss.session_id:
        ss.session_id, ss.last_turn = picked, None
    session_id: int = ss.session_id

    st.subheader("Recall")
    smart = st.toggle("Smart recall (Claude rewrites the search)", value=True,
                      help="Off = search memory with your raw message only (no extra call).")
    st.caption(f"Model: `{SETTINGS.model}` · memory file: `{SETTINGS.db_path}`")

# ── Layout ───────────────────────────────────────────────────────────────────
chat_col, mem_col = st.columns([3, 2], gap="large")

with chat_col:
    st.subheader(f"Chat · {labels[session_id].split(' · ')[0]}")
    for m in store.messages(session_id):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

prompt = st.chat_input("Tell me about yourself, or ask me anything…")
if prompt:
    with chat_col:
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            with st.status("Recalling and answering…", expanded=False) as status:
                turn = answer(store, get_client(), profile_id, session_id, prompt, smart_recall=smart)
                status.update(label="Remembering what's new…")
                turn = remember(store, get_client(), profile_id, session_id, turn, prompt)
                status.update(label="Done", state="complete")
            ss.last_turn = turn
            st.rerun()
        except Exception as exc:  # e.g. no API key, network down
            st.error(f"Claude call failed: {exc}. Is ANTHROPIC_API_KEY set in .env?")


# ── Memory inspector ─────────────────────────────────────────────────────────
def fact_line(f) -> str:
    return f"**[M{f.id}]** {f.statement}"


def edge(f) -> str:
    return f"`{f.subject}` —{f.predicate}→ `{f.object}`"


with mem_col:
    st.subheader("Memory inspector")
    active = store.facts(profile_id)
    everything = store.facts(profile_id, include_superseded=True)
    past = [f for f in everything if not f.active]
    tab_used, tab_all, tab_hist, tab_graph = st.tabs(
        [f"Used now", f"Memories ({len(active)})", f"History ({len(past)})", "Entities"])

    # 1. Which memories shaped the latest answer, and what was learned from it.
    with tab_used:
        msgs = store.messages(session_id)
        last_assistant = next((m for m in reversed(msgs) if m["role"] == "assistant"), None)
        turn = ss.last_turn
        if not last_assistant:
            st.caption("Nothing yet. Send a message and this tab shows which memories were used.")
        else:
            if turn and turn.assistant_message_id == last_assistant["id"]:
                p = turn.plan
                st.caption(f"Search: *{p.query}*" + (f" · entities: {', '.join(p.entities)}" if p.entities else "")
                           + (" · including past facts" if p.include_history else "")
                           + ("" if p.by_llm else " · (plain search)"))
            used = store.recalled_for(last_assistant["id"])
            st.markdown("**Used for the latest answer**")
            if not used:
                st.caption("No stored memory was relevant.")
            for f, score in used:
                if f is None:
                    continue
                tag = "" if f.active else " · ⏳ past"
                st.markdown(f"{fact_line(f)}  \n<small>score {score:.2f}{tag}</small>",
                            unsafe_allow_html=True)
            if turn and turn.assistant_message_id == last_assistant["id"]:
                with st.expander("Score breakdown"):
                    st.dataframe([{"id": r.fact.id, "score": r.score, **r.why} for r in turn.recalled],
                                 hide_index=True)
                st.markdown("**Learned from this exchange**")
                if not turn.writes:
                    st.caption("Nothing new worth remembering.")
                for w in turn.writes:
                    icon = {"added": "🆕", "confirmed": "✔️"}.get(w.action, "✏️")
                    st.markdown(f"{icon} {w.action}: {fact_line(w.fact)}")
                    for old in w.superseded:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;↳ replaced ~~{old.statement}~~ (M{old.id})")

    # 2. Everything currently believed, editable.
    with tab_all:
        if not active:
            st.caption("No memories yet.")
        flt = st.text_input("Filter", key="flt", placeholder="search memories…") if active else ""
        for f in active:
            if flt and flt.lower() not in (f.statement + f.subject + f.object).lower():
                continue
            with st.expander(f"[M{f.id}] {f.statement}"):
                st.markdown(f"{edge(f)}  \n"
                            f"{f.category} · importance {f.importance}"
                            f"{' · one value at a time' if f.exclusive else ''}"
                            f"{f' · event {f.occurred_at}' if f.occurred_at else ''}  \n"
                            f"true since {f.valid_from[:16].replace('T', ' ')} · learned "
                            f"{f.recorded_at[:16].replace('T', ' ')} ({f.origin}"
                            f"{f', session {f.source_session_id}' if f.source_session_id else ''})  \n"
                            f"last confirmed {f.last_confirmed_at[:16].replace('T', ' ')} · "
                            f"recalled {store.times_recalled(f.id)}×")
                with st.form(f"edit-{f.id}"):
                    new_statement = st.text_input("Statement", f.statement)
                    new_obj = st.text_input("Value", f.object)
                    new_imp = st.slider("Importance", 1, 5, f.importance)
                    c1, c2 = st.columns(2)
                    save = c1.form_submit_button("Save edit")
                    forget = c2.form_submit_button("🗑 Forget")
                if save:
                    store.edit_fact(f.id, new_statement, new_obj, new_imp)
                    st.rerun()
                if forget:
                    store.delete_fact(f.id)
                    st.rerun()

    # 3. What used to be true: the temporal part of the memory.
    with tab_hist:
        if not past:
            st.caption("Nothing has changed yet. When a fact is replaced (you moved, changed "
                       "jobs, changed your mind), the old version lands here with its dates.")
        for f in past:
            st.markdown(f"~~{f.statement}~~  \n<small>true {f.valid_from[:10]} → {(f.valid_to or '')[:10]}"
                        f"{f' · replaced by M{f.superseded_by}' if f.superseded_by else ''}</small>",
                        unsafe_allow_html=True)
            if st.button("🗑 Forget", key=f"forget-past-{f.id}"):
                store.delete_fact(f.id)
                st.rerun()

    # 4. The graph view: nodes and current edges.
    with tab_graph:
        ents = store.entities(profile_id)
        st.dataframe([{"entity": e["name"], "kind": e["kind"], "facts": e["active_facts"]} for e in ents],
                     hide_index=True)
        if active:
            st.markdown("**Relations (true now)**")
            st.dataframe([{"subject": f.subject, "relation": f.predicate, "object": f.object,
                           "since": f.valid_from[:10]} for f in active], hide_index=True)
