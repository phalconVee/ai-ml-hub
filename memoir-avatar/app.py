"""Memoir Avatar: the Streamlit app.  Run:  uv run streamlit run app.py

One message makes two script runs, so the avatar can start talking before
memory is updated:

  run 1  message in → recall + Claude reply → voice it → save as "pending"
         → st.rerun()
  run 2  the avatar renders the new reply (it starts speaking in the browser)
         → at the very end of the script, remember(): extract facts, merge
         into the graph → st.rerun() once more so the graph view is fresh.
         (The avatar's played-ids set stops it from speaking twice.)

Layout:  sidebar = voice + memory controls · tab 1 = avatar | chat ·
         tab 2 = interactive graph + facts table.
"""

from __future__ import annotations

import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from memoir.avatar import avatar_payload, mic_input, show_avatar
from memoir.companion import PERSONA, Companion, TurnResult
from memoir.config import DEFAULT_VOICES, SETTINGS
from memoir.graph import KnowledgeGraph
from memoir.graph_view import graph_view, layout
from memoir.llm import RefusedError, make_client
from memoir.voice import VoiceError, available_engines, speak

load_dotenv()
st.set_page_config(page_title="Memoir Avatar", page_icon="🗣️", layout="wide")


# ── Long-lived objects: one graph (one SQLite file) and one client per server ──

@st.cache_resource
def get_graph() -> KnowledgeGraph:
    # MEMOIR_DB lets you keep separate memories (and lets the tests use a temp file).
    return KnowledgeGraph(os.getenv("MEMOIR_DB", SETTINGS.db_path))


@st.cache_resource
def get_client():
    return make_client()


graph = get_graph()
ss = st.session_state
ss.setdefault("history", [])        # [{"role", "content"}] plain text, sent to Claude
ss.setdefault("log", [])            # what the chat shows: + facts used / memory changes
ss.setdefault("avatar", None)       # the latest reply's avatar payload
ss.setdefault("pending", None)      # (user_text, TurnResult) waiting to be remembered
ss.setdefault("selected", None)     # node clicked in the graph view


# ── Sidebar: voice and memory controls ──────────────────────────────────────

with st.sidebar:
    st.header("🗣️ Voice")
    engines = available_engines()
    labels = {"kokoro": "Kokoro (local neural, free)", "say": "macOS say (local, free)",
              "browser": "Browser speech (free)", "elevenlabs": "ElevenLabs (paid key)"}
    engine = st.selectbox("Engine", engines, format_func=labels.get,
                          help="Free local engines come first. ElevenLabs only appears when "
                               "ELEVENLABS_API_KEY is set.")
    voice = st.text_input("Voice", DEFAULT_VOICES[engine], key=f"voice_{engine}",
                          help="kokoro: af_heart, am_michael… · say: `say -v '?'` · "
                               "browser: part of a voice name, e.g. 'Samantha'")
    muted = st.toggle("Mute (text only)", value=False)

    st.header("🧠 Memory")
    stats = graph.stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Things", stats["entities"])
    c2.metric("Facts", stats["facts"])
    c3.metric("Turns", stats["turns"])
    forgettable = [n for n in graph.names() if n != SETTINGS.user_node]
    if forgettable:
        target = st.selectbox("Forget something", ["—", *forgettable])
        if target != "—" and st.button(f"Forget “{target}”", width="stretch"):
            graph.forget(target)
            st.rerun()
    if st.button("Wipe all memory", type="secondary", width="stretch"):
        graph.reset()
        ss.history, ss.log, ss.avatar, ss.selected = [], [], None, None
        st.rerun()
    st.caption(f"Stored locally in `{graph.path}`. Nothing leaves this machine "
               "except the messages sent to Claude.")


# ── The two tabs ────────────────────────────────────────────────────────────

st.title(f"Talk to {PERSONA}")
st.caption("A companion that remembers you. Every message grows a knowledge graph; "
           "answers draw on it; replies are spoken by the avatar.")
tab_chat, tab_graph = st.tabs(["💬 Conversation", "🕸️ Memory graph"])

with tab_chat:
    left, right = st.columns([2, 3], gap="large")
    with left:
        show_avatar(ss.avatar)
        spoken = mic_input()
    with right:
        chat_box = st.container(height=520)
        with chat_box:
            for item in ss.log:
                with st.chat_message(item["role"]):
                    st.write(item["content"])
                    if item.get("facts"):
                        with st.expander(f"🧠 used {len(item['facts'])} memories"):
                            st.markdown("\n".join(f"- {f}" for f in item["facts"]))
                    if item.get("delta"):
                        st.caption("📝 " + item["delta"])
        typed = st.chat_input(f"Message {PERSONA}…")

with tab_graph:
    last_recall = set(ss.log[-1].get("nodes", [])) if ss.log else set()
    clicked = graph_view(layout(graph, highlight=last_recall))
    if clicked:
        ss.selected = clicked
    st.caption("Drag to pan · scroll to zoom · hover for details · click a node for its facts. "
               "Gold ring = used to answer the last message. Dashed = no longer true.")
    facts = graph.facts()
    if ss.selected:
        node = graph.find(ss.selected)
        st.subheader(ss.selected)
        if node is not None and node["description"]:
            st.caption(f"{node['type']} · {node['description']}")
        if st.button("Show all facts"):
            ss.selected = None
            st.rerun()
        facts = [f for f in facts if ss.selected in (f.src, f.dst)]
    if facts:
        st.dataframe([{"subject": f.src, "predicate": f.predicate, "object": f.dst,
                       "said": f.weight, "turns": f"{f.first_turn}–{f.last_turn}",
                       "status": "current" if f.active else f"ended turn {f.ended_turn}"}
                      for f in facts], hide_index=True, width="stretch")


# ── Handling a new message (run 1) ──────────────────────────────────────────

def describe(result: TurnResult) -> str:
    """One line for the chat: what memory did with this turn."""
    if result.memory_error:
        return f"memory not updated ({result.memory_error})"
    d = result.delta
    parts = [f"+{len(d.new_facts)} new" if d.new_facts else "",
             f"{len(d.reinforced)} confirmed" if d.reinforced else "",
             f"{len(d.retired)} retired" if d.retired else ""]
    parts = [p for p in parts if p]
    return ("memory: " + ", ".join(parts) + " · " + "; ".join(d.new_facts + d.retired)[:220]
            if parts else "")


user_text = (typed or spoken or "").strip()
if user_text:
    companion = Companion(get_client(), graph)
    try:
        with st.spinner(f"{PERSONA} is thinking…"):
            result = companion.reply(user_text, ss.history)
    except RefusedError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:  # missing key, network... say it plainly, keep the app alive
        hint = (" Check ANTHROPIC_API_KEY in .env." if "auth" in type(e).__name__.lower()
                or "api_key" in str(e).lower() else "")
        st.error(f"Couldn't reach Claude: {type(e).__name__}.{hint}")
        st.stop()

    speech = None
    if not muted:
        try:
            with st.spinner("Finding my voice…"):
                speech = speak(result.reply, engine, voice)
        except (VoiceError, Exception) as e:   # no voice is not fatal: fall back to the browser
            st.toast(f"Voice engine failed ({e}); using browser speech.")
            speech = speak(result.reply, "browser")

    ss.history += [{"role": "user", "content": user_text},
                   {"role": "assistant", "content": result.reply}]
    ss.log += [{"role": "user", "content": user_text},
               {"role": "assistant", "content": result.reply,
                "facts": [f.sentence() for f in result.recall.facts],
                "nodes": [n for n, a in result.recall.activation.items() if a >= 0.5]}]
    ss.avatar = avatar_payload(uuid.uuid4().hex, result.reply, speech, voice=voice, muted=muted)
    ss.pending = (user_text, result)
    st.rerun()


# ── Remembering (run 2, after the avatar has started speaking) ──────────────

if ss.pending:
    user_text, result = ss.pending
    ss.pending = None
    with st.sidebar, st.spinner("Updating memory…"):
        result = Companion(get_client(), graph).remember(user_text, result)
    ss.log[-1]["delta"] = describe(result)
    st.rerun()
