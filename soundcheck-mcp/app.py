"""Streamlit client: upload audio, then explore it with or without Claude.

    uv run streamlit run app.py

Everything audio-related goes through the MCP server (soundcheck/server.py),
even in this app that lives right next to it. Three ways it's used:
  * Upload:   the app calls transcribe_audio itself (no LLM involved).
  * Insights: the app calls sentiment_timeline / extract_topics itself.
  * Chat:     Claude decides which tools to call (soundcheck/agent.py).
Same tools, same server, three different callers: that's the MCP idea.

Streamlit re-runs this whole script on every click. Anything that must
survive a rerun (chat history, the loaded recording) lives in st.session_state.
Each action opens a short MCP session (`asyncio.run(...)`) and closes it; the
server starts in well under a second because Whisper is only loaded when a
transcription actually runs, and transcripts are cached on disk.
"""

import asyncio
import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from soundcheck.agent import run_turn
from soundcheck.config import AUDIO_EXTENSIONS, settings
from soundcheck.llm import LLM
from soundcheck.toolbox import open_toolbox
from soundcheck.transcript import audio_id_for

st.set_page_config(page_title="Soundcheck", page_icon="🎧", layout="wide")


# --- Helpers: one MCP round-trip each -------------------------------------------


async def _call(name: str, args: dict) -> tuple[str, bool]:
    async with open_toolbox() as tb:
        return await tb.call(name, args)


def call_tool(name: str, args: dict) -> dict | list | str:
    """Call one server tool synchronously, raising on tool errors. JSON is decoded."""
    text, is_error = asyncio.run(_call(name, args))
    if is_error:
        raise RuntimeError(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text  # get_transcript returns plain lines, not JSON


def save_upload(uploaded) -> Path:
    """Write the upload to data/uploads/<hash>-<name>.

    The server needs a real file path (MCP passes the path, not the bytes).
    Prefixing the content hash means re-uploading the same file reuses it.
    """
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", uploaded.name)[-80:]
    tmp = settings.uploads_dir / f"incoming-{safe_name}"
    tmp.write_bytes(uploaded.getbuffer())
    final = settings.uploads_dir / f"{audio_id_for(tmp)}-{safe_name}"
    tmp.replace(final)
    return final


def has_claude() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")) \
        or settings.analysis_mode == "claude"


state = st.session_state
state.setdefault("recording", None)  # dict returned by transcribe_audio
state.setdefault("messages", [])     # full API history (append-only)
state.setdefault("chat_log", [])     # what the UI shows: (role, text, tool_events)
state.setdefault("upload_id", None)  # which upload widget file we already processed


def load_recording(info: dict) -> None:
    """Switch to a recording; a new recording means a fresh conversation."""
    state.recording = info
    state.messages, state.chat_log = [], []
    state.pop("insights", None)


# --- Sidebar ---------------------------------------------------------------------

with st.sidebar:
    st.header("🎧 Soundcheck")
    st.caption("Audio analysis over MCP")
    st.markdown(
        f"- Speech-to-text: **{settings.engine}** (Whisper `{settings.whisper_model}`)\n"
        f"- Summary/topics: **{'Claude' if settings.claude_available() else 'local fallback'}**\n"
        f"- Chat agent: **{settings.agent_model if has_claude() else 'needs ANTHROPIC_API_KEY'}**"
    )
    try:
        previous = call_tool("list_recordings", {})
    except RuntimeError:
        previous = []
    if isinstance(previous, list) and previous:
        st.subheader("Earlier recordings")
        for rec in previous[:10]:
            # Uploads are named "<audio_id>-<name>"; show just the name.
            label = rec["file"].removeprefix(f"{rec['audio_id']}-")
            if st.button(f"{label} · {rec['duration']}", key=rec["audio_id"]):
                try:  # cached, so this is instant; it fails only if the audio file was deleted
                    load_recording(call_tool("transcribe_audio", {"file_path": rec["path"]}))
                except RuntimeError as exc:
                    st.error(str(exc))
                st.rerun()


# --- Upload --------------------------------------------------------------------

st.title("Soundcheck")
uploaded = st.file_uploader(
    "Upload a recording (meeting, interview, podcast, voice memo)",
    type=sorted(e.lstrip(".") for e in AUDIO_EXTENSIONS),
)
# The uploader keeps its file across reruns; only process each upload once.
if uploaded is not None and uploaded.file_id != state.upload_id:
    state.upload_id = uploaded.file_id
    path = save_upload(uploaded)
    with st.spinner(f"Transcribing {uploaded.name} locally with Whisper '{settings.whisper_model}'… "
                    "(first run also downloads the model)"):
        try:
            load_recording(call_tool("transcribe_audio", {"file_path": str(path)}))
        except RuntimeError as exc:
            st.error(str(exc))

rec = state.recording
if not rec:
    st.info("Upload an audio file to begin. Transcription runs on this machine; no key needed.")
    st.stop()

if Path(rec["path"]).is_file():
    st.audio(rec["path"])
st.caption(f"`{rec['audio_id']}` · {rec['duration']} · language: {rec['language']} · "
           f"{rec['words']} words · {rec['engine']}{' · cached' if rec['cached'] else ''}")

chat_tab, transcript_tab, insights_tab = st.tabs(["💬 Chat", "📝 Transcript", "📈 Insights"])


# --- Transcript tab: direct tool calls, no LLM ------------------------------------

with transcript_tab:
    cols = st.columns([3, 1])
    query = cols[0].text_input("Search the transcript", placeholder="a name, a term, a phrase")
    if cols[1].button("Label speakers", help="Local voice-similarity heuristic"):
        with st.spinner("Comparing voices…"):
            st.json(call_tool("identify_speakers", {"audio_id": rec["audio_id"]}), expanded=False)
    if query:
        found = call_tool("search_transcript", {"audio_id": rec["audio_id"], "query": query})
        st.write(f"{found['total_hits']} hit(s)")
        for hit in found["hits"]:
            st.markdown(f"**[{hit['time']}]** _{hit['match']}_")
            st.code(hit["context"], language=None)
    st.text(call_tool("get_transcript", {"audio_id": rec["audio_id"], "max_chars": 200000}))


# --- Insights tab: sentiment curve + topics -----------------------------------------

with insights_tab:
    window = st.slider("Sentiment window (seconds)", 10, 120, 30, step=5)
    senti = call_tool("sentiment_timeline", {"audio_id": rec["audio_id"], "window_seconds": window})
    st.metric("Overall mood", senti["overall"]["mood"], f"{senti['overall']['score']:+.2f}")
    if senti["points"]:
        df = pd.DataFrame(senti["points"]).set_index("label_time")
        st.line_chart(df["score"], y_label="sentiment (-1 … +1)", x_label="time")
    c1, c2 = st.columns(2)
    c1.write("**Most positive moments**")
    for m in senti["most_positive"]:
        c1.markdown(f"- [{m['time']}] {m['text']}")
    c2.write("**Most negative moments**")
    for m in senti["most_negative"]:
        c2.markdown(f"- [{m['time']}] {m['text']}")
    if st.button("Extract topics & entities"):
        with st.spinner("Indexing…"):
            state.insights = call_tool("extract_topics", {"audio_id": rec["audio_id"]})
    if "insights" in state:
        st.json(state.insights)


# --- Chat tab: Claude drives the tools ----------------------------------------------

with chat_tab:
    if not has_claude():
        st.warning("Chat needs ANTHROPIC_API_KEY in .env (the other tabs work without it).")

    for role, text, events in state.chat_log:
        with st.chat_message(role):
            for ev in events:
                st.caption(f"🔧 {ev}")
            st.markdown(text)

    quick = st.columns(4)
    prompts = {
        "Summarise": "Summarise this recording with timestamped key points.",
        "Who spoke?": "Identify the speakers and tell me who said what, roughly.",
        "Mood shifts": "How does the sentiment change over the recording? Point to the moments that cause shifts.",
        "Topics": "What topics are covered, and when?",
    }
    clicked = next((p for (label, p), col in zip(prompts.items(), quick) if col.button(label)), None)
    question = st.chat_input("Ask anything about the recording") or clicked

    if question:
        state.chat_log.append(("user", question, []))
        checkpoint = len(state.messages)  # to roll back cleanly if the turn fails
        state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            events: list[str] = []
            status = st.status("Thinking…", expanded=False)

            def on_event(kind: str, data: dict) -> None:
                # Streamlit calls from inside asyncio.run are fine: same thread.
                if kind == "tool_call":
                    line = f"{data['name']}({', '.join(f'{k}={v!r}' for k, v in data['input'].items() if k != 'audio_id')})"
                    events.append(line)
                    status.update(label=f"Calling {data['name']}…")
                    status.write(f"🔧 {line}")

            async def go() -> str:
                async with open_toolbox() as tb:
                    return await run_turn(state.messages, tb, LLM(), rec, on_event)

            try:
                answer = asyncio.run(go())
                status.update(label=f"Used {len(events)} tool call(s)", state="complete")
            except Exception as exc:
                answer = f"⚠️ {type(exc).__name__}: {exc}"
                status.update(label="Failed", state="error")
                del state.messages[checkpoint:]  # drop the half-finished turn so history stays valid
            st.markdown(answer)
        state.chat_log.append(("assistant", answer, events))
