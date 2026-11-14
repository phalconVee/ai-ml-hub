"""Streamlit UI. Run with:  uv run streamlit run app.py

Streamlit re-runs this whole script top to bottom on every click. The
finished script and audio live in `st.session_state`, so they survive those
re-runs (playing, downloading, re-voicing).
"""

import streamlit as st
from dotenv import load_dotenv

from podcast.config import DEFAULT_VOICES
from podcast.crew import produce
from podcast.llm import RefusedError, make_client
from podcast.source import SourceError
from podcast.studio import record
from podcast.voices import VoiceError, available_engines, pick_engine

load_dotenv()  # picks up ANTHROPIC_API_KEY (and optional ELEVENLABS_API_KEY) from .env

st.set_page_config(page_title="Podcast Crew", page_icon="🎙️", layout="wide")
st.title("🎙️ Podcast Crew")
st.caption("Paste a link or some text. A researcher outlines it, a scriptwriter turns it into "
           "a two-host conversation, an editor tightens it, and local voices record it.")


@st.cache_resource(show_spinner="Loading the voice engine…")
def get_engine(name: str):
    """Load each engine once per server (Kokoro takes a few seconds to load)."""
    return pick_engine(name)


# ── Sidebar: the show's settings ────────────────────────────────────────────
with st.sidebar:
    st.header("The show")
    name_a = st.text_input("Host A (the guide)", "Maya")
    name_b = st.text_input("Host B (the curious one)", "Theo")
    minutes = st.slider("Length (minutes)", 2, 10, 4)
    style = st.text_input("Style note (optional)", placeholder="e.g. playful, for teenagers")

    st.header("Voices")
    engines = available_engines()
    if not engines:
        st.error("No voice engine found. See the README: install Kokoro, or use a Mac.")
    engine_name = st.selectbox("Engine", engines or ["none"],
                               help="Free local engines come first. ElevenLabs appears "
                                    "only when ELEVENLABS_API_KEY is set.")
    voice_a, voice_b = None, None
    if engines:
        defaults = DEFAULT_VOICES[engine_name]
        try:
            options = get_engine(engine_name).voices()
        except VoiceError as e:
            st.error(str(e))
            options = []
        if options:  # pick from the engine's own list…
            def idx(v):
                return options.index(v) if v in options else 0
            voice_a = st.selectbox(f"{name_a}'s voice", options, index=idx(defaults["A"]))
            voice_b = st.selectbox(f"{name_b}'s voice", options, index=idx(defaults["B"]))
        else:        # …or type an id (ElevenLabs voice ids)
            voice_a = st.text_input(f"{name_a}'s voice id", defaults["A"])
            voice_b = st.text_input(f"{name_b}'s voice id", defaults["B"])

# ── Main: the input ─────────────────────────────────────────────────────────
user_input = st.text_area("A link to an article, or the text itself", height=160,
                          placeholder="https://… or paste an article / notes / a transcript")
go = st.button("Produce the episode", type="primary", disabled=not user_input.strip())

names = {"A": name_a or "Host A", "B": name_b or "Host B"}
voices = {"A": voice_a, "B": voice_b}


def voice_it(script) -> None:
    """Record the script with the sidebar's engine + voices; store the episode."""
    if not engines:
        st.warning("No voice engine available, so there's no audio; the script is below.")
        return
    bar = st.progress(0.0, text="Recording…")
    try:
        episode = record(script, get_engine(engine_name), voices,
                         on_line=lambda i, n: bar.progress(i / n, text=f"Recording line {i}/{n}"))
    except VoiceError as e:
        st.error(str(e))
        return
    finally:
        bar.empty()
    st.session_state.episode = episode


if go:
    st.session_state.pop("episode", None)
    st.session_state.pop("production", None)  # never re-voice an old script by mistake
    with st.status("The crew is on it…", expanded=True) as status:
        try:
            st.session_state.production = produce(
                user_input, make_client(), names=names, minutes=minutes,
                style=style.strip(), on_step=st.write)
            status.update(label="Script ready", state="complete", expanded=False)
        except SourceError as e:
            status.update(label="Couldn't use that input", state="error")
            st.error(str(e))
        except RefusedError as e:
            status.update(label="Declined", state="error")
            st.error(str(e))
    if "production" in st.session_state:
        voice_it(st.session_state.production.script)

# ── Results ─────────────────────────────────────────────────────────────────
prod = st.session_state.get("production")
if prod:
    script = prod.script
    st.header(script.title)
    st.caption(f"{len(script.lines)} lines · {script.words()} words · "
               f"~{script.minutes():.1f} min spoken")

    episode = st.session_state.get("episode")
    if episode:
        st.audio(episode.wav, format="audio/wav")
        st.caption(f"{episode.seconds / 60:.1f} min · voiced by {episode.engine}")
    c1, c2, c3 = st.columns(3)
    if episode:
        c1.download_button("Download audio (.wav)", episode.wav,
                           file_name="episode.wav", mime="audio/wav")
    c2.download_button("Download script (.md)", script.markdown(names),
                       file_name="episode.md", mime="text/markdown")
    if c3.button("Re-record with current voices"):
        voice_it(script)
        st.rerun()

    st.subheader("Script")
    for line in script.lines:
        st.markdown(f"**{names[line.speaker]}:** {line.text}")

    with st.expander("Research outline"):
        o = prod.outline
        st.markdown(f"**{o.title}** — {o.hook}\n\n{o.summary}")
        for kp in o.key_points:
            st.markdown(f"- **{kp['point']}** — {kp['detail']}")
        st.markdown("**Listener questions:** " + " · ".join(o.listener_questions))
        st.markdown(f"**Takeaway:** {o.takeaway}")
    with st.expander("Editor's notes"):
        st.markdown("**Automatic check on the draft:**")
        for i in prod.edit.issues_before or ["No issues."]:
            st.write("- " + i)
        st.markdown("**Editor's changes:**")
        for c in prod.edit.changes or ["None."]:
            st.write("- " + c)
        if prod.edit.issues_after:
            st.markdown("**Still flagged after the edit:**")
            for i in prod.edit.issues_after:
                st.write("- " + i)
    with st.expander("First draft (before editing)"):
        st.text(prod.draft.as_text(names))
    with st.expander("Source text"):
        st.caption(prod.source.url or "pasted text")
        st.text(prod.source.text[:5000] + ("…" if len(prod.source.text) > 5000 else ""))
    with st.expander("How we got here"):
        for s in prod.trace:
            st.write(s)
