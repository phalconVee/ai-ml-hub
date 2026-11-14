"""Sourcebook - Streamlit UI.

Layout:
    sidebar  : notebook picker, add sources (upload / URL / paste), source list
               with include-checkboxes, settings
    main     : audio overview panel, then the chat with clickable citations

Run:  uv run streamlit run app.py
"""

from __future__ import annotations

import html
import os

import streamlit as st

from sourcebook import Notebook, ask, generate_script, list_notebooks, load_pdf, load_text, load_url
from sourcebook import llm, tts
from sourcebook.answer import Answer
from sourcebook.podcast import HOSTS, Script
from sourcebook.retrieve import try_dense_embedder

st.set_page_config(page_title="Sourcebook", page_icon="📓", layout="wide")

# Small CSS for citation markers and cards. Everything else is stock Streamlit.
st.markdown("""
<style>
.sb-cite-mark a { text-decoration: none; font-weight: 600; font-size: 0.75em; }
.sb-card { font-size: 0.86em; line-height: 1.45; padding: 0.45em 0.7em; margin: 0.3em 0;
           border-left: 3px solid #f0b429; background: rgba(240,180,41,0.07); border-radius: 4px;
           scroll-margin-top: 4rem; }
.sb-card:target { background: rgba(240,180,41,0.25); }
.sb-card .ctx { opacity: 0.6; }
.sb-card mark { background: rgba(240,180,41,0.45); padding: 0 2px; color: inherit; }
</style>
""", unsafe_allow_html=True)


# --- shared resources ---------------------------------------------------------

@st.cache_resource(show_spinner="Loading local embedding model...")
def dense_embedder():
    return try_dense_embedder()


def client():
    # Tests inject a fake client through session state; the app uses the real SDK.
    return st.session_state.get("_client") or llm.make_client()


def notebook() -> Notebook:
    return st.session_state.nb


def open_notebook(name: str) -> None:
    st.session_state.nb = Notebook.load(name)
    st.session_state.nb_name = name


if "nb" not in st.session_state:
    existing = list_notebooks()
    open_notebook(existing[0] if existing else "My first notebook")


# --- rendering helpers ----------------------------------------------------------

def answer_html(ans: Answer, key: str) -> str:
    """Answer text with each [n] as a superscript link to its citation card."""
    by_num = {c.number: c for c in ans.citations}
    out = []
    for text, nums in ans.segments:
        out.append(html.escape(text, quote=False))   # Markdown still renders; raw HTML can't sneak in
        for n in nums:
            tip = html.escape(by_num[n].location) if n in by_num else ""
            out.append(f'<sup class="sb-cite-mark"><a href="#{key}-c{n}" title="{tip}">[{n}]</a></sup>')
    return "".join(out)


def citation_card(c, key: str) -> str:
    """The cited sentences highlighted inside their passage, plus one sentence of context."""
    before = " ".join(c.sentences[max(0, c.start - 1):c.start])
    cited = " ".join(c.sentences[c.start:c.end])
    after = " ".join(c.sentences[c.end:c.end + 1])
    title = html.escape(c.location)
    if c.url:   # web source: the link scrolls to the passage on the live page
        title = f'<a href="{html.escape(c.url)}" target="_blank">{title} ↗</a>'
    return (f'<div class="sb-card" id="{key}-c{c.number}"><b>[{c.number}]</b> {title}<br>'
            f'<span class="ctx">{html.escape(before)}</span> <mark>{html.escape(cited)}</mark> '
            f'<span class="ctx">{html.escape(after)}</span></div>')


def render_turn(ans: Answer, key: str) -> None:
    with st.chat_message("user"):
        st.markdown(ans.question)
    with st.chat_message("assistant", avatar="📓"):
        st.markdown(answer_html(ans, key), unsafe_allow_html=True)
        if ans.citations:
            st.markdown("".join(citation_card(c, key) for c in ans.citations), unsafe_allow_html=True)
        if ans.search_query and ans.search_query != ans.question:
            st.caption(f"Searched for: “{ans.search_query}” · {ans.passages_used} passages read")


# --- sidebar: notebooks and sources ---------------------------------------------------

with st.sidebar:
    st.header("📓 Sourcebook")

    names = list_notebooks()
    current = st.session_state.nb_name
    options = ([current] if current not in names else []) + names + ["➕ New notebook"]
    picked = st.selectbox("Notebook", options, index=options.index(current))
    if picked == "➕ New notebook":
        new_name = st.text_input("Name", placeholder="e.g. Climate policy reading")
        if st.button("Create", disabled=not new_name.strip()):
            open_notebook(new_name.strip())
            notebook().save()
            st.rerun()
    elif picked != current:
        open_notebook(picked)
        st.rerun()

    nb = notebook()
    st.subheader("Add sources")
    tab_file, tab_url, tab_text = st.tabs(["Files", "Web page", "Paste"])

    def add(make, label: str) -> None:
        try:
            src = make()
        except Exception as e:   # bad PDF, unreachable URL, empty text...
            st.error(f"{label}: {e}")
            return
        if nb.add_source(src):
            nb.save()
            st.toast(f"Added {src.title} ({src.word_count:,} words)")
        else:
            st.info(f"{src.title} is already in this notebook.")

    with tab_file:
        files = st.file_uploader("PDF, text or Markdown", type=["pdf", "txt", "md", "markdown"],
                                 accept_multiple_files=True)
        if st.button("Add files", disabled=not files):
            for f in files:
                data = f.getvalue()
                if f.name.lower().endswith(".pdf"):
                    add(lambda: load_pdf(data, f.name), f.name)
                else:
                    add(lambda: load_text(data.decode("utf-8", errors="replace"), f.name), f.name)
    with tab_url:
        url = st.text_input("URL", placeholder="https://...")
        if st.button("Fetch page", disabled=not url.strip()):
            with st.spinner("Fetching..."):
                add(lambda: load_url(url.strip()), url)
    with tab_text:
        paste_title = st.text_input("Title", value="Pasted notes")
        paste_body = st.text_area("Text", height=120)
        if st.button("Add text", disabled=not paste_body.strip()):
            add(lambda: load_text(paste_body, paste_title or "Pasted notes"), "Pasted text")

    st.subheader(f"Sources ({len(nb.sources)})")
    if not nb.sources:
        st.caption("Nothing yet. Add a PDF, a web page or some notes above.")
    icons = {"pdf": "📄", "text": "📝", "web": "🌐"}
    for sid, src in list(nb.sources.items()):
        col_box, col_del = st.columns([5, 1], vertical_alignment="center")
        enabled = col_box.checkbox(f"{icons.get(src.kind, '•')} {src.title}", value=src.enabled, key=f"en-{sid}",
                                   help=f"{src.origin} · {src.word_count:,} words")
        if enabled != src.enabled:
            src.enabled = enabled
            nb.save()
        if col_del.button(":material/close:", key=f"del-{sid}", help="Remove source", type="tertiary"):
            nb.remove_source(sid)
            nb.save()
            st.rerun()

    with st.expander("Settings"):
        want_dense = st.toggle("Hybrid search (local embeddings)", value=nb.embedder is not None,
                               help="BM25 + a small local embedding model, fused with RRF. "
                                    "Needs `uv sync --extra dense`; downloads ~70 MB once.")
        if want_dense and nb.embedder is None:
            emb = dense_embedder()
            if emb is None:
                st.warning("fastembed isn't installed: run `uv sync --extra dense`. Using BM25.")
            else:
                nb.set_embedder(emb)
        elif not want_dense and nb.embedder is not None:
            nb.set_embedder(None)
        if st.button("Clear chat memory"):
            nb.memory.clear()
            nb.save()
            st.rerun()


# --- main: header, audio overview, chat --------------------------------------------------

st.title(nb.name)
if nb.sources:
    enabled = sum(s.enabled for s in nb.sources.values())
    st.caption(f"{enabled}/{len(nb.sources)} sources in use · {len(nb.passages):,} passages · "
               f"search: {nb.retriever.mode} · memory: {len(nb.memory.turns)} turns"
               + (" (older ones summarised)" if nb.memory.summary else ""))
if not os.getenv("ANTHROPIC_API_KEY") and "_client" not in st.session_state:
    st.warning("Set ANTHROPIC_API_KEY in `.env` to chat and generate overviews (or log in with `ant auth login`).")

with st.expander("🎧 Audio overview", expanded=False):
    backend = tts.pick_backend()
    st.caption(f"Two hosts, {HOSTS['A']} and {HOSTS['B']}, talk through your sources in a few minutes. "
               f"Voice: {backend.name if backend else 'none found (script only)'}.")
    focus = st.text_input("Focus (optional)", placeholder="e.g. the disagreements between sources")
    if st.button("Generate audio overview", type="primary", disabled=not nb.sources):
        with st.status("Writing the script...", expanded=True) as status:
            try:
                script = generate_script(nb, client(), focus)
                script.save(nb.script_path)
                if backend:
                    bar = st.progress(0.0, text="Recording voices...")
                    secs = tts.render(script.turns, backend, nb.audio_path,
                                      on_progress=lambda d, t: bar.progress(d / t, text=f"Recording turn {d}/{t}"))
                    status.update(label=f"Done: {secs / 60:.1f} min", state="complete")
                else:
                    nb.audio_path.unlink(missing_ok=True)
                    status.update(label="Script ready (no voice backend available)", state="complete")
            except Exception as e:
                status.update(label=f"Failed: {e}", state="error")
    script = Script.load(nb.script_path)
    if script:
        st.markdown(f"**{script.title}** · {script.word_count} words")
        if nb.audio_path.exists():
            st.audio(str(nb.audio_path), format="audio/wav")
        with st.popover("Transcript"):
            st.markdown(script.transcript())

for i, turn in enumerate(nb.memory.turns):
    render_turn(turn, f"t{i}")

if not nb.memory.turns:
    st.info("Ask anything about your sources. Every answer cites the exact passages it used; "
            "click a [n] to jump to it, or the ↗ on a web source to open the page at that spot.")

if question := st.chat_input("Ask about your sources", disabled=not nb.sources):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📓"):
        with st.status("Thinking...") as status:
            try:
                ask(nb, question, client(), on_event=lambda m: status.update(label=m))
                nb.save()
                status.update(label="Done", state="complete")
            except Exception as e:
                status.update(label=f"Failed: {e}", state="error")
                st.stop()
    st.rerun()   # redraw the whole conversation from memory, citations included
