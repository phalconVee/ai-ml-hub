"""Streamlit UI. Run with:  uv run streamlit run app.py

Streamlit re-runs this whole script top to bottom on every click. The
finished story is kept in `st.session_state`, so it survives those re-runs
(e.g. when you click the download button).
"""

import re

import streamlit as st
from dotenv import load_dotenv

from newsroom.desk import run_desk
from newsroom.llm import RefusedError, make_client

load_dotenv()  # picks up ANTHROPIC_API_KEY from a local .env file, if present

st.set_page_config(page_title="Newsroom Crew", page_icon="🗞️", layout="wide")
st.title("🗞️ Newsroom Crew")
st.caption("A reporter searches the web, a writer drafts a sourced story, an editor checks it.")

# ── Sidebar: the assignment ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Assignment")
    topic = st.text_input("Topic", placeholder="e.g. the latest on solid-state batteries")
    angle = st.text_input("Angle (optional)", placeholder="e.g. what it means for EV prices")
    words = st.slider("Length (words)", 250, 1200, 500, step=50)
    go = st.button("File the story", type="primary", disabled=not topic.strip())


def link_citations(body: str, story) -> str:
    """Turn [S2] into a clickable [[S2]](url) so readers can check each claim."""
    urls = {s.id: s.url for s in story.sources}
    return re.sub(r"\[(S\d+)\]",
                  lambda m: f"[[{m[1]}]]({urls[m[1]]})" if m[1] in urls else m[0], body)


def render(story) -> None:
    if story.status == "spiked":
        st.warning("Story spiked: the reporter found no usable sources, so nothing was written.")
    else:
        if story.status == "published_with_notes":
            st.warning("Published after the revision limit; the editor still had notes (see below).")
        d = story.draft
        st.header(d.headline)
        st.markdown(f"*{d.dek}*")
        st.markdown(link_citations(d.body, story))
        st.download_button("Download Markdown", story.markdown(),
                           file_name="story.md", mime="text/markdown")

        st.subheader(f"Sources cited ({len(story.used_sources())})")
        for s in story.used_sources():
            badge = " · read in full" if s.read else ""
            with st.expander(f"[{s.id}] {s.title} — {s.outlet}{badge}"):
                st.markdown(s.url)
                for q in s.quotes:
                    st.caption(f"“{q}”")

        with st.expander(f"Editor's desk ({len(story.drafts)} draft(s))"):
            for draft, review in zip(story.drafts, story.reviews):
                st.markdown(f"**Draft {draft.revision + 1}: {review.verdict}**")
                for n in review.notes:
                    st.write("- " + n)

    with st.expander("Reporter's notes"):
        st.text(story.notes.text if story.notes else "")
    with st.expander("How we got here"):
        for step in story.trace:
            st.write(step)
        if story.notes:
            st.write("Searches:", story.notes.queries)
            st.write("Pages read:", story.notes.fetched)


if go:
    # st.status shows a live, collapsible log while the crew works.
    with st.status("The crew is on it…", expanded=True) as status:
        try:
            story = run_desk(topic.strip(), make_client(), words=words, angle=angle.strip(),
                             on_step=st.write)
        except RefusedError as e:
            status.update(label="Declined", state="error")
            st.error(str(e))
            st.stop()
        except Exception as e:  # missing key, network error, rate limit…
            status.update(label="Something went wrong", state="error")
            st.error(f"{type(e).__name__}: {e}")
            st.stop()
        status.update(label="Story filed", state="complete", expanded=False)
    st.session_state.story = story

if "story" in st.session_state:
    render(st.session_state.story)
else:
    st.info("Enter a topic in the sidebar and click **File the story**.")
