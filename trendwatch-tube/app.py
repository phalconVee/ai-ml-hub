"""Streamlit front end. Run with:  uv run streamlit run app.py

The UI is thin on purpose: it collects inputs, calls `pipeline.run`, then
streams the editor's brief. All the real logic lives in `trendwatch/`.

Streamlit reruns this whole script on every click, so results are kept in
`st.session_state`; otherwise switching tabs would pay for a fresh analysis.
"""

import os

import streamlit as st

from trendwatch.config import get_settings
from trendwatch.llm import AgentError, ClaudeLLM
from trendwatch.pipeline import run, write_brief

st.set_page_config(page_title="TrendWatch Tube", page_icon="📈", layout="wide")
st.title("📈 TrendWatch Tube")
st.caption("A small crew of Claude agents watches your YouTube channels and explains what's trending.")

settings = get_settings()

# ---------------------------------------------------------------- sidebar: inputs
with st.sidebar:
    st.header("Channels")
    raw = st.text_area(
        "One per line: URL, @handle or UC… id",
        value="@fireship\n@ThePrimeTimeagen\n@t3dotgg",
        height=140,
    )
    settings.days_back = st.slider("Look back (days)", 1, 60, settings.days_back)
    settings.videos_per_channel = st.slider("Videos per channel", 1, 15, settings.videos_per_channel)
    audience = st.text_input("Brief written for", "a tech content creator deciding what to make next")

    st.divider()
    has_claude = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    st.write("Claude key:", "✅" if has_claude else "⚠️ not in env (an `ant auth login` profile also works)")
    st.write("YouTube key:", "✅ Data API" if settings.youtube_api_key else "— using RSS feeds (no key)")
    st.caption(f"Lead: `{settings.lead_model}` · Worker: `{settings.worker_model}`")
    go = st.button("Find trends", type="primary", width="stretch")

# ---------------------------------------------------------------- run the crew
if go:
    refs = [line.strip() for line in raw.splitlines() if line.strip()]
    llm = ClaudeLLM()
    try:
        with st.status("Working…", expanded=True) as status:
            result = run(refs, settings=settings, llm=llm, on_progress=status.write)
            status.update(label="Writing the brief…")
        st.session_state.result = result
        st.session_state.brief = None          # new data, so the old brief is stale
        st.session_state.audience = audience
        st.session_state.llm = llm
    except AgentError as e:
        st.error(str(e))

result = st.session_state.get("result")
if result is None:
    st.info("Add some channels on the left and press **Find trends**.")
    st.stop()

for w in result.warnings:
    st.warning(w, icon="⚠️")

brief_tab, trends_tab, signals_tab, videos_tab = st.tabs(["Brief", "Trends", "Signals", "Videos"])

# ---------------------------------------------------------------- tab 1: the brief
with brief_tab:
    if st.session_state.get("brief") is None:
        # First render after a run: stream it live, then keep the text.
        try:
            chunks = write_brief(result, st.session_state.audience, settings=settings, llm=st.session_state.llm)
            st.session_state.brief = st.write_stream(chunks)
        except AgentError as e:
            st.error(str(e))
    else:
        st.markdown(st.session_state.brief)
    if st.session_state.get("brief"):
        st.download_button("Download brief (.md)", st.session_state.brief, file_name="trend-brief.md")

# ---------------------------------------------------------------- tab 2: structured trends
titles = {v.video_id: v for v in result.videos}
with trends_tab:
    st.subheader(result.report.headline)
    for t in result.report.trends:
        with st.expander(f"**{t.name}** · {t.momentum} · confidence {t.confidence:.0%}"):
            st.write(t.explanation)
            st.markdown(f"**Why now:** {t.why_now}")
            st.markdown("**Channels:** " + ", ".join(t.channels))
            for vid in t.evidence_video_ids:
                st.markdown(f"- [{titles[vid].title}]({titles[vid].url})")
    if result.report.outliers:
        st.markdown("**Outliers:** " + " · ".join(result.report.outliers))
    if result.report.caveats:
        st.caption("Caveats: " + " ".join(result.report.caveats))

# ---------------------------------------------------------------- tab 3: the numbers
with signals_tab:
    st.caption("Computed in code from the analysts' topic tags. buzz = channels × log(1 + views/day) + videos.")
    rows = [s.as_row() for s in result.signals]
    if rows:
        st.bar_chart({r["topic"]: r["buzz"] for r in rows[:12]}, horizontal=True)
        st.dataframe(rows, width="stretch", hide_index=True)

# ---------------------------------------------------------------- tab 4: raw material
with videos_tab:
    digests = {d.video_id: d for d in result.digests}
    for v in result.videos:
        d = digests.get(v.video_id)
        with st.expander(f"{v.channel_title} — {v.title}"):
            st.markdown(f"[Watch]({v.url}) · {v.published_at:%Y-%m-%d} · views: {v.views if v.views is not None else 'n/a'}")
            if v.transcript is None:
                st.caption("No transcript: analysed from title and description.")
            elif v.transcript_truncated:
                st.caption(f"Transcript clipped to {settings.max_transcript_chars:,} characters.")
            if d:
                st.write(d.summary)
                st.markdown("**Topics:** " + ", ".join(d.topics))
                st.markdown("**Claims:**\n" + "\n".join(f"- {c}" for c in d.key_claims))
