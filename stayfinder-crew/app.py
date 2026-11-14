"""Streamlit UI. Run with:  uv run streamlit run app.py

Streamlit re-runs this whole script on every click. The last result is kept
in `st.session_state` so it survives those re-runs.
"""

import os

import pandas as pd  # ships with streamlit
import streamlit as st
from dotenv import load_dotenv

from stay import llm  # module import (not `from … import make_brain`) so tests can swap it
from stay.geo import Geocoder, NoGeocoder
from stay.pipeline import run
from stay.sources import pick_source

load_dotenv()  # ANTHROPIC_API_KEY / OLLAMA_MODEL / OLLAMA_HOST from a local .env

st.set_page_config(page_title="Stayfinder Crew", page_icon="🏨", layout="wide")
st.title("🏨 Stayfinder Crew")
st.caption("Describe your stay in plain words. A crew of agents parses it, finds hotels, "
           "compares them and recommends. Information only: nothing is booked or paid.")

# ── Sidebar: which brain, which hotel data ───────────────────────────────────
SOURCES = {
    "auto": "Auto (web search with Claude, OpenStreetMap with Ollama)",
    "web": "Claude web search (prices + reviews, indicative)",
    "osm": "OpenStreetMap (real hotels, no prices, no key)",
    "demo": "Demo data (made-up hotels, offline)",
}
with st.sidebar:
    st.header("Brain")
    kind = st.radio("LLM", ["claude", "ollama"],
                    format_func={"claude": "Claude (cloud)", "ollama": "Ollama (100% local)"}.get)
    if kind == "claude":
        st.write(("✅" if os.environ.get("ANTHROPIC_API_KEY") else "⬜")
                 + " ANTHROPIC_API_KEY (or an `ant auth login` profile)")
        brain_kwargs = {}
    else:
        probe = llm.OllamaBrain()
        pulled = probe.installed_models()
        if pulled:
            default = probe.model if probe.model in pulled else pulled[0]
            model = st.selectbox("Local model", pulled, index=pulled.index(default))
        else:
            model = st.text_input("Local model", probe.model)
            st.warning(f"No models found at {probe.host}. Start Ollama and run "
                       f"`ollama pull {probe.model}`.")
        think = st.toggle("Let the model think first", value=True,
                          help="Reasoning models (qwen3, deepseek-r1…) reason before answering. "
                               "Slower, usually more accurate.")
        brain_kwargs = {"model": model, "think": think}

    st.header("Hotel data")
    choice = st.radio("Source", list(SOURCES), format_func=SOURCES.get)
    st.info("Prices from web search are what a page said, not live quotes. Always "
            "confirm price, fees and cancellation on the hotel's own site.")

EXAMPLE = ("Lisbon, 3 nights from 12 December, two of us. Under €160 a night, near "
           "Alfama or Baixa. Must have wifi and air conditioning, breakfast would be "
           "nice. Rated 8+ please, and a quiet room.")
text = st.text_area("Your stay", value=EXAMPLE, height=110)

if st.button("Find hotels", type="primary") and text.strip():
    try:
        brain = llm.make_brain(kind, **brain_kwargs)
        source = pick_source(choice, brain)
    except Exception as err:  # bad choice combo, missing anthropic credentials…
        st.error(f"{type(err).__name__}: {err}")
        st.stop()
    geocoder = NoGeocoder() if source.name == "demo" else Geocoder()
    with st.status("The crew is working…", expanded=True) as status:
        try:
            st.session_state.result = run(text, brain=brain, source=source,
                                          geocoder=geocoder, on_step=status.write)
            status.update(label="Done", state="complete", expanded=False)
        except Exception as err:  # show API/network errors instead of a stack trace
            status.update(label="Failed", state="error")
            st.error(f"{type(err).__name__}: {err}")
            st.stop()

# ── Results ──────────────────────────────────────────────────────────────────
result = st.session_state.get("result")
if result is None:
    st.stop()
if result.question:
    st.warning(f"**One question first:** {result.question}")
    st.stop()

req, analysis, advice = result.request, result.analysis, result.advice
st.subheader(f"{req.target} · {req.check_in} → {req.check_out} · {req.nights} night(s)")
st.caption(f"Brain: {result.brain_label} · Data: {result.source_label} · Priority: "
           f"{req.priority} · Budget: {req.budget_per_night or 'any'} {req.currency}/night "
           f"· Must-haves: {', '.join(req.must_haves) or 'none'}")

if not analysis or not analysis.ranked:
    st.error("No hotels survived the search and filters. Try a wider area or fewer must-haves.")
else:
    by_id = {o.id: o for o in analysis.ranked}
    if advice:
        st.markdown(f"### {advice.headline}")
        cols = st.columns(min(len(advice.picks), 4) or 1)
        for col, pick in zip(cols, advice.picks):
            o = by_id[pick["hotel_id"]]
            with col.container(border=True):
                st.markdown(f"**{pick['role'].replace('_', ' ').title()}**  \n{o.id} · {o.name}")
                price = (f"{o.price_per_night:,.0f} {o.currency}/night"
                         + (" (est.)" if o.estimated else "")) if o.price_per_night else "price unknown"
                st.caption(" · ".join(x for x in [
                    price,
                    f"{o.distance_km} km" if o.distance_km is not None else "",
                    f"{o.rating}/10" if o.rating is not None else "",
                    f"{o.stars}★" if o.stars else ""] if x))
                st.write(pick["reason"])
                if o.source_url:
                    st.link_button("Open page", o.source_url)
        if advice.tradeoffs:
            st.markdown(f"**Trade-offs.** {advice.tradeoffs}")
        for c in advice.caveats:
            st.markdown(f"- ⚠️ {c}")

    st.markdown("### All hotels, ranked")
    rows = [{
        "id": o.id, "hotel": o.name, "fit": o.score,
        "per night": o.price_per_night, "cur": o.currency,
        "km": o.distance_km, "rating/10": o.rating, "stars": o.stars,
        "must-haves": ", ".join(f"{k}:{v}" for k, v in o.must_have_status.items()),
        "badges": ", ".join(o.badges), "over budget": o.over_budget, "link": o.source_url,
    } for o in analysis.ranked]
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={"link": st.column_config.LinkColumn("link")})

    pins = [{"lat": o.lat, "lon": o.lon} for o in analysis.ranked if o.lat is not None]
    if pins:
        st.map(pd.DataFrame(pins), size=40)

    if analysis.excluded:
        with st.expander(f"Ruled out ({len(analysis.excluded)})"):
            for o, why in analysis.excluded:
                st.write(f"{o.name}: {why}")

if result.search and result.search.sources:
    with st.expander("Sources"):
        for s in result.search.sources:
            st.markdown(f"- [{s['title']}]({s['url']})")
with st.expander("Agent trace"):
    st.code("\n".join(result.trace), language=None)
