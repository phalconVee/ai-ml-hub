"""Streamlit UI. Run with:  uv run streamlit run app.py

Streamlit re-runs this whole script on every click. The last result is kept
in `st.session_state` so it survives those re-runs.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from scout import llm  # module import (not `from … import make_client`) so tests can swap it
from scout.pipeline import run
from scout.sources import pick_source

load_dotenv()  # picks up ANTHROPIC_API_KEY / SERPAPI_API_KEY from a local .env

st.set_page_config(page_title="Skyfare Scout", page_icon="✈️", layout="wide")
st.title("✈️ Skyfare Scout")
st.caption("Describe a trip in plain words. Four agents parse it, search, compare "
           "and recommend. Information only: Skyfare Scout never books or pays.")

# ── Sidebar: where flights come from ─────────────────────────────────────────
SOURCE_CHOICES = {
    "auto": "Auto (SerpApi if key set, else web search)",
    "web": "Claude web search (no extra key)",
    "serpapi": "SerpApi / Google Flights (needs key)",
    "demo": "Demo data (made-up flights)",
}
with st.sidebar:
    st.header("Flight data")
    choice = st.radio("Source", list(SOURCE_CHOICES), format_func=SOURCE_CHOICES.get)
    has_serp = bool(os.environ.get("SERPAPI_API_KEY", "").strip())
    st.write(("✅" if has_serp else "⬜") + " SERPAPI_API_KEY")
    st.write(("✅" if os.environ.get("ANTHROPIC_API_KEY") else "⬜")
             + " ANTHROPIC_API_KEY (or an `ant auth login` profile)")
    st.info("Web-search prices are indicative and may be stale. Always confirm "
            "on the airline's own site before booking.")

EXAMPLE = ("Lagos to London around 18 December, back after New Year. 2 adults, "
           "economy, as cheap as possible but at most one stop, no overnight layovers.")
text = st.text_area("Your trip", value=EXAMPLE, height=100)

if st.button("Scout flights", type="primary") and text.strip():
    client = llm.make_client()
    try:
        source = pick_source(choice, client)
    except ValueError as err:  # e.g. SerpApi chosen without a key
        st.error(str(err))
        st.stop()
    with st.status("Agents at work…", expanded=True) as status:
        try:
            st.session_state.result = run(text, client=client, source=source,
                                          on_step=status.write)
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
    st.warning(f"🤔 {result.question}")
    st.stop()

trip = result.trip
cols = st.columns(5)
cols[0].metric("Route", f"{trip.origin} → {trip.destination}")
cols[1].metric("Dates", trip.depart_date + (f" ↩ {trip.return_date}" if trip.return_date else ""))
cols[2].metric("Travellers", f"{trip.adults} × {trip.cabin.replace('_', ' ')}")
cols[3].metric("Priority", trip.priority)
cols[4].metric("Budget / adult", f"{trip.budget:,.0f} {trip.currency}" if trip.budget else "—")
if trip.notes:
    st.caption(f"Notes: {trip.notes}")
st.caption(f"Data source: {result.source_label}")

analysis = result.analysis
by_id = {o.id: o for la in analysis.legs for o in la.options}


def fmt_duration(minutes):
    return "?" if minutes is None else f"{minutes // 60}h {minutes % 60:02d}m"


def fmt_price(o):
    if o.price is None:
        return "?"
    return f"{'~' if o.estimated else ''}{o.price:,.0f} {o.currency}"


# Recommendation first: it's what the traveller came for.
advice = result.advice
if advice:
    st.subheader("Recommendation")
    st.markdown(f"**{advice.headline}**")
    for pick in advice.picks:
        o = by_id[pick["option_id"]]  # safe: verify() guarantees the id exists
        with st.container(border=True):
            st.markdown(f"**{pick['leg'].title()} · {pick['role'].replace('_', ' ')} · "
                        f"`{o.id}` {o.airline}** — {fmt_price(o)}, "
                        f"{fmt_duration(o.duration_minutes)}, {o.stops} stop(s)")
            st.write(pick["reason"])
            if o.source_url:
                st.markdown(f"[Check this fare]({o.source_url})")
    st.markdown(advice.tradeoffs)
    for c in advice.caveats:
        st.markdown(f"- ⚠️ {c}")
    if analysis.cheapest_total is not None:
        msg = f"Cheapest possible trip: **{analysis.cheapest_total:,.0f} {trip.currency}** per adult"
        if analysis.within_budget is not None:
            msg += " ✅ within budget" if analysis.within_budget else " ❌ over budget"
        st.markdown(msg)
elif not any(la.options for la in analysis.legs):
    st.error("No flights found for this search. Try other dates, more stops, "
             "or another data source.")

# Full comparison tables, one per leg.
for la in analysis.legs:
    st.subheader(f"{la.leg.title()} options")
    for note in la.notes:
        st.caption(note)
    if not la.options:
        st.write("No options.")
        continue
    st.dataframe(
        [{"id": o.id, "airline": o.airline, "depart": o.depart_time,
          "arrive": o.arrive_time, "duration": fmt_duration(o.duration_minutes),
          "stops": o.stops, "layovers": ", ".join(o.layovers), "price": fmt_price(o),
          "score": o.score, "badges": ", ".join(o.badges),
          "over budget": "yes" if o.over_budget else "", "link": o.source_url}
         for o in la.options],
        hide_index=True,
        column_config={"link": st.column_config.LinkColumn("link", display_text="open")},
    )
    sources = result.searches[la.leg].sources
    if sources:
        with st.expander(f"Pages consulted ({len(sources)})"):
            for s in sources:
                st.markdown(f"- [{s['title']}]({s['url']})")

with st.expander("Agent trace"):
    for line in result.trace:
        st.text(line)
