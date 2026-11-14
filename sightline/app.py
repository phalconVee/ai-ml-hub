"""Streamlit UI. Run with:  uv run streamlit run app.py

Streamlit re-runs this whole script top to bottom on every click. Anything
that must survive a re-run (the last analysis, the review queue) lives in
`st.session_state`; things that must be created once (the camera, the
HTTP client) live behind `st.cache_resource`.

Layout: sidebar = what to ask + thresholds. Three tabs: Camera (capture and
results), Review queue (the "not sure" answers), History (the decision log).
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from sight import config
from sight.decisions import Decision, Thresholds
from sight.demo import DemoJevClient
from sight.draw import draw_boxes
from sight.frames import make_sample_image
from sight.jev import ChoiceAnswer, HttpJevClient, JevError
from sight.log import log_review, read_log
from sight.pipeline import Analysis, analyze
from sight.questions import Plan, parse_labels

load_dotenv()  # picks up TYPESAFE_API_KEY from a local .env file, if present

st.set_page_config(page_title="Sightline", page_icon="👁️", layout="wide")
st.title("👁️ Sightline")
st.caption("Your webcam, read by Jev: typed answers, calibrated confidence, "
           "and thresholds you set that decide what's automatic and what a person checks.")

state = st.session_state
state.setdefault("last", None)    # the latest Analysis
state.setdefault("queue", [])     # decisions waiting for a person
state.setdefault("seen", None)    # id of the last snapshot we analysed

has_key = bool(os.environ.get(config.API_KEY_ENV))

# ── Sidebar: what to ask, and how cautious to be ────────────────────────────
with st.sidebar:
    demo = st.toggle("Demo mode (no key, canned answers)", value=not has_key,
                     help="Answers come from sight/demo.py, not from Jev.")
    if not demo and not has_key:
        st.error(f"No {config.API_KEY_ENV} found. Add it to .env or use demo mode.")

    source = st.radio("Frame source", ["Snapshot", "Live", "Sample image"],
                      help="Snapshot: browser camera, one frame per click. "
                           "Live: this computer's camera, one frame every N seconds.")

    st.header("Questions (one request)")
    labels = parse_labels(st.text_area(
        "Classify: labels (Choice)", "mug, phone, laptop, book, nothing",
        help="Comma-separated, or one 'label: description' per line."))
    objects = [o.strip() for o in st.text_input(
        "Locate: objects (Noul + Box each)", "mug, laptop, book").split(",") if o.strip()]
    checks = [c.strip() for c in st.text_area(
        "Checks: yes/no (Noul)", "Is a person present?").splitlines() if c.strip()]
    quality = st.checkbox("Image quality gate (Score)", value=True)

    st.header("Decision thresholds")
    t = Thresholds(
        choice_accept=st.slider("Choice: accept if P(label) ≥", 0.5, 1.0, config.CHOICE_ACCEPT, 0.01),
        noul_yes=st.slider("Noul: 'yes' if P ≥", 0.5, 1.0, config.NOUL_YES, 0.01),
        noul_no=st.slider("Noul: 'no' if P ≤", 0.0, 0.5, config.NOUL_NO, 0.01),
        box_accept=st.slider("Box: accept if confidence ≥", 0.0, 1.0, config.BOX_ACCEPT, 0.01),
        quality_min=st.slider("Quality: review all if score <", 0.0, 2.0,
                              config.QUALITY_MIN_LEVEL, 0.1),
    )
    save_frames = st.checkbox("Save analysed frames to data/frames/", value=False)

plan = Plan(labels=labels, objects=objects, checks=checks, quality=quality)


@st.cache_resource
def real_client(api_key: str) -> HttpJevClient:
    return HttpJevClient(api_key)


@st.cache_resource
def camera(index: int = 0):
    from sight.camera import Camera  # imported lazily: only live mode needs OpenCV
    return Camera(index)


def client():
    return DemoJevClient() if demo else real_client(os.environ.get(config.API_KEY_ENV, ""))


def run(raw) -> None:
    """Analyse one frame; queue anything the thresholds didn't auto-accept."""
    try:
        a = analyze(client(), raw, plan, t, save_frame=save_frames)
    except (JevError, ValueError) as e:
        st.error(str(e))
        return
    state.last = a
    for d in a.decisions:
        if d.status == "review":
            state.queue.append({"analysis": a, "decision": d})


# ── Rendering one analysis ──────────────────────────────────────────────────
BADGE = {"auto": "✅ auto", "review": "🟠 review", "skipped": "⚪ skipped"}


def show(a: Analysis) -> None:
    left, right = st.columns([3, 2])
    with left:
        st.image(draw_boxes(a.image, a.decisions), width="stretch",
                 caption=f"frame {a.frame_id} · {a.image.width}×{a.image.height}px sent")
    with right:
        if demo:
            st.info("Demo mode: these answers are canned (sight/demo.py), not read "
                    "from the frame.")
        choice = next((d for d in a.decisions if d.role == "classify"), None)
        if choice:
            raw = a.result.answers[choice.question_id]
            assert isinstance(raw, ChoiceAnswer)
            st.metric("Classified as", choice.answer,
                      f"P = {raw.top_probability:.2f} · confidence {raw.confidence:.2f}",
                      delta_color="off")
            st.caption(BADGE[choice.status] + " · " + choice.reason)
            st.bar_chart(pd.DataFrame({"label": list(raw.probabilities),
                                       "probability": list(raw.probabilities.values())}),
                         x="label", y="probability", horizontal=True, height=220,
                         sort="-probability")  # most likely label first
        st.caption(f"Model `{a.result.model}` · {len(a.questions)} questions in ONE request · "
                   f"{a.result.input_tokens} input tokens")

    st.dataframe(pd.DataFrame([{
        "question": d.subject, "type": d.kind, "answer": d.answer,
        "value": None if d.value is None else round(d.value, 3),
        "decision": BADGE[d.status], "why": d.reason,
    } for d in a.decisions]), hide_index=True, width="stretch")
    with st.expander("The request that was sent (JSON part; the JPEG goes in its own part)"):
        st.json(a.request)


cam_tab, queue_tab, hist_tab = st.tabs(
    ["📷 Camera", f"🟠 Review queue ({len(state.queue)})", "📜 History"])

# ── Camera tab ──────────────────────────────────────────────────────────────
with cam_tab:
    if source == "Snapshot":
        shot = st.camera_input("Take a photo")
        # camera_input keeps returning the same photo on every rerun; analyse
        # each photo once, keyed by its file_id.
        if shot is not None and shot.file_id != state.seen:
            state.seen = shot.file_id
            with st.spinner("Asking Jev…"):
                run(shot.getvalue())
            # The tab labels (with the queue count) were drawn before this
            # analysis ran; rerun once so they show the new count.
            st.rerun()
        if state.last:
            show(state.last)

    elif source == "Sample image":
        if st.button("Analyse the sample desk", type="primary"):
            with st.spinner("Asking Jev…"):
                run(make_sample_image())
            st.rerun()  # refresh the queue count in the tab label
        if state.last:
            show(state.last)

    else:  # Live
        interval = st.slider("Seconds between frames (each frame = one paid request)",
                             2, 60, config.DEFAULT_INTERVAL_SECONDS)
        live = st.toggle("Start live analysis", value=False)

        # A fragment re-runs on its own timer without re-running the whole page.
        # run_every=None (toggle off) makes it an ordinary, static fragment.
        @st.fragment(run_every=interval if live else None)
        def live_view() -> None:
            if live:
                cam = camera()
                if not cam.is_open():
                    st.error("Couldn't open the camera. On macOS: System Settings → "
                             "Privacy & Security → Camera → allow your terminal app.")
                    return
                frame = cam.grab()
                if frame is not None:
                    run(frame)
            if state.last:
                show(state.last)
            if live:
                st.caption(f"Next frame in {interval}s. Review-queue count refreshes "
                           "on your next click (fragments redraw only themselves).")

        live_view()

# ── Review queue tab ────────────────────────────────────────────────────────
def review_options(a: Analysis, d: Decision) -> list[str]:
    """What a person can answer, per question type."""
    if d.kind == "choice":
        return list(a.result.answers[d.question_id].probabilities)
    if d.kind == "box":
        return ["found", "wrong place", "not present"]
    return ["yes", "no"]


with queue_tab:
    if not state.queue:
        st.success("Nothing waiting. Answers below your thresholds will appear here.")
    for i, item in enumerate(list(state.queue)):
        a, d = item["analysis"], item["decision"]
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            # Show just this box (if any) so the reviewer sees what Jev meant.
            c1.image(draw_boxes(a.image, [d] if d.box else [], show_skipped=True),
                     width="stretch")
            c2.markdown(f"**{d.subject}** · `{d.kind}` · frame `{a.frame_id}`")
            c2.write(f"Jev: **{d.answer}** · {d.reason}")
            opts = review_options(a, d)
            default = opts.index(d.answer) if d.answer in opts else 0
            pick = c2.radio("Your answer", opts, index=default, horizontal=True,
                            key=f"pick_{a.frame_id}_{d.question_id}")
            if c2.button("Save verdict", key=f"save_{a.frame_id}_{d.question_id}"):
                log_review(a.frame_id, d.question_id, d.subject, d.answer, pick)
                state.queue.remove(item)
                st.rerun()

# ── History tab ─────────────────────────────────────────────────────────────
with hist_tab:
    rows = read_log()
    if not rows:
        st.info("No decisions logged yet.")
    else:
        df = pd.DataFrame(rows)
        counts = df["status"].value_counts()
        cols = st.columns(len(counts))
        for col, (status, n) in zip(cols, counts.items()):
            col.metric(status, int(n))
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("Download log (JSONL)", config.LOG_PATH.read_bytes(),
                           file_name="decisions.jsonl")
