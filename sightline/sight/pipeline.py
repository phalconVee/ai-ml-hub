"""One frame, end to end: prepare -> ask Jev -> route -> log.

The Streamlit app calls only `analyze()`. Tests call it too, with a fake
client, so the whole flow is checked without a key or a camera.
"""

import uuid
from dataclasses import dataclass

from PIL import Image

from sight import config
from sight.decisions import Decision, Thresholds, route
from sight.frames import prepare_frame, to_jpeg
from sight.jev import JevClient, JevResult
from sight.log import log_decisions
from sight.questions import Plan, Question, build_questions, build_request


@dataclass
class Analysis:
    frame_id: str
    image: Image.Image          # the exact frame Jev saw (after resizing)
    questions: list[Question]
    result: JevResult
    decisions: list[Decision]
    request: dict               # kept so the UI can show the raw request


def analyze(client: JevClient, raw: bytes | Image.Image, plan: Plan,
            thresholds: Thresholds, *, log_path=None,
            save_frame: bool = False) -> Analysis:
    image = prepare_frame(raw)                 # upright RGB, within 280..1568 px
    jpeg = to_jpeg(image)
    questions = build_questions(plan)          # typed questions from the sidebar
    request = build_request(questions)         # ONE request, all questions, one image
    result = client.ask(request, jpeg)         # the only network call
    decisions = route(questions, result, thresholds)   # your thresholds decide

    frame_id = uuid.uuid4().hex[:8]
    log_decisions(frame_id, result.model, decisions, path=log_path)
    if save_frame:                             # off by default: it's your camera
        config.FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        (config.FRAMES_DIR / f"{frame_id}.jpg").write_bytes(jpeg)
    return Analysis(frame_id, image, questions, result, decisions, request)
