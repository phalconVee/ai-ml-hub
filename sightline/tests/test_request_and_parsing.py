"""The wire format: what we send to Jev and how we read what comes back.

No key, no network: the HTTP client talks to an httpx.MockTransport that
inspects the multipart upload and replies with a canned JSON answer.
"""

import json

import httpx
import pytest
from PIL import Image

from sight import config
from sight.frames import make_sample_image, prepare_frame, to_jpeg
from sight.jev import (BoxAnswer, ChoiceAnswer, HttpJevClient, JevError, NoulAnswer,
                       ScoreAnswer, parse_response)
from sight.questions import Plan, build_questions, build_request, parse_labels, slug

PLAN = Plan(labels=parse_labels("mug, phone, nothing"), objects=["coffee mug"],
            checks=["Is a person present?"], quality=True)

# A response in the exact shape of the API reference examples.
RAW = {
    "model": "jev-1.14.0",
    "answers": {
        "classify": {"type": "choice", "choice": "mug", "confidence": 0.85,
                     "probabilities": {"mug": 0.9, "phone": 0.07, "nothing": 0.03}},
        "present_coffee_mug": {"type": "noul", "noul": 0.97},
        "box_coffee_mug": {"type": "box", "confidence": 0.92,
                           "box": {"top": 0.25, "bottom": 0.82, "left": 0.44, "right": 0.84}},
        "check_0_is_a_person_present": {"type": "noul", "noul": 0.02},
        "quality": {"type": "score", "score": 1.8, "confidence": 0.7,
                    "legend": {"0": "dark", "1": "ok", "2": "clear"},
                    "probabilities": {"0": 0.0, "1": 0.2, "2": 0.8}},
    },
    "usage": {"input_tokens": 900, "output_tokens": 40},
}


def test_labels_parse_from_commas_or_lines():
    assert list(parse_labels("mug, phone,  , book")) == ["mug", "phone", "book"]
    lines = parse_labels("cat: a small furry pet\ndog")
    assert lines["cat"] == "a small furry pet"
    assert "`frame`" in lines["dog"]                  # default description points at the image
    assert "No clear object" in parse_labels("a, nothing")["nothing"]


def test_one_request_holds_every_question_and_one_image():
    qs = build_questions(PLAN)
    body = build_request(qs, image_ref="REF-123")

    assert body["model"] == "jev-latest"
    assert body["state"] == {"frame": "REF-123"}                     # image goes in state once
    assert body["assets"] == {"REF-123": {"type": "image", "part": "frame"}}
    assert [q["type"] for q in body["questions"].values()] == \
        ["choice", "noul", "box", "noul", "score"]
    # Every question points at the image by its state key.
    assert all("`frame`" in q["instructions"] for q in body["questions"].values())
    # Box takes no criteria; Choice criteria are the labels.
    assert "criteria" not in body["questions"]["box_coffee_mug"]
    assert list(body["questions"]["classify"]["criteria"]) == ["mug", "phone", "nothing"]
    # Each box is paired with a presence Noul under the same slug.
    assert {"present_coffee_mug", "box_coffee_mug"} <= set(body["questions"])


def test_fresh_reference_per_request_and_no_empty_requests():
    qs = build_questions(PLAN)
    assert build_request(qs)["state"] != build_request(qs)["state"]
    with pytest.raises(ValueError):
        build_request([])
    # A single label can't be a Choice: it is left out rather than sent broken.
    assert build_questions(Plan(labels={"mug": "x"}, quality=False)) == []


def test_slug_makes_safe_ids():
    assert slug("Coffee Mug!") == "coffee_mug"
    assert slug("???") == "item"


def test_parse_response_gives_typed_answers():
    r = parse_response(RAW)
    assert r.model == "jev-1.14.0" and r.input_tokens == 900
    c = r.answers["classify"]
    assert isinstance(c, ChoiceAnswer) and c.choice == "mug" and c.top_probability == 0.9
    s = r.answers["quality"]
    assert isinstance(s, ScoreAnswer) and s.legend[2] == "clear" and s.probabilities[2] == 0.8
    assert isinstance(r.answers["present_coffee_mug"], NoulAnswer)
    b = r.answers["box_coffee_mug"]
    assert isinstance(b, BoxAnswer) and (b.left, b.right, b.top, b.bottom) == (0.44, 0.84, 0.25, 0.82)


def test_box_confidence_may_be_null():
    raw = {"model": "m", "usage": {}, "answers": {"b": {
        "type": "box", "confidence": None,
        "box": {"top": 0.2, "bottom": 0.66, "left": 0.19, "right": 0.68}}}}
    assert parse_response(raw).answers["b"].confidence is None


def test_frames_are_resized_into_jevs_limits():
    big = prepare_frame(Image.new("RGB", (4000, 3000)))
    assert max(big.size) == config.SEND_LONG_SIDE
    tiny = prepare_frame(Image.new("RGBA", (200, 100)))
    assert min(tiny.size) >= config.MIN_SIDE and tiny.mode == "RGB"
    assert max(tiny.size) <= config.MAX_SIDE
    jpeg = to_jpeg(prepare_frame(make_sample_image()))
    assert jpeg[:2] == b"\xff\xd8"                    # a JPEG, ready to upload


def _multipart_parts(request: httpx.Request) -> list[tuple[str, bytes]]:
    """Tiny multipart reader: [(part name, body)] in order."""
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    parts = []
    for chunk in request.content.split(b"--" + boundary)[1:-1]:
        head, _, body = chunk.partition(b"\r\n\r\n")
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        parts.append((name, body.rstrip(b"\r\n")))
    return parts


def test_http_client_sends_multipart_with_images_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json=RAW)

    client = HttpJevClient("test-key", http=httpx.Client(transport=httpx.MockTransport(handler)))
    body = build_request(build_questions(PLAN))
    result = client.ask(body, b"\xff\xd8fake-jpeg")

    req = seen["request"]
    assert str(req.url) == config.JEV_URL
    assert req.headers["authorization"] == "Bearer test-key"
    assert req.headers["typesafe-danger-zone-images"] == "true"   # images/Box are opt-in
    parts = _multipart_parts(req)
    assert parts[0][0] == "request"                               # JSON must come FIRST
    assert json.loads(parts[0][1]) == body
    assert parts[1] == ("frame", b"\xff\xd8fake-jpeg")            # name == assets[ref].part
    assert result.answers["classify"].choice == "mug"


def test_http_client_retries_429_then_succeeds_and_raises_on_400():
    calls, waits = [], []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0.5"})
        return httpx.Response(200, json=RAW)

    c = HttpJevClient("k", http=httpx.Client(transport=httpx.MockTransport(handler)),
                      sleep=waits.append)
    c.ask(build_request(build_questions(PLAN)), b"x")
    assert len(calls) == 2 and waits == [0.5]

    bad = HttpJevClient("k", http=httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(400, json={"detail": "images not enabled"}))))
    with pytest.raises(JevError, match="400.*images not enabled"):
        bad.ask(build_request(build_questions(PLAN)), b"x")


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        HttpJevClient()
