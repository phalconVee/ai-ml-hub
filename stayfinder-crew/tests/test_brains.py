"""Both brains honour the same contract, without any network."""

import pytest

from stay.llm import BrainError, ClaudeBrain, OllamaBrain, loads_lenient, make_brain

from .fakes import FakeAnthropic, msg, text

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


def test_claude_sends_structured_output_and_effort():
    client = FakeAnthropic([msg(text('{"x": 1}'))])
    out = ClaudeBrain(client).json(system="S", user="U", schema=SCHEMA, effort="low")
    assert out == {"x": 1}
    sent = client.requests[0]
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_config"] == {"effort": "low",
                                     "format": {"type": "json_schema", "schema": SCHEMA}}
    assert sent["messages"] == [{"role": "user", "content": "U"}]


def test_claude_refusal_is_an_error():
    client = FakeAnthropic([msg(text(""), stop_reason="refusal")])
    with pytest.raises(BrainError):
        ClaudeBrain(client).json(system="S", user="U", schema=SCHEMA, effort="low")


def test_ollama_payload_uses_schema_format_and_think():
    sent = []

    def post(url, body):
        sent.append((url, body))
        return {"message": {"role": "assistant", "content": '{"x": 2}', "thinking": "hmm"}}

    brain = OllamaBrain(model="qwen3:4b", host="http://h:1", think=True, http_post=post)
    assert brain.json(system="S", user="U", schema=SCHEMA, effort="low") == {"x": 2}
    url, body = sent[0]
    assert url == "http://h:1/api/chat"
    assert body["format"] == SCHEMA and body["stream"] is False and body["think"] is True
    assert body["messages"][0] == {"role": "system", "content": "S"}
    assert body["options"]["temperature"] == 0


def test_ollama_retries_once_on_broken_json():
    replies = iter(['{"x": ', '{"x": 3}'])
    sent = []

    def post(url, body):
        sent.append(body)
        return {"message": {"content": next(replies)}}

    out = OllamaBrain(http_post=post).json(system="S", user="U", schema=SCHEMA, effort="low")
    assert out == {"x": 3}
    # The retry shows the model its own broken answer plus the error.
    assert sent[1]["messages"][-2] == {"role": "assistant", "content": '{"x": '}
    assert "not valid JSON" in sent[1]["messages"][-1]["content"]


def test_ollama_reports_missing_model_and_offline_server():
    brain = OllamaBrain(http_post=lambda u, b: {"error": "model 'qwen3:4b' not found"})
    with pytest.raises(BrainError, match="not found"):
        brain.json(system="S", user="U", schema=SCHEMA, effort="low")

    def offline(url, body):
        raise ConnectionRefusedError(61, "Connection refused")

    with pytest.raises(BrainError, match="ollama serve"):
        OllamaBrain(http_post=offline).json(system="S", user="U", schema=SCHEMA, effort="low")


@pytest.mark.parametrize("raw", ['{"x": 1}', '```json\n{"x": 1}\n```',
                                 '<think>reasoning…</think>{"x": 1}', 'Sure! {"x": 1} done'])
def test_loads_lenient(raw):
    assert loads_lenient(raw) == {"x": 1}


def test_make_brain_rejects_unknown():
    assert make_brain("ollama").name == "ollama"
    with pytest.raises(ValueError):
        make_brain("gpt")
