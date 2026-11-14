"""The real MCP server, connected in-process, with a fake speech-to-text engine.

Everything goes through MCP exactly as a client would see it: list_tools,
call_tool, resources, prompts. Only Whisper and audio decoding are faked.
"""

import json

import numpy as np
import pytest

from soundcheck import server as srv
from soundcheck import transcribers
from soundcheck.config import settings
from soundcheck.toolbox import open_toolbox
from soundcheck.transcribers import TranscriptionResult
from soundcheck.transcript import Segment
from tests.conftest import LINES, synth_voice

EXPECTED_TOOLS = {"transcribe_audio", "list_recordings", "get_transcript", "identify_speakers",
                  "summarize_recording", "extract_topics", "sentiment_timeline", "search_transcript"}


@pytest.fixture
def audio_file(tmp_path):
    p = tmp_path / "show.wav"
    p.write_bytes(b"RIFF fake audio bytes")  # never decoded: the engine is faked
    return p


@pytest.fixture
def fake_engine(monkeypatch):
    calls = []

    def fake(path, language=None, on_progress=None):
        calls.append(path)
        if on_progress:
            on_progress(0.5)
            on_progress(1.0)
        return TranscriptionResult([Segment(start=a, end=b, text=t) for a, b, t in LINES],
                                   duration=45.0, language="en", engine="fake")

    monkeypatch.setitem(transcribers.ENGINES, "local", fake)
    return calls


async def _transcribe(tb, path) -> dict:
    text, err = await tb.call("transcribe_audio", {"file_path": str(path)})
    assert not err, text
    return json.loads(text)


async def test_advertises_tools_with_schemas():
    async with open_toolbox(srv.server) as tb:
        assert EXPECTED_TOOLS <= set(tb.tools)
        defs = {d["name"]: d for d in tb.claude_tools()}
        schema = defs["transcribe_audio"]["input_schema"]
        assert schema["required"] == ["file_path"]
        assert "ctx" not in schema["properties"]  # the Context parameter is hidden from clients
        assert "audio_id" in defs["search_transcript"]["input_schema"]["required"]
        assert "transcribe_audio" in tb.instructions


async def test_transcribe_then_cache(audio_file, fake_engine):
    async with open_toolbox(srv.server) as tb:
        first = await _transcribe(tb, audio_file)
        second = await _transcribe(tb, audio_file)
        listing = json.loads((await tb.call("list_recordings", {}))[0])
    assert first["audio_id"] == second["audio_id"] and len(first["audio_id"]) == 12
    assert first["cached"] is False and second["cached"] is True
    assert len(fake_engine) == 1  # the engine ran once
    assert first["duration"] == "00:45" and "[00:21] Then Maria Lopez" in first["preview"]
    assert listing[0]["audio_id"] == first["audio_id"]


async def test_bad_inputs_are_tool_errors_with_reasons(tmp_path, audio_file, fake_engine):
    notes = tmp_path / "notes.txt"
    notes.write_text("hi")
    async with open_toolbox(srv.server) as tb:
        missing = await tb.call("transcribe_audio", {"file_path": str(tmp_path / "nope.mp3")})
        not_audio = await tb.call("transcribe_audio", {"file_path": str(notes)})
        bad_engine = await tb.call("transcribe_audio", {"file_path": str(audio_file), "engine": "magic"})
        unknown_id = await tb.call("search_transcript", {"audio_id": "ffffffffffff", "query": "x"})
    for text, is_error in (missing, not_audio, bad_engine, unknown_id):
        assert is_error
    assert "File not found" in missing[0]
    assert "doesn't look like audio" in not_audio[0]
    assert "Unknown engine" in bad_engine[0]
    assert "transcribe_audio first" in unknown_id[0]


async def test_assemblyai_needs_a_key(audio_file, monkeypatch):
    monkeypatch.setattr(settings, "assemblyai_key", "")
    async with open_toolbox(srv.server) as tb:
        text, err = await tb.call("transcribe_audio", {"file_path": str(audio_file), "engine": "assemblyai"})
    assert err and "ASSEMBLYAI_API_KEY" in text


async def test_analysis_tools_end_to_end(audio_file, fake_engine):
    async with open_toolbox(srv.server) as tb:
        aid = (await _transcribe(tb, audio_file))["audio_id"]
        window, _ = await tb.call("get_transcript", {"audio_id": aid, "start_seconds": 20, "end_seconds": 30})
        search = json.loads((await tb.call("search_transcript", {"audio_id": aid, "query": "billing service"}))[0])
        senti = json.loads((await tb.call("sentiment_timeline", {"audio_id": aid, "window_seconds": 15}))[0])
        summary = json.loads((await tb.call("summarize_recording", {"audio_id": aid}))[0])
        topics = json.loads((await tb.call("extract_topics", {"audio_id": aid}))[0])
    assert window.splitlines()[0].startswith("[00:15]") and "[00:39]" not in window
    assert search["hits"][0]["time"] == "00:39"
    assert len(senti["points"]) == 3 and senti["overall"]["mood"] in {"positive", "neutral", "negative"}
    assert summary["method"].startswith("local") and summary["key_sentences"]
    assert any(e["name"] == "Maria Lopez" for e in topics["entities"])


async def test_identify_speakers_labels_and_persists(audio_file, fake_engine, monkeypatch):
    # Alternate two synthetic voices to match the 8 transcript segments' timings.
    def fake_decode(path):
        parts = [synth_voice(b - a, *((110, 700) if i % 2 == 0 else (230, 2200)), seed=i)
                 for i, (a, b, _) in enumerate(LINES)]
        return np.concatenate(parts)

    monkeypatch.setattr(srv, "_decode", fake_decode)
    async with open_toolbox(srv.server) as tb:
        aid = (await _transcribe(tb, audio_file))["audio_id"]
        result = json.loads((await tb.call("identify_speakers", {"audio_id": aid}))[0])
        text, _ = await tb.call("get_transcript", {"audio_id": aid})
        senti = json.loads((await tb.call("sentiment_timeline", {"audio_id": aid}))[0])
    assert result["speakers"] == 2 and result["turns"] == 8
    assert text.splitlines()[0].startswith("[00:00] Speaker 1:")
    assert text.splitlines()[1].startswith("[00:04] Speaker 2:")
    assert set(senti["by_speaker"]) == {"Speaker 1", "Speaker 2"}


async def test_claude_summary_path_and_fallback(audio_file, fake_engine, monkeypatch):
    from soundcheck.llm import KeyPoint, Summary

    class FakeLLM:
        async def structured(self, *, system, prompt, schema):
            assert schema is Summary and "[00:21]" in prompt  # Claude sees timestamped lines
            return Summary(headline="A platform team story", summary="...",
                           key_points=[KeyPoint(point="Outage", timestamp="00:09")], action_items=["Move billing"])

    class BrokenLLM:
        async def structured(self, **kw):
            raise ConnectionError("offline")

    monkeypatch.setattr(settings, "analysis_mode", "claude")
    async with open_toolbox(srv.server) as tb:
        aid = (await _transcribe(tb, audio_file))["audio_id"]
        monkeypatch.setattr(srv, "make_llm", FakeLLM)
        good = json.loads((await tb.call("summarize_recording", {"audio_id": aid}))[0])
        monkeypatch.setattr(srv, "make_llm", BrokenLLM)
        degraded = json.loads((await tb.call("summarize_recording", {"audio_id": aid}))[0])
    assert good["method"].startswith("claude") and good["action_items"] == ["Move billing"]
    assert degraded["method"].startswith("local") and "ConnectionError" in degraded["note"]


async def test_resource_and_prompt(audio_file, fake_engine):
    async with open_toolbox(srv.server) as tb:
        aid = (await _transcribe(tb, audio_file))["audio_id"]
        templates = await tb.client.list_resource_templates()
        res = await tb.client.read_resource(f"transcript://{aid}")
        prompt = await tb.client.get_prompt("analyze_recording", {"file_path": "/tmp/x.mp3"})
    assert any(t.uri_template == "transcript://{audio_id}" for t in templates.resource_templates)
    assert "[00:39]" in res.contents[0].text
    assert "/tmp/x.mp3" in prompt.messages[0].content.text
