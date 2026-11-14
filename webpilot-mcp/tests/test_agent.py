"""The agent loop, offline: a scripted fake Claude + the real browser server.

Runs through PilotSession, the same sync-over-background-loop bridge the
Streamlit app uses, so the pause/resume across "reruns" is exercised for real.
"""

import itertools
from types import SimpleNamespace as NS

import pytest

from tests.conftest import make_settings
from webpilot.mcp_server import build_server
from webpilot.session import PilotSession

_ids = itertools.count()


def tool_use(name, **args):
    return NS(type="tool_use", id=f"tu_{next(_ids)}", name=name, input=args)


def say(text):
    return NS(type="text", text=text)


def reply(*blocks):
    stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
    return NS(content=list(blocks), stop_reason=stop)


def last_tool_results(messages):
    """The tool_result blocks in the latest user message."""
    return [b for b in messages[-1]["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]


def find_id(messages, label):
    """Pull an element id for `label` out of the latest read_page result, like Claude would."""
    for block in last_tool_results(messages):
        for line in block["content"][0]["text"].splitlines():
            if line.startswith("[") and f'"{label}' in line:
                return int(line[1 : line.index("]")])
    raise AssertionError(f"{label} not found")


class ScriptedLLM:
    """Plays Claude from a list of functions, one per turn: f(messages) -> reply."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []  # (messages snapshot length, tool names, tool_choice)

    async def step(self, *, system, messages, tools, tool_choice=None):
        self.calls.append({"tools": {t["name"] for t in tools}, "tool_choice": tool_choice, "system": system})
        turn = self.script.pop(0)
        return turn(messages)


def session(settings, llm):
    return PilotSession(settings, llm=llm, server=build_server(settings)).start()


@pytest.fixture
def pilot_factory():
    made = []

    def make(settings, llm):
        s = session(settings, llm)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


def test_read_only_goal_end_to_end(site_url, pilot_factory):
    llm = ScriptedLLM([
        lambda m: reply(say("Opening the site."), tool_use("goto", url=f"{site_url}/index.html")),
        lambda m: reply(tool_use("read_page")),
        lambda m: reply(tool_use("click", element_id=find_id(m, "Rust 2026"))),
        lambda m: reply(tool_use("read_page")),
        lambda m: reply(say("Top Rust story: **Rust 2026 edition ships** (async closures).")),
    ])
    pilot = pilot_factory(make_settings(read_only=True), llm)
    run = pilot.new_run("Find the top Rust story")
    events = list(pilot.advance(run))
    kinds = [e.kind for e in events]

    assert kinds[0] == "text" and kinds[-1] == "final"
    assert "Rust 2026 edition ships" in events[-1].text
    assert "image" in kinds  # the UI got screenshots after actions
    assert run.done and run.steps == 5 and not run.awaiting_approval
    # Read-only: Claude was never offered the write tools.
    assert not llm.calls[0]["tools"] & {"type_text", "submit_form"}
    assert "Read-only mode" in llm.calls[0]["system"]
    # Every tool_use got exactly one tool_result, in the right order.
    for i, msg in enumerate(run.messages):
        if msg["role"] == "assistant":
            uses = [b.id for b in msg["content"] if b.type == "tool_use"]
            if uses:
                results = [b["tool_use_id"] for b in run.messages[i + 1]["content"] if b.get("type") == "tool_result"]
                assert results == uses


def fill_and_submit_script(site_url, after_submit):
    return [
        lambda m: reply(tool_use("goto", url=f"{site_url}/index.html")),
        lambda m: reply(tool_use("read_page")),
        # Two calls in one turn: type (runs) then submit (must pause).
        lambda m: reply(
            tool_use("type_text", element_id=find_id(m, "Your email"), text="me@example.com"),
            tool_use("submit_form", element_id=find_id(m, "Sign up")),
        ),
        after_submit,
        lambda m: reply(say("Done.")),
    ]


def test_form_submission_pauses_until_approved(site_url, pilot_factory):
    seen = {}

    def after_submit(m):
        seen["results"] = last_tool_results(m)
        return reply(tool_use("read_page"))

    pilot = pilot_factory(make_settings(read_only=False), ScriptedLLM(fill_and_submit_script(site_url, after_submit)))
    run = pilot.new_run("Sign me up")

    events = list(pilot.advance(run))
    assert events[-1].kind == "approval"
    assert run.awaiting_approval
    assert "me@example.com" in events[-1].text and "/thanks.html" in events[-1].text  # form preview
    # Nothing was submitted yet: still on the index page.
    assert "index.html" in pilot.call(pilot.toolbox.call("read_page", {})).text.split("\n")[0]

    # ... the Streamlit script re-runs, the user clicks Approve ...
    events = list(pilot.advance(run, approved=True))
    assert events[-1].kind == "final" and run.done
    results = seen["results"]
    assert len(results) == 2 and not results[1]["is_error"]  # type_text + submit_form
    assert "thanks.html" in results[1]["content"][0]["text"]


def test_rejected_submission_is_reported_to_claude(site_url, pilot_factory):
    seen = {}

    def after_submit(m):
        seen["results"] = last_tool_results(m)
        return reply(tool_use("read_page"))

    pilot = pilot_factory(make_settings(read_only=False), ScriptedLLM(fill_and_submit_script(site_url, after_submit)))
    run = pilot.new_run("Sign me up")
    list(pilot.advance(run))
    events = list(pilot.advance(run, approved=False))

    declined = seen["results"][1]
    assert declined["is_error"] and "declined" in declined["content"][0]["text"]
    assert any(e.kind == "result" and e.is_error and e.tool == "submit_form" for e in events)
    # The browser never left the page.
    page = pilot.call(pilot.toolbox.call("read_page", {})).text
    assert "index.html" in page.split("\n")[0]


def test_step_limit_forces_a_final_answer(site_url, pilot_factory):
    settings = make_settings(max_steps=3)
    looping = [lambda m: reply(tool_use("goto", url=f"{site_url}/index.html"))] * 3
    llm = ScriptedLLM(looping + [lambda m: reply(say("Partial answer: found the front page."))])
    pilot = pilot_factory(settings, llm)
    run = pilot.new_run("Loop forever")
    events = list(pilot.advance(run))

    assert [e.kind for e in events][-2:] == ["limit", "final"]
    assert run.steps == 4  # 3 budgeted turns + 1 tools-off wrap-up
    assert llm.calls[-1]["tool_choice"] == {"type": "none"}
    assert all(c["tool_choice"] is None for c in llm.calls[:-1])
    # The wrap-up instruction rides in the same user turn as the last tool result.
    last_user = run.messages[-2]["content"]
    assert last_user[0]["type"] == "tool_result" and "Step limit reached" in last_user[-1]["text"]


def test_blocked_domain_comes_back_as_tool_error(site_url, pilot_factory):
    other = site_url.replace("127.0.0.1", "localhost")
    seen = {}

    def check(m):
        seen["r"] = last_tool_results(m)[0]
        return reply(say("That site is blocked."))

    llm = ScriptedLLM([lambda m: reply(tool_use("goto", url=other)), check])
    pilot = pilot_factory(make_settings(deny=("localhost",)), llm)
    run = pilot.new_run("Open localhost")
    events = list(pilot.advance(run))
    assert seen["r"]["is_error"] and "deny list" in seen["r"]["content"][0]["text"]
    assert "localhost" in llm.calls[0]["system"]  # the policy is in the prompt too
    assert events[-1].kind == "final"


def test_llm_failure_ends_run_with_error_event(site_url, pilot_factory):
    def boom(m):
        raise RuntimeError("API down")

    pilot = pilot_factory(make_settings(), ScriptedLLM([boom]))
    run = pilot.new_run("anything")
    events = list(pilot.advance(run))
    assert events[-1].kind == "error" and "API down" in events[-1].text and run.done
