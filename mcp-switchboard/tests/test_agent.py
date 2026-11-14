"""The agent loop end to end: scripted fake Claude, REAL MCP servers (stdio subprocesses).

The fake plays Claude's part from a script and records what it was sent, so we
can check the loop's contract: parallel calls, approval pauses, resume, denial,
results in one message, and the history staying valid for the API.
"""

import itertools
from types import SimpleNamespace as NS

import pytest

from switchboard.agent import close_dangling, run_turn
from switchboard.config import ServerSpec
from switchboard.hub import Switchboard

_ids = itertools.count()


def tool_use(name, args):
    return NS(type="tool_use", id=f"tu_{next(_ids)}", name=name, input=args)


def text(t):
    return NS(type="text", text=t)


def reply(*blocks, stop="tool_use"):
    return NS(content=[{"type": b.type, **{k: v for k, v in vars(b).items() if k != "type"}} for b in blocks],
              stop_reason=stop)


class ScriptedLLM:
    """Returns the next scripted response on each call; keeps every request it received."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    async def step(self, **kwargs):
        # Snapshot: the loop mutates `messages` after we return.
        self.requests.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        return self.responses.pop(0)


@pytest.fixture
async def board(tmp_path):
    b = Switchboard()
    spec = dict(command="${PYTHON}")
    await b.connect(ServerSpec(name="files", args=["-m", "example_servers.sandbox_files"],
                               env={"SANDBOX_ROOT": str(tmp_path)}, **spec))
    await b.connect(ServerSpec(name="calc", args=["-m", "example_servers.calc_clock"], **spec))
    assert not b.errors
    yield b
    await b.disconnect_all()


async def test_read_only_calls_run_without_asking(board):
    llm = ScriptedLLM(
        reply(text("Let me compute."), tool_use("calc__calculate", {"expression": "12*12"}),
              tool_use("calc__days_between", {"start": "2026-01-01", "end": "2026-01-31"})),
        reply(text("144, and 30 days."), stop="end_turn"),
    )
    messages = [{"role": "user", "content": "maths please"}]
    events = []
    result = await run_turn(llm, board, messages, model="m", effort="low",
                            on_event=lambda k, d: events.append(k))
    assert result.status == "done"

    # Both calls answered in ONE user message, in order.
    results = messages[2]["content"]
    assert [r["type"] for r in results] == ["tool_result", "tool_result"]
    assert "= 144" in results[0]["content"][0]["text"] and "30 days" in results[1]["content"][0]["text"]
    assert events.count("tool_start") == 2

    # Claude was offered every server's tools plus the built-in one, and server notes in the system prompt.
    names = {t["name"] for t in llm.requests[0]["tools"]}
    assert {"calc__calculate", "files__write_file", "switchboard__list_servers"} <= names
    assert "Notes from the connected servers" in llm.requests[0]["system"]


async def test_write_pauses_for_approval_then_resumes(board, tmp_path):
    write = tool_use("files__write_file", {"path": "plan.md", "content": "step 1"})
    read = tool_use("files__list_dir", {})
    llm = ScriptedLLM(
        reply(write, read),
        reply(text("Saved."), stop="end_turn"),
    )
    messages = [{"role": "user", "content": "save my plan"}]

    first = await run_turn(llm, board, messages, model="m", effort="low")
    assert first.status == "needs_approval"
    assert [p.tool for p in first.pending] == ["write_file"]       # list_dir is read-only: not pending
    assert first.pending[0].server == "files" and first.pending[0].args["path"] == "plan.md"
    assert not (tmp_path / "plan.md").exists()                      # nothing ran yet, not even list_dir
    assert len(llm.requests) == 1

    # The user clicks Approve: same messages list, now with a decision.
    second = await run_turn(llm, board, messages, model="m", effort="low", decisions={write.id: True})
    assert second.status == "done"
    assert (tmp_path / "plan.md").read_text() == "step 1"
    assert messages[-1]["content"][0]["text"] == "Saved."


async def test_denied_call_is_reported_to_claude(board, tmp_path):
    delete = tool_use("files__delete_path", {"path": "welcome.md"})
    llm = ScriptedLLM(reply(delete), reply(text("OK, I won't."), stop="end_turn"))
    messages = [{"role": "user", "content": "delete it"}]
    assert (await run_turn(llm, board, messages, model="m", effort="low")).status == "needs_approval"
    result = await run_turn(llm, board, messages, model="m", effort="low", decisions={delete.id: False})
    assert result.status == "done"
    tool_result = llm.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] and "denied" in tool_result["content"][0]["text"]


async def test_builtin_list_servers_and_tools_never_empty(board):
    await board.disconnect_all()  # every server switched off...
    llm = ScriptedLLM(reply(tool_use("switchboard__list_servers", {})),
                      reply(text("No servers are on."), stop="end_turn"))
    messages = [{"role": "user", "content": "what can you do?"}]
    assert (await run_turn(llm, board, messages, model="m", effort="low")).status == "done"
    assert [t["name"] for t in llm.requests[0]["tools"]] == ["switchboard__list_servers"]  # ...tools still non-empty
    assert "No servers" in messages[2]["content"][0]["content"][0]["text"]


async def test_step_limit_and_dangling_calls(board, monkeypatch):
    monkeypatch.setattr("switchboard.agent.MAX_AGENT_STEPS", 2)
    llm = ScriptedLLM(*[reply(tool_use("calc__now", {})) for _ in range(5)])  # Claude never stops
    messages = [{"role": "user", "content": "loop forever"}]
    assert (await run_turn(llm, board, messages, model="m", effort="low")).status == "step_limit"
    # History ends with an unanswered tool_use; the API would reject a new user message after it.
    close_dangling(messages)
    last = messages[-1]
    assert last["role"] == "user" and last["content"][0]["type"] == "tool_result"
    assert last["content"][0]["tool_use_id"] == messages[-2]["content"][0]["id"]
