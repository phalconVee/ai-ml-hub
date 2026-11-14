"""The Switchboard against REAL servers: stdio subprocesses and a Streamable HTTP server.

No mocks here. Each test launches the example servers exactly as the app does
(`${PYTHON} -m example_servers...`), or serves one over HTTP with uvicorn on a
free local port, and talks MCP to them.
"""

import socket
import threading
import time

import pytest
import uvicorn

from example_servers import calc_clock
from switchboard.config import ServerSpec
from switchboard.hub import Switchboard


def stdio(name: str, module: str, **kw) -> ServerSpec:
    return ServerSpec(name=name, command="${PYTHON}", args=["-m", f"example_servers.{module}"], **kw)


@pytest.fixture
async def board(tmp_path):
    b = Switchboard()
    await b.connect(stdio("files", "sandbox_files", env={"SANDBOX_ROOT": str(tmp_path)}))
    await b.connect(stdio("calc", "calc_clock"))
    yield b
    await b.disconnect_all()


async def test_merged_namespaced_tools(board):
    assert not board.errors, board.errors
    names = {t.qualified for t in board.tools()}
    assert {"files__read_file", "files__write_file", "calc__calculate", "calc__now"} <= names
    tool = board.find("calc__calculate").for_claude()
    assert tool["description"].startswith("[calc]")
    assert tool["input_schema"]["properties"]["expression"]["type"] == "string"
    # Server instructions are collected for the system prompt.
    assert "sandbox" in board.instructions()["files"]


async def test_routing_and_results(board, tmp_path):
    out = await board.call("calc__calculate", {"expression": "6*7"})
    assert not out.is_error and out.preview.endswith("= 42") and out.server == "calc"

    out = await board.call("files__write_file", {"path": "x.md", "content": "hello"})
    assert not out.is_error and (tmp_path / "x.md").read_text() == "hello"  # the subprocess honoured SANDBOX_ROOT

    out = await board.call("files__read_file", {"path": "../../etc/passwd"})
    assert out.is_error and "outside the sandbox" in out.preview

    out = await board.call("nope__missing", {})
    assert out.is_error and "not available" in out.preview


async def test_approval_policy(board):
    assert not board.needs_approval("files__read_file")      # read-only hint
    assert board.needs_approval("files__write_file")         # writes
    board.connections["files"].spec.approval = "never"
    assert not board.needs_approval("files__write_file")
    board.connections["calc"].spec.approval = "always"
    assert board.needs_approval("calc__calculate")
    board.trusted.add("calc__calculate")                     # "always allow" from the UI
    assert not board.needs_approval("calc__calculate")


async def test_unannotated_tools_count_as_writes():
    from mcp.server.mcpserver import MCPServer
    srv = MCPServer("bare")

    @srv.tool()
    def poke() -> str:
        """No annotations at all."""
        return "poked"

    b = Switchboard()
    await b.connect(ServerSpec(name="bare", url="http://unused"), target=srv)  # in-process
    t = b.find("bare__poke")
    assert not t.read_only and t.destructive and b.needs_approval("bare__poke")
    await b.disconnect_all()


async def test_disconnect_and_reconnect(board):
    await board.disconnect("calc")
    assert board.find("calc__now") is None
    assert (await board.call("calc__now", {})).is_error
    await board.connect(stdio("calc", "calc_clock"))
    assert board.find("calc__now") is not None


async def test_bad_servers_report_errors_not_crashes():
    b = Switchboard()
    await b.connect(ServerSpec(name="ghost", command="definitely-not-a-real-command-xyz"))
    await b.connect(ServerSpec(name="crashy", command="${PYTHON}", args=["-c", "import sys; sys.exit(3)"]))
    await b.connect(ServerSpec(name="needs-key", url="http://localhost/mcp",
                               headers={"Authorization": "Bearer ${SURELY_UNSET_TOKEN_123}"}))
    assert set(b.errors) == {"ghost", "crashy", "needs-key"}
    assert "SURELY_UNSET_TOKEN_123" in b.errors["needs-key"]
    assert b.connections == {}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def http_server_url():
    """Serve the calc server over Streamable HTTP in a background thread."""
    port = _free_port()
    app = calc_clock.server.streamable_http_app()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=5)


async def test_streamable_http_transport(http_server_url, monkeypatch):
    monkeypatch.setenv("TEST_TOKEN", "abc")
    b = Switchboard()
    # headers with ${VAR} expansion travel with every request
    await b.connect(ServerSpec(name="remote-calc", url=http_server_url,
                               headers={"Authorization": "Bearer ${TEST_TOKEN}"}))
    assert not b.errors, b.errors
    out = await b.call("remote-calc__calculate", {"expression": "2**8"})
    assert out.preview.endswith("= 256")
    await b.disconnect_all()
