"""MCP Switchboard: the Streamlit UI.

    uv run streamlit run app.py

Layout
    sidebar  model + effort, every server with an on/off toggle, its status,
             its tools (lock = needs approval), add/edit/save servers
    main     the chat. Each tool call shows its server, arguments and result.
             Calls that need approval stop the agent and show a form.

How a Streamlit app "remembers": the script re-runs top to bottom on every
click, so anything that must survive lives in `st.session_state`:
    rt        the Runtime (background event loop + Switchboard with live connections)
    specs     the server configs (the same objects the Switchboard holds)
    messages  the conversation in Messages API format: the single source of truth
    pending   tool calls awaiting approval (or None)
"""

import asyncio
import base64
import json
import os
import queue
import shlex

import streamlit as st
from pydantic import ValidationError

from switchboard import settings
from switchboard.agent import PendingCall, close_dangling, run_turn
from switchboard.config import ServerSpec, dump_config, load_config, parse_config, save_config
from switchboard.hub import SEP
from switchboard.llm import LLM
from switchboard.runtime import Runtime

st.set_page_config(page_title="MCP Switchboard", page_icon="🎛️", layout="wide")
S = st.session_state


# =============================================================================
# One-time setup per browser session
# =============================================================================

def connect_all(specs: list[ServerSpec]) -> None:
    """Connect every enabled server at once (concurrently: stdio servers take ~1s each to boot)."""
    board = S.rt.board

    async def go():
        await asyncio.gather(*(board.connect(s) for s in specs if s.enabled))

    with st.spinner("Connecting MCP servers..."):
        S.rt.run(go())


if "rt" not in S:
    S.rt = Runtime()
    S.specs = load_config(settings.CONFIG_PATH)
    S.messages = []
    S.pending = None
    S.last_error = None
    connect_all(S.specs)

board = S.rt.board


# =============================================================================
# Sidebar: model and servers
# =============================================================================

def toggle_server(spec: ServerSpec) -> None:
    """on_change callback for a server's toggle. Runs before the script re-runs."""
    spec.enabled = S[f"on_{spec.name}"]
    if spec.enabled:
        S.rt.run(board.connect(spec))
    else:
        S.rt.run(board.disconnect(spec.name))


def set_policy(spec: ServerSpec) -> None:
    spec.approval = S[f"policy_{spec.name}"]  # the Switchboard holds this same object: takes effect at once


def remove_server(spec: ServerSpec) -> None:
    S.rt.run(board.disconnect(spec.name))
    S.specs = [s for s in S.specs if s.name != spec.name]


def server_card(spec: ServerSpec) -> None:
    conn = board.connections.get(spec.name)
    st.toggle(f"**{spec.name}**", value=conn is not None, key=f"on_{spec.name}",
              on_change=toggle_server, args=(spec,), help=spec.description or None)
    if conn:
        st.caption(f"🟢 {spec.transport} · {len(conn.tools)} tools · {spec.description}")
    elif spec.name in board.errors:
        st.caption(f"🔴 {board.errors[spec.name]}")
    else:
        st.caption(f"⚪ off · {spec.transport} · {spec.description}")

    with st.expander("tools & settings", expanded=False):
        if conn:
            for t in conn.tools:
                lock = "🔒" if board.needs_approval(t.qualified) else "✅"
                first_line = (t.description or "").strip().splitlines()[0:1]
                st.markdown(f"{lock} `{t.name}`  \n<small>{first_line[0] if first_line else ''}</small>",
                            unsafe_allow_html=True)
        st.selectbox("Ask before calling", ["writes", "always", "never"],
                     index=["writes", "always", "never"].index(spec.approval),
                     key=f"policy_{spec.name}", on_change=set_policy, args=(spec,),
                     help="writes: tools not marked read-only by the server. always: every call. never: trust it.")
        target = " ".join([spec.command, *spec.args]) if spec.command else spec.url
        st.code(target, language="bash")
        st.button("Remove server", key=f"rm_{spec.name}", on_click=remove_server, args=(spec,))


def parse_pairs(text: str) -> dict[str, str]:
    """'KEY=value' lines -> dict (for env vars and HTTP headers)."""
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def add_server_form() -> None:
    with st.form("add_server", clear_on_submit=True):
        name = st.text_input("Name", placeholder="my-server")
        transport = st.radio("Transport", ["stdio", "Streamable HTTP"], horizontal=True)
        command = st.text_input("Command (stdio)", placeholder="npx -y @some/mcp-server ./data")
        url = st.text_input("URL (HTTP)", placeholder="http://127.0.0.1:8765/mcp")
        pairs = st.text_area("Env vars (stdio) or headers (HTTP), KEY=value per line",
                             placeholder="API_TOKEN=${MY_TOKEN}")
        approval = st.selectbox("Ask before calling", ["writes", "always", "never"])
        if not st.form_submit_button("Add & connect"):
            return
    try:
        if transport == "stdio":
            parts = shlex.split(command)
            spec = ServerSpec(name=name, command=parts[0] if parts else None, args=parts[1:],
                              env=parse_pairs(pairs), approval=approval)
        else:
            spec = ServerSpec(name=name, url=url or None, headers=parse_pairs(pairs), approval=approval)
    except ValidationError as exc:
        st.error(exc.errors()[0]["msg"])
        return
    if any(s.name == spec.name for s in S.specs):
        st.error(f"A server called {spec.name!r} already exists.")
        return
    S.specs.append(spec)
    S.rt.run(board.connect(spec))
    st.rerun()


def raw_json_editor() -> None:
    text = st.text_area("servers.json", dump_config(S.specs), height=300, label_visibility="collapsed")
    if st.button("Apply JSON"):
        try:
            specs = parse_config(json.loads(text))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            st.error(f"Invalid config: {exc}")
            return
        S.rt.run(board.disconnect_all())
        board.errors.clear()
        S.specs = specs
        connect_all(specs)
        st.rerun()


with st.sidebar:
    st.title("🎛️ MCP Switchboard")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("ANTHROPIC_API_KEY is not set (see .env.example). Servers and tools work; chat won't.")
    model = st.selectbox("Model", settings.MODELS, index=settings.MODELS.index(settings.DEFAULT_MODEL)
                         if settings.DEFAULT_MODEL in settings.MODELS else 0)
    effort = st.select_slider("Effort", settings.EFFORTS, value=settings.DEFAULT_EFFORT)

    st.subheader("Servers")
    for spec in S.specs:
        server_card(spec)
    with st.expander("➕ Add a server"):
        add_server_form()
    with st.expander("{ } Edit as JSON"):
        raw_json_editor()
    col1, col2 = st.columns(2)
    if col1.button("💾 Save config", use_container_width=True, help=str(settings.CONFIG_PATH)):
        save_config(settings.CONFIG_PATH, S.specs)
        st.toast(f"Saved {settings.CONFIG_PATH.name}")
    if col2.button("🧹 New chat", use_container_width=True):
        S.messages, S.pending, S.last_error = [], None, None
        st.rerun()


# =============================================================================
# Main area: the transcript
# =============================================================================

def split_name(qualified: str) -> tuple[str, str]:
    server, _, tool = qualified.partition(SEP)
    return server, tool or qualified


def render_result_blocks(blocks) -> None:
    for b in blocks if isinstance(blocks, list) else [{"type": "text", "text": str(blocks)}]:
        if b.get("type") == "image":
            st.image(base64.b64decode(b["source"]["data"]))
        elif b.get("type") == "text":
            text = b["text"]
            st.code(text[:4000] + ("\n..." if len(text) > 4000 else ""), language=None, wrap_lines=True)


def render_call(use: dict, result: dict | None, pending_ids: set[str]) -> None:
    """One tool call: which server, which tool, the arguments, the result."""
    server, tool = split_name(use["name"])
    if use["id"] in pending_ids:
        icon, state = "⏳", "awaiting approval"
    elif result is None:
        icon, state = "⏭️", "not run"
    elif result.get("is_error"):
        denied = "denied" in json.dumps(result.get("content"))[:200]
        icon, state = ("🚫", "denied") if denied else ("❌", "error")
    else:
        icon, state = "✅", "ok"
    with st.expander(f"{icon} **{server}** › `{tool}` · {state}"):
        st.caption("arguments")
        st.json(use.get("input") or {})
        if result is not None:
            st.caption("result")
            render_result_blocks(result.get("content"))


def render_transcript(messages: list[dict], pending_ids: set[str]) -> None:
    # Index every tool_result by the id of the call it answers.
    results = {b["tool_use_id"]: b for m in messages if m["role"] == "user" and isinstance(m["content"], list)
               for b in m["content"] if b.get("type") == "tool_result"}
    bubble = None
    for m in messages:
        if m["role"] == "user":
            if isinstance(m["content"], str):  # a real user message (tool results render with their calls)
                st.chat_message("user").markdown(m["content"])
                bubble = None
            continue
        # One assistant bubble per user message, however many loop steps it took.
        bubble = bubble or st.chat_message("assistant")
        with bubble:
            for b in m["content"]:
                if b["type"] == "text" and b["text"].strip():
                    st.markdown(b["text"])
                elif b["type"] == "tool_use":
                    render_call(b, results.get(b["id"]), pending_ids)
                # thinking blocks are kept in the history but not shown


def approval_form(pending: list[PendingCall]) -> dict[str, bool] | None:
    """Ask the human. Returns {tool_use_id: approved} once submitted."""
    with st.form("approve"):
        st.markdown(f"#### 🔒 Claude wants to run {len(pending)} call(s) that need your OK")
        choices, trust = {}, {}
        for p in pending:
            warn = " ⚠️ can destroy data" if p.destructive else ""
            st.markdown(f"**{p.server}** › `{p.tool}`{warn}")
            st.json(p.args)
            c1, c2 = st.columns([2, 1])
            choices[p.id] = c1.radio("Decision", ["Approve", "Deny"], key=f"d_{p.id}",
                                     horizontal=True, label_visibility="collapsed")
            trust[p.id] = (p.qualified, c2.checkbox("Always allow this tool", key=f"t_{p.id}"))
        if st.form_submit_button("Continue", type="primary"):
            decisions = {pid: choice == "Approve" for pid, choice in choices.items()}
            for pid, (qualified, always) in trust.items():
                if always and decisions[pid]:
                    board.trusted.add(qualified)  # no more prompts for this tool this session
            return decisions
    return None


def advance(decisions: dict[str, bool] | None = None) -> None:
    """Run the agent on the background loop, streaming progress into a status box."""
    events: queue.Queue = queue.Queue()
    # on_event is called on the background thread, where st.* calls are not
    # allowed. So it only enqueues; this (Streamlit) thread drains the queue.
    future = S.rt.submit(run_turn(LLM(), board, S.messages, model=model, effort=effort,
                                  decisions=decisions, on_event=lambda k, d: events.put((k, d))))
    with st.status("Working...", expanded=True) as status:
        while not future.done() or not events.empty():
            try:
                kind, data = events.get(timeout=0.1)
            except queue.Empty:
                continue
            if kind == "thinking":
                status.update(label="Claude is thinking...")
            elif kind == "tool_start":
                server, tool = split_name(data["name"])
                status.write(f"🔧 **{server}** › `{tool}` {json.dumps(data['args'])[:150]}")
                status.update(label=f"Calling {server} › {tool}...")
        try:
            result = future.result()
        except Exception as exc:  # API key missing, network down, bad request...
            S.last_error = f"{type(exc).__name__}: {exc}"
            status.update(label="Failed", state="error")
            return
        status.update(label="Done", state="complete", expanded=False)
    S.last_error = None
    S.pending = result.pending if result.status == "needs_approval" else None
    if result.status == "step_limit":
        S.last_error = f"Stopped after {settings.MAX_AGENT_STEPS} steps. Send a message to continue."
    elif result.status == "refused":
        S.last_error = "Claude declined this request."


st.title("Chat")
tool_count = len(board.tools())
st.caption(f"{len(board.connections)} server(s) on · {tool_count} tools · {model} · effort {effort}")

pending_ids = {p.id for p in S.pending} if S.pending else set()
render_transcript(S.messages, pending_ids)
if S.last_error:
    st.error(S.last_error)

if S.pending:
    decisions = approval_form(S.pending)
    if decisions is not None:
        S.pending = None
        advance(decisions)
        st.rerun()

if not S.messages:
    st.markdown("Try one of these, or ask anything:")
    examples = [
        "What time is it in Lagos, London and Tokyo right now?",
        "List the sandbox and summarise welcome.md.",
        "Look up the Model Context Protocol on Wikipedia and save a 5-line summary to notes/mcp.md.",
    ]
    for ex in examples:
        if st.button(ex):
            S.queued = ex
            st.rerun()

prompt = st.chat_input("Ask something (approve pending calls first)" if S.pending else "Ask something",
                       disabled=bool(S.pending)) or S.pop("queued", None)
if prompt:
    close_dangling(S.messages)  # answer any orphaned tool calls so the API accepts the history
    S.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    advance()
    st.rerun()
