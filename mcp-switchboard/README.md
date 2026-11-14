# mcp-switchboard: one chat, many MCP servers

A Streamlit chat app where Claude can use tools from **several MCP servers at once**. You plug servers in (stdio commands or Streamable HTTP URLs), flick each one on or off, see every tool each server offers, and chat. Every tool call shows up in the transcript with **which server ran it, the arguments and the result**. Anything that writes stops and waits for your **Approve / Deny**.

It ships with three keyless example servers, ready to switch on: a filesystem jailed to `./sandbox`, a calculator + clock, and a web reader (search, Wikipedia, read a page). The only key you need is `ANTHROPIC_API_KEY`.

No agent framework: the hub, the agent loop and the approval pause are each a page of plain Python, commented so they can be read line by line.

## The idea in one paragraph

MCP turns "give the model a tool" into a plug: any server that speaks the protocol works with any client. The interesting part of a *client* is what happens when you plug in many at once. Tool names collide (two servers both have `search`). You want to unplug one mid-conversation. Some tools read, some delete files, and you want a human in the loop for the second kind without babysitting the first. And a web UI that re-runs its script on every click has to hold long-lived async connections somewhere. This project is those four problems, solved small.

## Architecture

```
 ┌───────────────────────── Streamlit (app.py) ──────────────────────────┐
 │ sidebar: model · effort · servers [on/off] · tools 🔒/✅ · add / JSON   │
 │ main:    chat transcript · tool-call cards · approval form            │
 └───────────────┬───────────────────────────────────────▲───────────────┘
   runtime.run(coro) / submit(coro)          result, progress events (queue)
                 ▼                                       │
 ┌──────────── background asyncio loop (runtime.py, own thread) ─────────┐
 │                                                                        │
 │   agent.run_turn()  ──step()──▶  Claude (llm.py, Messages API + tools) │
 │        │   ▲                                                           │
 │        │   └── returns "needs_approval" ──▶ UI shows form, calls again │
 │        ▼                                                               │
 │   Switchboard (hub.py)   tools named  server__tool                     │
 │     ├─ keeper task ─ Client ─ stdio ──▶ python -m example_servers.sandbox_files
 │     ├─ keeper task ─ Client ─ stdio ──▶ python -m example_servers.calc_clock
 │     ├─ keeper task ─ Client ─ stdio ──▶ python -m example_servers.web_reader
 │     └─ keeper task ─ Client ─ HTTP  ──▶ http://127.0.0.1:8765/mcp  (any URL)
 └────────────────────────────────────────────────────────────────────────┘
```

One user message, start to finish:

1. The UI appends `{"role": "user", ...}` to the message list and calls `run_turn()` on the background loop.
2. `run_turn` sends the history plus **every enabled server's tools** (renamed `server__tool`) to Claude.
3. Claude answers with text, or with `tool_use` blocks (often several in parallel).
4. For each call, `needs_approval()` checks the server's policy and the tool's MCP annotations. If any call needs a human, `run_turn` **returns** `needs_approval`. Nothing has run yet.
5. The UI shows the form. On *Continue*, it calls `run_turn()` again with `{tool_use_id: approved?}`. The message list itself says where we were.
6. Approved calls run concurrently, each routed to its own server; denied ones go back to Claude as errors ("the user denied this"). All results return in one message, and the loop asks Claude again until it answers in plain text.

## Code map: read in this order

| File | Role |
|------|------|
| [`example_servers/calc_clock.py`](example_servers/calc_clock.py) | The simplest MCP server: four read-only tools, and a calculator that never calls `eval()` |
| [`example_servers/sandbox_files.py`](example_servers/sandbox_files.py) | A file server jailed to one folder, with `read_only` / `destructive` **annotations** on every tool |
| [`example_servers/web_reader.py`](example_servers/web_reader.py) | Keyless search + page reader. Runs over **stdio or Streamable HTTP**, and blocks private-network URLs |
| [`switchboard/settings.py`](switchboard/settings.py) | Every knob: models, effort, limits, timeouts |
| [`switchboard/config.py`](switchboard/config.py) | `ServerSpec`, the Claude-Desktop-compatible `servers.json` format, `${VAR}` expansion |
| [`switchboard/hub.py`](switchboard/hub.py) | **The core:** one keeper task per connection, merged + namespaced tools, the approval policy, call routing, MCP→Claude result conversion |
| [`switchboard/llm.py`](switchboard/llm.py) | The only file that talks to Claude (one method, `step()`) |
| [`switchboard/agent.py`](switchboard/agent.py) | **The other core:** the tool loop that can pause for approval and resume |
| [`switchboard/runtime.py`](switchboard/runtime.py) | A background event loop so Streamlit's re-runs don't kill async connections |
| [`app.py`](app.py) | The Streamlit UI: server cards, transcript, approval form, live progress |
| [`servers.json`](servers.json) | The example configuration (five servers, three on by default) |
| [`tests/`](tests/) | Offline tests: real MCP servers (in-process, stdio subprocesses, HTTP), scripted fake Claude |

## Run it

```bash
cd mcp-switchboard
uv sync
cp .env.example .env        # then put your ANTHROPIC_API_KEY in it
uv run streamlit run app.py
```

Things to try (also offered as buttons in an empty chat):

- *What time is it in Lagos, London and Tokyo right now?* runs `calc` tools, no approval needed.
- *List the sandbox and summarise welcome.md.* runs read-only `files` tools.
- *Look up the Model Context Protocol on Wikipedia and save a 5-line summary to notes/mcp.md.* runs `web` searches freely, then **stops at `write_file`** for your approval.

To try the Streamable HTTP transport, start the web server as a web service in a second terminal, then switch on `web-http` in the sidebar:

```bash
uv run python -m example_servers.web_reader --http
```

Tests (no API key, no paid calls; one test serves an MCP server over local HTTP):

```bash
uv run pytest -q
```

## Add your own server

Three ways, all ending in the same `ServerSpec`:

1. **Sidebar → ➕ Add a server.** Pick stdio (type the command line) or Streamable HTTP (paste the URL). Env vars / headers go in as `KEY=value` lines.
2. **Sidebar → { } Edit as JSON**, paste, *Apply*.
3. **Edit `servers.json`** and restart. The format is Claude Desktop's, so entries from a server's README paste straight in:

```json
{
  "mcpServers": {
    "my-stdio-server": {
      "command": "npx",
      "args": ["-y", "@scope/some-mcp-server", "${PROJECT_DIR}/data"],
      "env": {"SOME_TOKEN": "${SOME_TOKEN}"},
      "approval": "writes"
    },
    "my-http-server": {
      "url": "https://example.com/mcp",
      "headers": {"Authorization": "Bearer ${MY_SERVER_TOKEN}"},
      "approval": "always",
      "enabled": false
    }
  }
}
```

- `${NAME}` is filled from the environment / `.env` when connecting, so secrets stay out of the file. If one is missing, the sidebar says which. `${PYTHON}` (this virtualenv's Python) and `${PROJECT_DIR}` are built in.
- `approval` is `writes` (default: ask unless the server marks the tool read-only), `always`, or `never`. You can change it per server in the sidebar, and tick *Always allow this tool* in the approval form.
- **💾 Save config** writes the current servers (and their on/off state) back to `servers.json`.

To write a server in Python, copy [`calc_clock.py`](example_servers/calc_clock.py): `MCPServer("name")`, decorate typed functions with `@server.tool(annotations=...)`, end with `server.run()`. Raise `ToolError` for failures the model should read.

## What I learned

- **The client is where multi-server gets hard.** Each server is simple. The switchboard's work is the seams between them: unique names (`server__tool`), routing a call back to the right connection, merging every server's `instructions` into one system prompt, and staying valid when one is switched off mid-chat.
- **Async context managers want to be closed by the task that opened them.** MCP clients sit on anyio task groups. Open a connection in one Streamlit run and close it in another, and anyio refuses (a cancel scope must be exited by the task that entered it). A "keeper" task per connection, which enters `async with Client(...)` and then just waits for a stop signal, solves it in ten lines. A background event loop thread gives those tasks somewhere to live between re-runs.
- **"Pause for approval" is just *return*.** I first imagined threads blocking on a button click. The clean design keeps all state in the message list: an assistant turn with `tool_use` blocks and no results *is* the paused state, and calling `run_turn()` again resumes it. That's also why it's easy to test.
- **Tool annotations are hints, and MCP's defaults are pessimistic.** A tool with no annotations counts as "may write, may destroy". So unknown third-party tools ask first, and only tools whose server says `read_only_hint=True` run freely. A malicious server could lie, which is why the per-server `always` policy exists.
- **The API has opinions about history.** Every `tool_use` must be answered by a `tool_result` in the next message, all results for one turn go in *one* message, and a request whose history contains tool blocks must define at least one tool. Hence `close_dangling()`, the single results message, and the always-present `switchboard__list_servers` tool.
- **Protect against the model, not just the user.** A web page can tell the model to fetch `http://192.168.1.1/admin`, and a file can tell it to read `../../.ssh/id_rsa`. The web server refuses private addresses, and the file server resolves symlinks before checking that a path is inside the sandbox.
- **stdio servers don't inherit your environment.** The SDK launches them with a minimal, safe environment (PATH, HOME, ...). Anything else a server needs must go in its `env` block, which is why `${VAR}` expansion exists.
- **Small surprises from the wire:** Wikipedia returns 403 to user agents without a contact URL; MCP hides the text of plain exceptions raised in tools, so tools raise `ToolError` for messages the model should see; and `anthropic` / `mcp` 2.x are built on `httpx2`, not `httpx`.

## Limits

- Connections live per browser session, in that session's background thread. Two tabs = two sets of server processes.
- Only tools are used. MCP resources, prompts, sampling and elicitation are not wired into the chat.
- The HTML-to-text in `web_reader` is deliberately crude. JavaScript-rendered pages come back thin.
