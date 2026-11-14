# Local Handoff Swarm

A small multi-agent system built around **handoffs**. A **Triage** agent reads your message and passes the conversation to a specialist (**Researcher**, **Summarizer** or **Writer**). Specialists can pass it on to each other partway through a task, so *"research X and write a LinkedIn post about it"* goes Triage → Researcher → Writer in one turn. A Streamlit chat shows each handoff as it happens and highlights whichever agent is active.

It runs **entirely on your machine** through [Ollama](https://ollama.com) and needs no API key. Web search uses keyless DuckDuckGo (`ddgs`). Claude is an optional second backend that you can switch to from the sidebar.

There's no agent framework here. The orchestration loop is ~80 lines in [`swarm/loop.py`](swarm/loop.py), and every file is commented so you can read it line by line.

## The idea in one paragraph

An agent is just **a name, a system prompt and a list of Python functions**. A handoff is **a function that returns another agent**: `def transfer_to_writer(): return writer`. To the model it's an ordinary tool. The loop calls the active agent's model; if the model calls a tool, the loop runs it. A string result is returned to the model as the tool result. An `Agent` result **becomes the active agent**, so the next model call uses *its* system prompt and *its* tools over the same shared transcript. The loop ends when an agent replies in plain text. The active agent stays active after the turn, so a follow-up ("make it shorter") goes straight to the Writer.

## Architecture

```
                      ┌───────────────────── Streamlit (app.py) ─────────────────────┐
   your message ─────►│ SwarmState{active, messages, context} in session_state        │
                      │ draws each Event live: 🔀 handoff · 🔧 tool call · answer      │
                      └───────────────────────────────┬──────────────────────────────┘
                                                      │ run_turn(state, backend, text)
                                                      ▼
            ┌───────────────────────── loop.py ─────────────────────────┐
            │  repeat ≤ SWARM_MAX_STEPS:                                 │
            │    reply = backend.complete(active.instructions,          │
            │                             shared messages,               │
            │                             schemas(active.functions))     │
            │    no tool calls?          → yield message, stop           │
            │    tool returns str        → append tool result            │
            │    tool returns an Agent   → state.active = it  (HANDOFF)  │
            └───────────────┬───────────────────────────┬───────────────┘
                            │                           │
            backends.py     ▼                           ▼   team.py + tools.py
   ┌────────────────────────────────────┐   ┌────────────────────────────────────────┐
   │ OllamaBackend  (default, local)    │   │ 🧭 Triage ──► 🔎 Researcher (web_search)│
   │ ClaudeBackend  (optional, API key) │   │          ──► 🗜️ Summarizer (word_count) │
   │ neutral transcript ⇄ each API      │   │          ──► ✍️ Writer     (word_count) │
   └────────────────────────────────────┘   │ specialists ⇄ each other, ◄── back to   │
                                            │ Triage; sources saved in shared context │
                                            └────────────────────────────────────────┘
```

## Code map: read in this order

| File | Role |
|------|------|
| [`swarm/config.py`](swarm/config.py) | Every setting, read from env / `.env`: backend, models, step cap |
| [`swarm/agent.py`](swarm/agent.py) | `Agent` = name + instructions + functions. Explains the handoff trick |
| [`swarm/schema.py`](swarm/schema.py) | Python function → JSON-Schema tool description (name, docstring, type hints) |
| [`swarm/backends.py`](swarm/backends.py) | One `complete()` interface; Ollama and Claude translators to and from a neutral transcript |
| [`swarm/loop.py`](swarm/loop.py) | **The core:** the orchestration loop, tool execution, the handoff, events |
| [`swarm/tools.py`](swarm/tools.py) | `web_search` (keyless, records sources in shared context) and `word_count` |
| [`swarm/team.py`](swarm/team.py) | The four agents, their prompts, and the `transfer_to_*` handoff functions |
| [`app.py`](app.py) | Streamlit: backend picker, live handoff trail, team roster, sources |
| [`tests/test_loop.py`](tests/test_loop.py) | The full Triage → Researcher → Writer flow with a scripted fake LLM |
| [`tests/test_backends.py`](tests/test_backends.py) | Wire formats for Ollama and Claude, against fake SDK clients |
| [`tests/test_app.py`](tests/test_app.py) | The Streamlit page run headless, end to end |

## Run it

Needs [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/download).

**1. Set up Ollama (once).** Install it, make sure it's running (the desktop app starts the server; `ollama serve` also works), then pull a model that supports **tool calling**:

```bash
ollama pull qwen3:4b
```

`qwen3:4b` (~2.5 GB) is the default. Smaller models like `llama3.2:3b` also work but route less reliably. If you have the RAM, `qwen3:8b` is noticeably better at handoffs. To use a different model, set `OLLAMA_MODEL` in `.env` or type its name in the sidebar.

**2. Install the project:**

```bash
uv sync
```

**3. (Optional) settings.** None are required. To change the model or use Claude:

```bash
cp .env.example .env
```

**4. Run the tests** (offline, no Ollama or key needed, about a second):

```bash
uv run pytest
```

**5. Start the app:**

```bash
uv run streamlit run app.py
```

Things to try:
- *"Research the latest Ollama release and write a three-line LinkedIn post about it."* → 🧭 → 🔎 → ✍️
- Paste a long article with *"summarize this"* → 🧭 → 🗜️, then *"now turn it into a tweet"* → 🗜️ → ✍️
- Turn on **Start every message at Triage** in the sidebar to make Triage route every message, not just the first.

**Using Claude instead:** add `ANTHROPIC_API_KEY` to `.env` and pick **claude** in the sidebar (or set `SWARM_BACKEND=claude`). It defaults to `claude-opus-5-5` at `low` effort, since routing hops don't need deep thinking.

## Design choices, and why

| Choice | Why |
|--------|-----|
| A handoff is a tool that **returns an Agent** | The model needs no new concept: it just calls a tool. Only the loop knows a handoff is special, so the whole mechanism is one `isinstance` check. |
| One **shared transcript** for all agents | The Writer sees the Researcher's search results without any copying or summarizing. That's what makes a handoff mid-task work. |
| Each call uses the **active agent's** prompt and tools | Specialization is real: Triage *can't* search, because it has no `web_search` tool. Who may hand off to whom is simply which `transfer_*` functions an agent has. |
| A neutral transcript plus thin backend translators | The loop is backend-agnostic, and you can switch from Ollama to Claude mid-conversation. The differences (tool ids, `tool_result` blocks, alternating roles) stay inside `backends.py`. |
| `run_turn` is a **generator of Events** | The UI draws each handoff as it happens, and tests assert the exact sequence. |
| Tool errors become **text for the model** | Small local models sometimes invent tools or skip arguments. They read the error and recover; nothing crashes. |
| `SWARM_MAX_STEPS` cap | Two agents can bounce a task back and forth. The cap guarantees a turn ends. |
| Hidden `context` parameter | Tools can share state outside the transcript. `web_search` numbers sources once across agents and turns, and the sidebar lists them. |
| Thinking off for Ollama by default | qwen3-style models think by default, which is slow on a laptop and adds little to routing. Set `OLLAMA_THINK=true` to turn it on. |
| Settings prefixed `SWARM_` | Claude Code exports `CLAUDE_EFFORT` into the shells it starts, which silently overrode an unprefixed setting during development. |

## What was verified, and what wasn't

- ✅ **15 offline tests pass** (`uv run pytest`, ~1 s). They cover: tool schemas from signatures; the full Triage → Researcher → Writer handoff chain (events, the system prompt and tool list sent on every step, shared sources); every tool call answered in the transcript; follow-ups staying with the active agent; recovery from invented tools and bad arguments; the max-steps stop; search failure and retry; Ollama and Claude wire formats (merged `tool_result` turns, string arguments parsed, effort/fallback settings, refusals); the `think` flag retry; and the Streamlit page end to end with Streamlit's `AppTest`.
- ✅ **Live keyless web search** returned real results (`ddgs` 9.16.1). It is also intermittently throttled: one of the first runs failed, which is why `web_search` retries once and otherwise tells the model the search failed.
- ✅ **Live Ollama server** (0.31.1) reached through the real client. With no model pulled, the app showed the 404 and the `ollama pull` hint instead of crashing.
- ✅ The Streamlit server starts and answers its health check.
- ⬜ **Not yet run with a real local model.** No model was pulled on the build machine. The first `ollama pull qwen3:4b` and a real conversation is the true test of how reliably a small model follows the handoff prompts.
- ⬜ **Claude backend not run against the live API** (no key on the build machine). Its request shape is checked against the installed `anthropic` SDK (2.8.0) and the current API docs, but not on the wire.

## What I learned

- **A handoff is just a return value.** I expected multi-agent orchestration to need a framework. It turns out to be "swap the system prompt and tool list between two calls", and the cleanest place to signal the swap is a tool's return type.
- **Shared history is the hidden half of a handoff.** Changing the agent is easy. What makes it useful is that the new agent inherits everything said so far, including the previous agent's tool results.
- **Routing quality lives in the prompts and docstrings.** The model picks a `transfer_to_*` tool from its docstring alone. Making those docstrings say *when* to use each tool helped small models more than anything else.
- **Small models need guard rails, not trust.** Invented tool names, string-encoded arguments, ping-pong handoffs: each is one defensive line in the loop, and each would otherwise crash or hang the app.
- **Normalize at the edges.** Ollama and Claude disagree on tool ids, where tool results go, and whether roles must alternate. Keeping one neutral transcript and translating only inside the backends kept the loop small.

## Ideas for next steps

- Stream tokens from the model into the chat, not just events.
- Let a handoff carry a short "briefing" argument (`transfer_to_writer(brief=...)`) so the next agent gets a focused task statement.
- Give agents different models: a tiny fast one for Triage, a bigger one for the Writer.
- Add a `fetch_page` tool so the Researcher can read a whole article, not just snippets.
