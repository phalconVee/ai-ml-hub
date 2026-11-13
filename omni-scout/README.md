# omni-scout: one question, many platforms, one cited answer

A multi-agent researcher. You ask a question; a **planner** decides which corners of the internet are worth searching; one **scout** agent per platform (web, news, video, Reddit, Hacker News, Wikipedia) searches *in parallel* through an **MCP server**; a **synthesizer** merges everything into a report that says where the platforms agree, where they don't, and cites every claim.

No agent framework: the planner, the agent loop and the merge are each a page of plain Python, heavily commented so they can be read line by line. The search side needs **no API keys**. Only Claude does.

## The idea in one paragraph

Different platforms know different things. The vendor's site says what a product *should* do, the news says *when* things changed, Reddit says what *actually* happens to people using it, Hacker News says what engineers *think* of it. A single search agent blends these into mush. So give each platform its own scout with **only that platform's tools**, let them work at the same time, make each one hand back structured, URL-cited findings, and then have one stronger pass do the cross-checking. A claim that turns up on three platforms is stronger than one that turns up on one, and a fact that the official docs and the Reddit threads disagree on is the most interesting thing in the report.

## Architecture

```
                       ┌──────────────┐
  question ───────────▶│   planner    │  1 call, structured output (ResearchPlan)
                       └──────┬───────┘
          picks ≤4 platforms, writes a brief + queries for each
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
     ┌────────────┐    ┌────────────┐    ┌────────────┐
     │ web scout  │    │ news scout │    │ hn scout   │ … run concurrently (asyncio.gather)
     │ tool loop  │    │ tool loop  │    │ tool loop  │   each sees ONLY its platform's tools
     └─────┬──────┘    └─────┬──────┘    └─────┬──────┘
           │  tool_use → MCP call_tool (one shared connection, stdio)
           ▼                 ▼                 ▼
     ┌──────────────────────────────────────────────────┐
     │  MCP server  (scout/mcp_server.py, own process)  │
     │  search_web · search_news · search_videos ·      │
     │  search_reddit · search_hackernews ·             │
     │  search_wikipedia · read_page                    │
     └──────────────────────────────────────────────────┘
           │ each scout ends by calling submit_findings(ScoutReport)
           ▼
     ┌─────────────────────────┐
     │  merge: dedupe URLs     │  same article found by 2 scouts = 1 source, tagged with both
     │  number sources [1..n]  │
     └───────────┬─────────────┘
                 ▼
          ┌──────────────┐
          │ synthesizer  │  streamed, high effort → markdown report
          └──────┬───────┘   + a sources list written by code, never by the model
                 ▼
          Streamlit UI (app.py)
```

## Code map: read in this order

| File | Role |
|------|------|
| [`scout/config.py`](scout/config.py) | Every knob: platforms → tools, models, effort per role, budgets |
| [`scout/schemas.py`](scout/schemas.py) | The objects agents hand each other: `ResearchPlan`, `Finding`, `ScoutReport` |
| [`scout/sources.py`](scout/sources.py) | Raw platform access, no AI. Every source returns the same `Hit` shape |
| [`scout/mcp_server.py`](scout/mcp_server.py) | Wraps the sources as MCP tools. Docstrings are the prompts the scouts read |
| [`scout/toolbox.py`](scout/toolbox.py) | MCP client: launches the server, converts tools to Claude's format, runs calls |
| [`scout/llm.py`](scout/llm.py) | The only file that talks to Claude: structured / tool-step / streamed calls |
| [`scout/agents.py`](scout/agents.py) | **The core:** planner, the scout's agent loop, the merge + synthesizer |
| [`scout/pipeline.py`](scout/pipeline.py) | Plan → scouts in parallel → synthesize, in ~30 lines |
| [`app.py`](app.py) | Streamlit UI with live per-scout progress and a streamed report |
| [`scout/__main__.py`](scout/__main__.py) | Same pipeline from the command line |
| [`tests/`](tests/) | The whole crew, offline: fake Claude + real MCP server with fake sources |

## Run it

Setup (once). Needs [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

```bash
cp .env.example .env
```

Then put your `ANTHROPIC_API_KEY` in `.env` (or run `ant auth login` instead).

Run the tests (offline, no key, a couple of seconds):

```bash
uv run pytest
```

Launch the UI:

```bash
uv run streamlit run app.py
```

Or from the terminal:

```bash
uv run python -m scout "How are developers using MCP servers in production, and what breaks?"
```

The MCP server also works on its own, so you can plug it into Claude Desktop, Claude Code or the MCP Inspector:

```bash
uv run python -m scout.mcp_server
```

### Cost

Each question is 1 planner call, 2–6 tool-loop turns for each of ≤4 scouts, and 1 synthesizer call, all on Claude Opus 5.5 by default. Scouts run at `low` effort because their work is mechanical; the synthesizer runs at `high` because that's where the reasoning is. To spend less, set `OMNI_SCOUT_SCOUT_MODEL=claude-haiku-5-5` (or lower `OMNI_SCOUT_MAX_PLATFORMS`) in `.env`.

## Design choices

| Choice | Why |
|--------|-----|
| Own MCP server over free sources | No search-API keys to get started. Swapping in a paid source (e.g. a scraping MCP server) changes the server, not the agents. |
| One scout per platform, tools scoped per scout | Short, focused prompts; a scout can't drift onto another platform; runs in parallel. |
| `submit_findings` as a tool | The scout's "done" signal *is* its structured output. Invalid reports get the validation error back, and the scout retries. |
| Manual agent loop (not the SDK's tool runner) | The loop is the thing being learned; it's ~40 lines and stops on submission, not on silence. |
| Sources list built in code | The model only cites numbers; URLs come from tool output, so links can't be invented. |
| Tool errors are returned, not raised | A rate-limited search is information ("try another query"), not a crash. One failed scout doesn't sink the others. |
| Server-side refusal fallback on every call | If a request is declined, the API retries on a fallback model in the same call. |

## What was verified, and what wasn't

- ✅ `uv run pytest`: 6 tests. The full pipeline runs offline: planner validation (unknown platforms dropped), tool scoping, MCP round-trip, invalid-submission retry, cross-scout URL merging, report assembly, and the scout that never submits.
- ✅ Every MCP tool called live through the real stdio subprocess against the free sources (web, news, Reddit, HN, Wikipedia, `read_page`, and a bad URL that came back as a readable error).
- ✅ Requests checked on the wire: the real SDK pointed at a mock HTTP transport, confirming the beta header, `fallbacks`, effort, the JSON schema and streaming.
- ✅ Streamlit UI driven headlessly (AppTest), with a fake Claude: question → status panels → streamed report → findings tables, with no exceptions.
- ⚠️ DuckDuckGo's keyless video endpoint gets throttled after a burst of calls, so `videos()` falls back to `site:youtube.com` web results. Heavy back-to-back use can throttle the web search too.
- ❌ Not yet run against the real Claude API. Run one real question to see how good the reports are.

## What I learned

- **MCP is a boundary, not a feature.** The agents never import the search code. They see tool names, descriptions and JSON schemas over a pipe. That made testing trivial (connect the client to the server object in-process) and makes the server reusable by any MCP client.
- **A tool's docstring is a prompt.** "Use short queries" on the Hacker News tool, or "you cannot watch the video" on the video tool, changes how the scout behaves more than anything in its system prompt.
- **mcp 2.x hides tool crashes on purpose.** A plain exception reaches the client as "Error executing tool X" with the message withheld. Raising `ToolError` marks a failure as expected and passes the message through, which the model needs in order to recover.
- **Structured output can be the stop condition.** Giving the scout a `submit_findings` tool whose schema is the report turns "when is the agent done?" and "what shape is its answer?" into one question.
- **Merging is where multi-agent pays off.** Numbering sources globally and deduplicating URLs across scouts turns "two scouts found the same article" from noise into evidence.
- **Put effort where the thinking is.** Scouts search and summarise (low effort); one synthesizer cross-checks (high effort). Same model, very different token bills.
- **Read the installed SDK, not your memory.** `FastMCP` is now `MCPServer`, the Anthropic SDK now uses `httpx2`, and tool results use `is_error` / `input_schema`. All of these were caught by inspecting the installed packages before writing code.
