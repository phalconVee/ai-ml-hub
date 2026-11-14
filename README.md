# AI/ML Hub

A working lab of **agentic AI projects** I'm building to learn and practise: multi-agent systems, tool-using agents, MCP, agent memory, and the RAG pipelines that feed them.

Every project is built **from scratch, in my own code**, with heavy comments, so each one can be read line by line and explained. Nothing here is a copy-paste fork.

**Where this is going:** this repo is the source behind my **portfolio**. Each project gets a short write-up, a demo (most have a Streamlit UI), and a link from the portfolio back to its folder here.

## How the repo is organised

- **One project per top-level folder.** Each folder is self-contained, with its own `README.md`, dependencies (`pyproject.toml` / `uv`), `.env.example`, and app entry point.
- **Every project README covers:** what it does, the architecture, a code map to read in order, how to run it, and what I learned.
- **Secrets never get committed.** Keys go in a local `.env`, and each project ships a `.env.example`.
- **Way of working:** documented code first, then a line-by-line walkthrough.

## Projects

### Multi-agent systems and workflows
| Project | What it does | Built with |
|---|---|---|
| [deepdive-research-crew](deepdive-research-crew/) | Planner, parallel scouts, analyst and writer research a question in depth | Claude, MCP |
| [omni-scout](omni-scout/) | Scouts search several platforms in parallel, then one report merges the findings | Claude, MCP |
| [trendwatch-tube](trendwatch-tube/) | Agents read recent videos from chosen YouTube channels and explain emerging trends | Claude, YouTube feeds |
| [fallback-sage](fallback-sage/) | Answers from your documents first, and searches the web only when they fall short | Claude, BM25, web search |
| [groundtruth-agent](groundtruth-agent/) | Agent answers questions about complex PDFs (tables, figures) with citations | Claude tool use, citations |
| [newsroom-crew](newsroom-crew/) | Reporter, writer and editor produce a sourced article, with a citation audit | Claude, web search |
| [paper-lab-analyst](paper-lab-analyst/) | Reads a scientific paper, figures included, and writes a structured analysis | Claude vision, PDF extraction |
| [brand-radar](brand-radar/) | Collects brand mentions across sources and reports sentiment and themes | Claude |
| [context-forge](context-forge/) | Research assistant that picks the best context from docs, web, papers and memory | Claude, arXiv, web search |
| [skyfare-scout](skyfare-scout/) | Turns a trip description into flight searches and ranked recommendations | Claude |
| [stayfinder-crew](stayfinder-crew/) | Hotel finder crew that runs on Claude or fully locally | Claude or Ollama, OpenStreetMap |
| [counsel-copilot](counsel-copilot/) | Plain-English, cited answers about contracts and other legal PDFs | Claude, RAG, web search |
| [local-handoff-swarm](local-handoff-swarm/) | Agents hand the conversation to each other mid-task, via a hand-written loop | Ollama, Claude |
| [podcast-crew](podcast-crew/) | Turns a web page or text into a two-host podcast episode | Claude, local TTS |
| [sourcebook](sourcebook/) | Research notebook: cited answers from your sources, plus an audio overview | Claude citations, hybrid search, local TTS |

### MCP agents
| Project | What it does | Built with |
|---|---|---|
| [soundcheck-mcp](soundcheck-mcp/) | Audio-analysis tools served over MCP, with a chat app to question a recording | MCP, local Whisper, Claude |
| [webpilot-mcp](webpilot-mcp/) | Claude drives a real browser through an MCP server, within guard rails | MCP, Playwright, Claude |
| [mcp-switchboard](mcp-switchboard/) | One chat UI over several MCP servers, with approval before anything writes | MCP (stdio and HTTP), Claude |

### Agent memory
| Project | What it does | Built with |
|---|---|---|
| [recall-companion](recall-companion/) | Assistant that remembers facts across sessions and updates them when they change | Claude, SQLite |
| [vault-memory-agent](vault-memory-agent/) | Agent that stores and recalls documents, threads and its own notes | Claude, SQLite + sqlite-vec |
| [memoir-avatar](memoir-avatar/) | Talking avatar backed by a knowledge graph of what you've told it | Claude, networkx, local TTS |

### Vision
| Project | What it does | Built with |
|---|---|---|
| [sightline](sightline/) | Classifies and locates objects from your webcam; confidence thresholds decide what is auto-accepted and what goes to human review | TypeSafe Jev |

### Model training
| Project | What it does | Built with |
|---|---|---|
| [build-reasoning-model](build-reasoning-model/) | A small reasoning model trained with GRPO, with the algorithm written from scratch | PyTorch, GRPO |

## Running a project

```bash
cd <project-folder>
cp .env.example .env   # then add your keys
uv run streamlit run app.py
```

Each project's own README has the exact steps and the keys it needs (`build-reasoning-model` is a training project and has its own instructions).

## Acknowledgements

Some project ideas were sparked by [patchy631/ai-engineering-hub](https://github.com/patchy631/ai-engineering-hub).
