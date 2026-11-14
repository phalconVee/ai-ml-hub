# AI/ML Hub

A working lab of **agentic AI projects** I'm building to learn and practise: multi-agent systems, tool-using agents, MCP, agent memory, and the RAG pipelines that feed them.

Every project is built **from scratch, in my own code**, with heavy comments, so each one can be read line by line and explained. Nothing here is a copy-paste fork.

**Where this is going:** this repo becomes the source behind a **portfolio**. Each finished project gets a short write-up, a demo (most use a Streamlit UI), and a link from the portfolio site back to its folder here.

## How the repo is organised

- **One project per top-level folder.** Each folder is self-contained, with its own `README.md`, dependencies (`pyproject.toml` / `uv`), `.env.example`, and app entry point.
- **Every project README covers:** what it does, the architecture, a code map to read in order, how to run it, and what I learned.
- **Secrets never get committed.** Keys go in a local `.env`, and each project ships a `.env.example`.
- **Way of working:** documented code first, then a line-by-line walkthrough.

## Projects

| Project | What it is | Stack | Status |
|---|---|---|---|
| [build-reasoning-model](build-reasoning-model/) | A small reasoning model trained with GRPO, written from scratch | PyTorch, GRPO | ✅ Done |

## Original projects

Net-new ideas of my own, not drawn from any reference list.

| Project | Idea | Stack | Status |
|---|---|---|---|
| [sightline](sightline/) | Live webcam vision: classify and locate objects, with confidence thresholds that decide between auto-accept and human review | TypeSafe Jev decision API, Streamlit | 🧪 |

## Roadmap: Streamlit projects

The next wave is inspired by the Streamlit-based projects in [patchy631/ai-engineering-hub](https://github.com/patchy631/ai-engineering-hub). I rebuild each idea independently, under my own name, in its own folder. Groups are ordered by how agentic they are.

Status key: ⬜ planned · 🚧 in progress · 🧪 built, offline tests pass, not yet run with live API keys · ✅ done

### 1. Multi-agent systems and workflows
| Project | Idea | Status |
|---|---|---|
| [deepdive-research-crew](deepdive-research-crew/) | Multi-agent deep web researcher, wired together over MCP | 🧪 |
| [omni-scout](omni-scout/) | Researcher that searches across many platforms at once | 🧪 |
| [trendwatch-tube](trendwatch-tube/) | Agents that spot and explain YouTube trends | 🧪 |
| [fallback-sage](fallback-sage/) | Answers from your docs first, falls back to the web | 🧪 |
| [groundtruth-agent](groundtruth-agent/) | Agent with a custom document-retrieval tool | 🧪 |
| [newsroom-crew](newsroom-crew/) | Research-and-write news desk | 🧪 |
| [paper-lab-analyst](paper-lab-analyst/) | Reads paper figures with OCR, then agents analyse them | 🧪 |
| [brand-radar](brand-radar/) | Watches the web for mentions of a brand | 🧪 |
| [context-forge](context-forge/) | Multi-source research assistant built on agent flows | 🧪 |
| [skyfare-scout](skyfare-scout/) | Multi-agent flight search | 🧪 |
| [stayfinder-crew](stayfinder-crew/) | Local multi-agent hotel finder | 🧪 |
| [counsel-copilot](counsel-copilot/) | Legal-document Q&A crew | 🧪 |
| [local-handoff-swarm](local-handoff-swarm/) | Lightweight agent handoffs on a local model | 🧪 |
| [podcast-crew](podcast-crew/) | Turns web content into a two-voice podcast, via agents | 🧪 |
| [sourcebook](sourcebook/) | Answers grounded in your documents, with citations and audio | 🧪 |

### 2. MCP agents
| Project | Idea | Status |
|---|---|---|
| soundcheck-mcp | Audio analysis toolkit served over MCP | ⬜ |
| webpilot-mcp | Chat UI that drives a browser-automation MCP server | ⬜ |
| mcp-switchboard | One chat UI over several configurable MCP servers | ⬜ |

### 3. Agent memory
| Project | Idea | Status |
|---|---|---|
| recall-companion | Assistant with long-term, human-like memory | ⬜ |
| vault-memory-agent | Agent with vector-database memory and RAG | ⬜ |
| memoir-avatar | Knowledge-graph memory behind a talking avatar | ⬜ |

### 4. Tool-using agents and agentic RAG
| Project | Idea | Status |
|---|---|---|
| webcrawl-sage | RAG agent with live web search | ⬜ |
| query-dispatcher | Routes each question to vector search or SQL | ⬜ |
| self-check-rag | RAG that grades and corrects its own retrieval | ⬜ |
| site2api | Turns any website into a structured API | ⬜ |
| article-to-airwaves | Article to podcast script and audio | ⬜ |

### 5. RAG applications
| Project | Idea | Status |
|---|---|---|
| listen-and-ask | Ask questions about audio recordings | ⬜ |
| repo-whisperer | Chat with a GitHub repository | ⬜ |
| docs-dialogue | Chat with your documents | ⬜ |
| sub-15ms-rag | RAG stack tuned for retrieval under 15 ms | ⬜ |
| quickdraw-rag | Fast chat-with-docs stack | ⬜ |
| repo-digest-rag | Flattens a repo to markdown, then runs RAG over it | ⬜ |
| llama-four-lens | RAG on Llama 4 | ⬜ |
| local-bert-rag | Fully local RAG with ModernBERT embeddings | ⬜ |
| spreadsheet-sage | RAG over spreadsheets and other complex documents | ⬜ |
| confidence-rag | RAG answers that come with a trust score | ⬜ |
| dense-doc-pipeline | Pipeline for documents full of tables and figures | ⬜ |
| frame-talk | Chat with uploaded videos | ⬜ |
| pixel-page-rag | Multimodal RAG over web pages captured as images | ⬜ |
| janus-vision-rag | Multimodal RAG with Janus-Pro | ⬜ |
| polyglot-minutes | Detects the meeting's language, then writes English notes | ⬜ |

### 6. Evaluation and model comparison
| Project | Idea | Status |
|---|---|---|
| code-arena | Pick models and compare their code generation | ⬜ |
| reasoning-arena | Reasoning evals scored with G-Eval | ⬜ |
| rag-model-duel | Side-by-side RAG model comparison, with tracing | ⬜ |
| frontier-coder-faceoff | Four-way code-generation comparison | ⬜ |
| rag-eval-pipeline | End-to-end RAG evaluation and observability | ⬜ |
| retrieval-duel | Two reasoning models compared on RAG | ⬜ |
| coder-scorecard | Code-generation evals using DeepEval | ⬜ |
| open-vs-closed-coder | Open-weight vs closed-model coding evals | ⬜ |

### 7. Local model UIs and vision
| Project | Idea | Status |
|---|---|---|
| think-aloud-chat | Local chat UI that shows the model's reasoning | ⬜ |
| open-mind-chat | Local reasoning UI for an open-weight model | ⬜ |
| pocket-thinker | Small local reasoning model in a chat UI | ⬜ |
| glyph-extractor | Local OCR and structured data extraction | ⬜ |
| page-reader | Local OCR app | ⬜ |
| equation-snap | Photo of an equation to LaTeX | ⬜ |
| vision-scribe | OCR with a vision-language model | ⬜ |
| dream-and-describe | Local image generation and visual Q&A | ⬜ |

## Credits

Project ideas in the roadmap are inspired by [patchy631/ai-engineering-hub](https://github.com/patchy631/ai-engineering-hub). All names and implementations here are my own.
