"""Two interchangeable brains behind one tiny interface.

Every agent only ever calls

    brain.json(system=..., user=..., schema=..., effort=...) -> dict

and gets back a dict that matches `schema`. That single method is the whole
contract, so the parser and advisor don't know (or care) whether the answer
came from Claude in the cloud or a small model on your laptop.

  ClaudeBrain   Anthropic Messages API with *structured outputs*: the server
                constrains decoding to the schema, so the JSON always parses.
                Also exposes `create(...)` for the web-search scout, which is
                Claude-only (it uses Claude's server-side web_search tool).
  OllamaBrain   Ollama's local /api/chat with `format=<schema>`. Ollama also
                constrains output to the schema, but small models can still
                stop early or wrap JSON in prose, so we parse defensively and
                retry once with the error message.

Both take their transport as an argument (an `anthropic.Anthropic` client, or
an `http_post` function), so the tests swap in fakes and never touch a network.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable

from .config import SETTINGS


class BrainError(RuntimeError):
    """The model couldn't produce a usable answer (refusal, bad JSON, offline)."""


# ═════════════════════════════════ Claude ═══════════════════════════════════
class ClaudeBrain:
    name = "claude"
    can_search_web = True  # Claude has a server-side web search tool

    def __init__(self, client: Any = None, model: str | None = None):
        if client is None:
            import anthropic  # lazy: tests and Ollama-only runs never need it configured
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.client = client
        self.model = model or SETTINGS.claude_model
        self.label = f"Claude ({self.model})"

    def create(self, *, system: str, messages: list, effort: str,
               tools: list | None = None, schema: dict | None = None) -> Any:
        """One raw Messages API call. Returns the SDK response object."""
        output_config: dict = {"effort": effort}
        if schema is not None:
            # Structured outputs: the final text block is guaranteed valid JSON.
            output_config["format"] = {"type": "json_schema", "schema": schema}
        kwargs: dict = dict(model=self.model, max_tokens=SETTINGS.claude_max_tokens,
                            system=system, messages=messages, output_config=output_config)
        if tools:
            kwargs["tools"] = tools
        response = self.client.messages.create(**kwargs)
        # A refusal still comes back as HTTP 200, so check before trusting content.
        if response.stop_reason == "refusal":
            raise BrainError("Claude declined this request.")
        return response

    def json(self, *, system: str, user: str, schema: dict, effort: str) -> dict:
        response = self.create(system=system, effort=effort, schema=schema,
                               messages=[{"role": "user", "content": user}])
        if response.stop_reason == "max_tokens":
            raise BrainError("Claude ran out of tokens before finishing the JSON.")
        return json.loads(text_of(response))


def text_of(response: Any) -> str:
    """Join the text blocks of a Claude response (skips thinking/tool blocks)."""
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")


# ═════════════════════════════════ Ollama ═══════════════════════════════════
class OllamaBrain:
    name = "ollama"
    can_search_web = False  # a local model has no search tool; we pair it with OSM

    def __init__(self, model: str | None = None, host: str | None = None,
                 think: bool | None = None, http_post: Callable | None = None):
        self.model = model or SETTINGS.ollama_model
        self.host = (host or SETTINGS.ollama_host).rstrip("/")
        self.think = SETTINGS.ollama_think if think is None else think
        self.http_post = http_post or _post_json
        self.label = f"Ollama ({self.model}, local)"

    def json(self, *, system: str, user: str, schema: dict, effort: str) -> dict:
        # `effort` is Claude's dial. Locally the nearest thing is thinking on/off,
        # which is a per-brain setting, so `effort` is accepted and ignored here.
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        last_error = ""
        for _ in range(SETTINGS.json_retries + 1):
            reply = self._chat(messages, schema)
            try:
                return loads_lenient(reply)
            except ValueError as err:
                last_error = str(err)
                # Show the model its own broken output and ask again: cheap and
                # usually enough for a small model to fix a truncated object.
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": "That was not valid JSON "
                              f"({err}). Reply with ONLY the JSON object."}]
        raise BrainError(f"{self.model} did not return valid JSON: {last_error}")

    def _chat(self, messages: list, schema: dict) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "format": schema,          # Ollama structured outputs: JSON schema → grammar
            "stream": False,           # one JSON response instead of a token stream
            "think": self.think,       # reasoning models put their thoughts in .thinking
            "options": {"temperature": 0, "num_ctx": SETTINGS.ollama_num_ctx},
        }
        try:
            data = self.http_post(f"{self.host}/api/chat", body)
        except OSError as err:  # URLError, connection refused, timeouts
            raise BrainError(f"Can't reach Ollama at {self.host} ({err}). Is "
                             "`ollama serve` running?") from err
        if "error" in data:  # e.g. "model 'qwen3:4b' not found, try pulling it first"
            raise BrainError(f"Ollama: {data['error']}")
        return data.get("message", {}).get("content", "")

    def installed_models(self) -> list[str]:
        """Names of locally pulled models (for the UI). [] if Ollama is down."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as r:
                return [m["name"] for m in json.loads(r.read())["models"]]
        except (OSError, ValueError, KeyError):
            return []


def loads_lenient(text: str) -> dict:
    """Parse JSON that may be wrapped in ```fences``` or a stray <think> block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object found")
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError as err:
        raise ValueError(err.msg) from err
    if not isinstance(value, dict):
        raise ValueError("top-level JSON is not an object")
    return value


def _post_json(url: str, body: dict) -> dict:
    """Stdlib POST → JSON. One endpoint doesn't justify an HTTP dependency."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=SETTINGS.ollama_timeout_s) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as err:  # Ollama puts the reason in the body
        try:
            return json.loads(err.read())
        except ValueError:
            return {"error": f"HTTP {err.code}"}


def make_brain(kind: str, **kwargs) -> ClaudeBrain | OllamaBrain:
    """'claude' or 'ollama' → a ready brain. The UI's only entry point."""
    if kind == "claude":
        return ClaudeBrain(**kwargs)
    if kind == "ollama":
        return OllamaBrain(**kwargs)
    raise ValueError(f"unknown brain {kind!r}")
