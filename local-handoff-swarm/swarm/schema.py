"""Turn a plain Python function into a tool description an LLM can read.

Both Ollama and Claude want each tool described as JSON Schema:

    {"name": "web_search",
     "description": "Search the web ...",
     "parameters": {"type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]}}

Writing those by hand is tedious and drifts out of sync with the code, so we
derive them from the function itself: its name, its docstring and its type
hints. ~30 lines, no pydantic, no decorators.
"""

from __future__ import annotations

import inspect
import typing
from typing import Any, Callable

# Python type → JSON Schema type. Anything unknown is described as a string,
# which is what small models are best at producing anyway.
_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean",
               list: "array", dict: "object"}

# Parameters with this name are *not* shown to the model. The loop fills them
# in with the shared context dict (see loop.py). This is how a tool can read
# or write state that lives outside the chat transcript.
CONTEXT_PARAM = "context"


def function_to_schema(fn: Callable) -> dict[str, Any]:
    """Describe `fn` as a tool: {"name", "description", "parameters"}."""
    hints = typing.get_type_hints(fn)          # resolves "str" annotations to real types
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in inspect.signature(fn).parameters.items():
        if name == CONTEXT_PARAM:
            continue                            # injected by the loop, hidden from the model
        py_type = hints.get(name, str)
        properties[name] = {"type": _JSON_TYPES.get(py_type, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(name)               # no default → the model must supply it

    return {
        "name": fn.__name__,
        # The docstring IS the tool description. Write it for the model:
        # say what the tool does and when to use it.
        "description": inspect.getdoc(fn) or fn.__name__,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }
