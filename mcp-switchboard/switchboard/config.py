"""What a server *is*, before we connect to it: the config model + JSON load/save.

The file format deliberately matches Claude Desktop's `claude_desktop_config.json`
(`{"mcpServers": {name: {...}}}`), so you can paste servers from there or from
any server's README. We add three optional fields of our own:

    enabled      connect on startup?                        (default true)
    approval     "writes" | "always" | "never"              (default "writes")
    description  one line shown in the sidebar

A server is either
    stdio:  {"command": "...", "args": [...], "env": {...}, "cwd": "..."}
    http:   {"url": "http://host/mcp", "headers": {...}}       (Streamable HTTP)

String values may contain ${VAR} placeholders, expanded from the environment
(and .env) at connect time, so secrets stay out of servers.json:

    "headers": {"Authorization": "Bearer ${MY_SERVER_TOKEN}"}

Two built-in placeholders make the example servers portable:
    ${PYTHON}       the Python running the switchboard (its virtualenv included)
    ${PROJECT_DIR}  this project's folder
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from switchboard.settings import PROJECT_DIR

Approval = Literal["writes", "always", "never"]

# Server names become tool-name prefixes ("files__read_file"), and Claude's tool
# names must match ^[a-zA-Z0-9_-]+$, so names are restricted to the same set.
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,30}$")


class ServerSpec(BaseModel):
    name: str
    description: str = ""
    enabled: bool = True
    approval: Approval = "writes"
    # stdio transport
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    # Streamable HTTP transport
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "ServerSpec":
        if not NAME_RE.match(self.name) or "__" in self.name:
            # "__" is reserved as the server/tool separator in tool names.
            raise ValueError(f"Server name {self.name!r}: use letters, digits, '-' or single '_' (max 30).")
        if bool(self.command) == bool(self.url):
            raise ValueError(f"Server {self.name!r}: set exactly one of 'command' (stdio) or 'url' (HTTP).")
        return self

    @property
    def transport(self) -> str:
        return "stdio" if self.command else "http"

    def to_json(self) -> dict:
        """The entry as it appears under mcpServers (no name, no empty fields)."""
        return self.model_dump(exclude={"name"}, exclude_defaults=True) | {"enabled": self.enabled}


def expand(value: str) -> str:
    """Replace ${VAR} with its value. Unknown variables become "" (and the server will complain)."""
    builtins = {"PYTHON": sys.executable, "PROJECT_DIR": str(PROJECT_DIR)}
    return re.sub(r"\$\{(\w+)\}", lambda m: builtins.get(m[1], os.environ.get(m[1], "")), value)


def missing_vars(spec: ServerSpec) -> list[str]:
    """Placeholders this server needs that aren't set: shown in the UI instead of a cryptic failure."""
    blob = json.dumps(spec.model_dump())
    return sorted({v for v in re.findall(r"\$\{(\w+)\}", blob)
                   if v not in ("PYTHON", "PROJECT_DIR") and not os.environ.get(v)})


def parse_config(data: dict) -> list[ServerSpec]:
    """Accept {"mcpServers": {...}} (Claude Desktop style) or a bare {name: {...}} mapping."""
    servers = data.get("mcpServers", data)
    return [ServerSpec(name=name, **entry) for name, entry in servers.items()]


def load_config(path: Path) -> list[ServerSpec]:
    if not path.exists():
        return []
    return parse_config(json.loads(path.read_text(encoding="utf-8")))


def dump_config(specs: list[ServerSpec]) -> str:
    return json.dumps({"mcpServers": {s.name: s.to_json() for s in specs}}, indent=2) + "\n"


def save_config(path: Path, specs: list[ServerSpec]) -> None:
    path.write_text(dump_config(specs), encoding="utf-8")


def resolve_cwd(spec: ServerSpec) -> str:
    """stdio servers run from their `cwd`, relative paths being relative to the project."""
    if not spec.cwd:
        return str(PROJECT_DIR)
    p = Path(expand(spec.cwd))
    return str(p if p.is_absolute() else PROJECT_DIR / p)
