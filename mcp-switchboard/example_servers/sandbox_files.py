"""A filesystem MCP server that can only touch one folder: the sandbox.

Run it standalone (speaks MCP over stdin/stdout):

    uv run python -m example_servers.sandbox_files [--root PATH]

Two ideas worth noticing:

1. **Jail every path.** The model sends paths as strings, and strings like
   "../../.ssh/id_rsa" or "/etc/passwd" are one hallucination (or one prompt
   injection) away. `_resolve()` turns every incoming path into an absolute
   path and refuses anything that lands outside the root, *after* following
   symlinks, so a symlink inside the sandbox can't be used as a tunnel out.

2. **Annotate what writes.** Each tool declares MCP `ToolAnnotations`.
   `read_only_hint=True` says "safe to call freely"; the writing tools set
   `read_only_hint=False` (and `destructive_hint=True` when they can destroy
   data). The switchboard reads these hints to decide which calls need a
   human's approval. They are *hints*, so a client should only trust them
   from servers it trusts, which is why the switchboard also lets you force
   approval per server.
"""

import argparse
import os
import shutil
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

# Where the sandbox lives. The env var lets the switchboard (or a test) point
# the server somewhere else; `--root` on the command line wins over both.
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "sandbox"
ROOT = Path(os.environ.get("SANDBOX_ROOT", DEFAULT_ROOT)).resolve()

MAX_READ_CHARS = 50_000  # don't pour a 50 MB log file into the model's context

# `instructions` is sent to the client when it connects. The switchboard puts
# it in Claude's system prompt, so this is how a server explains itself.
server = MCPServer(
    "sandbox-files",
    instructions=(
        "Files live in a sandbox folder. Use paths relative to it, like 'notes/todo.md'. "
        "Call list_dir before guessing file names."
    ),
)

# Two reusable annotation presets, so each tool below reads at a glance.
READ = ToolAnnotations(read_only_hint=True)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True)
DESTROY = ToolAnnotations(read_only_hint=False, destructive_hint=True)


def _resolve(path: str) -> Path:
    """Map a user/model-supplied path to a real path inside ROOT, or refuse."""
    # Treat "/notes.md" as "notes.md": models often add a leading slash.
    candidate = (ROOT / path.lstrip("/\\")).resolve()  # resolve() follows symlinks and "..".
    if candidate != ROOT and ROOT not in candidate.parents:
        # ToolError (not a plain exception) so the message reaches the model:
        # MCP hides the text of unexpected exceptions.
        raise ToolError(f"Path {path!r} is outside the sandbox. Use a path relative to the sandbox root.")
    return candidate


def _rel(p: Path) -> str:
    """Show paths relative to the sandbox, never the real absolute path."""
    return str(p.relative_to(ROOT)) or "."


# --- read-only tools ----------------------------------------------------------
# The function signature becomes the tool's JSON schema; the docstring becomes
# the description Claude reads when deciding which tool to call.


@server.tool(annotations=READ)
def list_dir(path: str = ".") -> str:
    """List a folder in the sandbox. Folders end with '/'. Sizes are in bytes."""
    folder = _resolve(path)
    if not folder.is_dir():
        raise ToolError(f"{path!r} is not a folder.")
    lines = []
    for child in sorted(folder.iterdir(), key=lambda c: (c.is_file(), c.name.lower())):
        lines.append(f"{child.name}/" if child.is_dir() else f"{child.name}  ({child.stat().st_size} B)")
    return "\n".join(lines) or "(empty folder)"


@server.tool(annotations=READ)
def read_file(path: str) -> str:
    """Read a UTF-8 text file from the sandbox."""
    file = _resolve(path)
    if not file.is_file():
        raise ToolError(f"No file at {path!r}. Use list_dir to see what exists.")
    text = file.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + f"\n...[truncated, file is {len(text)} chars]"
    return text


@server.tool(annotations=READ)
def search_files(query: str, path: str = ".") -> str:
    """Find files under a folder whose name or contents contain `query` (case-insensitive)."""
    base = _resolve(path)
    q = query.lower()
    hits = []
    for file in base.rglob("*"):
        if not file.is_file():
            continue
        if q in file.name.lower():
            hits.append(f"{_rel(file)}  (name match)")
            continue
        try:
            for n, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
                if q in line.lower():
                    hits.append(f"{_rel(file)}:{n}: {line.strip()[:200]}")
                    break  # one hit per file keeps the output short
        except (UnicodeDecodeError, OSError):
            pass  # binary or unreadable file: skip
        if len(hits) >= 50:
            break
    return "\n".join(hits) or f"Nothing matched {query!r}."


# --- writing tools --------------------------------------------------------------


@server.tool(annotations=WRITE)
def write_file(path: str, content: str) -> str:
    """Create or overwrite a text file in the sandbox (parent folders are created)."""
    file = _resolve(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {_rel(file)}."


@server.tool(annotations=WRITE)
def make_dir(path: str) -> str:
    """Create a folder (and any missing parents) in the sandbox."""
    folder = _resolve(path)
    folder.mkdir(parents=True, exist_ok=True)
    return f"Folder {_rel(folder)} exists."


@server.tool(annotations=DESTROY)
def move_path(source: str, destination: str) -> str:
    """Move or rename a file or folder inside the sandbox. Overwrites nothing."""
    src, dst = _resolve(source), _resolve(destination)
    if not src.exists():
        raise ToolError(f"No such path: {source!r}.")
    if dst.exists():
        raise ToolError(f"{destination!r} already exists; refusing to overwrite.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(src, dst)
    return f"Moved {_rel(src)} -> {_rel(dst)}."


@server.tool(annotations=DESTROY)
def delete_path(path: str) -> str:
    """Delete a file, or an EMPTY folder, from the sandbox."""
    target = _resolve(path)
    if target == ROOT:
        raise ToolError("Refusing to delete the sandbox root.")
    if target.is_dir():
        if any(target.iterdir()):
            raise ToolError(f"{path!r} is not empty; delete its contents first.")
        target.rmdir()
    elif target.exists():
        target.unlink()
    else:
        raise ToolError(f"No such path: {path!r}.")
    return f"Deleted {_rel(target)}."


def _seed() -> None:
    """First run: create the sandbox with a file to play with."""
    if not ROOT.exists():
        ROOT.mkdir(parents=True)
        (ROOT / "welcome.md").write_text(
            "# Sandbox\n\nThe switchboard's file server can only see this folder.\n"
            "Ask Claude to list it, read this file, or write a new one.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help="sandbox folder (default: ./sandbox or $SANDBOX_ROOT)")
    cli = parser.parse_args()
    if cli.root:
        ROOT = Path(cli.root).resolve()  # module-level rebind: the tools read ROOT at call time
    _seed()
    server.run()  # stdio transport: the client launched us and talks over our stdin/stdout
