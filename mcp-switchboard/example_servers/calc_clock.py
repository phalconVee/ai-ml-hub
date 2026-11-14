"""A tiny maths + time MCP server. All tools are read-only.

    uv run python -m example_servers.calc_clock

Why give an LLM a calculator? Models are good at *setting up* arithmetic and
unreliable at *doing* it digit by digit. And they don't know what time it is
at all: "now" for a model is whenever its training data ends. Two tools,
two whole classes of mistakes gone.

`calculate` evaluates maths WITHOUT `eval()`. Python's `eval` would happily
run `__import__('os').system('rm -rf ~')`. Instead we parse the expression
into a syntax tree and walk it ourselves, allowing only numbers, arithmetic
operators and a whitelist of math functions. Anything else is rejected.
"""

import ast
import math
import operator
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

server = MCPServer(
    "calc-clock",
    instructions="Use calculate for any arithmetic beyond the trivial, and now() before reasoning about dates.",
)
READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)

# Which syntax-tree nodes map to which operation. Anything not listed is refused.
_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    name: getattr(math, name)
    for name in ("sqrt", "log", "log10", "log2", "exp", "sin", "cos", "tan",
                 "asin", "acos", "atan", "floor", "ceil", "factorial", "gcd", "radians", "degrees")
}
_FUNCS |= {"abs": abs, "round": round, "min": min, "max": max}


def _factorial(n: int) -> int:
    if n > 1000:  # factorial(10**6) would hang the server for minutes
        raise ToolError("factorial argument too large (limit 1000).")
    return math.factorial(n)


_FUNCS["factorial"] = _factorial
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _eval(node: ast.AST) -> float | int:
    """Recursively evaluate a whitelisted expression tree."""
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        # 9**9**9 would freeze the server computing a number with millions of digits.
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ToolError("Exponent too large (limit 1000).")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ToolError(f"Not allowed in an expression: {ast.dump(node)[:80]}")


@server.tool(annotations=READ)
def calculate(expression: str) -> str:
    """Evaluate a maths expression exactly, e.g. '(1.07 ** 10) * 2500' or 'sqrt(2) * pi'.

    Supports + - * / // % **, parentheses, pi, e, tau and: sqrt, log, log10, log2,
    exp, sin, cos, tan, asin, acos, atan, floor, ceil, factorial, gcd, radians,
    degrees, abs, round, min, max.
    """
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")  # people type ^ for power
        value = _eval(tree)
    except ToolError:
        raise
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        raise ToolError(f"Could not evaluate {expression!r}: {exc}") from exc
    return f"{expression} = {value}"


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        # Help the model recover: suggest real zone names that look similar.
        close = sorted(z for z in available_timezones() if name.split("/")[-1].lower() in z.lower())[:5]
        raise ToolError(f"Unknown time zone {name!r}. Use IANA names like 'Africa/Lagos'. Close: {close}") from exc


@server.tool(annotations=READ)
def now(timezone: str = "UTC") -> str:
    """The current date and time in an IANA time zone (e.g. 'Europe/London', 'America/New_York')."""
    t = datetime.now(_zone(timezone))
    return f"{t:%A %Y-%m-%d %H:%M:%S} {timezone} (UTC{t:%z})"


@server.tool(annotations=READ)
def convert_time(time: str, from_timezone: str, to_timezone: str) -> str:
    """Convert a wall-clock time between zones. `time` is ISO format, e.g. '2026-11-13 09:30'."""
    try:
        naive = datetime.fromisoformat(time)
    except ValueError as exc:
        raise ToolError(f"Use ISO format like '2026-11-13 09:30', got {time!r}.") from exc
    src = naive.replace(tzinfo=_zone(from_timezone))
    dst = src.astimezone(_zone(to_timezone))
    return f"{src:%Y-%m-%d %H:%M} {from_timezone} = {dst:%A %Y-%m-%d %H:%M} {to_timezone}"


@server.tool(annotations=READ)
def days_between(start: str, end: str) -> str:
    """Whole days from `start` to `end` (ISO dates like '2026-12-25'). Negative if end is earlier."""
    try:
        d = (date.fromisoformat(end) - date.fromisoformat(start)).days
    except ValueError as exc:
        raise ToolError(f"Dates must be ISO format YYYY-MM-DD: {exc}") from exc
    return f"{d} days from {start} to {end} ({d / 7:.1f} weeks)."


if __name__ == "__main__":
    server.run()
