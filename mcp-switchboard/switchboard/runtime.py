"""Bridge between Streamlit (synchronous, re-runs on every click) and asyncio.

Streamlit runs your script top to bottom on each interaction, in a plain
thread with no event loop that survives between runs. MCP connections are
asyncio objects that must live on ONE loop for as long as they're open.

So the switchboard gets its own event loop, running forever in a daemon
thread. The UI hands it coroutines with `run(coro)` and waits for the answer:

    Streamlit thread                         background loop thread
    ----------------                         ----------------------
    runtime.run(board.connect(spec))  --->   connect() runs, keeper task parks
          (blocks until done)         <---   result
    ... user clicks, script re-runs ...      keeper tasks still parked, connections open
    runtime.run(run_turn(...))        --->   agent loop, tool calls over those connections
"""

import asyncio
import concurrent.futures
import threading
from typing import Any, Coroutine

from switchboard.hub import Switchboard


class Runtime:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="switchboard-loop", daemon=True)
        self.thread.start()
        self.board = Switchboard()

    def submit(self, coro: Coroutine[Any, Any, Any]) -> concurrent.futures.Future:
        """Start a coroutine on the background loop; return at once with a Future."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def run(self, coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any:
        """Run a coroutine on the background loop and wait for its result (thread-safe)."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def close(self) -> None:
        self.run(self.board.disconnect_all(), timeout=30)
        self.loop.call_soon_threadsafe(self.loop.stop)
