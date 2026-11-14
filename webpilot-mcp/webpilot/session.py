"""Keeping one browser alive across Streamlit reruns.

The problem: Streamlit re-runs app.py from top to bottom on every click, and
that code is synchronous. But the MCP connection (and the browser behind it)
is async and must survive between reruns, e.g. while the user decides
whether to approve a form submission.

The fix: a background thread running its own asyncio event loop, forever.
  * `PilotSession.start()` opens the MCP toolbox inside a long-lived task on
    that loop (MCP's anyio scopes must be entered and exited by the same task,
    so one "holder" task owns the connection until `close()`).
  * The UI hands coroutines to the loop with `run_coroutine_threadsafe` and
    waits for the result: `call()` for one result, `iterate()` to pull events
    out of an async generator one at a time, so each renders immediately.

The command-line runner (__main__.py) uses the same class.
"""

import asyncio
import threading
from typing import Any, AsyncIterator, Iterator

from webpilot.agent import AgentRun, Event
from webpilot.config import Settings
from webpilot.llm import LLM
from webpilot.toolbox import Toolbox, open_toolbox


class PilotSession:
    def __init__(self, settings: Settings, llm: Any = None, server: Any = None):
        self.settings = settings
        self._llm = llm  # injected in tests; built lazily otherwise
        self._server = server  # None = launch the stdio subprocess
        self.toolbox: Toolbox | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="webpilot-loop")
        self._thread.start()
        self._stop: asyncio.Event | None = None
        self._holder: Any = None

    # --- bridging sync <-> async ---------------------------------------------

    def call(self, coro, timeout: float | None = 300) -> Any:
        """Run a coroutine on the background loop and wait for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def iterate(self, agen: AsyncIterator[Event]) -> Iterator[Event]:
        """Turn an async generator on the background loop into a normal iterator."""
        while True:
            try:
                yield self.call(agen.__anext__())
            except StopAsyncIteration:
                return

    # --- lifecycle --------------------------------------------------------------

    def start(self) -> "PilotSession":
        """Launch the browser server and connect to it. Blocks until ready."""
        ready = self.call(self._spawn_holder(), timeout=30)

        async def wait_ready():
            await ready  # re-raises if the connection failed

        self.call(wait_ready(), timeout=60)
        return self

    async def _spawn_holder(self) -> asyncio.Future:
        self._stop = asyncio.Event()
        ready = asyncio.get_running_loop().create_future()
        self._holder = asyncio.create_task(self._hold(ready))
        return ready

    async def _hold(self, ready: asyncio.Future) -> None:
        """Owns the MCP connection from open to close, in a single task."""
        try:
            async with open_toolbox(self.settings, self._server) as toolbox:
                self.toolbox = toolbox
                ready.set_result(True)
                await self._stop.wait()
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
        finally:
            self.toolbox = None

    def close(self) -> None:
        """Shut the browser server down and stop the background loop."""
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
            if self._holder is not None:
                try:
                    self.call(asyncio.wait_for(asyncio.shield(self._holder), 15), timeout=20)
                except Exception:
                    pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    # --- the agent ---------------------------------------------------------------

    def new_run(self, goal: str) -> AgentRun:
        if self.toolbox is None:
            raise RuntimeError("session not started")
        if self._llm is None:
            self._llm = LLM(self.settings)
        return AgentRun(goal=goal, toolbox=self.toolbox, llm=self._llm, settings=self.settings)

    def advance(self, run: AgentRun, approved: bool | None = None) -> Iterator[Event]:
        """Stream a run's events, synchronously, for the UI."""
        return self.iterate(run.advance(approved))
