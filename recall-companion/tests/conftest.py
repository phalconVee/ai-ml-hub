from datetime import datetime, timedelta, timezone

import pytest

from recall.store import MemoryStore


class Clock:
    """A clock the test can move forward: clock.advance(days=7)."""

    def __init__(self):
        self.t = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(clock):
    return MemoryStore(":memory:", clock=clock)
