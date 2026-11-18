"""The clock the safety layer times things by.

Upstream reads `time.monotonic()`: confirmation tokens expire after 60 s, the
rate limiter counts calls per minute, and navigation is refused for 3 s after
a take-off is commanded. Here the aircraft flies in simulated time, hundreds of
times faster than the wall clock, so those windows must be measured on the
simulation's clock or they would mean nothing (a take-off that climbed for
ten simulated seconds would still be "0.01 s ago").

`use(source)` installs the flight stack's clock; with none installed this is
`time.monotonic()`, so the vendored modules behave exactly as upstream.
"""

from __future__ import annotations

import time
from collections.abc import Callable

_source: Callable[[], float] | None = None


def monotonic() -> float:
    return _source() if _source is not None else time.monotonic()


def use(source: Callable[[], float] | None) -> None:
    global _source
    _source = source
