"""Lightweight observability interface owned by ``pythc``.

The library only declares *what* can be observed (metrics, checkpoints);
the benchmark driver provides the implementation (see ``pythc-benchmark``).

Usage inside ``pythc``::

    from pythc import observe

    observe.log_metric("n_thc", X.shape[0])
    observe.checkpoint("grid_build")

If no run is active both helpers are no-ops, so library code needs no
``if active:`` guards and no heavy third-party imports.
"""

from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from typing import Any, Iterator, Protocol, runtime_checkable


@runtime_checkable
class RunTracer(Protocol):
    """Minimal interface a benchmark run must implement."""

    def log_metric(self, key: str, value: Any) -> None: ...
    def checkpoint(self, name: str) -> None: ...
    def append_metric(self, key: str, value: Any) -> None:
        """Append ``value`` to the list metric stored under ``key``."""
        ...


class NullTracer:
    """No-op tracer used when observability is disabled."""

    def log_metric(self, key: str, value: Any) -> None:
        return None

    def checkpoint(self, name: str) -> None:
        return None

    def append_metric(self, key: str, value: Any) -> None:
        return None


_current: contextvars.ContextVar[RunTracer | None] = contextvars.ContextVar(
    "pythc_active_run", default=None
)


def get_active() -> RunTracer | None:
    """Return the currently active run tracer, or ``None``."""
    return _current.get()


def set_active(run: RunTracer | None):
    """Install ``run`` as the active tracer; returns a token for :func:`reset`."""
    return _current.set(run)


def reset(token) -> None:
    """Restore the tracer active before the matching :func:`set_active` call."""
    _current.reset(token)


def log_metric(key: str, value: Any) -> None:
    """Log ``value`` under ``key`` on the active run, if any."""
    run = _current.get()
    if run is not None:
        run.log_metric(key, value)


def checkpoint(name: str) -> None:
    """Record a named checkpoint on the active run, if any."""
    run = _current.get()
    if run is not None:
        run.checkpoint(name)


def append_metric(key: str, value: Any) -> None:
    """Append ``value`` to the list metric ``key`` on the active run, if any."""
    run = _current.get()
    if run is not None:
        run.append_metric(key, value)


@contextmanager
def measure(name: str) -> Iterator[None]:
    """Measure a block of code as ``<name>_time`` on the active run, if any."""
    run = _current.get()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if run is not None:
            run.log_metric(f"{name}_time", time.perf_counter() - t0)
