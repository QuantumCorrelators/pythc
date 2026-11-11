"""Local no-op fallback for the optional external ``tracking`` package."""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from typing import Any, Iterator


class Configurable:
    @classmethod
    def from_config(cls, config: dict):
        """
        Factory method: inspects the child class's __init__,
        extracts relevant keys from 'config', and instantiates.
        """
        signature = inspect.signature(cls.__init__)

        valid_params = {}
        for param_name in signature.parameters:
            if param_name == "self":
                continue

            if param_name in config:
                valid_params[param_name] = config[param_name]

        return cls(**valid_params)


class _NoopObserve:
    """No-op replacement for ``tracking.observe``."""

    def log_metric(self, key: str, value: Any) -> None:
        return None

    def checkpoint(self, name: str) -> None:
        return None

    def append_metric(self, key: str, value: Any) -> None:
        return None

    def add_artifact(
        self,
        name: str,
        data: Any,
        *,
        media_type: str | None = None,
        format: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {}

    def add_array(
        self, name: str, arrays: dict[str, Any], *, axes: Any = None
    ) -> dict[str, Any]:
        return {}

    def get_active(self):
        return None

    def set_active(self, run):
        return None

    def reset(self, token) -> None:
        return None

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        yield

    @contextmanager
    def isolated(self) -> Iterator[dict[str, Any]]:
        yield {}


observe = _NoopObserve()
