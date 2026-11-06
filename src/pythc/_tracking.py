"""Optional integration with the external ``tracking`` package.

``pythc`` only needs ``Configurable`` and the no-op-by-default ``observe``
interface.  When ``tracking`` is not installed, use a local no-op fallback so
that the library remains fully usable without it.
"""

from __future__ import annotations

try:
    import tracking as _tracking
except ModuleNotFoundError as exc:
    if exc.name != "tracking":
        raise
    _tracking = None

if _tracking is not None and hasattr(_tracking, "Configurable") and hasattr(
    _tracking, "observe"
):
    Configurable = _tracking.Configurable
    observe = _tracking.observe
else:
    from ._tracking_fallback import Configurable, observe

__all__ = ["Configurable", "observe"]
