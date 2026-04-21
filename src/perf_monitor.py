"""
perf_monitor.py — Per-stage latency tracking for the pose pipeline.

Tracks rolling-window average latency for named stages (e.g. "extract",
"flow_validate", "score") so we can verify the real-time budget is met.

Targets:
  - REALTIME_BUDGET_MS = 33.0 ms/frame  (30 FPS)
  - WARNING_THRESHOLD_MS = 25.0 ms/frame

Provides:
  - LatencyMonitor      — rolling-window stage timing
  - monitor             — module-level singleton for app-wide use
  - timed(stage_name)   — decorator that records a function's runtime
"""

from __future__ import annotations

import time
from collections import deque
from functools import wraps
from typing import Any, Callable

REALTIME_BUDGET_MS: float = 33.0
WARNING_THRESHOLD_MS: float = 25.0


class LatencyMonitor:
    """Rolling-window latency accumulator."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self.timings: dict[str, deque[float]] = {}

    def record(self, stage_name: str, duration_ms: float) -> None:
        if stage_name not in self.timings:
            self.timings[stage_name] = deque(maxlen=self.window_size)
        self.timings[stage_name].append(float(duration_ms))

    def avg(self, stage_name: str) -> float:
        q = self.timings.get(stage_name)
        if not q:
            return 0.0
        return sum(q) / len(q)

    def total_avg(self) -> float:
        return sum(self.avg(s) for s in self.timings)

    def reset(self) -> None:
        self.timings.clear()

    def report(self) -> dict[str, Any]:
        total = self.total_avg()
        fps_estimate = 1000.0 / total if total > 0 else 0.0
        return {
            "stages": {s: round(self.avg(s), 2) for s in self.timings},
            "total_ms": round(total, 2),
            "fps_estimate": round(fps_estimate, 1),
            "budget_ms": REALTIME_BUDGET_MS,
            "warning_ms": WARNING_THRESHOLD_MS,
            "over_budget": total > REALTIME_BUDGET_MS,
            "over_warning": total > WARNING_THRESHOLD_MS,
        }


# Module-level singleton.
monitor = LatencyMonitor()


def timed(stage_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: record a function's runtime into the module monitor."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000.0
                monitor.record(stage_name, duration_ms)

        return wrapped

    return decorator
