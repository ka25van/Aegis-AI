"""Reliability primitives: CircuitBreaker + in-memory idempotency fallback."""
from __future__ import annotations

import time
from typing import Any, Optional


class CircuitBreaker:
    """Simple circuit breaker — no extra deps.

    Closed -> Open after `failure_threshold` failures within `window_s`.
    Open -> HalfOpen after `recovery_timeout` s.
    """

    def __init__(self, failure_threshold: int = 5, window_s: int = 60, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.window_s = window_s
        self.recovery_timeout = recovery_timeout
        self._failures: list[float] = []
        self._state = "closed"
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> str:
        if self._state == "open" and self._opened_at and time.monotonic() - self._opened_at >= self.recovery_timeout:
            self._state = "half_open"
        return self._state

    def record_success(self) -> None:
        self._failures.clear()
        self._state = "closed"
        self._opened_at = None

    def record_failure(self) -> None:
        now = time.monotonic()
        self._failures = [t for t in self._failures if t > now - self.window_s]
        self._failures.append(now)
        if len(self._failures) >= self.failure_threshold:
            self._state = "open"
            self._opened_at = now

    def allow_request(self) -> bool:
        s = self.state
        return s in ("closed", "half_open")

    def remaining(self) -> float:
        if self._state == "open" and self._opened_at:
            return max(0, self.recovery_timeout - (time.monotonic() - self._opened_at))
        return 0
