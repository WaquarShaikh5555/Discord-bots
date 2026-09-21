"""Concurrency primitives that keep the bot inside free-tier API budgets.

Three independent guards are used by :class:`bot.services.ai_service.AIService`:

``TokenBucket``
    Smooths requests per minute so a burst of tickets in one server cannot burn
    a whole free-tier allowance in a few seconds (and trigger HTTP 429).

``DailyBudget``
    Hard cap on requests per UTC day. Groq's free tier is 14,400/day; once the
    budget is gone the provider is skipped instead of returning 429s.

``CircuitBreaker``
    Stops hammering a provider that is erroring, and lets it recover on its own.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"        # healthy — requests flow
    OPEN = "open"            # tripped — requests are rejected until cooldown ends
    HALF_OPEN = "half_open"  # cooldown elapsed — a single probe request is allowed


class TokenBucket:
    """Async token bucket with a bounded wait.

    ``rate_per_minute`` refills continuously; ``burst`` caps how many tokens can
    be banked so an idle bot does not fire a huge burst on wake-up.
    """

    __slots__ = ("rate_per_minute", "burst", "_tokens", "_updated", "_lock")

    def __init__(self, rate_per_minute: float, burst: int | None = None) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self.rate_per_minute = float(rate_per_minute)
        self.burst = float(burst if burst is not None else max(1.0, min(rate_per_minute, 5.0)))
        self._tokens = self.burst
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - self._updated)
        self._tokens = min(self.burst, self._tokens + elapsed * (self.rate_per_minute / 60.0))
        self._updated = now

    def wait_time(self) -> float:
        """Seconds until one token is available (0.0 if available now)."""
        self._refill()
        if self._tokens >= 1.0:
            return 0.0
        deficit = 1.0 - self._tokens
        return deficit / (self.rate_per_minute / 60.0)

    async def acquire(self, timeout: float | None = None) -> bool:
        """Block until a token is available. Returns False if ``timeout`` elapses."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                wait = (1.0 - self._tokens) / (self.rate_per_minute / 60.0)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)
            await asyncio.sleep(min(wait, 5.0))

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self._tokens


class DailyBudget:
    """Request counter that resets at UTC midnight."""

    __slots__ = ("limit", "_used", "_day", "_lock", "_clock")

    def __init__(self, limit: int, *, clock=time.time) -> None:
        self.limit = int(limit)
        self._clock = clock
        self._used = 0
        self._day = self._today(clock)
        self._lock = asyncio.Lock()

    @staticmethod
    def _today(clock=time.time) -> str:
        return datetime.fromtimestamp(clock(), tz=timezone.utc).strftime("%Y-%m-%d")

    def _rollover(self) -> None:
        today = self._today(self._clock)
        if today != self._day:
            self._day = today
            self._used = 0

    async def try_consume(self, amount: int = 1) -> bool:
        if self.limit <= 0:  # 0 or negative disables the cap
            return True
        async with self._lock:
            self._rollover()
            if self._used + amount > self.limit:
                return False
            self._used += amount
            return True

    async def restore(self, amount: int = 1) -> None:
        """Give tokens back (used when a request failed before hitting the API)."""
        async with self._lock:
            self._rollover()
            self._used = max(0, self._used - amount)

    @property
    def used(self) -> int:
        self._rollover()
        return self._used

    @property
    def remaining(self) -> int:
        if self.limit <= 0:
            return -1
        return max(0, self.limit - self.used)


@dataclass
class BreakerSnapshot:
    state: CircuitState
    failures: int
    opened_at: float | None
    retry_in: float


class CircuitBreaker:
    """Trip after ``threshold`` consecutive failures; recover after ``cooldown``."""

    __slots__ = ("threshold", "cooldown", "_failures", "_opened_at", "_half_open_in_flight", "_clock")

    def __init__(self, threshold: int = 4, cooldown: float = 60.0, *, clock=time.monotonic) -> None:
        self.threshold = max(1, threshold)
        self.cooldown = max(0.1, cooldown)
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open_in_flight = False
        self._clock = clock

    @property
    def state(self) -> CircuitState:
        if self._opened_at is None:
            return CircuitState.CLOSED
        if self._clock() - self._opened_at < self.cooldown:
            return CircuitState.OPEN
        return CircuitState.HALF_OPEN

    def allow_request(self) -> bool:
        """Reserve permission to send one request, if the breaker allows it."""
        state = self.state
        if state is CircuitState.CLOSED:
            return True
        if state is CircuitState.OPEN:
            return False
        # Half-open: let exactly one probe through at a time.
        if self._half_open_in_flight:
            return False
        self._half_open_in_flight = True
        return True

    def release(self) -> None:
        """Give back a half-open probe reservation without recording an outcome.

        Called when a request was permitted by the breaker but then skipped for
        an unrelated reason (daily budget spent, no RPM slot, deadline hit).
        Without this the breaker would wait for a probe that never happens.
        """
        self._half_open_in_flight = False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._half_open_in_flight = False

    def record_failure(self) -> None:
        self._half_open_in_flight = False
        self._failures += 1
        if self.state is CircuitState.HALF_OPEN or self._failures >= self.threshold:
            self._opened_at = self._clock()

    def retry_in(self) -> float:
        if self._opened_at is None:
            return 0.0
        return max(0.0, self.cooldown - (self._clock() - self._opened_at))

    def snapshot(self) -> BreakerSnapshot:
        return BreakerSnapshot(
            state=self.state,
            failures=self._failures,
            opened_at=self._opened_at,
            retry_in=self.retry_in(),
        )
