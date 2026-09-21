"""Inference orchestration: retries, provider failover, rate limits, budgets.

One call to :meth:`AIService.generate` does the following, in order:

1. Walk the configured provider chain (default: Groq -> Cerebras -> Gemini).
2. Skip a provider whose circuit breaker is open or whose daily budget is spent.
3. Wait for a per-provider token bucket slot so a burst of tickets cannot trip
   the free-tier RPM limit (Groq's free tier is 15 RPM / 14,400 RPD).
4. Retry transient failures (429, 5xx, timeouts, network) with exponential
   backoff + jitter, honouring ``Retry-After`` when the provider sends one.
5. Never retry a fatal error (401/403/404/400) — fail over to the next provider
   immediately instead of burning the deadline on a broken key.
6. Respect a hard end-to-end deadline, because a ticket answer that arrives
   after a minute is worse than an escalation to staff.

Everything is telemetry-instrumented: provider, model, latency and attempt
counts land in the ``llm_usage`` table and in the ``/ai-status`` command.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Sequence

from bot.config import Settings
from bot.services.providers.base import LLMProvider, ProviderError, ProviderResponse
from bot.utils.logging_setup import get_logger
from bot.utils.ratelimit import CircuitBreaker, DailyBudget, TokenBucket
from bot.utils.text import sanitize_ai_text, strip_code_fence, strip_echoed_prefix

log = get_logger(__name__)

UsageSink = Callable[[str, str, bool, int, int, int], Awaitable[None]]

#: Safety ceiling for a provider-supplied ``Retry-After`` so a bogus header
#: cannot stall the worker forever.
_MAX_RETRY_AFTER_SECONDS = 120.0


class AIServiceError(RuntimeError):
    """Every provider in the chain failed for one request."""

    def __init__(self, message: str, *, attempts: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.attempts = list(attempts)


@dataclass(frozen=True)
class AIResult:
    """Successful (or definitively failed) outcome of one inference call."""

    ok: bool
    text: str
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    attempts: tuple[str, ...] = ()
    finish_reason: str | None = None
    error: str = ""
    truncated: bool = False

    @property
    def summary(self) -> str:
        if self.ok:
            return f"{self.provider}/{self.model} in {self.latency_ms}ms ({len(self.text)} chars)"
        return f"failed after {len(self.attempts)} attempt(s): {self.error}"


@dataclass
class ProviderState:
    """Runtime guards attached to one provider instance."""

    provider: LLMProvider
    bucket: TokenBucket
    breaker: CircuitBreaker
    budget: DailyBudget
    successes: int = 0
    failures: int = 0
    total_latency_ms: int = 0
    last_error: str = ""
    last_success_at: datetime | None = None

    @property
    def average_latency_ms(self) -> int:
        return int(self.total_latency_ms / self.successes) if self.successes else 0

    def health(self) -> dict[str, Any]:
        snapshot = self.breaker.snapshot()
        return {
            "provider": self.provider.name,
            "model": self.provider.model,
            "available": self.provider.available,
            "circuit": snapshot.state.value,
            "consecutive_failures": snapshot.failures,
            "retry_in_seconds": round(snapshot.retry_in, 1),
            "successes": self.successes,
            "failures": self.failures,
            "avg_latency_ms": self.average_latency_ms,
            "rpm_limit": self.provider.settings.requests_per_minute,
            "requests_today": self.budget.used,
            "daily_limit": self.budget.limit,
            "daily_remaining": self.budget.remaining,
            "last_error": self.last_error,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
        }


@dataclass
class AIServiceStats:
    """Process-lifetime counters surfaced by ``/ai-status``."""

    requests: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    failovers: int = 0
    skipped_rate_limit: int = 0
    skipped_breaker: int = 0
    skipped_budget: int = 0
    total_latency_ms: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def average_latency_ms(self) -> int:
        return int(self.total_latency_ms / self.successes) if self.successes else 0

    @property
    def success_rate(self) -> float:
        return (self.successes / self.requests * 100.0) if self.requests else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "successes": self.successes,
            "failures": self.failures,
            "retries": self.retries,
            "failovers": self.failovers,
            "skipped_rate_limit": self.skipped_rate_limit,
            "skipped_breaker": self.skipped_breaker,
            "skipped_budget": self.skipped_budget,
            "avg_latency_ms": self.average_latency_ms,
            "success_rate_pct": round(self.success_rate, 2),
            "uptime_seconds": int((datetime.now(timezone.utc) - self.started_at).total_seconds()),
        }


class AIService:
    """Provider-agnostic inference facade used by the ticket listener."""

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        settings: Settings,
        *,
        usage_sink: UsageSink | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.stats = AIServiceStats()
        self._usage_sink = usage_sink
        self._sleep = sleep
        self._clock = clock
        self._states: list[ProviderState] = [self._make_state(p) for p in providers]
        self._by_name: dict[str, ProviderState] = {s.provider.name: s for s in self._states}

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _make_state(self, provider: LLMProvider) -> ProviderState:
        sp = provider.settings
        return ProviderState(
            provider=provider,
            bucket=TokenBucket(sp.requests_per_minute),
            breaker=CircuitBreaker(
                self.settings.circuit_breaker_threshold,
                self.settings.circuit_breaker_cooldown,
                clock=self._clock,
            ),
            budget=DailyBudget(sp.requests_per_day, clock=time.time),
        )

    @property
    def providers(self) -> list[str]:
        return [state.provider.name for state in self._states]

    def describe_chain(self) -> str:
        """Human-readable chain order, e.g. ``groq(llama-3.3-70b) → gemini(...)``."""
        if not self._states:
            return "no providers configured"
        return " → ".join(
            f"{state.provider.name}({state.provider.model})" for state in self._states
        )

    @property
    def active_provider(self) -> str:
        """First provider that is currently able to serve a request."""
        for state in self._states:
            if state.provider.available and state.breaker.state.value == "closed":
                return state.provider.name
        return self.providers[0] if self.providers else "none"

    def health(self) -> list[dict[str, Any]]:
        return [state.health() for state in self._states]

    def reset_breakers(self) -> None:
        """Force every circuit closed (used by ``/ai-status`` recovery action)."""
        for state in self._states:
            state.breaker.record_success()

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #
    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        guild_id: int | str | None = None,
    ) -> AIResult:
        """Produce one completion, failing over across providers as needed."""
        if not self._states:
            return AIResult(ok=False, text="", error="no AI provider is configured")

        self.stats.requests += 1
        started = self._clock()
        deadline = started + self.settings.total_deadline
        attempts: list[str] = []
        errors: list[str] = []

        for index, state in enumerate(self._states):
            if index > 0:
                self.stats.failovers += 1

            remaining = deadline - self._clock()
            if remaining <= 0.5:
                errors.append(f"{state.provider.name}: skipped, total deadline exceeded")
                log.warning(
                    "AI deadline exhausted before trying %s (guild=%s)", state.provider.name, guild_id
                )
                break

            if not state.provider.available:
                continue

            if not state.breaker.allow_request():
                self.stats.skipped_breaker += 1
                note = (
                    f"{state.provider.name}: circuit open, retry in "
                    f"{state.breaker.retry_in():.0f}s"
                )
                attempts.append(note)
                errors.append(note)
                log.info("Skipping %s — %s", note, "circuit breaker open")
                continue

            probe_reserved = True
            try:
                if not await state.budget.try_consume():
                    self.stats.skipped_budget += 1
                    note = (
                        f"{state.provider.name}: daily budget exhausted "
                        f"({state.budget.limit}/day)"
                    )
                    attempts.append(note)
                    errors.append(note)
                    log.warning("Skipping %s — daily request budget spent", state.provider.name)
                    continue

                # Wait for an RPM slot, but never longer than the remaining deadline.
                acquire_timeout = max(0.5, min(remaining - 0.5, self._rate_wait_ceiling(state)))
                acquired = await state.bucket.acquire(timeout=acquire_timeout)
                if not acquired:
                    self.stats.skipped_rate_limit += 1
                    note = (
                        f"{state.provider.name}: rate limit slot unavailable within "
                        f"{acquire_timeout:.1f}s"
                    )
                    attempts.append(note)
                    errors.append(note)
                    log.warning(
                        "Skipping %s — RPM slot unavailable (%s)", state.provider.name, note
                    )
                    continue

                result = await self._try_provider(
                    state,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    deadline=deadline,
                    attempts=attempts,
                    errors=errors,
                )
                if result is not None:
                    self.stats.successes += 1
                    self.stats.total_latency_ms += result.latency_ms
                    await self._record_usage(state, result, success=True)
                    return result

                await self._record_usage(state, None, success=False)
            finally:
                # The attempt loop above always records an outcome except when it
                # bails on the deadline; release any unspent half-open probe.
                if probe_reserved:
                    state.breaker.release()
                probe_reserved = False

        elapsed_ms = int((self._clock() - started) * 1000)
        self.stats.failures += 1
        error_text = "; ".join(dict.fromkeys(errors)) or "all providers unavailable"
        log.error(
            "AI generation failed in %dms (guild=%s): %s", elapsed_ms, guild_id, error_text
        )
        return AIResult(
            ok=False,
            text="",
            attempts=tuple(attempts),
            error=error_text,
            latency_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------ #
    # per-provider retry loop
    # ------------------------------------------------------------------ #
    async def _try_provider(
        self,
        state: ProviderState,
        *,
        system_prompt: str,
        user_prompt: str,
        deadline: float,
        attempts: list[str],
        errors: list[str],
    ) -> AIResult | None:
        """Attempt one provider with bounded retries. ``None`` means give up on it."""
        provider = state.provider
        max_attempts = self.settings.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            remaining = deadline - self._clock()
            if remaining <= 0.5:
                errors.append(f"{provider.name}: deadline exceeded on attempt {attempt}")
                state.breaker.release()
                break

            started = self._clock()
            try:
                response: ProviderResponse = await asyncio.wait_for(
                    provider.complete(system_prompt=system_prompt, user_prompt=user_prompt),
                    timeout=min(provider.settings.timeout, remaining),
                )
            except asyncio.CancelledError:
                raise
            except ProviderError as exc:
                latency_ms = int((self._clock() - started) * 1000)
                state.last_error = str(exc)
                backoff = self._backoff_delay(attempt, exc.retry_after, deadline)
                self._handle_provider_error(
                    state,
                    exc,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    latency_ms=latency_ms,
                    attempts=attempts,
                    errors=errors,
                    next_backoff=backoff,
                )
                if not exc.retryable:
                    return None  # fatal for this provider — fail over now
                if attempt >= max_attempts or backoff is None:
                    return None
                await self._sleep(backoff)
                continue
            except asyncio.TimeoutError:
                latency_ms = int((self._clock() - started) * 1000)
                exc = ProviderError("request timed out", retryable=True, provider=provider.name)
                state.last_error = str(exc)
                backoff = self._backoff_delay(attempt, None, deadline)
                self._handle_provider_error(
                    state,
                    exc,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    latency_ms=latency_ms,
                    attempts=attempts,
                    errors=errors,
                    next_backoff=backoff,
                )
                if attempt >= max_attempts or backoff is None:
                    return None
                await self._sleep(backoff)
                continue
            except Exception as exc:  # pragma: no cover - unexpected SDK/driver bug
                state.failures += 1
                state.breaker.record_failure()
                state.last_error = f"{type(exc).__name__}: {exc}"
                errors.append(f"{provider.name}: unexpected {type(exc).__name__}: {exc}")
                log.exception("Unexpected error calling provider %s", provider.name)
                return None

            # --- success path ------------------------------------------- #
            text = postprocess_completion(response.text)
            if not text:
                errors.append(f"{provider.name}: completion was empty after post-processing")
                state.breaker.record_failure()
                backoff = self._backoff_delay(attempt, None, deadline)
                if attempt >= max_attempts or backoff is None:
                    return None
                await self._sleep(backoff)
                continue

            state.breaker.record_success()
            state.successes += 1
            state.failures = 0
            state.total_latency_ms += response.latency_ms
            state.last_error = ""
            state.last_success_at = datetime.now(timezone.utc)

            return AIResult(
                ok=True,
                text=text,
                provider=response.provider or provider.name,
                model=response.model or provider.model,
                latency_ms=response.latency_ms,
                attempts=tuple(attempts),
                finish_reason=response.finish_reason,
                truncated=response.truncated,
            )

        return None

    def _handle_provider_error(
        self,
        state: ProviderState,
        exc: ProviderError,
        *,
        attempt: int,
        max_attempts: int,
        latency_ms: int,
        attempts: list[str],
        errors: list[str],
        next_backoff: float | None = None,
    ) -> None:
        """Bookkeeping for one failed attempt (breaker, counters, log line)."""
        state.failures += 1
        note = f"{state.provider.name} attempt {attempt}/{max_attempts}: {exc}"
        attempts.append(note)
        errors.append(note)

        if exc.retryable:
            self.stats.retries += 1
            state.breaker.record_failure()
            if attempt >= max_attempts:
                note = " — no attempts left"
            elif next_backoff is None:
                note = " — deadline exhausted, failing over"
            else:
                note = f" — retrying in {next_backoff:.2f}s"
            log.warning(
                "Provider %s failed (attempt %d/%d, %dms): %s%s",
                state.provider.name,
                attempt,
                max_attempts,
                latency_ms,
                exc,
                note,
            )
        else:
            # Fatal configuration errors should not trip the breaker into a long
            # cooldown loop; they are simply skipped for this request.
            state.breaker.record_failure()
            log.error("Provider %s rejected the request permanently: %s", state.provider.name, exc)

    def _backoff_delay(self, attempt: int, retry_after: float | None, deadline: float) -> float | None:
        """Exponential backoff with jitter, capped by the remaining deadline.

        Returns ``None`` when there is not enough time left to bother waiting.
        """
        base = self.settings.backoff_base * (2 ** (attempt - 1))
        delay = min(base, self.settings.backoff_cap)
        # Full jitter avoids synchronised retry storms across shards/guilds.
        delay = random.uniform(delay * 0.5, delay)  # noqa: S311 - not a security context
        if retry_after is not None:
            # The provider told us exactly how long to wait. Respect it (bounded
            # by a sanity ceiling); the remaining deadline decides whether we can
            # afford to, and if not we fail over to the next provider instead.
            delay = max(delay, min(retry_after, _MAX_RETRY_AFTER_SECONDS))
        remaining = deadline - self._clock()
        if delay + 0.5 >= remaining:
            return None
        return delay

    def _rate_wait_ceiling(self, state: ProviderState) -> float:
        """How long we are willing to wait for an RPM slot on this provider."""
        wait = state.bucket.wait_time()
        return max(0.5, min(wait + 0.5, 20.0))

    async def _record_usage(
        self, state: ProviderState, result: AIResult | None, *, success: bool
    ) -> None:
        if self._usage_sink is None:
            return
        try:
            await self._usage_sink(
                state.provider.name,
                state.provider.model,
                success,
                result.latency_ms if result else 0,
                len(result.text) if result else 0,
                0,
            )
        except Exception:  # pragma: no cover - telemetry must never break a reply
            log.exception("Failed to persist LLM usage telemetry")


def postprocess_completion(text: str) -> str:
    """Normalise raw model output before escalation parsing and delivery.

    * unwrap a stray code fence
    * drop an echoed ``AI Agent Response:`` prefix
    * strip mentions (the only allowed mention is added by the escalation layer)
    """
    if not text:
        return ""
    cleaned = strip_code_fence(text)
    cleaned = strip_echoed_prefix(cleaned)
    cleaned = sanitize_ai_text(cleaned)
    return cleaned.strip()


def build_ai_service(
    settings: Settings,
    session: Any,
    *,
    usage_sink: UsageSink | None = None,
) -> AIService:
    """Convenience factory wiring providers + settings into an :class:`AIService`."""
    from bot.services.providers.factory import build_providers  # local import avoids a cycle

    providers = build_providers(settings, session)
    return AIService(providers, settings, usage_sink=usage_sink)
