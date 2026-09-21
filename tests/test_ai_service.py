"""Retry, failover, circuit-breaking and budget behaviour of the AI service."""

from __future__ import annotations

import asyncio

import pytest

from bot.services.ai_service import AIService, postprocess_completion
from bot.services.providers import CerebrasProvider, GeminiProvider, GroqProvider
from bot.services.providers.base import (
    ProviderError,
    classify_http_error,
    extract_completion_text,
)
from bot.services.providers.factory import build_providers
from bot.utils.ratelimit import CircuitBreaker, CircuitState, DailyBudget, TokenBucket
from fakes import FakeResponse, FakeSession, gemini_ok, openai_ok, rate_limited
from conftest import make_ai_service


@pytest.fixture
def session() -> FakeSession:
    return FakeSession()


# --------------------------------------------------------------------------- #
# happy paths
# --------------------------------------------------------------------------- #
async def test_successful_completion(settings, session):
    session._responses = [openai_ok("Refunds take 14 days.", model="llama-3.3-70b-versatile")]
    service = make_ai_service(settings, session)
    result = await service.generate(system_prompt="sys", user_prompt="refund?")

    assert result.ok is True
    assert result.text == "Refunds take 14 days."
    assert result.provider == "groq"
    assert result.model == "llama-3.3-70b-versatile"
    assert session.call_count == 1
    assert service.stats.successes == 1


async def test_request_payload_matches_openai_chat_schema(settings, session):
    session._responses = [openai_ok("ok")]
    service = make_ai_service(settings, session)
    await service.generate(system_prompt="SYSTEM TEXT", user_prompt="USER TEXT")

    call = session.calls[0]
    assert call["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer test-groq-key"
    payload = call["json"]
    assert payload["model"] == "llama-3.3-70b-versatile"
    assert payload["messages"][0] == {"role": "system", "content": "SYSTEM TEXT"}
    assert payload["messages"][1] == {"role": "user", "content": "USER TEXT"}
    assert payload["stream"] is False
    assert payload["temperature"] == 0.2


async def test_cerebras_uses_its_own_endpoint_and_token_field(settings, session):
    session._responses = [openai_ok("fast answer")]
    service = make_ai_service(settings, session, providers=("cerebras",))
    result = await service.generate(system_prompt="s", user_prompt="u")

    assert result.provider == "cerebras"
    assert session.calls[0]["url"] == "https://api.cerebras.ai/v1/chat/completions"
    assert "max_completion_tokens" in session.calls[0]["json"]


async def test_gemini_payload_and_parsing(settings, session):
    session._responses = [gemini_ok("Gemini answer")]
    service = make_ai_service(settings, session, providers=("gemini",))
    result = await service.generate(system_prompt="SYS", user_prompt="USR")

    assert result.ok and result.text == "Gemini answer"
    call = session.calls[0]
    assert ":generateContent" in call["url"]
    assert call["headers"]["x-goog-api-key"] == "test-gemini-key"
    assert call["json"]["system_instruction"]["parts"][0]["text"] == "SYS"


# --------------------------------------------------------------------------- #
# retries
# --------------------------------------------------------------------------- #
async def test_retries_on_429_then_succeeds(settings, session):
    session._responses = [rate_limited(0.01), rate_limited(0.01), openai_ok("finally")]
    service = make_ai_service(settings, session)
    result = await service.generate(system_prompt="s", user_prompt="u")

    assert result.ok and result.text == "finally"
    assert session.call_count == 3
    assert service.stats.retries >= 2


async def test_retry_after_header_is_respected(settings, session):
    observed: list[float] = []

    async def fake_sleep(delay: float) -> None:
        observed.append(delay)
        await asyncio.sleep(0)

    session._responses = [rate_limited(3.5), openai_ok("ok")]
    service = make_ai_service(settings, session, sleep=fake_sleep)
    result = await service.generate(system_prompt="s", user_prompt="u")

    assert result.ok
    assert observed and observed[0] >= 3.5, "Retry-After must override the backoff floor"


async def test_retries_on_server_error_then_fails_over(settings, session):
    session._responses = [
        FakeResponse(status=503, body={"error": {"message": "groq down"}}),
        FakeResponse(status=503, body={"error": {"message": "groq down"}}),
        FakeResponse(status=503, body={"error": {"message": "groq down"}}),
        openai_ok("cerebras saved us"),
    ]
    service = make_ai_service(settings, session, providers=("groq", "cerebras"))
    result = await service.generate(system_prompt="s", user_prompt="u")

    assert result.ok and result.provider == "cerebras"
    assert service.stats.failovers == 1


async def test_empty_completion_is_retried_then_escalated_as_failure(settings, session):
    session._responses = [openai_ok("   "), openai_ok("")]
    service = make_ai_service(settings, session, providers=("groq",))
    result = await service.generate(system_prompt="s", user_prompt="u")

    assert result.ok is False
    assert "empty" in result.error


# --------------------------------------------------------------------------- #
# failover semantics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_fatal_errors_do_not_retry_but_fail_over(settings, session, status):
    session._responses = [
        FakeResponse(status=status, body={"error": {"message": "nope"}}),
        openai_ok("cerebras answered"),
    ]
    service = make_ai_service(settings, session, providers=("groq", "cerebras"))
    result = await service.generate(system_prompt="s", user_prompt="u")

    assert result.ok and result.provider == "cerebras"
    assert session.call_count == 2, "a fatal error must not be retried on the same provider"
    assert service.stats.retries == 0


async def test_all_providers_failing_returns_error_with_attempts(settings, session):
    session._responses = [FakeResponse(status=500, body={"error": {"message": "boom"}})]
    service = make_ai_service(
        settings, session, providers=("groq", "cerebras", "gemini")
    )
    result = await service.generate(system_prompt="s", user_prompt="u")

    assert result.ok is False
    assert result.text == ""
    assert len(result.attempts) > 0
    assert "groq" in result.error and "cerebras" in result.error
    assert service.stats.failures == 1


async def test_gemini_safety_block_fails_over_without_retry(settings, session):
    blocked = FakeResponse(
        status=200,
        body={"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []},
    )
    session._responses = [blocked, openai_ok("cerebras answered")]
    service = make_ai_service(settings, session, providers=("gemini", "cerebras"))
    result = await service.generate(system_prompt="s", user_prompt="u")

    assert result.ok and result.provider == "cerebras"
    assert session.call_count == 2


# --------------------------------------------------------------------------- #
# guards: circuit breaker, budget, deadline
# --------------------------------------------------------------------------- #
async def test_open_circuit_skips_provider(settings, session):
    session._responses = [FakeResponse(status=500, body={"error": {"message": "boom"}})]
    settings = _replace(settings, max_retries=0, circuit_breaker_threshold=1)
    service = make_ai_service(settings, session, providers=("groq",))

    first = await service.generate(system_prompt="s", user_prompt="u")
    assert first.ok is False
    assert service.health()[0]["circuit"] == "open"

    calls_before = session.call_count
    second = await service.generate(system_prompt="s", user_prompt="u")
    assert second.ok is False
    assert session.call_count == calls_before, "an open circuit must not hit the API"
    assert service.stats.skipped_breaker == 1
    assert "circuit open" in second.error


async def test_circuit_recovers_after_cooldown(settings, session):
    clock = {"now": 1000.0}
    session._responses = [FakeResponse(status=500, body={"error": {"message": "boom"}})]
    settings = _replace(settings, max_retries=0, circuit_breaker_threshold=1,
                        circuit_breaker_cooldown=30.0)
    service = make_ai_service(settings, session, providers=("groq",))
    service._states[0].breaker = CircuitBreaker(1, 30.0, clock=lambda: clock["now"])

    await service.generate(system_prompt="s", user_prompt="u")
    assert service._states[0].breaker.state is CircuitState.OPEN

    clock["now"] += 31.0
    assert service._states[0].breaker.state is CircuitState.HALF_OPEN

    session._responses = [openai_ok("recovered")]
    result = await service.generate(system_prompt="s", user_prompt="u")
    assert result.ok and result.text == "recovered"
    assert service._states[0].breaker.state is CircuitState.CLOSED


async def test_daily_budget_blocks_provider(settings, session):
    session._responses = [openai_ok("first"), openai_ok("should not be used")]
    from bot.config import ProviderSettings

    limited = ProviderSettings("groq", "k", "m", 6000, 1, 5.0)
    settings = _replace(settings, groq=limited)
    service = make_ai_service(settings, session, providers=("groq",))

    assert (await service.generate(system_prompt="s", user_prompt="u")).ok is True
    second = await service.generate(system_prompt="s", user_prompt="u")
    assert second.ok is False
    assert service.stats.skipped_budget == 1
    assert session.call_count == 1


async def test_budget_fails_over_to_next_provider(settings, session):
    from bot.config import ProviderSettings

    session._responses = [openai_ok("groq"), openai_ok("cerebras")]
    settings = _replace(settings, groq=ProviderSettings("groq", "k", "m", 6000, 1, 5.0))
    service = make_ai_service(settings, session, providers=("groq", "cerebras"))

    await service.generate(system_prompt="s", user_prompt="u")
    second = await service.generate(system_prompt="s", user_prompt="u")
    assert second.ok and second.provider == "cerebras"


async def test_total_deadline_stops_the_attempt_loop(settings, session):
    """An exhausted deadline must stop work instead of answering minutes late."""
    session._responses = [openai_ok("late answer")]
    settings = _replace(settings, total_deadline=0.2, max_retries=5)
    service = make_ai_service(settings, session, providers=("groq", "cerebras"))
    result = await service.generate(system_prompt="s", user_prompt="u")

    assert result.ok is False
    assert "deadline" in result.error
    assert session.call_count == 0


async def test_deadline_bounds_a_slow_provider(settings):
    """A provider slower than the remaining deadline is abandoned."""
    session._responses = [_Delayed(openai_ok("late answer"), delay=1.5)]
    settings = _replace(settings, total_deadline=1.0, max_retries=0)
    service = make_ai_service(settings, session, providers=("groq",))
    result = await service.generate(system_prompt="s", user_prompt="u")

    assert result.ok is False


async def test_no_providers_configured(settings, session):
    service = AIService([], settings)
    result = await service.generate(system_prompt="s", user_prompt="u")
    assert result.ok is False
    assert "no AI provider" in result.error


# --------------------------------------------------------------------------- #
# post-processing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("plain answer", "plain answer"),
        ("  padded answer  ", "padded answer"),
        ("```markdown\nfenced\n```", "fenced"),
        ("AI Agent Response: here you go", "here you go"),
        ("Assistant: hi", "hi"),
        ("ping <@&123> now", "ping  now"),
        ("@everyone look", "@\u200beveryone look"),
        ("", ""),
    ],
)
def test_postprocess_completion(raw, expected):
    assert postprocess_completion(raw) == expected.strip()


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
async def test_token_bucket_limits_burst():
    bucket = TokenBucket(rate_per_minute=60, burst=2)  # 1 token/second
    assert await bucket.acquire(timeout=0.01)
    assert await bucket.acquire(timeout=0.01)
    assert await bucket.acquire(timeout=0.01) is False, "burst exhausted"
    assert bucket.wait_time() > 0


async def test_token_bucket_refills_over_time():
    bucket = TokenBucket(rate_per_minute=600, burst=1)  # 10 tokens/second
    await bucket.acquire(timeout=0.01)
    assert await bucket.acquire(timeout=0.01) is False
    await asyncio.sleep(0.15)
    assert await bucket.acquire(timeout=0.01) is True


async def test_daily_budget_resets_on_new_utc_day():
    clock = {"now": 1_700_000_000.0}
    budget = DailyBudget(2, clock=lambda: clock["now"])
    assert await budget.try_consume()
    assert await budget.try_consume()
    assert await budget.try_consume() is False
    assert budget.remaining == 0

    clock["now"] += 86_400  # next UTC day
    assert await budget.try_consume() is True
    assert budget.used == 1


async def test_unlimited_budget_when_limit_is_zero():
    budget = DailyBudget(0)
    for _ in range(5):
        assert await budget.try_consume()
    assert budget.remaining == -1


def test_circuit_breaker_transitions():
    clock = {"now": 0.0}
    breaker = CircuitBreaker(threshold=2, cooldown=10.0, clock=lambda: clock["now"])
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow_request()

    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.allow_request() is False

    clock["now"] += 11
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow_request() is True, "one probe is permitted"
    assert breaker.allow_request() is False, "but only one at a time"

    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


def test_circuit_breaker_release_frees_a_probe():
    clock = {"now": 0.0}
    breaker = CircuitBreaker(threshold=1, cooldown=10.0, clock=lambda: clock["now"])
    breaker.record_failure()
    clock["now"] += 11
    assert breaker.allow_request() is True
    assert breaker.allow_request() is False
    breaker.release()
    assert breaker.allow_request() is True


# --------------------------------------------------------------------------- #
# error classification + factory
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "status,retryable",
    [(400, False), (401, False), (403, False), (404, False), (422, False),
     (408, True), (429, True), (500, True), (502, True), (503, True), (504, True)],
)
def test_classify_http_error(status, retryable):
    error = classify_http_error("groq", status, '{"error":{"message":"m"}}', {})
    assert isinstance(error, ProviderError)
    assert error.retryable is retryable
    assert error.status == status


def test_classify_http_error_extracts_retry_after():
    error = classify_http_error("groq", 429, "", {"Retry-After": "2.5"})
    assert error.retry_after == 2.5


def test_classify_http_error_reads_provider_message():
    body = '{"error":{"message":"Model not found","code":"model_not_found"}}'
    assert "model_not_found" in classify_http_error("groq", 404, body, {}).detail


def test_extract_completion_text_handles_missing_choices():
    assert extract_completion_text({}) == ("", None)
    assert extract_completion_text({"choices": []}) == ("", None)
    text, reason = extract_completion_text(
        {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
    )
    assert (text, reason) == ("hi", "stop")


def test_build_providers_skips_missing_keys(settings):
    session = FakeSession([])
    from bot.config import ProviderSettings

    settings = _replace(
        settings,
        provider_chain=("groq", "cerebras", "gemini"),
        cerebras=ProviderSettings("cerebras", "", "m", 60, 100, 5.0),  # no key
    )
    providers = build_providers(settings, session)  # type: ignore[arg-type]
    assert [p.name for p in providers] == ["groq", "gemini"]
    assert all(isinstance(p, (GroqProvider, GeminiProvider)) for p in providers)
    assert not any(isinstance(p, CerebrasProvider) for p in providers)


def test_build_providers_dedupes_chain(settings):
    session = FakeSession([])
    from bot.config import Settings as _S

    settings = _replace(settings, provider_chain=("groq", "groq"))
    providers = build_providers(settings, session)  # type: ignore[arg-type]
    assert [p.name for p in providers] == ["groq"]
    assert _S is not None


def _replace(settings, **kwargs):
    from dataclasses import replace as _dc_replace

    return _dc_replace(settings, **kwargs)


class _Delayed(FakeResponse):
    def __init__(self, inner: FakeResponse, delay: float) -> None:
        super().__init__(status=inner.status, body=inner.body, headers=inner.headers, delay=delay)
