"""Provider contract, error taxonomy and shared HTTP helpers."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

import aiohttp

from bot.config import ProviderSettings
from bot.utils.logging_setup import get_logger

log = get_logger(__name__)


class ProviderError(Exception):
    """A provider-level failure with an explicit retry verdict.

    ``retryable`` is the single source of truth for the retry/failover logic in
    :class:`bot.services.ai_service.AIService`: 429/5xx/network/timeout are
    retryable, while 401/403/404/400 are configuration problems that would fail
    identically forever — those must fail over to the next provider instead.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status: int | None = None,
        retry_after: float | None = None,
        provider: str = "",
        detail: str = "",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.status = status
        self.retry_after = retry_after
        self.provider = provider
        self.detail = detail

    def __str__(self) -> str:  # pragma: no cover - logging aid
        base = f"[{self.provider or 'provider'}] {self.message}"
        if self.status:
            base += f" (HTTP {self.status})"
        if self.retry_after is not None:
            base += f" retry_after={self.retry_after:.1f}s"
        return base


@dataclass(frozen=True)
class ProviderResponse:
    """A successful completion, plus the telemetry we persist in ``llm_usage``."""

    text: str
    provider: str
    model: str
    latency_ms: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    finish_reason: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def truncated(self) -> bool:
        return (self.finish_reason or "").lower() in {"length", "max_tokens"}

    @property
    def ok(self) -> bool:
        return bool(self.text and self.text.strip())


class LLMProvider(ABC):
    """Minimal contract every inference backend implements."""

    name: ClassVar[str] = "base"
    base_url: ClassVar[str] = ""

    def __init__(
        self,
        settings: ProviderSettings,
        session: aiohttp.ClientSession,
        *,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> None:
        self.settings = settings
        self.session = session
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self.settings.model

    @property
    def available(self) -> bool:
        return self.settings.available

    @abstractmethod
    async def complete(self, *, system_prompt: str, user_prompt: str) -> ProviderResponse:
        """Run one chat completion. Must raise :class:`ProviderError` on failure."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model={self.model!r} available={self.available}>"


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #

#: Statuses that mean "try again later".
RETRYABLE_STATUSES: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

#: Statuses that mean "this provider is misconfigured; skip to the next one".
FATAL_STATUSES: frozenset[int] = frozenset({400, 401, 402, 403, 404, 405, 422})


def _parse_retry_after(headers: Mapping[str, str]) -> float | None:
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _error_detail(body_text: str, limit: int = 500) -> str:
    """Extract a human-readable reason from a JSON error body when possible."""
    if not body_text:
        return ""
    try:
        payload = json.loads(body_text)
    except (ValueError, TypeError):
        return body_text[:limit]
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code") or ""
            code = error.get("code") or error.get("type") or ""
            return f"{code}: {message}".strip(": ")[:limit] or json.dumps(error)[:limit]
        if isinstance(error, str):
            return error[:limit]
        for key in ("message", "msg", "detail", "reason"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value[:limit]
    return body_text[:limit]


def classify_http_error(
    provider: str, status: int, body_text: str, headers: Mapping[str, str]
) -> ProviderError:
    """Turn an HTTP status into a :class:`ProviderError` with a retry verdict."""
    detail = _error_detail(body_text)
    retryable = status in RETRYABLE_STATUSES or status >= 500
    message = {
        400: "malformed request rejected by provider",
        401: "invalid API key",
        402: "billing/credits exhausted",
        403: "API key not permitted for this model or region",
        404: "model not found — check the configured model name",
        429: "rate limited by provider",
    }.get(status, f"provider returned HTTP {status}")
    return ProviderError(
        message,
        retryable=retryable,
        status=status,
        retry_after=_parse_retry_after(headers) if status == 429 else None,
        provider=provider,
        detail=detail,
    )


def wrap_transport_error(provider: str, exc: Exception) -> ProviderError:
    """Normalise network/timeout failures into retryable provider errors."""
    if isinstance(exc, asyncio.TimeoutError):
        return ProviderError("request timed out", retryable=True, provider=provider)
    if isinstance(exc, aiohttp.ClientResponseError):
        return classify_http_error(provider, exc.status, str(exc.message), exc.headers or {})
    if isinstance(exc, aiohttp.ClientError):
        return ProviderError(
            f"network error: {type(exc).__name__}", retryable=True, provider=provider, detail=str(exc)
        )
    return ProviderError(
        f"unexpected error: {type(exc).__name__}", retryable=False, provider=provider, detail=str(exc)
    )


def extract_completion_text(data: Mapping[str, Any]) -> tuple[str, str | None]:
    """Pull ``choices[0].message.content`` out of an OpenAI-compatible payload."""
    choices = data.get("choices") or []
    if not choices:
        return "", None
    first = choices[0] if isinstance(choices[0], Mapping) else {}
    message = first.get("message") or {}
    content = message.get("content")
    if content is None:  # some gateways return tool-style part arrays
        parts = message.get("parts") or []
        content = "".join(
            part.get("text", "") for part in parts if isinstance(part, Mapping)
        )
    text_value = content if isinstance(content, str) else str(content or "")
    finish_reason = first.get("finish_reason")
    return text_value, (finish_reason if isinstance(finish_reason, str) else None)
