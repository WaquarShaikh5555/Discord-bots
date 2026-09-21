"""OpenAI-compatible ``/chat/completions`` providers: Groq and Cerebras.

Both expose the identical request/response schema, so a single implementation
covers them — only the base URL, default model and free-tier ceilings differ.
Requests go out over the bot's shared :mod:`aiohttp` session (no extra SDK),
which keeps the dependency surface small and makes the provider trivially
mockable in tests.
"""

from __future__ import annotations

import json
import time
from typing import Any, ClassVar

import aiohttp

from bot.services.providers.base import (
    LLMProvider,
    ProviderError,
    ProviderResponse,
    classify_http_error,
    extract_completion_text,
    wrap_transport_error,
)
from bot.utils.logging_setup import get_logger

log = get_logger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Base class for chat-completions style inference APIs."""

    name: ClassVar[str] = "openai_compatible"
    base_url: ClassVar[str] = ""
    #: ``max_tokens`` is deprecated by some gateways in favour of this name.
    token_limit_field: ClassVar[str] = "max_tokens"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "discord-ai-ticket-responder/1.0",
        }

    def _payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            self.token_limit_field: self.max_tokens,
            "stream": False,
            # Keep support answers deterministic; sampling noise hurts KB fidelity.
            "top_p": 0.9,
        }

    async def complete(self, *, system_prompt: str, user_prompt: str) -> ProviderResponse:
        if not self.available:
            raise ProviderError(
                f"{self.name} API key is not configured", provider=self.name, retryable=False
            )
        if not self.base_url:  # pragma: no cover - programming guard
            raise ProviderError(f"{self.name} has no base_url", provider=self.name, retryable=False)

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = self._payload(system_prompt, user_prompt)
        timeout = aiohttp.ClientTimeout(total=self.settings.timeout)
        started = time.perf_counter()

        try:
            async with self.session.post(
                url, json=payload, headers=self._headers(), timeout=timeout
            ) as response:
                body_text = await response.text()
                if response.status != 200:
                    raise classify_http_error(self.name, response.status, body_text, response.headers)
                try:
                    data = json.loads(body_text) if body_text else {}
                except ValueError:
                    raise ProviderError(
                        "provider returned a non-JSON body",
                        retryable=True,
                        status=response.status,
                        provider=self.name,
                        detail=body_text[:200],
                    ) from None
        except ProviderError:
            raise
        except Exception as exc:  # network, timeout, TLS, DNS
            raise wrap_transport_error(self.name, exc) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        text_value, finish_reason = extract_completion_text(data)

        if not text_value.strip():
            # An empty completion is useless to a ticket; treat it as retryable so
            # the caller can fail over instead of posting a blank message.
            raise ProviderError(
                "provider returned an empty completion",
                retryable=True,
                status=200,
                provider=self.name,
                detail=f"finish_reason={finish_reason}",
            )

        return ProviderResponse(
            text=text_value,
            provider=self.name,
            model=data.get("model") or self.model,
            latency_ms=latency_ms,
            prompt_chars=len(system_prompt) + len(user_prompt),
            completion_chars=len(text_value),
            finish_reason=finish_reason,
            raw=data,
        )


class GroqProvider(OpenAICompatibleProvider):
    """Groq LPU inference — the primary free high-volume backend (14,400 req/day)."""

    name: ClassVar[str] = "groq"
    base_url: ClassVar[str] = "https://api.groq.com/openai/v1"


class CerebrasProvider(OpenAICompatibleProvider):
    """Cerebras inference — sub-second latency, used as the second primary."""

    name: ClassVar[str] = "cerebras"
    base_url: ClassVar[str] = "https://api.cerebras.ai/v1"
    #: Cerebras validates this field name for output limits.
    token_limit_field: ClassVar[str] = "max_completion_tokens"
