"""Google Gemini provider (``generateContent`` REST API) — the fallback backend.

Gemini does not speak the OpenAI chat schema: the system prompt goes in
``system_instruction``, the user turn in ``contents``, and the text comes back
as a list of ``parts``. Safety blocks are surfaced as a retryable error so the
caller can fail over rather than posting nothing.
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
    wrap_transport_error,
)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(LLMProvider):
    """``gemini-2.0-flash`` via the Generative Language REST API."""

    name: ClassVar[str] = "gemini"
    base_url: ClassVar[str] = GEMINI_BASE_URL

    def _headers(self) -> dict[str, str]:
        return {
            # Preferred over the ?key= query param: keeps the key out of URL logs.
            "x-goog-api-key": self.settings.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "discord-ai-ticket-responder/1.0",
        }

    def _payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "candidateCount": 1,
                "topP": 0.9,
            },
        }

    def _url(self) -> str:
        return f"{self.base_url.rstrip('/')}/models/{self.model}:generateContent"

    async def complete(self, *, system_prompt: str, user_prompt: str) -> ProviderResponse:
        if not self.available:
            raise ProviderError(
                "gemini API key is not configured", provider=self.name, retryable=False
            )

        timeout = aiohttp.ClientTimeout(total=self.settings.timeout)
        started = time.perf_counter()

        try:
            async with self.session.post(
                self._url(),
                json=self._payload(system_prompt, user_prompt),
                headers=self._headers(),
                timeout=timeout,
            ) as response:
                body_text = await response.text()
                if response.status != 200:
                    raise classify_http_error(self.name, response.status, body_text, response.headers)
                try:
                    data = json.loads(body_text) if body_text else {}
                except ValueError:
                    raise ProviderError(
                        "gemini returned a non-JSON body",
                        retryable=True,
                        status=response.status,
                        provider=self.name,
                        detail=body_text[:200],
                    ) from None
        except ProviderError:
            raise
        except Exception as exc:
            raise wrap_transport_error(self.name, exc) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        text_value, finish_reason, block_reason = self._extract(data)

        if block_reason:
            # Prompt-level safety block. Retrying the identical prompt will not
            # help, so this is fatal for Gemini — the caller fails over.
            raise ProviderError(
                f"prompt blocked by safety filters ({block_reason})",
                retryable=False,
                status=200,
                provider=self.name,
                detail=block_reason,
            )
        if not text_value.strip():
            raise ProviderError(
                "gemini returned an empty completion",
                retryable=True,
                status=200,
                provider=self.name,
                detail=f"finish_reason={finish_reason}",
            )

        return ProviderResponse(
            text=text_value,
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
            prompt_chars=len(system_prompt) + len(user_prompt),
            completion_chars=len(text_value),
            finish_reason=finish_reason,
            raw=data,
        )

    @staticmethod
    def _extract(data: dict[str, Any]) -> tuple[str, str | None, str | None]:
        """Return ``(text, finish_reason, block_reason)`` from a Gemini response."""
        prompt_feedback = data.get("promptFeedback") or {}
        block_reason = prompt_feedback.get("blockReason") or prompt_feedback.get("block_reason")

        candidates = data.get("candidates") or []
        if not candidates:
            return "", None, block_reason if isinstance(block_reason, str) else "no_candidates"

        candidate = candidates[0] if isinstance(candidates[0], dict) else {}
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        text_value = "".join(
            part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")
        )
        finish_reason = candidate.get("finishReason") or candidate.get("finish_reason")
        if not text_value and not block_reason and candidate.get("safetyRatings"):
            block_reason = "safety_ratings"
        return (
            text_value,
            finish_reason if isinstance(finish_reason, str) else None,
            block_reason if isinstance(block_reason, str) else None,
        )
