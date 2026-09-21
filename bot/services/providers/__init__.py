"""LLM provider implementations (Groq, Cerebras, Gemini) over plain REST."""

from __future__ import annotations

from bot.services.providers.base import (
    LLMProvider,
    ProviderError,
    ProviderResponse,
    classify_http_error,
)
from bot.services.providers.factory import build_providers
from bot.services.providers.gemini import GeminiProvider
from bot.services.providers.openai_compatible import (
    CerebrasProvider,
    GroqProvider,
    OpenAICompatibleProvider,
)

PROVIDER_CLASSES: dict[str, type[LLMProvider]] = {
    "groq": GroqProvider,
    "cerebras": CerebrasProvider,
    "gemini": GeminiProvider,
}

__all__ = [
    "PROVIDER_CLASSES",
    "CerebrasProvider",
    "GeminiProvider",
    "GroqProvider",
    "LLMProvider",
    "OpenAICompatibleProvider",
    "ProviderError",
    "ProviderResponse",
    "build_providers",
    "classify_http_error",
]
