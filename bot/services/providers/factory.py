"""Build the ordered provider chain from :class:`bot.config.Settings`."""

from __future__ import annotations

import aiohttp

from bot.config import Settings
from bot.services.providers.base import LLMProvider
from bot.services.providers.gemini import GeminiProvider
from bot.services.providers.openai_compatible import CerebrasProvider, GroqProvider
from bot.utils.logging_setup import get_logger

log = get_logger(__name__)

_REGISTRY: dict[str, type[LLMProvider]] = {
    GroqProvider.name: GroqProvider,
    CerebrasProvider.name: CerebrasProvider,
    GeminiProvider.name: GeminiProvider,
}


def build_providers(settings: Settings, session: aiohttp.ClientSession) -> list[LLMProvider]:
    """Instantiate one provider per chain entry that has an API key.

    Order is preserved so the AI service tries the primary backend first and
    only falls through on rate limits or outages.
    """
    providers: list[LLMProvider] = []
    seen: set[str] = set()

    for name in settings.provider_chain:
        if name in seen:
            log.warning("Provider %r appears twice in AI_PROVIDER_CHAIN; ignoring duplicate.", name)
            continue
        seen.add(name)

        provider_settings = settings.provider(name)
        provider_cls = _REGISTRY.get(name)
        if provider_cls is None or provider_settings is None:
            log.warning("Unknown provider %r in AI_PROVIDER_CHAIN; skipping.", name)
            continue
        if not provider_settings.available:
            log.info(
                "Provider %r is in the chain but %s is not set; it will be skipped.",
                name,
                f"{name.upper()}_API_KEY",
            )
            continue

        providers.append(
            provider_cls(
                provider_settings,
                session,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
        )

    if not providers:
        log.error("No usable AI provider: every entry in AI_PROVIDER_CHAIN is missing an API key.")
    return providers
