"""Shared pytest fixtures: an isolated in-memory database and a stubbed AI stack."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Repo root as well, so `bot` and `scripts` import regardless of how pytest
# is invoked (bare `pytest` does not add the cwd the way `python -m pytest` does).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.config import ProviderSettings, Settings  # noqa: E402
from bot.db.repository import TicketRepository  # noqa: E402
from bot.db.session import Database  # noqa: E402
from bot.services.ai_service import AIService  # noqa: E402
from bot.services.providers import CerebrasProvider, GeminiProvider, GroqProvider  # noqa: E402
from fakes import (  # noqa: E402
    BOT_USER_ID,
    CATEGORY_ID,
    GUILD_ID,
    STAFF_ROLE_ID,
    FakeBot,
    FakeSession,
    FakeUser,
)

KNOWLEDGE_BASE = """## Refund Policy
Refunds are available within 14 days of purchase for unused credits.
To request one, open a ticket and include your order ID.

## Server Rules
1. Be respectful. No harassment or hate speech.
2. No advertising without staff approval.
3. Keep support questions in ticket channels.

## FAQ
Q: How do I get the Member role?
A: React in #roles with the Member emoji.

Q: What are the support hours?
A: Staff are available 09:00-18:00 UTC on weekdays."""


@pytest.fixture
def settings() -> Settings:
    """A fully populated Settings object with fast test-friendly timings."""
    groq = ProviderSettings("groq", "test-groq-key", "llama-3.3-70b-versatile", 6000, 14_400, 5.0)
    cerebras = ProviderSettings("cerebras", "test-cerebras-key", "llama3.3-70b", 6000, 14_400, 5.0)
    gemini = ProviderSettings("gemini", "test-gemini-key", "gemini-2.0-flash", 6000, 1_500, 5.0)
    return Settings(
        discord_token="test-token",
        database_url="sqlite+aiosqlite:///:memory:",
        provider_chain=("groq", "cerebras", "gemini"),
        groq=groq,
        cerebras=cerebras,
        gemini=gemini,
        temperature=0.2,
        max_tokens=400,
        max_retries=2,
        backoff_base=0.01,
        backoff_cap=0.05,
        total_deadline=10.0,
        circuit_breaker_threshold=3,
        circuit_breaker_cooldown=30.0,
        history_message_limit=10,
        history_char_budget=4000,
        knowledge_base_char_limit=40_000,
        require_configured_category=True,
        user_cooldown_seconds=0.0,
        max_concurrent_tickets=5,
        escalation_cooldown_seconds=0.0,
        escalation_preflight_safety=True,
        escalation_preflight_requests=False,
        config_cache_ttl=0.0,  # keep tests deterministic: always read through
    )


@pytest.fixture
async def database(settings: Settings) -> AsyncIterator[Database]:
    db = Database(settings.database_url)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


@pytest.fixture
async def repo(database: Database, settings: Settings) -> AsyncIterator[TicketRepository]:
    yield TicketRepository(database, cache_ttl=settings.config_cache_ttl)


@pytest.fixture
async def configured_repo(repo: TicketRepository) -> TicketRepository:
    """A repository whose test guild is fully configured."""
    await repo.ensure_guild(GUILD_ID, "Test Server")
    await repo.set_knowledge_base(GUILD_ID, KNOWLEDGE_BASE, server_name="Test Server")
    await repo.set_staff_role(GUILD_ID, STAFF_ROLE_ID)
    await repo.set_ticket_category(GUILD_ID, CATEGORY_ID)
    return repo


def make_ai_service(
    settings: Settings,
    session: FakeSession,
    *,
    providers: tuple[str, ...] = ("groq",),
    sleep: Callable[[float], Any] | None = None,
) -> AIService:
    """Build an AIService whose providers all talk to a fake HTTP session."""
    classes = {"groq": GroqProvider, "cerebras": CerebrasProvider, "gemini": GeminiProvider}
    instances = [
        classes[name](
            getattr(settings, name),
            session,  # type: ignore[arg-type]
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        for name in providers
    ]
    return AIService(
        instances,
        settings,
        sleep=sleep or (lambda _delay: asyncio.sleep(0)),
    )


@pytest.fixture
def fake_bot_user() -> FakeUser:
    return FakeUser(id=BOT_USER_ID, name="TicketBot", bot=True)


@pytest.fixture
def bot_factory(settings: Settings) -> Callable[..., FakeBot]:
    def _make(repo: TicketRepository, session: FakeSession, **kwargs: Any) -> FakeBot:
        ai = make_ai_service(settings, session, **{k: v for k, v in kwargs.items()
                                                   if k in {"providers", "sleep"}})
        return FakeBot(settings=settings, repository=repo, ai_service=ai)

    return _make
