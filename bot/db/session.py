"""Async engine + session management (SQLite for dev, PostgreSQL for prod).

The same code path serves both because only ``DATABASE_URL`` changes:

* ``sqlite+aiosqlite:///data/tickets.db``  – zero-setup local development
* ``postgresql+asyncpg://…``               – production / Supabase

SQLite is configured with WAL and enforced foreign keys so multi-tenant writes
from concurrent ticket tasks do not lock each other out.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urlparse

from sqlalchemy import event, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from bot.db.models import Base
from bot.utils.logging_setup import get_logger

log = get_logger(__name__)


def _prepare_sqlite_path(url_str: str) -> None:
    """Create the parent directory for a file-backed SQLite database."""
    parsed = urlparse(url_str)
    # sqlite+aiosqlite:///data/tickets.db -> path == "data/tickets.db"
    path = (parsed.path or "").lstrip("/")
    if parsed.netloc:  # absolute Windows-ish path or host component
        path = f"{parsed.netloc}/{path}" if path else parsed.netloc
    if not path or path == ":memory:":
        return
    parent = Path(path).parent
    if str(parent) not in {"", "."}:
        parent.mkdir(parents=True, exist_ok=True)


def build_engine(url_str: str, *, echo: bool = False, pool_size: int = 10) -> AsyncEngine:
    """Create an async engine tuned for the backing database."""
    url = make_url(url_str)
    is_sqlite = url.get_backend_name() == "sqlite"

    kwargs: dict[str, object] = {"echo": echo, "future": True, "pool_pre_ping": True}

    if is_sqlite:
        _prepare_sqlite_path(url_str)
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        if ":memory:" in url_str or url.database in (None, "", ":memory:"):
            # A single shared connection is required for in-memory databases.
            kwargs["poolclass"] = StaticPool
    else:
        kwargs.update(
            {
                "pool_size": pool_size,
                "max_overflow": pool_size,
                "pool_recycle": 1800,
                "pool_timeout": 30,
            }
        )

    engine = create_async_engine(url, **kwargs)

    if is_sqlite:

        @event.listens_for(engine.sync_engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - driver hook
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

    return engine


class Database:
    """Owns the engine and session factory for the process lifetime."""

    def __init__(self, url: str, *, echo: bool = False, pool_size: int = 10) -> None:
        self.url = url
        self.engine: AsyncEngine = build_engine(url, echo=echo, pool_size=pool_size)
        self.session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            class_=AsyncSession,
            autoflush=False,
        )
        self._ready = asyncio.Event()

    @property
    def dialect(self) -> str:
        return self.engine.dialect.name

    async def initialize(self) -> None:
        """Create any missing tables. Idempotent and safe on every boot."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await self._sanity_check()
        self._ready.set()
        log.info("Database ready (dialect=%s, url=%s)", self.dialect, _safe_url(self.url))

    async def _sanity_check(self) -> None:
        async with self.session_factory() as session:
            await session.execute(text("SELECT 1"))

    async def wait_until_ready(self, timeout: float | None = None) -> None:
        await asyncio.wait_for(self._ready.wait(), timeout=timeout)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Transaction-per-block session scope with automatic rollback."""
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    @asynccontextmanager
    async def read_session(self) -> AsyncIterator[AsyncSession]:
        """Session scope for read-only work (no commit overhead)."""
        session = self.session_factory()
        try:
            yield session
        finally:
            await session.close()

    async def close(self) -> None:
        await self.engine.dispose()
        log.info("Database connections closed.")


def _safe_url(url: str) -> str:
    """Mask credentials when logging a database URL."""
    try:
        parsed = make_url(url)
        return str(parsed.render_as_string(hide_password=True))
    except Exception:  # pragma: no cover - defensive
        return "<unparseable database url>"
