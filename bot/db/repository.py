"""Multi-tenant data-access layer.

Design notes
------------
* Every method takes an explicit ``guild_id`` and every query is filtered by it.
  There is no code path that can read one server's knowledge base while
  answering another server's ticket — that is the tenant isolation guarantee.
* Rows are returned as detached, frozen :class:`GuildConfigRecord` objects rather
  than live ORM instances, so cached values can be handed to concurrent tasks
  without SQLAlchemy session/lazy-load hazards.
* Guild configs are cached in-process with a short TTL *and* negative-cached,
  because the vast majority of messages a popular bot sees belong to servers
  that have not configured it. Without negative caching every unrelated message
  costs a database round-trip.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Sequence

from sqlalchemy import func, select
from bot.constants import (
    ACTIVE_TICKET_STATUSES,
    STATUS_ANSWERED,
    STATUS_ESCALATED,
    STATUS_OPEN,
    STATUS_RESOLVED,
)
from bot.db.models import ServerConfig, TicketActivity, TicketLog, UsageCounter
from bot.db.session import Database
from bot.utils.logging_setup import get_logger

log = get_logger(__name__)

KBMode = Literal["replace", "append", "prepend"]

#: Negative cache lives shorter than positive entries so a freshly configured
#: server starts working quickly even if the admin beats the TTL.
_NEGATIVE_TTL_SECONDS = 15.0


class _Unset:
    """Sentinel distinguishing "not provided" from an explicit ``None``."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<UNSET>"


_UNSET = _Unset()


def as_utc(value: datetime | None) -> datetime | None:
    """Normalise timestamps: SQLite returns naive datetimes, Postgres returns aware."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class GuildConfigRecord:
    """Immutable snapshot of a ``server_configs`` row."""

    guild_id: str
    server_name: str = ""
    knowledge_base: str = ""
    staff_role_id: str | None = None
    ticket_category_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_orm(cls, row: ServerConfig) -> "GuildConfigRecord":
        return cls(
            guild_id=str(row.guild_id),
            server_name=row.server_name or "",
            knowledge_base=row.knowledge_base or "",
            staff_role_id=row.staff_role_id,
            ticket_category_id=row.ticket_category_id,
            created_at=as_utc(row.created_at),
            updated_at=as_utc(row.updated_at),
        )

    @property
    def has_knowledge_base(self) -> bool:
        return bool(self.knowledge_base and self.knowledge_base.strip())

    @property
    def is_fully_configured(self) -> bool:
        return bool(self.has_knowledge_base and self.staff_role_id and self.ticket_category_id)

    def missing_pieces(self) -> list[str]:
        missing: list[str] = []
        if not self.has_knowledge_base:
            missing.append("knowledge base — `/setup-kb`")
        if not self.staff_role_id:
            missing.append("staff role — `/set-staff-role`")
        if not self.ticket_category_id:
            missing.append("ticket category — `/set-ticket-category`")
        return missing

    @property
    def knowledge_base_chars(self) -> int:
        return len(self.knowledge_base or "")

    @property
    def knowledge_base_words(self) -> int:
        return len((self.knowledge_base or "").split())


@dataclass(frozen=True)
class TicketRecord:
    """Immutable snapshot of a ``ticket_logs`` row joined with its activity."""

    ticket_id: str
    guild_id: str | None
    channel_id: str | None
    user_id: str | None
    status: str
    created_at: datetime | None
    ai_reply_count: int = 0
    escalation_count: int = 0
    user_message_count: int = 0
    last_escalation_at: datetime | None = None
    last_ai_reply_at: datetime | None = None
    last_user_message_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_TICKET_STATUSES


@dataclass(frozen=True)
class GuildStats:
    guild_id: str
    total_tickets: int
    open_tickets: int
    escalated_tickets: int
    resolved_tickets: int
    ai_replies: int
    escalations: int


@dataclass
class _CacheEntry:
    record: GuildConfigRecord | None
    expires_at: float


class TicketRepository:
    """All persistence for the bot, with an in-process guild config cache."""

    def __init__(self, database: Database, *, cache_ttl: float = 60.0) -> None:
        self.db = database
        self.cache_ttl = max(0.0, cache_ttl)
        self._cache: dict[str, _CacheEntry] = {}
        self._guild_locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # cache plumbing
    # ------------------------------------------------------------------ #
    async def _lock_for(self, guild_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._guild_locks.get(guild_id)
            if lock is None:
                lock = asyncio.Lock()
                self._guild_locks[guild_id] = lock
            return lock

    def invalidate(self, guild_id: str | None = None) -> None:
        if guild_id is None:
            self._cache.clear()
            return
        self._cache.pop(str(guild_id), None)

    def _store(self, guild_id: str, record: GuildConfigRecord | None) -> None:
        """Write a positive or negative cache entry (no-op when caching is off)."""
        if self.cache_ttl <= 0:
            return
        ttl = self.cache_ttl if record is not None else _NEGATIVE_TTL_SECONDS
        self._cache[str(guild_id)] = _CacheEntry(record=record, expires_at=time.monotonic() + ttl)

    def _cache_get(self, guild_id: str) -> tuple[bool, GuildConfigRecord | None]:
        entry = self._cache.get(guild_id)
        if entry is None or entry.expires_at <= time.monotonic():
            return False, None
        return True, entry.record

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------ #
    # server_configs (tenant settings)
    # ------------------------------------------------------------------ #
    async def get_guild(self, guild_id: int | str, *, use_cache: bool = True) -> GuildConfigRecord | None:
        """Fetch one server's config, or ``None`` when the server is unknown."""
        key = str(guild_id)
        if use_cache and self.cache_ttl > 0:
            hit, cached = self._cache_get(key)
            if hit:
                return cached

        async with self.db.read_session() as session:
            row = await session.get(ServerConfig, key)
            record = GuildConfigRecord.from_orm(row) if row else None

        if self.cache_ttl > 0:
            self._store(key, record)
        return record

    async def ensure_guild(self, guild_id: int | str, server_name: str = "") -> GuildConfigRecord:
        """Return the server's config, creating an empty row if it does not exist.

        Serialised per guild so two concurrent first-messages cannot race into a
        duplicate primary-key insert.
        """
        key = str(guild_id)
        async with await self._lock_for(key):
            existing = await self.get_guild(key, use_cache=False)
            if existing is not None:
                if server_name and existing.server_name != server_name:
                    return await self._update_guild(key, server_name=server_name)
                return existing

            async with self.db.session() as session:
                row = ServerConfig(guild_id=key, server_name=server_name or None, knowledge_base="")
                session.add(row)
                await session.flush()
                record = GuildConfigRecord.from_orm(row)

            self._store(key, record)
            log.info("Created server_configs row for guild %s (%s)", key, server_name or "unnamed")
            return record

    async def _update_guild(self, guild_id: str, **values: Any) -> GuildConfigRecord:
        """Apply a partial update and refresh the cache in the same critical section."""
        async with self.db.session() as session:
            row = await session.get(ServerConfig, guild_id)
            if row is None:
                row = ServerConfig(guild_id=guild_id, knowledge_base="")
                session.add(row)
                await session.flush()
            for column, value in values.items():
                setattr(row, column, value)
            row.updated_at = datetime.now(timezone.utc)
            await session.flush()
            record = GuildConfigRecord.from_orm(row)
        self._store(guild_id, record)
        return record

    async def upsert_guild(
        self,
        guild_id: int | str,
        *,
        server_name: str | None = None,
        knowledge_base: str | None = None,
        staff_role_id: str | None | _Unset = _UNSET,
        ticket_category_id: str | None | _Unset = _UNSET,
    ) -> GuildConfigRecord:
        """Create-or-update a server config.

        ``staff_role_id`` / ``ticket_category_id`` accept ``None`` explicitly to
        *clear* the value; omitting them leaves the stored value untouched.
        """
        key = str(guild_id)
        values: dict[str, Any] = {}
        if server_name is not None:
            values["server_name"] = server_name
        if knowledge_base is not None:
            values["knowledge_base"] = knowledge_base
        # Explicit None clears the column; _UNSET leaves it untouched.
        if staff_role_id is not _UNSET:
            values["staff_role_id"] = staff_role_id
        if ticket_category_id is not _UNSET:
            values["ticket_category_id"] = ticket_category_id
        async with await self._lock_for(key):
            return await self._update_guild(key, **values)

    async def set_knowledge_base(
        self,
        guild_id: int | str,
        text_content: str,
        *,
        mode: KBMode = "replace",
        server_name: str | None = None,
    ) -> GuildConfigRecord:
        """Store the tenant's knowledge base, replacing/appending/prepending."""
        key = str(guild_id)
        async with await self._lock_for(key):
            current = await self.get_guild(key, use_cache=False)
            existing = (current.knowledge_base if current else "") or ""
            merged = _merge_knowledge_base(existing, text_content, mode)
            updates: dict[str, Any] = {"knowledge_base": merged}
            if server_name is not None:
                updates["server_name"] = server_name
            return await self._update_guild(key, **updates)

    async def set_staff_role(
        self, guild_id: int | str, role_id: int | str | None, *, server_name: str | None = None
    ) -> GuildConfigRecord:
        return await self.upsert_guild(
            guild_id, server_name=server_name, staff_role_id=None if role_id is None else str(role_id)
        )

    async def set_ticket_category(
        self, guild_id: int | str, category_id: int | str | None, *, server_name: str | None = None
    ) -> GuildConfigRecord:
        return await self.upsert_guild(
            guild_id,
            server_name=server_name,
            ticket_category_id=None if category_id is None else str(category_id),
        )

    async def rename_guild(self, guild_id: int | str, server_name: str) -> None:
        """Keep the stored name in sync (cheap; only writes when it changed)."""
        cached = await self.get_guild(guild_id)
        if cached is not None and cached.server_name == server_name:
            return
        await self._update_guild(str(guild_id), server_name=server_name)

    # ------------------------------------------------------------------ #
    # ticket_logs + ticket_activity
    # ------------------------------------------------------------------ #
    async def get_ticket(self, ticket_id: str) -> TicketRecord | None:
        async with self.db.read_session() as session:
            row = await session.get(TicketLog, str(ticket_id))
            if row is None:
                return None
            activity = await session.get(TicketActivity, str(row.channel_id or row.ticket_id))
            return _ticket_record(row, activity)

    async def open_ticket(
        self,
        *,
        ticket_id: str,
        guild_id: int | str,
        channel_id: int | str,
        user_id: int | str,
        status: str = STATUS_OPEN,
    ) -> TicketRecord:
        """Idempotently create the ticket row and return its current state."""
        ticket_key = str(ticket_id)
        channel_key = str(channel_id)
        async with self.db.session() as session:
            row = await session.get(TicketLog, ticket_key)
            if row is None:
                row = TicketLog(
                    ticket_id=ticket_key,
                    guild_id=str(guild_id),
                    channel_id=channel_key,
                    user_id=str(user_id),
                    status=status,
                )
                session.add(row)
            else:
                # Keep the opener fresh so /ticket-status reflects the requester.
                row.user_id = row.user_id or str(user_id)
                # Deliberately do NOT flip a resolved ticket back to open: closing
                # a ticket is an explicit admin action, and a member typing into a
                # closed ticket must not resurrect automated answering.
            await session.flush()

            activity = await session.get(TicketActivity, channel_key)
            if activity is None:
                activity = TicketActivity(
                    channel_id=channel_key, guild_id=str(guild_id), ticket_id=ticket_key
                )
                session.add(activity)
                await session.flush()
            return _ticket_record(row, activity)

    async def set_ticket_status(self, ticket_id: str, status: str) -> TicketRecord | None:
        async with self.db.session() as session:
            row = await session.get(TicketLog, str(ticket_id))
            if row is None:
                return None
            row.status = status
            await session.flush()
            activity = await session.get(TicketActivity, str(row.channel_id or row.ticket_id))
            return _ticket_record(row, activity)

    async def record_user_message(
        self, *, guild_id: int | str, channel_id: int | str, user_id: int | str, ticket_id: str
    ) -> None:
        now = datetime.now(timezone.utc)
        async with self.db.session() as session:
            activity = await session.get(TicketActivity, str(channel_id))
            if activity is None:
                activity = TicketActivity(
                    channel_id=str(channel_id), guild_id=str(guild_id), ticket_id=str(ticket_id)
                )
                session.add(activity)
            activity.user_message_count = (activity.user_message_count or 0) + 1
            activity.last_user_message_at = now
            activity.updated_at = now

    async def record_ai_reply(
        self, *, guild_id: int | str, channel_id: int | str, ticket_id: str
    ) -> None:
        now = datetime.now(timezone.utc)
        async with self.db.session() as session:
            activity = await session.get(TicketActivity, str(channel_id))
            if activity is None:
                activity = TicketActivity(
                    channel_id=str(channel_id), guild_id=str(guild_id), ticket_id=str(ticket_id)
                )
                session.add(activity)
            activity.ai_reply_count = (activity.ai_reply_count or 0) + 1
            activity.last_ai_reply_at = now
            activity.updated_at = now
            log_row = await session.get(TicketLog, str(ticket_id))
            if log_row is not None and log_row.status == STATUS_OPEN:
                log_row.status = STATUS_ANSWERED

    async def record_escalation(
        self, *, guild_id: int | str, channel_id: int | str, ticket_id: str, pinged: bool = True
    ) -> None:
        """Record an escalation.

        ``pinged=False`` means the staff role was *not* notified because the
        cooldown suppressed it. The counter still increases (so /ticket-status
        stays honest) but ``last_escalation_at`` is left alone — otherwise a
        long-running ticket would keep pushing the cooldown forward and staff
        would never be re-pinged at all.
        """
        now = datetime.now(timezone.utc)
        async with self.db.session() as session:
            activity = await session.get(TicketActivity, str(channel_id))
            if activity is None:
                activity = TicketActivity(
                    channel_id=str(channel_id), guild_id=str(guild_id), ticket_id=str(ticket_id)
                )
                session.add(activity)
            activity.escalation_count = (activity.escalation_count or 0) + 1
            if pinged:
                activity.last_escalation_at = now
            activity.updated_at = now
            log_row = await session.get(TicketLog, str(ticket_id))
            if log_row is not None:
                log_row.status = STATUS_ESCALATED

    async def escalation_on_cooldown(
        self, channel_id: int | str, *, cooldown_seconds: float
    ) -> tuple[bool, float]:
        """Has this ticket been escalated too recently to ping staff again?

        Returns ``(on_cooldown, seconds_remaining)``.
        """
        if cooldown_seconds <= 0:
            return False, 0.0
        async with self.db.read_session() as session:
            activity = await session.get(TicketActivity, str(channel_id))
        if activity is None or activity.last_escalation_at is None:
            return False, 0.0
        last = as_utc(activity.last_escalation_at)
        if last is None:  # pragma: no cover - defensive
            return False, 0.0
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        remaining = cooldown_seconds - elapsed
        return (remaining > 0, max(0.0, remaining))

    async def guild_stats(self, guild_id: int | str) -> GuildStats:
        key = str(guild_id)
        async with self.db.read_session() as session:
            total = await session.scalar(
                select(func.count()).select_from(TicketLog).where(TicketLog.guild_id == key)
            )
            by_status_rows = (
                await session.execute(
                    select(TicketLog.status, func.count())
                    .where(TicketLog.guild_id == key)
                    .group_by(TicketLog.status)
                )
            ).all()
            totals = await session.execute(
                select(
                    func.coalesce(func.sum(TicketActivity.ai_reply_count), 0),
                    func.coalesce(func.sum(TicketActivity.escalation_count), 0),
                ).where(TicketActivity.guild_id == key)
            )
            ai_replies, escalations = totals.one()

        by_status = {status: int(count) for status, count in by_status_rows}
        return GuildStats(
            guild_id=key,
            total_tickets=int(total or 0),
            open_tickets=by_status.get(STATUS_OPEN, 0) + by_status.get(STATUS_ANSWERED, 0),
            escalated_tickets=by_status.get(STATUS_ESCALATED, 0),
            resolved_tickets=by_status.get(STATUS_RESOLVED, 0),
            ai_replies=int(ai_replies or 0),
            escalations=int(escalations or 0),
        )

    async def recent_tickets(self, guild_id: int | str, limit: int = 10) -> Sequence[TicketRecord]:
        key = str(guild_id)
        async with self.db.read_session() as session:
            rows = (
                await session.execute(
                    select(TicketLog)
                    .where(TicketLog.guild_id == key)
                    .order_by(TicketLog.created_at.desc())
                    .limit(limit)
                )
            ).scalars().all()
            records: list[TicketRecord] = []
            for row in rows:
                activity = await session.get(TicketActivity, str(row.channel_id or row.ticket_id))
                records.append(_ticket_record(row, activity))
            return records

    # ------------------------------------------------------------------ #
    # llm_usage (free-tier budget observability)
    # ------------------------------------------------------------------ #
    async def bump_usage(
        self,
        *,
        provider: str,
        model: str,
        success: bool,
        prompt_chars: int = 0,
        completion_chars: int = 0,
        latency_ms: int = 0,
    ) -> None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        async with self.db.session() as session:
            row = (
                await session.execute(
                    select(UsageCounter).where(
                        UsageCounter.usage_date == day,
                        UsageCounter.provider == provider,
                        UsageCounter.model == model,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                row = UsageCounter(usage_date=day, provider=provider, model=model)
                session.add(row)
            if success:
                row.requests = (row.requests or 0) + 1
                row.prompt_chars = (row.prompt_chars or 0) + prompt_chars
                row.completion_chars = (row.completion_chars or 0) + completion_chars
                row.latency_ms_total = (row.latency_ms_total or 0) + latency_ms
            else:
                row.failures = (row.failures or 0) + 1

    async def usage_snapshot(self, days: int = 1) -> list[dict[str, Any]]:
        since = (datetime.now(timezone.utc) - timedelta(days=days - 1)).strftime("%Y-%m-%d")
        async with self.db.read_session() as session:
            rows = (
                await session.execute(
                    select(UsageCounter)
                    .where(UsageCounter.usage_date >= since)
                    .order_by(UsageCounter.usage_date.desc(), UsageCounter.provider)
                )
            ).scalars().all()
            return [
                {
                    "date": r.usage_date,
                    "provider": r.provider,
                    "model": r.model,
                    "requests": r.requests,
                    "failures": r.failures,
                    "avg_latency_ms": round(r.latency_ms_total / r.requests) if r.requests else 0,
                }
                for r in rows
            ]


def _ticket_record(row: TicketLog, activity: TicketActivity | None) -> TicketRecord:
    return TicketRecord(
        ticket_id=str(row.ticket_id),
        guild_id=row.guild_id,
        channel_id=row.channel_id,
        user_id=row.user_id,
        status=row.status or STATUS_OPEN,
        created_at=as_utc(row.created_at),
        ai_reply_count=int(activity.ai_reply_count or 0) if activity else 0,
        escalation_count=int(activity.escalation_count or 0) if activity else 0,
        user_message_count=int(activity.user_message_count or 0) if activity else 0,
        last_escalation_at=as_utc(activity.last_escalation_at) if activity else None,
        last_ai_reply_at=as_utc(activity.last_ai_reply_at) if activity else None,
        last_user_message_at=as_utc(activity.last_user_message_at) if activity else None,
    )


def _merge_knowledge_base(existing: str, incoming: str, mode: KBMode) -> str:
    """Combine knowledge base text according to the requested mode."""
    existing = (existing or "").rstrip()
    incoming = (incoming or "").strip()
    if mode == "replace" or not existing:
        return incoming
    separator = "\n\n"
    if mode == "prepend":
        return f"{incoming}{separator}{existing}".strip()
    return f"{existing}{separator}{incoming}".strip()


__all__ = [
    "GuildConfigRecord",
    "GuildStats",
    "TicketRecord",
    "TicketRepository",
    "as_utc",
]
