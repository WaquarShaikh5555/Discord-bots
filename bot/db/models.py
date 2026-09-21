"""SQLAlchemy 2.0 ORM models.

``server_configs`` and ``ticket_logs`` implement the schema from the project
specification verbatim (same tables, columns, types and defaults). Two additive
tables extend it with operational state that a production multi-tenant bot
needs — they are created alongside the spec tables and never alter them:

``ticket_activity``
    Per-ticket counters and escalation timestamps, used for the "do not re-ping
    staff every message" cooldown and for ``/ticket-status``.

``llm_usage``
    Per-day, per-provider request counters so the free-tier daily budget survives
    restarts and is visible to operators.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp used for Python-side defaults/onupdate."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for every model."""


class ServerConfig(Base):
    """One row per Discord server — the multi-tenant isolation boundary.

    Every knowledge base, staff role and ticket category is scoped to
    ``guild_id``; no query in the bot ever crosses guilds.
    """

    __tablename__ = "server_configs"

    guild_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    server_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    knowledge_base: Mapped[str] = mapped_column(
        Text, nullable=False, default="", server_default=""
    )
    staff_role_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    ticket_category_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=utcnow,
        onupdate=utcnow,
    )

    # --- derived helpers (not persisted) --------------------------------- #
    @property
    def has_knowledge_base(self) -> bool:
        return bool(self.knowledge_base and self.knowledge_base.strip())

    @property
    def is_ready(self) -> bool:
        """A server is fully configured when KB, staff role and category exist."""
        return bool(self.has_knowledge_base and self.staff_role_id and self.ticket_category_id)

    def missing_pieces(self) -> list[str]:
        missing: list[str] = []
        if not self.has_knowledge_base:
            missing.append("knowledge base (/setup-kb)")
        if not self.staff_role_id:
            missing.append("staff role (/set-staff-role)")
        if not self.ticket_category_id:
            missing.append("ticket category (/set-ticket-category)")
        return missing

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<ServerConfig guild_id={self.guild_id!r} name={self.server_name!r} "
            f"kb_chars={len(self.knowledge_base or '')} staff_role={self.staff_role_id!r} "
            f"category={self.ticket_category_id!r}>"
        )


class TicketLog(Base):
    """Lifecycle record for a ticket channel/thread."""

    __tablename__ = "ticket_logs"
    __table_args__ = (
        Index("ix_ticket_logs_guild_status", "guild_id", "status"),
    )

    ticket_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    guild_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("server_configs.guild_id", ondelete="CASCADE"), nullable=True
    )
    channel_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    user_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="open", server_default="open"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=utcnow
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<TicketLog ticket_id={self.ticket_id!r} guild_id={self.guild_id!r} "
            f"status={self.status!r}>"
        )


class TicketActivity(Base):
    """Extension table: counters + escalation cooldown state per ticket."""

    __tablename__ = "ticket_activity"

    channel_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    guild_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    ticket_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    ai_reply_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    escalation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    user_message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_user_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_ai_reply_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_escalation_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=utcnow,
        onupdate=utcnow,
    )


class UsageCounter(Base):
    """Extension table: daily LLM request counters per provider/model."""

    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    usage_date: Mapped[str] = mapped_column(String(10), nullable=False, index=True)  # YYYY-MM-DD (UTC)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    requests: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prompt_chars: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_chars: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=utcnow,
        onupdate=utcnow,
    )


ALL_MODELS: tuple[type[Base], ...] = (ServerConfig, TicketLog, TicketActivity, UsageCounter)
