"""Tenant persistence: isolation, KB merging, ticket lifecycle, caching."""

from __future__ import annotations

import pytest

from bot.constants import STATUS_ESCALATED, STATUS_OPEN, STATUS_RESOLVED
from bot.db.repository import TicketRepository, _merge_knowledge_base
from bot.db.session import Database

GUILD_A = 111
GUILD_B = 222


@pytest.fixture
async def fresh_repo(database: Database) -> TicketRepository:
    return TicketRepository(database, cache_ttl=0.0)


# --------------------------------------------------------------------------- #
# tenant isolation — the single most important property of a multi-tenant bot
# --------------------------------------------------------------------------- #
async def test_unknown_guild_returns_none(fresh_repo: TicketRepository):
    assert await fresh_repo.get_guild(999_999) is None


async def test_knowledge_bases_are_isolated_between_guilds(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "Server A")
    await fresh_repo.ensure_guild(GUILD_B, "Server B")
    await fresh_repo.set_knowledge_base(GUILD_A, "A: refunds in 14 days.")
    await fresh_repo.set_knowledge_base(GUILD_B, "B: refunds in 30 days.")

    record_a = await fresh_repo.get_guild(GUILD_A)
    record_b = await fresh_repo.get_guild(GUILD_B)
    assert record_a.knowledge_base == "A: refunds in 14 days."
    assert record_b.knowledge_base == "B: refunds in 30 days."
    assert "B:" not in record_a.knowledge_base
    assert "A:" not in record_b.knowledge_base


async def test_staff_roles_and_categories_are_per_guild(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "A")
    await fresh_repo.ensure_guild(GUILD_B, "B")
    await fresh_repo.set_staff_role(GUILD_A, 11)
    await fresh_repo.set_ticket_category(GUILD_B, 22)

    a = await fresh_repo.get_guild(GUILD_A)
    b = await fresh_repo.get_guild(GUILD_B)
    assert a.staff_role_id == "11" and a.ticket_category_id is None
    assert b.staff_role_id is None and b.ticket_category_id == "22"


# --------------------------------------------------------------------------- #
# knowledge base management
# --------------------------------------------------------------------------- #
async def test_ensure_guild_is_idempotent(fresh_repo: TicketRepository):
    first = await fresh_repo.ensure_guild(GUILD_A, "Server A")
    second = await fresh_repo.ensure_guild(GUILD_A, "Server A")
    assert first.guild_id == second.guild_id == str(GUILD_A)
    assert second.created_at == first.created_at


async def test_ensure_guild_updates_renamed_server(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "Old Name")
    renamed = await fresh_repo.ensure_guild(GUILD_A, "New Name")
    assert renamed.server_name == "New Name"


@pytest.mark.parametrize(
    "existing,incoming,mode,expected",
    [
        ("", "new rules", "replace", "new rules"),
        ("old rules", "new rules", "replace", "new rules"),
        ("old rules", "new rules", "append", "old rules\n\nnew rules"),
        ("old rules", "new rules", "prepend", "new rules\n\nold rules"),
        ("", "new rules", "append", "new rules"),
    ],
)
def test_merge_knowledge_base_modes(existing, incoming, mode, expected):
    assert _merge_knowledge_base(existing, incoming, mode) == expected


async def test_append_preserves_existing_content(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "A")
    await fresh_repo.set_knowledge_base(GUILD_A, "Part one.")
    record = await fresh_repo.set_knowledge_base(GUILD_A, "Part two.", mode="append")
    assert record.knowledge_base == "Part one.\n\nPart two."
    assert record.knowledge_base_words == 4


async def test_append_does_not_wipe_server_name(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "Named Server")
    record = await fresh_repo.set_knowledge_base(GUILD_A, "rules", mode="append")
    assert record.server_name == "Named Server"


async def test_clearing_staff_role_and_category(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "A")
    await fresh_repo.set_staff_role(GUILD_A, 11)
    await fresh_repo.set_ticket_category(GUILD_A, 22)
    cleared = await fresh_repo.upsert_guild(GUILD_A, staff_role_id=None, ticket_category_id=None)
    assert cleared.staff_role_id is None
    assert cleared.ticket_category_id is None


async def test_unset_fields_are_left_untouched(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "A")
    await fresh_repo.set_staff_role(GUILD_A, 11)
    record = await fresh_repo.upsert_guild(GUILD_A, server_name="Renamed")
    assert record.staff_role_id == "11", "omitted fields must not be cleared"
    assert record.server_name == "Renamed"


def test_missing_pieces_checklist():
    from bot.db.repository import GuildConfigRecord

    empty = GuildConfigRecord(guild_id="1")
    assert len(empty.missing_pieces()) == 3
    assert empty.is_fully_configured is False

    ready = GuildConfigRecord(
        guild_id="1", knowledge_base="kb", staff_role_id="2", ticket_category_id="3"
    )
    assert ready.missing_pieces() == []
    assert ready.is_fully_configured is True


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #
async def test_cache_serves_repeated_reads(database: Database):
    repo = TicketRepository(database, cache_ttl=60.0)
    await repo.ensure_guild(GUILD_A, "A")
    first = await repo.get_guild(GUILD_A)
    second = await repo.get_guild(GUILD_A)
    assert first == second
    assert repo.cache_size >= 1


async def test_negative_cache_avoids_repeated_lookups(database: Database):
    repo = TicketRepository(database, cache_ttl=60.0)
    assert await repo.get_guild(GUILD_B) is None
    assert repo.cache_size == 1, "unknown guilds must be negative-cached"
    assert await repo.get_guild(GUILD_B) is None


async def test_write_invalidates_cache(database: Database):
    repo = TicketRepository(database, cache_ttl=60.0)
    await repo.ensure_guild(GUILD_A, "A")
    assert (await repo.get_guild(GUILD_A)).knowledge_base == ""
    await repo.set_knowledge_base(GUILD_A, "fresh rules")
    assert (await repo.get_guild(GUILD_A)).knowledge_base == "fresh rules"


async def test_cache_ttl_zero_always_reads_through(database: Database):
    repo = TicketRepository(database, cache_ttl=0.0)
    await repo.ensure_guild(GUILD_A, "A")
    await repo.get_guild(GUILD_A)
    assert repo.cache_size == 0


async def test_invalidate_single_guild(database: Database):
    repo = TicketRepository(database, cache_ttl=60.0)
    await repo.ensure_guild(GUILD_A, "A")
    await repo.ensure_guild(GUILD_B, "B")
    await repo.get_guild(GUILD_A)
    await repo.get_guild(GUILD_B)
    repo.invalidate(GUILD_A)
    assert str(GUILD_B) in repo._cache
    assert str(GUILD_A) not in repo._cache
    repo.invalidate()
    assert repo.cache_size == 0


# --------------------------------------------------------------------------- #
# ticket lifecycle
# --------------------------------------------------------------------------- #
async def test_open_ticket_creates_log_and_activity(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "A")
    ticket = await fresh_repo.open_ticket(
        ticket_id="777", guild_id=GUILD_A, channel_id="777", user_id="1001"
    )
    assert ticket.ticket_id == "777"
    assert ticket.status == STATUS_OPEN
    assert ticket.is_active is True
    assert ticket.ai_reply_count == 0


async def test_open_ticket_is_idempotent(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "A")
    first = await fresh_repo.open_ticket(
        ticket_id="777", guild_id=GUILD_A, channel_id="777", user_id="1001"
    )
    second = await fresh_repo.open_ticket(
        ticket_id="777", guild_id=GUILD_A, channel_id="777", user_id="1002"
    )
    assert second.created_at == first.created_at
    assert second.user_id == "1001", "the original requester is preserved"


async def test_reply_transitions_status_to_answered(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "A")
    await fresh_repo.open_ticket(ticket_id="777", guild_id=GUILD_A, channel_id="777", user_id="1")
    await fresh_repo.record_user_message(guild_id=GUILD_A, channel_id="777", user_id="1", ticket_id="777")
    await fresh_repo.record_ai_reply(guild_id=GUILD_A, channel_id="777", ticket_id="777")

    ticket = await fresh_repo.get_ticket("777")
    assert ticket.status == "answered"
    assert ticket.ai_reply_count == 1
    assert ticket.user_message_count == 1
    assert ticket.last_ai_reply_at is not None


async def test_escalation_transitions_status_and_counts(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "A")
    await fresh_repo.open_ticket(ticket_id="778", guild_id=GUILD_A, channel_id="778", user_id="1")
    await fresh_repo.record_escalation(guild_id=GUILD_A, channel_id="778", ticket_id="778")

    ticket = await fresh_repo.get_ticket("778")
    assert ticket.status == STATUS_ESCALATED
    assert ticket.escalation_count == 1
    assert ticket.last_escalation_at is not None


async def test_escalation_cooldown_window(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "A")
    await fresh_repo.open_ticket(ticket_id="779", guild_id=GUILD_A, channel_id="779", user_id="1")

    on_cooldown, remaining = await fresh_repo.escalation_on_cooldown("779", cooldown_seconds=900)
    assert on_cooldown is False and remaining == 0.0

    await fresh_repo.record_escalation(guild_id=GUILD_A, channel_id="779", ticket_id="779")
    on_cooldown, remaining = await fresh_repo.escalation_on_cooldown("779", cooldown_seconds=900)
    assert on_cooldown is True
    assert 890 < remaining <= 900


async def test_suppressed_escalation_keeps_the_original_ping_time(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "A")
    await fresh_repo.open_ticket(ticket_id="790", guild_id=GUILD_A, channel_id="790", user_id="1")

    await fresh_repo.record_escalation(guild_id=GUILD_A, channel_id="790", ticket_id="790")
    first = (await fresh_repo.get_ticket("790")).last_escalation_at

    await fresh_repo.record_escalation(
        guild_id=GUILD_A, channel_id="790", ticket_id="790", pinged=False
    )
    ticket = await fresh_repo.get_ticket("790")
    assert ticket.escalation_count == 2, "the counter still tracks every escalation"
    assert ticket.last_escalation_at == first, "but the cooldown anchor does not move"


async def test_cooldown_disabled_when_zero(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "A")
    await fresh_repo.open_ticket(ticket_id="780", guild_id=GUILD_A, channel_id="780", user_id="1")
    await fresh_repo.record_escalation(guild_id=GUILD_A, channel_id="780", ticket_id="780")
    assert await fresh_repo.escalation_on_cooldown("780", cooldown_seconds=0) == (False, 0.0)


async def test_resolve_stops_the_ticket(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "A")
    await fresh_repo.open_ticket(ticket_id="781", guild_id=GUILD_A, channel_id="781", user_id="1")
    record = await fresh_repo.set_ticket_status("781", STATUS_RESOLVED)
    assert record.status == STATUS_RESOLVED
    assert record.is_active is False


async def test_set_status_on_unknown_ticket_returns_none(fresh_repo: TicketRepository):
    assert await fresh_repo.set_ticket_status("nope", STATUS_OPEN) is None


async def test_guild_stats_and_recent_tickets(fresh_repo: TicketRepository):
    await fresh_repo.ensure_guild(GUILD_A, "A")
    for index in range(3):
        channel = 800 + index
        await fresh_repo.open_ticket(
            ticket_id=str(channel), guild_id=GUILD_A, channel_id=str(channel), user_id="1"
        )
    await fresh_repo.record_ai_reply(guild_id=GUILD_A, channel_id="800", ticket_id="800")
    await fresh_repo.record_escalation(guild_id=GUILD_A, channel_id="801", ticket_id="801")
    await fresh_repo.set_ticket_status("802", STATUS_RESOLVED)

    stats = await fresh_repo.guild_stats(GUILD_A)
    assert stats.total_tickets == 3
    assert stats.ai_replies == 1
    assert stats.escalations == 1
    assert stats.resolved_tickets == 1

    recent = await fresh_repo.recent_tickets(GUILD_A, limit=2)
    assert len(recent) == 2


async def test_usage_counters(fresh_repo: TicketRepository):
    await fresh_repo.bump_usage(
        provider="groq", model="llama-3.3-70b-versatile", success=True,
        prompt_chars=1000, completion_chars=200, latency_ms=350,
    )
    await fresh_repo.bump_usage(provider="groq", model="llama-3.3-70b-versatile", success=False)
    rows = await fresh_repo.usage_snapshot(days=1)
    assert len(rows) == 1
    assert rows[0]["requests"] == 1
    assert rows[0]["failures"] == 1
    assert rows[0]["avg_latency_ms"] == 350
