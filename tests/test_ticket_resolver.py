"""Ticket gating: which messages the AI is allowed to answer."""

from __future__ import annotations

import re

import pytest

from bot.db.repository import GuildConfigRecord
from bot.services.ticket_resolver import (
    SkipReason,
    TicketResolver,
    extract_query,
    resolve_category_id,
)
from fakes import (
    BOT_USER_ID,
    CATEGORY_ID,
    GUILD_ID,
    FakeAttachment,
    FakeChannel,
    FakeGuild,
    FakeMessage,
    FakeThread,
    FakeUser,
)


def make_config(**overrides) -> GuildConfigRecord:
    base = dict(
        guild_id=str(GUILD_ID),
        server_name="Test Server",
        knowledge_base="Refunds within 14 days.",
        staff_role_id="999",
        ticket_category_id=str(CATEGORY_ID),
    )
    base.update(overrides)
    return GuildConfigRecord(**base)


def make_resolver(**kwargs) -> TicketResolver:
    kwargs.setdefault("bot_user_id", BOT_USER_ID)
    return TicketResolver(**kwargs)


def ticket_message(content: str = "how do refunds work?", **kwargs) -> FakeMessage:
    channel = kwargs.pop("channel", FakeChannel(id=777, name="ticket-0001", category_id=CATEGORY_ID))
    return FakeMessage(
        id=kwargs.pop("id", 1),
        content=content,
        channel=channel,
        guild=kwargs.pop("guild", FakeGuild(id=GUILD_ID)),
        author=kwargs.pop("author", FakeUser(id=1001, name="alice")),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# allow path
# --------------------------------------------------------------------------- #
def test_message_in_configured_category_is_allowed():
    decision = make_resolver().resolve(ticket_message(), make_config())
    assert decision.allowed is True
    assert decision.reason is SkipReason.ALLOWED
    assert bool(decision) is True
    assert decision.context.ticket_id == "777"
    assert decision.context.guild_id == GUILD_ID
    assert decision.context.category_id == CATEGORY_ID


def test_thread_inside_configured_category_is_allowed():
    parent = FakeChannel(id=300, name="support-desk", category_id=CATEGORY_ID)
    thread = FakeThread(id=301, name="ticket-alice", parent=parent)
    decision = make_resolver().resolve(ticket_message(channel=thread), make_config())
    assert decision.allowed is True
    assert decision.context.is_thread is True
    assert decision.context.parent_channel_id == 300
    assert decision.context.ticket_id == "301"


def test_channel_name_fallback_when_category_is_not_required():
    resolver = make_resolver(require_configured_category=False)
    channel = FakeChannel(id=778, name="ticket-9999", category_id=None)
    assert resolver.resolve(ticket_message(channel=channel), make_config()).allowed is True


def test_permissive_mode_reports_name_mismatch_for_ordinary_channels():
    resolver = make_resolver(require_configured_category=False)
    other = FakeChannel(id=779, name="general", category_id=None)
    decision = resolver.resolve(
        ticket_message(channel=other), make_config(ticket_category_id=None)
    )
    assert decision.allowed is False
    assert decision.reason is SkipReason.NAME_MISMATCH


def test_permissive_mode_still_reports_wrong_category_when_one_is_configured():
    resolver = make_resolver(require_configured_category=False)
    other = FakeChannel(id=779, name="general", category_id=12_345)
    decision = resolver.resolve(ticket_message(channel=other), make_config())
    assert decision.allowed is False
    assert decision.reason is SkipReason.WRONG_CATEGORY


def test_custom_name_pattern_is_honoured():
    resolver = make_resolver(
        require_configured_category=False,
        ticket_name_pattern=re.compile(r"^helpdesk", re.IGNORECASE),
    )
    channel = FakeChannel(id=1, name="HelpDesk-42", category_id=None)
    assert resolver.resolve(ticket_message(channel=channel), make_config()).allowed is True


# --------------------------------------------------------------------------- #
# deny paths
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"guild": None}, SkipReason.NOT_GUILD),
        ({"author": FakeUser(id=2, bot=True)}, SkipReason.BOT_AUTHOR),
        ({"author": FakeUser(id=BOT_USER_ID, bot=True)}, SkipReason.SELF_AUTHOR),
        ({"webhook_id": 12345}, SkipReason.WEBHOOK),
        ({"type": 7}, SkipReason.SYSTEM_MESSAGE),           # user join
        ({"type": 19}, SkipReason.SYSTEM_MESSAGE),          # reply notification
        ({"interaction_metadata": object()}, SkipReason.COMMAND_ECHO),
        ({"content": "   "}, SkipReason.EMPTY),
    ],
)
def test_denied_messages(kwargs, expected):
    decision = make_resolver().resolve(ticket_message(**kwargs), make_config())
    assert decision.allowed is False
    assert decision.reason is expected


def test_unconfigured_server_is_never_answered():
    """Tenant isolation: no config row means silence, even in a ticket channel."""
    decision = make_resolver().resolve(ticket_message(), None)
    assert decision.allowed is False
    assert decision.reason is SkipReason.NOT_CONFIGURED


def test_other_tenants_category_does_not_leak():
    """A category configured for guild A must not enable guild B's channel."""
    foreign_config = make_config(guild_id="200", ticket_category_id="888")
    decision = make_resolver().resolve(ticket_message(), foreign_config)
    assert decision.allowed is False
    assert decision.reason is SkipReason.WRONG_CATEGORY


def test_strict_mode_ignores_ticket_named_channels_without_a_category():
    resolver = make_resolver(require_configured_category=True)
    channel = FakeChannel(id=780, name="ticket-1", category_id=None)
    decision = resolver.resolve(ticket_message(channel=channel), make_config())
    assert decision.allowed is False
    assert decision.reason is SkipReason.WRONG_CATEGORY


def test_config_with_no_category_and_strict_mode_denies_everything():
    """Strict mode is a real restriction: unconfigured servers get silence."""
    resolver = make_resolver(require_configured_category=True)
    decision = resolver.resolve(ticket_message(), make_config(ticket_category_id=None))
    assert decision.allowed is False
    assert decision.reason is SkipReason.CATEGORY_UNSET


def test_strict_mode_denies_ticket_named_channel_when_category_unset():
    resolver = make_resolver(require_configured_category=True)
    channel = FakeChannel(id=781, name="ticket-42", category_id=None)
    decision = resolver.resolve(
        ticket_message(channel=channel), make_config(ticket_category_id=None)
    )
    assert decision.allowed is False
    assert decision.reason is SkipReason.CATEGORY_UNSET


def test_resolved_ticket_is_ignored():
    decision = make_resolver().resolve(ticket_message(), make_config(), ticket_status="resolved")
    assert decision.allowed is False
    assert decision.reason is SkipReason.TICKET_CLOSED


def test_active_statuses_are_allowed():
    resolver = make_resolver()
    for status in ("open", "answered", "escalated"):
        assert resolver.resolve(ticket_message(), make_config(), ticket_status=status).allowed is True


def test_empty_knowledge_base_gate_when_enabled():
    resolver = make_resolver(require_knowledge_base=True)
    decision = resolver.resolve(ticket_message(), make_config(knowledge_base=""))
    assert decision.allowed is False
    assert decision.reason is SkipReason.NO_KNOWLEDGE_BASE


def test_none_message_is_handled_gracefully():
    decision = make_resolver().resolve(None, make_config())
    assert decision.allowed is False


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_resolve_category_id_handles_channels_and_threads():
    assert resolve_category_id(FakeChannel(id=1, category_id=CATEGORY_ID)) == CATEGORY_ID
    parent = FakeChannel(id=2, category_id=CATEGORY_ID)
    assert resolve_category_id(FakeThread(id=3, parent=parent)) == CATEGORY_ID
    assert resolve_category_id(FakeChannel(id=4, category_id=None)) is None
    assert resolve_category_id(None) is None


def test_extract_query_includes_content_and_attachments():
    message = ticket_message(
        content="my payment failed",
        attachments=(FakeAttachment(filename="receipt.pdf", size=2048),),
    )
    query = extract_query(message)
    assert "my payment failed" in query
    assert "receipt.pdf" in query


def test_extract_query_flags_oversized_attachments_as_unread():
    message = ticket_message(
        content="see attached", attachments=(FakeAttachment(filename="log.txt", size=9_000_000),)
    )
    assert "not read" in extract_query(message)


def test_extract_query_empty_when_nothing_to_read():
    assert extract_query(ticket_message(content="")) == ""
