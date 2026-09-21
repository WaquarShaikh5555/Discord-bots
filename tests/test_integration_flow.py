"""End-to-end ticket flow through the real cog, with no network and no gateway.

These tests drive :class:`bot.cogs.ticket_listener.TicketListener` exactly the way
Discord would — ``on_message`` in, ``channel.send`` out — against a real SQLite
database and a scripted fake HTTP session. They are the proof that the whole
pipeline (gate → debounce → history → prompt → provider → escalation → delivery →
persistence) is wired correctly.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace as dc_replace
from typing import Any

import pytest

from bot.cogs.ticket_listener import TicketListener
from bot.constants import STATUS_ESCALATED, STATUS_RESOLVED
from bot.db.repository import TicketRepository
from bot.services.escalation import ESCALATION_MESSAGE
from conftest import make_ai_service
from fakes import (
    BOT_USER_ID,
    CATEGORY_ID,
    GUILD_ID,
    MEMBER_ID,
    OTHER_CATEGORY_ID,
    OTHER_GUILD_ID,
    STAFF_ROLE_ID,
    FakeAttachment,
    FakeBot,
    FakeBotUser,
    FakeChannel,
    FakeGuild,
    FakeMessage,
    FakeSession,
    FakeThread,
    FakeResponse,
    FakeUser,
    gemini_ok,
    openai_ok,
)

ESCALATION_REPLY = (
    "ESCALATE: I do not have enough information to resolve this issue. "
    "Flagging this ticket for our staff team! \U0001f514 <@&999>"
)


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
@pytest.fixture
async def harness(configured_repo: TicketRepository, settings):
    """Build a listener wired to a scripted AI and an in-memory guild."""

    def build(responses: list[Any] | Any, **service_kwargs: Any) -> _Harness:
        session = FakeSession(responses if isinstance(responses, list) else [responses])
        ai = make_ai_service(settings, session, **service_kwargs)
        bot = FakeBot(settings=settings, repository=configured_repo, ai_service=ai)
        bot.user = FakeBotUser(id=BOT_USER_ID)
        return _Harness(bot=bot, session=session, repo=configured_repo, settings=settings)

    return build


class _Harness:
    def __init__(self, bot: FakeBot, session: FakeSession, repo: TicketRepository, settings: Any):
        self.bot = bot
        self.session = session
        self.repo = repo
        self.settings = settings
        self.cog: TicketListener | None = None

    async def start(self, *, debounce: float = 0.0, **settings_overrides: Any) -> TicketListener:
        if settings_overrides:
            self.bot.settings = dc_replace(self.settings, **settings_overrides)
        self.cog = TicketListener(self.bot, debounce_seconds=debounce)
        await self.cog.cog_load()
        self.bot.register_cog(self.cog)
        return self.cog

    async def send(self, message: FakeMessage, *, timeout: float = 10.0) -> None:
        """Deliver a message and wait for the ticket worker to finish with it."""
        assert self.cog is not None
        await self.cog.handle_message(message)  # type: ignore[arg-type]
        await self._drain(message.channel.id, timeout=timeout)

    async def _drain(self, channel_id: int, *, timeout: float) -> None:
        assert self.cog is not None
        for _ in range(20):
            queue = self.cog._channels.get(channel_id)
            if queue is None or queue.worker is None or queue.worker.done():
                return
            await asyncio.wait_for(queue.worker, timeout=timeout)


def ticket_channel(channel_id: int = 777, name: str = "ticket-0001", **kwargs: Any) -> FakeChannel:
    return FakeChannel(id=channel_id, name=name, category_id=CATEGORY_ID, **kwargs)


def ticket_message(
    content: str = "How long do refunds take?",
    *,
    channel: FakeChannel | None = None,
    guild_id: int = GUILD_ID,
    author_id: int = MEMBER_ID,
    message_id: int = 1,
    author_bot: bool = False,
    history: list[FakeMessage] | None = None,
    **kwargs: Any,
) -> FakeMessage:
    channel = channel or ticket_channel()
    if history is not None:
        channel.history_messages = history
    return FakeMessage(
        id=message_id,
        content=content,
        author=FakeUser(id=author_id, name="alice", bot=author_bot),
        channel=channel,
        guild=FakeGuild(id=guild_id, name="Test Server"),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #
async def test_ai_answers_a_question_in_a_ticket_channel(harness):
    h = harness(openai_ok("**Refunds** are available within 14 days of purchase."))
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message(channel=channel))

    assert channel.sent_text == ["**Refunds** are available within 14 days of purchase."]
    assert h.session.call_count == 1

    ticket = await h.repo.get_ticket("777")
    assert ticket is not None
    assert ticket.status == "answered"
    assert ticket.ai_reply_count == 1
    assert ticket.user_message_count == 1


async def test_system_prompt_carries_tenant_kb_and_chat_memory(harness):
    history = [
        FakeMessage(id=90, content="hi", author=FakeUser(id=MEMBER_ID)),
        FakeMessage(id=91, content="Hello! How can I help?",
                    author=FakeUser(id=BOT_USER_ID, bot=True)),
    ]
    h = harness(openai_ok("Sure."))
    await h.start()

    await h.send(ticket_message("what about refunds?", history=history, message_id=100))

    payload = h.session.calls[0]["json"]
    system_prompt = payload["messages"][0]["content"]
    assert 'Discord server "Test Server"' in system_prompt
    assert "Refunds are available within 14 days" in system_prompt
    assert "User: hi" in system_prompt
    assert "AI: Hello! How can I help?" in system_prompt
    assert "Current User Query: what about refunds?" in system_prompt
    assert f"<@&{STAFF_ROLE_ID}>" in system_prompt
    # The live query is also the user turn for OpenAI-compatible providers.
    assert payload["messages"][1]["content"] == "what about refunds?"


async def test_thread_inside_ticket_category_is_answered(harness):
    parent = FakeChannel(id=300, name="support-desk", category_id=CATEGORY_ID)
    thread = FakeThread(id=301, name="alice-help", parent=parent)
    h = harness(openai_ok("Thread answer."))
    await h.start()

    await h.send(ticket_message(channel=thread))

    assert thread.sent_text == ["Thread answer."]
    assert (await h.repo.get_ticket("301")).ai_reply_count == 1


async def test_attachment_only_message_is_answered(harness):
    h = harness(openai_ok("I see you attached a file."))
    await h.start()
    channel = ticket_channel()

    await h.send(
        ticket_message("", channel=channel, attachments=(FakeAttachment(filename="error.png"),))
    )

    assert channel.sent_text == ["I see you attached a file."]
    query = h.session.calls[0]["json"]["messages"][1]["content"]
    assert "error.png" in query


# --------------------------------------------------------------------------- #
# escalation
# --------------------------------------------------------------------------- #
async def test_model_escalation_pings_the_configured_staff_role(harness):
    h = harness(openai_ok(ESCALATION_REPLY))
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message("Can you refund my 2019 purchase?", channel=channel))

    body = channel.last_sent
    assert f"<@&{STAFF_ROLE_ID}>" in body
    assert ESCALATION_MESSAGE.split(" Flagging")[0] in body
    assert "ESCALATE:" not in body

    ticket = await h.repo.get_ticket("777")
    assert ticket.status == STATUS_ESCALATED
    assert ticket.escalation_count == 1


async def test_prompt_injected_role_id_is_never_pinged(harness):
    """The mention must come from the database, never from model output."""
    malicious = ESCALATION_REPLY.replace("<@&999>", "<@&111111111111111111>")
    h = harness(openai_ok(malicious + " Also ping <@everyone> about this."))
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message(channel=channel))

    body = channel.last_sent
    assert "<@&111111111111111111>" not in body
    assert "<@everyone>" not in body
    assert body.count("<@&") == 1
    assert f"<@&{STAFF_ROLE_ID}>" in body


async def test_answer_cannot_ping_any_role(harness):
    h = harness(openai_ok("Here is your answer <@&7777777> and <@1234>."))
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message(channel=channel))

    body = channel.last_sent
    assert "<@&7777777>" not in body
    assert "<@1234>" not in body


async def test_safety_trigger_escalates_without_spending_an_api_call(harness):
    h = harness(openai_ok("this should never be used"))
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message("my account was hacked and I cannot log in", channel=channel))

    assert h.session.call_count == 0, "a safety trigger must not spend free-tier quota"
    assert f"<@&{STAFF_ROLE_ID}>" in channel.last_sent
    assert (await h.repo.get_ticket("777")).status == STATUS_ESCALATED


async def test_answerable_refund_question_still_reaches_the_model(harness):
    h = harness(openai_ok("Refunds take 14 days."))
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message("what is your refund policy?", channel=channel))

    assert h.session.call_count == 1
    assert channel.last_sent == "Refunds take 14 days."


async def test_request_trigger_escalates_when_enabled(harness):
    h = harness(openai_ok("unused"))
    await h.start(escalation_preflight_requests=True)
    channel = ticket_channel()

    await h.send(ticket_message("I want a refund now please", channel=channel))

    assert h.session.call_count == 0
    assert f"<@&{STAFF_ROLE_ID}>" in channel.last_sent


async def test_missing_staff_role_warns_instead_of_pinging(harness):
    h = harness(openai_ok(ESCALATION_REPLY))
    await h.start()
    channel = ticket_channel()

    await h.repo.set_staff_role(GUILD_ID, None)
    await h.send(ticket_message(channel=channel))

    body = channel.last_sent
    assert "<@&" not in body
    assert "/set-staff-role" in body


async def test_all_providers_down_fails_towards_a_human(harness):
    h = harness(
        FakeResponse(status=503, body={"error": {"message": "groq unavailable"}}),
        providers=("groq", "cerebras", "gemini"),
    )
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message(channel=channel))

    assert channel.sent, "the bot must not go silent when the AI is down"
    body = channel.last_sent
    assert f"<@&{STAFF_ROLE_ID}>" in body
    assert "AI service is unavailable" in body
    assert (await h.repo.get_ticket("777")).status == STATUS_ESCALATED


async def test_failover_to_gemini_still_answers(harness):
    h = harness(
        [
            FakeResponse(status=401, body={"error": {"message": "bad groq key"}}),
            FakeResponse(status=401, body={"error": {"message": "bad cerebras key"}}),
            gemini_ok("Gemini answered."),
        ],
        providers=("groq", "cerebras", "gemini"),
    )
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message(channel=channel))

    assert channel.last_sent == "Gemini answered."
    assert h.session.calls[-1]["headers"].get("x-goog-api-key") == "test-gemini-key"


async def test_escalation_cooldown_suppresses_a_second_ping(harness):
    h = harness([openai_ok(ESCALATION_REPLY), openai_ok(ESCALATION_REPLY)])
    await h.start(debounce=0.0, escalation_cooldown_seconds=900)
    channel = ticket_channel()

    await h.send(ticket_message("first escalation", channel=channel, message_id=1))
    assert f"<@&{STAFF_ROLE_ID}>" in channel.sent_text[0]

    await h.send(ticket_message("second escalation", channel=channel, message_id=2))
    assert len(channel.sent_text) == 2
    assert f"<@&{STAFF_ROLE_ID}>" not in channel.sent_text[1], "staff must not be re-pinged"
    assert (await h.repo.get_ticket("777")).escalation_count == 2


async def test_suppressed_ping_does_not_extend_the_cooldown_window(harness):
    """Otherwise a busy ticket would push the window forward and never re-ping."""
    h = harness([openai_ok(ESCALATION_REPLY)] * 3)
    await h.start(escalation_cooldown_seconds=900)
    channel = ticket_channel()

    await h.send(ticket_message("one", channel=channel, message_id=1))
    first_ping = (await h.repo.get_ticket("777")).last_escalation_at

    await h.send(ticket_message("two", channel=channel, message_id=2))
    second = (await h.repo.get_ticket("777")).last_escalation_at
    assert second == first_ping, "a suppressed escalation must not move last_escalation_at"

    on_cooldown, remaining = await h.repo.escalation_on_cooldown(
        "777", cooldown_seconds=900
    )
    assert on_cooldown is True and remaining <= 900


async def test_idle_channel_queues_are_released(harness):
    """The listener must not accumulate one queue per ticket channel ever seen."""
    h = harness([openai_ok("one"), openai_ok("two")])
    cog = await h.start()

    await h.send(ticket_message("first", channel=ticket_channel(channel_id=777), message_id=1))
    await h.send(
        ticket_message(
            "second", channel=ticket_channel(channel_id=778, name="ticket-0002"), message_id=2
        )
    )

    assert cog.stats["answered"] == 2
    assert cog._channels == {}, "idle queues must be discarded"
    assert cog.stats["active_channels"] == 0


# --------------------------------------------------------------------------- #
# gating
# --------------------------------------------------------------------------- #
async def test_channel_outside_the_ticket_category_is_ignored(harness):
    h = harness(openai_ok("should not be used"))
    await h.start()
    channel = FakeChannel(id=900, name="general", category_id=OTHER_CATEGORY_ID)

    await h.send(ticket_message("hello", channel=channel))

    assert channel.sent == []
    assert h.session.call_count == 0


async def test_bot_messages_are_ignored(harness):
    h = harness(openai_ok("unused"))
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message("beep boop", channel=channel, author_bot=True, author_id=2))

    assert channel.sent == []
    assert h.session.call_count == 0


async def test_bot_never_answers_itself(harness):
    h = harness(openai_ok("unused"))
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message("loop?", channel=channel, author_bot=True, author_id=BOT_USER_ID))

    assert channel.sent == []
    assert h.session.call_count == 0


async def test_unconfigured_server_is_ignored(harness):
    h = harness(openai_ok("unused"))
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message("hello", channel=channel, guild_id=OTHER_GUILD_ID))

    assert channel.sent == []
    assert h.session.call_count == 0
    assert await h.repo.get_guild(OTHER_GUILD_ID) is None, "no row may be created for a stranger"


async def test_direct_messages_are_ignored(harness):
    h = harness(openai_ok("unused"))
    await h.start()
    channel = ticket_channel()
    message = ticket_message("hello", channel=channel)
    message.guild = None

    await h.send(message)

    assert channel.sent == []
    assert h.session.call_count == 0


async def test_slash_command_echo_is_ignored(harness):
    h = harness(openai_ok("unused"))
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message("/setup-kb", channel=channel, interaction_metadata=object()))

    assert channel.sent == []
    assert h.session.call_count == 0


async def test_resolved_ticket_stops_the_ai(harness):
    h = harness(openai_ok("unused"))
    await h.start()
    channel = ticket_channel()

    await h.repo.ensure_guild(GUILD_ID, "Test Server")
    await h.repo.open_ticket(
        ticket_id="777", guild_id=GUILD_ID, channel_id="777", user_id=str(MEMBER_ID)
    )
    await h.repo.set_ticket_status("777", STATUS_RESOLVED)

    await h.send(ticket_message("anyone there?", channel=channel))

    assert channel.sent == []
    assert h.session.call_count == 0
    assert (await h.repo.get_ticket("777")).status == STATUS_RESOLVED


async def test_empty_message_is_ignored(harness):
    h = harness(openai_ok("unused"))
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message("   ", channel=channel))

    assert channel.sent == []
    assert h.session.call_count == 0


# --------------------------------------------------------------------------- #
# multi-tenancy
# --------------------------------------------------------------------------- #
async def test_two_tenants_get_their_own_knowledge_base(harness):
    other_category = 888
    h = harness([openai_ok("A answer"), openai_ok("B answer")])
    await h.start()

    # Configure a second tenant with its own category, KB and staff role.
    await h.repo.ensure_guild(OTHER_GUILD_ID, "Server B")
    await h.repo.set_knowledge_base(OTHER_GUILD_ID, "B: refunds in 30 days.", server_name="Server B")
    await h.repo.set_staff_role(OTHER_GUILD_ID, 4242)
    await h.repo.set_ticket_category(OTHER_GUILD_ID, other_category)

    channel_a = ticket_channel(channel_id=777)
    channel_b = FakeChannel(id=889, name="ticket-b", category_id=other_category)

    await h.send(ticket_message("refunds?", channel=channel_a))
    await h.send(
        ticket_message("refunds?", channel=channel_b, guild_id=OTHER_GUILD_ID, message_id=2)
    )

    assert channel_a.last_sent == "A answer"
    assert channel_b.last_sent == "B answer"

    prompt_a = h.session.calls[0]["json"]["messages"][0]["content"]
    prompt_b = h.session.calls[1]["json"]["messages"][0]["content"]
    assert "Refunds are available within 14 days" in prompt_a
    assert "B: refunds in 30 days" in prompt_b
    assert "B: refunds in 30 days" not in prompt_a, "tenant B KB leaked into tenant A"
    assert "Refunds are available within 14 days" not in prompt_b, "tenant A KB leaked into tenant B"
    assert 'Discord server "Test Server"' in prompt_a
    assert 'Discord server "Server B"' in prompt_b


# --------------------------------------------------------------------------- #
# delivery mechanics
# --------------------------------------------------------------------------- #
async def test_debounce_merges_a_burst_into_one_reply(harness):
    h = harness(openai_ok("Answered both questions."))
    await h.start(debounce=0.08)
    channel = ticket_channel()

    first = ticket_message("hi", channel=channel, message_id=1)
    second = ticket_message("also, how do refunds work?", channel=channel, message_id=2)
    await h.cog.handle_message(first)  # type: ignore[union-attr]
    await h.cog.handle_message(second)  # type: ignore[union-attr]
    await h._drain(channel.id, timeout=10)

    assert h.session.call_count == 1, "a burst must cost exactly one API call"
    assert len(channel.sent_text) == 1
    query = h.session.calls[0]["json"]["messages"][1]["content"]
    assert "hi" in query and "how do refunds work?" in query


async def test_long_reply_is_chunked_under_the_discord_limit(harness):
    long_answer = "\n\n".join(f"**Step {index}.** Do the thing." for index in range(400))
    assert len(long_answer) > 2000
    h = harness(openai_ok(long_answer))
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message(channel=channel))

    assert len(channel.sent_text) > 1
    assert all(len(chunk) <= 2000 for chunk in channel.sent_text)
    assert "".join(channel.sent_text).startswith("**Step 0.**")


async def test_messages_are_answered_in_order_per_channel(harness):
    h = harness([openai_ok("reply one"), openai_ok("reply two")])
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message("first", channel=channel, message_id=1))
    await h.send(ticket_message("second", channel=channel, message_id=2))

    assert channel.sent_text == ["reply one", "reply two"]


class _StubResponse:
    """Minimal stand-in for the aiohttp response discord.py wraps in errors."""

    status = 403
    reason = "Forbidden"


async def test_send_failure_does_not_crash_the_worker(harness):
    import discord

    h = harness([openai_ok("first"), openai_ok("second")])
    await h.start()
    channel = ticket_channel()
    channel.send_error = discord.Forbidden(
        _StubResponse(), "Cannot send messages in this channel"  # type: ignore[arg-type]
    )

    await h.send(ticket_message("hello", channel=channel))

    assert channel.sent == []
    assert h.cog is not None  # the cog survives a delivery failure

    channel.send_error = None
    await h.send(ticket_message("hello again", channel=channel, message_id=2))
    assert channel.sent_text == ["second"]


async def test_listener_stats_are_tracked(harness):
    h = harness([openai_ok("answer"), openai_ok(ESCALATION_REPLY)])
    cog = await h.start()
    channel = ticket_channel()

    await h.send(ticket_message("question", channel=channel, message_id=1))
    await h.send(ticket_message("refund me now", channel=channel, message_id=2))

    stats = cog.stats
    assert stats["messages_processed"] == 2
    assert stats["answered"] == 1
    assert stats["escalated"] == 1
    assert h.bot.listener_stats()["answered"] == 1


async def test_guild_join_creates_tenant_row_and_syncs_commands(harness):
    h = harness(openai_ok("unused"))
    cog = await h.start()

    guild = FakeGuild(id=OTHER_GUILD_ID, name="Fresh Server")
    await cog.sync_new_guild(guild)  # type: ignore[arg-type]

    record = await h.repo.get_guild(OTHER_GUILD_ID)
    assert record is not None
    assert record.server_name == "Fresh Server"
    assert record.knowledge_base == ""
    assert h.bot.synced_guilds == [OTHER_GUILD_ID]


async def test_empty_completion_escalates_rather_than_posting_nothing(harness):
    h = harness(openai_ok("   "))
    await h.start()
    channel = ticket_channel()

    await h.send(ticket_message(channel=channel))

    assert channel.sent, "an empty model reply must still produce a message"
    assert f"<@&{STAFF_ROLE_ID}>" in channel.last_sent
