"""The ticket listener: watch ticket channels, answer with the tenant's KB, escalate when unsure.

Message flow
------------
::

    on_message
      └─ cheap pre-filters (DM / bot / webhook / self)      ← no DB, no await
      └─ tenant config lookup                               ← cached, negative-cached
      └─ TicketResolver gate (category / thread / status)   ← pure function
      └─ debounce into a per-channel batch                  ← coalesces bursts
      └─ per-channel worker (serialised, semaphore-bounded)
            ├─ fetch last N messages → chat memory
            ├─ PromptBuilder → tenant-scoped system prompt
            ├─ pre-flight escalation triggers (safety)      ← no API call spent
            ├─ AIService.generate → Groq → Cerebras → Gemini
            ├─ escalation parsing → safe staff-role mention
            └─ chunked delivery + ticket_logs/ticket_activity updates

Three properties matter for a multi-tenant bot and are enforced here:

* **No cross-talk.** Config, KB and staff role are always keyed by the message's
  own ``guild_id``.
* **No infinite loops.** The bot never answers itself, never answers other bots,
  and serialises per channel so replies stay ordered.
* **Fail towards humans.** If every AI provider fails, the ticket is escalated to
  staff rather than silently dropped.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import discord
from discord.ext import commands

from bot.constants import (
    DISCORD_MESSAGE_LIMIT,
    STATUS_RESOLVED,
)
from bot.db.repository import GuildConfigRecord, TicketRecord
from bot.services.ai_service import AIResult
from bot.services.escalation import (
    EscalationDecision,
    ReplyKind,
    detect_trigger,
    escalation_message,
    parse_ai_reply,
)
from bot.services.prompt_builder import PromptBuilder
from bot.services.ticket_resolver import SkipReason, TicketDecision, TicketResolver, extract_query
from bot.utils.logging_setup import get_logger
from bot.utils.text import chunk_message, truncate

if TYPE_CHECKING:  # pragma: no cover
    from bot.main import TicketBot

log = get_logger(__name__)

#: How long to wait for a burst of messages to settle before answering once.
DEFAULT_DEBOUNCE_SECONDS = 1.75


@dataclass
class ChannelQueue:
    """Per-ticket-channel buffer + worker, guaranteeing serialised replies."""

    channel_id: int
    guild_id: int
    buffer: list[discord.Message] = field(default_factory=list)
    worker: asyncio.Task | None = None
    last_reply_at: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def drain(self) -> list[discord.Message]:
        pending, self.buffer = self.buffer, []
        return pending


class TicketListener(commands.Cog):
    """Answers member questions inside configured ticket channels/threads."""

    def __init__(
        self,
        bot: "TicketBot",
        *,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
    ) -> None:
        self.bot = bot
        self.settings = bot.settings
        self.repo = bot.repository
        self.ai = bot.ai
        self.debounce_seconds = max(0.0, debounce_seconds)

        self.resolver = TicketResolver(
            bot_user_id=0,  # patched in cog_load once the gateway gives us our id
            require_configured_category=self.settings.require_configured_category,
            ticket_name_pattern=self.settings.ticket_name_pattern,
        )
        self.prompt_builder = PromptBuilder(
            history_message_limit=self.settings.history_message_limit,
            history_char_budget=self.settings.history_char_budget,
        )

        self._channels: dict[int, ChannelQueue] = {}
        self._channels_guard = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.settings.max_concurrent_tickets)
        self._processed = 0
        self._answered = 0
        self._escalated = 0
        self._skipped: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def cog_load(self) -> None:
        if self.bot.user is not None:
            self.resolver.bot_user_id = self.bot.user.id
        log.info(
            "Ticket listener ready (history=%d messages, debounce=%.2fs, max_concurrent=%d)",
            self.settings.history_message_limit,
            self.debounce_seconds,
            self.settings.max_concurrent_tickets,
        )

    async def cog_unload(self) -> None:
        for queue in list(self._channels.values()):
            if queue.worker is not None and not queue.worker.done():
                queue.worker.cancel()
        self._channels.clear()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "messages_processed": self._processed,
            "answered": self._answered,
            "escalated": self._escalated,
            "active_channels": len(self._channels),
            "skipped": dict(self._skipped),
        }

    def _count_skip(self, reason: SkipReason) -> None:
        key = reason.value
        self._skipped[key] = self._skipped.get(key, 0) + 1

    # ------------------------------------------------------------------ #
    # Discord events
    # ------------------------------------------------------------------ #
    @commands.Cog.listener("on_message")
    async def handle_message(self, message: discord.Message) -> None:
        """Entry point for every message the bot can see."""
        # --- Stage 1: free filters (no I/O, no DB) ----------------------- #
        if message.guild is None or message.author.bot or message.webhook_id is not None:
            return
        if self.bot.user is not None and message.author.id == self.bot.user.id:
            return

        config = await self._config_for(message)
        if config is None:
            return  # unconfigured tenant — negative-cached, so this is cheap

        decision = self.resolver.resolve(message, config)
        if not decision.allowed or decision.context is None:
            if decision.reason not in {SkipReason.NOT_CONFIGURED, SkipReason.EMPTY}:
                log.debug(
                    "Ignoring message %s in guild %s: %s",
                    message.id, message.guild.id, decision.reason.value,
                )
            self._count_skip(decision.reason)
            return

        self._processed += 1
        await self._enqueue(decision, message, config)

    @commands.Cog.listener("on_guild_join")
    async def sync_new_guild(self, guild: discord.Guild) -> None:
        """Create the tenant row eagerly and push slash commands to the new server."""
        try:
            await self.repo.ensure_guild(guild.id, guild.name)
        except Exception:  # pragma: no cover - never fail a join on bookkeeping
            log.exception("Could not create server_configs row for guild %s", guild.id)
        await self.bot.sync_commands_for(guild)

    @commands.Cog.listener("on_guild_available")
    async def refresh_guild_name(self, guild: discord.Guild) -> None:
        """Keep stored server names fresh (used inside the system prompt)."""
        try:
            await self.repo.rename_guild(guild.id, guild.name)
        except Exception:  # pragma: no cover
            log.debug("Name refresh skipped for guild %s", guild.id, exc_info=True)

    # ------------------------------------------------------------------ #
    # queueing / debounce
    # ------------------------------------------------------------------ #
    async def _config_for(self, message: discord.Message) -> GuildConfigRecord | None:
        try:
            return await self.repo.get_guild(message.guild.id)
        except Exception:
            log.exception("Database error while loading config for guild %s", message.guild.id)
            return None

    async def _enqueue(
        self, decision: TicketDecision, message: discord.Message, config: GuildConfigRecord
    ) -> None:
        """Add the message to its channel batch and make sure a worker exists."""
        context = decision.context
        assert context is not None  # guaranteed by the caller
        queue = await self._queue_for(context.channel_id, context.guild_id)

        async with queue.lock:
            queue.buffer.append(message)
            if queue.worker is None or queue.worker.done():
                queue.worker = asyncio.create_task(
                    self._channel_worker(queue),
                    name=f"ticket-worker-{context.channel_id}",
                )

    async def _discard_queue(self, queue: ChannelQueue) -> None:
        """Forget an idle channel queue (its worker has exited, buffer empty)."""
        async with self._channels_guard:
            current = self._channels.get(queue.channel_id)
            if current is queue and not current.buffer and current.worker is None:
                del self._channels[queue.channel_id]

    async def _queue_for(self, channel_id: int, guild_id: int) -> ChannelQueue:
        async with self._channels_guard:
            queue = self._channels.get(channel_id)
            if queue is None:
                queue = ChannelQueue(channel_id=channel_id, guild_id=guild_id)
                self._channels[channel_id] = queue
            return queue

    async def _channel_worker(self, queue: ChannelQueue) -> None:
        """Serialised worker for one ticket channel.

        Debouncing here is what makes the bot feel calm: members typically fire
        three short messages in a row, and answering each one separately both
        wastes free-tier quota and produces overlapping replies.
        """
        try:
            while True:
                if self.debounce_seconds:
                    await asyncio.sleep(self.debounce_seconds)
                batch = queue.drain()
                if not batch:
                    return

                # Honour the per-user cooldown by delaying, never by dropping.
                wait = self._min_interval_wait(queue)
                if wait > 0:
                    await asyncio.sleep(wait)
                    # More messages may have landed while we waited; include them.
                    batch = batch + queue.drain()

                async with self._semaphore:
                    await self._answer_batch(queue, batch)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Unhandled error in ticket worker for channel %s", queue.channel_id)
        finally:
            async with queue.lock:
                queue.worker = None
                if queue.buffer:
                    # Messages can arrive between the last drain and this point.
                    queue.worker = asyncio.create_task(
                        self._channel_worker(queue), name=f"ticket-worker-{queue.channel_id}"
                    )
                else:
                    # Nothing pending: drop the queue so long-running processes
                    # do not accumulate one entry per ticket channel ever seen.
                    await self._discard_queue(queue)

    def _min_interval_wait(self, queue: ChannelQueue) -> float:
        cooldown = self.settings.user_cooldown_seconds
        if cooldown <= 0 or queue.last_reply_at == 0:
            return 0.0
        elapsed = time.monotonic() - queue.last_reply_at
        return max(0.0, cooldown - elapsed)

    # ------------------------------------------------------------------ #
    # core answering path
    # ------------------------------------------------------------------ #
    async def _answer_batch(self, queue: ChannelQueue, batch: Sequence[discord.Message]) -> None:
        """Answer one debounced batch of member messages."""
        latest = batch[-1]
        guild = latest.guild
        channel = latest.channel
        if guild is None:
            return

        config = await self.repo.get_guild(guild.id)
        if config is None:  # config removed while the batch was pending
            return

        ticket = await self._ensure_ticket(latest, config, queue)
        if ticket is not None and ticket.status == STATUS_RESOLVED:
            log.info(
                "Ticket %s is resolved; ignoring %d message(s).", ticket.ticket_id, len(batch)
            )
            self._count_skip(SkipReason.TICKET_CLOSED)
            return

        query = self._combine_query(batch)
        if not query.strip():
            return

        ticket_id = ticket.ticket_id if ticket else str(channel.id)
        history = await self._fetch_history(channel, exclude={m.id for m in batch})

        # --- pre-flight escalation (saves an API call, protects members) --- #
        trigger = detect_trigger(
            query,
            check_safety=self.settings.escalation_preflight_safety,
            check_requests=self.settings.escalation_preflight_requests,
        )

        async with self._typing(channel):
            if trigger.matched:
                log.info(
                    "Pre-flight escalation (%s: %r) in guild %s channel %s",
                    trigger.tier.value, trigger.matched_text, guild.id, channel.id,
                )
                decision = EscalationDecision(
                    kind=ReplyKind.ESCALATE,
                    content=escalation_message(
                        f"Automated {trigger.tier.value} trigger matched: “{trigger.matched_text}”."
                    ),
                    staff_mention=self._staff_mention(config),
                    reason=f"preflight:{trigger.tier.value}",
                    trigger=trigger,
                    staff_role_missing=config.staff_role_id is None,
                )
                await self._deliver(queue, channel, decision, ticket_id, guild)
                return

            prompt = self.prompt_builder.build(
                server_name=config.server_name or guild.name,
                knowledge_base=config.knowledge_base,
                staff_role_id=config.staff_role_id,
                latest_message=query,
                history=history,
                bot_user_id=self.bot.user.id if self.bot.user else 0,
            )
            if prompt.truncated:
                log.warning(
                    "Knowledge base for guild %s exceeded the prompt cap and was truncated.", guild.id
                )

            result: AIResult = await self.ai.generate(
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt,
                guild_id=guild.id,
            )

        if not result.ok:
            await self._handle_ai_failure(queue, channel, ticket_id, guild, config, result)
            return

        decision = parse_ai_reply(result.text, staff_role_id=config.staff_role_id)
        log.info(
            "AI answered ticket %s via %s in %dms → %s",
            ticket_id, result.provider, result.latency_ms, decision.kind.value,
        )
        await self._deliver(queue, channel, decision, ticket_id, guild, ai_result=result)

    async def _deliver(
        self,
        queue: ChannelQueue,
        channel: discord.abc.Messageable,
        decision: EscalationDecision,
        ticket_id: str,
        guild: discord.Guild,
        *,
        ai_result: AIResult | None = None,
    ) -> None:
        """Post the answer or the escalation, then persist the outcome."""
        if ai_result is not None:
            log.debug(
                "Ticket %s served by %s/%s in %dms (finish=%s, truncated=%s)",
                ticket_id, ai_result.provider, ai_result.model, ai_result.latency_ms,
                ai_result.finish_reason, ai_result.truncated,
            )
        escalated = decision.should_escalate
        ping_staff = escalated and decision.staff_mention is not None
        suppressed = False

        if escalated and ping_staff:
            on_cooldown, remaining = await self.repo.escalation_on_cooldown(
                channel.id, cooldown_seconds=self.settings.escalation_cooldown_seconds
            )
            if on_cooldown:
                # Staff were already pinged for this ticket recently: post the
                # status update without pinging them a second time.
                ping_staff = False
                suppressed = True
                log.info(
                    "Escalation ping suppressed in channel %s (cooldown, %.0fs left)",
                    channel.id, remaining,
                )

        body = decision.content if decision.content else escalation_message()
        mention = decision.staff_mention if ping_staff else None
        text = f"{body}\n\n{mention}" if mention else body

        if escalated and decision.staff_role_missing:
            text = (
                f"{body}\n\n⚠️ *No staff role is configured, so nobody was pinged. "
                f"An administrator should run `/set-staff-role`.*"
            )

        sent = await self._send(channel, text)
        queue.last_reply_at = time.monotonic()

        try:
            if escalated:
                self._escalated += 1
                await self.repo.record_escalation(
                    guild_id=guild.id,
                    channel_id=channel.id,
                    ticket_id=ticket_id,
                    pinged=not suppressed,
                )
            elif sent:
                self._answered += 1
                await self.repo.record_ai_reply(
                    guild_id=guild.id, channel_id=channel.id, ticket_id=ticket_id
                )
        except Exception:  # pragma: no cover - telemetry must not break delivery
            log.exception("Failed to persist ticket outcome for %s", ticket_id)

        if suppressed:
            log.debug("Escalation delivered without ping for ticket %s", ticket_id)

    async def _handle_ai_failure(
        self,
        queue: ChannelQueue,
        channel: discord.abc.Messageable,
        ticket_id: str,
        guild: discord.Guild,
        config: GuildConfigRecord,
        result: AIResult,
    ) -> None:
        """Every provider failed — fail towards a human instead of going silent."""
        log.error(
            "All AI providers failed for ticket %s (guild %s): %s", ticket_id, guild.id, result.error
        )
        mention = self._staff_mention(config)
        body = (
            "I'm temporarily unable to process this ticket because the AI service is unavailable. "
            "Your question has not been lost."
        )
        decision = EscalationDecision(
            kind=ReplyKind.ESCALATE,
            content=escalation_message(body),
            staff_mention=mention,
            reason=f"ai_unavailable:{result.error[:120]}",
            staff_role_missing=mention is None,
        )
        await self._deliver(queue, channel, decision, ticket_id, guild)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _staff_mention(self, config: GuildConfigRecord) -> str | None:
        role_id = config.staff_role_id
        if not role_id or not str(role_id).isdigit():
            return None
        return f"<@&{role_id}>"

    def _combine_query(self, batch: Sequence[discord.Message]) -> str:
        """Merge a debounced burst into one query string."""
        parts: list[str] = []
        for message in batch:
            text = extract_query(message)
            if text:
                parts.append(text)
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        joined = "\n".join(parts)
        return truncate(joined, DISCORD_MESSAGE_LIMIT)

    async def _ensure_ticket(
        self, message: discord.Message, config: GuildConfigRecord, queue: ChannelQueue
    ) -> TicketRecord | None:
        """Create/refresh the ``ticket_logs`` row for this channel."""
        try:
            await self.repo.record_user_message(
                guild_id=message.guild.id,
                channel_id=message.channel.id,
                user_id=message.author.id,
                ticket_id=str(message.channel.id),
            )
            return await self.repo.open_ticket(
                ticket_id=str(message.channel.id),
                guild_id=message.guild.id,
                channel_id=message.channel.id,
                user_id=message.author.id,
            )
        except Exception:
            log.exception("Could not persist ticket row for channel %s", message.channel.id)
            return None

    async def _fetch_history(
        self, channel: discord.abc.Messageable, *, exclude: set[int]
    ) -> list[discord.Message]:
        """Last N messages, oldest first, excluding the ones we are answering now."""
        limit = max(1, self.settings.history_message_limit)
        history: list[discord.Message] = []
        try:
            async for message in channel.history(limit=limit + len(exclude) + 5):
                if message.id in exclude:
                    continue
                history.append(message)
                if len(history) >= limit:
                    break
        except (discord.Forbidden, discord.NotFound) as exc:
            log.warning("Cannot read history in channel %s: %s", getattr(channel, "id", "?"), exc)
        except Exception:  # pragma: no cover - unexpected gateway issue
            log.exception("Unexpected error reading history for channel %s", getattr(channel, "id", "?"))
        history.reverse()  # Discord yields newest-first; the prompt wants chronological
        return history

    def _typing(self, channel: discord.abc.Messageable) -> "_TypingGuard":
        return _TypingGuard(channel)

    async def _send(self, channel: discord.abc.Messageable, text: str) -> bool:
        """Send a reply, chunked to Discord's 2000-character limit."""
        if not text.strip():
            return False
        chunks = chunk_message(text)
        sent_any = False
        for index, chunk in enumerate(chunks):
            try:
                await channel.send(chunk)
                sent_any = True
            except discord.Forbidden:
                log.error("Missing permission to send messages in channel %s", getattr(channel, "id", "?"))
                return sent_any
            except discord.NotFound:
                log.info("Channel %s disappeared before the reply was sent.", getattr(channel, "id", "?"))
                return sent_any
            except discord.HTTPException:
                log.exception("Failed to send chunk %d/%d in channel %s", index + 1, len(chunks),
                              getattr(channel, "id", "?"))
            except Exception:  # pragma: no cover - defensive
                log.exception("Unexpected error sending reply to channel %s", getattr(channel, "id", "?"))
            if index + 1 < len(chunks):
                await asyncio.sleep(0.4)  # stay well inside Discord's rate limits
        return sent_any


class _TypingGuard:
    """Best-effort typing indicator that never breaks the reply path."""

    def __init__(self, channel: discord.abc.Messageable) -> None:
        self.channel = channel
        self._cm: Any = None

    async def __aenter__(self) -> "_TypingGuard":
        try:
            self._cm = self.channel.typing()
            await self._cm.__aenter__()
        except Exception:  # pragma: no cover - typing is cosmetic
            self._cm = None
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._cm is None:
            return
        try:
            await self._cm.__aexit__(*exc_info)
        except Exception:  # pragma: no cover
            pass


async def setup(bot: "TicketBot") -> None:
    await bot.add_cog(TicketListener(bot))
