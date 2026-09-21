"""Decide *whether* a message should be answered, independent of Discord I/O.

All of the gating rules live here as a pure function of (message, tenant config)
so they can be exhaustively unit tested without a gateway connection. The
listener cog simply acts on the returned :class:`TicketDecision`.

Gate order (cheapest and most common rejections first):

1. Not from a guild (DM)                     -> ignore
2. Author is a bot / webhook / system message -> ignore  (spec: "ignore bots")
3. Slash-command echo                        -> ignore
4. Empty content and no attachments          -> ignore
5. Server has no config row                  -> ignore  (never answer unconfigured tenants)
6. Ticket closed/resolved                    -> ignore
7. Channel/thread not inside the configured ticket category -> ignore
   (falls back to a name pattern only when REQUIRE_CONFIGURED_CATEGORY=false)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Final

from bot.constants import ATTACHMENT_PLACEHOLDER, MAX_ATTACHMENT_BYTES
from bot.db.repository import GuildConfigRecord
from bot.utils.logging_setup import get_logger

log = get_logger(__name__)


class SkipReason(str, Enum):
    """Why a message was not answered. Logged at DEBUG for operator triage."""

    ALLOWED = "allowed"
    NOT_GUILD = "direct message"
    BOT_AUTHOR = "author is a bot"
    SELF_AUTHOR = "message from this bot"
    WEBHOOK = "message from a webhook"
    SYSTEM_MESSAGE = "system/non-default message type"
    COMMAND_ECHO = "slash command invocation"
    EMPTY = "no text content"
    NOT_CONFIGURED = "server has no configuration row"
    NO_KNOWLEDGE_BASE = "server knowledge base is empty"
    TICKET_CLOSED = "ticket is resolved or closed"
    WRONG_CATEGORY = "channel is outside the configured ticket category"
    CATEGORY_UNSET = "server has no ticket category configured"
    NAME_MISMATCH = "channel name does not match the ticket pattern"
    UNKNOWN_CHANNEL = "channel could not be resolved"


@dataclass(frozen=True)
class TicketContext:
    """Tenant + channel identifiers resolved for one message."""

    guild_id: int
    guild_name: str
    channel_id: int
    channel_name: str
    ticket_id: str
    category_id: int | None
    is_thread: bool
    parent_channel_id: int | None = None
    author_id: int = 0
    author_name: str = ""

    @property
    def log_label(self) -> str:
        return f"guild={self.guild_id} channel={self.channel_id} ticket={self.ticket_id}"


@dataclass(frozen=True)
class TicketDecision:
    """Result of the gate: answer this message, or ignore it (with a reason)."""

    allowed: bool
    reason: SkipReason
    context: TicketContext | None = None
    detail: str = ""

    def __bool__(self) -> bool:
        return self.allowed


class TicketResolver:
    """Stateless message gate. One instance is shared by the whole process."""

    def __init__(
        self,
        *,
        bot_user_id: int = 0,
        require_configured_category: bool = True,
        ticket_name_pattern: re.Pattern[str] | None = None,
        require_knowledge_base: bool = False,
    ) -> None:
        self.bot_user_id = bot_user_id
        self.require_configured_category = require_configured_category
        self.ticket_name_pattern = ticket_name_pattern or re.compile(
            r"^(ticket|support|help)[-_]", re.IGNORECASE
        )
        # When true, a server without a KB gets silence instead of a stream of
        # escalations. Default false: the bot escalates to staff, which is the
        # more useful behaviour while an admin is still setting up.
        self.require_knowledge_base = require_knowledge_base

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def resolve(
        self,
        message: object,
        config: GuildConfigRecord | None,
        *,
        ticket_status: str | None = None,
    ) -> TicketDecision:
        """Apply every gate in order and return the verdict."""
        if message is None:
            return TicketDecision(False, SkipReason.UNKNOWN_CHANNEL, detail="message is None")

        guild = getattr(message, "guild", None)
        if guild is None:
            return TicketDecision(False, SkipReason.NOT_GUILD)

        channel = getattr(message, "channel", None)
        if channel is None:
            return TicketDecision(False, SkipReason.UNKNOWN_CHANNEL)

        author = getattr(message, "author", None)
        author_id = getattr(author, "id", 0) or 0

        if self.bot_user_id and author_id == self.bot_user_id:
            return TicketDecision(False, SkipReason.SELF_AUTHOR)
        if getattr(author, "bot", False):
            return TicketDecision(False, SkipReason.BOT_AUTHOR)
        if getattr(message, "webhook_id", None):
            return TicketDecision(False, SkipReason.WEBHOOK)
        if not _is_default_message_type(message):
            return TicketDecision(False, SkipReason.SYSTEM_MESSAGE)
        if getattr(message, "interaction_metadata", None) is not None:
            return TicketDecision(False, SkipReason.COMMAND_ECHO)

        query = extract_query(message)
        if not query.strip():
            return TicketDecision(False, SkipReason.EMPTY)

        guild_id = getattr(guild, "id", 0)
        context = TicketContext(
            guild_id=int(guild_id),
            guild_name=getattr(guild, "name", "") or "",
            channel_id=int(getattr(channel, "id", 0) or 0),
            channel_name=getattr(channel, "name", "") or "",
            # A Discord ticket channel/thread *is* the ticket, so its snowflake
            # is the natural, stable ticket_id.
            ticket_id=str(getattr(channel, "id", 0) or 0),
            category_id=resolve_category_id(channel),
            is_thread=_is_thread(channel),
            parent_channel_id=_parent_id(channel),
            author_id=int(author_id),
            author_name=getattr(author, "display_name", "") or getattr(author, "name", "") or "",
        )

        if config is None:
            return TicketDecision(False, SkipReason.NOT_CONFIGURED, context=context)

        if self.require_knowledge_base and not config.has_knowledge_base:
            return TicketDecision(False, SkipReason.NO_KNOWLEDGE_BASE, context=context)

        if ticket_status is not None and str(ticket_status).lower() in {"resolved", "closed"}:
            return TicketDecision(False, SkipReason.TICKET_CLOSED, context=context)

        area_reason = self._in_ticket_area(context, config)
        if area_reason is not SkipReason.ALLOWED:
            return TicketDecision(False, area_reason, context=context)

        return TicketDecision(True, SkipReason.ALLOWED, context=context)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _in_ticket_area(
        self, context: TicketContext, config: GuildConfigRecord
    ) -> SkipReason:
        """Is this channel/thread one the tenant asked us to serve?

        Returns :attr:`SkipReason.ALLOWED` when it is, otherwise the precise
        reason it is not — operators read these reasons in the logs to work out
        why the bot is silent in a given channel.

        Strict mode (``REQUIRE_CONFIGURED_CATEGORY=true``, the default) means the
        category set by ``/set-ticket-category`` is the *only* scope: a matching
        channel is served, everything else — including a server that has not
        configured a category at all — is ignored. That is what makes the setting
        a real restriction rather than a hint.

        Permissive mode additionally serves channels whose name matches
        ``TICKET_NAME_PATTERN``, which suits servers whose ticket tool creates
        channels outside a dedicated category.
        """
        configured = config.ticket_category_id
        if configured:
            if context.category_id is not None and str(context.category_id) == str(configured):
                return SkipReason.ALLOWED
            if self.require_configured_category:
                return SkipReason.WRONG_CATEGORY
        elif self.require_configured_category:
            # No category configured and strict mode on: stay silent until an
            # admin runs /set-ticket-category.
            return SkipReason.CATEGORY_UNSET

        # Permissive mode: a ticket-looking channel name is enough.
        if self.ticket_name_pattern.match(context.channel_name or ""):
            return SkipReason.ALLOWED
        return (
            SkipReason.NAME_MISMATCH
            if not configured
            else SkipReason.WRONG_CATEGORY
        )


# --------------------------------------------------------------------------- #
# helpers (module-level so they are reusable and directly testable)
# --------------------------------------------------------------------------- #

DEFAULT_MESSAGE_TYPE_VALUE: Final[int] = 0


def _is_default_message_type(message: object) -> bool:
    """True for ordinary chat messages (not joins, boosts, pins, replies-notices…)."""
    message_type = getattr(message, "type", None)
    if message_type is None:
        return True
    value = getattr(message_type, "value", message_type)
    try:
        return int(value) == DEFAULT_MESSAGE_TYPE_VALUE
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return True


def _is_thread(channel: object) -> bool:
    """Detect a thread without importing discord (keeps this module I/O-free)."""
    if getattr(channel, "parent", None) is not None and hasattr(channel, "type"):
        # discord.Thread exposes `parent`; a TextChannel's parent is a category.
        channel_type = getattr(channel.type, "value", channel.type)
        # 10/11/12 = news/public/private threads; forum/media threads too.
        if channel_type in {10, 11, 12, 15}:
            return True
        class_name = type(channel).__name__.lower()
        if "thread" in class_name:
            return True
    return False


def _parent_id(channel: object) -> int | None:
    parent = getattr(channel, "parent", None)
    if parent is None:
        return None
    # For a TextChannel, `parent` is its Category — not a ticket parent.
    if _is_thread(channel):
        return int(getattr(parent, "id", 0) or 0) or None
    return None


def resolve_category_id(channel: object) -> int | None:
    """Best-effort category id for a channel *or* a thread inside a channel."""
    if channel is None:
        return None

    if _is_thread(channel):
        parent = getattr(channel, "parent", None)
        if parent is not None:
            return resolve_category_id(parent)
        return None

    category_id = getattr(channel, "category_id", None)
    if category_id is not None:
        try:
            return int(category_id)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    category = getattr(channel, "category", None)
    if category is not None:
        try:
            return int(getattr(category, "id", 0) or 0) or None
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None
    return None


def extract_query(message: object) -> str:
    """Member-visible text of a message, plus a note for any attachments.

    Attachments are described (not transcribed) because the configured models
    are text-only; naming the file lets the model — and staff — understand that
    the member sent evidence the AI cannot read.
    """
    parts: list[str] = []
    content = (getattr(message, "content", "") or "").strip()
    if content:
        parts.append(content)

    for attachment in getattr(message, "attachments", None) or ():
        filename = getattr(attachment, "filename", None) or "file"
        size = getattr(attachment, "size", 0) or 0
        if size > MAX_ATTACHMENT_BYTES:
            note = f"[attached file: {filename} ({size} bytes — not read)]"
        else:
            note = ATTACHMENT_PLACEHOLDER.format(name=filename)
        parts.append(note)

    for embed in getattr(message, "embeds", None) or ():
        title = getattr(embed, "title", None)
        if title:
            parts.append(f"[embed: {title}]")

    return "\n".join(parts).strip()
