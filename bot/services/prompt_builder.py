"""Render the per-tenant system prompt and the chat-memory transcript.

This module is deliberately free of ``discord.py`` imports so it can be unit
tested with plain strings, and so the exact prompt contract from the
specification stays visible in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from bot.constants import (
    EMPTY_KB_PLACEHOLDER,
    ESCALATION_MESSAGE,
    ESCALATION_TOKEN,
    HISTORY_MESSAGE_CHAR_LIMIT,
    MAX_SYSTEM_PROMPT_CHARS,
    SYSTEM_PROMPT_TEMPLATE,
    UNCONFIGURED_STAFF_ROLE,
)
from bot.utils.text import truncate


class HistoryMessage(Protocol):
    """The minimum surface a message object must expose to enter chat memory.

    ``discord.Message`` satisfies this structurally; tests use a tiny stub.
    """

    @property
    def author_id(self) -> int: ...

    @property
    def author_is_bot(self) -> bool: ...

    @property
    def content(self) -> str: ...


@dataclass(frozen=True)
class TranscriptLine:
    """One normalised chat-history entry."""

    speaker: str  # "User" or "AI"
    text: str

    def render(self) -> str:
        return f"{self.speaker}: {self.text}"


def _clean_content(content: str) -> str:
    """Collapse whitespace so one message cannot span dozens of history lines."""
    return " ".join((content or "").split())


def build_transcript(
    messages: Sequence[object],
    *,
    bot_user_id: int,
    message_limit: int = 10,
    per_message_char_limit: int = HISTORY_MESSAGE_CHAR_LIMIT,
    char_budget: int = 6000,
    attachment_text: str = "",
    staff_role_id: str | int | None = None,
) -> list[TranscriptLine]:
    """Convert raw messages into ``User: …`` / ``AI: …`` history lines.

    * Oldest-first ordering is enforced regardless of the input order.
    * Only the last ``message_limit`` messages are kept (spec: last 10).
    * The bot's own messages are labelled ``AI``; everything else is ``User``.
      Other humans (staff replying in the ticket) are labelled ``Staff`` so the
      model does not mistake a moderator's answer for its own prior turn.
    * The newest turns are protected: when the character budget is exceeded the
      oldest lines are dropped first.
    """
    if not messages:
        return []

    ordered = _sort_chronologically(messages)[-message_limit:]

    lines: list[TranscriptLine] = []
    for message in ordered:
        text = _clean_content(getattr(message, "content", "") or "")
        if not text:
            text = attachment_text or _attachment_placeholder(message)
        if not text:
            continue
        speaker = _speaker_for(message, bot_user_id, staff_role_id)
        lines.append(TranscriptLine(speaker=speaker, text=truncate(text, per_message_char_limit)))

    return _fit_budget(lines, char_budget)


def _sort_chronologically(messages: Iterable[object]) -> list[object]:
    def key(message: object) -> tuple[int, int]:
        created = getattr(message, "created_at", None)
        raw_id = getattr(message, "id", 0) or 0
        # Discord snowflake IDs are monotonic, so they are a reliable fallback
        # when timestamps are equal or absent (as in unit-test stubs).
        epoch = int(getattr(created, "timestamp", lambda: 0)() * 1000) if created else 0
        return (epoch, int(raw_id))

    try:
        return sorted(messages, key=key)
    except TypeError:  # pragma: no cover - incomparable stubs
        return list(messages)


def _speaker_for(message: object, bot_user_id: int, staff_role_id: str | int | None = None) -> str:
    """Label a history message as ``AI``, ``Staff`` or ``User``.

    Getting this right matters: if a moderator's reply were labelled ``User``,
    the model would treat staff guidance as another customer question, and if
    the bot's own reply were labelled ``User`` it would repeat itself.
    """
    author = getattr(message, "author", None)
    author_id = getattr(author, "id", None) or getattr(message, "author_id", None)
    is_bot = getattr(author, "bot", None)
    if is_bot is None:
        is_bot = getattr(message, "author_is_bot", False)

    if bot_user_id and author_id is not None and int(author_id) == int(bot_user_id):
        return "AI"
    if is_bot:
        # Another bot (e.g. a ticket-tool webhook) — never treat as the user.
        return "AI"
    if staff_role_id and _has_role(author, staff_role_id):
        return "Staff"
    return "User"


def _has_role(author: object, staff_role_id: str | int) -> bool:
    """True when the author carries the tenant's staff role."""
    roles = getattr(author, "roles", None) or ()
    try:
        target = int(staff_role_id)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    for role in roles:
        role_id = getattr(role, "id", role)
        try:
            if int(role_id) == target:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _attachment_placeholder(message: object) -> str:
    attachments = getattr(message, "attachments", None) or ()
    names = [getattr(attachment, "filename", "file") for attachment in attachments]
    if not names:
        return ""
    return "[attached: " + ", ".join(str(name) for name in names[:3]) + "]"


def _fit_budget(lines: Sequence[TranscriptLine], char_budget: int) -> list[TranscriptLine]:
    """Drop the oldest lines until the transcript fits the character budget."""
    if char_budget <= 0:
        return list(lines)
    total = sum(len(line.render()) + 1 for line in lines)
    result = list(lines)
    while total > char_budget and len(result) > 1:
        removed = result.pop(0)
        total -= len(removed.render()) + 1
    return result


def render_chat_history(lines: Sequence[TranscriptLine]) -> str:
    """Serialise transcript lines for the ``=== CHAT HISTORY ===`` block."""
    if not lines:
        return "(no previous messages in this ticket)"
    return "\n".join(line.render() for line in lines)


def escalation_instruction(staff_role_id: str | int | None) -> str:
    """The exact string the model must emit, with the tenant's staff role baked in.

    When a server has not configured a staff role we still emit the token so the
    escalation is detected and logged — the mention itself is simply omitted.
    """
    if staff_role_id:
        mention = f"<@&{staff_role_id}>"
        return f"{ESCALATION_TOKEN} {ESCALATION_MESSAGE} {mention}"
    return f"{ESCALATION_TOKEN} {ESCALATION_MESSAGE} ({UNCONFIGURED_STAFF_ROLE})"


@dataclass(frozen=True)
class RenderedPrompt:
    """The fully rendered prompt pair for one inference call."""

    system_prompt: str
    user_prompt: str
    knowledge_base_chars: int
    history_lines: int
    truncated: bool = False

    @property
    def total_chars(self) -> int:
        return len(self.system_prompt) + len(self.user_prompt)


class PromptBuilder:
    """Assembles tenant-scoped prompts from config + chat memory."""

    def __init__(
        self,
        *,
        history_message_limit: int = 10,
        history_char_budget: int = 6000,
        max_system_prompt_chars: int = MAX_SYSTEM_PROMPT_CHARS,
    ) -> None:
        self.history_message_limit = history_message_limit
        self.history_char_budget = history_char_budget
        self.max_system_prompt_chars = max_system_prompt_chars

    def build(
        self,
        *,
        server_name: str,
        knowledge_base: str,
        staff_role_id: str | int | None,
        latest_message: str,
        history: Sequence[object] = (),
        bot_user_id: int = 0,
    ) -> RenderedPrompt:
        kb_text = (knowledge_base or "").strip() or EMPTY_KB_PLACEHOLDER
        transcript = build_transcript(
            history,
            bot_user_id=bot_user_id,
            message_limit=self.history_message_limit,
            char_budget=self.history_char_budget,
            staff_role_id=staff_role_id,
        )
        chat_history = render_chat_history(transcript)
        escalation_response = escalation_instruction(staff_role_id)

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            server_name=(server_name or "this server").strip(),
            knowledge_base=kb_text,
            escalation_response=escalation_response,
            chat_history=chat_history,
            latest_message=_clean_content(latest_message) or "(empty message)",
        )

        truncated = False
        if len(system_prompt) > self.max_system_prompt_chars:
            # Shrink the knowledge base (largest block) rather than dropping the
            # directives or the current query, which must always survive intact.
            overflow = len(system_prompt) - self.max_system_prompt_chars
            kb_text = truncate(kb_text, max(500, len(kb_text) - overflow), suffix="\n…[truncated]")
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                server_name=(server_name or "this server").strip(),
                knowledge_base=kb_text,
                escalation_response=escalation_response,
                chat_history=chat_history,
                latest_message=_clean_content(latest_message) or "(empty message)",
            )
            truncated = True
            if len(system_prompt) > self.max_system_prompt_chars:
                system_prompt = system_prompt[: self.max_system_prompt_chars]

        # The user turn is short and redundant with the template's final line;
        # providers that require a user message get the live query.
        user_prompt = _clean_content(latest_message) or "(empty message)"

        return RenderedPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            knowledge_base_chars=len(kb_text),
            history_lines=len(transcript),
            truncated=truncated,
        )
