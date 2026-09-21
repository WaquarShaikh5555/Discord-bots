"""Test doubles: a fake aiohttp session and fake Discord objects.

These let the whole bot pipeline (resolver -> prompt -> provider -> escalation
-> delivery) run in unit tests with no network and no gateway connection.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# --------------------------------------------------------------------------- #
# Shared test identifiers
# --------------------------------------------------------------------------- #
GUILD_ID = 100
OTHER_GUILD_ID = 200
CATEGORY_ID = 555
OTHER_CATEGORY_ID = 888
STAFF_ROLE_ID = 999
TICKET_CHANNEL_ID = 777
BOT_USER_ID = 4242
MEMBER_ID = 1001
STAFF_MEMBER_ID = 2002


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
@dataclass
class FakeResponse:
    status: int = 200
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0
    raise_exc: Exception | None = None

    async def text(self) -> str:
        if self.delay:
            await asyncio.sleep(self.delay)
        if isinstance(self.body, str):
            return self.body
        return json.dumps(self.body if self.body is not None else {})

    async def json(self) -> Any:
        return json.loads(await self.text())

    async def __aenter__(self) -> "FakeResponse":
        if self.raise_exc is not None:
            raise self.raise_exc
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None


class FakeSession:
    """Records requests and replays a scripted queue of responses."""

    def __init__(self, responses: Sequence[FakeResponse] | Callable[..., FakeResponse] = ()) -> None:
        self._responses = list(responses)
        self._script = responses if callable(responses) else None
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, *, json: Any = None, headers: dict | None = None,
             timeout: Any = None, **kwargs: Any) -> FakeResponse:
        payload = {"url": url, "json": json, "headers": headers or {}, "timeout": timeout}
        self.calls.append(payload)
        if self._script is not None:
            return self._script(len(self.calls), payload)
        if not self._responses:
            return FakeResponse(status=500, body={"error": {"message": "no scripted response"}})
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def closed(self) -> bool:
        return False


def openai_ok(text: str, *, model: str = "test-model", finish_reason: str = "stop") -> FakeResponse:
    return FakeResponse(
        status=200,
        body={
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": finish_reason}],
        },
    )


def gemini_ok(text: str, *, finish_reason: str = "STOP") -> FakeResponse:
    return FakeResponse(
        status=200,
        body={"candidates": [{"content": {"parts": [{"text": text}]},
                              "finishReason": finish_reason}]},
    )


def rate_limited(retry_after: float = 1.0) -> FakeResponse:
    return FakeResponse(status=429, body={"error": {"message": "slow down"}},
                        headers={"Retry-After": str(retry_after)})


# --------------------------------------------------------------------------- #
# Discord-shaped stubs
# --------------------------------------------------------------------------- #
@dataclass
class FakeRole:
    id: int
    name: str = "role"


@dataclass
class FakeUser:
    id: int
    name: str = "member"
    bot: bool = False
    roles: tuple[FakeRole, ...] = ()

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def mention(self) -> str:
        return f"<@{self.id}>"


@dataclass
class FakeAttachment:
    filename: str = "screenshot.png"
    size: int = 2048


@dataclass
class FakeGuild:
    id: int
    name: str = "Test Server"


@dataclass
class FakeCategory:
    id: int
    name: str = "Tickets"


@dataclass
class FakeChannel:
    id: int = 777
    name: str = "ticket-0001"
    category_id: int | None = None
    category: FakeCategory | None = None
    parent: Any = None
    type: Any = 0
    sent: list[dict[str, Any]] = field(default_factory=list)
    history_messages: list[Any] = field(default_factory=list)
    send_error: Exception | None = None

    async def send(self, content: str | None = None, **kwargs: Any) -> dict[str, Any]:
        if self.send_error is not None:
            raise self.send_error
        record = {"content": content, **kwargs}
        self.sent.append(record)
        return record

    @property
    def sent_text(self) -> list[str]:
        return [str(item.get("content") or "") for item in self.sent]

    @property
    def last_sent(self) -> str:
        return self.sent_text[-1] if self.sent else ""

    def history(self, limit: int | None = None, **kwargs: Any) -> "_AsyncHistoryIterator":
        """Newest-first iterator, matching discord.py's semantics."""
        messages = list(reversed(self.history_messages))
        if limit is not None:
            messages = messages[:limit]
        return _AsyncHistoryIterator(messages)

    def typing(self) -> "_TypingContext":
        return _TypingContext()


class _AsyncHistoryIterator:
    def __init__(self, messages: Sequence[Any]) -> None:
        self._messages = list(messages)

    def __aiter__(self) -> "_AsyncHistoryIterator":
        return self

    async def __anext__(self) -> Any:
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def flatten(self) -> list[Any]:
        return list(self._messages)


class _TypingContext:
    async def __aenter__(self) -> "_TypingContext":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


@dataclass
class FakeThread(FakeChannel):
    type: Any = 12  # discord.ChannelType.public_thread


@dataclass
class FakeBotUser:
    id: int = 4242
    name: str = "TicketBot"
    bot: bool = True

    def __str__(self) -> str:
        return self.name


class FakeBot:
    """The slice of TicketBot the cogs actually touch."""

    def __init__(self, settings: Any, repository: Any, ai_service: Any,
                 user: FakeBotUser | None = None) -> None:
        self.settings = settings
        self.repository = repository
        self.ai_service = ai_service
        self.user = user or FakeBotUser()
        self.synced_guilds: list[int] = []
        self._cogs: dict[str, Any] = {}

    @property
    def ai(self) -> Any:
        return self.ai_service

    def get_cog(self, name: str) -> Any:
        return self._cogs.get(name)

    def register_cog(self, cog: Any) -> None:
        self._cogs[type(cog).__name__] = cog

    async def sync_commands_for(self, guild: Any) -> None:
        if guild is not None:
            self.synced_guilds.append(getattr(guild, "id", 0))

    def listener_stats(self) -> dict[str, Any]:
        cog = self.get_cog("TicketListener")
        return cog.stats if cog is not None else {}


@dataclass
class FakeMessage:
    id: int = 1
    content: str = ""
    author: FakeUser = field(default_factory=lambda: FakeUser(id=1001))
    channel: FakeChannel = field(default_factory=lambda: FakeChannel(id=777))
    guild: FakeGuild | None = field(default_factory=lambda: FakeGuild(id=100))
    attachments: tuple[FakeAttachment, ...] = ()
    embeds: tuple[Any, ...] = ()
    webhook_id: int | None = None
    type: Any = 0
    interaction_metadata: Any = None
    created_at: Any = None


@dataclass
class FakeInteractionResponse:
    deferred: bool = False
    replied: bool = False
    messages: list[dict[str, Any]] = field(default_factory=list)

    async def defer(self, *, ephemeral: bool = False, **kwargs: Any) -> None:
        self.deferred = True
        self.messages.append({"type": "defer", "ephemeral": ephemeral})

    async def send_message(self, content: str | None = None, *, ephemeral: bool = False,
                           **kwargs: Any) -> None:
        self.replied = True
        self.messages.append({"type": "message", "content": content, "ephemeral": ephemeral, **kwargs})


@dataclass
class FakeFollowup:
    messages: list[dict[str, Any]] = field(default_factory=list)

    async def send(self, content: str | None = None, **kwargs: Any) -> None:
        self.messages.append({"content": content, **kwargs})


@dataclass
class FakeInteraction:
    guild: FakeGuild | None = field(default_factory=lambda: FakeGuild(id=100))
    guild_id: int | None = 100
    user: FakeUser = field(default_factory=lambda: FakeUser(id=7, name="admin"))
    response: FakeInteractionResponse = field(default_factory=FakeInteractionResponse)
    followup: FakeFollowup = field(default_factory=FakeFollowup)
    command_name: str = "setup-kb"
    client: Any = None

    @property
    def channel(self) -> FakeChannel:
        return FakeChannel(id=1, name="staff-room", category_id=None)
