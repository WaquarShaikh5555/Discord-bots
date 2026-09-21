"""Application entrypoint: build the bot, wire the services, load the cogs.

Run with::

    python -m bot

Startup order matters and is enforced here:

1. Load and validate configuration (fail fast with an actionable message).
2. Initialise the database (creates tables if missing).
3. Open a shared ``aiohttp`` session for all AI providers.
4. Build the AI service (provider chain + rate limits + circuit breakers).
5. Load cogs: admin commands and the ticket listener.
6. Sync slash commands (per-guild for dev servers, globally otherwise).
"""

from __future__ import annotations

import asyncio
import signal
import sys
from typing import Any

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from bot import __version__
from bot.config import Settings, load_settings, redacted_dump
from bot.db.repository import TicketRepository
from bot.db.session import Database
from bot.services.ai_service import AIService, build_ai_service
from bot.utils.logging_setup import get_logger, setup_logging

log = get_logger(__name__)

COG_MODULES: tuple[str, ...] = ("bot.cogs.commands", "bot.cogs.ticket_listener")


class TicketCommandTree(app_commands.CommandTree):
    """Command tree with a single, user-friendly error path."""

    async def on_error(
        self,
        interaction: discord.Interaction,
        error: app_commands.AppCommandError,
    ) -> None:
        log.warning(
            "Unhandled app-command error (%s, guild=%s): %s",
            interaction.command.name if interaction.command else "?",
            interaction.guild_id,
            error,
        )
        text = "Something went wrong while running that command. Details are in the bot logs."
        try:
            if interaction.response.is_done():
                await interaction.followup.send(text, ephemeral=True)
            else:
                await interaction.response.send_message(text, ephemeral=True)
        except discord.HTTPException:  # pragma: no cover
            log.exception("Could not report an app-command error to the user")


class TicketBot(commands.Bot):
    """The bot process: owns config, database, HTTP session and AI service."""

    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.default()
        # Privileged intent — required to read ticket text. Must also be enabled
        # in the Discord developer portal, or the gateway rejects the connection.
        intents.message_content = True
        intents.members = False  # not needed; avoids another privileged grant

        super().__init__(
            command_prefix=commands.when_mentioned,  # prefix commands are unused
            intents=intents,
            tree_cls=TicketCommandTree,
            help_command=None,
            case_insensitive=True,
        )

        self.settings = settings
        self.database = Database(
            settings.database_url, echo=settings.db_echo, pool_size=settings.db_pool_size
        )
        self.repository = TicketRepository(self.database, cache_ttl=settings.config_cache_ttl)
        self.ai_service: AIService | None = None
        self.http_session: aiohttp.ClientSession | None = None
        self._cogs_ready = asyncio.Event()
        self._startup_error: str | None = None

    # ------------------------------------------------------------------ #
    # service accessors (typed non-Optional for cogs)
    # ------------------------------------------------------------------ #
    @property
    def ai(self) -> AIService:
        if self.ai_service is None:  # pragma: no cover - guarded by setup_hook
            raise RuntimeError("AI service is not initialised yet")
        return self.ai_service

    @property
    def session(self) -> aiohttp.ClientSession:
        if self.http_session is None:  # pragma: no cover
            raise RuntimeError("HTTP session is not initialised yet")
        return self.http_session

    def listener_stats(self) -> dict[str, Any]:
        cog = self.get_cog("TicketListener")
        return cog.stats if cog is not None else {  # type: ignore[attr-defined]
            "messages_processed": 0,
            "answered": 0,
            "escalated": 0,
            "active_channels": 0,
            "skipped": {},
        }

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def setup_hook(self) -> None:
        try:
            await self._bootstrap()
        except Exception as exc:
            self._startup_error = f"{type(exc).__name__}: {exc}"
            log.critical("Startup failed: %s", self._startup_error, exc_info=True)
            raise

    async def _bootstrap(self) -> None:
        log.info("Starting %s v%s", "AI Ticket Responder", __version__)
        for line in _format_config_dump(redacted_dump(self.settings)):
            log.info("config | %s", line)

        await self.database.initialize()

        self.http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_read=None),
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=30, ttl_dns_cache=300),
            headers={"Accept": "application/json"},
        )

        self.ai_service = build_ai_service(
            self.settings, self.http_session, usage_sink=self._record_usage
        )
        log.info("AI provider chain: %s", self.ai_service.describe_chain())

        for module in COG_MODULES:
            await self.load_extension(module)
            log.info("Loaded extension %s", module)

        self._cogs_ready.set()

        # Global sync can take up to an hour to reach every client; dev guilds
        # are synced per-guild in on_ready for instant iteration.
        #
        # A sync failure is recoverable (it is retried on the next boot, and the
        # bot answers tickets regardless of whether command metadata reached
        # Discord), so it must never take the whole process down. Network errors
        # surface as bare aiohttp exceptions rather than discord.HTTPException,
        # hence the broad catch.
        try:
            synced = await self.tree.sync()
            log.info("Synced %d global slash command(s).", len(synced))
        except discord.Forbidden:
            log.error(
                "Global command sync was forbidden. The bot token cannot register "
                "application commands — re-invite it with the applications.commands scope."
            )
        except Exception as exc:
            log.error(
                "Global command sync failed (%s: %s). The bot will keep running and "
                "answer tickets; slash commands may be missing until the next restart.",
                type(exc).__name__,
                exc,
            )

    async def _record_usage(
        self,
        provider: str,
        model: str,
        success: bool,
        latency_ms: int,
        completion_chars: int,
        prompt_chars: int,
    ) -> None:
        """Persist provider telemetry (best effort — never blocks a reply)."""
        try:
            await self.repository.bump_usage(
                provider=provider,
                model=model,
                success=success,
                latency_ms=latency_ms,
                completion_chars=completion_chars,
                prompt_chars=prompt_chars,
            )
        except Exception:  # pragma: no cover - telemetry is non-critical
            log.debug("Could not persist usage telemetry", exc_info=True)

    async def on_ready(self) -> None:
        guild_count = len(self.guilds)
        log.info(
            "Logged in as %s (id=%s) — serving %d guild(s), discord.py %s",
            self.user, self.user.id if self.user else "?", guild_count, discord.__version__,
        )
        for guild_id in self.settings.dev_guild_ids:
            guild = self.get_guild(guild_id)
            if guild is None:
                log.warning("DEV_GUILD_IDS contains %s but the bot is not in that server.", guild_id)
                continue
            await self.sync_commands_for(guild)

        try:
            await self.change_presence(
                activity=discord.Activity(
                    type=discord.ActivityType.listening, name="tickets | /setup-kb"
                ),
                status=discord.Status.online,
            )
        except discord.HTTPException:  # pragma: no cover - cosmetic
            pass

    async def sync_commands_for(self, guild: discord.Guild | None) -> None:
        """Instantly sync commands for one guild (dev servers and new joins)."""
        if guild is None:
            return
        try:
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            log.info("Synced %d command(s) to guild %s (%s).", len(synced), guild.id, guild.name)
        except discord.Forbidden:
            log.warning("No permission to sync commands in guild %s.", guild.id)
        except Exception as exc:
            log.warning(
                "Could not sync commands to guild %s (%s: %s)", guild.id, type(exc).__name__, exc
            )

    async def close(self) -> None:
        """Release every resource we own, in reverse order of creation."""
        log.info("Shutting down…")
        if self.http_session is not None and not self.http_session.closed:
            await self.http_session.close()
            # aiohttp needs a moment to close SSL transports cleanly.
            await asyncio.sleep(0.15)
        try:
            await self.database.close()
        except Exception:  # pragma: no cover
            log.exception("Error while closing the database")
        await super().close()


def _format_config_dump(dump: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, value in dump.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {sub_value}")
        else:
            lines.append(f"{key}: {value}")
    return lines


async def run_bot(settings: Settings) -> None:
    """Start the bot and run until cancelled or disconnected."""
    bot = TicketBot(settings)

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _request_stop(signame: str) -> None:
        log.info("Received %s — stopping.", signame)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop, sig.name)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - Windows
            pass

    async with bot:
        starter = asyncio.create_task(bot.start(settings.discord_token), name="bot-start")
        stopper = asyncio.create_task(stop.wait(), name="signal-stop")
        done, pending = await asyncio.wait(
            {starter, stopper}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc is not None:
                raise exc


def cli() -> int:
    """Console entrypoint (``python -m bot`` / ``ticket-bot``)."""
    try:
        settings = load_settings()
    except Exception as exc:
        setup_logging("INFO")
        log.critical("Configuration error: %s", exc)
        print(f"\nConfiguration error:\n  {exc}\n\nSee .env.example for the full list of settings.",
              file=sys.stderr)
        return 2

    setup_logging(settings.log_level)
    try:
        asyncio.run(run_bot(settings))
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        return 0
    except discord.LoginFailure:
        log.critical(
            "Discord rejected the bot token. Check DISCORD_BOT_TOKEN in your .env file."
        )
        return 3
    except discord.PrivilegedIntentsRequired:
        log.critical(
            "Message Content Intent is required but not enabled. Turn it on at "
            "https://discord.com/developers/applications → Bot → Privileged Gateway Intents."
        )
        return 4
    except Exception:
        log.critical("Fatal error", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
