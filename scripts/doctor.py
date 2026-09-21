#!/usr/bin/env python3
"""Setup doctor: validate configuration, database and AI providers before launch.

Run it after editing ``.env`` and before starting the bot::

    python -m scripts.doctor            # offline checks
    python -m scripts.doctor --live     # + one real request per provider

It exits non-zero when something would stop the bot from working, and prints an
actionable message for each failure — no Discord connection required.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aiohttp  # noqa: E402

from bot.config import ConfigurationError, load_settings, redacted_dump  # noqa: E402
from bot.db.repository import TicketRepository  # noqa: E402
from bot.db.session import Database  # noqa: E402
from bot.services.providers.factory import build_providers  # noqa: E402
from bot.utils.logging_setup import setup_logging  # noqa: E402

OK = "\033[92m✓\033[0m"
WARN = "\033[93m!\033[0m"
FAIL = "\033[91m✗\033[0m"


def report(symbol: str, message: str) -> None:
    print(f"  {symbol} {message}")


async def check_database(settings) -> bool:
    print("\nDatabase")
    database = Database(settings.database_url, echo=False)
    try:
        await database.initialize()
    except Exception as exc:
        report(FAIL, f"could not initialise: {exc}")
        return False
    report(OK, f"connected ({database.dialect}) and tables created")

    repo = TicketRepository(database, cache_ttl=0.0)
    try:
        record = await repo.ensure_guild(1, "doctor-probe")
        assert record.guild_id == "1"
        report(OK, "read/write round-trip succeeded")
    except Exception as exc:
        report(FAIL, f"read/write round-trip failed: {exc}")
        await database.close()
        return False

    await database.close()
    return True


async def check_providers(settings, *, live: bool) -> bool:
    print("\nAI providers")
    async with aiohttp.ClientSession() as session:
        providers = build_providers(settings, session)
        if not providers:
            report(FAIL, "no provider has an API key; the bot cannot answer tickets")
            return False

        for provider in providers:
            report(OK, f"{provider.name} configured with model `{provider.model}`")

        missing = [
            name
            for name in settings.provider_chain
            if (settings.provider(name) is None or not settings.provider(name).available)
        ]
        for name in missing:
            report(WARN, f"{name} is in the chain but has no API key (will be skipped)")

        if not live:
            report(WARN, "skipped live inference check (pass --live to send one real request)")
            return True

        print("\nLive inference check (one request per provider)")
        all_ok = True
        for provider in providers:
            try:
                response = await provider.complete(
                    system_prompt="You are a health-check probe. Reply with exactly: OK",
                    user_prompt="ping",
                )
                report(
                    OK,
                    f"{provider.name}/{provider.model} replied in {response.latency_ms}ms: "
                    f"{response.text[:60]!r}",
                )
            except Exception as exc:
                all_ok = False
                report(FAIL, f"{provider.name}/{provider.model} failed: {exc}")
        if not all_ok:
            report(WARN, "a failing primary is survivable only if another provider works")
        return all_ok


def check_intents(settings) -> None:
    print("\nDiscord application")
    if settings.discord_token:
        report(OK, "DISCORD_BOT_TOKEN is set")
    else:
        report(FAIL, "DISCORD_BOT_TOKEN is missing")
    report(
        WARN,
        "the bot needs the MESSAGE CONTENT intent — enable it at "
        "discord.com/developers/applications → your app → Bot → Privileged Gateway Intents",
    )
    report(
        WARN,
        "invite URL must include the scopes `bot` and `applications.commands`",
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the ticket bot's configuration.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="send one real request to each configured provider (uses free-tier quota)",
    )
    parser.add_argument("--env", default=None, help="path to a .env file (default: ./.env)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print full config dump")
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else "WARNING")

    print("Discord AI Ticket Responder — setup doctor")
    print("=" * 52)

    try:
        settings = load_settings(args.env, load_dotenv_file=True)
    except ConfigurationError as exc:
        print(f"\n  {FAIL} Configuration error:\n     {exc}\n")
        print("  Copy .env.example to .env and fill in the required values.")
        return 2

    if args.verbose:
        print("\nEffective configuration")
        for key, value in redacted_dump(settings).items():
            print(f"  {key}: {value}")

    results = [
        await check_database(settings),
        await check_providers(settings, live=args.live),
    ]
    check_intents(settings)

    print("\n" + "=" * 52)
    if all(results):
        print(f"  {OK} Ready. Start the bot with:  python -m bot")
        return 0
    print(f"  {FAIL} Fix the items above before starting the bot.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
