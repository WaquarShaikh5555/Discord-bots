"""Bot wiring: the process must boot, load cogs and expose valid slash commands.

``tree.to_list()`` renders exactly the payload Discord receives on sync, so a bad
option type or an over-long description fails here instead of in production.
"""

from __future__ import annotations

import aiohttp
import pytest

from bot.config import redacted_dump
from bot.main import COG_MODULES, TicketBot, _format_config_dump
from bot.services.ai_service import AIService

MANAGE_GUILD_BIT = 32  # discord.Permissions(manage_guild=True).value
GUILD_CONTEXT = 0      # discord.InteractionContext.guild

#: Discord application-command option types we assert against.
OPTION_STRING = 3
OPTION_BOOLEAN = 5
OPTION_CHANNEL = 7
OPTION_ROLE = 8
OPTION_ATTACHMENT = 11
CHANNEL_TYPE_CATEGORY = 4

EXPECTED_COMMANDS = {
    "setup-kb",
    "set-staff-role",
    "set-ticket-category",
    "view-kb",
    "ticket-status",
    "resolve-ticket",
    "ai-status",
}


async def command_payload(bot: TicketBot) -> list[dict]:
    """The exact payload Discord receives on ``tree.sync()``."""
    return [command.to_dict(bot.tree) for command in bot.tree.get_commands()]


@pytest.fixture
async def booted_bot(settings):
    """A TicketBot with database, HTTP session, AI service and cogs initialised.

    Nothing here touches the network: no login, no command sync, no API calls.
    """
    bot = TicketBot(settings)
    await bot.database.initialize()
    bot.http_session = aiohttp.ClientSession()
    bot.ai_service = AIService([], settings)
    for module in COG_MODULES:
        await bot.load_extension(module)
    try:
        yield bot
    finally:
        if not bot.http_session.closed:
            await bot.http_session.close()
        await bot.database.close()


async def test_all_extensions_load(booted_bot: TicketBot):
    assert booted_bot.get_cog("AdminCommands") is not None
    assert booted_bot.get_cog("TicketListener") is not None


async def test_intents_include_message_content(booted_bot: TicketBot):
    """Without this privileged intent the listener can never read a ticket."""
    assert booted_bot.intents.message_content is True
    assert booted_bot.intents.guilds is True
    assert booted_bot.intents.guild_messages is True
    assert booted_bot.intents.members is False, "members intent is not needed; avoid the grant"


async def test_slash_commands_are_registered_with_valid_schemas(booted_bot: TicketBot):
    payload = await command_payload(booted_bot)
    names = {entry["name"] for entry in payload}

    assert EXPECTED_COMMANDS <= names, f"missing commands: {EXPECTED_COMMANDS - names}"
    for entry in payload:
        assert entry["type"] == 1, f"{entry['name']} must be a CHAT_INPUT slash command"
        assert 0 < len(entry["description"]) <= 100, entry["name"]


async def test_every_command_is_guild_only_and_admin_gated(booted_bot: TicketBot):
    """Admin configuration must not be reachable from DMs or by ordinary members."""
    payload = await command_payload(booted_bot)
    assert payload, "no commands were registered"
    for entry in payload:
        assert entry["dm_permission"] is False, f"{entry['name']} must not be usable in DMs"
        assert entry["contexts"] == [GUILD_CONTEXT], f"{entry['name']} must be guild-only"
        assert entry["default_member_permissions"] == MANAGE_GUILD_BIT, (
            f"{entry['name']} must require Manage Guild"
        )


@pytest.mark.parametrize(
    "command,expected_options",
    [
        ("setup-kb", {"text", "attachment", "mode"}),
        ("set-staff-role", {"role"}),
        ("set-ticket-category", {"category"}),
        ("view-kb", {"full"}),
        ("ticket-status", set()),
        ("resolve-ticket", {"action"}),
        ("ai-status", set()),
    ],
)
async def test_command_options(booted_bot: TicketBot, command: str, expected_options: set[str]):
    payload = await command_payload(booted_bot)
    entry = next(item for item in payload if item["name"] == command)
    options = {option["name"] for option in entry.get("options", [])}
    assert options == expected_options

    for option in entry.get("options", []):
        assert 0 < len(option["description"]) <= 100, f"{command}.{option['name']}"


async def test_setup_kb_exposes_the_three_modes(booted_bot: TicketBot):
    payload = await command_payload(booted_bot)
    entry = next(item for item in payload if item["name"] == "setup-kb")
    by_name = {option["name"]: option for option in entry["options"]}

    assert by_name["text"]["type"] == OPTION_STRING
    assert by_name["attachment"]["type"] == OPTION_ATTACHMENT
    assert by_name["mode"]["type"] == OPTION_STRING
    assert {choice["value"] for choice in by_name["mode"]["choices"]} == {
        "replace",
        "append",
        "prepend",
    }
    # Every option is optional so `/setup-kb` can be invoked with any combination.
    assert all(option.get("required") is not True for option in entry["options"])


async def test_set_staff_role_takes_a_role_object(booted_bot: TicketBot):
    payload = await command_payload(booted_bot)
    entry = next(item for item in payload if item["name"] == "set-staff-role")
    role_option = next(option for option in entry["options"] if option["name"] == "role")
    assert role_option["type"] == OPTION_ROLE, "must be a ROLE picker so admins cannot typo an id"
    assert role_option["required"] is True


async def test_set_ticket_category_accepts_only_categories(booted_bot: TicketBot):
    payload = await command_payload(booted_bot)
    entry = next(item for item in payload if item["name"] == "set-ticket-category")
    option = next(item for item in entry["options"] if item["name"] == "category")

    assert option["type"] == OPTION_CHANNEL
    assert option["channel_types"] == [CHANNEL_TYPE_CATEGORY], (
        "the picker must offer only category channels"
    )
    assert option["required"] is False, "omitting it clears the restriction"


async def test_view_kb_and_resolve_ticket_option_types(booted_bot: TicketBot):
    payload = await command_payload(booted_bot)
    view_kb = next(item for item in payload if item["name"] == "view-kb")
    assert view_kb["options"][0]["type"] == OPTION_BOOLEAN

    resolve = next(item for item in payload if item["name"] == "resolve-ticket")
    action = next(option for option in resolve["options"] if option["name"] == "action")
    assert {choice["value"] for choice in action["choices"]} == {"resolve", "reopen"}


async def test_ticket_status_and_ai_status_take_no_options(booted_bot: TicketBot):
    payload = await command_payload(booted_bot)
    for name in ("ticket-status", "ai-status"):
        entry = next(item for item in payload if item["name"] == name)
        assert not entry.get("options"), f"{name} must not require any arguments"


async def test_listener_is_wired_to_the_real_services(booted_bot: TicketBot):
    cog = booted_bot.get_cog("TicketListener")
    assert cog.repo is booted_bot.repository
    assert cog.ai is booted_bot.ai_service
    assert cog.settings is booted_bot.settings
    assert cog.resolver.bot_user_id == 0, "patched in cog_load, which needs a logged-in user"
    assert cog.resolver.require_configured_category is booted_bot.settings.require_configured_category


async def test_cog_load_patches_the_bot_user_id(booted_bot: TicketBot):
    cog = booted_bot.get_cog("TicketListener")

    class _User:
        id = 4242

    # commands.Bot.user is read-only; the gateway populates it through the
    # connection state, which is what we simulate here.
    booted_bot._connection.user = _User()
    await cog.cog_load()
    assert cog.resolver.bot_user_id == 4242
    await cog.cog_unload()


async def test_ai_service_property_raises_before_initialisation(settings):
    bot = TicketBot(settings)
    with pytest.raises(RuntimeError):
        _ = bot.ai


async def test_listener_stats_default_shape(booted_bot: TicketBot):
    stats = booted_bot.listener_stats()
    assert set(stats) == {
        "messages_processed",
        "answered",
        "escalated",
        "active_channels",
        "skipped",
    }


# --------------------------------------------------------------------------- #
# config redaction — secrets must never reach a log file
# --------------------------------------------------------------------------- #
def test_redacted_dump_masks_secrets(settings):
    dump = redacted_dump(settings)
    assert "test-token" not in str(dump)
    assert "test-groq-key" not in str(dump)
    assert dump["enabled_providers"] == ["groq", "cerebras", "gemini"]
    assert dump["provider_chain"] == ["groq", "cerebras", "gemini"]


def test_redacted_dump_masks_database_password():
    from bot.config import Settings

    settings = Settings(
        discord_token="tok",
        database_url="postgresql+asyncpg://user:supersecret@db.example.com:5432/tickets",
    )
    dump = redacted_dump(settings)
    assert "supersecret" not in dump["database_url"]
    assert "db.example.com" in dump["database_url"]


def test_config_dump_is_loggable(settings):
    lines = _format_config_dump(redacted_dump(settings))
    assert any("provider_chain" in line for line in lines)
    assert all(isinstance(line, str) for line in lines)
