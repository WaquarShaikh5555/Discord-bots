"""Configuration loading and the unfilled-placeholder guard."""

from __future__ import annotations

import pytest

from bot.config import (
    ConfigurationError,
    _looks_unfilled,
    load_settings,
    redacted_dump,
)


@pytest.fixture
def env_clean(monkeypatch):
    """A minimal, valid environment with no stray variables."""
    for name in (
        "DISCORD_BOT_TOKEN",
        "DATABASE_URL",
        "GROQ_API_KEY",
        "CEREBRAS_API_KEY",
        "GEMINI_API_KEY",
        "AI_PROVIDER_CHAIN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "MTIz.real.token")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_realtoken")
    return monkeypatch


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_valid_environment_loads(env_clean):
    settings = load_settings(load_dotenv_file=False)
    assert settings.discord_token == "MTIz.real.token"
    assert settings.groq.available is True
    assert settings.cerebras.available is False
    assert [p.name for p in settings.enabled_providers()] == ["groq"]


def test_missing_token_is_rejected(env_clean):
    env_clean.delenv("DISCORD_BOT_TOKEN")
    with pytest.raises(ConfigurationError, match="DISCORD_BOT_TOKEN is missing"):
        load_settings(load_dotenv_file=False)


def test_missing_all_provider_keys_is_rejected(env_clean):
    env_clean.delenv("GROQ_API_KEY")
    with pytest.raises(ConfigurationError, match="No AI provider is usable"):
        load_settings(load_dotenv_file=False)


def test_unknown_provider_in_chain_is_rejected(env_clean):
    env_clean.setenv("AI_PROVIDER_CHAIN", "groq,openai")
    with pytest.raises(ConfigurationError, match="unknown provider"):
        load_settings(load_dotenv_file=False)


def test_blank_chain_falls_back_to_the_default(env_clean):
    """An empty AI_PROVIDER_CHAIN means 'use the defaults', not 'fail to boot'."""
    env_clean.setenv("AI_PROVIDER_CHAIN", "  ")
    assert load_settings(load_dotenv_file=False).provider_chain == ("groq", "cerebras", "gemini")


def test_non_numeric_setting_is_rejected(env_clean):
    env_clean.setenv("AI_MAX_RETRIES", "lots")
    with pytest.raises(ConfigurationError, match="must be an integer"):
        load_settings(load_dotenv_file=False)


def test_out_of_range_setting_is_rejected(env_clean):
    env_clean.setenv("AI_TEMPERATURE", "7")
    with pytest.raises(ConfigurationError, match="must be between"):
        load_settings(load_dotenv_file=False)


def test_invalid_boolean_is_rejected(env_clean):
    env_clean.setenv("REQUIRE_CONFIGURED_CATEGORY", "maybe")
    with pytest.raises(ConfigurationError, match="must be a boolean"):
        load_settings(load_dotenv_file=False)


def test_invalid_regex_pattern_is_rejected(env_clean):
    env_clean.setenv("TICKET_NAME_PATTERN", "^([unclosed")
    with pytest.raises(ConfigurationError, match="not a valid regex"):
        load_settings(load_dotenv_file=False)


def test_non_numeric_dev_guild_id_is_rejected(env_clean):
    env_clean.setenv("DEV_GUILD_IDS", "my-server")
    with pytest.raises(ConfigurationError, match="numeric Discord snowflakes"):
        load_settings(load_dotenv_file=False)


def test_dev_guild_ids_are_parsed(env_clean):
    env_clean.setenv("DEV_GUILD_IDS", "111, 222 ,333")
    assert load_settings(load_dotenv_file=False).dev_guild_ids == (111, 222, 333)


# --------------------------------------------------------------------------- #
# the placeholder guard — an unfilled template must not become a "real" secret
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value",
    [
        "# <<< PASTE HERE",           # dotenv keeps an inline comment as the value
        "<your-token-here>",
        "your-api-key",
        "YOUR_API_KEY",
        "changeme",
        "change-me",
        "placeholder",
        "TODO",
        "xxxxxxxx",
        "paste your key here",
        "your token here",
        "example-key",
        "gsk_abc def",          # a real-looking prefix with prose after it
    ],
)
def test_unfilled_placeholders_are_detected(value):
    assert _looks_unfilled(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "MTIzNDU2.real.discord.token",
        "gsk_A1b2C3d4E5f6",
        "csk-realkey",
        "AIzaSyRealKey",
    ],
)
def test_real_credentials_are_not_flagged(value):
    assert _looks_unfilled(value) is False


def test_placeholder_token_is_rejected_with_guidance(env_clean):
    """The classic mistake: `DISCORD_BOT_TOKEN=  # paste here`."""
    env_clean.setenv("DISCORD_BOT_TOKEN", "# <<< PASTE HERE")
    with pytest.raises(ConfigurationError) as excinfo:
        load_settings(load_dotenv_file=False)
    message = str(excinfo.value)
    assert "DISCORD_BOT_TOKEN" in message
    assert "placeholder" in message
    assert "inline comment" in message, "must explain the root cause"


def test_placeholder_api_key_is_rejected_by_name(env_clean):
    env_clean.setenv("GROQ_API_KEY", "your-groq-key")
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        load_settings(load_dotenv_file=False)


def test_empty_optional_key_is_allowed_but_placeholder_is_not(env_clean):
    """Empty means 'skip this provider'; a marker means 'you forgot'."""
    env_clean.setenv("CEREBRAS_API_KEY", "")
    assert load_settings(load_dotenv_file=False).cerebras.available is False

    env_clean.setenv("CEREBRAS_API_KEY", "<paste here>")
    with pytest.raises(ConfigurationError, match="CEREBRAS_API_KEY"):
        load_settings(load_dotenv_file=False)


# --------------------------------------------------------------------------- #
# defaults + redaction
# --------------------------------------------------------------------------- #
def test_defaults_match_the_specification(env_clean):
    settings = load_settings(load_dotenv_file=False)
    assert settings.history_message_limit == 10, "spec: last 10 messages of chat memory"
    assert settings.require_configured_category is True
    assert settings.escalation_preflight_safety is True
    assert settings.escalation_preflight_requests is False
    assert settings.groq.model == "llama-3.3-70b-versatile"
    assert settings.gemini.model == "gemini-2.0-flash"
    assert settings.provider_chain == ("groq", "cerebras", "gemini")


def test_validate_is_idempotent(env_clean):
    settings = load_settings(load_dotenv_file=False)
    assert settings.validate() is settings


def test_redaction_never_leaks_credentials(env_clean):
    dump = str(redacted_dump(load_settings(load_dotenv_file=False)))
    assert "MTIz.real.token" not in dump
    assert "gsk_realtoken" not in dump


def test_settings_are_frozen(env_clean):
    settings = load_settings(load_dotenv_file=False)
    with pytest.raises(Exception):
        settings.discord_token = "mutated"  # type: ignore[misc]


def test_provider_lookup_helper(env_clean):
    settings = load_settings(load_dotenv_file=False)
    assert settings.provider("groq") is settings.groq
    assert settings.provider("nonexistent") is None
