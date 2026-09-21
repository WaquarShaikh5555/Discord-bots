"""Environment-driven configuration.

Everything the bot needs is loaded once at startup into a frozen
:class:`Settings` object, validated eagerly, and injected into services. No
module reads ``os.environ`` directly — that keeps the runtime deterministic and
the unit tests trivial (just build a ``Settings`` with the values you want).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

from dotenv import load_dotenv

from bot.constants import OPENAI_COMPATIBLE_PROVIDERS

VALID_PROVIDERS: Final[frozenset[str]] = frozenset({"groq", "cerebras", "gemini"})
DEFAULT_PROVIDER_CHAIN: Final[tuple[str, ...]] = ("groq", "cerebras", "gemini")

PROVIDER_API_KEY_ENV: Final[dict[str, str]] = {
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class ConfigurationError(RuntimeError):
    """Raised when the environment is not usable. Message is operator-facing."""


def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


#: Substrings that mean "the operator never replaced the example value".
_PLACEHOLDER_MARKERS: Final[tuple[str, ...]] = (
    "paste",
    "your-",
    "your_",
    "changeme",
    "change-me",
    "placeholder",
    "todo",
    "xxx",
    "example",
    "insert-",
    "put-",
)


def _looks_unfilled(value: str) -> bool:
    """True when a secret still holds a template marker instead of a real key.

    Catches ``KEY=`` left with a trailing inline comment (python-dotenv keeps the
    comment as the value), ``KEY=<your-token>``, ``KEY=your-api-key``, etc.
    """
    candidate = value.strip().lower()
    if not candidate:
        return False
    if candidate.startswith("#"):
        return True
    if candidate.startswith("<") and candidate.endswith(">"):
        return True
    # Real credentials (Discord tokens, gsk_/csk-/AIza keys) never contain
    # internal whitespace, so a space means prose was pasted instead of a key.
    if any(char.isspace() for char in candidate):
        return True
    return any(marker in candidate for marker in _PLACEHOLDER_MARKERS)


def _env_secret(name: str) -> str:
    """Read a credential, rejecting unfilled placeholders with a clear error."""
    value = _env_str(name)
    if value and _looks_unfilled(value):
        raise ConfigurationError(
            f"{name} still contains a placeholder value ({value[:40]!r}). "
            f"Replace it with the real credential, or leave it completely empty "
            f"to disable that provider. Note that .env values must not be followed "
            f"by an inline comment — put comments on their own line."
        )
    return value


def _env_int(name: str, default: int, *, minimum: int = 0, maximum: int | None = None) -> int:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:  # pragma: no cover - operator input guard
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}.") from exc
    if value < minimum or (maximum is not None and value > maximum):
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}, got {value}.")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    raw = _env_str(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}.") from exc
    if value < minimum or (maximum is not None and value > maximum):
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}, got {value}.")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean (true/false), got {raw!r}.")


def _env_list(name: str, default: Sequence[str]) -> tuple[str, ...]:
    raw = _env_str(name)
    if not raw:
        return tuple(default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _snowflake_list(name: str) -> tuple[int, ...]:
    ids: list[int] = []
    for raw in _env_list(name, ()):
        if not raw.isdigit():
            raise ConfigurationError(f"{name} must contain numeric Discord snowflakes, got {raw!r}.")
        ids.append(int(raw))
    return tuple(ids)


@dataclass(frozen=True)
class ProviderSettings:
    """Per-provider model, credentials and free-tier ceilings."""

    name: str
    api_key: str
    model: str
    requests_per_minute: float
    requests_per_day: int
    timeout: float

    @property
    def available(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class Settings:
    """Validated runtime configuration for the whole process."""

    discord_token: str
    database_url: str
    log_level: str = "INFO"
    dev_guild_ids: tuple[int, ...] = ()

    # --- provider chain -------------------------------------------------- #
    provider_chain: tuple[str, ...] = DEFAULT_PROVIDER_CHAIN
    groq: ProviderSettings | None = None
    cerebras: ProviderSettings | None = None
    gemini: ProviderSettings | None = None

    # --- inference ------------------------------------------------------- #
    temperature: float = 0.2
    max_tokens: int = 700
    max_retries: int = 3
    backoff_base: float = 0.6
    backoff_cap: float = 12.0
    total_deadline: float = 45.0
    circuit_breaker_threshold: int = 4
    circuit_breaker_cooldown: float = 60.0

    # --- conversation ---------------------------------------------------- #
    history_message_limit: int = 10
    history_char_budget: int = 6000
    knowledge_base_char_limit: int = 40_000

    # --- ticket behaviour ------------------------------------------------ #
    require_configured_category: bool = True
    ticket_name_pattern: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"^(ticket|support|help)[-_]", re.IGNORECASE)
    )
    user_cooldown_seconds: float = 5.0
    max_concurrent_tickets: int = 12
    escalation_cooldown_seconds: float = 900.0
    escalation_preflight_safety: bool = True
    escalation_preflight_requests: bool = False

    # --- database -------------------------------------------------------- #
    config_cache_ttl: float = 60.0
    db_echo: bool = False
    db_pool_size: int = 10

    def provider(self, name: str) -> ProviderSettings | None:
        return getattr(self, name, None)

    def enabled_providers(self) -> list[ProviderSettings]:
        """Providers in chain order that actually have an API key."""
        enabled: list[ProviderSettings] = []
        for name in self.provider_chain:
            provider = self.provider(name)
            if provider is not None and provider.available:
                enabled.append(provider)
        return enabled

    def validate(self) -> "Settings":
        if not self.discord_token:
            raise ConfigurationError(
                "DISCORD_BOT_TOKEN is missing. Create a bot at "
                "https://discord.com/developers/applications and set it in your .env file."
            )
        if not self.database_url:
            raise ConfigurationError("DATABASE_URL is missing (e.g. sqlite+aiosqlite:///data/tickets.db).")
        if not self.provider_chain:
            raise ConfigurationError("AI_PROVIDER_CHAIN cannot be empty.")
        unknown = [name for name in self.provider_chain if name not in VALID_PROVIDERS]
        if unknown:
            raise ConfigurationError(
                f"AI_PROVIDER_CHAIN contains unknown provider(s): {', '.join(unknown)}. "
                f"Valid values: {', '.join(sorted(VALID_PROVIDERS))}."
            )
        if not self.enabled_providers():
            raise ConfigurationError(
                "No AI provider is usable: set at least one of GROQ_API_KEY, CEREBRAS_API_KEY "
                "or GEMINI_API_KEY in your .env file."
            )
        if self.history_message_limit < 1:
            raise ConfigurationError("HISTORY_MESSAGE_LIMIT must be >= 1.")
        return self


def _provider_settings(
    name: str,
    *,
    model_default: str,
    rpm_default: float,
    rpd_default: int,
    timeout: float,
) -> ProviderSettings:
    upper = name.upper()
    return ProviderSettings(
        name=name,
        api_key=_env_secret(PROVIDER_API_KEY_ENV[name]),
        model=_env_str(f"{upper}_MODEL", model_default),
        requests_per_minute=_env_float(f"{upper}_REQUESTS_PER_MINUTE", rpm_default, minimum=0.1),
        requests_per_day=_env_int(f"{upper}_REQUESTS_PER_DAY", rpd_default, minimum=0),
        timeout=timeout,
    )


def _compile_ticket_pattern(raw: str) -> re.Pattern[str]:
    try:
        return re.compile(raw, re.IGNORECASE)
    except re.error as exc:
        raise ConfigurationError(f"TICKET_NAME_PATTERN is not a valid regex: {exc}") from exc


def load_settings(env_file: str | Path | None = None, *, load_dotenv_file: bool = True) -> Settings:
    """Read the environment (optionally seeding it from a ``.env`` file)."""
    if load_dotenv_file:
        load_dotenv(dotenv_path=env_file, override=False)

    request_timeout = _env_float("AI_REQUEST_TIMEOUT", 25.0, minimum=1.0)
    provider_chain = tuple(name.lower() for name in _env_list("AI_PROVIDER_CHAIN", DEFAULT_PROVIDER_CHAIN))

    settings = Settings(
        discord_token=_env_secret("DISCORD_BOT_TOKEN"),
        database_url=_env_str("DATABASE_URL", "sqlite+aiosqlite:///data/tickets.db"),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        dev_guild_ids=_snowflake_list("DEV_GUILD_IDS"),
        provider_chain=provider_chain,
        groq=_provider_settings(
            "groq", model_default="llama-3.3-70b-versatile", rpm_default=15.0, rpd_default=14_400,
            timeout=request_timeout,
        ),
        cerebras=_provider_settings(
            "cerebras", model_default="llama3.3-70b", rpm_default=60.0, rpd_default=14_400,
            timeout=request_timeout,
        ),
        gemini=_provider_settings(
            "gemini", model_default="gemini-2.0-flash", rpm_default=15.0, rpd_default=1_500,
            timeout=request_timeout,
        ),
        temperature=_env_float("AI_TEMPERATURE", 0.2, minimum=0.0, maximum=2.0),
        max_tokens=_env_int("AI_MAX_TOKENS", 700, minimum=64, maximum=8192),
        max_retries=_env_int("AI_MAX_RETRIES", 3, minimum=0, maximum=10),
        backoff_base=_env_float("AI_BACKOFF_BASE", 0.6, minimum=0.05),
        backoff_cap=_env_float("AI_BACKOFF_CAP", 12.0, minimum=0.1),
        total_deadline=_env_float("AI_TOTAL_DEADLINE", 45.0, minimum=2.0),
        circuit_breaker_threshold=_env_int("CIRCUIT_BREAKER_THRESHOLD", 4, minimum=1),
        circuit_breaker_cooldown=_env_float("CIRCUIT_BREAKER_COOLDOWN", 60.0, minimum=1.0),
        history_message_limit=_env_int("HISTORY_MESSAGE_LIMIT", 10, minimum=1, maximum=100),
        history_char_budget=_env_int("HISTORY_CHAR_BUDGET", 6000, minimum=500),
        knowledge_base_char_limit=_env_int("KNOWLEDGE_BASE_CHAR_LIMIT", 40_000, minimum=200),
        require_configured_category=_env_bool("REQUIRE_CONFIGURED_CATEGORY", True),
        ticket_name_pattern=_compile_ticket_pattern(
            _env_str("TICKET_NAME_PATTERN", r"^(ticket|support|help)[-_]")
        ),
        user_cooldown_seconds=_env_float("USER_COOLDOWN_SECONDS", 5.0, minimum=0.0),
        max_concurrent_tickets=_env_int("MAX_CONCURRENT_TICKETS", 12, minimum=1, maximum=500),
        escalation_cooldown_seconds=_env_float("ESCALATION_COOLDOWN_SECONDS", 900.0, minimum=0.0),
        escalation_preflight_safety=_env_bool("ESCALATION_PREFLIGHT_SAFETY", True),
        escalation_preflight_requests=_env_bool("ESCALATION_PREFLIGHT_REQUESTS", False),
        config_cache_ttl=_env_float("CONFIG_CACHE_TTL", 60.0, minimum=0.0),
        db_echo=_env_bool("DB_ECHO", False),
        db_pool_size=_env_int("DB_POOL_SIZE", 10, minimum=1, maximum=100),
    )
    return settings.validate()


def redacted_dump(settings: Settings) -> dict[str, Any]:
    """Settings snapshot safe to log (API keys and token masked)."""
    def mask(value: str) -> str:
        if not value:
            return "<unset>"
        return f"{value[:3]}...{value[-2:]} (len={len(value)})" if len(value) > 8 else "***"

    providers = {
        name: (
            {
                "model": getattr(settings, name).model,
                "api_key": mask(getattr(settings, name).api_key),
                "rpm": getattr(settings, name).requests_per_minute,
                "rpd": getattr(settings, name).requests_per_day,
            }
            if getattr(settings, name)
            else None
        )
        for name in ("groq", "cerebras", "gemini")
    }
    return {
        "discord_token": mask(settings.discord_token),
        "database_url": re.sub(r"://([^:/@]+):([^@]+)@", r"://\1:***@", settings.database_url),
        "provider_chain": list(settings.provider_chain),
        "enabled_providers": [p.name for p in settings.enabled_providers()],
        "providers": providers,
        "model_is_openai_compatible": {
            name: name in OPENAI_COMPATIBLE_PROVIDERS for name in settings.provider_chain
        },
        "history_message_limit": settings.history_message_limit,
        "require_configured_category": settings.require_configured_category,
        "max_concurrent_tickets": settings.max_concurrent_tickets,
    }
