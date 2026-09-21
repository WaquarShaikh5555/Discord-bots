"""Logging setup with a single, consistent format across the process."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-26s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: discord.py's HTTP logger is extremely chatty at DEBUG level.
_NOISY_LOGGERS = ("discord.http", "discord.gateway", "asyncio", "aiosqlite")


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once. Safe to call repeatedly."""
    resolved = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)

    # Keep third-party noise one level above the app unless we're fully verbose.
    noisy_level = logging.INFO if resolved > logging.DEBUG else logging.DEBUG
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(noisy_level)

    logging.getLogger("discord").setLevel(max(resolved, logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
