"""Text helpers: Discord-safe sanitising, truncation and message chunking.

Model output is *untrusted input*: it may contain role/user mentions injected
through a ticket, literal ``@everyone``, or replies longer than Discord's 2000
character limit. Everything the bot sends goes through these helpers first.
"""

from __future__ import annotations

import re
from typing import Iterable

from bot.constants import (
    BROADCAST_MENTION_RE,
    DISCORD_MESSAGE_LIMIT,
    MENTION_TOKEN_RE,
    ZERO_WIDTH_SPACE,
)

_FENCE = "```"
_ECHO_PREFIX_RE = re.compile(
    r"^\s*(?:ai\s*(?:agent)?\s*(?:response|reply)?|assistant|bot)\s*[:\-]\s*",
    re.IGNORECASE,
)
_MANY_NEWLINES_RE = re.compile(r"\n{4,}")


def strip_code_fence(text: str) -> str:
    """Remove a single wrapping code fence, which models sometimes add."""
    stripped = text.strip()
    if not stripped.startswith(_FENCE) or not stripped.endswith(_FENCE):
        return text
    inner = stripped[len(_FENCE) : -len(_FENCE)]
    # Drop an optional language hint on the opening fence line.
    if "\n" in inner:
        first, rest = inner.split("\n", 1)
        if len(first.strip()) <= 20 and " " not in first.strip():
            inner = rest
    return inner.strip()


def strip_echoed_prefix(text: str) -> str:
    """Drop a leading ``AI Agent Response:`` the model may repeat verbatim."""
    return _ECHO_PREFIX_RE.sub("", text, count=1)


def sanitize_ai_text(text: str) -> str:
    """Neutralise mentions and tidy whitespace in model output.

    * Discord mention tokens (``<@123>``, ``<@&123>``, ``<#123>``) are removed —
      the only mention the bot ever sends is the staff role, appended explicitly
      by :mod:`bot.services.escalation` from the database value.
    * Literal ``@everyone`` / ``@here`` get a zero-width space so they cannot ping.
    """
    cleaned = MENTION_TOKEN_RE.sub("", text)
    cleaned = BROADCAST_MENTION_RE.sub(lambda m: f"@{ZERO_WIDTH_SPACE}{m.group(1)}", cleaned)
    cleaned = _MANY_NEWLINES_RE.sub("\n\n\n", cleaned)
    return cleaned.strip()


def truncate(text: str, limit: int = DISCORD_MESSAGE_LIMIT, *, suffix: str = "…") -> str:
    """Clip ``text`` to ``limit`` characters, preferring a word boundary."""
    if len(text) <= limit:
        return text
    if limit <= len(suffix):
        return text[:limit]
    window = text[: limit - len(suffix)]
    cut = max(window.rfind(" "), window.rfind("\n"))
    if cut < int(limit * 0.6):  # avoid a stubby cut when there is no space nearby
        cut = len(window)
    return f"{window[:cut].rstrip()}{suffix}"


#: Reserve headroom so a chunk joined with separators still fits the limit.
_BLOCK_LIMIT = DISCORD_MESSAGE_LIMIT - 64


def _slice_long_line(line: str) -> tuple[str, str]:
    """Split an over-long line, preferring a word boundary over a mid-word cut."""
    head = line[:_BLOCK_LIMIT]
    cut = head.rfind(" ")
    # Only honour the space when it is not so early that we waste most of the
    # budget (a single enormous token has no spaces at all).
    if cut < int(_BLOCK_LIMIT * 0.5):
        cut = _BLOCK_LIMIT
    return head[:cut].rstrip(), line[cut:].lstrip()


def _split_blocks(text: str) -> list[str]:
    """Break text into packable units: paragraphs, then lines, then long lines."""
    blocks: list[str] = []
    for paragraph in text.split("\n\n"):
        if len(paragraph) <= _BLOCK_LIMIT:
            blocks.append(paragraph)
            continue
        for line in paragraph.split("\n"):
            while len(line) > _BLOCK_LIMIT:
                head, line = _slice_long_line(line)
                blocks.append(head)
            blocks.append(line)
    return blocks


def chunk_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Split a reply into sendable chunks, keeping code fences balanced.

    Discord rejects messages over 2000 characters, so long AI answers (and long
    knowledge-base previews) must be split. Naive slicing can cut through a
    fenced code block: the closing fence would be lost, and the rest of the
    message would render as code. So the fence state of the *source* text is
    tracked, the block is closed at the end of one chunk and re-opened at the
    start of the next, which keeps every chunk independently renderable.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    inside_fence = False   # fence state of the source text
    reopen_fence = False   # next chunk must re-open a block we closed

    def flush() -> None:
        nonlocal current, current_len, reopen_fence
        if not current:
            return
        body = "\n\n".join(current).strip()
        current, current_len = [], 0
        if not body:
            return
        if inside_fence:
            # Close the block here; the continuation re-opens it below.
            body = f"{body}\n{_FENCE}"
            reopen_fence = True
        chunks.append(body[:limit])

    for block in _split_blocks(text):
        addition = len(block) + (2 if current else 0)
        if current and current_len + addition > limit:
            flush()
        if not current and reopen_fence:
            current.append(_FENCE)
            current_len += len(_FENCE) + 2
            reopen_fence = False
        current.append(block)
        current_len += len(block) + (2 if len(current) > 1 else 0)
        if block.count(_FENCE) % 2 == 1:
            inside_fence = not inside_fence

    flush()

    # Final safety net: a single token longer than the limit still has to fit.
    final: list[str] = []
    for chunk in chunks:
        while len(chunk) > limit:
            final.append(chunk[:limit])
            chunk = chunk[limit:]
        if chunk:
            final.append(chunk)
    return final or [truncate(text, limit)]


def chunk_sequence(parts: Iterable[str], limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Chunk and flatten several texts (used for paginated knowledge bases)."""
    out: list[str] = []
    for part in parts:
        out.extend(chunk_message(part, limit))
    return out


def humanize_count(count: int, singular: str, plural: str | None = None) -> str:
    plural = plural or f"{singular}s"
    return f"{count} {singular if count == 1 else plural}"
