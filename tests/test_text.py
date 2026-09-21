"""Text safety: chunking under Discord's limit and neutralising mentions."""

from __future__ import annotations

import pytest

from bot.utils.text import (
    chunk_message,
    chunk_sequence,
    humanize_count,
    sanitize_ai_text,
    strip_code_fence,
    strip_echoed_prefix,
    truncate,
)


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #
def test_short_text_is_a_single_chunk():
    assert chunk_message("hello") == ["hello"]


def test_empty_text_produces_no_chunks():
    assert chunk_message("") == []
    assert chunk_message("   \n  ") == []


def test_every_chunk_stays_within_the_discord_limit():
    text = "\n\n".join(f"Paragraph {index} with some words to bulk it out." for index in range(300))
    chunks = chunk_message(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 2000 for chunk in chunks)


def test_no_content_is_lost_when_chunking():
    """Every word must survive the split — a dropped answer is a wrong answer."""
    words = [f"word{index}" for index in range(1500)]
    text = " ".join(words)
    chunks = chunk_message(text)

    assert len(chunks) > 1
    reconstructed = " ".join(chunks)
    missing = [word for word in words if word not in reconstructed]
    assert not missing, f"{len(missing)} words were dropped, e.g. {missing[:5]}"
    # Order must be preserved too.
    assert reconstructed.index("word0") < reconstructed.index("word1499")


def test_long_unbroken_token_is_hard_split():
    text = "A" * 5000
    chunks = chunk_message(text)
    assert len(chunks) >= 3
    assert all(len(chunk) <= 2000 for chunk in chunks)
    assert "".join(chunks) == text


def test_code_fences_stay_balanced_across_chunks():
    body = "\n".join(f"line {index} of a very long code block" for index in range(200))
    text = f"Here you go:\n```python\n{body}\n```\nDone."
    chunks = chunk_message(text)
    assert len(chunks) > 1
    for chunk in chunks:
        # An unbalanced fence count would render the rest of the message as code.
        assert chunk.count("```") % 2 == 0, f"unbalanced fence in chunk: {chunk[:80]!r}"


def test_code_block_continuation_is_reopened_in_the_next_chunk():
    """A chunk that ends inside a code block must start the next one with a fence."""
    body = "\n".join(f"line {index} of a very long code block" for index in range(200))
    text = f"Here you go:\n```python\n{body}\n```\nDone."
    chunks = chunk_message(text)

    assert len(chunks) >= 3
    assert not chunks[0].lstrip().startswith("```"), "the first chunk opens naturally"
    assert chunks[0].rstrip().endswith("```"), "and must close the block it started"
    for middle in chunks[1:-1]:
        assert middle.lstrip().startswith("```"), "continuation must re-open the block"
        assert middle.rstrip().endswith("```"), "and close it again"
    assert chunks[-1].lstrip().startswith("```")
    assert "Done." in chunks[-1]
    assert "line 0 of" in chunks[0]
    assert "line 199 of" in "".join(chunks)


def test_chunking_prefers_paragraph_boundaries():
    text = "\n\n".join(f"{'x' * 900}" for _ in range(4))
    chunks = chunk_message(text)
    assert len(chunks) == 2
    assert all("x" * 900 in chunk for chunk in chunks)


def test_chunk_sequence_flattens_multiple_parts():
    parts = ["first part", "y" * 3000, "last part"]
    chunks = chunk_sequence(parts)
    assert chunks[0] == "first part"
    assert chunks[-1] == "last part"
    assert all(len(chunk) <= 2000 for chunk in chunks)


# --------------------------------------------------------------------------- #
# mention sanitising (prompt-injection defence)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw",
    [
        "ping <@123456789012345678>",
        "ping <@!123456789012345678>",
        "role <@&123456789012345678>",
        "channel <#123456789012345678>",
        "role and user mixed <@&123456789012345678> <@123456789012345678>",
    ],
)
def test_discord_tokens_are_removed(raw: str):
    """A model that emits a mention token must not cause a real ping."""
    cleaned = sanitize_ai_text(raw)
    assert "123456789012345678" not in cleaned
    assert "<@" not in cleaned
    assert "<#" not in cleaned


@pytest.mark.parametrize(
    "raw",
    [
        "emoji <:pepe:123456789012345678>",
        "animated <a:pepe:123456789012345678>",
        "timestamp <t:1700000000:R>",
    ],
)
def test_harmless_tokens_are_preserved(raw: str):
    """Emoji/timestamp tokens ping nobody and may come from the KB — keep them."""
    assert sanitize_ai_text(raw) == raw


@pytest.mark.parametrize("raw", ["hey @everyone look", "hey @here look", "@everyone"])
def test_broadcast_mentions_are_neutralised(raw: str):
    cleaned = sanitize_ai_text(raw)
    assert "@everyone" not in cleaned
    assert "@here" not in cleaned
    assert "\u200b" in cleaned, "a zero-width space must break the ping"


def test_plain_text_is_left_alone():
    text = "**Bold** and `code` and a list:\n- one\n- two"
    assert sanitize_ai_text(text) == text


def test_excessive_newlines_are_collapsed():
    assert sanitize_ai_text("a\n\n\n\n\n\n\nb") == "a\n\n\nb"


def test_surrounding_whitespace_is_trimmed():
    assert sanitize_ai_text("  hello  ") == "hello"


# --------------------------------------------------------------------------- #
# fences / echoed prefixes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("```text\nwrapped\n```", "wrapped"),
        ("```\nplain fence\n```", "plain fence"),
        ("```json\n{}\n```", "{}"),
        ("not fenced", "not fenced"),
        ("```partial fence", "```partial fence"),
    ],
)
def test_strip_code_fence(raw, expected):
    assert strip_code_fence(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("AI Agent Response: hello", "hello"),
        ("AI: hello", "hello"),
        ("Assistant: hello", "hello"),
        ("Bot - hello", "hello"),
        ("hello", "hello"),
        ("The AI Agent Response is here", "The AI Agent Response is here"),
    ],
)
def test_strip_echoed_prefix(raw, expected):
    assert strip_echoed_prefix(raw) == expected


# --------------------------------------------------------------------------- #
# truncation
# --------------------------------------------------------------------------- #
def test_truncate_leaves_short_text_untouched():
    assert truncate("short", 100) == "short"


def test_truncate_clips_at_a_word_boundary():
    result = truncate("alpha beta gamma delta", 15)
    assert len(result) <= 15
    assert result.endswith("…")
    assert result.startswith("alpha")


def test_truncate_handles_no_spaces():
    result = truncate("x" * 50, 20)
    assert len(result) == 20


def test_truncate_with_tiny_limit():
    result = truncate("abcdef", 3)
    assert len(result) == 3
    assert result == "ab…"


def test_humanize_count():
    assert humanize_count(1, "character") == "1 character"
    assert humanize_count(2, "character") == "2 characters"
    assert humanize_count(2, "entry", "entries") == "2 entries"
