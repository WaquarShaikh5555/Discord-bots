"""Prompt rendering: the exact contract from the specification."""

from __future__ import annotations

from string import Formatter

from bot.constants import ESCALATION_TOKEN, SYSTEM_PROMPT_TEMPLATE
from bot.services.prompt_builder import (
    PromptBuilder,
    build_transcript,
    escalation_instruction,
    render_chat_history,
)
from fakes import FakeMessage, FakeRole, FakeUser

BOT_ID = 4242
STAFF_ROLE = 999


def test_template_placeholders_match_specification():
    """The template must expose exactly the documented placeholders."""
    fields = {
        field_name
        for _literal, field_name, _spec, _conv in Formatter().parse(SYSTEM_PROMPT_TEMPLATE)
        if field_name
    }
    assert fields == {
        "server_name",
        "knowledge_base",
        "escalation_response",
        "chat_history",
        "latest_message",
    }


def test_escalation_instruction_is_the_mandated_string():
    expected = (
        "ESCALATE: I do not have enough information to resolve this issue. "
        "Flagging this ticket for our staff team! \U0001f514 <@&999>"
    )
    assert escalation_instruction(999) == expected


def test_escalation_instruction_without_staff_role_still_uses_token():
    result = escalation_instruction(None)
    assert result.startswith(ESCALATION_TOKEN)
    assert "STAFF_ROLE_NOT_CONFIGURED" in result
    assert "<@&" not in result  # never invent a mention


def test_build_injects_tenant_values():
    builder = PromptBuilder()
    rendered = builder.build(
        server_name="Acme Gaming",
        knowledge_base="Refunds within 14 days.",
        staff_role_id=STAFF_ROLE,
        latest_message="How long do refunds take?",
        history=(),
        bot_user_id=BOT_ID,
    )
    prompt = rendered.system_prompt
    assert 'Discord server "Acme Gaming"' in prompt
    assert "=== SERVER KNOWLEDGE BASE ===" in prompt
    assert "Refunds within 14 days." in prompt
    assert "<@&999>" in prompt
    assert "Current User Query: How long do refunds take?" in prompt
    assert prompt.endswith("AI Agent Response:")
    assert rendered.knowledge_base_chars == len("Refunds within 14 days.")


def test_empty_knowledge_base_forces_escalation_guidance():
    rendered = PromptBuilder().build(
        server_name="Acme",
        knowledge_base="   ",
        staff_role_id=STAFF_ROLE,
        latest_message="hello",
        bot_user_id=BOT_ID,
    )
    assert "No knowledge base has been configured" in rendered.system_prompt
    assert ESCALATION_TOKEN in rendered.system_prompt


def test_knowledge_base_with_braces_does_not_break_formatting():
    """`.format()` on a KB containing `{}` must not explode."""
    rendered = PromptBuilder().build(
        server_name="Acme",
        knowledge_base='Use {"json": "like this"} and {placeholders} freely.',
        staff_role_id=None,
        latest_message="how do I format {this}?",
        bot_user_id=BOT_ID,
    )
    assert '{"json": "like this"}' in rendered.system_prompt
    assert "{this}" in rendered.system_prompt


def test_history_uses_user_and_ai_labels():
    messages = [
        FakeMessage(id=1, content="hi", author=FakeUser(id=1001, name="alice")),
        FakeMessage(id=2, content="Hello! How can I help?", author=FakeUser(id=BOT_ID, bot=True)),
        FakeMessage(id=3, content="refund policy?", author=FakeUser(id=1001, name="alice")),
    ]
    lines = build_transcript(messages, bot_user_id=BOT_ID)
    assert [line.render() for line in lines] == [
        "User: hi",
        "AI: Hello! How can I help?",
        "User: refund policy?",
    ]
    assert render_chat_history(lines).startswith("User: hi")


def test_staff_replies_are_labelled_separately_from_the_ai():
    staff = FakeUser(id=2002, name="mod", roles=(FakeRole(id=STAFF_ROLE),))
    messages = [
        FakeMessage(id=1, content="need help", author=FakeUser(id=1001)),
        FakeMessage(id=2, content="I've refunded you manually", author=staff),
    ]
    lines = build_transcript(messages, bot_user_id=BOT_ID, staff_role_id=STAFF_ROLE)
    assert lines[-1].speaker == "Staff"


def test_history_is_chronological_even_if_input_is_reversed():
    messages = [
        FakeMessage(id=3, content="third"),
        FakeMessage(id=1, content="first"),
        FakeMessage(id=2, content="second"),
    ]
    lines = build_transcript(messages, bot_user_id=BOT_ID)
    assert [line.text for line in lines] == ["first", "second", "third"]


def test_history_is_capped_at_ten_messages():
    messages = [FakeMessage(id=index, content=f"message {index}") for index in range(1, 26)]
    lines = build_transcript(messages, bot_user_id=BOT_ID, message_limit=10)
    assert len(lines) == 10
    assert lines[-1].text == "message 25"  # newest survives
    assert lines[0].text == "message 16"   # oldest nine are dropped


def test_history_char_budget_drops_oldest_lines_first():
    messages = [FakeMessage(id=index, content="x" * 400) for index in range(1, 11)]
    lines = build_transcript(messages, bot_user_id=BOT_ID, char_budget=1500)
    assert len(lines) < 10
    assert all(len(line.text) <= 400 for line in lines)


def test_empty_history_renders_placeholder():
    assert render_chat_history(()) == "(no previous messages in this ticket)"


def test_whitespace_is_collapsed_in_transcript_lines():
    messages = [FakeMessage(id=1, content="line one\n\n\nline two     trailing   ")]
    lines = build_transcript(messages, bot_user_id=BOT_ID)
    assert lines[0].text == "line one line two trailing"


def test_attachment_only_message_enters_history():
    from fakes import FakeAttachment

    message = FakeMessage(id=1, content="", attachments=(FakeAttachment(filename="error.png"),))
    lines = build_transcript([message], bot_user_id=BOT_ID)
    assert lines and "error.png" in lines[0].text


def test_oversized_prompt_truncates_the_knowledge_base_not_the_query():
    builder = PromptBuilder(max_system_prompt_chars=4000)
    rendered = builder.build(
        server_name="Acme",
        knowledge_base="K" * 20_000,
        staff_role_id=STAFF_ROLE,
        latest_message="THE CRITICAL QUERY",
        bot_user_id=BOT_ID,
    )
    assert rendered.truncated is True
    assert len(rendered.system_prompt) <= 4000
    # Directives and the live query must survive even after truncation.
    assert "=== STRICT DIRECTIVES ===" in rendered.system_prompt
    assert "THE CRITICAL QUERY" in rendered.system_prompt
