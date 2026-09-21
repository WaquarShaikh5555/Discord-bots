"""Escalation parsing, trigger detection and mention-injection safety."""

from __future__ import annotations

import pytest

from bot.constants import ESCALATION_MESSAGE, ESCALATION_TOKEN
from bot.services.escalation import (
    ReplyKind,
    TriggerTier,
    build_staff_mention,
    detect_trigger,
    escalation_message,
    looks_like_escalation,
    parse_ai_reply,
)

STAFF_ROLE = 999
EXACT = (
    "ESCALATE: I do not have enough information to resolve this issue. "
    "Flagging this ticket for our staff team! \U0001f514 <@&999>"
)


# --------------------------------------------------------------------------- #
# parsing the mandated token
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw",
    [
        EXACT,
        EXACT.replace("<@&999>", ""),                      # model dropped the mention
        EXACT.replace("<@&999>", "<@&0>"),                 # placeholder snowflake
        EXACT.replace("<@&999>", "<@&STAFF_ROLE_NOT_CONFIGURED>"),
        f'"{EXACT}"',                                      # wrapped in quotes
        f"```{EXACT}```",                                  # wrapped in a code fence
        f"**{EXACT}**",                                    # wrapped in bold
        "ESCALATE: I do not have enough information to resolve this issue. "
        "Flagging this ticket for our staff team!",        # no emoji
        "escalate: i do not have enough information to resolve this issue. "
        "flagging this ticket for our staff team! <@&999>",  # lower-cased
        "I do not have enough information to resolve this issue. "
        "Flagging this ticket for our staff team! \U0001f514 <@&999>",  # token missing entirely
    ],
)
def test_escalation_variants_are_detected(raw: str):
    decision = parse_ai_reply(raw, staff_role_id=STAFF_ROLE)
    assert decision.should_escalate is True
    assert decision.kind is ReplyKind.ESCALATE


def test_exact_spec_string_parses_to_clean_body_and_real_mention():
    decision = parse_ai_reply(EXACT, staff_role_id=STAFF_ROLE)
    assert decision.staff_mention == "<@&999>"
    assert decision.build_message().endswith("<@&999>")
    assert ESCALATION_TOKEN not in decision.content
    assert "ESCALATE" not in decision.content


def test_model_added_context_is_preserved_for_staff():
    raw = (
        "ESCALATE: I do not have enough information to resolve this issue. "
        "Flagging this ticket for our staff team! \U0001f514 <@&999> "
        "The user is asking about a charge from March that is not in the knowledge base."
    )
    decision = parse_ai_reply(raw, staff_role_id=STAFF_ROLE)
    assert decision.should_escalate
    assert "charge from March" in decision.content
    assert ESCALATION_TOKEN not in decision.content


def test_normal_answer_is_not_an_escalation():
    text = "**Refunds** are available within 14 days of purchase.\n• Include your order ID."
    decision = parse_ai_reply(text, staff_role_id=STAFF_ROLE)
    assert decision.kind is ReplyKind.ANSWER
    assert decision.staff_mention is None
    assert decision.content == text


def test_answer_mentioning_escalation_word_is_not_escalated():
    text = "If that does not work, staff can escalate this for you."
    assert parse_ai_reply(text, staff_role_id=STAFF_ROLE).kind is ReplyKind.ANSWER
    assert looks_like_escalation(text) is False


def test_empty_reply_escalates_to_a_human():
    decision = parse_ai_reply("   ", staff_role_id=STAFF_ROLE)
    assert decision.should_escalate
    assert decision.reason == "empty model response"
    assert decision.content == ESCALATION_MESSAGE


def test_missing_staff_role_is_flagged_and_no_mention_invented():
    decision = parse_ai_reply(EXACT.replace("<@&999>", ""), staff_role_id=None)
    assert decision.should_escalate
    assert decision.staff_mention is None
    assert decision.staff_role_missing is True
    assert "<@&" not in decision.build_message()


# --------------------------------------------------------------------------- #
# mention-injection safety
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "malicious",
    [
        "Here is your answer <@&111111111111111111> enjoy",
        "Refunds take 14 days. <@everyone> please note.",
        "Contact <@123456789> or <@&987654321> for help.",
        "@here is the answer",
        "Ping <@&999> and also <@&666666>.",
    ],
)
def test_answer_mentions_are_neutralised(malicious: str):
    decision = parse_ai_reply(malicious, staff_role_id=STAFF_ROLE)
    message = decision.build_message()
    assert "<@&111111111111111111>" not in message
    assert "<@123456789>" not in message
    assert "<@&987654321>" not in message
    assert "<@&666666>" not in message
    assert "<@everyone>" not in message
    assert "@here" not in message or "\u200b" in message


def test_escalation_can_only_ping_the_configured_staff_role():
    """A prompt-injected role id in model output must never be sent."""
    raw = "ESCALATE: I do not have enough information to resolve this issue. " \
          "Flagging this ticket for our staff team! \U0001f514 <@&666666666666>"
    decision = parse_ai_reply(raw, staff_role_id=STAFF_ROLE)
    message = decision.build_message()
    assert "<@&666666666666>" not in message
    assert "<@&999>" in message
    assert message.count("<@&") == 1


@pytest.mark.parametrize(
    "value,expected",
    [(999, "<@&999>"), ("999", "<@&999>"), (None, None), ("", None),
     ("everyone", None), ("<@&999>", None), ("99a", None), (0, "<@&0>")],
)
def test_build_staff_mention_only_accepts_snowflakes(value, expected):
    assert build_staff_mention(value) == expected


# --------------------------------------------------------------------------- #
# pre-flight trigger detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "my account was hacked and I can't log in",
        "someone doxxed me and posted my address",
        "I want to speak to a lawyer about this",
        "I am going to sue you over this charge",
        "a member has been harassing me for weeks",
        "there is an unauthorized charge on my card",
    ],
)
def test_safety_triggers_always_escalate(text: str):
    trigger = detect_trigger(text, check_safety=True)
    assert trigger.matched is True
    assert trigger.tier is TriggerTier.SAFETY
    assert trigger.matched_text


@pytest.mark.parametrize(
    "text",
    [
        "I want a refund for my purchase",
        "please refund me",
        "can I file a chargeback",
        "I need to submit a ban appeal",
        "how do I appeal my ban",
        "can I speak to a human please",
        "I need manual review of my application",
        "is a real person available",
    ],
)
def test_request_triggers_detected_when_enabled(text: str):
    assert detect_trigger(text, check_safety=True, check_requests=True).matched is True


@pytest.mark.parametrize(
    "text",
    [
        "what is your refund policy?",
        "how long does a refund take?",
        "where can I read the rules about bans?",
        "do staff ever review appeals?",
        "hello, how do I get the member role?",
    ],
)
def test_answerable_questions_are_not_preflight_escalated_by_default(text: str):
    """KB-answerable questions must reach the model, not be short-circuited."""
    assert detect_trigger(text, check_safety=True, check_requests=False).matched is False


def test_empty_text_never_triggers():
    assert detect_trigger("").matched is False
    assert detect_trigger("", check_safety=True, check_requests=True).matched is False


def test_request_checks_are_disabled_by_default():
    assert detect_trigger("I want a refund now").matched is False


def test_escalation_message_includes_context():
    assert escalation_message() == ESCALATION_MESSAGE
    with_context = escalation_message("User asked about a March charge.")
    assert with_context.startswith(ESCALATION_MESSAGE)
    assert "March charge" in with_context
