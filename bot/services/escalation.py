"""Escalation detection, parsing and safe mention construction.

Two independent paths can escalate a ticket:

1. **Pre-flight** (:func:`detect_trigger`) — pattern matching on the member's
   message *before* any API call. High-risk phrasing (self-harm, legal threats,
   compromised accounts) is escalated immediately: no knowledge base should be
   the only responder to those, and skipping the API call also conserves
   free-tier quota.

2. **Post-hoc** (:func:`parse_ai_reply`) — the model emits the ``ESCALATE:``
   token mandated by directive #2 of the system prompt.

Both converge on :class:`EscalationDecision`, which the ticket listener turns
into a Discord message. The staff role mention is *always* rebuilt from the
tenant's database value; mentions appearing in model output are stripped, so a
prompt-injected ticket cannot make the bot ping ``@everyone`` or an arbitrary
role.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Sequence

from bot.constants import (
    ESCALATION_MESSAGE,
    ESCALATION_TOKEN,
    REQUEST_PATTERNS,
    SAFETY_PATTERNS,
    UNCONFIGURED_STAFF_ROLE,
)
from bot.utils.text import sanitize_ai_text

#: The canned sentence the model is told to emit, as a tolerant matcher: models
#: occasionally reflow whitespace or drop the bell emoji.
_CANNED_RE: Final[re.Pattern[str]] = re.compile(
    r"i\s+do\s+n[o0]t\s+have\s+enough\s+information.*?staff\s+team!?",
    re.IGNORECASE | re.DOTALL,
)
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(rf"\b{re.escape(ESCALATION_TOKEN)}\s*", re.IGNORECASE)
_PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(
    rf"<@&?\s*(?:\d+|{re.escape(UNCONFIGURED_STAFF_ROLE)}|none|null|role_?id|staff_?role)>",
    re.IGNORECASE,
)
_BELL_RE: Final[re.Pattern[str]] = re.compile("[🔔🛎🚨]")
#: Punctuation/markdown/emoji residue left after the token and canned sentence
#: are removed (e.g. the ``**`` of a bolded escalation).
_EDGE_NOISE: Final[str] = "\\s:>*_`~\"'.!🔔🛎🚨-"
_LEADING_NOISE_RE: Final[re.Pattern[str]] = re.compile(rf"^[{_EDGE_NOISE}]+")
_TRAILING_NOISE_RE: Final[re.Pattern[str]] = re.compile(rf"[{_EDGE_NOISE}]+$")


class ReplyKind(str, Enum):
    """What the bot should do with a model reply."""

    ANSWER = "answer"
    ESCALATE = "escalate"


class TriggerTier(str, Enum):
    """Why a pre-flight escalation fired."""

    SAFETY = "safety"
    REQUEST = "request"
    NONE = "none"


@dataclass(frozen=True)
class TriggerMatch:
    """A pre-flight escalation trigger that matched the member's message."""

    tier: TriggerTier
    pattern: str
    matched_text: str

    @property
    def matched(self) -> bool:
        return self.tier is not TriggerTier.NONE


NO_TRIGGER: Final[TriggerMatch] = TriggerMatch(TriggerTier.NONE, "", "")


@dataclass(frozen=True)
class EscalationDecision:
    """Fully resolved instruction for the ticket listener."""

    kind: ReplyKind
    #: Message body to post, already sanitised and mention-free.
    content: str
    #: Real ``<@&id>`` mention to append, or ``None`` when no staff role is set.
    staff_mention: str | None = None
    reason: str = ""
    trigger: TriggerMatch = field(default=NO_TRIGGER)
    #: True when the decision came from the model's ESCALATE: token.
    from_model: bool = False
    #: Set when the escalation could not be delivered because no role exists.
    staff_role_missing: bool = False

    @property
    def should_escalate(self) -> bool:
        return self.kind is ReplyKind.ESCALATE

    def build_message(self) -> str:
        """Final Discord-ready text, mention appended last and exactly once."""
        body = self.content.strip()
        if self.should_escalate and self.staff_mention:
            return f"{body}\n\n{self.staff_mention}" if body else self.staff_mention
        return body


def build_staff_mention(staff_role_id: str | int | None) -> str | None:
    """Validate and render the tenant's staff role mention.

    Only a purely numeric snowflake from the database is accepted — this is what
    stops an injected ``<@&…>`` in model output from ever being sent.
    """
    if staff_role_id is None:
        return None
    candidate = str(staff_role_id).strip()
    if not candidate or not candidate.isdigit():
        return None
    return f"<@&{candidate}>"


def detect_trigger(
    text: str,
    *,
    check_safety: bool = True,
    check_requests: bool = False,
    safety_patterns: Sequence[re.Pattern[str]] = SAFETY_PATTERNS,
    request_patterns: Sequence[re.Pattern[str]] = REQUEST_PATTERNS,
) -> TriggerMatch:
    """Scan a member message for escalation triggers (no API call involved).

    ``check_requests`` defaults to False on purpose: phrases like *refund* also
    appear in perfectly answerable questions ("what is your refund policy?"),
    and the request-shaped patterns below require an actual ask ("I want a
    refund") to reduce false positives. The model is separately instructed by
    directive #2 to escalate refund/ban-appeal/manual-support requests, so the
    behaviour the specification asks for still holds by default.
    """
    if not text:
        return NO_TRIGGER
    if check_safety:
        for pattern in safety_patterns:
            match = pattern.search(text)
            if match:
                return TriggerMatch(TriggerTier.SAFETY, pattern.pattern, match.group(0))
    if check_requests:
        for pattern in request_patterns:
            match = pattern.search(text)
            if match:
                return TriggerMatch(TriggerTier.REQUEST, pattern.pattern, match.group(0))
    return NO_TRIGGER


def looks_like_escalation(text: str) -> bool:
    """Cheap check for the ``ESCALATE:`` token anywhere near the start."""
    if not text:
        return False
    stripped = text.strip()
    if stripped.upper().startswith(ESCALATION_TOKEN.upper()):
        return True
    # Some models wrap the token in quotes, bold markers or a leading apology.
    head = re.sub(r"^[\s\"'*>_`#\-]+", "", stripped)[:400]
    return bool(_TOKEN_RE.match(head))


def parse_ai_reply(
    raw_text: str,
    *,
    staff_role_id: str | int | None = None,
) -> EscalationDecision:
    """Classify a model reply as an answer or an escalation.

    Handles the realistic variations models produce:
      * the exact mandated string
      * the token with extra explanation appended
      * the token wrapped in quotes/markdown or a code fence
      * the canned sentence *without* the token (models drop it occasionally)
      * a literal ``<@&0>`` / ``<@&STAFF_ROLE_ID>`` placeholder mention
    """
    text = (raw_text or "").strip()
    mention = build_staff_mention(staff_role_id)
    staff_missing = mention is None

    if not text:
        # An empty completion cannot resolve a ticket — hand it to a human.
        return EscalationDecision(
            kind=ReplyKind.ESCALATE,
            content=ESCALATION_MESSAGE,
            staff_mention=mention,
            reason="empty model response",
            staff_role_missing=staff_missing,
        )

    if looks_like_escalation(text) or _CANNED_RE.search(text):
        remainder = _strip_escalation_noise(text)
        reason = remainder or ESCALATION_MESSAGE
        return EscalationDecision(
            kind=ReplyKind.ESCALATE,
            content=reason,
            staff_mention=mention,
            reason="model emitted ESCALATE token",
            from_model=True,
            staff_role_missing=staff_missing,
        )

    return EscalationDecision(
        kind=ReplyKind.ANSWER,
        content=sanitize_ai_text(text),
        staff_mention=None,
        reason="",
        staff_role_missing=False,
    )


def _strip_escalation_noise(text: str) -> str:
    """Remove the token, the canned sentence and any placeholder mention.

    Whatever survives is the model's own extra explanation, which is useful
    context for the staff member picking the ticket up.
    """
    working = text.strip()

    # Unwrap a code fence if the model quoted the whole response.
    if working.startswith("```") and working.endswith("```"):
        working = working.strip("`")
        if "\n" in working:
            first, rest = working.split("\n", 1)
            if len(first.strip()) <= 20 and " " not in first.strip():
                working = rest

    working = _TOKEN_RE.sub("", working, count=1)
    working = _PLACEHOLDER_RE.sub("", working)
    working = _CANNED_RE.sub("", working)
    working = sanitize_ai_text(working)
    working = _BELL_RE.sub("", working)

    # Strip the punctuation/markdown residue the removals leave behind. Repeating
    # handles interleaved cases such as "** 🔔 **".
    for _ in range(3):
        cleaned = _LEADING_NOISE_RE.sub("", working)
        cleaned = _TRAILING_NOISE_RE.sub("", cleaned).strip()
        if cleaned == working:
            break
        working = cleaned
    return working.strip()


def escalation_message(reason: str = "") -> str:
    """Canned escalation body, optionally followed by the model's explanation."""
    base = ESCALATION_MESSAGE
    cleaned = sanitize_ai_text(reason or "").strip()
    if not cleaned:
        return base
    return f"{base}\n**Context for staff:** {cleaned}"
