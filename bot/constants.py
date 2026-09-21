"""Immutable constants: the AI system prompt template, limits and trigger phrases.

The prompt template below is the single source of truth for how server-specific
knowledge is injected into every LLM call. It is rendered by
:mod:`bot.services.prompt_builder`.
"""

from __future__ import annotations

import re
from typing import Final

# --------------------------------------------------------------------------- #
# Prompt contract
# --------------------------------------------------------------------------- #

#: Literal token the model must emit when it cannot resolve a ticket.
ESCALATION_TOKEN: Final[str] = "ESCALATE:"

#: Canned escalation sentence, mandated verbatim by the escalation protocol.
ESCALATION_MESSAGE: Final[str] = (
    "I do not have enough information to resolve this issue. "
    "Flagging this ticket for our staff team! \U0001f514"
)

#: Full string the model is instructed to output (mention placeholder included).
ESCALATION_FULL_RESPONSE: Final[str] = f"{ESCALATION_TOKEN} {ESCALATION_MESSAGE}"

SYSTEM_PROMPT_TEMPLATE: Final[str] = """You are the official AI Support Agent for the Discord server "{server_name}".
Your sole job is to answer user tickets instantly using ONLY the facts explicitly provided in the Knowledge Base below.

=== SERVER KNOWLEDGE BASE ===
{knowledge_base}
=============================

=== STRICT DIRECTIVES ===
1. STAY IN BOUNDS: You are strictly forbidden from inventing rules, prices, URLs, or policies not directly written in the Knowledge Base.
2. ESCALATION PROTOCOL: If the answer is NOT in the Knowledge Base, or if the user asks for refund/ban appeals/manual staff support, output EXACTLY this string and nothing else:
   "{escalation_response}"
3. TONE & FORMATTING:
   - Be clear, polite, and direct.
   - Keep replies under 4 sentences unless listing multi-step guides.
   - Use Markdown (**bold**, bullet points) for readability on Discord.
4. CHAT HISTORY: Review the conversation history below to maintain continuity. Never repeat information you already provided.

=== CHAT HISTORY ===
{chat_history}
====================

Current User Query: {latest_message}
AI Agent Response:"""

#: Sent when a server has not uploaded a knowledge base yet.
EMPTY_KB_PLACEHOLDER: Final[str] = (
    "(No knowledge base has been configured for this server yet. "
    "Because you have no approved facts to draw from, you must use the ESCALATION PROTOCOL.)"
)

#: Rendered when the guild has no staff role configured.
UNCONFIGURED_STAFF_ROLE: Final[str] = "STAFF_ROLE_NOT_CONFIGURED"

# --------------------------------------------------------------------------- #
# Discord limits
# --------------------------------------------------------------------------- #

DISCORD_MESSAGE_LIMIT: Final[int] = 2000
DISCORD_EMBED_LIMIT: Final[int] = 4096
MAX_ATTACHMENT_BYTES: Final[int] = 1_000_000  # 1 MB is plenty for a KB text file
ALLOWED_KB_EXTENSIONS: Final[frozenset[str]] = frozenset({".txt", ".md", ".markdown", ".text"})

# --------------------------------------------------------------------------- #
# Conversation shaping
# --------------------------------------------------------------------------- #

#: Individual history lines are clipped so one wall-of-text cannot crowd out the KB.
HISTORY_MESSAGE_CHAR_LIMIT: Final[int] = 1200

#: The rendered system prompt is hard-capped to keep free-tier latency low.
MAX_SYSTEM_PROMPT_CHARS: Final[int] = 24_000

ATTACHMENT_PLACEHOLDER: Final[str] = "[attached file: {name}]"

# --------------------------------------------------------------------------- #
# Ticket status vocabulary (server_configs.ticket_logs.status)
# --------------------------------------------------------------------------- #

STATUS_OPEN: Final[str] = "open"
STATUS_ANSWERED: Final[str] = "answered"
STATUS_ESCALATED: Final[str] = "escalated"
STATUS_RESOLVED: Final[str] = "resolved"

#: Statuses in which the AI keeps answering. ``resolved``/``closed`` stop it.
ACTIVE_TICKET_STATUSES: Final[frozenset[str]] = frozenset({STATUS_OPEN, STATUS_ANSWERED, STATUS_ESCALATED})

# --------------------------------------------------------------------------- #
# Escalation triggers
# --------------------------------------------------------------------------- #
# Two tiers, both evaluated *before* spending an API call:
#
#   SAFETY_PATTERNS   – high-risk phrasing. Always escalated (configurable via
#                       ESCALATION_PREFLIGHT_SAFETY) because no knowledge base
#                       should be the only responder to these.
#   REQUEST_PATTERNS  – "I want a refund" style asks. Off by default so that
#                       answerable questions such as "what is your refund
#                       policy?" are still served from the KB; the model itself
#                       is instructed to escalate these via directive #2.

SAFETY_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:kill(?:ing)?\s+myself|suicid\w*|self[-\s]?harm\w*|end\s+my\s+(?:own\s+)?life)\b",
        r"\b(?:sue\s+you|suing\s+you|lawyer|attorney|legal\s+action|take\s+(?:you|this)\s+to\s+court)\b",
        r"\b(?:doxx\w*|leak(?:ed|ing)?\s+(?:my|their|his|her)\s+(?:address|location|info\w*|data))\b",
        r"\b(?:my\s+account\s+(?:was|is|got)\s+hack\w*|account\s+compromis\w*|stolen\s+account|"
        r"unauthori[sz]ed\s+(?:access|login|charge|purchase))\b",
        r"\b(?:report(?:ing)?\s+(?:a\s+)?(?:user|member|server|staff)|harass(?:ed|ing|ment)|"
        r"threaten(?:ed|ing)?\s+me|death\s+threat)\b",
        r"\b(?:minor|child)\s+(?:abuse|exploitation|pornograph\w*)\b",
    )
)

REQUEST_PATTERNS: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:want|need|request\w*|demand|get|issue|give|process|file|open|start|claim)\s+"
        r"(?:a\s+|my\s+|the\s+|full\s+|partial\s+)*refund\b",
        r"\brefund\s+(?:me|my|this|that|the)\b",
        r"\bcharge\s*back\b|\bchargeback\b|\bpayment\s+dispute\b|\bdispute\s+(?:the\s+)?charge\b",
        r"\bban\s+appeal\b|\bappeal\s+(?:my|this|a|the)\s+ban\b|\bunban\s+(?:me|my)\b|"
        r"\b(?:was|been|got)\s+(?:wrongly\s+|unfairly\s+)?(?:ban|kick|mute)\w*\b",
        r"\b(?:speak|talk|chat)\s+(?:to|with)\s+(?:a\s+|an\s+|the\s+)?(?:human|person|real\s+person|"
        r"staff|mod(?:erator)?|admin|agent|representative|support\s+rep)\b",
        r"\b(?:need|want|request\w*)\s+(?:manual|human)\s+(?:support|review|help|intervention)\b",
        r"\b(?:is|are)\s+(?:a\s+|any\s+)?(?:human|real\s+person|staff\s+member|moderator)\s+"
        r"(?:available|there|online|around)\b",
        r"\bmanual(?:ly)?\s+(?:review|refund|approve|override)\b",
    )
)

#: Ping-capable tokens are stripped from model output; the only mention the bot
#: ever sends is the staff role, appended by :mod:`bot.services.escalation` from
#: the database value. Covers every form that notifies someone:
#:   <@id>  <@!id>  <@&id>  <#id>  <@everyone>  <@here>
#: Emoji (<:name:id>) and timestamp (<t:...:R>) tokens are intentionally kept —
#: they ping nobody and may legitimately appear in a server's knowledge base.
MENTION_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"<@!?&?\d+>|<#\d+>|<@(?:everyone|here)>", re.IGNORECASE
)
BROADCAST_MENTION_RE: Final[re.Pattern[str]] = re.compile(r"@(everyone|here)\b", re.IGNORECASE)

#: Zero-width space used to neutralise literal broadcast mentions.
ZERO_WIDTH_SPACE: Final[str] = "\u200b"

# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #

BOT_NAME: Final[str] = "AI Ticket Responder"
DB_ECHO_LABEL: Final[str] = "ticket-bot"

#: Providers that speak the OpenAI-compatible ``/chat/completions`` schema.
OPENAI_COMPATIBLE_PROVIDERS: Final[frozenset[str]] = frozenset({"groq", "cerebras"})
