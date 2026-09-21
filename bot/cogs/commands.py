"""Admin slash commands — the tenant configuration surface.

Every command is ``guild_only`` and gated on ``Manage Guild``, and every write is
scoped to ``interaction.guild_id`` so an admin can only ever touch their own
server's knowledge base. Responses are ephemeral: a knowledge base is private
configuration and should not be broadcast to the channel.

Commands
--------
``/setup-kb``              upload or update the knowledge base (text and/or file)
``/set-staff-role``        role pinged on escalation
``/set-ticket-category``   category the AI is allowed to answer in (omit to clear)
``/view-kb``               show the stored knowledge base
``/ticket-status``         per-server ticket and AI statistics
``/resolve-ticket``        stop the AI answering in this ticket
``/ai-status``             live provider health, rate limits and daily budget
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.ext import commands

from bot.constants import (
    ALLOWED_KB_EXTENSIONS,
    DISCORD_MESSAGE_LIMIT,
    MAX_ATTACHMENT_BYTES,
    STATUS_RESOLVED,
)
from bot.db.repository import GuildConfigRecord
from bot.utils.logging_setup import get_logger
from bot.utils.text import truncate

if TYPE_CHECKING:  # pragma: no cover
    from bot.main import TicketBot

log = get_logger(__name__)

EMBED_COLOR = discord.Color.from_str("#5865F2")  # blurple
WARN_COLOR = discord.Color.orange()
OK_COLOR = discord.Color.green()

MAX_TEXT_OPTION_CHARS = 4000  # Discord caps a single slash-command string option


def _embed(title: str, description: str = "", *, color: discord.Color = EMBED_COLOR) -> discord.Embed:
    embed = discord.Embed(title=title, description=description or None, color=color)
    embed.set_footer(text="AI Ticket Responder")
    embed.timestamp = datetime.now(timezone.utc)
    return embed


def _config_fields(config: GuildConfigRecord) -> list[tuple[str, str, bool]]:
    """Readiness checklist rendered into admin responses."""
    kb_state = (
        f"✅ {config.knowledge_base_chars:,} characters / {config.knowledge_base_words:,} words"
        if config.has_knowledge_base
        else "❌ empty"
    )
    return [
        ("Knowledge base", kb_state, False),
        ("Staff role", f"✅ `<@&{config.staff_role_id}>`" if config.staff_role_id else "❌ not set", False),
        (
            "Ticket category",
            f"✅ `{config.ticket_category_id}`" if config.ticket_category_id else "❌ not set",
            False,
        ),
    ]


def _readiness_embed(config: GuildConfigRecord, *, heading: str) -> discord.Embed:
    missing = config.missing_pieces()
    if missing:
        embed = _embed(
            heading,
            "Setup is incomplete. The bot will keep escalating to staff until these are done:\n"
            + "\n".join(f"• {item}" for item in missing),
            color=WARN_COLOR,
        )
    else:
        embed = _embed(heading, "✅ This server is fully configured. The AI is answering tickets.",
                       color=OK_COLOR)
    for name, value, inline in _config_fields(config):
        embed.add_field(name=name, value=value[:1024], inline=inline)
    return embed


class AdminCommands(commands.Cog):
    """Slash commands for server administrators."""

    def __init__(self, bot: "TicketBot") -> None:
        self.bot = bot
        self.repo = bot.repository

    # ------------------------------------------------------------------ #
    # shared helpers
    # ------------------------------------------------------------------ #
    async def _config(self, interaction: discord.Interaction) -> GuildConfigRecord:
        guild = interaction.guild
        name = guild.name if guild else ""
        return await self.repo.ensure_guild(interaction.guild_id, name)

    @staticmethod
    async def _read_attachment(attachment: discord.Attachment) -> tuple[str | None, str | None]:
        """Return ``(text, error)`` for an uploaded knowledge-base file."""
        suffix = ""
        if "." in attachment.filename:
            suffix = "." + attachment.filename.rsplit(".", 1)[-1].lower()
        if suffix and suffix not in ALLOWED_KB_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_KB_EXTENSIONS))
            return None, f"`{attachment.filename}` is not a text file. Allowed extensions: {allowed}."
        if attachment.size > MAX_ATTACHMENT_BYTES:
            return None, (
                f"`{attachment.filename}` is {attachment.size / 1024:.0f} KB. "
                f"The limit is {MAX_ATTACHMENT_BYTES / 1024:.0f} KB — split it into smaller parts "
                f"and use `mode: Append`."
            )
        try:
            raw = await attachment.read()
        except (discord.HTTPException, discord.NotFound) as exc:
            return None, f"Discord could not deliver the file: {exc}"
        try:
            return raw.decode("utf-8"), None
        except UnicodeDecodeError:
            return None, f"`{attachment.filename}` is not valid UTF-8 text."

    # ------------------------------------------------------------------ #
    # /setup-kb
    # ------------------------------------------------------------------ #
    @app_commands.command(
        name="setup-kb",
        description="Upload or update this server's AI knowledge base (rules, FAQs, procedures).",
    )
    @app_commands.guild_only()
    @app_commands.describe(
        text="Knowledge base text (paste rules/FAQs directly).",
        attachment="Or upload a .txt / .md file containing the knowledge base.",
        mode="Replace overwrites everything; Append adds to the end; Prepend adds to the top.",
    )
    @app_commands.choices(
        mode=[
            app_commands.Choice(name="Replace (default)", value="replace"),
            app_commands.Choice(name="Append to end", value="append"),
            app_commands.Choice(name="Prepend to top", value="prepend"),
        ]
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def setup_kb(
        self,
        interaction: discord.Interaction,
        text: str | None = None,
        attachment: discord.Attachment | None = None,
        mode: str | None = None,
    ) -> None:
        if not text and attachment is None:
            await interaction.response.send_message(
                embed=_embed(
                    "Nothing to save",
                    "Provide `text`, attach a `.txt`/`.md` file, or both.",
                    color=WARN_COLOR,
                ),
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=True, thinking=True)

        pieces: list[str] = []
        if text:
            pieces.append(text.strip())
        if attachment is not None:
            file_text, error = await self._read_attachment(attachment)
            if error:
                await interaction.followup.send(embed=_embed("Upload rejected", error, color=WARN_COLOR),
                                                ephemeral=True)
                return
            if file_text:
                pieces.append(file_text.strip())

        combined = "\n\n".join(piece for piece in pieces if piece).strip()
        if not combined:
            await interaction.followup.send(
                embed=_embed("Nothing to save", "The supplied knowledge base was empty.", color=WARN_COLOR),
                ephemeral=True,
            )
            return

        mode_value = (mode or "replace").lower()
        if mode_value not in {"replace", "append", "prepend"}:
            mode_value = "replace"
        limit = self.bot.settings.knowledge_base_char_limit
        if len(combined) > limit:
            await interaction.followup.send(
                embed=_embed(
                    "Knowledge base too large",
                    f"This upload is **{len(combined):,} characters**, over the {limit:,} limit.\n"
                    "Trim it, or split it into several files and use `mode: Append`.",
                    color=WARN_COLOR,
                ),
                ephemeral=True,
            )
            return

        existing = await self.repo.get_guild(interaction.guild_id, use_cache=False)
        if mode_value in {"append", "prepend"} and existing and existing.knowledge_base:
            projected = len(existing.knowledge_base) + len(combined) + 2
            if projected > limit:
                await interaction.followup.send(
                    embed=_embed(
                        "Knowledge base would exceed the limit",
                        f"Existing {len(existing.knowledge_base):,} + new {len(combined):,} "
                        f"= {projected:,} characters (limit {limit:,}).\n"
                        "Use `mode: Replace` or remove outdated sections first.",
                        color=WARN_COLOR,
                    ),
                    ephemeral=True,
                )
                return

        config = await self.repo.set_knowledge_base(
            interaction.guild_id,
            combined,
            mode=mode_value,  # type: ignore[arg-type]  # validated above
            server_name=interaction.guild.name if interaction.guild else None,
        )
        await self.bot.sync_commands_for(interaction.guild)

        verb = {
            "replace": "replaced",
            "append": "appended to",
            "prepend": "prepended to",
        }.get(mode_value, "replaced")
        log.info(
            "Knowledge base %s for guild %s by %s (%d chars)",
            verb, interaction.guild_id, interaction.user.id, config.knowledge_base_chars,
        )
        await interaction.followup.send(
            embed=_readiness_embed(
                config, heading=f"Knowledge base {verb} ({config.knowledge_base_chars:,} characters)"
            ),
            ephemeral=True,
        )

    # ------------------------------------------------------------------ #
    # /set-staff-role
    # ------------------------------------------------------------------ #
    @app_commands.command(
        name="set-staff-role", description="Set the role the bot pings when it escalates a ticket."
    )
    @app_commands.guild_only()
    @app_commands.describe(role="Staff/moderator role to ping on escalation.")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def set_staff_role(self, interaction: discord.Interaction, role: discord.Role) -> None:
        guild = interaction.guild
        warnings: list[str] = []

        if guild is not None:
            if role.is_default():
                await interaction.response.send_message(
                    embed=_embed(
                        "Invalid staff role",
                        "That is the `@everyone` role — pinging it would notify the entire server "
                        "on every escalation. Pick a dedicated staff role.",
                        color=WARN_COLOR,
                    ),
                    ephemeral=True,
                )
                return
            me = guild.me
            if me is not None and role >= me.top_role:
                warnings.append(
                    f"**{role.name}** is equal to or above my highest role, so Discord will not let "
                    "me ping it. Move my role above the staff role in Server Settings → Roles."
                )
            if not role.mentionable and (me is None or role >= me.top_role):
                warnings.append(f"**{role.name}** is not mentionable by members with lower roles.")

        config = await self.repo.set_staff_role(
            interaction.guild_id, role.id, server_name=guild.name if guild else None
        )
        log.info("Staff role for guild %s set to %s", interaction.guild_id, role.id)

        embed = _readiness_embed(config, heading=f"Staff role set to {role.mention}")
        if warnings:
            embed.add_field(name="⚠️ Heads up", value="\n".join(warnings)[:1024], inline=False)
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # ------------------------------------------------------------------ #
    # /set-ticket-category
    # ------------------------------------------------------------------ #
    @app_commands.command(
        name="set-ticket-category",
        description="Restrict the AI to ticket channels/threads inside one category.",
    )
    @app_commands.guild_only()
    @app_commands.describe(
        category="Ticket category to serve. Leave empty to clear the restriction."
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def set_ticket_category(
        self, interaction: discord.Interaction, category: discord.CategoryChannel | None = None
    ) -> None:
        guild = interaction.guild
        config = await self.repo.set_ticket_category(
            interaction.guild_id,
            category.id if category else None,
            server_name=guild.name if guild else None,
        )

        if category is None:
            heading = "Ticket category cleared"
            if self.bot.settings.require_configured_category:
                tail = (
                    " — and because `REQUIRE_CONFIGURED_CATEGORY=true`, the AI will answer "
                    "**nothing** until you set a category again."
                )
            else:
                tail = "."
            detail = (
                "The AI will now only answer channels whose name matches the configured ticket "
                "pattern" + tail
            )
            embed = _embed(heading, detail, color=WARN_COLOR)
            for name, value, inline in _config_fields(config):
                embed.add_field(name=name, value=value[:1024], inline=inline)
        else:
            channel_count = len(getattr(category, "channels", []) or [])
            log.info(
                "Ticket category for guild %s set to %s (%s)",
                interaction.guild_id, category.id, category.name,
            )
            embed = _readiness_embed(
                config, heading=f"AI responder limited to **{category.name}**"
            )
            embed.add_field(
                name="Scope",
                value=(
                    f"All channels in this category, plus any threads opened inside them "
                    f"({channel_count} channel(s) currently)."
                ),
                inline=False,
            )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # ------------------------------------------------------------------ #
    # /view-kb
    # ------------------------------------------------------------------ #
    @app_commands.command(name="view-kb", description="Show the knowledge base currently stored for this server.")
    @app_commands.guild_only()
    @app_commands.describe(full="Send the entire knowledge base as a downloadable file.")
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def view_kb(self, interaction: discord.Interaction, full: bool = False) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        config = await self._config(interaction)

        if not config.has_knowledge_base:
            await interaction.followup.send(
                embed=_embed(
                    "No knowledge base yet",
                    "Upload one with `/setup-kb`. Until then the AI has no approved facts to use and "
                    "will escalate every ticket to your staff role.",
                    color=WARN_COLOR,
                ),
                ephemeral=True,
            )
            return

        kb = config.knowledge_base
        header = (
            f"**Knowledge base for {config.server_name or 'this server'}**\n"
            f"{config.knowledge_base_chars:,} characters · {config.knowledge_base_words:,} words · "
            f"last updated <t:{int((config.updated_at or datetime.now(timezone.utc)).timestamp())}:R>\n"
        )

        if full or len(kb) > DISCORD_MESSAGE_LIMIT - len(header):
            file = discord.File(
                io.BytesIO(kb.encode("utf-8")), filename=f"knowledge-base-{config.guild_id}.md"
            )
            await interaction.followup.send(content=truncate(header, DISCORD_MESSAGE_LIMIT), file=file,
                                            ephemeral=True)
            return

        await interaction.followup.send(
            content=truncate(f"{header}\n```md\n{kb}\n```", DISCORD_MESSAGE_LIMIT), ephemeral=True
        )

    # ------------------------------------------------------------------ #
    # /ticket-status
    # ------------------------------------------------------------------ #
    @app_commands.command(name="ticket-status", description="Ticket and AI statistics for this server.")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def ticket_status(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        stats = await self.repo.guild_stats(interaction.guild_id)
        config = await self._config(interaction)
        listener = self.bot.listener_stats()

        embed = _embed(f"Ticket status — {config.server_name or 'this server'}")
        embed.add_field(name="Tickets", value=str(stats.total_tickets), inline=True)
        embed.add_field(name="Open / answered", value=str(stats.open_tickets), inline=True)
        embed.add_field(name="Escalated", value=str(stats.escalated_tickets), inline=True)
        embed.add_field(name="Resolved", value=str(stats.resolved_tickets), inline=True)
        embed.add_field(name="AI replies", value=str(stats.ai_replies), inline=True)
        embed.add_field(name="Staff pings", value=str(stats.escalations), inline=True)
        embed.add_field(
            name="Bot-wide (this process)",
            value=(
                f"messages processed: {listener['messages_processed']}\n"
                f"answered: {listener['answered']} · escalated: {listener['escalated']}\n"
                f"active ticket channels: {listener['active_channels']}"
            ),
            inline=False,
        )

        recent = await self.repo.recent_tickets(interaction.guild_id, limit=5)
        if recent:
            lines = [
                f"• <#{record.channel_id}> — `{record.status}` "
                f"({record.ai_reply_count} AI, {record.escalation_count} escalations)"
                for record in recent
                if record.channel_id
            ]
            if lines:
                embed.add_field(name="Recent tickets", value="\n".join(lines)[:1024], inline=False)

        await interaction.followup.send(embed=embed, ephemeral=True)

    # ------------------------------------------------------------------ #
    # /resolve-ticket
    # ------------------------------------------------------------------ #
    @app_commands.command(
        name="resolve-ticket",
        description="Mark the current ticket resolved so the AI stops answering in it.",
    )
    @app_commands.guild_only()
    @app_commands.describe(
        action="`resolve` stops the AI here; `reopen` lets it answer again."
    )
    @app_commands.choices(
        action=[
            app_commands.Choice(name="Resolve (stop AI)", value="resolve"),
            app_commands.Choice(name="Reopen (resume AI)", value="reopen"),
        ]
    )
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def resolve_ticket(
        self, interaction: discord.Interaction, action: str = "resolve"
    ) -> None:
        channel = interaction.channel
        if channel is None:
            await interaction.response.send_message(
                "This command must be used inside the ticket channel.", ephemeral=True
            )
            return

        ticket_id = str(channel.id)
        existing = await self.repo.get_ticket(ticket_id)
        if existing is None:
            await interaction.response.send_message(
                "No ticket has been recorded for this channel yet.", ephemeral=True
            )
            return

        new_status = STATUS_RESOLVED if action == "resolve" else "open"
        record = await self.repo.set_ticket_status(ticket_id, new_status)
        log.info(
            "Ticket %s in guild %s set to %s by %s",
            ticket_id, interaction.guild_id, new_status, interaction.user.id,
        )
        await interaction.response.send_message(
            embed=_embed(
                "Ticket resolved — the AI will stay quiet here"
                if new_status == STATUS_RESOLVED
                else "Ticket reopened — the AI will answer again",
                f"Status is now `{record.status if record else new_status}`.",
                color=OK_COLOR if new_status == STATUS_RESOLVED else EMBED_COLOR,
            ),
            ephemeral=True,
        )

    # ------------------------------------------------------------------ #
    # /ai-status
    # ------------------------------------------------------------------ #
    @app_commands.command(name="ai-status", description="Show AI provider health, rate limits and usage.")
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_guild=True)
    @app_commands.checks.has_permissions(manage_guild=True)
    async def ai_status(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        health = self.bot.ai.health()
        stats = self.bot.ai.stats

        embed = _embed("AI provider health")
        embed.add_field(
            name="Requests",
            value=(
                f"{stats.requests} total · {stats.successes} ok · {stats.failures} failed\n"
                f"{stats.retries} retries · {stats.failovers} failovers · "
                f"avg {stats.average_latency_ms} ms"
            ),
            inline=False,
        )

        for entry in health:
            badge = {"closed": "🟢", "half_open": "🟡", "open": "🔴"}.get(entry["circuit"], "⚪")
            daily = (
                f"{entry['requests_today']:,}/{entry['daily_limit']:,}"
                if entry["daily_limit"] > 0
                else "unlimited"
            )
            value = (
                f"{badge} `{entry['model']}` — circuit **{entry['circuit']}**\n"
                f"RPM limit {entry['rpm_limit']:.0f} · today {daily} requests\n"
                f"{entry['successes']} ok / {entry['failures']} failed · avg {entry['avg_latency_ms']} ms"
            )
            if entry["last_error"]:
                value += f"\nlast error: {truncate(entry['last_error'], 200)}"
            embed.add_field(name=entry["provider"].title(), value=value[:1024], inline=False)

        usage = await self.repo.usage_snapshot(days=1)
        if usage:
            embed.add_field(
                name="Persisted usage (UTC today)",
                value=truncate(
                    "\n".join(
                        f"• {row['provider']}/{row['model']}: {row['requests']} ok, "
                        f"{row['failures']} failed, avg {row['avg_latency_ms']} ms"
                        for row in usage
                    ),
                    1024,
                ),
                inline=False,
            )

        if embed.description is None:
            embed.description = f"Chain order: {' → '.join(self.bot.ai.providers)}"
        await interaction.followup.send(embed=embed, ephemeral=True)

    # ------------------------------------------------------------------ #
    # error handling
    # ------------------------------------------------------------------ #
    async def cog_command_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        """Friendly, ephemeral error messages for every admin command."""
        message, color = _describe_app_command_error(error)
        log.warning(
            "Slash command error in guild %s: %s", getattr(interaction, "guild_id", "?"), error
        )
        try:
            if interaction.response.is_done():
                await interaction.followup.send(
                    embed=_embed("Command failed", message, color=color), ephemeral=True
                )
            else:
                await interaction.response.send_message(
                    embed=_embed("Command failed", message, color=color), ephemeral=True
                )
        except discord.HTTPException:  # pragma: no cover - Discord already unhappy
            log.exception("Could not deliver a slash-command error message")


def _describe_app_command_error(error: app_commands.AppCommandError) -> tuple[str, discord.Color]:
    """Map an app-command error onto operator-readable advice."""
    if isinstance(error, app_commands.MissingPermissions):
        needed = ", ".join(f"`{perm}`" for perm in error.missing_permissions)
        return f"You need {needed} to manage the AI ticket responder.", WARN_COLOR
    if isinstance(error, app_commands.BotMissingPermissions):
        needed = ", ".join(f"`{perm}`" for perm in error.missing_permissions)
        return (
            f"I am missing {needed} in this server. Ask an administrator to grant them, "
            "otherwise I cannot read tickets or ping staff.",
            WARN_COLOR,
        )
    if isinstance(error, app_commands.NoPrivateMessage):
        return "This command can only be used inside a server.", WARN_COLOR
    if isinstance(error, app_commands.CommandOnCooldown):
        return f"Slow down — try again in {error.retry_after:.1f}s.", WARN_COLOR
    if isinstance(error, app_commands.CommandSignatureError):
        return "Those arguments were not valid. Check the command's options and try again.", WARN_COLOR
    if isinstance(error, app_commands.CommandNotFound):
        return "That command is not registered here yet. Re-invite the bot or wait a few minutes.", WARN_COLOR
    return "Something went wrong while running that command. Check the bot logs for details.", WARN_COLOR


async def setup(bot: "TicketBot") -> None:
    await bot.add_cog(AdminCommands(bot))
