-- ===========================================================================
-- Discord AI Ticket Responder — reference schema
--
-- The application creates these tables automatically on first boot via
-- SQLAlchemy (`Base.metadata.create_all`), so running this file is OPTIONAL.
-- It is provided for DBAs who want to provision PostgreSQL/Supabase by hand,
-- review the shape of the data, or wire the schema into their own migrations.
--
-- Dialect notes:
--   * The two specification tables use portable types that work on both SQLite
--     and PostgreSQL.
--   * The two extension tables add operational state (escalation cooldowns and
--     free-tier usage counters). They are additive: removing them does not
--     affect the specified behaviour, but the cooldown/budget features degrade.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- Tenant configuration: one row per Discord server (the isolation boundary)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS server_configs (
    guild_id           VARCHAR(32) PRIMARY KEY,
    server_name        VARCHAR(255),
    knowledge_base     TEXT        NOT NULL DEFAULT '',
    staff_role_id      VARCHAR(32),
    ticket_category_id VARCHAR(32),
    created_at         TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    updated_at         TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- Ticket lifecycle
--   status: open -> answered | escalated -> resolved
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ticket_logs (
    ticket_id  VARCHAR(32) PRIMARY KEY,
    guild_id   VARCHAR(32),
    channel_id VARCHAR(32),
    user_id    VARCHAR(32),
    status     VARCHAR(20) DEFAULT 'open',
    created_at TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (guild_id) REFERENCES server_configs (guild_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_ticket_logs_guild_status ON ticket_logs (guild_id, status);

-- ---------------------------------------------------------------------------
-- EXTENSION: per-ticket counters + escalation cooldown state
-- Used to avoid re-pinging the staff role on every message of a long ticket.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ticket_activity (
    channel_id           VARCHAR(32) PRIMARY KEY,
    guild_id             VARCHAR(32),
    ticket_id            VARCHAR(32),
    ai_reply_count       INTEGER NOT NULL DEFAULT 0,
    escalation_count     INTEGER NOT NULL DEFAULT 0,
    user_message_count   INTEGER NOT NULL DEFAULT 0,
    last_user_message_at TIMESTAMP,
    last_ai_reply_at     TIMESTAMP,
    last_escalation_at   TIMESTAMP,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_ticket_activity_guild_id ON ticket_activity (guild_id);

-- ---------------------------------------------------------------------------
-- EXTENSION: daily LLM usage, so the free-tier budget survives restarts
--   usage_date is a UTC 'YYYY-MM-DD' string (portable across SQLite/Postgres)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_usage (
    id               INTEGER PRIMARY KEY,   -- PostgreSQL: use BIGSERIAL
    usage_date       VARCHAR(10) NOT NULL,
    provider         VARCHAR(32) NOT NULL,
    model            VARCHAR(64) NOT NULL DEFAULT '',
    requests         INTEGER     NOT NULL DEFAULT 0,
    failures         INTEGER     NOT NULL DEFAULT 0,
    prompt_chars     INTEGER     NOT NULL DEFAULT 0,
    completion_chars INTEGER     NOT NULL DEFAULT 0,
    latency_ms_total INTEGER     NOT NULL DEFAULT 0,
    updated_at       TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_llm_usage_usage_date ON llm_usage (usage_date);

-- ---------------------------------------------------------------------------
-- Useful operational views (optional)
-- ---------------------------------------------------------------------------
-- Servers that have not finished setup:
--   SELECT guild_id, server_name,
--          CASE WHEN knowledge_base = ''       THEN 'needs /setup-kb '            ELSE '' END ||
--          CASE WHEN staff_role_id IS NULL     THEN 'needs /set-staff-role '      ELSE '' END ||
--          CASE WHEN ticket_category_id IS NULL THEN 'needs /set-ticket-category' ELSE '' END AS missing
--   FROM server_configs
--   WHERE knowledge_base = '' OR staff_role_id IS NULL OR ticket_category_id IS NULL;
--
-- Today's spend against the free tier:
--   SELECT provider, model, SUM(requests) AS requests, SUM(failures) AS failures
--   FROM llm_usage WHERE usage_date = date('now') GROUP BY provider, model;
