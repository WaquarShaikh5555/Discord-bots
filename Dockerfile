# Runtime image ---------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so the layer is cached across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    # Production deployments usually target PostgreSQL/Supabase:
    && pip install --no-cache-dir "asyncpg>=0.29"

COPY bot/ ./bot/
COPY schema.sql README.md ./

# Run unprivileged; the SQLite database lives in /app/data (mount a volume).
RUN useradd --create-home --uid 10001 botuser \
    && mkdir -p /app/data \
    && chown -R botuser:botuser /app
USER botuser

VOLUME ["/app/data"]

HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import bot.config" || exit 1

CMD ["python", "-m", "bot"]
