FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir .

# Copy application code (secrets excluded via .dockerignore)
COPY kucoin_bot/ kucoin_bot/

# Re-install in editable-ish mode so entry point resolves
RUN pip install --no-cache-dir --no-deps .

# Non-root user for security
RUN useradd -m botuser && mkdir -p /app/data && chown botuser:botuser /app/data
USER botuser

ENV BOT_MODE=BACKTEST
ENV KILL_SWITCH=false
ENV DB_URL=sqlite:////app/data/kucoin_bot.db

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9090/healthz')" || exit 1

CMD ["python", "-m", "kucoin_bot"]
