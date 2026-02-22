FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir . 2>/dev/null || pip install --no-cache-dir aiohttp websockets sqlalchemy pyyaml numpy pandas

COPY . .
RUN pip install --no-cache-dir -e .

# Non-root user for security
RUN useradd -m botuser
USER botuser

ENV BOT_MODE=BACKTEST
ENV KILL_SWITCH=false

CMD ["python", "-m", "kucoin_bot"]
