FROM python:3.12-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir .

# Non-root user for security
RUN useradd -m botuser
USER botuser

ENV BOT_MODE=BACKTEST
ENV KILL_SWITCH=false

CMD ["python", "-m", "kucoin_bot"]
