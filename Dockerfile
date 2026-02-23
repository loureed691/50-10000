FROM python:3.12-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir .

# Non-root user for security
RUN useradd -m botuser && mkdir -p /app/data && chown botuser:botuser /app/data
USER botuser

ENV BOT_MODE=BACKTEST
ENV KILL_SWITCH=false
ENV DB_URL=sqlite:////app/data/kucoin_bot.db

CMD ["python", "-m", "kucoin_bot"]
