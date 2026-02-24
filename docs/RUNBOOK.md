# Runbook – KuCoin Trading Bot

> **Disclaimer**: This bot trades real money on KuCoin. Profitability is NOT guaranteed.
> Use at your own risk. This is NOT financial advice.

---

## 1. Kill Switch

**When**: Unexpected market conditions, suspected bugs, or manual intervention needed.

**Steps**:

1. Set the environment variable `KILL_SWITCH=true` (or update `.env`).
2. The bot detects the flag within one fast-cycle (≤ 30 s by default).
3. It **cancels all open orders** (spot + futures).
4. It **flattens all positions** using `reduceOnly` market orders on futures.
5. The bot then exits cleanly.

**Verification**:
```bash
# Check /healthz returns 503 (bot is shutting down)
curl -s http://localhost:9090/healthz

# Confirm positions are closed via KuCoin UI or API
```

**Recovery**: Set `KILL_SWITCH=false`, restart the bot.  It will reconcile
exchange state on startup.

---

## 2. Circuit Breaker

**Trigger conditions** (automatic):
- Daily PnL drops below `-MAX_DAILY_LOSS_PCT` of equity (default 3 %).
- Drawdown from peak equity exceeds `MAX_DRAWDOWN_PCT` (default 10 %).

**Behaviour**:
- New entries are **blocked**; exits and stop-loss orders continue.
- Resets automatically at UTC midnight (daily PnL reset).

**Monitoring**:
```promql
circuit_breaker_active == 1       # Alert: CircuitBreakerActive
daily_pnl_usdt                    # Gauge: current daily PnL
```

**Manual override**: No direct override.  To resume trading before midnight,
restart the bot (which resets `daily_pnl` and `circuit_breaker_active`).

---

## 3. API Outage (KuCoin)

**Detection**:
- HTTP 5xx / connection errors from `KuCoinClient._request`.
- Automatic retry with exponential back-off (3 attempts, honours `Retry-After`).
- After max retries, raises `KuCoinAPIError(code='max_retries')`.

**Behaviour**:
- Market data refresh fails → slow path skips signal computation for that cycle.
- Order placement fails → `OrderResult.success=False`.
- **No phantom position booking** — only confirmed fills update positions.

**Monitoring**:
```promql
rate(api_429_total[5m])           # Rate-limit hits
rate(api_errors_total[5m])        # General API errors
```

**Response**:
1. Check KuCoin status page: https://status.kucoin.com
2. If prolonged (> 5 min), consider activating the kill switch.
3. On recovery, the bot resumes automatically; startup reconciliation catches
   any positions opened/closed while the bot was degraded.

---

## 4. Database Outage

**Detection**:
- `init_db()` or session commits raise `SQLAlchemy` exceptions.
- Logged as `"Failed to write signal snapshots to DB"`.

**Behaviour**:
- Signal snapshots are **dropped** (not queued) — trading continues.
- Order/trade persistence in the current design is exchange-side; the DB is an
  **audit log**, not the source of truth for positions.
- The bot does **not** crash on DB errors.

**Monitoring**:
```promql
# Check for recurring DB write failures in log aggregation (e.g. Loki):
{app="kucoin-bot"} |= "Failed to write signal snapshots"
```

**Response**:
1. Check database connectivity (Postgres: `pg_isready`, SQLite: disk space).
2. Restart the database service.
3. The bot will reconnect via SQLAlchemy's connection pool on the next write.

---

## 5. DB Retention / Compaction

`SignalSnapshot` rows are purged automatically every ~100 slow cycles.
Default retention: **7 days** (configurable via `SIGNAL_RETENTION_DAYS`).

To run a manual purge:
```python
from kucoin_bot.models import init_db
from kucoin_bot.reporting.retention import purge_old_snapshots

Session = init_db("sqlite:///kucoin_bot.db")
deleted = purge_old_snapshots(Session, days=3)
print(f"Purged {deleted} old snapshots")
```

---

## 6. Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `BOT_MODE` | `BACKTEST` | LIVE / PAPER / SHADOW / BACKTEST |
| `LIVE_TRADING` | `false` | Must be `true` for LIVE mode |
| `KILL_SWITCH` | `false` | Emergency stop |
| `METRICS_HOST` | `0.0.0.0` | HTTP server bind address |
| `METRICS_PORT` | `9090` | HTTP server port |
| `SIGNAL_RETENTION_DAYS` | `7` | Days to keep signal snapshots |
| `MAX_KLINE_CONCURRENCY` | `8` | Parallel kline fetches |
| `BATCH_DB_WRITES` | `1` | Batch signal snapshot inserts |

See `.env.example` for the full list.
