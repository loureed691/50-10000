# KuCoin Auto Trading Bot

A fully automatic, multi-strategy KuCoin API trading bot built in Python.

> **⚠️ WARNING:** This bot trades real money. Leverage amplifies losses.
> Profitability is **NOT guaranteed**. Use at your own risk.

## Features

- **Automatic market discovery** – finds all USDT-quoted pairs + futures contracts, filters by liquidity/spread
- **Multi-source signal engine** – momentum, trend strength, mean reversion, volatility, volume anomalies, orderbook imbalance, funding rate
- **Regime classification** – trending-up/down, ranging, high-volatility, low-liquidity, news-spike
- **6 strategy modules** – trend following, mean reversion, volatility breakout, scalping, hedging, risk-off
- **Robust short trading** – via USDT-margined perpetual futures (preferred) or margin borrow (fallback)
- **Short squeeze protection** – blocks new shorts on volatility spikes, momentum bursts, and crowded-short funding rates
- **Unified cost-aware EV gate** – fees + slippage + funding + borrow vs expected edge; marginal trades rejected in both backtest and live
- **Portfolio-level risk management** – drawdown limits, daily loss caps, exposure limits, circuit breaker, correlated exposure limits with prospective notional
- **Dynamic leverage** – only when justified by signal confidence and risk budget
- **Order lifecycle management** – orders are polled for fill status (pending → partial → filled/cancelled); positions only update from confirmed fills
- **Futures-aware execution** – integer contract sizing (lot-size aligned), `reduceOnly` for exits, separate cancel_all for spot and futures
- **Startup reconciliation** – on restart, fetches open positions and orders from the exchange to rebuild risk state
- **Strategy monitor** – auto-disables modules with persistently negative net expectancy after costs
- **Internal transfers** – automated Funding ↔ Trading ↔ Futures account transfers (opt-in, default enabled), with DB persistence and idempotency keys
- **Kill switch** – set `KILL_SWITCH=true` to immediately stop all trading
- **Full audit trail** – every signal, decision, order, and fill is logged to the database
- **Backtesting** – realistic fees, slippage, funding, borrow, partial fills, per-side PnL, walk-forward evaluation

## Architecture

```
kucoin_bot/
├── __main__.py              # Entry point (LIVE / PAPER / SHADOW / BACKTEST modes)
├── config.py                # Configuration (env vars only); ShortConfig added
├── models.py                # SQLAlchemy models (orders, trades, signals, PnL, transfers)
├── api/
│   ├── client.py            # KuCoin REST client (HMAC auth, rate limiting, typed errors, futures parity)
│   └── websocket.py         # KuCoin WebSocket client
├── services/
│   ├── market_data.py       # Market universe discovery & filtering (spot + futures kline routing)
│   ├── signal_engine.py     # Multi-timeframe feature computation & regime detection
│   ├── risk_manager.py      # Position sizing, leverage, circuit breaker, squeeze check
│   ├── cost_model.py        # Unified cost model: fees, slippage, funding (directional), borrow, EV gate
│   ├── side_selector.py     # LONG/SHORT/FLAT decision + squeeze filter + crowded-short filter
│   ├── strategy_monitor.py  # Rolling per-module PnL tracking; auto-disables losers
│   ├── portfolio.py         # Portfolio allocation & internal transfers (DB-persisted)
│   └── execution.py         # Order lifecycle polling, futures-aware sizing, reduceOnly support
├── strategies/
│   ├── base.py              # Strategy interface
│   ├── trend.py             # Trend following (breakout + trailing exit; long and short)
│   ├── mean_reversion.py    # Bollinger band reversion (long and short)
│   ├── volatility_breakout.py  # Fast momentum with strict stops
│   ├── scalping.py          # Liquidity-aware scalping
│   ├── hedge.py             # Spot/futures hedging
│   └── risk_off.py          # Capital preservation
├── backtest/
│   └── engine.py            # Walk-forward backtester; per-side stats, cost breakdown
└── reporting/
    ├── cli.py               # CLI dashboard & performance export
    └── metrics.py           # Prometheus-style metrics (counters, gauges, histograms)
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/loureed691/50-10000.git
cd 50-10000
pip install -e ".[dev]"
```

### 2. Configure

Copy the example env and set your API credentials:

```bash
cp .env.example .env
# Edit .env with your KuCoin API key, secret, and passphrase
```

### 3. Run Backtest (default)

```bash
python -m kucoin_bot
```

### 4. Run Live Trading

```bash
BOT_MODE=LIVE LIVE_TRADING=true python -m kucoin_bot
```

### 5. Docker

```bash
docker-compose up --build
```

For production with Postgres + Redis:

```bash
docker-compose --profile production up --build
```

## Modes

| Mode | Description |
|---|---|
| **BACKTEST** | Historical simulation on kline data (default, safe) |
| **SHADOW** | `BOT_MODE=SHADOW` – signals computed, no orders placed |
| **PAPER** | `BOT_MODE=PAPER` – simulated fills with live market data, no real orders |
| **LIVE** | `BOT_MODE=LIVE LIVE_TRADING=true` – real orders placed on KuCoin |

**Safe defaults:** `BOT_MODE=BACKTEST`, `LIVE_TRADING=false`, `KILL_SWITCH=false`.
LIVE mode requires both `BOT_MODE=LIVE` AND `LIVE_TRADING=true` to be set.

## Configuration

All configuration is via environment variables (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `KUCOIN_API_KEY` | (required) | API key |
| `KUCOIN_API_SECRET` | (required) | API secret |
| `KUCOIN_API_PASSPHRASE` | (required) | API passphrase |
| `BOT_MODE` | `BACKTEST` | `LIVE`, `PAPER`, `SHADOW`, or `BACKTEST` |
| `LIVE_TRADING` | `false` | Second gate for LIVE mode |
| `KILL_SWITCH` | `false` | Set to `true` to stop all trading immediately |
| `ALLOW_INTERNAL_TRANSFERS` | `true` | Allow Funding↔Trading↔Futures transfers |
| `MAX_DAILY_LOSS_PCT` | `3.0` | Circuit breaker: max daily loss % |
| `MAX_DRAWDOWN_PCT` | `10.0` | Circuit breaker: max drawdown % |
| `MAX_TOTAL_EXPOSURE_PCT` | `80.0` | Max portfolio exposure % |
| `MAX_LEVERAGE` | `3.0` | Max allowed leverage |
| `MAX_PER_POSITION_RISK_PCT` | `2.0` | Max risk per position % |
| `MIN_EV_BPS` | `15.0` | Min expected value above round-trip cost (basis points) |
| `COOLDOWN_BARS` | `3` | Minimum candles between entries per symbol |
| `KLINE_TYPE` | `15min` | Candle interval for signals (`1min`, `5min`, `15min`, `1hour`, etc.) |
| `DB_TYPE` | `sqlite` | `sqlite` or `postgres` |

### Short Trading Variables

| Variable | Default | Description |
|---|---|---|
| `ALLOW_SHORTS` | `true` | Master toggle for short trading |
| `SHORT_PREFER_FUTURES` | `true` | Prefer futures over margin for shorts |
| `REQUIRE_FUTURES_FOR_SHORT` | `true` | Block shorts on spot-only instruments |
| `FUNDING_RATE_PER_8H` | `0.0001` | Expected funding rate per 8h period (0.01%) |
| `BORROW_RATE_PER_HOUR` | `0.00003` | Margin borrow interest per hour (0.003%) |
| `EXPECTED_HOLDING_HOURS` | `4.0` | Expected holding period used in EV gate |

## Security Guidance

- **Use trade-only API keys** – do NOT enable withdrawal permissions
- API secrets are **never logged** – loaded from env vars only
- Internal transfers require explicit `ALLOW_INTERNAL_TRANSFERS=true`
- Docker runs as non-root user
- All transfer routes are allow-listed with idempotency keys
- Futures exits always use `reduceOnly=True` to prevent unintentional position flips

## Risk Controls

- **Circuit breaker** – stops trading on daily loss > threshold, drawdown > threshold, or abnormal exposure
- **Short squeeze filter** – mandatory per trade; cannot be bypassed
- **Cost-aware EV gate** – rejects all trades where costs exceed edge
- **No martingale** – no doubling down behavior
- **Every strategy has stop/exit logic** – stops, take-profits, trailing stops, time stops
- **Dynamic leverage** – only applied when signal confidence is high AND volatility is low AND drawdown is low
- **Exposure limits** – per-position and total portfolio caps, correlated exposure check includes pending entries
- **reduceOnly exits** – futures exit orders always use `reduceOnly=True`
- **Kill switch** – `KILL_SWITCH=true` cancels all orders and flattens positions

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_execution.py -v
pytest tests/test_api_client.py -v

# Lint
ruff check kucoin_bot/ tests/

# Format check
black --check --line-length 120 kucoin_bot/ tests/

# Type checking
mypy kucoin_bot/ --ignore-missing-imports
```

## Compliance

- This software is provided as-is with no warranty
- Leverage trading carries significant risk of loss
- Past performance (including backtests) does not guarantee future results
- Users are responsible for compliance with local regulations

## Operational Runbook

### Stop-Loss Strategy

The bot uses a **software-based stop-loss** checked on every fast-loop iteration:

1. **Fast-loop price check**: Every `FAST_INTERVAL` seconds (default 30s), the bot evaluates all open positions against their `stop_price`.
2. **Flatten on trigger**: When price crosses the stop level, `flatten_position()` places a market order with `reduceOnly=True` for futures.
3. **Exchange-side stops (recommended for production)**: For latency-sensitive setups, place native stop-market orders on KuCoin using:
   - Futures: `POST /api/v1/st-orders` with `reduceOnly=true`, `stop="down"` (long) or `stop="up"` (short), `stopPriceType="TP"`.
   - Monitor stop order status via `GET /api/v1/st-orders` and reconcile in the slow loop.

### Kill Switch

Set `KILL_SWITCH=true` in the environment (or `.env`). The bot will:
1. Cancel all open orders (spot + futures) via `cancel_all()`.
2. Flatten all tracked positions using market orders with:
   - Full `MarketInfo` context (price increment, lot size, contract multiplier).
   - `reduceOnly=True` for all futures positions.
3. Exit the trading loop.

### Open-Order Reconciliation

Every slow-loop cycle (candle boundary), the bot:
1. Fetches all open spot and futures orders from the exchange.
2. Cancels any order older than 10 minutes (configurable via `_ORDER_STALE_SECONDS`).
3. Logs all reconciliation actions for audit.

### Metrics & Monitoring

The bot emits metrics via the `kucoin_bot.reporting.metrics.METRICS` singleton:

```python
from kucoin_bot.reporting.metrics import METRICS

# Export as JSON
print(METRICS.to_json())

# Export as Prometheus text format
print(METRICS.to_prometheus())
```

Key metrics:
- `orders_placed_total` – counter by symbol/side/status
- `order_latency_seconds` – histogram of order round-trip times
- `equity_usdt`, `daily_pnl_usdt`, `drawdown_pct` – gauges
- `circuit_breaker_active` – 0/1 gauge
- `open_positions`, `total_exposure_usdt` – gauges

Example Prometheus alert rules are in `alert_rules.yml`.

### Docker Production Deployment

```bash
# Set secrets in .env (never commit .env to Git!)
cp .env.example .env
# Edit .env: set POSTGRES_PASSWORD, API credentials, BOT_MODE, etc.

# Start with production profile (Postgres + Redis)
docker-compose --profile production up --build -d

# Check health
docker-compose ps
```

All services have health checks. Postgres password is read from `POSTGRES_PASSWORD` env var (required, no default).

## Validation Checklist

### PAPER Smoke Test
- [ ] Set `BOT_MODE=PAPER` and run for at least 2 slow cycles
- [ ] Verify signal computation logs appear (`[PAPER] Simulated ...`)
- [ ] Verify no real orders are placed (check KuCoin order history)
- [ ] Verify circuit breaker triggers on simulated losses
- [ ] Verify EV gate blocks marginal trades in logs

### SHADOW Mode
- [ ] Set `BOT_MODE=SHADOW` and run for 1+ hours
- [ ] Verify `[SHADOW] Would ...` logs appear for entry/exit decisions
- [ ] Verify no orders are placed and no positions are opened
- [ ] Check risk summary output at dashboard intervals

### LIVE Gate
- [ ] Confirm `LIVE_TRADING=true` is required alongside `BOT_MODE=LIVE`
- [ ] Verify startup reconciliation loads existing positions
- [ ] Verify `reduceOnly=True` is set on all futures exits
- [ ] Verify kill switch flattens all positions with market context
- [ ] Verify poll timeout does NOT mark orders as filled
- [ ] Verify open-order reconciliation cancels stale orders
- [ ] Monitor metrics export: equity, PnL, exposure, latency
- [ ] Confirm no hardcoded passwords in docker-compose or code
