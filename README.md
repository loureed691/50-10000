# KuCoin Auto Trading Bot

A fully automatic, multi-strategy KuCoin API trading bot built in Python.

> **⚠️ WARNING:** This bot trades real money. Leverage amplifies losses.
> Profitability is **NOT guaranteed**. Use at your own risk.

## Features

- **Automatic market discovery** – finds all USDT-quoted pairs, filters by liquidity/spread
- **Multi-source signal engine** – momentum, trend strength, mean reversion, volatility, volume anomalies, orderbook imbalance
- **Regime classification** – trending, ranging, high-volatility, low-liquidity
- **6 strategy modules** – trend following, mean reversion, volatility breakout, scalping, hedging, risk-off
- **Portfolio-level risk management** – drawdown limits, daily loss caps, exposure limits, circuit breaker
- **Dynamic leverage** – only when justified by signal confidence and risk budget
- **Smart execution** – limit/post-only preference, slippage controls, partial fill handling
- **Internal transfers** – automated Funding ↔ Trading ↔ Futures account transfers (opt-in)
- **Kill switch** – set `KILL_SWITCH=true` to immediately stop all trading
- **Full audit trail** – every signal, decision, order, and fill is logged to the database
- **Backtesting** – realistic fees, slippage, partial fills, walk-forward evaluation

## Architecture

```
kucoin_bot/
├── __main__.py              # Entry point (LIVE / BACKTEST modes)
├── config.py                # Configuration (env vars + optional YAML)
├── models.py                # SQLAlchemy models (orders, trades, signals, PnL)
├── api/
│   ├── client.py            # KuCoin REST client (HMAC auth, rate limiting)
│   └── websocket.py         # KuCoin WebSocket client
├── services/
│   ├── market_data.py       # Market universe discovery & filtering
│   ├── signal_engine.py     # Multi-timeframe feature computation & regime detection
│   ├── risk_manager.py      # Position sizing, leverage, circuit breaker
│   ├── portfolio.py         # Portfolio allocation & internal transfers
│   └── execution.py         # Smart order routing & slippage controls
├── strategies/
│   ├── base.py              # Strategy interface
│   ├── trend.py             # Trend following (breakout + trailing exit)
│   ├── mean_reversion.py    # Bollinger band reversion
│   ├── volatility_breakout.py  # Fast momentum with strict stops
│   ├── scalping.py          # Liquidity-aware scalping
│   ├── hedge.py             # Spot/futures hedging
│   └── risk_off.py          # Capital preservation
├── backtest/
│   └── engine.py            # Walk-forward backtester
└── reporting/
    └── cli.py               # CLI dashboard & performance export
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
BOT_MODE=LIVE python -m kucoin_bot
```

### 5. Docker

```bash
docker-compose up --build
```

For production with Postgres + Redis:

```bash
docker-compose --profile production up --build
```

## Configuration

All configuration is via environment variables (see `.env.example`). An optional `config.yaml` is auto-generated on first run for overrides.

| Variable | Default | Description |
|---|---|---|
| `KUCOIN_API_KEY` | (required) | API key |
| `KUCOIN_API_SECRET` | (required) | API secret |
| `KUCOIN_API_PASSPHRASE` | (required) | API passphrase |
| `BOT_MODE` | `BACKTEST` | `LIVE` or `BACKTEST` |
| `KILL_SWITCH` | `false` | Set to `true` to stop all trading immediately |
| `ALLOW_INTERNAL_TRANSFERS` | `false` | Allow Funding↔Trading↔Futures transfers |
| `MAX_DAILY_LOSS_PCT` | `3.0` | Circuit breaker: max daily loss % |
| `MAX_DRAWDOWN_PCT` | `10.0` | Circuit breaker: max drawdown % |
| `MAX_TOTAL_EXPOSURE_PCT` | `80.0` | Max portfolio exposure % |
| `MAX_LEVERAGE` | `3.0` | Max allowed leverage |
| `MAX_PER_POSITION_RISK_PCT` | `2.0` | Max risk per position % |
| `DB_TYPE` | `sqlite` | `sqlite` or `postgres` |

## Risk Controls

- **Circuit breaker** – automatically stops trading on daily loss > threshold, drawdown > threshold, or abnormal volatility
- **No martingale** – no doubling down behavior
- **Every strategy has stop/exit logic** – stops, take-profits, trailing stops, time stops
- **Dynamic leverage** – only applied when signal confidence is high AND volatility is low AND drawdown is low
- **Exposure limits** – per-position and total portfolio caps
- **Kill switch** – `KILL_SWITCH=true` cancels all orders and flattens positions

## Security Notes

- API secrets are **never logged** – loaded from env vars only
- Internal transfers require explicit `ALLOW_INTERNAL_TRANSFERS=true`
- Docker runs as non-root user
- All transfer routes are allow-listed with idempotency keys

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_risk_manager.py -v

# Type checking
mypy kucoin_bot/ --ignore-missing-imports
```

## Compliance

- This software is provided as-is with no warranty
- Leverage trading carries significant risk of loss
- Past performance (including backtests) does not guarantee future results
- Users are responsible for compliance with local regulations