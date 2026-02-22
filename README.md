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
- **Portfolio-level risk management** – drawdown limits, daily loss caps, exposure limits, circuit breaker, correlated exposure limits
- **Dynamic leverage** – only when justified by signal confidence and risk budget
- **Smart execution** – limit/post-only preference, slippage controls, partial fill handling; `reduceOnly` for futures exits
- **Strategy monitor** – auto-disables modules with persistently negative net expectancy after costs
- **Internal transfers** – automated Funding ↔ Trading ↔ Futures account transfers (opt-in)
- **Kill switch** – set `KILL_SWITCH=true` to immediately stop all trading
- **Full audit trail** – every signal, decision, order, and fill is logged to the database
- **Backtesting** – realistic fees, slippage, funding, borrow, partial fills, per-side PnL, walk-forward evaluation

## Architecture

```
kucoin_bot/
├── __main__.py              # Entry point (LIVE / PAPER / SHADOW / BACKTEST modes)
├── config.py                # Configuration (env vars + optional YAML); ShortConfig added
├── models.py                # SQLAlchemy models (orders, trades, signals, PnL)
├── api/
│   ├── client.py            # KuCoin REST client (HMAC auth, rate limiting, reduceOnly)
│   └── websocket.py         # KuCoin WebSocket client
├── services/
│   ├── market_data.py       # Market universe discovery & filtering (spot + futures)
│   ├── signal_engine.py     # Multi-timeframe feature computation & regime detection
│   ├── risk_manager.py      # Position sizing, leverage, circuit breaker, squeeze check
│   ├── cost_model.py        # ★ Unified cost model: fees, slippage, funding, borrow, EV gate
│   ├── side_selector.py     # ★ LONG/SHORT/FLAT decision + squeeze filter + crowded-short filter
│   ├── strategy_monitor.py  # ★ Rolling per-module PnL tracking; auto-disables losers
│   ├── portfolio.py         # Portfolio allocation & internal transfers
│   └── execution.py         # Smart order routing, slippage controls, reduceOnly support
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
    └── cli.py               # CLI dashboard & performance export
```

★ = new module added for short trading

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
| `BOT_MODE` | `BACKTEST` | `LIVE`, `PAPER`, `SHADOW`, or `BACKTEST` |
| `KILL_SWITCH` | `false` | Set to `true` to stop all trading immediately |
| `ALLOW_INTERNAL_TRANSFERS` | `false` | Allow Funding↔Trading↔Futures transfers |
| `MAX_DAILY_LOSS_PCT` | `3.0` | Circuit breaker: max daily loss % |
| `MAX_DRAWDOWN_PCT` | `10.0` | Circuit breaker: max drawdown % |
| `MAX_TOTAL_EXPOSURE_PCT` | `80.0` | Max portfolio exposure % |
| `MAX_LEVERAGE` | `3.0` | Max allowed leverage |
| `MAX_PER_POSITION_RISK_PCT` | `2.0` | Max risk per position % |
| `MIN_EV_BPS` | `10.0` | Min expected value above round-trip cost (basis points) |
| `DB_TYPE` | `sqlite` | `sqlite` or `postgres` |

### Short Trading Variables

| Variable | Default | Description |
|---|---|---|
| `ALLOW_SHORTS` | `true` | Master toggle for short trading |
| `SHORT_PREFER_FUTURES` | `true` | Prefer futures over margin for shorts |
| `REQUIRE_FUTURES_FOR_SHORT` | `true` | Block shorts on spot-only instruments |
| `FUNDING_RATE_PER_8H` | `0.0001` | Expected funding rate per 8h period (0.01%) |
| `BORROW_RATE_PER_HOUR` | `0.00003` | Margin borrow interest per hour (0.003%) |
| `EXPECTED_HOLDING_HOURS` | `24.0` | Expected holding period used in EV gate |

## Short Trading

### Instrument Selection

1. **USDT-margined perpetual futures** (preferred) – no borrow mechanics, clean PnL, funding every 8h.
2. **Margin borrow** (fallback) – borrow the asset, sell it, buy back to close; interest accrues hourly.
3. **Spot** – no short trading; positions are long only.

The `SideSelector` service enforces these rules automatically.

### Short Squeeze Protection (Mandatory)

New short entries are blocked when **any** of the following conditions are met:

| Condition | Threshold | Why |
|---|---|---|
| Volatility spike | `volatility > 0.6` (normalized ATR) | Wide candle range = covering pressure |
| Momentum + volume burst | `momentum > 0.5 AND volume_anomaly > 2.5` | Aggressive buying / forced covering |
| Crowded-short funding | `funding_rate < -0.0003` (per 8h) | Deeply negative = shorts crowded, squeeze risk elevated |

### Cost-Aware EV Gate

Every trade (long or short) is gated by expected value after full costs:

```
expected_edge_bps = volatility * 100 * confidence

round_trip_cost_bps = (fee_rate * 2 * 10_000)   # entry + exit fees
                    + (slippage_bps * 2)          # entry + exit slippage
                    + funding_bps                 # futures funding (for expected holding)
                    + borrow_bps                  # margin borrow (for expected holding)

gate_passes = expected_edge_bps > round_trip_cost_bps + safety_buffer_bps (MIN_EV_BPS)
```

Identical logic runs in both backtest and live.

### Strategy Monitor (Auto-Adaptation)

The `StrategyMonitor` tracks rolling net PnL (after costs) per strategy module over the last 20 trades. If a module's rolling net expectancy turns negative after at least 5 trades, it is **automatically disabled**. Re-enable manually with `monitor.enable(module_name)`.

## Risk Controls

- **Circuit breaker** – stops trading on daily loss > threshold, drawdown > threshold, or abnormal volatility
- **Short squeeze filter** – mandatory per trade; cannot be bypassed
- **Cost-aware EV gate** – rejects all trades where costs exceed edge
- **No martingale** – no doubling down behavior
- **Every strategy has stop/exit logic** – stops, take-profits, trailing stops, time stops
- **Dynamic leverage** – only applied when signal confidence is high AND volatility is low AND drawdown is low
- **Exposure limits** – per-position and total portfolio caps
- **reduceOnly exits** – futures exit orders always use `reduceOnly=True`
- **Kill switch** – `KILL_SWITCH=true` cancels all orders and flattens positions

## Backtest Output

The backtest engine now reports:
- Combined, long-only, and short-only PnL
- Full cost breakdown: `fees / slippage / funding / borrow`
- Walk-forward out-of-sample evaluation

Example summary line:
```
Backtest: 12 trades (L:8 S:4) | Return: 2.31% | Max DD: 3.14% | Sharpe: 1.42 |
Win Rate: 58.3% | Fees: 4.23 | Expectancy: 1.9250 | Long PnL: 18.5 | Short PnL: 4.6 |
Costs[fee=4.23 slip=1.10 fund=0.05 borr=0.00]
```

## Live Rollout Plan

| Phase | Description |
|---|---|
| **SHADOW** | `BOT_MODE=SHADOW` – signals computed, no orders placed |
| **PAPER** | `BOT_MODE=PAPER` – simulated fills with live data, no real orders |
| **LIVE small** | `BOT_MODE=LIVE LIVE_TRADING=true` with small position caps |
| **LIVE scaled** | Increase `MAX_PER_POSITION_RISK_PCT` only after positive PAPER/LIVE-small results |

## Security Notes

- API secrets are **never logged** – loaded from env vars only
- Internal transfers require explicit `ALLOW_INTERNAL_TRANSFERS=true` + `INTERNAL_TRANSFERS_ACK`
- Docker runs as non-root user
- All transfer routes are allow-listed with idempotency keys
- Futures exits always use `reduceOnly=True` to prevent unintentional position flips

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run short-trading specific tests
pytest tests/test_short_trading.py -v

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
