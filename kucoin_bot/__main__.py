"""Main entry point for the KuCoin Trading Bot."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys

from kucoin_bot.config import load_config, BotConfig
from kucoin_bot.models import init_db, SignalSnapshot
from kucoin_bot.api.client import KuCoinClient
from kucoin_bot.services.market_data import MarketDataService
from kucoin_bot.services.signal_engine import SignalEngine
from kucoin_bot.services.risk_manager import RiskManager, PositionInfo
from kucoin_bot.services.portfolio import PortfolioManager
from kucoin_bot.services.execution import ExecutionEngine, OrderRequest
from kucoin_bot.strategies.base import BaseStrategy
from kucoin_bot.strategies.trend import TrendFollowing
from kucoin_bot.strategies.mean_reversion import MeanReversion
from kucoin_bot.strategies.volatility_breakout import VolatilityBreakout
from kucoin_bot.strategies.scalping import Scalping
from kucoin_bot.strategies.hedge import HedgeMode
from kucoin_bot.strategies.risk_off import RiskOff
from kucoin_bot.backtest.engine import BacktestEngine, DEFAULT_TAKER_FEE, DEFAULT_SLIPPAGE_BPS
from kucoin_bot.reporting.cli import print_dashboard

logger = logging.getLogger(__name__)

# ⚠️ WARNING: This bot trades real money. Use at your own risk.
# Profitability is NOT guaranteed. Leverage amplifies losses.
# The authors are not responsible for any financial losses.

DISCLAIMER = """
╔══════════════════════════════════════════════════════════════╗
║  WARNING: This bot can trade with real money and leverage.  ║
║  • Profitability is NOT guaranteed.                         ║
║  • Leverage amplifies both gains AND losses.                ║
║  • You may lose your entire investment.                     ║
║  • Use at your own risk.                                    ║
╚══════════════════════════════════════════════════════════════╝
"""

# Strategy registry
STRATEGY_MAP: dict[str, type[BaseStrategy]] = {
    "trend_following": TrendFollowing,
    "mean_reversion": MeanReversion,
    "volatility_breakout": VolatilityBreakout,
    "scalping": Scalping,
    "hedge": HedgeMode,
    "risk_off": RiskOff,
}


def _build_strategies() -> list[BaseStrategy]:
    return [cls() for cls in STRATEGY_MAP.values()]


async def run_live(cfg: BotConfig) -> None:
    """Main live trading loop (also handles PAPER and SHADOW modes)."""
    print(DISCLAIMER)

    is_paper = cfg.is_paper
    is_shadow = cfg.is_shadow
    mode_label = cfg.mode.upper()

    if not cfg.api_key or not cfg.api_secret:
        logger.error("API credentials not set. Set KUCOIN_API_KEY, KUCOIN_API_SECRET, KUCOIN_API_PASSPHRASE.")
        return

    if is_shadow:
        logger.info("SHADOW mode: signals will be computed and logged; NO orders will be placed.")
    elif is_paper:
        logger.info("PAPER mode: simulated fills using live market data; NO real orders.")
    else:
        logger.info("LIVE mode: real orders will be placed.")

    # Initialize
    db_session_factory = init_db(cfg.db_url)
    client = KuCoinClient(cfg.api_key, cfg.api_secret, cfg.api_passphrase, cfg.rest_url, cfg.futures_rest_url)
    await client.start()

    market_data = MarketDataService(client=client)
    signal_engine = SignalEngine()
    risk_mgr = RiskManager(config=cfg.risk)
    portfolio_mgr = PortfolioManager(client=client, risk_mgr=risk_mgr, allow_transfers=cfg.allow_internal_transfers)
    exec_engine = ExecutionEngine(client=client, risk_mgr=risk_mgr)
    strategies = _build_strategies()

    # Validate API connectivity
    try:
        balance = await client.get_account_balance("USDT")
        risk_mgr.update_equity(balance)
        logger.info("[%s] Connected. USDT balance: %.2f", mode_label, balance)
    except Exception:
        logger.error("Failed to connect to KuCoin API", exc_info=True)
        await client.close()
        return

    # Discover markets
    await market_data.refresh_universe()

    # Kill switch handler
    stop_event = asyncio.Event()

    def _shutdown(*_: object) -> None:
        logger.warning("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_event_loop().add_signal_handler(sig, _shutdown)
        except NotImplementedError:
            pass

    # Cooldown tracking: symbol → last entry cycle index
    last_entry_cycle: dict[str, int] = {}
    cooldown_cycles = cfg.risk.cooldown_bars  # 1 cycle ≈ 1 bar

    logger.info(
        "Starting %s loop with %d strategies | EV gate: %.0f bps | cooldown: %d bars",
        mode_label, len(strategies), cfg.risk.min_ev_bps, cooldown_cycles,
    )
    cycle = 0

    while not stop_event.is_set():
        # Check kill switch
        if cfg.kill_switch or os.getenv("KILL_SWITCH", "false").lower() == "true":
            logger.critical("KILL SWITCH activated – cancelling all orders and flattening positions")
            if not is_shadow and not is_paper:
                await exec_engine.cancel_all()
            for sym, pos in list(risk_mgr.positions.items()):
                if pos.size > 0:
                    close_price = pos.current_price or pos.entry_price
                    if close_price <= 0:
                        logger.error("No price for %s, cannot flatten", sym)
                        continue
                    try:
                        if not is_shadow and not is_paper:
                            await exec_engine.flatten_position(
                                sym, pos.size, close_price, pos.side,
                            )
                        risk_mgr.update_position(sym, PositionInfo(symbol=sym, side=pos.side, size=0))
                    except Exception:
                        logger.error("Failed to flatten %s", sym, exc_info=True)
            break

        cycle += 1
        try:
            # Refresh universe periodically
            if cycle % 60 == 1:
                await market_data.refresh_universe()
                balance = await client.get_account_balance("USDT")
                risk_mgr.update_equity(balance)

            # Process each pair
            active_strategies: dict[str, str] = {}
            for sym in market_data.get_symbols()[:20]:  # Top 20 pairs
                try:
                    klines = await market_data.get_klines(sym)
                    if len(klines) < signal_engine.lookback:
                        continue

                    signals = signal_engine.compute(sym, klines)
                    market = market_data.get_info(sym)

                    if cfg.live_diagnostic:
                        logger.debug("[DIAG] %s signals: %s", sym, signals.to_dict())

                    # Portfolio allocation
                    allocs = portfolio_mgr.compute_allocations(
                        {sym: signals}, [sym]
                    )
                    alloc = allocs.get(sym)
                    if not alloc or alloc.weight <= 0:
                        continue

                    active_strategies[sym] = alloc.strategy

                    # Get matching strategy
                    strat_cls = STRATEGY_MAP.get(alloc.strategy)
                    if not strat_cls:
                        continue
                    strat = strat_cls()

                    if not strat.preconditions_met(signals):
                        continue

                    pos = risk_mgr.positions.get(sym)
                    decision = strat.evaluate(
                        signals,
                        pos.side if pos else None,
                        pos.entry_price if pos else None,
                        market.last_price if market else 0,
                    )

                    # Log decision
                    with db_session_factory() as session:
                        session.add(SignalSnapshot(
                            symbol=sym,
                            regime=signals.regime.value,
                            strategy_name=alloc.strategy,
                            signal_data=json.dumps(signals.to_dict()),
                            decision=decision.action,
                            reason=decision.reason,
                        ))
                        session.commit()

                    # EV gate: same gate applied in both backtest and live
                    if decision.action.startswith("entry_"):
                        cost_bps = (DEFAULT_TAKER_FEE * 2) * 10_000 + DEFAULT_SLIPPAGE_BPS * 2 + cfg.risk.min_ev_bps
                        expected_bps = signals.volatility * 100.0 * signals.confidence
                        if expected_bps < cost_bps:
                            if cfg.live_diagnostic:
                                logger.debug(
                                    "[EV-GATE] Blocked %s: expected %.1f bps < cost %.1f bps",
                                    sym, expected_bps, cost_bps,
                                )
                            continue

                    # Cooldown: enforce minimum bars between entries
                    if decision.action.startswith("entry_") and not pos:
                        last_cycle = last_entry_cycle.get(sym, 0)
                        if cycle - last_cycle < cooldown_cycles:
                            continue

                    # Correlated exposure check: group symbols by base asset
                    if decision.action.startswith("entry_") and not pos:
                        base = market.base if market else sym.split("-")[0]
                        correlated = [s for s in risk_mgr.positions if s.startswith(base + "-")]
                        correlated.append(sym)
                        if risk_mgr.check_correlated_exposure(correlated):
                            logger.warning("Correlated exposure limit reached for base %s, skipping %s", base, sym)
                            continue

                    # Execute
                    if decision.action.startswith("entry_"):
                        notional = risk_mgr.compute_position_size(
                            sym, market.last_price if market else 0, signals.volatility, signals
                        )
                        if notional > 0:
                            side = "buy" if "long" in decision.action else "sell"

                            if is_shadow:
                                logger.info(
                                    "[SHADOW] Would %s %s notional=%.2f @ %.4f reason=%s",
                                    side, sym, notional * alloc.weight,
                                    market.last_price if market else 0, decision.reason,
                                )
                            else:
                                result = await exec_engine.execute(
                                    OrderRequest(
                                        symbol=sym,
                                        side=side,
                                        notional=notional * alloc.weight,
                                        order_type=decision.order_type,
                                        price=market.last_price if market else None,
                                        leverage=alloc.max_leverage,
                                        reason=decision.reason,
                                    ),
                                    market,
                                )
                                if result.success and result.filled_qty > 0:
                                    pos_side = "long" if "long" in decision.action else "short"
                                    risk_mgr.update_position(sym, PositionInfo(
                                        symbol=sym,
                                        side=pos_side,
                                        size=result.filled_qty,
                                        entry_price=result.avg_price,
                                        current_price=result.avg_price,
                                        leverage=alloc.max_leverage,
                                    ))
                                    last_entry_cycle[sym] = cycle

                    elif decision.action == "exit":
                        if pos:
                            side = "sell" if pos.side == "long" else "buy"

                            if is_shadow:
                                logger.info("[SHADOW] Would exit %s side=%s", sym, pos.side)
                            else:
                                result = await exec_engine.execute(
                                    OrderRequest(
                                        symbol=sym,
                                        side=side,
                                        notional=abs(pos.size * pos.current_price),
                                        order_type=decision.order_type,
                                        reason=decision.reason,
                                    ),
                                    market,
                                )
                                if result.success:
                                    pnl = (result.avg_price - pos.entry_price) * pos.size
                                    if pos.side == "short":
                                        pnl = -pnl
                                    risk_mgr.record_pnl(pnl)
                                    risk_mgr.update_position(sym, PositionInfo(
                                        symbol=sym, side=pos.side, size=0,
                                    ))

                except Exception:
                    logger.error("Error processing %s", sym, exc_info=True)

            # Dashboard
            if cycle % 10 == 0:
                print_dashboard(risk_mgr, active_strategies)

        except Exception:
            logger.error("Trading loop error", exc_info=True)

        await asyncio.sleep(60)  # 1 min cycle

    # Cleanup
    logger.info("Shutting down...")
    await client.close()


def run_backtest(cfg: BotConfig) -> None:
    """Run backtesting mode."""
    print(DISCLAIMER)
    logger.info("Starting backtest mode")

    strategies = _build_strategies()
    engine = BacktestEngine(
        strategies=strategies,
        risk_config=cfg.risk,
    )

    # Generate sample data for demo
    import numpy as np
    rng = np.random.RandomState(42)
    n = 500
    price = 30000.0
    klines = []
    for i in range(n):
        ret = rng.normal(0, 0.02)
        o = price
        c = price * (1 + ret)
        h = max(o, c) * (1 + abs(rng.normal(0, 0.005)))
        l_ = min(o, c) * (1 - abs(rng.normal(0, 0.005)))
        vol = rng.uniform(100, 1000)
        klines.append([i * 3600, str(o), str(c), str(h), str(l_), str(vol), str(vol * c)])
        price = c

    result = engine.run(klines, "BTC-USDT", initial_equity=10_000)
    print("\n" + result.summary())
    logger.info(result.summary())


def main() -> None:
    cfg = load_config()

    if cfg.mode.upper() == "BACKTEST":
        run_backtest(cfg)
    elif cfg.mode.upper() == "LIVE":
        if not cfg.live_trading:
            print(
                "ERROR: LIVE_TRADING=true is required to start in LIVE mode.\n"
                "Set the environment variable LIVE_TRADING=true to confirm you understand real money is at risk."
            )
            sys.exit(1)
        asyncio.run(run_live(cfg))
    elif cfg.mode.upper() in ("PAPER", "SHADOW"):
        asyncio.run(run_live(cfg))
    else:
        print(f"Unknown mode: {cfg.mode}. Use LIVE, PAPER, SHADOW, or BACKTEST.")
        sys.exit(1)


if __name__ == "__main__":
    main()
