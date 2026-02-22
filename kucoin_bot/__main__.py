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
from kucoin_bot.services.risk_manager import RiskManager
from kucoin_bot.services.portfolio import PortfolioManager
from kucoin_bot.services.execution import ExecutionEngine, OrderRequest
from kucoin_bot.strategies.base import BaseStrategy
from kucoin_bot.strategies.trend import TrendFollowing
from kucoin_bot.strategies.mean_reversion import MeanReversion
from kucoin_bot.strategies.volatility_breakout import VolatilityBreakout
from kucoin_bot.strategies.scalping import Scalping
from kucoin_bot.strategies.hedge import HedgeMode
from kucoin_bot.strategies.risk_off import RiskOff
from kucoin_bot.backtest.engine import BacktestEngine
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
    """Main live trading loop."""
    print(DISCLAIMER)

    if not cfg.api_key or not cfg.api_secret:
        logger.error("API credentials not set. Set KUCOIN_API_KEY, KUCOIN_API_SECRET, KUCOIN_API_PASSPHRASE.")
        return

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
        logger.info("Connected. USDT balance: %.2f", balance)
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

    logger.info("Starting live trading loop with %d strategies", len(strategies))
    cycle = 0

    while not stop_event.is_set():
        # Check kill switch
        if cfg.kill_switch or os.getenv("KILL_SWITCH", "false").lower() == "true":
            logger.critical("KILL SWITCH activated – cancelling all orders")
            await exec_engine.cancel_all()
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

                    # Execute
                    if decision.action.startswith("entry_"):
                        notional = risk_mgr.compute_position_size(
                            sym, market.last_price if market else 0, signals.volatility, signals
                        )
                        if notional > 0:
                            side = "buy" if "long" in decision.action else "sell"
                            await exec_engine.execute(
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
                    elif decision.action == "exit":
                        if pos:
                            side = "sell" if pos.side == "long" else "buy"
                            await exec_engine.execute(
                                OrderRequest(
                                    symbol=sym,
                                    side=side,
                                    notional=abs(pos.size * pos.current_price),
                                    order_type=decision.order_type,
                                    reason=decision.reason,
                                ),
                                market,
                            )

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
        asyncio.run(run_live(cfg))
    else:
        print(f"Unknown mode: {cfg.mode}. Use LIVE or BACKTEST.")
        sys.exit(1)


if __name__ == "__main__":
    main()
