"""Main entry point for the KuCoin Trading Bot."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys

from kucoin_bot.config import load_config, BotConfig
from kucoin_bot.api.client import KuCoinClient
from kucoin_bot.services.market_data import MarketDataService
from kucoin_bot.services.signal_engine import SignalEngine
from kucoin_bot.services.risk_manager import RiskManager, PositionInfo
from kucoin_bot.services.portfolio import PortfolioManager
from kucoin_bot.services.execution import ExecutionEngine, OrderRequest, OrderResult
from kucoin_bot.services.cost_model import CostModel
from kucoin_bot.services.side_selector import SideSelector
from kucoin_bot.services.strategy_monitor import StrategyMonitor
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
    from kucoin_bot.models import init_db, SignalSnapshot  # requires sqlalchemy
    print(DISCLAIMER)

    is_paper = cfg.is_paper
    is_shadow = cfg.is_shadow
    mode_label = cfg.mode.upper()

    if not cfg.api_key or not cfg.api_secret or not cfg.api_passphrase:
        raise RuntimeError(
            "API credentials missing. "
            "Set KUCOIN_API_KEY, KUCOIN_API_SECRET, and KUCOIN_API_PASSPHRASE "
            "before starting in LIVE, PAPER, or SHADOW mode."
        )

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
    cost_model = CostModel(
        taker_fee=DEFAULT_TAKER_FEE,
        slippage_bps=DEFAULT_SLIPPAGE_BPS,
        funding_rate_per_8h=cfg.short.funding_rate_per_8h,
        borrow_rate_per_hour=cfg.short.borrow_rate_per_hour,
        safety_buffer_bps=cfg.risk.min_ev_bps,
    )
    side_selector = SideSelector(
        allow_shorts=cfg.short.allow_shorts,
        require_futures_for_short=cfg.short.require_futures_for_short,
    )
    strategy_monitor = StrategyMonitor()
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

    # Cooldown tracking: symbol → last entry cycle index.
    # Each cycle is 60 seconds; default klines are 1-hour bars.
    # Scale cooldown_bars → cycles: 1 bar (1 hour) = 60 cycles.
    last_entry_cycle: dict[str, int] = {}
    cooldown_cycles = cfg.risk.cooldown_bars * 60  # convert bars to 60-second cycles

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

                    # Check strategy monitor – skip if module auto-disabled
                    if not strategy_monitor.is_enabled(alloc.strategy):
                        if cfg.live_diagnostic:
                            logger.debug(
                                "[MONITOR] Skipping %s: module '%s' auto-disabled",
                                sym, alloc.strategy,
                            )
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

                    # Side selector: validate proposed side (squeeze filter, feasibility)
                    if decision.action.startswith("entry_") and not pos:
                        mkt_type = market.market_type if market else "spot"
                        proposed_side = "long" if "long" in decision.action else "short"
                        side_dec = side_selector.select(
                            signals, market_type=mkt_type, proposed_side=proposed_side
                        )
                        if side_dec.side == "flat":
                            if cfg.live_diagnostic:
                                logger.debug(
                                    "[SIDE-SELECTOR] Blocked %s %s: %s",
                                    sym, proposed_side, side_dec.reason,
                                )
                            continue

                    # EV gate: cost-aware, shared with backtest
                    if decision.action.startswith("entry_"):
                        mkt_type = market.market_type if market else "spot"
                        is_futures = mkt_type == "futures"
                        proposed_side_ev = "long" if "long" in decision.action else "short"
                        is_margin_short = mkt_type == "margin" and proposed_side_ev == "short"
                        costs = cost_model.estimate(
                            order_type="taker",
                            holding_hours=cfg.short.expected_holding_hours,
                            is_futures=is_futures,
                            is_margin_short=is_margin_short,
                            live_funding_rate=signals.funding_rate if signals.funding_rate != 0 else None,
                            position_side=proposed_side_ev,
                        )
                        expected_bps = signals.volatility * 100.0 * signals.confidence
                        if not cost_model.ev_gate(expected_bps, costs):
                            if cfg.live_diagnostic:
                                logger.debug(
                                    "[EV-GATE] Blocked %s: expected %.1f bps < cost %.1f bps + buffer %.1f",
                                    sym, expected_bps, costs.total_bps, cfg.risk.min_ev_bps,
                                )
                            continue

                    # Cooldown: enforce minimum bars between entries
                    if decision.action.startswith("entry_") and not pos:
                        last_cycle = last_entry_cycle.get(sym, 0)
                        if cycle - last_cycle < cooldown_cycles:
                            continue

                    # Compute notional early so it can be included in the correlated
                    # exposure check below (pending entry not yet in risk_mgr.positions).
                    if decision.action.startswith("entry_") and not pos:
                        notional = risk_mgr.compute_position_size(
                            sym, market.last_price if market else 0, signals.volatility, signals
                        )
                    else:
                        notional = 0.0

                    # Correlated exposure check: group symbols by base asset.
                    # Prospective notional is included to prevent same-cycle bypass.
                    if decision.action.startswith("entry_") and not pos:
                        if notional <= 0:
                            continue
                        base = market.base if market else sym.split("-")[0]
                        correlated = [s for s in risk_mgr.positions if s.startswith(base + "-")]
                        if sym not in correlated:
                            correlated.append(sym)
                        if risk_mgr.check_correlated_exposure(
                            correlated, prospective_notional=notional * alloc.weight
                        ):
                            logger.warning("Correlated exposure limit reached for base %s, skipping %s", base, sym)
                            continue

                    # Execute
                    if decision.action.startswith("entry_"):
                        if notional > 0:
                            side = "buy" if "long" in decision.action else "sell"
                            trade_notional = notional * alloc.weight

                            if is_shadow:
                                logger.info(
                                    "[SHADOW] Would %s %s notional=%.2f @ %.4f reason=%s",
                                    side, sym, trade_notional,
                                    market.last_price if market else 0, decision.reason,
                                )
                            elif is_paper:
                                # Simulate fill: use last price with taker slippage and taker fee.
                                # No real API call is made – paper mode only.
                                last_px = market.last_price if market else 0
                                slip_dir = 1 if side == "buy" else -1
                                fill_px = last_px * (1 + slip_dir * DEFAULT_SLIPPAGE_BPS / 10_000) if last_px > 0 else 0
                                if fill_px > 0:
                                    filled_qty = trade_notional / fill_px
                                    paper_fee = filled_qty * fill_px * DEFAULT_TAKER_FEE
                                    risk_mgr.update_equity(risk_mgr.current_equity - paper_fee)
                                    result = OrderResult(
                                        success=True,
                                        order_id=f"paper-{sym}-{cycle}",
                                        filled_qty=filled_qty,
                                        avg_price=fill_px,
                                        message="paper_fill",
                                    )
                                    logger.info(
                                        "[PAPER] Simulated %s %s qty=%.6f @ %.4f fee=%.4f",
                                        side, sym, filled_qty, fill_px, paper_fee,
                                    )
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
                            else:
                                result = await exec_engine.execute(
                                    OrderRequest(
                                        symbol=sym,
                                        side=side,
                                        notional=trade_notional,
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
                            is_futures_exit = market.market_type == "futures" if market else False

                            if is_shadow:
                                logger.info("[SHADOW] Would exit %s side=%s", sym, pos.side)
                            elif is_paper:
                                # Simulate exit fill with taker slippage and taker fee.
                                # No real API call is made – paper mode only.
                                last_px = market.last_price if market else 0
                                slip_dir = 1 if side == "buy" else -1
                                fill_px = last_px * (1 + slip_dir * DEFAULT_SLIPPAGE_BPS / 10_000) if last_px > 0 else pos.entry_price
                                paper_fee = pos.size * fill_px * DEFAULT_TAKER_FEE
                                pnl = (fill_px - pos.entry_price) * pos.size - paper_fee
                                if pos.side == "short":
                                    pnl = -((fill_px - pos.entry_price) * pos.size) - paper_fee
                                risk_mgr.record_pnl(pnl)
                                risk_mgr.update_position(sym, PositionInfo(
                                    symbol=sym, side=pos.side, size=0,
                                ))
                                strategy_monitor.record_trade(alloc.strategy, pnl, paper_fee)
                                logger.info("[PAPER] Simulated exit %s pnl=%.4f fee=%.4f", sym, pnl, paper_fee)
                            else:
                                result = await exec_engine.execute(
                                    OrderRequest(
                                        symbol=sym,
                                        side=side,
                                        notional=abs(pos.size * pos.current_price),
                                        order_type=decision.order_type,
                                        reason=decision.reason,
                                        reduce_only=is_futures_exit,
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
                                    # Estimate exit fee from fill to track net expectancy
                                    trade_cost = (
                                        pos.size * result.avg_price * DEFAULT_TAKER_FEE
                                        if result.avg_price > 0 else 0.0
                                    )
                                    strategy_monitor.record_trade(alloc.strategy, pnl, trade_cost)

                except Exception:
                    logger.error("Error processing %s", sym, exc_info=True)

            # Dashboard
            if cycle % 10 == 0:
                print_dashboard(risk_mgr, active_strategies)
                if cfg.live_diagnostic:
                    logger.info("[MONITOR] %s", strategy_monitor.get_status())

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
        try:
            asyncio.run(run_live(cfg))
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            logger.debug("Fatal error in LIVE mode", exc_info=True)
            sys.exit(1)
    elif cfg.mode.upper() in ("PAPER", "SHADOW"):
        try:
            asyncio.run(run_live(cfg))
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            logger.debug("Fatal error in %s mode", cfg.mode, exc_info=True)
            sys.exit(1)
    else:
        print(f"Unknown mode: {cfg.mode}. Use LIVE, PAPER, SHADOW, or BACKTEST.")
        sys.exit(1)


if __name__ == "__main__":
    main()
