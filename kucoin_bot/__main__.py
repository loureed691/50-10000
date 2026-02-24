"""Main entry point for the KuCoin Trading Bot."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time

from kucoin_bot.api.client import KuCoinClient
from kucoin_bot.backtest.engine import DEFAULT_SLIPPAGE_BPS, DEFAULT_TAKER_FEE, BacktestEngine
from kucoin_bot.config import BotConfig, load_config
from kucoin_bot.reporting.cli import print_dashboard
from kucoin_bot.reporting.metrics import METRICS
from kucoin_bot.services.cost_model import CostModel
from kucoin_bot.services.execution import ExecutionEngine, OrderRequest, OrderResult
from kucoin_bot.services.market_data import _KLINE_PERIOD_SECONDS, MarketDataService
from kucoin_bot.services.portfolio import AllocationTarget, PortfolioManager
from kucoin_bot.services.risk_manager import PositionInfo, RiskManager
from kucoin_bot.services.side_selector import SideSelector
from kucoin_bot.services.signal_engine import SignalEngine, SignalScores
from kucoin_bot.services.strategy_monitor import StrategyMonitor
from kucoin_bot.strategies.base import BaseStrategy
from kucoin_bot.strategies.hedge import HedgeMode
from kucoin_bot.strategies.mean_reversion import MeanReversion
from kucoin_bot.strategies.risk_off import RiskOff
from kucoin_bot.strategies.scalping import Scalping
from kucoin_bot.strategies.trend import TrendFollowing
from kucoin_bot.strategies.volatility_breakout import VolatilityBreakout

logger = logging.getLogger(__name__)

# ⚠️ WARNING: This bot trades real money. Use at your own risk.
# Profitability is NOT guaranteed. Leverage amplifies losses.
# The authors are not responsible for any financial losses.

# Cancel open orders older than this threshold during periodic reconciliation.
_ORDER_STALE_SECONDS = 600  # 10 minutes

# Order statuses that represent confirmed fills (safe for position updates).
_CONFIRMED_FILL_STATUSES = {"filled", "partially_filled"}

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


async def _compute_total_equity(
    client: KuCoinClient,
    market_data: "MarketDataService",
) -> float:
    """Return total portfolio equity in USDT by summing all asset balances.

    Non-USDT assets are converted to USDT using available market prices.
    Falls back to 0.0 if the accounts endpoint is unavailable.
    """
    try:
        accounts = await client.get_accounts("trade")
        total = 0.0
        for acc in accounts:
            currency = acc.get("currency", "")
            balance = float(acc.get("balance", 0) or 0)
            if balance <= 0:
                continue
            if currency == "USDT":
                total += balance
            else:
                # Convert asset to USDT via cached market price
                sym = f"{currency}-USDT"
                info = market_data.get_info(sym)
                price = info.last_price if info and info.last_price > 0 else 0.0
                if price <= 0:
                    # Fallback: fetch ticker directly
                    try:
                        ticker = await client.get_ticker(sym)
                        price = float(ticker.get("price", 0) or 0)
                    except Exception:
                        pass
                if price > 0:
                    total += balance * price
        return total
    except Exception:
        logger.error("Failed to compute total portfolio equity", exc_info=True)
        return 0.0


async def _reconcile_positions(
    client: KuCoinClient,
    risk_mgr: RiskManager,
    market_data: "MarketDataService",
    log: logging.Logger,
    mode_label: str,
) -> None:
    """Reconcile open positions from the exchange on startup.

    Fetches open futures positions and rebuilds risk manager state so the bot
    knows about positions that were opened in a previous session.
    """
    # Reconcile futures positions
    try:
        positions = await client.get_futures_positions()
        if isinstance(positions, list):
            for pos in positions:
                if not pos:
                    continue
                sym = pos.get("symbol", "")
                qty = float(pos.get("currentQty", 0) or 0)
                if qty == 0:
                    continue
                entry_price = float(pos.get("avgEntryPrice", 0) or 0)
                mark_price = float(pos.get("markPrice", 0) or 0)
                leverage = float(pos.get("realLeverage", 1) or 1)
                side = "long" if qty > 0 else "short"
                unrealized = float(pos.get("unrealisedPnl", 0) or 0)
                multiplier = float(pos.get("multiplier", 1) or 1)
                risk_mgr.update_position(
                    sym,
                    PositionInfo(
                        symbol=sym,
                        side=side,
                        size=abs(qty),
                        entry_price=entry_price,
                        current_price=mark_price if mark_price > 0 else entry_price,
                        leverage=leverage,
                        account_type="futures",
                        unrealized_pnl=unrealized,
                        contract_multiplier=abs(multiplier) if multiplier != 0 else 1.0,
                    ),
                )
                log.info(
                    "[%s] Reconciled futures position: %s %s qty=%.6f entry=%.4f",
                    mode_label,
                    side,
                    sym,
                    abs(qty),
                    entry_price,
                )
    except Exception:
        log.warning("Failed to reconcile futures positions", exc_info=True)

    # Reconcile open spot orders (log them, don't rebuild positions from orders)
    try:
        open_orders = await client.get_open_orders()
        if open_orders:
            log.info("[%s] Found %d open spot orders on startup", mode_label, len(open_orders))
            for o in open_orders:
                log.info(
                    "[%s]   Order %s: %s %s %s @ %s",
                    mode_label,
                    o.get("id", "?"),
                    o.get("side", "?"),
                    o.get("symbol", "?"),
                    o.get("size", "?"),
                    o.get("price", "?"),
                )
    except Exception:
        log.warning("Failed to fetch open spot orders", exc_info=True)

    # Reconcile open futures orders
    try:
        fut_orders = await client.get_futures_open_orders()
        if fut_orders:
            log.info("[%s] Found %d open futures orders on startup", mode_label, len(fut_orders))
    except Exception:
        log.warning("Failed to fetch open futures orders", exc_info=True)

    log.info("[%s] Reconciliation complete. Positions: %d", mode_label, len(risk_mgr.positions))


async def run_live(cfg: BotConfig) -> None:
    """Main live trading loop (also handles PAPER and SHADOW modes)."""
    from kucoin_bot.models import SignalSnapshot, init_db  # requires sqlalchemy
    from kucoin_bot.reporting.http_server import set_healthy, start_metrics_server
    from kucoin_bot.reporting.retention import purge_old_snapshots

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

    # Start observability HTTP server (/healthz, /metrics)
    metrics_runner = await start_metrics_server()
    METRICS.set("bot_info", 1, labels={"mode": mode_label})

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

    # Now compute full portfolio equity (all assets converted to USDT)
    total_equity = await _compute_total_equity(client, market_data)
    if total_equity > 0:
        risk_mgr.update_equity(total_equity)
        logger.info("[%s] Total portfolio equity: %.2f USDT", mode_label, total_equity)

    # Startup reconciliation: rebuild risk state from exchange truth
    if not is_paper and not is_shadow:
        await _reconcile_positions(client, risk_mgr, market_data, logger, mode_label)

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

    # Cooldown tracking: symbol → last slow-cycle index.
    # Cooldown bars map directly to slow-path cycles (one cycle per candle).
    last_entry_slow: dict[str, int] = {}
    cooldown_bars = cfg.risk.cooldown_bars

    # Daily reset tracking (UTC date string)
    last_reset_day: str = time.strftime("%Y-%m-%d", time.gmtime())

    # Slow-path timing: run klines + signals + allocation only after a new
    # candle has closed.  The period equals the default kline interval.
    if cfg.kline_type not in _KLINE_PERIOD_SECONDS:
        logger.warning(
            "Unknown KLINE_TYPE %r, falling back to 15min (900s). Valid: %s",
            cfg.kline_type,
            ", ".join(sorted(_KLINE_PERIOD_SECONDS)),
        )
    slow_period = _KLINE_PERIOD_SECONDS.get(cfg.kline_type, 900)
    last_slow_ts: float = 0.0  # force immediate first slow run
    slow_cycle = 0

    # Dashboard interval: print roughly every 10 minutes.
    dashboard_interval = max(1, 600 // max(cfg.fast_interval, 1))

    logger.info(
        "Starting %s loop with %d strategies | EV gate: %.0f bps | cooldown: %d bars"
        " | fast interval: %ds | slow period: %ds",
        mode_label,
        len(strategies),
        cfg.risk.min_ev_bps,
        cfg.risk.cooldown_bars,
        cfg.fast_interval,
        slow_period,
    )
    cycle = 0
    active_strategies: dict[str, str] = {}

    while not stop_event.is_set():
        # Check kill switch
        if cfg.kill_switch or os.getenv("KILL_SWITCH", "false").lower() == "true":
            logger.critical("KILL SWITCH activated – cancelling all orders and flattening positions")
            if not is_shadow and not is_paper:
                await exec_engine.cancel_all()
            for sym, kill_pos in list(risk_mgr.positions.items()):
                if kill_pos.size > 0:
                    close_price = kill_pos.current_price or kill_pos.entry_price
                    if close_price <= 0:
                        logger.error("No price for %s, cannot flatten", sym)
                        continue
                    try:
                        if not is_shadow and not is_paper:
                            kill_market = market_data.get_info(sym)
                            await exec_engine.flatten_position(
                                sym,
                                kill_pos.size,
                                close_price,
                                kill_pos.side,
                                kill_market,
                                contract_multiplier=kill_pos.contract_multiplier,
                            )
                        risk_mgr.update_position(sym, PositionInfo(symbol=sym, side=kill_pos.side, size=0))
                    except Exception:
                        logger.error("Failed to flatten %s", sym, exc_info=True)
            break

        cycle += 1
        now = time.time()
        try:
            # ── Fast path (every cycle) ────────────────────────────
            # Daily PnL / circuit-breaker reset at UTC midnight
            current_day = time.strftime("%Y-%m-%d", time.gmtime())
            if current_day != last_reset_day:
                risk_mgr.reset_daily()
                last_reset_day = current_day
                logger.info("Daily PnL reset. New day: %s", current_day)

            # Evaluate circuit breaker each cycle (matches backtest behaviour)
            if risk_mgr.check_circuit_breaker():
                logger.critical("Circuit breaker active – skipping entries this cycle")

            # Stop-price checks: flatten positions whose stop has been hit
            for sym, stop_pos in list(risk_mgr.positions.items()):
                if (
                    stop_pos.stop_price is not None
                    and stop_pos.stop_price > 0
                    and stop_pos.current_price > 0
                    and stop_pos.size > 0
                ):
                    triggered = (stop_pos.side == "long" and stop_pos.current_price <= stop_pos.stop_price) or (
                        stop_pos.side == "short" and stop_pos.current_price >= stop_pos.stop_price
                    )
                    if triggered:
                        logger.warning(
                            "Stop triggered for %s (%s): price=%.4f stop=%.4f",
                            sym,
                            stop_pos.side,
                            stop_pos.current_price,
                            stop_pos.stop_price,
                        )
                        if not is_shadow and not is_paper:
                            try:
                                stop_market = market_data.get_info(sym)
                                await exec_engine.flatten_position(
                                    sym,
                                    stop_pos.size,
                                    stop_pos.current_price,
                                    stop_pos.side,
                                    stop_market,
                                )
                            except Exception:
                                logger.error("Failed to flatten %s on stop", sym, exc_info=True)
                        risk_mgr.update_position(sym, PositionInfo(symbol=sym, side=stop_pos.side, size=0))

            # ── Slow path (candle boundary) ────────────────────────
            run_slow = last_slow_ts == 0.0 or now >= last_slow_ts + slow_period
            if run_slow:
                slow_start = time.time()
                last_slow_ts = now
                slow_cycle += 1

                # Refresh universe and full portfolio equity each slow cycle
                await market_data.refresh_universe()
                # Only refresh equity from the API in live mode.
                # In paper/shadow mode the equity is tracked internally;
                # overwriting it would lose simulated P&L.
                if not is_paper and not is_shadow:
                    total_equity = await _compute_total_equity(client, market_data)
                    if total_equity > 0:
                        risk_mgr.update_equity(total_equity)

                    # Open-order reconciliation: check for stale orders every slow cycle
                    try:
                        for oo in await client.get_open_orders() or []:
                            oid = oo.get("id", "")
                            created = float(oo.get("createdAt", 0) or 0) / 1000.0
                            if created > 0 and now - created > _ORDER_STALE_SECONDS:
                                logger.warning(
                                    "Cancelling stale spot order %s (%s %s) age=%.0fs",
                                    oid, oo.get("side", "?"), oo.get("symbol", "?"), now - created,
                                )
                                try:
                                    await client.cancel_order(oid)
                                except Exception:
                                    logger.error("Failed to cancel stale spot order %s", oid, exc_info=True)
                    except Exception:
                        logger.debug("Open-order reconciliation (spot) failed", exc_info=True)
                    try:
                        for fo in await client.get_futures_open_orders() or []:
                            oid = fo.get("id", "")
                            created = float(fo.get("createdAt", 0) or 0) / 1000.0
                            if created > 0 and now - created > _ORDER_STALE_SECONDS:
                                logger.warning(
                                    "Cancelling stale futures order %s (%s %s) age=%.0fs",
                                    oid, fo.get("side", "?"), fo.get("symbol", "?"), now - created,
                                )
                                try:
                                    await client.cancel_futures_order(oid)
                                except Exception:
                                    logger.error("Failed to cancel stale futures order %s", oid, exc_info=True)
                    except Exception:
                        logger.debug("Open-order reconciliation (futures) failed", exc_info=True)

                # Process each pair
                # Build the symbol list: universe symbols (optionally capped) plus
                # any symbols that have open positions so they are always managed.
                active_strategies.clear()
                universe_syms = market_data.get_symbols()
                if cfg.max_symbols > 0:
                    universe_syms = universe_syms[: cfg.max_symbols]
                position_syms = [s for s in risk_mgr.positions if s not in universe_syms]
                symbols_to_process = universe_syms + position_syms

                # Collect signals for all symbols first so portfolio allocation
                # can normalise weights across the whole universe.
                all_signals: dict[str, SignalScores] = {}

                # Fetch klines in parallel with bounded concurrency
                max_conc = int(os.getenv("MAX_KLINE_CONCURRENCY", "8"))
                sem = asyncio.Semaphore(max_conc)

                async def _fetch_klines(s: str) -> tuple[str, list]:
                    async with sem:
                        return s, await market_data.get_klines(s, kline_type=cfg.kline_type)

                kline_results = await asyncio.gather(
                    *(_fetch_klines(s) for s in symbols_to_process),
                    return_exceptions=True,
                )

                for item in kline_results:
                    if isinstance(item, Exception):
                        logger.error("Error fetching klines: %s", item, exc_info=False)
                        continue
                    sym_k, kl = item
                    try:
                        if len(kl) < signal_engine.lookback:
                            continue
                        all_signals[sym_k] = signal_engine.compute(sym_k, kl)
                    except Exception:
                        logger.error("Error computing signals for %s", sym_k, exc_info=True)

                # Compute portfolio allocations across all symbols at once
                batch_allocs = portfolio_mgr.compute_allocations(all_signals, list(all_signals.keys()))

                batch_db_enabled = os.getenv("BATCH_DB_WRITES", "1") != "0"
                _snapshots: list = []

                for sym in symbols_to_process:
                    if sym not in all_signals:
                        continue
                    try:
                        signals = all_signals[sym]
                        market = market_data.get_info(sym)

                        if cfg.live_diagnostic:
                            logger.debug("[DIAG] %s signals: %s", sym, signals.to_dict())

                        # Get existing position before any gates so exits can always be evaluated
                        pos = risk_mgr.positions.get(sym)

                        # Update position's current price for accurate risk calculations
                        if pos and market and market.last_price > 0:
                            pos.current_price = market.last_price

                        # Portfolio allocation (from batch computation)
                        alloc = batch_allocs.get(sym)
                        if not alloc or alloc.weight <= 0:
                            if not pos:
                                continue
                            # Open position with zero allocation – fall back to risk_off to evaluate exit
                            alloc = AllocationTarget(symbol=sym, weight=0.0, strategy="risk_off")

                        active_strategies[sym] = alloc.strategy

                        # Get matching strategy
                        strat_cls = STRATEGY_MAP.get(alloc.strategy)
                        if not strat_cls:
                            continue
                        strat = strat_cls()

                        if not strat.preconditions_met(signals):
                            if not pos:
                                continue
                            # For existing positions bypass preconditions and allow exit via risk_off
                            strat = RiskOff()

                        # Check strategy monitor – skip if module auto-disabled;
                        # still allow exits for open positions
                        if not strategy_monitor.is_enabled(alloc.strategy):
                            if cfg.live_diagnostic:
                                logger.debug(
                                    "[MONITOR] Skipping %s: module '%s' auto-disabled",
                                    sym,
                                    alloc.strategy,
                                )
                            if not pos:
                                continue
                            strat = RiskOff()

                        decision = strat.evaluate(
                            signals,
                            pos.side if pos else None,
                            pos.entry_price if pos else None,
                            market.last_price if market else 0,
                        )

                        # Collect snapshot for batched DB write
                        _snapshots.append(
                            SignalSnapshot(
                                symbol=sym,
                                regime=signals.regime.value,
                                strategy_name=alloc.strategy,
                                signal_data=json.dumps(signals.to_dict()),
                                decision=decision.action,
                                reason=decision.reason,
                            )
                        )

                        # Side selector: validate proposed side (squeeze filter, feasibility)
                        if decision.action.startswith("entry_") and not pos:
                            mkt_type = market.market_type if market else "spot"
                            proposed_side = "long" if "long" in decision.action else "short"
                            side_dec = side_selector.select(signals, market_type=mkt_type, proposed_side=proposed_side)
                            if side_dec.side == "flat":
                                if cfg.live_diagnostic:
                                    logger.debug(
                                        "[SIDE-SELECTOR] Blocked %s %s: %s",
                                        sym,
                                        proposed_side,
                                        side_dec.reason,
                                    )
                                continue

                        # EV gate: cost-aware, shared with backtest
                        if decision.action.startswith("entry_"):
                            mkt_type = market.market_type if market else "spot"
                            is_futures = mkt_type == "futures"
                            proposed_side_ev = "long" if "long" in decision.action else "short"
                            is_margin_short = mkt_type == "margin" and proposed_side_ev == "short"
                            ev_order_type = (
                                "maker" if getattr(decision, "post_only", False)
                                else "taker"
                            )
                            costs = cost_model.estimate(
                                order_type=ev_order_type,
                                holding_hours=cfg.short.expected_holding_hours,
                                is_futures=is_futures,
                                is_margin_short=is_margin_short,
                                live_funding_rate=signals.funding_rate if signals.funding_rate != 0 else None,
                                position_side=proposed_side_ev,
                            )
                            expected_bps = max(signals.volatility, signals.trend_strength) * 100.0 * signals.confidence
                            if not cost_model.ev_gate(expected_bps, costs):
                                if cfg.live_diagnostic:
                                    logger.debug(
                                        "[EV-GATE] Blocked %s: expected %.1f bps < cost %.1f bps + buffer %.1f",
                                        sym,
                                        expected_bps,
                                        costs.total_bps,
                                        cfg.risk.min_ev_bps,
                                    )
                                continue

                        # Cooldown: enforce minimum bars between entries
                        if decision.action.startswith("entry_") and not pos:
                            last_sc = last_entry_slow.get(sym, -cooldown_bars)
                            if slow_cycle - last_sc < cooldown_bars:
                                continue

                        # Compute notional early so it can be included in the correlated
                        # exposure check below (pending entry not yet in risk_mgr.positions).
                        if decision.action.startswith("entry_") and not pos:
                            notional = risk_mgr.compute_position_size(
                                sym,
                                market.last_price if market else 0,
                                signals.volatility,
                                signals,
                                leverage=alloc.max_leverage,
                            )
                        else:
                            notional = 0.0

                        # Correlated exposure check: group symbols by base asset.
                        # Prospective notional is included to prevent same-cycle bypass.
                        # We use the full per-position notional (not weight-adjusted)
                        # because trade_notional is no longer scaled by alloc.weight;
                        # position sizing already constrains each position.
                        if decision.action.startswith("entry_") and not pos:
                            if notional <= 0:
                                continue
                            base = market.base if market else sym.split("-")[0]
                            correlated = [s for s in risk_mgr.positions if s.startswith(base + "-")]
                            if sym not in correlated:
                                correlated.append(sym)
                            if risk_mgr.check_correlated_exposure(correlated, prospective_notional=notional):
                                logger.warning("Correlated exposure limit reached for base %s, skipping %s", base, sym)
                                continue

                        # Execute
                        if decision.action.startswith("entry_"):
                            if notional > 0:
                                side = "buy" if "long" in decision.action else "sell"
                                trade_notional = notional

                                if is_shadow:
                                    logger.info(
                                        "[SHADOW] Would %s %s notional=%.2f @ %.4f reason=%s",
                                        side,
                                        sym,
                                        trade_notional,
                                        market.last_price if market else 0,
                                        decision.reason,
                                    )
                                elif is_paper:
                                    # Simulate fill: use last price with taker slippage and taker fee.
                                    # No real API call is made – paper mode only.
                                    last_px = market.last_price if market else 0
                                    slip_dir = 1 if side == "buy" else -1
                                    fill_px = (
                                        last_px * (1 + slip_dir * DEFAULT_SLIPPAGE_BPS / 10_000) if last_px > 0 else 0
                                    )
                                    if fill_px > 0:
                                        filled_qty = trade_notional * alloc.max_leverage / fill_px
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
                                            side,
                                            sym,
                                            filled_qty,
                                            fill_px,
                                            paper_fee,
                                        )
                                        pos_side = "long" if "long" in decision.action else "short"
                                        risk_mgr.update_position(
                                            sym,
                                            PositionInfo(
                                                symbol=sym,
                                                side=pos_side,
                                                size=result.filled_qty,
                                                entry_price=result.avg_price,
                                                current_price=result.avg_price,
                                                leverage=alloc.max_leverage,
                                                stop_price=getattr(decision, "stop_price", None),
                                            ),
                                        )
                                        last_entry_slow[sym] = slow_cycle
                                else:
                                    # Transfer funds if trading on a futures market
                                    mkt_type_entry = market.market_type if market else "spot"
                                    if mkt_type_entry == "futures" and cfg.allow_internal_transfers:
                                        # Add 5 % buffer so the futures wallet covers exchange
                                        # fees and initial-margin rounding after the transfer.
                                        xfer_amount = trade_notional * 1.05
                                        xfer = await portfolio_mgr.transfer_if_needed(
                                            "USDT",
                                            "trade",
                                            "futures",
                                            xfer_amount,
                                        )
                                        if xfer is None:
                                            logger.warning("Futures transfer failed for %s, skipping entry", sym)
                                            continue
                                        # Brief pause to let the balance propagate on the exchange.
                                        await asyncio.sleep(0.5)

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
                                    if result.success and result.filled_qty > 0 and result.status in _CONFIRMED_FILL_STATUSES:
                                        pos_side = "long" if "long" in decision.action else "short"
                                        acct = "futures" if mkt_type_entry == "futures" else "trade"
                                        cm = market.contract_multiplier if market and market.contract_multiplier > 0 else 1.0
                                        risk_mgr.update_position(
                                            sym,
                                            PositionInfo(
                                                symbol=sym,
                                                side=pos_side,
                                                size=result.filled_qty,
                                                entry_price=result.avg_price,
                                                current_price=result.avg_price,
                                                leverage=alloc.max_leverage,
                                                account_type=acct,
                                                stop_price=getattr(decision, "stop_price", None),
                                                contract_multiplier=cm,
                                            ),
                                        )
                                        last_entry_slow[sym] = slow_cycle

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
                                    fill_px = (
                                        last_px * (1 + slip_dir * DEFAULT_SLIPPAGE_BPS / 10_000)
                                        if last_px > 0
                                        else pos.entry_price
                                    )
                                    paper_fee = pos.size * fill_px * DEFAULT_TAKER_FEE
                                    raw_pnl = (fill_px - pos.entry_price) * pos.size
                                    if pos.side == "short":
                                        raw_pnl = -raw_pnl
                                    pnl = raw_pnl - paper_fee
                                    risk_mgr.record_pnl(pnl)
                                    risk_mgr.update_equity(risk_mgr.current_equity + pnl)
                                    risk_mgr.update_position(
                                        sym,
                                        PositionInfo(
                                            symbol=sym,
                                            side=pos.side,
                                            size=0,
                                        ),
                                    )
                                    strategy_monitor.record_trade(alloc.strategy, raw_pnl, paper_fee)
                                    logger.info("[PAPER] Simulated exit %s pnl=%.4f fee=%.4f", sym, pnl, paper_fee)
                                else:
                                    exit_cm = pos.contract_multiplier
                                    exit_price = pos.current_price if pos.current_price > 0 else pos.entry_price
                                    result = await exec_engine.execute(
                                        OrderRequest(
                                            symbol=sym,
                                            side=side,
                                            notional=abs(pos.size * exit_cm * exit_price),
                                            order_type=decision.order_type,
                                            reason=decision.reason,
                                            reduce_only=is_futures_exit,
                                        ),
                                        market,
                                    )
                                    if result.success and result.status in _CONFIRMED_FILL_STATUSES:
                                        pnl = (result.avg_price - pos.entry_price) * pos.size * exit_cm
                                        if pos.side == "short":
                                            pnl = -pnl
                                        risk_mgr.record_pnl(pnl)
                                        risk_mgr.update_position(
                                            sym,
                                            PositionInfo(
                                                symbol=sym,
                                                side=pos.side,
                                                size=0,
                                            ),
                                        )
                                        # Estimate exit fee from fill to track net expectancy
                                        trade_cost = (
                                            pos.size * result.avg_price * DEFAULT_TAKER_FEE
                                            if result.avg_price > 0
                                            else 0.0
                                        )
                                        strategy_monitor.record_trade(alloc.strategy, pnl, trade_cost)
                                        # Transfer proceeds back from futures account
                                        if is_futures_exit:
                                            margin = abs(pos.size * pos.entry_price / max(pos.leverage, 1.0))
                                            exit_value = max(0, margin + pnl)
                                            await portfolio_mgr.transfer_if_needed(
                                                "USDT",
                                                "futures",
                                                "trade",
                                                exit_value,
                                            )

                        # HOLD path: update stop_price if strategy provides a new one
                        elif decision.action == "hold" and pos:
                            new_stop = getattr(decision, "stop_price", None)
                            if new_stop is not None:
                                pos.stop_price = new_stop
                                logger.debug(
                                    "Updated stop_price for %s to %.4f",
                                    sym,
                                    new_stop,
                                )

                    except Exception:
                        logger.error("Error processing %s", sym, exc_info=True)

                # Flush signal snapshots to DB
                if _snapshots:
                    try:
                        with db_session_factory() as db_sess:
                            for snap in _snapshots:
                                db_sess.add(snap)
                                if not batch_db_enabled:
                                    db_sess.commit()
                            if batch_db_enabled:
                                db_sess.commit()
                    except Exception:
                        logger.exception("Failed to write signal snapshots to DB")

                # Slow-cycle duration metric
                METRICS.observe("slow_cycle_duration_seconds", time.time() - slow_start)

                # Periodic DB retention (once every ~100 slow cycles)
                if slow_cycle % 100 == 0:
                    purge_old_snapshots(db_session_factory)

            # Update health probe and equity/exposure gauges each fast cycle
            set_healthy(not risk_mgr.circuit_breaker_active)
            METRICS.set("equity_usdt", risk_mgr.current_equity)
            METRICS.set("daily_pnl_usdt", risk_mgr.daily_pnl)
            METRICS.set("circuit_breaker_active", 1.0 if risk_mgr.circuit_breaker_active else 0.0)
            total_exposure = sum(
                risk_mgr.position_notional(p) for p in risk_mgr.positions.values()
            )
            METRICS.set("total_exposure_usdt", total_exposure)

            # Dashboard
            if cycle % dashboard_interval == 0:
                print_dashboard(risk_mgr, active_strategies)
                if cfg.live_diagnostic:
                    logger.info("[MONITOR] %s", strategy_monitor.get_status())

        except Exception:
            logger.error("Trading loop error", exc_info=True)

        await asyncio.sleep(cfg.fast_interval)

    # Cleanup
    logger.info("Shutting down...")
    await metrics_runner.cleanup()
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
    # Reject unknown CLI args; print a hint for --help/-h.
    argv = sys.argv[1:]
    if argv:
        if len(argv) == 1 and argv[0] in ("-h", "--help"):
            print(
                "KuCoin Trading Bot\n"
                "\n"
                "Usage:\n"
                "  python -m kucoin_bot\n"
                "\n"
                "This bot is configured entirely via environment variables.\n"
                "Copy .env.example to .env and set your values before starting.\n"
                "Key variables: BOT_MODE, LIVE_TRADING, KUCOIN_API_KEY, KUCOIN_API_SECRET, KUCOIN_API_PASSPHRASE"
            )
            sys.exit(0)
        else:
            print(
                f"ERROR: Unknown command-line arguments: {' '.join(argv)}\n"
                "This bot does not accept CLI options; configure it via environment variables."
            )
            sys.exit(2)
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
