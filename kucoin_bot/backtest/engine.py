"""Backtesting engine with realistic fees, slippage, partial fills, and walk-forward."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import numpy as np

from kucoin_bot.config import RiskConfig
from kucoin_bot.services.signal_engine import SignalEngine, SignalScores
from kucoin_bot.services.risk_manager import RiskManager, PositionInfo
from kucoin_bot.services.cost_model import CostModel, TradeCosts
from kucoin_bot.services.side_selector import SideSelector
from kucoin_bot.strategies.base import BaseStrategy, StrategyDecision

logger = logging.getLogger(__name__)

# Realistic KuCoin fee tiers
DEFAULT_MAKER_FEE = 0.001  # 0.1%
DEFAULT_TAKER_FEE = 0.001
DEFAULT_SLIPPAGE_BPS = 5  # 0.05%
DEFAULT_FILL_RATE = 0.95  # 95% fill probability for limit orders

# Latency slippage scaling: each extra second of latency adds this fraction of
# one slippage_bps unit as additional adverse drift during order transit time.
LATENCY_DRIFT_FACTOR = 0.1


@dataclass
class BacktestTrade:
    """Record of a backtest fill."""

    timestamp: int
    symbol: str
    side: str
    price: float
    quantity: float
    fee: float
    pnl: float = 0.0
    funding_cost: float = 0.0   # cumulative futures funding cost for this position
    borrow_cost: float = 0.0    # cumulative margin borrow cost for this position
    position_side: str = ""     # "long" or "short" for exit records


@dataclass
class BacktestResult:
    """Summary of a backtest run."""

    initial_equity: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    total_fees: float
    expectancy: float = 0.0   # avg PnL per closed round-trip trade
    turnover: float = 0.0     # total notional traded / initial equity
    # Per-side breakdown
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    long_trades: int = 0
    short_trades: int = 0
    # Cost breakdown
    cost_breakdown: Dict[str, float] = field(default_factory=lambda: {
        "fees": 0.0,
        "slippage": 0.0,
        "funding": 0.0,
        "borrow": 0.0,
    })
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    def summary(self) -> str:
        cb = self.cost_breakdown
        return (
            f"Backtest: {self.total_trades} trades "
            f"(L:{self.long_trades} S:{self.short_trades}) | "
            f"Return: {self.total_return_pct:.2f}% | "
            f"Max DD: {self.max_drawdown_pct:.2f}% | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"Win Rate: {self.win_rate:.1f}% | "
            f"Fees: {self.total_fees:.2f} | "
            f"Expectancy: {self.expectancy:.4f} | "
            f"Long PnL: {self.long_pnl:.4f} | "
            f"Short PnL: {self.short_pnl:.4f} | "
            f"Costs[fee={cb.get('fees', 0):.2f} "
            f"slip={cb.get('slippage', 0):.2f} "
            f"fund={cb.get('funding', 0):.2f} "
            f"borr={cb.get('borrow', 0):.2f}]"
        )


@dataclass
class BacktestEngine:
    """Walk-forward backtester with realistic next-bar execution and EV gate."""

    strategies: List[BaseStrategy] = field(default_factory=list)
    signal_engine: SignalEngine = field(default_factory=SignalEngine)
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    maker_fee: float = DEFAULT_MAKER_FEE
    taker_fee: float = DEFAULT_TAKER_FEE
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    fill_rate: float = DEFAULT_FILL_RATE
    seed: int = 42
    # Simulated execution latency: adds proportional extra slippage
    latency_ms: int = 0
    # EV gate: override from risk_config if > 0
    min_ev_bps: float = -1.0
    # Cooldown: override from risk_config if >= 0
    cooldown_bars: int = -1
    # Cost model (created from maker/taker/slippage params if not provided)
    cost_model: Optional[CostModel] = None
    # Side selector (created from default settings if not provided)
    side_selector: Optional[SideSelector] = None

    def _effective_min_ev_bps(self) -> float:
        return self.min_ev_bps if self.min_ev_bps >= 0 else self.risk_config.min_ev_bps

    def _effective_cooldown_bars(self) -> int:
        return self.cooldown_bars if self.cooldown_bars >= 0 else self.risk_config.cooldown_bars

    def _get_cost_model(self) -> CostModel:
        if self.cost_model is not None:
            return self.cost_model
        return CostModel(
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
            slippage_bps=self.slippage_bps,
            safety_buffer_bps=self._effective_min_ev_bps(),
        )

    def _get_side_selector(self) -> SideSelector:
        if self.side_selector is not None:
            return self.side_selector
        return SideSelector()

    def run(
        self,
        klines: List[list],
        symbol: str = "BTC-USDT",
        initial_equity: float = 10_000.0,
        warmup: int = 60,
        market_type: str = "spot",
    ) -> BacktestResult:
        """Run backtest on kline data with next-bar execution (no look-ahead).

        Signal is generated at the close of bar i.
        Fill is executed at the open of bar i+1 (realistic: you see the close,
        then submit an order that fills when the next bar opens).

        Args:
            klines: KuCoin-format klines [[time, open, close, high, low, vol, turnover], ...]
            symbol: Symbol name.
            initial_equity: Starting capital in USDT.
            warmup: Bars to skip for indicator warm-up.
            market_type: ``"spot"``, ``"futures"``, or ``"margin"`` – controls
                whether funding/borrow costs are modelled and reduceOnly is
                applied to exits.
        """
        rng = random.Random(self.seed)
        risk_mgr = RiskManager(config=self.risk_config)
        risk_mgr.update_equity(initial_equity)

        cost_model = self._get_cost_model()
        side_selector = self._get_side_selector()
        is_futures = market_type == "futures"
        is_margin = market_type == "margin"
        cooldown_bars = self._effective_cooldown_bars()

        equity = initial_equity
        equity_curve = [equity]
        trades: List[BacktestTrade] = []
        position_side: Optional[str] = None
        entry_price: float = 0.0
        position_size: float = 0.0
        total_fees: float = 0.0
        total_funding_cost: float = 0.0
        total_borrow_cost: float = 0.0
        total_slippage_cost: float = 0.0
        holding_bars: int = 0  # bars held in current position

        # Allow entry at warmup bar (last_entry_bar set before warmup)
        last_entry_bar: int = warmup - cooldown_bars - 1

        # Pending orders queued at bar i, executed at bar i+1 open
        _pending_entry: Optional[Tuple[StrategyDecision, SignalScores]] = None
        _pending_exit: bool = False

        # Effective slippage for fills, including latency impact.
        effective_slippage_bps = self.slippage_bps + (self.latency_ms / 1000.0) * self.slippage_bps * LATENCY_DRIFT_FACTOR

        for i in range(warmup, len(klines)):
            bar = klines[i]
            ts = int(bar[0])
            open_price = float(bar[1])
            close = float(bar[2])

            # ---- STEP 1: Execute pending orders at THIS bar's open (next-bar fill) ----
            if _pending_entry is not None and not position_side:
                pend_decision, pend_signals = _pending_entry
                _pending_entry = None
                slip_dir = 1 if "long" in pend_decision.action else -1
                slip = open_price * (effective_slippage_bps / 10_000) * slip_dir
                fill_price = open_price + slip
                slip_cost = abs(open_price * (effective_slippage_bps / 10_000))

                notional = risk_mgr.compute_position_size(
                    symbol, fill_price, pend_signals.volatility, pend_signals
                )
                if notional > 0:
                    size = notional / fill_price
                    if pend_decision.order_type == "limit" and rng.random() > self.fill_rate:
                        size *= rng.uniform(0.3, 0.9)
                    fee_rate = (
                        self.taker_fee if pend_decision.order_type == "market" else self.maker_fee
                    )
                    fee = size * fill_price * fee_rate
                    total_fees += fee
                    total_slippage_cost += slip_cost * size
                    equity -= fee

                    position_side = "long" if "long" in pend_decision.action else "short"
                    entry_price = fill_price
                    position_size = size
                    holding_bars = 0
                    last_entry_bar = i
                    trades.append(
                        BacktestTrade(ts, symbol, pend_decision.action, fill_price, size, fee)
                    )

            if _pending_exit and position_side:
                _pending_exit = False
                slip_dir = -1 if position_side == "long" else 1
                slip = open_price * (effective_slippage_bps / 10_000) * slip_dir
                fill_price = open_price + slip
                slip_cost = abs(open_price * (effective_slippage_bps / 10_000))

                fee = position_size * fill_price * self.taker_fee
                total_fees += fee
                total_slippage_cost += slip_cost * position_size
                pnl = (fill_price - entry_price) * position_size
                if position_side == "short":
                    pnl = -pnl
                pnl -= fee

                trades.append(
                    BacktestTrade(
                        ts, symbol, "exit", fill_price, position_size, fee, pnl,
                        position_side=position_side,
                    )
                )
                equity += pnl
                risk_mgr.record_pnl(pnl)
                position_side = None
                position_size = 0.0
                entry_price = 0.0
                holding_bars = 0

            # ---- STEP 2: MTM and per-bar carrying costs ----
            if position_side and position_size > 0:
                holding_bars += 1

                # Futures funding: paid every 8 bars (1-hour bars → every 8 hours)
                # Positive rate: longs pay, shorts receive. Negative rate: the reverse.
                if is_futures and holding_bars % 8 == 0:
                    funding = position_size * entry_price * cost_model.funding_rate_per_8h
                    if position_side == "short":
                        funding = -funding  # shorts receive when rate is positive, pay when negative
                    equity -= funding
                    total_funding_cost += funding

                # Margin borrow: paid every bar for margin shorts
                if is_margin and position_side == "short":
                    borrow = position_size * entry_price * cost_model.borrow_rate_per_hour
                    equity -= borrow
                    total_borrow_cost += borrow

                unrealized = (close - entry_price) * position_size
                if position_side == "short":
                    unrealized = -unrealized
                risk_mgr.update_equity(equity + unrealized)
            else:
                risk_mgr.update_equity(equity)

            # ---- STEP 3: Circuit breaker ----
            if risk_mgr.check_circuit_breaker():
                _pending_entry = None  # Cancel any pending entry
                if position_side and not _pending_exit:
                    _pending_exit = True  # Queue exit for next bar
                equity_curve.append(equity)
                continue

            # ---- STEP 4: Signals at end of bar i (no look-ahead) ----
            window = klines[max(0, i - self.signal_engine.lookback) : i + 1]
            signals = self.signal_engine.compute(symbol, window)

            # ---- STEP 5: Strategy evaluation ----
            decision = StrategyDecision(action="hold", symbol=symbol)
            for strat in self.strategies:
                if strat.preconditions_met(signals):
                    decision = strat.evaluate(signals, position_side, entry_price, close)
                    if decision.action != "hold":
                        break

            # ---- STEP 5b: Side selector filter ----
            if decision.action.startswith("entry_") and not position_side:
                proposed = "long" if "long" in decision.action else "short"
                side_dec = side_selector.select(signals, market_type=market_type, proposed_side=proposed)
                if side_dec.side == "flat":
                    logger.debug(
                        "Side selector blocked %s %s: %s",
                        symbol, proposed, side_dec.reason,
                    )
                    equity_curve.append(equity)
                    continue

            # ---- STEP 6: EV gate – only trade when expected edge exceeds round-trip cost ----
            if decision.action.startswith("entry_") and not position_side:
                order_type_str = "taker" if decision.order_type == "market" else "maker"
                proposed_side = "long" if "long" in decision.action else "short"
                is_margin_short = is_margin and proposed_side == "short"
                costs = cost_model.estimate(
                    order_type=order_type_str,
                    holding_hours=24.0,  # conservative expected holding
                    is_futures=is_futures,
                    is_margin_short=is_margin_short,
                    live_funding_rate=signals.funding_rate if signals.funding_rate != 0 else None,
                    position_side=proposed_side,
                )
                expected_bps = signals.volatility * 100.0 * signals.confidence
                if not cost_model.ev_gate(expected_bps, costs):
                    logger.debug(
                        "EV gate blocked %s entry: expected %.1f bps < cost %.1f bps + buffer %.1f",
                        symbol, expected_bps, costs.total_bps, cost_model.safety_buffer_bps,
                    )
                    equity_curve.append(equity)
                    continue

            # ---- STEP 7: Cooldown – minimum bars between entries ----
            if decision.action.startswith("entry_") and not position_side:
                if i - last_entry_bar < cooldown_bars:
                    equity_curve.append(equity)
                    continue

            # ---- STEP 8: Queue pending decision for execution at NEXT bar's open ----
            if decision.action.startswith("entry_") and not position_side:
                _pending_entry = (decision, signals)
            elif decision.action == "exit" and position_side:
                _pending_exit = True

            equity_curve.append(equity)

        # Close any open position at the final bar's close (no future bar available)
        if position_side and position_size > 0:
            close = float(klines[-1][2])
            fee = position_size * close * self.taker_fee
            pnl = (close - entry_price) * position_size
            if position_side == "short":
                pnl = -pnl
            pnl -= fee
            total_fees += fee
            trades.append(
                BacktestTrade(
                    int(klines[-1][0]), symbol, "exit", close, position_size, fee, pnl,
                    position_side=position_side,
                )
            )
            equity += pnl

        return self._compute_result(
            initial_equity, equity, equity_curve, trades, total_fees,
            total_funding_cost, total_borrow_cost, total_slippage_cost,
        )

    def walk_forward(
        self,
        klines: List[list],
        symbol: str = "BTC-USDT",
        n_splits: int = 5,
        initial_equity: float = 10_000.0,
    ) -> List[BacktestResult]:
        """Walk-forward out-of-sample evaluation.

        Divides klines into (n_splits + 1) equal windows.  Each test fold is
        the window immediately following the first window (which acts as
        warm-up), sliding forward one window at a time.

        warmup=0 is passed to run() because each test slice is already out-of-sample
        data; consuming warmup bars within the slice would discard valid test bars
        in small folds.  The signal engine lookback is still respected inside run()
        via the rolling window slicing.

        Returns a list of BacktestResult, one per fold.
        """
        n = len(klines)
        fold_size = n // (n_splits + 1)
        results: List[BacktestResult] = []
        for fold in range(n_splits):
            test_start = fold_size * (fold + 1)
            test_end = fold_size * (fold + 2)
            test_klines = klines[test_start:test_end]
            min_required = self.signal_engine.lookback + 10
            if len(test_klines) > min_required:
                # warmup=0: this slice is already OOS; no need to discard bars
                # for indicator warm-up within the slice itself.
                result = self.run(
                    test_klines, symbol, initial_equity=initial_equity, warmup=0
                )
                results.append(result)
        logger.info("Walk-forward complete: %d/%d folds evaluated", len(results), n_splits)
        return results

    @staticmethod
    def _compute_result(
        initial: float,
        final: float,
        curve: List[float],
        trades: List[BacktestTrade],
        fees: float,
        funding_cost: float = 0.0,
        borrow_cost: float = 0.0,
        slippage_cost: float = 0.0,
    ) -> BacktestResult:
        ret_pct = (final - initial) / initial * 100 if initial > 0 else 0

        # Max drawdown
        peak = initial
        max_dd = 0.0
        for eq in curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (daily-ish returns)
        returns = []
        for i in range(1, len(curve)):
            if curve[i - 1] > 0:
                returns.append((curve[i] - curve[i - 1]) / curve[i - 1])
        sharpe = 0.0
        if returns and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

        # Win rate and per-side breakdown
        closed = [t for t in trades if t.side == "exit"]
        wins = sum(1 for t in closed if t.pnl > 0)
        win_rate = (wins / len(closed) * 100) if closed else 0

        long_exits = [t for t in closed if t.position_side == "long"]
        short_exits = [t for t in closed if t.position_side == "short"]
        long_pnl = sum(t.pnl for t in long_exits)
        short_pnl = sum(t.pnl for t in short_exits)

        # Expectancy: average PnL per closed trade
        expectancy = sum(t.pnl for t in closed) / len(closed) if closed else 0.0

        # Turnover: total notional traded / initial equity
        volume = sum(t.price * t.quantity for t in trades)
        turnover = volume / initial if initial > 0 else 0.0

        return BacktestResult(
            initial_equity=initial,
            final_equity=final,
            total_return_pct=round(ret_pct, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 2),
            win_rate=round(win_rate, 1),
            total_trades=len(trades),
            total_fees=round(fees, 4),
            expectancy=round(expectancy, 4),
            turnover=round(turnover, 2),
            long_pnl=round(long_pnl, 4),
            short_pnl=round(short_pnl, 4),
            long_trades=len(long_exits),
            short_trades=len(short_exits),
            cost_breakdown={
                "fees": round(fees, 4),
                "slippage": round(slippage_cost, 4),
                "funding": round(funding_cost, 4),
                "borrow": round(borrow_cost, 4),
            },
            trades=trades,
            equity_curve=curve,
        )
