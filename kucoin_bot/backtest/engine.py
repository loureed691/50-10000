"""Backtesting engine with realistic fees, slippage, partial fills, and walk-forward."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import numpy as np

from kucoin_bot.config import RiskConfig
from kucoin_bot.services.signal_engine import SignalEngine, SignalScores
from kucoin_bot.services.risk_manager import RiskManager, PositionInfo
from kucoin_bot.strategies.base import BaseStrategy, StrategyDecision

logger = logging.getLogger(__name__)

# Realistic KuCoin fee tiers
DEFAULT_MAKER_FEE = 0.001  # 0.1%
DEFAULT_TAKER_FEE = 0.001
DEFAULT_SLIPPAGE_BPS = 5  # 0.05%
DEFAULT_FILL_RATE = 0.95  # 95% fill probability for limit orders
DEFAULT_LATENCY_MS = 50  # simulated order latency in milliseconds


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
    latency_ms: float = 0.0


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
    expectancy: float = 0.0
    turnover: float = 0.0
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Backtest: {self.total_trades} trades | "
            f"Return: {self.total_return_pct:.2f}% | "
            f"Max DD: {self.max_drawdown_pct:.2f}% | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"Win Rate: {self.win_rate:.1f}% | "
            f"Expectancy: {self.expectancy:.2f} | "
            f"Turnover: {self.turnover:.2f} | "
            f"Fees: {self.total_fees:.2f}"
        )


@dataclass
class BacktestEngine:
    """Walk-forward backtester with realistic execution simulation."""

    strategies: List[BaseStrategy] = field(default_factory=list)
    signal_engine: SignalEngine = field(default_factory=SignalEngine)
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    maker_fee: float = DEFAULT_MAKER_FEE
    taker_fee: float = DEFAULT_TAKER_FEE
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    fill_rate: float = DEFAULT_FILL_RATE
    latency_ms: float = DEFAULT_LATENCY_MS
    seed: int = 42

    def run(
        self,
        klines: List[list],
        symbol: str = "BTC-USDT",
        initial_equity: float = 10_000.0,
        warmup: int = 60,
    ) -> BacktestResult:
        """Run backtest on kline data.

        Args:
            klines: KuCoin-format klines [[time, open, close, high, low, vol, turnover], ...]
            symbol: Symbol name.
            initial_equity: Starting capital in USDT.
            warmup: Bars to skip for indicator warm-up.
        """
        rng = random.Random(self.seed)
        risk_mgr = RiskManager(config=self.risk_config)
        risk_mgr.update_equity(initial_equity)

        equity = initial_equity
        equity_curve = [equity]
        trades: List[BacktestTrade] = []
        position_side: Optional[str] = None
        entry_price: float = 0.0
        position_size: float = 0.0
        total_fees: float = 0.0

        for i in range(warmup, len(klines)):
            bar = klines[i]
            ts = int(bar[0])
            close = float(bar[2])

            # Compute signals on window up to this bar
            window = klines[max(0, i - self.signal_engine.lookback) : i + 1]
            signals = self.signal_engine.compute(symbol, window)

            # Update risk manager
            if position_side and position_size > 0:
                unrealized = (close - entry_price) * position_size
                if position_side == "short":
                    unrealized = -unrealized
                risk_mgr.update_equity(equity + unrealized)
            else:
                risk_mgr.update_equity(equity)

            # Circuit breaker check
            if risk_mgr.check_circuit_breaker():
                if position_side:
                    # Force exit
                    fee = position_size * close * self.taker_fee
                    pnl = (close - entry_price) * position_size
                    if position_side == "short":
                        pnl = -pnl
                    pnl -= fee
                    trades.append(BacktestTrade(ts, symbol, "exit", close, position_size, fee, pnl))
                    equity += pnl
                    total_fees += fee
                    position_side = None
                    position_size = 0
                    entry_price = 0
                equity_curve.append(equity)
                continue

            # Evaluate strategies
            decision = StrategyDecision(action="hold", symbol=symbol)
            for strat in self.strategies:
                if strat.preconditions_met(signals):
                    decision = strat.evaluate(signals, position_side, entry_price, close)
                    if decision.action != "hold":
                        break

            # Execute decision
            if decision.action.startswith("entry_") and not position_side:
                # Position sizing
                notional = risk_mgr.compute_position_size(symbol, close, signals.volatility, signals)
                if notional > 0:
                    size = notional / close
                    # Latency simulation: use a slightly later price proxy via noise
                    latency = rng.gauss(self.latency_ms, self.latency_ms * 0.2)
                    latency_slip = close * (latency / 1_000) * 0.0001  # tiny drift per ms
                    # Slippage
                    slip = close * (self.slippage_bps / 10_000) * (1 if "long" in decision.action else -1)
                    fill_price = close + slip + latency_slip

                    # Partial fill simulation
                    if decision.order_type == "limit" and rng.random() > self.fill_rate:
                        size *= rng.uniform(0.3, 0.9)

                    fee = size * fill_price * (self.taker_fee if decision.order_type == "market" else self.maker_fee)
                    total_fees += fee
                    equity -= fee

                    position_side = "long" if "long" in decision.action else "short"
                    entry_price = fill_price
                    position_size = size
                    trades.append(BacktestTrade(ts, symbol, decision.action, fill_price, size, fee, latency_ms=latency))

            elif decision.action == "exit" and position_side:
                latency = rng.gauss(self.latency_ms, self.latency_ms * 0.2)
                latency_slip = close * (latency / 1_000) * 0.0001
                slip = close * (self.slippage_bps / 10_000) * (-1 if position_side == "long" else 1)
                fill_price = close + slip + latency_slip
                fee = position_size * fill_price * self.taker_fee
                total_fees += fee

                pnl = (fill_price - entry_price) * position_size
                if position_side == "short":
                    pnl = -pnl
                pnl -= fee

                trades.append(BacktestTrade(ts, symbol, "exit", fill_price, position_size, fee, pnl, latency_ms=latency))
                equity += pnl
                risk_mgr.record_pnl(pnl)
                position_side = None
                position_size = 0
                entry_price = 0

            equity_curve.append(equity)

        # Close any open position at end
        if position_side and position_size > 0:
            close = float(klines[-1][2])
            pnl = (close - entry_price) * position_size
            if position_side == "short":
                pnl = -pnl
            fee = position_size * close * self.taker_fee
            pnl -= fee
            total_fees += fee
            trades.append(BacktestTrade(int(klines[-1][0]), symbol, "exit", close, position_size, fee, pnl))
            equity += pnl

        return self._compute_result(initial_equity, equity, equity_curve, trades, total_fees)

    @staticmethod
    def _compute_result(
        initial: float,
        final: float,
        curve: List[float],
        trades: List[BacktestTrade],
        fees: float,
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

        # Win rate
        closed = [t for t in trades if t.side == "exit"]
        wins = sum(1 for t in closed if t.pnl > 0)
        win_rate = (wins / len(closed) * 100) if closed else 0

        # Expectancy = avg PnL per closed trade
        expectancy = float(np.mean([t.pnl for t in closed])) if closed else 0.0

        # Turnover = total traded notional relative to initial equity
        total_notional = sum(t.price * t.quantity for t in trades)
        turnover = (total_notional / initial) if initial > 0 else 0.0

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
            turnover=round(turnover, 4),
            trades=trades,
            equity_curve=curve,
        )

    def walk_forward(
        self,
        klines: List[list],
        symbol: str = "BTC-USDT",
        initial_equity: float = 10_000.0,
        n_splits: int = 5,
        warmup: int = 60,
    ) -> List[BacktestResult]:
        """Walk-forward evaluation: split data into n_splits folds, run OOS on each.

        Each fold uses the previous data as in-sample (warm-up) and the current
        segment as out-of-sample (test).  Returns one BacktestResult per fold.

        Args:
            klines: Full kline history.
            symbol: Symbol name.
            initial_equity: Starting capital for each fold.
            n_splits: Number of out-of-sample folds.
            warmup: Minimum in-sample bars before each test segment.
        """
        results: List[BacktestResult] = []
        n = len(klines)
        if n < warmup + n_splits * 2:
            logger.warning("Not enough data for walk-forward with %d splits", n_splits)
            return [self.run(klines, symbol, initial_equity, warmup)]

        fold_size = (n - warmup) // n_splits
        for fold_idx in range(n_splits):
            oos_start = warmup + fold_idx * fold_size
            oos_end = oos_start + fold_size if fold_idx < n_splits - 1 else n
            # Include all data up to OOS end so warm-up is available
            segment = klines[:oos_end]
            # Warmup for this fold is everything before OOS start
            fold_warmup = oos_start
            result = self.run(segment, symbol, initial_equity, warmup=fold_warmup)
            results.append(result)
            logger.info(
                "Walk-forward fold %d/%d: bars %d-%d → %s",
                fold_idx + 1, n_splits, oos_start, oos_end, result.summary(),
            )
        return results
