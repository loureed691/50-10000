"""Trend Following strategy – breakout entries with trailing exits."""

from __future__ import annotations

from typing import Optional

from kucoin_bot.services.signal_engine import Regime, SignalScores
from kucoin_bot.strategies.base import BaseStrategy, StrategyDecision


class TrendFollowing(BaseStrategy):
    """Enter on strong trend, trail stops."""

    name = "trend_following"

    def __init__(self, momentum_threshold: float = 0.2, trail_pct: float = 0.02) -> None:
        self.momentum_threshold = momentum_threshold
        self.trail_pct = trail_pct

    def preconditions_met(self, signals: SignalScores) -> bool:
        return (
            signals.regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN)
            and signals.trend_strength >= 0.4
            and signals.confidence >= 0.3
        )

    def evaluate(
        self,
        signals: SignalScores,
        current_position_side: Optional[str],
        entry_price: Optional[float],
        current_price: float,
    ) -> StrategyDecision:
        # Determine direction
        if signals.momentum > self.momentum_threshold:
            desired_side = "long"
        elif signals.momentum < -self.momentum_threshold:
            desired_side = "short"
        else:
            if current_position_side:
                return StrategyDecision(action="hold", symbol=signals.symbol, reason="weak_momentum")
            return StrategyDecision(action="hold", symbol=signals.symbol, reason="no_signal")

        # Exit if holding wrong side
        if current_position_side and current_position_side != desired_side:
            return StrategyDecision(
                action="exit",
                symbol=signals.symbol,
                order_type="market",
                reason="regime_flip",
            )

        # Entry
        if not current_position_side:
            stop = current_price * (1 - self.trail_pct) if desired_side == "long" else current_price * (1 + self.trail_pct)
            tp = current_price * (1 + self.trail_pct * 3) if desired_side == "long" else current_price * (1 - self.trail_pct * 3)
            return StrategyDecision(
                action=f"entry_{desired_side}",
                symbol=signals.symbol,
                confidence=signals.confidence,
                stop_price=stop,
                take_profit=tp,
                reason=f"trend_{desired_side}_momentum={signals.momentum:.2f}",
            )

        # Trailing stop check
        if current_position_side == "long" and entry_price:
            trail_stop = max(entry_price, current_price) * (1 - self.trail_pct)
            if current_price < trail_stop:
                return StrategyDecision(
                    action="exit", symbol=signals.symbol, order_type="market", reason="trailing_stop",
                )
        elif current_position_side == "short" and entry_price:
            trail_stop = min(entry_price, current_price) * (1 + self.trail_pct)
            if current_price > trail_stop:
                return StrategyDecision(
                    action="exit", symbol=signals.symbol, order_type="market", reason="trailing_stop",
                )

        return StrategyDecision(action="hold", symbol=signals.symbol, reason="in_trend")
