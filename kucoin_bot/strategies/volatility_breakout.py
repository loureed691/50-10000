"""Volatility Breakout strategy – fast momentum with strict stops."""

from __future__ import annotations

from typing import Optional

from kucoin_bot.services.signal_engine import Regime, SignalScores
from kucoin_bot.strategies.base import BaseStrategy, StrategyDecision


class VolatilityBreakout(BaseStrategy):
    """Trade volatility spikes with tight risk management."""

    name = "volatility_breakout"

    def __init__(self, momentum_min: float = 0.4, stop_pct: float = 0.015) -> None:
        self.momentum_min = momentum_min
        self.stop_pct = stop_pct

    def preconditions_met(self, signals: SignalScores) -> bool:
        return (
            signals.regime == Regime.HIGH_VOLATILITY
            and abs(signals.momentum) >= self.momentum_min
            and signals.volume_anomaly > 0.0
        )

    def evaluate(
        self,
        signals: SignalScores,
        current_position_side: Optional[str],
        entry_price: Optional[float],
        current_price: float,
    ) -> StrategyDecision:
        # Strict time/price stop for existing positions
        if current_position_side and entry_price:
            pnl_pct = (current_price - entry_price) / entry_price
            if current_position_side == "short":
                pnl_pct = -pnl_pct
            if pnl_pct < -self.stop_pct:
                return StrategyDecision(
                    action="exit",
                    symbol=signals.symbol,
                    order_type="market",
                    reason="vol_stop",
                )
            if pnl_pct > self.stop_pct * 2:
                return StrategyDecision(
                    action="exit",
                    symbol=signals.symbol,
                    reason="vol_take_profit",
                )
            return StrategyDecision(action="hold", symbol=signals.symbol, reason="in_breakout")

        # Entry on strong momentum + volume
        if signals.momentum > self.momentum_min:
            return StrategyDecision(
                action="entry_long",
                symbol=signals.symbol,
                confidence=signals.confidence,
                order_type="market",
                stop_price=current_price * (1 - self.stop_pct),
                reason=f"vol_breakout_long_mom={signals.momentum:.2f}",
            )
        elif signals.momentum < -self.momentum_min:
            return StrategyDecision(
                action="entry_short",
                symbol=signals.symbol,
                confidence=signals.confidence,
                order_type="market",
                stop_price=current_price * (1 + self.stop_pct),
                reason=f"vol_breakout_short_mom={signals.momentum:.2f}",
            )

        return StrategyDecision(action="hold", symbol=signals.symbol, reason="no_breakout")
