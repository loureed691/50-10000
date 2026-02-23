"""Hedge Mode strategy – hedge spot exposure with futures when risk rises."""

from __future__ import annotations

from typing import Optional

from kucoin_bot.services.signal_engine import Regime, SignalScores
from kucoin_bot.strategies.base import BaseStrategy, StrategyDecision


class HedgeMode(BaseStrategy):
    """Use futures to hedge spot when regime becomes risky."""

    name = "hedge"

    def preconditions_met(self, signals: SignalScores) -> bool:
        return signals.regime in (Regime.HIGH_VOLATILITY, Regime.TRENDING_DOWN) and signals.volatility > 0.4

    def evaluate(
        self,
        signals: SignalScores,
        current_position_side: Optional[str],
        entry_price: Optional[float],
        current_price: float,
    ) -> StrategyDecision:
        # If we have a long spot, open a short futures hedge
        if current_position_side == "long":
            if signals.regime == Regime.HIGH_VOLATILITY or signals.regime == Regime.TRENDING_DOWN:
                return StrategyDecision(
                    action="entry_short",
                    symbol=signals.symbol,
                    confidence=signals.confidence,
                    reason="hedge_short_futures",
                    order_type="market",
                )
            return StrategyDecision(action="hold", symbol=signals.symbol, reason="holding_hedge")

        # Unwind hedge when conditions improve
        if current_position_side == "short":
            if signals.regime not in (Regime.HIGH_VOLATILITY, Regime.TRENDING_DOWN):
                return StrategyDecision(
                    action="exit",
                    symbol=signals.symbol,
                    reason="hedge_unwind",
                )
            return StrategyDecision(action="hold", symbol=signals.symbol, reason="hedge_active")

        return StrategyDecision(action="hold", symbol=signals.symbol, reason="no_hedge_needed")
