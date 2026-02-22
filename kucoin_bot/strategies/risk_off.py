"""Risk-Off / Capital Preservation strategy – minimal exposure, exit positions."""

from __future__ import annotations

from typing import Optional

from kucoin_bot.services.signal_engine import SignalScores
from kucoin_bot.strategies.base import BaseStrategy, StrategyDecision


class RiskOff(BaseStrategy):
    """Exit all positions and preserve capital."""

    name = "risk_off"

    def preconditions_met(self, signals: SignalScores) -> bool:
        # Always applicable as a fallback
        return True

    def evaluate(
        self,
        signals: SignalScores,
        current_position_side: Optional[str],
        entry_price: Optional[float],
        current_price: float,
    ) -> StrategyDecision:
        # If we have any position, exit it
        if current_position_side:
            return StrategyDecision(
                action="exit",
                symbol=signals.symbol,
                order_type="market",
                reason="risk_off_exit",
            )
        return StrategyDecision(action="hold", symbol=signals.symbol, reason="risk_off_flat")
