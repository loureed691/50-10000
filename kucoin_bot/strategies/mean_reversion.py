"""Mean Reversion strategy – Bollinger band entries with reversion exits."""

from __future__ import annotations

from typing import Optional

from kucoin_bot.services.signal_engine import Regime, SignalScores
from kucoin_bot.strategies.base import BaseStrategy, StrategyDecision


class MeanReversion(BaseStrategy):
    """Trade reversions in ranging markets."""

    name = "mean_reversion"

    def __init__(self, reversion_threshold: float = 0.4, exit_threshold: float = 0.1) -> None:
        self.reversion_threshold = reversion_threshold
        self.exit_threshold = exit_threshold

    def preconditions_met(self, signals: SignalScores) -> bool:
        return (
            signals.regime == Regime.RANGING
            and signals.volatility < 0.5
            and abs(signals.mean_reversion) >= self.reversion_threshold
        )

    def evaluate(
        self,
        signals: SignalScores,
        current_position_side: Optional[str],
        entry_price: Optional[float],
        current_price: float,
    ) -> StrategyDecision:
        # Exit if reversion target reached
        if current_position_side and entry_price:
            if current_position_side == "long" and signals.mean_reversion < self.exit_threshold:
                return StrategyDecision(
                    action="exit", symbol=signals.symbol, reason="reversion_target_reached",
                )
            if current_position_side == "short" and signals.mean_reversion > -self.exit_threshold:
                return StrategyDecision(
                    action="exit", symbol=signals.symbol, reason="reversion_target_reached",
                )
            # Stop loss at 2x expected move
            pnl_pct = (current_price - entry_price) / entry_price
            if current_position_side == "long" and pnl_pct < -0.03:
                return StrategyDecision(
                    action="exit", symbol=signals.symbol, order_type="market", reason="stop_loss",
                )
            if current_position_side == "short" and pnl_pct > 0.03:
                return StrategyDecision(
                    action="exit", symbol=signals.symbol, order_type="market", reason="stop_loss",
                )
            return StrategyDecision(action="hold", symbol=signals.symbol, reason="waiting_reversion")

        # Entry
        if signals.mean_reversion > self.reversion_threshold:
            # Oversold -> buy
            stop = current_price * 0.97
            return StrategyDecision(
                action="entry_long",
                symbol=signals.symbol,
                confidence=signals.confidence,
                stop_price=stop,
                reason=f"mean_reversion_long={signals.mean_reversion:.2f}",
            )
        elif signals.mean_reversion < -self.reversion_threshold:
            # Overbought -> sell
            stop = current_price * 1.03
            return StrategyDecision(
                action="entry_short",
                symbol=signals.symbol,
                confidence=signals.confidence,
                stop_price=stop,
                reason=f"mean_reversion_short={signals.mean_reversion:.2f}",
            )

        return StrategyDecision(action="hold", symbol=signals.symbol, reason="no_reversion_signal")
