"""Mean Reversion strategy – Bollinger band entries with reversion exits."""

from __future__ import annotations

from typing import Optional

from kucoin_bot.services.signal_engine import Regime, SignalScores
from kucoin_bot.strategies.base import BaseStrategy, StrategyDecision


class MeanReversion(BaseStrategy):
    """Trade reversions in ranging or weak-trend markets."""

    name = "mean_reversion"

    def __init__(self, reversion_threshold: float = 0.4, exit_threshold: float = 0.1) -> None:
        self.reversion_threshold = reversion_threshold
        self.exit_threshold = exit_threshold

    def preconditions_met(self, signals: SignalScores) -> bool:
        # Allow mean reversion when ranging OR when trend is weak with extreme reversion
        if signals.regime == Regime.RANGING and signals.volatility < 0.5:
            return abs(signals.mean_reversion) >= self.reversion_threshold
        # Also allow at regime edges: weak trend + very extreme reversion signal
        if signals.trend_strength < 0.4 and abs(signals.mean_reversion) >= self.reversion_threshold * 1.25:
            return signals.volatility < 0.6
        return False

    def evaluate(
        self,
        signals: SignalScores,
        current_position_side: Optional[str],
        entry_price: Optional[float],
        current_price: float,
    ) -> StrategyDecision:
        # Exit if reversion target reached (proportional to entry signal strength)
        if current_position_side and entry_price:
            if current_position_side == "long" and signals.mean_reversion < self.exit_threshold:
                return StrategyDecision(
                    action="exit",
                    symbol=signals.symbol,
                    reason="reversion_target_reached",
                )
            if current_position_side == "short" and signals.mean_reversion > -self.exit_threshold:
                return StrategyDecision(
                    action="exit",
                    symbol=signals.symbol,
                    reason="reversion_target_reached",
                )
            # Volatility-adjusted stop loss: wider stops for deeper reversions
            stop_pct = 0.03 * max(1.0, abs(signals.mean_reversion) / self.reversion_threshold)
            pnl_pct = (current_price - entry_price) / entry_price
            if current_position_side == "long" and pnl_pct < -stop_pct:
                return StrategyDecision(
                    action="exit",
                    symbol=signals.symbol,
                    order_type="market",
                    reason="stop_loss",
                )
            if current_position_side == "short" and pnl_pct > stop_pct:
                return StrategyDecision(
                    action="exit",
                    symbol=signals.symbol,
                    order_type="market",
                    reason="stop_loss",
                )
            return StrategyDecision(action="hold", symbol=signals.symbol, reason="waiting_reversion")

        # Entry
        if signals.mean_reversion > self.reversion_threshold:
            # Oversold -> buy
            stop = current_price * (1 - 0.03 * max(1.0, signals.mean_reversion / self.reversion_threshold))
            return StrategyDecision(
                action="entry_long",
                symbol=signals.symbol,
                confidence=signals.confidence,
                stop_price=stop,
                reason=f"mean_reversion_long={signals.mean_reversion:.2f}",
            )
        elif signals.mean_reversion < -self.reversion_threshold:
            # Overbought -> sell
            stop = current_price * (1 + 0.03 * max(1.0, abs(signals.mean_reversion) / self.reversion_threshold))
            return StrategyDecision(
                action="entry_short",
                symbol=signals.symbol,
                confidence=signals.confidence,
                stop_price=stop,
                reason=f"mean_reversion_short={signals.mean_reversion:.2f}",
            )

        return StrategyDecision(action="hold", symbol=signals.symbol, reason="no_reversion_signal")
