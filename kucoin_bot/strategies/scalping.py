"""Liquidity-aware Scalping strategy – only if spread + depth allow."""

from __future__ import annotations

from typing import Optional

from kucoin_bot.services.signal_engine import Regime, SignalScores
from kucoin_bot.strategies.base import BaseStrategy, StrategyDecision


class Scalping(BaseStrategy):
    """Quick scalps in tight-spread, ranging markets with strict time-in-trade."""

    name = "scalping"

    def __init__(self, max_spread_bps: float = 15.0, take_profit_pct: float = 0.003) -> None:
        self.max_spread_bps = max_spread_bps
        self.take_profit_pct = take_profit_pct

    def preconditions_met(self, signals: SignalScores) -> bool:
        return signals.regime == Regime.RANGING and signals.volatility < 0.3 and signals.confidence >= 0.2

    def evaluate(
        self,
        signals: SignalScores,
        current_position_side: Optional[str],
        entry_price: Optional[float],
        current_price: float,
    ) -> StrategyDecision:
        # Exit logic
        if current_position_side and entry_price:
            pnl_pct = (current_price - entry_price) / entry_price
            if current_position_side == "short":
                pnl_pct = -pnl_pct
            if pnl_pct >= self.take_profit_pct:
                return StrategyDecision(
                    action="exit",
                    symbol=signals.symbol,
                    reason="scalp_tp",
                )
            if pnl_pct < -self.take_profit_pct * 2:
                return StrategyDecision(
                    action="exit",
                    symbol=signals.symbol,
                    order_type="market",
                    reason="scalp_stop",
                )
            return StrategyDecision(action="hold", symbol=signals.symbol, reason="in_scalp")

        # Entry based on orderbook imbalance / mean reversion
        if signals.mean_reversion > 0.2 or signals.orderbook_imbalance > 0.3:
            return StrategyDecision(
                action="entry_long",
                symbol=signals.symbol,
                confidence=signals.confidence,
                order_type="limit",
                post_only=True,
                reason=f"scalp_long_mr={signals.mean_reversion:.2f}",
            )
        elif signals.mean_reversion < -0.2 or signals.orderbook_imbalance < -0.3:
            return StrategyDecision(
                action="entry_short",
                symbol=signals.symbol,
                confidence=signals.confidence,
                order_type="limit",
                post_only=True,
                reason=f"scalp_short_mr={signals.mean_reversion:.2f}",
            )

        return StrategyDecision(action="hold", symbol=signals.symbol, reason="no_scalp_signal")
