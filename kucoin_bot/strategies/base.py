"""Base strategy interface – all strategies implement this protocol."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

from kucoin_bot.services.signal_engine import SignalScores


@dataclass
class StrategyDecision:
    """Output of a strategy evaluation."""

    action: str  # "entry_long", "entry_short", "exit", "hold"
    symbol: str = ""
    confidence: float = 0.0
    order_type: str = "limit"  # limit / market
    post_only: bool = False
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    notional: float = 0.0
    leverage: float = 1.0


class BaseStrategy(abc.ABC):
    """Abstract base for all trading strategies."""

    name: str = "base"

    @abc.abstractmethod
    def preconditions_met(self, signals: SignalScores) -> bool:
        """Return True if this strategy is applicable."""
        ...

    @abc.abstractmethod
    def evaluate(
        self,
        signals: SignalScores,
        current_position_side: Optional[str],
        entry_price: Optional[float],
        current_price: float,
    ) -> StrategyDecision:
        """Evaluate the strategy and return a decision."""
        ...

    def __repr__(self) -> str:
        return f"<Strategy:{self.name}>"
