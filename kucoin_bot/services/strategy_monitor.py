"""Strategy Monitor: rolling per-module PnL tracking with automatic disabling.

Tracks net expectancy (PnL after costs) per strategy module over a rolling
window of trades.  Modules with persistently negative net expectancy are
automatically disabled to prevent the bot from repeating losing behaviour.

Usage::

    monitor = StrategyMonitor()
    monitor.record_trade("trend_following", pnl=12.5, cost=3.2)
    if monitor.is_enabled("trend_following"):
        # proceed with trade
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from typing import Deque, Dict

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW = 20  # rolling trade window for expectancy calculation
_DEFAULT_MIN_TRADES = 5  # minimum trades before auto-disable is considered


@dataclass
class ModuleStats:
    """Rolling statistics for one strategy module."""

    name: str
    pnl_window: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=_DEFAULT_WINDOW))
    cost_window: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=_DEFAULT_WINDOW))
    enabled: bool = True
    disabled_reason: str = ""

    @property
    def trade_count(self) -> int:
        return len(self.pnl_window)

    @property
    def net_expectancy(self) -> float:
        """Average net PnL (after costs) per trade over the rolling window."""
        if not self.pnl_window:
            return 0.0
        net = [p - c for p, c in zip(self.pnl_window, self.cost_window)]
        return sum(net) / len(net)


class StrategyMonitor:
    """Track rolling performance and auto-adjust risk budget per strategy module.

    Call :meth:`record_trade` after each closed round-trip trade.  Modules
    whose rolling net expectancy turns negative (after at least ``min_trades``
    observations) are automatically disabled.  A disabled module can be
    manually re-enabled via :meth:`enable`.

    Args:
        window: Number of recent trades to consider for expectancy calculation.
        min_trades: Minimum trades required before auto-disable can trigger.
    """

    def __init__(self, window: int = _DEFAULT_WINDOW, min_trades: int = _DEFAULT_MIN_TRADES) -> None:
        self.window = window
        self.min_trades = min_trades
        self._modules: Dict[str, ModuleStats] = {}

    def _get_or_create(self, module: str) -> ModuleStats:
        if module not in self._modules:
            self._modules[module] = ModuleStats(name=module)
        return self._modules[module]

    def record_trade(self, module: str, pnl: float, cost: float = 0.0) -> None:
        """Record a closed trade for a strategy module.

        Args:
            module: Strategy module name (e.g. ``"trend_following"``).
            pnl: Realised PnL (before cost subtraction).
            cost: Total cost (fees + slippage + funding + borrow) for this trade.
        """
        stats = self._get_or_create(module)
        stats.pnl_window.append(pnl)
        stats.cost_window.append(cost)
        self._evaluate(stats)

    def is_enabled(self, module: str) -> bool:
        """Return True if the module is currently allowed to trade."""
        return self._get_or_create(module).enabled

    def enable(self, module: str) -> None:
        """Manually re-enable a previously disabled module."""
        stats = self._get_or_create(module)
        stats.enabled = True
        stats.disabled_reason = ""
        logger.info("Strategy module '%s' re-enabled", module)

    def get_status(self) -> Dict[str, dict]:
        """Return a summary dict of all tracked module stats."""
        return {
            name: {
                "enabled": s.enabled,
                "trade_count": s.trade_count,
                "net_expectancy": round(s.net_expectancy, 4),
                "disabled_reason": s.disabled_reason,
            }
            for name, s in self._modules.items()
        }

    # ------------------------------------------------------------------

    def _evaluate(self, stats: ModuleStats) -> None:
        """Auto-disable the module if rolling net expectancy is negative."""
        if not stats.enabled:
            return
        if stats.trade_count < self.min_trades:
            return
        if stats.net_expectancy < 0:
            stats.enabled = False
            stats.disabled_reason = (
                f"negative_net_expectancy={stats.net_expectancy:.4f} " f"over_last_{stats.trade_count}_trades"
            )
            logger.warning(
                "Auto-disabled strategy module '%s': %s",
                stats.name,
                stats.disabled_reason,
            )
