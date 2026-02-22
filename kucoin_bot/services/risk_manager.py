"""Risk Manager – portfolio-level and per-trade risk controls with circuit breaker."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from kucoin_bot.config import RiskConfig
from kucoin_bot.services.signal_engine import SignalScores

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Tracks a live position."""

    symbol: str
    side: str  # long / short
    size: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    leverage: float = 1.0
    account_type: str = "trade"  # trade / margin / futures
    unrealized_pnl: float = 0.0
    stop_price: Optional[float] = None


@dataclass
class RiskManager:
    """Enforces global and per-trade risk limits with circuit breaker."""

    config: RiskConfig
    positions: Dict[str, PositionInfo] = field(default_factory=dict)
    daily_pnl: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    circuit_breaker_active: bool = False
    _cb_triggered_at: float = 0.0

    # ------------------------------------------------------------------
    # Core checks
    # ------------------------------------------------------------------

    def check_daily_loss(self) -> bool:
        """Returns True if daily loss limit breached."""
        if self.peak_equity <= 0:
            return False
        loss_pct = abs(min(self.daily_pnl, 0)) / self.peak_equity * 100
        return loss_pct >= self.config.max_daily_loss_pct

    def check_drawdown(self) -> bool:
        """Returns True if max drawdown breached."""
        if self.peak_equity <= 0:
            return False
        dd = (self.peak_equity - self.current_equity) / self.peak_equity * 100
        return dd >= self.config.max_drawdown_pct

    def check_total_exposure(self) -> bool:
        """Returns True if total exposure exceeds limit."""
        if self.current_equity <= 0:
            return False
        total = sum(
            abs(p.size * p.current_price * p.leverage) for p in self.positions.values()
        )
        exposure_pct = total / self.current_equity * 100
        return exposure_pct >= self.config.max_total_exposure_pct

    def check_circuit_breaker(self) -> bool:
        """Evaluate all circuit-breaker conditions. Activates if breached."""
        if self.circuit_breaker_active:
            return True
        if self.check_daily_loss() or self.check_drawdown() or self.check_total_exposure():
            self.circuit_breaker_active = True
            self._cb_triggered_at = time.time()
            logger.critical("CIRCUIT BREAKER ACTIVATED – stopping all trading")
            return True
        return False

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def compute_position_size(
        self,
        symbol: str,
        price: float,
        volatility: float,
        signals: SignalScores,
    ) -> float:
        """Volatility-adjusted position sizing with hard caps.

        Returns the max USD notional to allocate.
        """
        if self.circuit_breaker_active or self.current_equity <= 0:
            return 0.0

        # Per-position risk cap
        max_risk_usd = self.current_equity * (self.config.max_per_position_risk_pct / 100)

        # Volatility adjustment: reduce size when vol is high
        vol_factor = max(0.1, 1.0 - volatility)

        # Confidence adjustment
        conf_factor = max(0.1, signals.confidence)

        notional = max_risk_usd * vol_factor * conf_factor

        # Ensure doesn't blow total exposure
        existing_exposure = sum(
            abs(p.size * p.current_price * p.leverage) for p in self.positions.values()
        )
        max_total = self.current_equity * (self.config.max_total_exposure_pct / 100)
        remaining = max(0, max_total - existing_exposure)

        return min(notional, remaining)

    def compute_leverage(self, signals: SignalScores, volatility: float) -> float:
        """Dynamic leverage: only when confidence high, vol low, drawdown low."""
        if self.circuit_breaker_active:
            return 1.0

        # Conditions for leverage > 1
        if signals.confidence < 0.6:
            return 1.0
        if volatility > 0.5:
            return 1.0
        if self.peak_equity > 0:
            dd_pct = (self.peak_equity - self.current_equity) / self.peak_equity * 100
            if dd_pct > self.config.max_drawdown_pct * 0.5:
                return 1.0

        # Scale leverage with confidence
        lev = 1.0 + (signals.confidence - 0.5) * 4  # max ~3x at confidence=1
        return min(lev, self.config.max_leverage)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def update_position(self, symbol: str, info: PositionInfo) -> None:
        if info.size == 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = info

    def update_equity(self, equity: float) -> None:
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

    def record_pnl(self, pnl: float) -> None:
        self.daily_pnl += pnl

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0
        # Reset circuit breaker after daily reset if conditions allow
        if self.circuit_breaker_active:
            if not self.check_drawdown():
                self.circuit_breaker_active = False
                logger.info("Circuit breaker reset after daily review")

    def get_risk_summary(self) -> dict:
        total_exposure = sum(
            abs(p.size * p.current_price * p.leverage) for p in self.positions.values()
        )
        dd = 0.0
        if self.peak_equity > 0:
            dd = (self.peak_equity - self.current_equity) / self.peak_equity * 100
        return {
            "equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "daily_pnl": self.daily_pnl,
            "drawdown_pct": round(dd, 2),
            "total_exposure": round(total_exposure, 2),
            "positions": len(self.positions),
            "circuit_breaker": self.circuit_breaker_active,
        }
