"""Risk Manager – portfolio-level and per-trade risk controls with circuit breaker."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from kucoin_bot.config import RiskConfig
from kucoin_bot.reporting.metrics import METRICS
from kucoin_bot.services.signal_engine import SignalScores

logger = logging.getLogger(__name__)

# Minimum notional fraction of one unit at current price
_MIN_NOTIONAL_FACTOR = 1e-8


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
    contract_multiplier: float = 1.0  # futures: value per contract in base asset


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
    # Notional helpers
    # ------------------------------------------------------------------

    @staticmethod
    def position_notional(pos: PositionInfo) -> float:
        """Return the USD notional value of a position.

        For futures positions the notional is ``contracts * multiplier * price``
        (the multiplier converts contracts to base-asset units).  For spot /
        margin positions the multiplier is 1.0, so this reduces to
        ``size * price``.
        """
        return abs(pos.size * pos.contract_multiplier * pos.current_price)

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
        total = sum(self.position_notional(p) for p in self.positions.values())
        exposure_pct = total / self.current_equity * 100
        return exposure_pct >= self.config.max_total_exposure_pct

    def check_circuit_breaker(self) -> bool:
        """Evaluate all circuit-breaker conditions. Activates if breached."""
        if not self.config.circuit_breaker_enabled:
            return False
        if self.circuit_breaker_active:
            return True
        if self.check_daily_loss() or self.check_drawdown() or self.check_total_exposure():
            self.circuit_breaker_active = True
            self._cb_triggered_at = time.time()
            logger.critical("CIRCUIT BREAKER ACTIVATED – stopping all trading")
            return True
        return False

    def check_correlated_exposure(self, symbols: List[str], prospective_notional: float = 0.0) -> bool:
        """Returns True if total exposure for the given correlated symbol group exceeds limit.

        Pass a list of symbols sharing the same base asset (e.g. all BTC pairs).
        ``prospective_notional`` (USD) can be provided to include a pending entry
        that is not yet recorded in ``self.positions``, preventing the same-cycle
        bypass where multiple correlated entries could all pass the check before
        any fill updates positions.
        """
        if self.current_equity <= 0:
            return False
        total = (
            sum(self.position_notional(p) for sym, p in self.positions.items() if sym in symbols)
            + prospective_notional
        )
        exposure_pct = total / self.current_equity * 100
        return exposure_pct >= self.config.max_correlated_exposure_pct

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def compute_position_size(
        self,
        symbol: str,
        price: float,
        volatility: float,
        signals: SignalScores,
        leverage: float = 1.0,
    ) -> float:
        """Volatility-adjusted position sizing with hard caps.

        Returns the max USD notional to allocate.
        """
        if self.circuit_breaker_active or self.current_equity <= 0:
            return 0.0

        if price <= 0:
            return 0.0

        # Per-position risk cap.  Volatility and confidence are already
        # captured in the allocation weight produced by PortfolioManager, so
        # we intentionally do NOT scale down here to avoid double-penalising
        # small-account futures orders.
        max_risk_usd = self.current_equity * (self.config.max_per_position_risk_pct / 100)

        notional = max_risk_usd

        # Ensure doesn't blow total exposure
        existing_exposure = sum(self.position_notional(p) for p in self.positions.values())
        max_total = self.current_equity * (self.config.max_total_exposure_pct / 100)
        remaining = max(0, max_total - existing_exposure)
        # remaining is in exposure units; convert to margin by dividing by leverage
        notional = float(min(notional, remaining / max(leverage, 1.0)))

        # Skip if notional too small for one unit at current price
        if notional < price * _MIN_NOTIONAL_FACTOR:
            return 0.0

        return notional

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
            # Preserve existing stop_price when incoming stop_price is None
            existing = self.positions.get(symbol)
            if existing is not None and info.stop_price is None and existing.stop_price is not None:
                info.stop_price = existing.stop_price
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

    def check_squeeze_risk(self, signals: SignalScores) -> bool:
        """Return True if short squeeze risk is elevated.

        Delegates to the shared squeeze-risk heuristic used by :class:`SideSelector`.
        Exposed here so live code can call ``risk_mgr.check_squeeze_risk(signals)``
        without importing ``SideSelector`` directly.

        Squeeze risk is flagged when:
        - Recent volatility spike (normalized ATR > threshold).
        - Strong positive momentum burst coincides with a volume anomaly.
        """
        # Import here to avoid circular imports; SideSelector does not import RiskManager.
        from kucoin_bot.services.side_selector import SideSelector

        return SideSelector()._squeeze_risk_high(signals)

    def get_risk_summary(self) -> dict:
        total_exposure = sum(self.position_notional(p) for p in self.positions.values())
        dd = 0.0
        if self.peak_equity > 0:
            dd = (self.peak_equity - self.current_equity) / self.peak_equity * 100
        # Emit metrics for external monitoring
        METRICS.set("equity_usdt", self.current_equity)
        METRICS.set("daily_pnl_usdt", self.daily_pnl)
        METRICS.set("drawdown_pct", round(dd, 2))
        METRICS.set("total_exposure_usdt", round(total_exposure, 2))
        METRICS.set("open_positions", len(self.positions))
        METRICS.set("circuit_breaker_active", 1.0 if self.circuit_breaker_active else 0.0)
        return {
            "equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "daily_pnl": self.daily_pnl,
            "drawdown_pct": round(dd, 2),
            "total_exposure": round(total_exposure, 2),
            "positions": len(self.positions),
            "circuit_breaker": self.circuit_breaker_active,
        }
