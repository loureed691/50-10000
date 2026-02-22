"""Side selector: decide LONG, SHORT, or FLAT given regime, signals, and squeeze risk.

Used identically in backtest and live to guarantee consistent side selection.
Short trades are guarded by:

1. **Feasibility** – futures or margin required (configurable).
2. **Short squeeze risk filter** – blocks new shorts on volatility spikes, momentum
   bursts, or volume anomalies that suggest covering pressure.
3. **Crowded-short filter** – blocks shorts when funding rate is deeply negative
   (shorts are paying longs, indicating a crowded positioning).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kucoin_bot.services.signal_engine import Regime, SignalScores

logger = logging.getLogger(__name__)

# Short squeeze risk thresholds
_SQUEEZE_VOL_THRESHOLD = 0.6           # volatility spike blocks new shorts
_SQUEEZE_MOMENTUM_THRESHOLD = 0.5      # strong positive momentum burst
_SQUEEZE_VOLUME_ANOMALY_THRESHOLD = 2.5  # volume anomaly accompanying momentum burst

# Funding rate below which shorts are considered crowded
# (negative funding = shorts pay longs; deeply negative = very crowded)
_CROWDED_SHORT_FUNDING_THRESHOLD = -0.0003  # per 8-hour period


@dataclass
class SideDecision:
    """Output of the side selector."""

    side: str            # "long", "short", or "flat"
    reason: str = ""
    squeeze_risk: bool = False


class SideSelector:
    """Validate and determine trade side from regime + signals + market constraints.

    Args:
        allow_shorts: Master toggle for short trading.
        require_futures_for_short: If True, shorts are only allowed on
            ``"futures"`` or ``"margin"`` instruments.
    """

    def __init__(
        self,
        allow_shorts: bool = True,
        require_futures_for_short: bool = True,
    ) -> None:
        self.allow_shorts = allow_shorts
        self.require_futures_for_short = require_futures_for_short

    def select(
        self,
        signals: SignalScores,
        market_type: str = "spot",
        proposed_side: str | None = None,
    ) -> SideDecision:
        """Return the validated trade side.

        Args:
            signals: Current signal scores for the symbol.
            market_type: ``"spot"``, ``"futures"``, or ``"margin"``.
            proposed_side: If provided (e.g., from a strategy decision), that
                side is validated rather than derived from signals.
        """
        side = proposed_side if proposed_side is not None else self._derive_side(signals)

        if side == "flat":
            return SideDecision(side="flat", reason="no_signal")

        if side == "short":
            can_short = self.allow_shorts and (
                market_type in ("futures", "margin")
                or not self.require_futures_for_short
            )
            if not can_short:
                return SideDecision(side="flat", reason="short_not_available")

            if self._squeeze_risk_high(signals):
                logger.debug("Short squeeze risk blocked short for %s", signals.symbol)
                return SideDecision(side="flat", reason="squeeze_risk", squeeze_risk=True)

            if signals.funding_rate < _CROWDED_SHORT_FUNDING_THRESHOLD:
                logger.debug(
                    "Crowded short blocked for %s (funding=%.4f)",
                    signals.symbol,
                    signals.funding_rate,
                )
                return SideDecision(side="flat", reason="crowded_short")

        return SideDecision(side=side, reason=f"{side}_signal")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _derive_side(self, signals: SignalScores) -> str:
        """Derive preferred side from regime and signal features."""
        if signals.regime == Regime.TRENDING_UP and signals.momentum > 0.1:
            return "long"
        if signals.regime == Regime.RANGING and signals.mean_reversion > 0.3:
            return "long"
        if signals.regime == Regime.TRENDING_DOWN and signals.momentum < -0.1:
            return "short"
        if signals.regime == Regime.RANGING and signals.mean_reversion < -0.3:
            return "short"
        return "flat"

    def _squeeze_risk_high(self, signals: SignalScores) -> bool:
        """Return True if short squeeze risk conditions are present.

        Squeeze risk is flagged when:

        * Recent volatility is spiking (wide candle ranges).
        * Strong positive momentum burst coincides with a volume anomaly,
          suggesting aggressive buying / covering pressure.
        """
        if signals.volatility > _SQUEEZE_VOL_THRESHOLD:
            return True
        if (
            signals.momentum > _SQUEEZE_MOMENTUM_THRESHOLD
            and signals.volume_anomaly > _SQUEEZE_VOLUME_ANOMALY_THRESHOLD
        ):
            return True
        return False
