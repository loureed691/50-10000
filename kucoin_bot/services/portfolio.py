"""Portfolio & Treasury Manager – risk budgets, rebalancing, internal transfers."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from kucoin_bot.api.client import KuCoinClient
from kucoin_bot.services.risk_manager import RiskManager
from kucoin_bot.services.signal_engine import Regime, SignalScores

logger = logging.getLogger(__name__)

# Allowed transfer routes (user-facing names; "futures" is normalised to
# "contract" inside KuCoinClient.inner_transfer before hitting the API).
_ALLOWED_TRANSFERS = {
    ("main", "trade"),
    ("trade", "main"),
    ("main", "futures"),
    ("futures", "main"),
    ("trade", "futures"),
    ("futures", "trade"),
}


@dataclass
class AllocationTarget:
    """Per-pair risk budget allocation."""

    symbol: str
    weight: float = 0.0  # 0-1 fraction of total risk budget
    strategy: str = "risk_off"
    max_leverage: float = 1.0


@dataclass
class PortfolioManager:
    """Allocates risk across pairs and manages internal transfers."""

    client: KuCoinClient
    risk_mgr: RiskManager
    allow_transfers: bool = True
    allocations: Dict[str, AllocationTarget] = field(default_factory=dict)
    _transfer_log: List[dict] = field(default_factory=list)
    _db_session_factory: Optional[Any] = None

    def compute_allocations(
        self,
        signals: Dict[str, SignalScores],
        universe: List[str],
    ) -> Dict[str, AllocationTarget]:
        """Allocate risk budgets across pairs based on signals/regime."""
        if self.risk_mgr.circuit_breaker_active:
            # Risk-off: zero allocation
            self.allocations = {s: AllocationTarget(symbol=s) for s in universe}
            return self.allocations

        # Score each pair
        scored: list[tuple[str, float, SignalScores]] = []
        for sym in universe:
            sig = signals.get(sym)
            if not sig or sig.confidence < 0.1:
                continue
            # Prefer higher confidence, lower vol; penalise high-vol regimes
            vol_floor = max(1.0 - sig.volatility, 0.2)
            # Regime bonus: trending regimes with alignment get a boost
            regime_bonus = 1.0
            if sig.regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN):
                if sig.trend_strength > 0.5:
                    regime_bonus = 1.2
            elif sig.regime == Regime.HIGH_VOLATILITY:
                regime_bonus = 0.7  # reduce allocation in chaotic regimes
            score = sig.confidence * vol_floor * regime_bonus
            scored.append((sym, score, sig))

        # Normalize weights
        total_score = sum(s[1] for s in scored) or 1.0
        allocs: Dict[str, AllocationTarget] = {}

        for sym, score, sig in scored:
            weight = score / total_score
            strategy = self._select_strategy(sig)
            lev = self.risk_mgr.compute_leverage(sig, sig.volatility)
            allocs[sym] = AllocationTarget(symbol=sym, weight=weight, strategy=strategy, max_leverage=lev)

        # Zero-weight the rest
        for sym in universe:
            if sym not in allocs:
                allocs[sym] = AllocationTarget(symbol=sym)

        self.allocations = allocs
        return allocs

    @staticmethod
    def _select_strategy(sig: SignalScores) -> str:
        """Select best strategy for the current regime."""
        if sig.regime == Regime.NEWS_SPIKE:
            return "risk_off"  # never trade into a news spike
        if sig.regime == Regime.LOW_LIQUIDITY:
            return "risk_off"  # avoid illiquid markets
        if sig.regime == Regime.HIGH_VOLATILITY:
            if abs(sig.momentum) > 0.5:
                return "volatility_breakout"
            return "risk_off"
        if sig.regime in (Regime.TRENDING_UP, Regime.TRENDING_DOWN):
            return "trend_following"
        if sig.regime == Regime.RANGING:
            if sig.mean_reversion > 0.3 or sig.mean_reversion < -0.3:
                return "mean_reversion"
            return "scalping"
        return "risk_off"

    # ------------------------------------------------------------------
    # Internal transfers
    # ------------------------------------------------------------------

    async def transfer_if_needed(
        self,
        currency: str,
        from_account: str,
        to_account: str,
        amount: float,
    ) -> Optional[str]:
        """Execute an internal transfer with safety checks."""
        if not self.allow_transfers:
            logger.warning("Internal transfers disabled, skipping transfer request")
            return None

        route = (from_account, to_account)
        if route not in _ALLOWED_TRANSFERS:
            logger.error("Transfer route %s not allowed", route)
            return None

        if amount <= 0:
            return None

        idempotency_key = str(uuid.uuid4())
        try:
            result = await self.client.inner_transfer(
                currency=currency,
                from_account=from_account,
                to_account=to_account,
                amount=amount,
                client_oid=idempotency_key,
            )

            # KuCoin may return HTTP 200 with a non-success code in the body.
            result_code = str(result.get("code", ""))
            if result_code != "200000":
                logger.error(
                    "Transfer rejected by KuCoin: code=%s msg=%s payload=%s",
                    result_code,
                    result.get("msg", ""),
                    {"from": from_account, "to": to_account, "currency": currency, "amount": amount},
                )
                return None

            self._transfer_log.append(
                {
                    "key": idempotency_key,
                    "from": from_account,
                    "to": to_account,
                    "currency": currency,
                    "amount": amount,
                    "result": result,
                }
            )
            # Persist transfer record to DB if available
            if self._db_session_factory is not None:
                try:
                    from kucoin_bot.models import TransferRecord

                    with self._db_session_factory() as session:
                        session.add(
                            TransferRecord(
                                idempotency_key=idempotency_key,
                                from_account=from_account,
                                to_account=to_account,
                                currency=currency,
                                amount=amount,
                                status="success",
                            )
                        )
                        session.commit()
                except Exception:
                    logger.warning("Failed to persist transfer record to DB", exc_info=True)
            logger.info(
                "Transfer %s %s from %s to %s (key=%s)",
                amount,
                currency,
                from_account,
                to_account,
                idempotency_key,
            )
            return idempotency_key
        except Exception:
            logger.error("Transfer failed", exc_info=True)
            return None

    async def wait_for_futures_balance(
        self,
        currency: str,
        min_available: float,
        timeout: float = 10.0,
        poll_interval: float = 1.0,
    ) -> bool:
        """Poll the futures account until *min_available* is reached or *timeout* expires."""
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            try:
                overview = await self.client.get_futures_account_overview(currency=currency)
                available = float(overview.get("availableBalance", 0))
                if available >= min_available:
                    return True
            except Exception:
                logger.debug("Polling futures balance failed, retrying", exc_info=True)
            await asyncio.sleep(poll_interval)
        logger.warning(
            "Futures balance did not reach %.4f %s within %.0fs",
            min_available,
            currency,
            timeout,
        )
        return False
