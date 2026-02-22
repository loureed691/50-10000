"""Unified cost model: fees, slippage, funding, and borrow cost estimation.

Shared by both backtest engine and live trading pipeline to ensure the same
cost accounting is applied everywhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# KuCoin default fee tiers (Level 1)
DEFAULT_MAKER_FEE = 0.001     # 0.1 %
DEFAULT_TAKER_FEE = 0.001     # 0.1 %
DEFAULT_SLIPPAGE_BPS = 5.0    # 0.05 % one-way

# Perpetual funding is settled every 8 hours on KuCoin.
# Typical neutral rate; heavily directional markets can see higher absolute values.
DEFAULT_FUNDING_RATE_PER_8H = 0.0001  # 0.01 % per 8-hour period

# Margin borrow interest on KuCoin (hourly rate)
DEFAULT_BORROW_RATE_PER_HOUR = 0.00003  # ~0.003 %/hr → ~0.072 %/day


@dataclass
class TradeCosts:
    """Decomposed round-trip cost breakdown in basis points."""

    fee_bps: float = 0.0        # entry + exit fees
    slippage_bps: float = 0.0   # entry + exit slippage
    funding_bps: float = 0.0    # futures funding cost while open
    borrow_bps: float = 0.0     # margin borrow interest while open

    @property
    def total_bps(self) -> float:
        return self.fee_bps + self.slippage_bps + self.funding_bps + self.borrow_bps

    def to_dict(self) -> dict:
        return {
            "fee_bps": round(self.fee_bps, 4),
            "slippage_bps": round(self.slippage_bps, 4),
            "funding_bps": round(self.funding_bps, 4),
            "borrow_bps": round(self.borrow_bps, 4),
            "total_bps": round(self.total_bps, 4),
        }


class CostModel:
    """Unified cost estimator shared by backtest and live trading.

    Implements the EV gate::

        trade_allowed = expected_edge_bps > total_cost_bps + safety_buffer_bps

    The ``safety_buffer_bps`` (default 10.0) is equivalent to ``min_ev_bps`` in
    :class:`~kucoin_bot.config.RiskConfig`, so default behaviour is identical to
    the previous inline gate.
    """

    def __init__(
        self,
        maker_fee: float = DEFAULT_MAKER_FEE,
        taker_fee: float = DEFAULT_TAKER_FEE,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
        funding_rate_per_8h: float = DEFAULT_FUNDING_RATE_PER_8H,
        borrow_rate_per_hour: float = DEFAULT_BORROW_RATE_PER_HOUR,
        safety_buffer_bps: float = 10.0,
    ) -> None:
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_bps = slippage_bps
        self.funding_rate_per_8h = funding_rate_per_8h
        self.borrow_rate_per_hour = borrow_rate_per_hour
        self.safety_buffer_bps = safety_buffer_bps

    def estimate(
        self,
        order_type: str = "taker",
        holding_hours: float = 8.0,
        is_futures: bool = False,
        is_margin_short: bool = False,
        live_funding_rate: float | None = None,
        position_side: str | None = None,
    ) -> TradeCosts:
        """Estimate full round-trip costs in basis points.

        Args:
            order_type: ``"maker"`` or ``"taker"`` – determines the fee tier.
            holding_hours: Expected holding period in hours.
            is_futures: If True, add estimated perpetual funding cost.
            is_margin_short: If True, add margin borrow interest cost.
            live_funding_rate: Override the default per-8-hour funding rate (as
                a decimal fraction, e.g. 0.0001 for 0.01 %).
            position_side: ``"long"`` or ``"short"``.  When provided, funding is
                applied directionally:

                * Positive rate → longs **pay**, shorts **receive** (negative cost).
                * Negative rate → longs **receive** (negative cost), shorts **pay**.

                When ``None`` (default), ``abs(rate)`` is used as a conservative
                worst-case estimate independent of direction.
        """
        fee_rate = self.taker_fee if order_type == "taker" else self.maker_fee
        fee_bps = fee_rate * 2 * 10_000           # round-trip (entry + exit)
        slippage_bps = self.slippage_bps * 2      # round-trip

        funding_bps = 0.0
        if is_futures:
            rate = live_funding_rate if live_funding_rate is not None else self.funding_rate_per_8h
            periods = holding_hours / 8.0
            if position_side is None:
                # Conservative: treat funding as a cost regardless of direction
                funding_bps = abs(rate) * periods * 10_000
            else:
                # Directional: positive rate means longs pay, shorts receive
                signed = rate * periods * 10_000
                funding_bps = signed if position_side == "long" else -signed

        borrow_bps = 0.0
        if is_margin_short:
            borrow_bps = self.borrow_rate_per_hour * holding_hours * 10_000

        return TradeCosts(
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            funding_bps=funding_bps,
            borrow_bps=borrow_bps,
        )

    def ev_gate(self, expected_bps: float, costs: TradeCosts) -> bool:
        """Return True if the trade has positive expected value after all costs.

        Gate passes when::

            expected_bps > costs.total_bps + safety_buffer_bps
        """
        return expected_bps > costs.total_bps + self.safety_buffer_bps
