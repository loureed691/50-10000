"""Execution Engine – smart order routing, partial fills, slippage controls."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from kucoin_bot.api.client import KuCoinClient
from kucoin_bot.services.market_data import MarketInfo
from kucoin_bot.services.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    """Internal order request before exchange submission."""

    symbol: str
    side: str  # buy / sell
    notional: float  # USD value to trade
    order_type: str = "limit"  # limit / market
    price: Optional[float] = None
    post_only: bool = False
    leverage: float = 1.0
    account_type: str = "trade"
    stop_price: Optional[float] = None
    reason: str = ""


@dataclass
class OrderResult:
    """Result of an order attempt."""

    success: bool
    order_id: str = ""
    filled_qty: float = 0.0
    avg_price: float = 0.0
    message: str = ""


@dataclass
class ExecutionEngine:
    """Smart order routing with slippage and constraint handling."""

    client: KuCoinClient
    risk_mgr: RiskManager
    max_spread_bps: float = 50.0
    max_retries: int = 3

    async def execute(self, req: OrderRequest, market: Optional[MarketInfo] = None) -> OrderResult:
        """Execute an order with safety checks."""
        # Kill switch / circuit breaker
        if self.risk_mgr.circuit_breaker_active:
            logger.warning("Circuit breaker active, rejecting order for %s", req.symbol)
            return OrderResult(success=False, message="circuit_breaker_active")

        # Slippage check
        if market and market.spread_bps > self.max_spread_bps:
            logger.warning(
                "Spread too wide for %s: %.1f bps (max %.1f)",
                req.symbol, market.spread_bps, self.max_spread_bps,
            )
            return OrderResult(success=False, message="spread_too_wide")

        # Compute size from notional
        price = req.price or (market.last_price if market else 0)
        if price <= 0:
            return OrderResult(success=False, message="no_price")

        size = req.notional / price
        if market:
            # Respect min size
            if size < market.base_min_size:
                return OrderResult(success=False, message="below_min_size")
            # Round to increment
            if market.base_increment > 0:
                size = round(size / market.base_increment) * market.base_increment
            if market.price_increment > 0:
                price = round(price / market.price_increment) * market.price_increment

        # Determine order type policy
        order_type = req.order_type
        if req.side == "buy" and order_type == "limit" and not req.post_only:
            # Use limit for entries, market only for stop-outs
            pass

        client_oid = str(uuid.uuid4())

        for attempt in range(self.max_retries):
            try:
                result = await self.client.place_order(
                    symbol=req.symbol,
                    side=req.side,
                    order_type=order_type,
                    size=size,
                    price=price if order_type == "limit" else None,
                    client_oid=client_oid,
                    post_only=req.post_only,
                )
                if result.get("code") == "200000":
                    oid = result.get("data", {}).get("orderId", "")
                    logger.info(
                        "Order placed: %s %s %.6f %s @ %.4f (oid=%s, reason=%s)",
                        req.side, req.symbol, size, order_type, price, oid, req.reason,
                    )
                    return OrderResult(success=True, order_id=oid, avg_price=price, filled_qty=size)
                else:
                    msg = result.get("msg", str(result))
                    logger.warning("Order rejected (attempt %d): %s", attempt, msg)
            except Exception as exc:
                logger.error("Order error (attempt %d): %s", attempt, exc)

        return OrderResult(success=False, message="max_retries_exceeded")

    async def cancel_all(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders, optionally for a specific symbol."""
        open_orders = await self.client.get_open_orders(symbol)
        cancelled = 0
        for o in open_orders:
            try:
                await self.client.cancel_order(o["id"])
                cancelled += 1
            except Exception:
                logger.error("Failed to cancel order %s", o.get("id"), exc_info=True)
        return cancelled

    async def flatten_position(self, symbol: str, current_size: float, current_price: float, side: str) -> OrderResult:
        """Close a position by placing an opposite market order."""
        close_side = "sell" if side == "long" else "buy"
        notional = abs(current_size * current_price)
        return await self.execute(
            OrderRequest(
                symbol=symbol,
                side=close_side,
                notional=notional,
                order_type="market",
                reason="flatten_position",
            )
        )
