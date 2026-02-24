"""Execution Engine – smart order routing, partial fills, slippage controls."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Optional, Set

from kucoin_bot.api.client import KuCoinClient
from kucoin_bot.services.market_data import MarketInfo
from kucoin_bot.services.risk_manager import RiskManager

logger = logging.getLogger(__name__)

# Maximum time (seconds) to poll an order before timing out
_ORDER_POLL_TIMEOUT = 120
_ORDER_POLL_INTERVAL = 2
_RETRY_BASE_DELAY = 0.5  # seconds – base delay between order retries
_BALANCE_RETRY_DELAY = 1.5  # seconds – extra delay for balance-related rejections


def _quantize(value: float, increment: float, rounding: str = ROUND_DOWN) -> float:
    """Quantize *value* to the nearest *increment* using exact Decimal math."""
    inc = Decimal(str(increment))
    return float(Decimal(str(value)).quantize(inc, rounding=rounding))


def _quantize_futures_size(raw_size: float, lot_size: int = 1) -> int:
    """Quantize a futures size to an integer number of contracts (lot-size aligned)."""
    if lot_size <= 0:
        lot_size = 1
    contracts = int(raw_size / lot_size) * lot_size
    return max(contracts, 0)


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
    reduce_only: bool = False  # set True for exit orders on futures


@dataclass
class OrderResult:
    """Result of an order attempt."""

    success: bool
    order_id: str = ""
    filled_qty: float = 0.0
    avg_price: float = 0.0
    message: str = ""
    status: str = ""  # pending / filled / partially_filled / cancelled


@dataclass
class ExecutionEngine:
    """Smart order routing with slippage and constraint handling."""

    client: KuCoinClient
    risk_mgr: RiskManager
    max_spread_bps: float = 50.0
    max_retries: int = 3
    poll_fills: bool = True  # whether to poll order status after placement
    _margin_mode_ready: Set[str] = field(default_factory=set, repr=False)
    _position_mode_ready: bool = field(default=False, repr=False)

    async def _ensure_position_mode(self) -> None:
        """Ensure the futures account is set to One-Way position mode.

        Only calls the API once per engine lifetime.  Failures are logged but
        do **not** prevent order placement (the order may still succeed if the
        mode already matches).
        """
        if self._position_mode_ready:
            return
        try:
            await self.client.change_position_mode(one_way=True)
            logger.info("Position mode set to One-Way")
        except Exception:
            logger.debug("Failed to switch position mode, order will proceed anyway", exc_info=True)
        self._position_mode_ready = True

    async def _ensure_margin_mode(self, symbol: str) -> None:
        """Ensure the futures symbol is set to ISOLATED margin mode.

        Only calls the API once per symbol per engine lifetime.  Failures are
        logged but do **not** prevent order placement (the order itself may
        still succeed if the mode already matches).
        """
        if symbol in self._margin_mode_ready:
            return
        try:
            await self.client.change_margin_mode(symbol, "ISOLATED")
            logger.info("Margin mode set to ISOLATED for %s", symbol)
        except Exception:
            logger.debug("Failed to switch margin mode for %s, order will proceed anyway", symbol, exc_info=True)
        self._margin_mode_ready.add(symbol)

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
                req.symbol,
                market.spread_bps,
                self.max_spread_bps,
            )
            return OrderResult(success=False, message="spread_too_wide")

        # Compute size from notional
        price = req.price or (market.last_price if market else 0)
        if price <= 0:
            return OrderResult(success=False, message="no_price")

        is_futures = market is not None and market.market_type == "futures"
        size = req.notional * req.leverage / price

        if market:
            if is_futures:
                # Futures: size is in integer contracts, use lot_size for rounding
                multiplier = market.contract_multiplier if market.contract_multiplier > 0 else 1.0
                raw_contracts = size / multiplier if multiplier != 0 else size
                lot_size = market.lot_size if market.lot_size > 0 else 1
                int_size = _quantize_futures_size(raw_contracts, lot_size)
                if int_size < lot_size:
                    return OrderResult(success=False, message="below_min_size")
                size = float(int_size)
            else:
                # Spot: respect min size and round to base_increment
                if size < market.base_min_size:
                    return OrderResult(success=False, message="below_min_size")
                if market.base_increment > 0:
                    size = _quantize(size, market.base_increment, ROUND_DOWN)

            if market.price_increment > 0:
                price = _quantize(price, market.price_increment, ROUND_HALF_UP)

        # Determine order type policy
        order_type = req.order_type

        client_oid = str(uuid.uuid4())

        for attempt in range(self.max_retries):
            try:
                # Route futures orders through the dedicated futures endpoint
                if is_futures:
                    await self._ensure_position_mode()
                    await self._ensure_margin_mode(req.symbol)
                    result = await self.client.place_futures_order(
                        symbol=req.symbol,
                        side=req.side,
                        size=int(size),
                        leverage=req.leverage,
                        order_type=order_type,
                        price=price if order_type == "limit" else None,
                        client_oid=client_oid,
                        reduce_only=req.reduce_only,
                    )
                else:
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
                        req.side,
                        req.symbol,
                        size,
                        order_type,
                        price,
                        oid,
                        req.reason,
                    )

                    # Poll for fill confirmation instead of assuming filled
                    if self.poll_fills and oid:
                        return await self._poll_order(oid, is_futures, price, size)

                    return OrderResult(success=True, order_id=oid, avg_price=price, filled_qty=size, status="pending")
                else:
                    msg = result.get("msg", str(result))
                    logger.warning("Order rejected (attempt %d): %s", attempt, msg)
                    # Back-off before next retry; longer for balance-related errors
                    # so that a recently-completed internal transfer can propagate.
                    if "insufficient balance" in msg.lower():
                        await asyncio.sleep(_BALANCE_RETRY_DELAY * (attempt + 1))
                    else:
                        await asyncio.sleep(_RETRY_BASE_DELAY * (attempt + 1))
            except Exception as exc:
                logger.error("Order error (attempt %d): %s", attempt, exc)
                await asyncio.sleep(_RETRY_BASE_DELAY * (attempt + 1))

        return OrderResult(success=False, message="max_retries_exceeded")

    async def _poll_order(
        self, order_id: str, is_futures: bool, expected_price: float, expected_qty: float
    ) -> OrderResult:
        """Poll order status until filled, partially filled, or timed out."""
        elapsed = 0.0
        while elapsed < _ORDER_POLL_TIMEOUT:
            try:
                if is_futures:
                    order = await self.client.get_futures_order(order_id)
                else:
                    order = await self.client.get_order(order_id)

                if not order:
                    break

                is_active = order.get("isActive", True)
                deal_size = float(order.get("dealSize", 0) or 0)
                deal_funds = float(order.get("dealFunds", 0) or 0)
                cancel_exist = order.get("cancelExist", False)

                if not is_active:
                    # Order is done (filled or cancelled)
                    if deal_size > 0:
                        avg_price = deal_funds / deal_size if deal_size > 0 else expected_price
                        status = "filled" if not cancel_exist else "partially_filled"
                        return OrderResult(
                            success=True,
                            order_id=order_id,
                            filled_qty=deal_size,
                            avg_price=avg_price,
                            status=status,
                        )
                    else:
                        return OrderResult(
                            success=False,
                            order_id=order_id,
                            message="order_cancelled",
                            status="cancelled",
                        )

                if deal_size > 0:
                    # Partial fill while still active
                    logger.debug("Order %s partially filled: %.6f", order_id, deal_size)

            except Exception:
                logger.warning("Error polling order %s", order_id, exc_info=True)

            await asyncio.sleep(_ORDER_POLL_INTERVAL)
            elapsed += _ORDER_POLL_INTERVAL

        # Timeout: return what we know
        logger.warning("Order %s poll timeout after %.0fs", order_id, elapsed)
        return OrderResult(
            success=True,
            order_id=order_id,
            avg_price=expected_price,
            filled_qty=expected_qty,
            message="poll_timeout",
            status="pending",
        )

    async def cancel_all(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders on both spot and futures."""
        cancelled = 0

        # Cancel spot orders
        try:
            spot_orders = await self.client.get_open_orders(symbol)
            for o in spot_orders:
                try:
                    await self.client.cancel_order(o["id"])
                    cancelled += 1
                except Exception:
                    logger.error("Failed to cancel spot order %s", o.get("id"), exc_info=True)
        except Exception:
            logger.error("Failed to fetch spot open orders", exc_info=True)

        # Cancel futures orders
        try:
            futures_orders = await self.client.get_futures_open_orders(symbol)
            for o in futures_orders:
                try:
                    await self.client.cancel_futures_order(o["id"])
                    cancelled += 1
                except Exception:
                    logger.error("Failed to cancel futures order %s", o.get("id"), exc_info=True)
        except Exception:
            logger.error("Failed to fetch futures open orders", exc_info=True)

        logger.info("Cancelled %d orders (spot + futures)", cancelled)
        return cancelled

    async def flatten_position(
        self, symbol: str, current_size: float, current_price: float, side: str, market: Optional[MarketInfo] = None
    ) -> OrderResult:
        """Close a position by placing an opposite market order."""
        close_side = "sell" if side == "long" else "buy"
        notional = abs(current_size * current_price)
        is_futures = market.market_type == "futures" if market else False
        return await self.execute(
            OrderRequest(
                symbol=symbol,
                side=close_side,
                notional=notional,
                order_type="market",
                reason="flatten_position",
                reduce_only=is_futures,
            ),
            market,
        )
