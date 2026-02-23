"""Tests for Execution Engine: sizing, quantization, cancel_all, order lifecycle."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kucoin_bot.config import RiskConfig
from kucoin_bot.services.execution import (
    ExecutionEngine,
    OrderRequest,
    OrderResult,
    _quantize,
    _quantize_futures_size,
)
from kucoin_bot.services.market_data import MarketInfo
from kucoin_bot.services.risk_manager import RiskManager


class TestQuantize:
    """Test spot size/price rounding."""

    def test_quantize_round_down(self) -> None:
        assert _quantize(1.23456, 0.001) == 1.234

    def test_quantize_round_down_exact(self) -> None:
        assert _quantize(1.0, 0.1) == 1.0

    def test_quantize_small_increment(self) -> None:
        assert _quantize(0.000123456, 0.00001) == 0.00012

    def test_quantize_price_round_half_up(self) -> None:
        from decimal import ROUND_HALF_UP

        assert _quantize(100.125, 0.01, ROUND_HALF_UP) == 100.13


class TestQuantizeFuturesSize:
    """Test futures contract sizing."""

    def test_basic_lot_size(self) -> None:
        assert _quantize_futures_size(10.5, lot_size=1) == 10

    def test_lot_size_multiple(self) -> None:
        assert _quantize_futures_size(25.0, lot_size=10) == 20

    def test_below_lot_size_returns_min(self) -> None:
        assert _quantize_futures_size(0.5, lot_size=1) == 1

    def test_zero_lot_size_defaults(self) -> None:
        assert _quantize_futures_size(5.0, lot_size=0) == 5


class TestExecutionEngine:
    """Test execution engine with mocked client."""

    def _make_engine(self, poll_fills: bool = False) -> tuple[ExecutionEngine, AsyncMock]:
        client = AsyncMock()
        risk_mgr = RiskManager(config=RiskConfig())
        risk_mgr.update_equity(10000)
        engine = ExecutionEngine(client=client, risk_mgr=risk_mgr, poll_fills=poll_fills)
        return engine, client

    @pytest.mark.asyncio
    async def test_spot_order_placement(self) -> None:
        engine, client = self._make_engine()
        client.place_order.return_value = {"code": "200000", "data": {"orderId": "spot-123"}}

        market = MarketInfo(
            symbol="BTC-USDT",
            base="BTC",
            quote="USDT",
            base_min_size=0.0001,
            base_increment=0.0001,
            price_increment=0.01,
            last_price=50000.0,
        )
        req = OrderRequest(symbol="BTC-USDT", side="buy", notional=100.0, order_type="market")
        result = await engine.execute(req, market)

        assert result.success
        assert result.order_id == "spot-123"
        client.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_futures_order_uses_int_size(self) -> None:
        engine, client = self._make_engine()
        client.place_futures_order.return_value = {"code": "200000", "data": {"orderId": "fut-456"}}

        market = MarketInfo(
            symbol="XBTUSDTM",
            base="BTC",
            quote="USDT",
            base_min_size=1.0,
            price_increment=0.1,
            last_price=50000.0,
            market_type="futures",
            contract_multiplier=0.001,
            lot_size=1,
        )
        req = OrderRequest(symbol="XBTUSDTM", side="buy", notional=500.0, order_type="market")
        result = await engine.execute(req, market)

        assert result.success
        # Check that size passed to futures is an int
        call_args = client.place_futures_order.call_args
        assert isinstance(call_args.kwargs["size"], int)

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_order(self) -> None:
        engine, client = self._make_engine()
        engine.risk_mgr.circuit_breaker_active = True

        req = OrderRequest(symbol="BTC-USDT", side="buy", notional=100.0)
        result = await engine.execute(req)
        assert not result.success
        assert result.message == "circuit_breaker_active"

    @pytest.mark.asyncio
    async def test_below_min_size_rejected(self) -> None:
        engine, client = self._make_engine()
        market = MarketInfo(
            symbol="BTC-USDT",
            base="BTC",
            quote="USDT",
            base_min_size=0.001,
            base_increment=0.001,
            last_price=50000.0,
        )
        # Notional too small to meet min size
        req = OrderRequest(symbol="BTC-USDT", side="buy", notional=0.01)
        result = await engine.execute(req, market)
        assert not result.success
        assert result.message == "below_min_size"

    @pytest.mark.asyncio
    async def test_cancel_all_spot_and_futures(self) -> None:
        engine, client = self._make_engine()
        client.get_open_orders.return_value = [{"id": "s1"}, {"id": "s2"}]
        client.cancel_order.return_value = {}
        client.get_futures_open_orders.return_value = [{"id": "f1"}]
        client.cancel_futures_order.return_value = {}

        count = await engine.cancel_all()
        assert count == 3
        assert client.cancel_order.call_count == 2
        assert client.cancel_futures_order.call_count == 1

    @pytest.mark.asyncio
    async def test_flatten_position_sets_reduce_only_for_futures(self) -> None:
        engine, client = self._make_engine()
        client.place_futures_order.return_value = {"code": "200000", "data": {"orderId": "flat-1"}}
        market = MarketInfo(
            symbol="XBTUSDTM",
            base="BTC",
            quote="USDT",
            base_min_size=1.0,
            price_increment=0.1,
            last_price=50000.0,
            market_type="futures",
            contract_multiplier=0.001,
            lot_size=1,
        )
        result = await engine.flatten_position("XBTUSDTM", 10.0, 50000.0, "long", market)
        assert result.success
        call_args = client.place_futures_order.call_args
        assert call_args.kwargs["reduce_only"] is True

    @pytest.mark.asyncio
    async def test_spread_too_wide_rejected(self) -> None:
        engine, client = self._make_engine()
        market = MarketInfo(
            symbol="BTC-USDT",
            base="BTC",
            quote="USDT",
            spread_bps=100.0,
            last_price=50000.0,
        )
        req = OrderRequest(symbol="BTC-USDT", side="buy", notional=100.0)
        result = await engine.execute(req, market)
        assert not result.success
        assert result.message == "spread_too_wide"
