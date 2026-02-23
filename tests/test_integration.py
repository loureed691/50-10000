"""Tests for execution engine and portfolio manager with mocked KuCoin API."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kucoin_bot.config import RiskConfig
from kucoin_bot.api.client import KuCoinClient
from kucoin_bot.services.execution import ExecutionEngine, OrderRequest
from kucoin_bot.services.market_data import MarketInfo
from kucoin_bot.services.risk_manager import RiskManager
from kucoin_bot.services.portfolio import PortfolioManager
from kucoin_bot.services.signal_engine import SignalScores, Regime


class TestExecutionEngine:
    def _make_engine(self, circuit_breaker: bool = False) -> ExecutionEngine:
        client = MagicMock(spec=KuCoinClient)
        client.place_order = AsyncMock(return_value={
            "code": "200000",
            "data": {"orderId": "test-order-123"},
        })
        client.get_open_orders = AsyncMock(return_value=[])
        client.cancel_order = AsyncMock(return_value={})
        risk_mgr = RiskManager(config=RiskConfig())
        risk_mgr.update_equity(10_000)
        risk_mgr.circuit_breaker_active = circuit_breaker
        return ExecutionEngine(client=client, risk_mgr=risk_mgr)

    @pytest.mark.asyncio
    async def test_execute_order(self):
        engine = self._make_engine()
        market = MarketInfo(
            symbol="BTC-USDT", base="BTC", quote="USDT",
            base_min_size=0.0001, base_increment=0.0001,
            price_increment=0.01, last_price=30000.0, spread_bps=5.0,
        )
        result = await engine.execute(
            OrderRequest(symbol="BTC-USDT", side="buy", notional=100, price=30000.0, reason="test"),
            market,
        )
        assert result.success is True
        assert result.order_id == "test-order-123"

    @pytest.mark.asyncio
    async def test_circuit_breaker_rejects(self):
        engine = self._make_engine(circuit_breaker=True)
        result = await engine.execute(
            OrderRequest(symbol="BTC-USDT", side="buy", notional=100),
        )
        assert result.success is False
        assert "circuit_breaker" in result.message

    @pytest.mark.asyncio
    async def test_spread_too_wide_rejects(self):
        engine = self._make_engine()
        market = MarketInfo(
            symbol="BTC-USDT", base="BTC", quote="USDT",
            last_price=30000.0, spread_bps=100.0,  # too wide
        )
        result = await engine.execute(
            OrderRequest(symbol="BTC-USDT", side="buy", notional=100, price=30000.0),
            market,
        )
        assert result.success is False
        assert "spread" in result.message

    @pytest.mark.asyncio
    async def test_size_quantized_to_increment(self):
        """Size must be exactly on the base_increment grid (no float artefacts)."""
        engine = self._make_engine()
        # ZEC-USDT style: base_increment = 0.0001, price ~30
        market = MarketInfo(
            symbol="ZEC-USDT", base="ZEC", quote="USDT",
            base_min_size=0.0001, base_increment=0.0001,
            price_increment=0.01, last_price=30.0, spread_bps=5.0,
        )
        # notional / price = 100 / 30 = 3.3333... -> should truncate to 3.3333
        result = await engine.execute(
            OrderRequest(symbol="ZEC-USDT", side="buy", notional=100, price=30.0, reason="test"),
            market,
        )
        assert result.success is True
        # Verify the size string passed to client has no float artefacts
        call_kwargs = engine.client.place_order.call_args
        sent_size = call_kwargs.kwargs.get("size") or call_kwargs[1].get("size")
        # str(float) must not produce trailing garbage digits
        assert "." in str(sent_size)
        parts = str(sent_size).split(".")
        assert len(parts[1]) <= 4  # 0.0001 -> max 4 decimal places

    @pytest.mark.asyncio
    async def test_size_truncated_not_rounded_up(self):
        """Size rounding must truncate (floor) to avoid exceeding available funds."""
        engine = self._make_engine()
        market = MarketInfo(
            symbol="ZEC-USDT", base="ZEC", quote="USDT",
            base_min_size=0.01, base_increment=0.01,
            price_increment=0.01, last_price=40.0, spread_bps=5.0,
        )
        # notional / price = 100 / 40 = 2.5 -> truncate to 2.50
        result = await engine.execute(
            OrderRequest(symbol="ZEC-USDT", side="buy", notional=99, price=40.0, reason="test"),
            market,
        )
        assert result.success is True
        call_kwargs = engine.client.place_order.call_args
        sent_size = call_kwargs.kwargs.get("size") or call_kwargs[1].get("size")
        # 99/40 = 2.475 -> should truncate to 2.47, not round to 2.48
        assert sent_size == 2.47


class TestQuantize:
    """Unit tests for the _quantize helper used by ExecutionEngine."""

    def test_basic_rounding(self):
        from kucoin_bot.services.execution import _quantize
        assert _quantize(1.23456, 0.0001) == 1.2345

    def test_no_float_artefacts(self):
        from kucoin_bot.services.execution import _quantize
        # This would fail with naive float math: round(1236 * 0.0001) != 0.1236
        result = _quantize(0.12369, 0.0001)
        assert result == 0.1236
        assert str(result) == "0.1236"

    def test_round_down(self):
        from kucoin_bot.services.execution import _quantize
        from decimal import ROUND_DOWN
        assert _quantize(2.999, 0.01, ROUND_DOWN) == 2.99

    def test_round_half_up(self):
        from kucoin_bot.services.execution import _quantize
        from decimal import ROUND_HALF_UP
        assert _quantize(2.995, 0.01, ROUND_HALF_UP) == 3.0

    def test_whole_number_increment(self):
        from kucoin_bot.services.execution import _quantize
        assert _quantize(3.7, 1) == 3.0


class TestPortfolioManager:
    def test_strategy_selection(self):
        client = MagicMock(spec=KuCoinClient)
        risk_mgr = RiskManager(config=RiskConfig())
        risk_mgr.update_equity(10_000)
        pm = PortfolioManager(client=client, risk_mgr=risk_mgr)

        signals = {
            "BTC-USDT": SignalScores(
                symbol="BTC-USDT", regime=Regime.TRENDING_UP,
                confidence=0.7, volatility=0.3, momentum=0.4, trend_strength=0.6,
            ),
        }
        allocs = pm.compute_allocations(signals, ["BTC-USDT"])
        assert "BTC-USDT" in allocs
        assert allocs["BTC-USDT"].strategy == "trend_following"

    def test_risk_off_on_circuit_breaker(self):
        client = MagicMock(spec=KuCoinClient)
        risk_mgr = RiskManager(config=RiskConfig())
        risk_mgr.update_equity(10_000)
        risk_mgr.circuit_breaker_active = True
        pm = PortfolioManager(client=client, risk_mgr=risk_mgr)

        allocs = pm.compute_allocations({}, ["BTC-USDT"])
        assert allocs["BTC-USDT"].weight == 0.0

    def test_high_volatility_pair_gets_nonzero_weight(self):
        """Pairs with volatility=1.0 should still get non-zero portfolio weight."""
        client = MagicMock(spec=KuCoinClient)
        risk_mgr = RiskManager(config=RiskConfig())
        risk_mgr.update_equity(10_000)
        pm = PortfolioManager(client=client, risk_mgr=risk_mgr)

        signals = {
            "ZEC-USDT": SignalScores(
                symbol="ZEC-USDT", regime=Regime.HIGH_VOLATILITY,
                confidence=0.55, volatility=1.0, momentum=1.0, trend_strength=0.68,
            ),
        }
        allocs = pm.compute_allocations(signals, ["ZEC-USDT"])
        assert allocs["ZEC-USDT"].weight > 0

    @pytest.mark.asyncio
    async def test_transfer_disabled(self):
        client = MagicMock(spec=KuCoinClient)
        risk_mgr = RiskManager(config=RiskConfig())
        pm = PortfolioManager(client=client, risk_mgr=risk_mgr, allow_transfers=False)
        result = await pm.transfer_if_needed("USDT", "main", "trade", 100)
        assert result is None

    @pytest.mark.asyncio
    async def test_transfer_bad_route(self):
        client = MagicMock(spec=KuCoinClient)
        risk_mgr = RiskManager(config=RiskConfig())
        pm = PortfolioManager(client=client, risk_mgr=risk_mgr, allow_transfers=True)
        result = await pm.transfer_if_needed("USDT", "unknown", "trade", 100)
        assert result is None


class TestCooldownLogic:
    """Verify cooldown allows first-ever entries immediately."""

    def test_first_entry_not_blocked_by_cooldown(self):
        """Symbols that have never been traded should not be blocked by cooldown."""
        last_entry_cycle: dict[str, int] = {}
        cooldown_cycles = 300  # 5 bars * 60 cycles

        # Simulate first cycle
        cycle = 1
        sym = "BTC-USDT"
        last_cycle = last_entry_cycle.get(sym, -cooldown_cycles)
        assert cycle - last_cycle >= cooldown_cycles, (
            "First-ever entry should not be blocked by cooldown"
        )

    def test_cooldown_blocks_recent_entry(self):
        """After an entry, the same symbol should be blocked for cooldown_cycles."""
        last_entry_cycle: dict[str, int] = {"BTC-USDT": 10}
        cooldown_cycles = 300

        cycle = 50
        sym = "BTC-USDT"
        last_cycle = last_entry_cycle.get(sym, -cooldown_cycles)
        assert cycle - last_cycle < cooldown_cycles, (
            "Recent entry should be in cooldown"
        )

    def test_cooldown_allows_after_expiry(self):
        """After cooldown expires, the symbol should be allowed to enter."""
        last_entry_cycle: dict[str, int] = {"BTC-USDT": 10}
        cooldown_cycles = 300

        cycle = 311
        sym = "BTC-USDT"
        last_cycle = last_entry_cycle.get(sym, -cooldown_cycles)
        assert cycle - last_cycle >= cooldown_cycles, (
            "Entry should be allowed after cooldown expires"
        )


class TestPaperMode:
    """Verify PAPER mode never calls the real order API."""

    @pytest.mark.asyncio
    async def test_paper_mode_does_not_call_place_order(self, monkeypatch):
        """In PAPER mode run_live() must NOT call client.place_order for entries."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch
        from kucoin_bot.config import BotConfig, RiskConfig
        from kucoin_bot.api.client import KuCoinClient
        from kucoin_bot.services.market_data import MarketDataService, MarketInfo
        from kucoin_bot.services.signal_engine import SignalScores, Regime
        from kucoin_bot.__main__ import run_live

        cfg = BotConfig(
            mode="PAPER",
            api_key="k", api_secret="s", api_passphrase="p",
            risk=RiskConfig(),
        )

        # Build a mock client that records place_order calls
        mock_client = MagicMock(spec=KuCoinClient)
        mock_client.start = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client.get_account_balance = AsyncMock(return_value=10_000.0)
        mock_client.place_order = AsyncMock(return_value={"code": "200000", "data": {"orderId": "real-order"}})
        mock_client.get_open_orders = AsyncMock(return_value=[])

        mock_market = MarketInfo(
            symbol="BTC-USDT", base="BTC", quote="USDT",
            last_price=30_000.0, spread_bps=5.0,
            base_min_size=0.0001, base_increment=0.0001, price_increment=0.01,
        )

        # KuCoin kline format: [time, open, close, high, low, volume, quote_volume]
        mock_klines = [
            [i, "30000", "30000", "30100", "29900", "100", "3000000"]
            for i in range(200)
        ]
        mock_mds = MagicMock(spec=MarketDataService)
        mock_mds.refresh_universe = AsyncMock()
        mock_mds.get_symbols = MagicMock(return_value=["BTC-USDT"])
        mock_mds.get_klines = AsyncMock(return_value=mock_klines)
        mock_mds.get_info = MagicMock(return_value=mock_market)

        # Use a simple context manager mock for init_db
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session_factory = MagicMock(return_value=mock_session)

        async def fake_sleep(_: float) -> None:
            raise asyncio.CancelledError()

        with patch("kucoin_bot.__main__.KuCoinClient", return_value=mock_client), \
             patch("kucoin_bot.__main__.MarketDataService", return_value=mock_mds), \
             patch("asyncio.sleep", side_effect=fake_sleep), \
             patch("kucoin_bot.models.init_db", return_value=mock_session_factory):
            try:
                await run_live(cfg)
            except asyncio.CancelledError:
                pass

        # PAPER mode must never call the real exchange order API
        mock_client.place_order.assert_not_called()


class TestComputeTotalEquity:
    """Tests for _compute_total_equity – multi-asset portfolio valuation."""

    @pytest.mark.asyncio
    async def test_usdt_only(self):
        """Only USDT balance → equity equals USDT amount."""
        from kucoin_bot.__main__ import _compute_total_equity
        from kucoin_bot.services.market_data import MarketDataService

        client = MagicMock(spec=KuCoinClient)
        client.get_accounts = AsyncMock(return_value=[
            {"currency": "USDT", "balance": "5000", "available": "5000"},
        ])

        mds = MagicMock(spec=MarketDataService)
        mds.get_info = MagicMock(return_value=None)

        equity = await _compute_total_equity(client, mds)
        assert equity == pytest.approx(5000.0)

    @pytest.mark.asyncio
    async def test_multi_asset(self):
        """USDT + BTC + ETH combined via market prices."""
        from kucoin_bot.__main__ import _compute_total_equity
        from kucoin_bot.services.market_data import MarketDataService, MarketInfo

        client = MagicMock(spec=KuCoinClient)
        client.get_accounts = AsyncMock(return_value=[
            {"currency": "USDT", "balance": "1000", "available": "1000"},
            {"currency": "BTC", "balance": "0.5", "available": "0.5"},
            {"currency": "ETH", "balance": "2", "available": "2"},
        ])

        btc_info = MarketInfo(symbol="BTC-USDT", base="BTC", quote="USDT", last_price=40_000.0)
        eth_info = MarketInfo(symbol="ETH-USDT", base="ETH", quote="USDT", last_price=2_000.0)

        def get_info(sym: str):
            return {"BTC-USDT": btc_info, "ETH-USDT": eth_info}.get(sym)

        mds = MagicMock(spec=MarketDataService)
        mds.get_info = MagicMock(side_effect=get_info)

        equity = await _compute_total_equity(client, mds)
        # 1000 USDT + 0.5 BTC * 40000 + 2 ETH * 2000 = 1000 + 20000 + 4000 = 25000
        assert equity == pytest.approx(25_000.0)

    @pytest.mark.asyncio
    async def test_zero_balance_skipped(self):
        """Assets with zero balance must not affect total equity."""
        from kucoin_bot.__main__ import _compute_total_equity
        from kucoin_bot.services.market_data import MarketDataService

        client = MagicMock(spec=KuCoinClient)
        client.get_accounts = AsyncMock(return_value=[
            {"currency": "USDT", "balance": "2000", "available": "2000"},
            {"currency": "BTC", "balance": "0", "available": "0"},
        ])

        mds = MagicMock(spec=MarketDataService)
        mds.get_info = MagicMock(return_value=None)

        equity = await _compute_total_equity(client, mds)
        assert equity == pytest.approx(2000.0)

    @pytest.mark.asyncio
    async def test_fallback_to_ticker_when_no_market_info(self):
        """Non-USDT asset falls back to live ticker when market_data has no price."""
        from kucoin_bot.__main__ import _compute_total_equity
        from kucoin_bot.services.market_data import MarketDataService

        client = MagicMock(spec=KuCoinClient)
        client.get_accounts = AsyncMock(return_value=[
            {"currency": "USDT", "balance": "500", "available": "500"},
            {"currency": "XRP", "balance": "1000", "available": "1000"},
        ])
        client.get_ticker = AsyncMock(return_value={"price": "0.5"})

        mds = MagicMock(spec=MarketDataService)
        mds.get_info = MagicMock(return_value=None)  # no cached info

        equity = await _compute_total_equity(client, mds)
        # 500 USDT + 1000 XRP * 0.5 = 500 + 500 = 1000
        assert equity == pytest.approx(1000.0)

    @pytest.mark.asyncio
    async def test_api_error_returns_zero(self):
        """API error in get_accounts must return 0.0 gracefully."""
        from kucoin_bot.__main__ import _compute_total_equity
        from kucoin_bot.services.market_data import MarketDataService

        client = MagicMock(spec=KuCoinClient)
        client.get_accounts = AsyncMock(side_effect=Exception("network error"))

        mds = MagicMock(spec=MarketDataService)

        equity = await _compute_total_equity(client, mds)
        assert equity == 0.0
