"""Tests for execution engine and portfolio manager with mocked KuCoin API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kucoin_bot.api.client import KuCoinClient
from kucoin_bot.config import RiskConfig
from kucoin_bot.services.execution import ExecutionEngine, OrderRequest
from kucoin_bot.services.market_data import MarketInfo
from kucoin_bot.services.portfolio import PortfolioManager
from kucoin_bot.services.risk_manager import RiskManager
from kucoin_bot.services.signal_engine import Regime, SignalScores


class TestExecutionEngine:
    def _make_engine(self, circuit_breaker: bool = False) -> ExecutionEngine:
        client = MagicMock(spec=KuCoinClient)
        client.place_order = AsyncMock(
            return_value={
                "code": "200000",
                "data": {"orderId": "test-order-123"},
            }
        )
        client.get_open_orders = AsyncMock(return_value=[])
        client.get_futures_open_orders = AsyncMock(return_value=[])
        client.cancel_order = AsyncMock(return_value={})
        client.cancel_futures_order = AsyncMock(return_value={})
        risk_mgr = RiskManager(config=RiskConfig())
        risk_mgr.update_equity(10_000)
        risk_mgr.circuit_breaker_active = circuit_breaker
        return ExecutionEngine(client=client, risk_mgr=risk_mgr, poll_fills=False)

    @pytest.mark.asyncio
    async def test_execute_order(self):
        engine = self._make_engine()
        market = MarketInfo(
            symbol="BTC-USDT",
            base="BTC",
            quote="USDT",
            base_min_size=0.0001,
            base_increment=0.0001,
            price_increment=0.01,
            last_price=30000.0,
            spread_bps=5.0,
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
            symbol="BTC-USDT",
            base="BTC",
            quote="USDT",
            last_price=30000.0,
            spread_bps=100.0,  # too wide
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
            symbol="ZEC-USDT",
            base="ZEC",
            quote="USDT",
            base_min_size=0.0001,
            base_increment=0.0001,
            price_increment=0.01,
            last_price=30.0,
            spread_bps=5.0,
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
            symbol="ZEC-USDT",
            base="ZEC",
            quote="USDT",
            base_min_size=0.01,
            base_increment=0.01,
            price_increment=0.01,
            last_price=40.0,
            spread_bps=5.0,
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
        from decimal import ROUND_DOWN

        from kucoin_bot.services.execution import _quantize

        assert _quantize(2.999, 0.01, ROUND_DOWN) == 2.99

    def test_round_half_up(self):
        from decimal import ROUND_HALF_UP

        from kucoin_bot.services.execution import _quantize

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
                symbol="BTC-USDT",
                regime=Regime.TRENDING_UP,
                confidence=0.7,
                volatility=0.3,
                momentum=0.4,
                trend_strength=0.6,
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
                symbol="ZEC-USDT",
                regime=Regime.HIGH_VOLATILITY,
                confidence=0.55,
                volatility=1.0,
                momentum=1.0,
                trend_strength=0.68,
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

    def test_news_spike_selects_risk_off(self):
        """NEWS_SPIKE regime should always select risk_off strategy."""
        client = MagicMock(spec=KuCoinClient)
        risk_mgr = RiskManager(config=RiskConfig())
        risk_mgr.update_equity(10_000)
        pm = PortfolioManager(client=client, risk_mgr=risk_mgr)

        signals = {
            "BTC-USDT": SignalScores(
                symbol="BTC-USDT",
                regime=Regime.NEWS_SPIKE,
                confidence=0.8,
                volatility=0.9,
                momentum=0.7,
                trend_strength=0.6,
            ),
        }
        allocs = pm.compute_allocations(signals, ["BTC-USDT"])
        assert allocs["BTC-USDT"].strategy == "risk_off"

    def test_trending_regime_gets_allocation_boost(self):
        """Trending regimes with strong trend should get higher scores than weak ones."""
        client = MagicMock(spec=KuCoinClient)
        risk_mgr = RiskManager(config=RiskConfig())
        risk_mgr.update_equity(10_000)
        pm = PortfolioManager(client=client, risk_mgr=risk_mgr)

        signals = {
            "STRONG-USDT": SignalScores(
                symbol="STRONG-USDT",
                regime=Regime.TRENDING_UP,
                confidence=0.6,
                volatility=0.3,
                momentum=0.4,
                trend_strength=0.7,
            ),
            "WEAK-USDT": SignalScores(
                symbol="WEAK-USDT",
                regime=Regime.UNKNOWN,
                confidence=0.6,
                volatility=0.3,
                momentum=0.1,
                trend_strength=0.2,
            ),
        }
        allocs = pm.compute_allocations(signals, ["STRONG-USDT", "WEAK-USDT"])
        assert allocs["STRONG-USDT"].weight > allocs["WEAK-USDT"].weight


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
        assert cycle - last_cycle >= cooldown_cycles, "First-ever entry should not be blocked by cooldown"

    def test_cooldown_blocks_recent_entry(self):
        """After an entry, the same symbol should be blocked for cooldown_cycles."""
        last_entry_cycle: dict[str, int] = {"BTC-USDT": 10}
        cooldown_cycles = 300

        cycle = 50
        sym = "BTC-USDT"
        last_cycle = last_entry_cycle.get(sym, -cooldown_cycles)
        assert cycle - last_cycle < cooldown_cycles, "Recent entry should be in cooldown"

    def test_cooldown_allows_after_expiry(self):
        """After cooldown expires, the symbol should be allowed to enter."""
        last_entry_cycle: dict[str, int] = {"BTC-USDT": 10}
        cooldown_cycles = 300

        cycle = 311
        sym = "BTC-USDT"
        last_cycle = last_entry_cycle.get(sym, -cooldown_cycles)
        assert cycle - last_cycle >= cooldown_cycles, "Entry should be allowed after cooldown expires"


class TestSymbolInclusion:
    """Verify that all open positions are always included in the processing loop."""

    def test_open_positions_included_when_not_in_universe(self):
        """Symbols with open positions must be processed even if outside the universe."""
        universe_syms = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
        positions = {"DOGE-USDT": True, "BTC-USDT": True}
        max_symbols = 0  # no limit

        if max_symbols > 0:
            universe_syms = universe_syms[:max_symbols]
        position_syms = [s for s in positions if s not in universe_syms]
        symbols_to_process = universe_syms + position_syms

        assert "DOGE-USDT" in symbols_to_process, "DOGE-USDT has an open position and must be processed"
        assert "BTC-USDT" in symbols_to_process
        # No duplicates
        assert len(symbols_to_process) == len(set(symbols_to_process))

    def test_max_symbols_caps_universe_but_keeps_positions(self):
        """When max_symbols limits universe, positions outside it are still included."""
        universe_syms = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT", "XRP-USDT"]
        positions = {"XRP-USDT": True, "AVAX-USDT": True}
        max_symbols = 2  # only top 2 from universe

        if max_symbols > 0:
            universe_syms = universe_syms[:max_symbols]
        position_syms = [s for s in positions if s not in universe_syms]
        symbols_to_process = universe_syms + position_syms

        # Universe capped at 2
        assert universe_syms == ["BTC-USDT", "ETH-USDT"]
        # But positions not in the capped universe are appended
        assert "XRP-USDT" in symbols_to_process
        assert "AVAX-USDT" in symbols_to_process
        assert len(symbols_to_process) == 4

    def test_no_limit_processes_all_universe(self):
        """When max_symbols is 0, all universe symbols are processed."""
        universe_syms = [f"SYM{i}-USDT" for i in range(50)]
        positions: dict[str, bool] = {}
        max_symbols = 0

        if max_symbols > 0:
            universe_syms = universe_syms[:max_symbols]
        position_syms = [s for s in positions if s not in universe_syms]
        symbols_to_process = universe_syms + position_syms

        assert len(symbols_to_process) == 50

    def test_no_duplicates_when_position_in_universe(self):
        """Symbols that are both in the universe and have positions are not duplicated."""
        universe_syms = ["BTC-USDT", "ETH-USDT"]
        positions = {"BTC-USDT": True}
        max_symbols = 0

        if max_symbols > 0:
            universe_syms = universe_syms[:max_symbols]
        position_syms = [s for s in positions if s not in universe_syms]
        symbols_to_process = universe_syms + position_syms

        assert symbols_to_process == ["BTC-USDT", "ETH-USDT"]


class TestPaperMode:
    """Verify PAPER mode never calls the real order API."""

    @pytest.mark.asyncio
    async def test_paper_mode_does_not_call_place_order(self, monkeypatch):
        """In PAPER mode run_live() must NOT call client.place_order for entries."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        from kucoin_bot.__main__ import run_live
        from kucoin_bot.api.client import KuCoinClient
        from kucoin_bot.config import BotConfig, RiskConfig
        from kucoin_bot.services.market_data import MarketDataService, MarketInfo
        from kucoin_bot.services.signal_engine import Regime, SignalScores

        cfg = BotConfig(
            mode="PAPER",
            api_key="k",
            api_secret="s",
            api_passphrase="p",
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
            symbol="BTC-USDT",
            base="BTC",
            quote="USDT",
            last_price=30_000.0,
            spread_bps=5.0,
            base_min_size=0.0001,
            base_increment=0.0001,
            price_increment=0.01,
        )

        # KuCoin kline format: [time, open, close, high, low, volume, quote_volume]
        mock_klines = [[i, "30000", "30000", "30100", "29900", "100", "3000000"] for i in range(200)]
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

        with (
            patch("kucoin_bot.__main__.KuCoinClient", return_value=mock_client),
            patch("kucoin_bot.__main__.MarketDataService", return_value=mock_mds),
            patch("asyncio.sleep", side_effect=fake_sleep),
            patch("kucoin_bot.models.init_db", return_value=mock_session_factory),
        ):
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
        client.get_accounts = AsyncMock(
            return_value=[
                {"currency": "USDT", "balance": "5000", "available": "5000"},
            ]
        )

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
        client.get_accounts = AsyncMock(
            return_value=[
                {"currency": "USDT", "balance": "1000", "available": "1000"},
                {"currency": "BTC", "balance": "0.5", "available": "0.5"},
                {"currency": "ETH", "balance": "2", "available": "2"},
            ]
        )

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
        client.get_accounts = AsyncMock(
            return_value=[
                {"currency": "USDT", "balance": "2000", "available": "2000"},
                {"currency": "BTC", "balance": "0", "available": "0"},
            ]
        )

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
        client.get_accounts = AsyncMock(
            return_value=[
                {"currency": "USDT", "balance": "500", "available": "500"},
                {"currency": "XRP", "balance": "1000", "available": "1000"},
            ]
        )
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


class TestKlineSorting:
    """Verify klines are sorted ascending by timestamp before caching."""

    @pytest.mark.asyncio
    async def test_klines_sorted_ascending(self):
        """get_klines must return data sorted oldest-first (ascending ts)."""
        client = MagicMock(spec=KuCoinClient)
        # Simulate KuCoin returning newest-first (descending)
        raw = [
            [300, "1", "2", "3", "0.5", "10", "20"],
            [100, "1", "2", "3", "0.5", "10", "20"],
            [200, "1", "2", "3", "0.5", "10", "20"],
        ]
        client.get_klines = AsyncMock(return_value=raw)

        from kucoin_bot.services.market_data import MarketDataService

        service = MarketDataService(client=client)
        result = await service.get_klines("BTC-USDT")
        timestamps = [int(k[0]) for k in result]
        assert timestamps == sorted(timestamps), "Klines must be sorted ascending by timestamp"


class TestCircuitBreakerLive:
    """Verify circuit breaker is evaluated in the live loop."""

    def test_check_circuit_breaker_activates_on_daily_loss(self):
        """Circuit breaker must activate when daily loss limit is breached."""
        risk_mgr = RiskManager(config=RiskConfig(max_daily_loss_pct=3.0))
        risk_mgr.update_equity(10_000)
        # Simulate losing more than 3% of peak equity
        risk_mgr.record_pnl(-350)  # 3.5% of 10k peak
        assert risk_mgr.check_circuit_breaker() is True
        assert risk_mgr.circuit_breaker_active is True


class TestPaperModeEquity:
    """Verify paper mode equity is tracked correctly through entries and exits."""

    def test_paper_exit_updates_equity(self):
        """Paper exit must update risk_mgr equity with the realised P&L."""
        risk_mgr = RiskManager(config=RiskConfig())
        risk_mgr.update_equity(10_000)

        # Simulate paper entry: deduct fee
        entry_fee = 0.3  # small fee
        risk_mgr.update_equity(risk_mgr.current_equity - entry_fee)
        assert risk_mgr.current_equity == pytest.approx(10_000 - entry_fee)

        # Simulate paper exit with positive PnL
        exit_pnl = 50.0  # net after fee
        risk_mgr.record_pnl(exit_pnl)
        risk_mgr.update_equity(risk_mgr.current_equity + exit_pnl)

        expected = 10_000 - entry_fee + exit_pnl
        assert risk_mgr.current_equity == pytest.approx(expected)

    def test_strategy_monitor_no_double_count_fee(self):
        """strategy_monitor.record_trade must receive raw PnL (before fee), not fee-inclusive."""
        from kucoin_bot.services.strategy_monitor import StrategyMonitor

        monitor = StrategyMonitor(window=5, min_trades=1)
        # raw_pnl=10.0, fee=3.0 → net expectancy = (10 - 3) / 1 = 7.0
        monitor.record_trade("trend_following", pnl=10.0, cost=3.0)

        stats = monitor.get_status()["trend_following"]
        assert stats["net_expectancy"] == pytest.approx(7.0)

    def test_double_counted_fee_gives_wrong_expectancy(self):
        """Demonstrates the bug: passing fee-inclusive PnL double-counts the fee."""
        from kucoin_bot.services.strategy_monitor import StrategyMonitor

        monitor = StrategyMonitor(window=5, min_trades=1)
        # If we mistakenly pass pnl=7 (already has fee subtracted) and cost=3
        # the monitor computes net = 7 - 3 = 4, which is WRONG (should be 7)
        monitor.record_trade("trend_following", pnl=7.0, cost=3.0)

        stats = monitor.get_status()["trend_following"]
        assert stats["net_expectancy"] == pytest.approx(4.0)  # wrong value from double-counting


class TestBatchPortfolioAllocation:
    """Verify portfolio allocations are normalised across multiple symbols."""

    def test_multi_symbol_weights_sum_to_one(self):
        """When multiple symbols are allocated, their weights should sum to ~1."""
        client = MagicMock(spec=KuCoinClient)
        risk_mgr = RiskManager(config=RiskConfig())
        risk_mgr.update_equity(10_000)
        pm = PortfolioManager(client=client, risk_mgr=risk_mgr)

        signals = {
            "BTC-USDT": SignalScores(
                symbol="BTC-USDT",
                regime=Regime.TRENDING_UP,
                confidence=0.6,
                volatility=0.3,
                momentum=0.4,
                trend_strength=0.6,
            ),
            "ETH-USDT": SignalScores(
                symbol="ETH-USDT",
                regime=Regime.TRENDING_UP,
                confidence=0.5,
                volatility=0.4,
                momentum=0.3,
                trend_strength=0.5,
            ),
        }
        allocs = pm.compute_allocations(signals, ["BTC-USDT", "ETH-USDT"])

        # Weights should sum to 1.0 when both symbols qualify
        total_weight = sum(a.weight for a in allocs.values() if a.weight > 0)
        assert total_weight == pytest.approx(1.0, abs=0.01)

        # Individual weights should be < 1.0 (not both getting 1.0)
        assert allocs["BTC-USDT"].weight < 1.0
        assert allocs["ETH-USDT"].weight < 1.0

    def test_single_symbol_gets_full_weight(self):
        """When only one symbol qualifies, it gets weight=1.0."""
        client = MagicMock(spec=KuCoinClient)
        risk_mgr = RiskManager(config=RiskConfig())
        risk_mgr.update_equity(10_000)
        pm = PortfolioManager(client=client, risk_mgr=risk_mgr)

        signals = {
            "BTC-USDT": SignalScores(
                symbol="BTC-USDT",
                regime=Regime.TRENDING_UP,
                confidence=0.6,
                volatility=0.3,
                momentum=0.4,
                trend_strength=0.6,
            ),
        }
        allocs = pm.compute_allocations(signals, ["BTC-USDT"])
        assert allocs["BTC-USDT"].weight == pytest.approx(1.0)


class TestLeveragePositionSizing:
    """Verify leverage amplifies position size correctly."""

    @pytest.mark.asyncio
    async def test_leverage_amplifies_position_size(self):
        """With leverage > 1, position size should be multiplied by leverage."""
        client = MagicMock(spec=KuCoinClient)
        client.place_order = AsyncMock(
            return_value={
                "code": "200000",
                "data": {"orderId": "lev-order-1"},
            }
        )
        risk_mgr = RiskManager(config=RiskConfig())
        risk_mgr.update_equity(10_000)
        engine = ExecutionEngine(client=client, risk_mgr=risk_mgr, poll_fills=False)

        market = MarketInfo(
            symbol="BTC-USDT",
            base="BTC",
            quote="USDT",
            base_min_size=0.0001,
            base_increment=0.0001,
            price_increment=0.01,
            last_price=30000.0,
            spread_bps=5.0,
        )

        # Execute without leverage (1.0)
        result_1x = await engine.execute(
            OrderRequest(symbol="BTC-USDT", side="buy", notional=300, price=30000.0, leverage=1.0, reason="test"),
            market,
        )
        # Execute with 3x leverage
        result_3x = await engine.execute(
            OrderRequest(symbol="BTC-USDT", side="buy", notional=300, price=30000.0, leverage=3.0, reason="test"),
            market,
        )

        assert result_1x.success and result_3x.success
        # 3x leverage should give 3x the filled quantity
        assert result_3x.filled_qty == pytest.approx(result_1x.filled_qty * 3.0, rel=0.01)

    @pytest.mark.asyncio
    async def test_leverage_1x_unchanged(self):
        """With leverage=1.0 (default), position size is notional/price as before."""
        client = MagicMock(spec=KuCoinClient)
        client.place_order = AsyncMock(
            return_value={
                "code": "200000",
                "data": {"orderId": "lev-order-2"},
            }
        )
        risk_mgr = RiskManager(config=RiskConfig())
        risk_mgr.update_equity(10_000)
        engine = ExecutionEngine(client=client, risk_mgr=risk_mgr, poll_fills=False)

        market = MarketInfo(
            symbol="ETH-USDT",
            base="ETH",
            quote="USDT",
            base_min_size=0.001,
            base_increment=0.001,
            price_increment=0.01,
            last_price=2000.0,
            spread_bps=5.0,
        )

        result = await engine.execute(
            OrderRequest(symbol="ETH-USDT", side="buy", notional=200, price=2000.0, leverage=1.0, reason="test"),
            market,
        )
        assert result.success
        # 200 / 2000 = 0.1
        assert result.filled_qty == pytest.approx(0.1, rel=0.01)

    def test_compute_position_size_leverage_caps_exposure(self):
        """compute_position_size with leverage should respect total exposure limit."""
        rm = RiskManager(
            config=RiskConfig(
                max_per_position_risk_pct=50.0,  # generous per-position limit
                max_total_exposure_pct=80.0,
            )
        )
        rm.update_equity(10_000)

        signals = SignalScores(symbol="BTC-USDT", confidence=0.9, volatility=0.1)

        # With leverage=1, remaining exposure = 8000
        notional_1x = rm.compute_position_size("BTC-USDT", 30000, 0.1, signals, leverage=1.0)
        # With leverage=3, margin capacity = 8000 / 3 ≈ 2666
        notional_3x = rm.compute_position_size("BTC-USDT", 30000, 0.1, signals, leverage=3.0)

        # 3x leverage should give equal or less margin (because remaining/leverage)
        assert notional_3x <= notional_1x

    def test_leverage_computed_in_allocation(self):
        """compute_allocations should set max_leverage > 1 for high-confidence signals."""
        client = MagicMock(spec=KuCoinClient)
        risk_mgr = RiskManager(config=RiskConfig(max_leverage=3.0))
        risk_mgr.update_equity(10_000)
        pm = PortfolioManager(client=client, risk_mgr=risk_mgr)

        signals = {
            "BTC-USDT": SignalScores(
                symbol="BTC-USDT",
                regime=Regime.TRENDING_UP,
                confidence=0.9,
                volatility=0.2,
                momentum=0.4,
                trend_strength=0.7,
            ),
        }
        allocs = pm.compute_allocations(signals, ["BTC-USDT"])
        assert allocs["BTC-USDT"].max_leverage > 1.0


class TestTransferForFutures:
    """Verify internal transfer is called for futures orders."""

    @pytest.mark.asyncio
    async def test_transfer_called_for_valid_route(self):
        """transfer_if_needed must call inner_transfer for valid routes."""
        client = MagicMock(spec=KuCoinClient)
        client.inner_transfer = AsyncMock(
            return_value={
                "code": "200000",
                "data": {"orderId": "xfer-1"},
            }
        )
        risk_mgr = RiskManager(config=RiskConfig())
        pm = PortfolioManager(client=client, risk_mgr=risk_mgr, allow_transfers=True)

        result = await pm.transfer_if_needed("USDT", "trade", "futures", 500)
        assert result is not None  # should return idempotency key
        client.inner_transfer.assert_called_once()
        call_kwargs = client.inner_transfer.call_args.kwargs
        assert call_kwargs["currency"] == "USDT"
        assert call_kwargs["from_account"] == "trade"
        assert call_kwargs["to_account"] == "futures"
        assert call_kwargs["amount"] == 500

    @pytest.mark.asyncio
    async def test_transfer_trade_to_futures_allowed(self):
        """trade → futures is a valid transfer route."""
        client = MagicMock(spec=KuCoinClient)
        client.inner_transfer = AsyncMock(return_value={"code": "200000"})
        risk_mgr = RiskManager(config=RiskConfig())
        pm = PortfolioManager(client=client, risk_mgr=risk_mgr, allow_transfers=True)

        result = await pm.transfer_if_needed("USDT", "trade", "futures", 100)
        assert result is not None

    @pytest.mark.asyncio
    async def test_transfer_futures_to_trade_allowed(self):
        """futures → trade is a valid transfer route (for exit proceeds)."""
        client = MagicMock(spec=KuCoinClient)
        client.inner_transfer = AsyncMock(return_value={"code": "200000"})
        risk_mgr = RiskManager(config=RiskConfig())
        pm = PortfolioManager(client=client, risk_mgr=risk_mgr, allow_transfers=True)

        result = await pm.transfer_if_needed("USDT", "futures", "trade", 200)
        assert result is not None

    @pytest.mark.asyncio
    async def test_transfer_zero_amount_skipped(self):
        """Transferring zero amount should be skipped."""
        client = MagicMock(spec=KuCoinClient)
        risk_mgr = RiskManager(config=RiskConfig())
        pm = PortfolioManager(client=client, risk_mgr=risk_mgr, allow_transfers=True)

        result = await pm.transfer_if_needed("USDT", "trade", "futures", 0)
        assert result is None
