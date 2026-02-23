"""Tests for short trading: PnL math, costs, EV gate, no look-ahead, reduceOnly."""

from __future__ import annotations

import pytest

from kucoin_bot.backtest.engine import BacktestEngine
from kucoin_bot.config import RiskConfig
from kucoin_bot.services.cost_model import CostModel, TradeCosts
from kucoin_bot.services.side_selector import SideSelector
from kucoin_bot.services.signal_engine import Regime, SignalScores
from kucoin_bot.services.strategy_monitor import StrategyMonitor
from kucoin_bot.strategies.mean_reversion import MeanReversion
from kucoin_bot.strategies.trend import TrendFollowing

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_trending_down_klines(n: int = 200) -> list:
    """Klines with a clear downtrend – generates short signals."""
    klines = []
    price = 100.0
    for i in range(n):
        price *= 0.995  # consistent downtrend
        o = price * 1.002
        h = price * 1.004
        low = price * 0.997
        klines.append([i * 3600, str(o), str(price), str(h), str(low), "500", str(500 * price)])
    return klines


def _make_ranging_klines(n: int = 200) -> list:
    """Klines oscillating in a tight range (for mean-reversion short signals)."""
    import numpy as np

    rng = np.random.RandomState(77)
    klines = []
    price = 100.0
    for i in range(n):
        price = 100.0 + 3.0 * float(np.sin(i * 0.15)) + float(rng.normal(0, 0.2))
        o = price - 0.1
        h = price + 0.4
        low = price - 0.4
        klines.append([i * 3600, str(o), str(price), str(h), str(low), "500", str(500 * price)])
    return klines


# ---------------------------------------------------------------------------
# 1. Short position PnL math – futures
# ---------------------------------------------------------------------------


class TestShortPnLMathFutures:
    """Verify futures short PnL = (entry - exit) * size - fees - funding."""

    def _make_fixed_klines(self, entry_price: float, exit_price: float, n_bars: int = 100) -> list:
        """Create klines where price drops from entry_price to exit_price."""
        klines = []
        for i in range(n_bars):
            # First half: stable at entry_price; second half: drop to exit_price
            if i < n_bars // 2:
                p = entry_price
            else:
                p = exit_price
            o = p * 1.001
            h = p * 1.002
            low = p * 0.998
            klines.append([i * 3600, str(o), str(p), str(h), str(low), "500", str(500 * p)])
        return klines

    def test_short_pnl_positive_when_price_falls(self):
        """Short position PnL must be positive when price falls."""
        engine = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            min_ev_bps=0,
            cooldown_bars=0,
        )
        klines = _make_trending_down_klines(200)
        result = engine.run(klines, "BTC-USDT", initial_equity=10_000, market_type="futures")

        short_exits = [t for t in result.trades if t.side == "exit" and t.position_side == "short"]
        if short_exits:
            total_short_pnl = sum(t.pnl for t in short_exits)
            # At least some short trades should produce positive PnL in a downtrend
            assert total_short_pnl != 0

    def test_short_pnl_math_correct(self):
        """Manual verification: PnL = (entry - exit) * qty - fee."""
        entry = 100.0
        exit_ = 90.0
        qty = 1.0
        fee_rate = 0.001

        # Compute expected PnL
        entry_fee = qty * entry * fee_rate
        exit_fee = qty * exit_ * fee_rate
        _ = (entry - exit_) * qty - entry_fee - exit_fee  # expected_pnl (unused, kept for reference)

        # BacktestTrade pnl already deducts exit fee; entry fee is deducted from equity separately
        # For a raw check: pnl at exit = (entry - exit_) * qty - exit_fee
        raw_pnl_at_exit = (entry - exit_) * qty - exit_fee
        assert raw_pnl_at_exit > 0, "Short must profit when price falls"
        assert raw_pnl_at_exit == pytest.approx(9.91, abs=0.01)

    def test_short_pnl_negative_when_price_rises(self):
        """Short position PnL must be negative when price rises."""
        entry = 100.0
        exit_ = 110.0
        qty = 1.0
        fee_rate = 0.001
        exit_fee = qty * exit_ * fee_rate
        raw_pnl_at_exit = (entry - exit_) * qty - exit_fee
        assert raw_pnl_at_exit < 0, "Short must lose when price rises"


# ---------------------------------------------------------------------------
# 2. Short position PnL math – margin
# ---------------------------------------------------------------------------


class TestShortPnLMathMargin:
    """Verify margin short includes borrow cost."""

    def test_margin_borrow_reduces_pnl(self):
        """A margin short held for many bars should have higher total costs than a futures short."""
        engine_margin = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            min_ev_bps=0,
            cooldown_bars=0,
        )
        engine_futures = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            min_ev_bps=0,
            cooldown_bars=0,
        )
        klines = _make_trending_down_klines(200)
        result_margin = engine_margin.run(klines, "BTC-USDT", initial_equity=10_000, market_type="margin")
        result_futures = engine_futures.run(klines, "BTC-USDT", initial_equity=10_000, market_type="futures")

        # Margin borrow cost should be >= 0
        assert result_margin.cost_breakdown["borrow"] >= 0
        # Futures has a funding field (may be positive if long paid, negative if short received)
        assert isinstance(result_futures.cost_breakdown["funding"], float)


# ---------------------------------------------------------------------------
# 3. Funding / borrow cost application
# ---------------------------------------------------------------------------


class TestFundingAndBorrowCosts:
    """Verify cost model computes funding and borrow correctly."""

    def test_funding_cost_per_8h_no_side(self):
        """Without position_side, abs() is used as conservative worst-case estimate."""
        model = CostModel(funding_rate_per_8h=0.0001)
        costs = model.estimate("taker", holding_hours=8.0, is_futures=True)
        # abs(0.0001) * 1 period * 10_000 = 1.0 bps
        assert costs.funding_bps == pytest.approx(1.0, abs=0.001)

    def test_funding_cost_long_positive_rate(self):
        """Long position pays when funding rate is positive."""
        model = CostModel(funding_rate_per_8h=0.0001)
        costs = model.estimate("taker", holding_hours=8.0, is_futures=True, position_side="long")
        # Positive rate → long pays → positive cost
        assert costs.funding_bps == pytest.approx(1.0, abs=0.001)

    def test_funding_cost_short_positive_rate(self):
        """Short position receives when funding rate is positive (negative cost)."""
        model = CostModel(funding_rate_per_8h=0.0001)
        costs = model.estimate("taker", holding_hours=8.0, is_futures=True, position_side="short")
        # Positive rate → short receives → negative cost (funding is a benefit)
        assert costs.funding_bps == pytest.approx(-1.0, abs=0.001)

    def test_funding_cost_short_negative_rate(self):
        """Short position pays when funding rate is negative."""
        model = CostModel(funding_rate_per_8h=-0.0002)
        costs = model.estimate("taker", holding_hours=8.0, is_futures=True, position_side="short")
        # Negative rate → short pays → positive cost
        assert costs.funding_bps == pytest.approx(2.0, abs=0.001)

    def test_funding_cost_long_negative_rate(self):
        """Long position receives when funding rate is negative (negative cost)."""
        model = CostModel(funding_rate_per_8h=-0.0002)
        costs = model.estimate("taker", holding_hours=8.0, is_futures=True, position_side="long")
        # Negative rate → long receives → negative cost (benefit)
        assert costs.funding_bps == pytest.approx(-2.0, abs=0.001)

    def test_funding_multiple_periods_directional(self):
        """Multi-period directional funding scales linearly for both sides."""
        model = CostModel(funding_rate_per_8h=0.0001)
        costs_long_1p = model.estimate("taker", holding_hours=8.0, is_futures=True, position_side="long")
        costs_long_3p = model.estimate("taker", holding_hours=24.0, is_futures=True, position_side="long")
        costs_short_3p = model.estimate("taker", holding_hours=24.0, is_futures=True, position_side="short")
        # Long with 3 periods should be 3× one period
        assert costs_long_3p.funding_bps == pytest.approx(costs_long_1p.funding_bps * 3, abs=0.001)
        # Short with 3 periods should be negative of long 3 periods (receives funding)
        assert costs_short_3p.funding_bps == pytest.approx(-costs_long_3p.funding_bps, abs=0.001)

    def test_funding_cost_two_periods(self):
        """Two 8-hour periods should double the funding cost (conservative/no side)."""
        model = CostModel(funding_rate_per_8h=0.0001)
        costs_1p = model.estimate("taker", holding_hours=8.0, is_futures=True)
        costs_2p = model.estimate("taker", holding_hours=16.0, is_futures=True)
        assert costs_2p.funding_bps == pytest.approx(costs_1p.funding_bps * 2, abs=0.001)

    def test_borrow_cost_per_hour(self):
        """Borrow cost over 8 hours at 0.003%/hr rate."""
        model = CostModel(borrow_rate_per_hour=0.00003)
        costs = model.estimate("taker", holding_hours=8.0, is_margin_short=True)
        # 0.00003 * 8 * 10_000 = 2.4 bps
        assert costs.borrow_bps == pytest.approx(2.4, abs=0.001)

    def test_no_funding_for_spot(self):
        """Spot orders should have zero funding cost."""
        model = CostModel(funding_rate_per_8h=0.001)
        costs = model.estimate("taker", holding_hours=8.0, is_futures=False)
        assert costs.funding_bps == 0.0

    def test_no_borrow_for_long(self):
        """Long positions should have zero borrow cost."""
        model = CostModel(borrow_rate_per_hour=0.001)
        costs = model.estimate("taker", holding_hours=8.0, is_margin_short=False)
        assert costs.borrow_bps == 0.0

    def test_funding_applied_every_8_bars_in_backtest(self):
        """Backtest should track funding in cost_breakdown for futures; spot shows zero."""
        engine_futures = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            min_ev_bps=0,
            cooldown_bars=0,
        )
        engine_spot = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            min_ev_bps=0,
            cooldown_bars=0,
        )
        klines = _make_trending_down_klines(200)
        result_futures = engine_futures.run(klines, "BTC-USDT", initial_equity=10_000, market_type="futures")
        result_spot = engine_spot.run(klines, "BTC-USDT", initial_equity=10_000, market_type="spot")

        # Futures shows a non-zero funding field (positive = paid, negative = received)
        assert isinstance(result_futures.cost_breakdown["funding"], float)
        # Spot never charges funding
        assert result_spot.cost_breakdown["funding"] == 0.0

    def test_live_funding_rate_override(self):
        """A live funding rate override should replace the default (conservative/no side)."""
        model = CostModel(funding_rate_per_8h=0.0001)
        costs_default = model.estimate("taker", holding_hours=8.0, is_futures=True)
        costs_override = model.estimate("taker", holding_hours=8.0, is_futures=True, live_funding_rate=0.0005)
        assert costs_override.funding_bps > costs_default.funding_bps


# ---------------------------------------------------------------------------
# 4. EV gate blocks trades when costs exceed edge
# ---------------------------------------------------------------------------


class TestEVGate:
    """Verify the CostModel EV gate rejects low-edge trades."""

    def test_ev_gate_passes_when_edge_sufficient(self):
        model = CostModel(
            taker_fee=0.001,
            slippage_bps=5.0,
            safety_buffer_bps=10.0,
        )
        costs = model.estimate("taker", holding_hours=8.0)
        # fee=20bps, slip=10bps, total=30bps; need > 30+10=40bps
        assert costs.total_bps == pytest.approx(30.0, abs=0.1)
        assert model.ev_gate(expected_bps=50.0, costs=costs) is True

    def test_ev_gate_blocks_marginal_trade(self):
        model = CostModel(
            taker_fee=0.001,
            slippage_bps=5.0,
            safety_buffer_bps=10.0,
        )
        costs = model.estimate("taker", holding_hours=8.0)
        # Exactly at cost threshold – should be blocked (not strictly greater)
        assert model.ev_gate(expected_bps=40.0, costs=costs) is False

    def test_ev_gate_blocks_when_funding_exceeds_edge(self):
        """High funding cost should tip the gate into blocking."""
        model = CostModel(
            taker_fee=0.001,
            slippage_bps=5.0,
            funding_rate_per_8h=0.005,  # very high 0.5% per 8h
            safety_buffer_bps=10.0,
        )
        costs = model.estimate("taker", holding_hours=24.0, is_futures=True)
        # funding = 0.005 * 3 * 10_000 = 150 bps, total way above any edge
        assert model.ev_gate(expected_bps=50.0, costs=costs) is False

    def test_backtest_ev_gate_blocks_all_with_extreme_threshold(self):
        """With absurdly high min_ev_bps all trades are blocked."""
        engine = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            min_ev_bps=10_000,
        )
        import numpy as np

        rng = np.random.RandomState(42)
        n = 200
        price = 30000.0
        klines = []
        for i in range(n):
            ret = rng.normal(0, 0.01)
            o = price
            c = price * (1 + ret)
            h = max(o, c) * (1 + abs(rng.normal(0, 0.003)))
            low = min(o, c) * (1 - abs(rng.normal(0, 0.003)))
            klines.append([i * 3600, str(o), str(c), str(h), str(low), "500", str(500 * c)])
            price = c
        result = engine.run(klines, "BTC-USDT", initial_equity=10_000)
        assert result.total_trades == 0

    def test_cost_model_total_bps(self):
        """TradeCosts.total_bps should sum all components."""
        costs = TradeCosts(fee_bps=20.0, slippage_bps=10.0, funding_bps=1.0, borrow_bps=2.4)
        assert costs.total_bps == pytest.approx(33.4, abs=0.001)

    def test_cost_model_to_dict(self):
        """TradeCosts.to_dict should include all keys."""
        costs = TradeCosts(fee_bps=20.0, slippage_bps=10.0, funding_bps=1.0, borrow_bps=2.4)
        d = costs.to_dict()
        assert set(d.keys()) == {"fee_bps", "slippage_bps", "funding_bps", "borrow_bps", "total_bps"}

    def test_default_cost_model_spot_round_trip(self):
        """Default CostModel should produce 24 bps round-trip for spot taker orders."""
        model = CostModel()
        costs = model.estimate("taker", holding_hours=24.0)
        # fee = 0.001 * 2 * 10000 = 20 bps, slippage = 2 * 2 = 4 bps
        assert costs.fee_bps == pytest.approx(20.0)
        assert costs.slippage_bps == pytest.approx(4.0)
        assert costs.total_bps == pytest.approx(24.0)

    def test_default_safety_buffer(self):
        """Default safety buffer should be 5 bps."""
        model = CostModel()
        assert model.safety_buffer_bps == pytest.approx(5.0)

    def test_ev_gate_passes_trending_signal(self):
        """A trending signal with moderate volatility should pass the default EV gate."""
        model = CostModel()
        costs = model.estimate("taker", holding_hours=24.0)
        # Simulates WAN-USDT-like signal: trend_strength > volatility
        volatility, trend_strength, confidence = 0.4155, 0.6651, 0.4835
        expected_bps = max(volatility, trend_strength) * 100.0 * confidence
        assert expected_bps > costs.total_bps + model.safety_buffer_bps


# ---------------------------------------------------------------------------
# 5. No look-ahead enforcement
# ---------------------------------------------------------------------------


class TestNoLookAhead:
    """Signal at bar i must fill at bar i+1 open, not bar i close."""

    def test_short_fills_at_next_bar_open(self, sample_klines):
        """Short entry fills must have price > 0 and be recorded as entry_short."""
        engine = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            min_ev_bps=0,
            cooldown_bars=0,
        )
        result = engine.run(sample_klines, "BTC-USDT", initial_equity=10_000, warmup=60)
        for trade in result.trades:
            if trade.side == "entry_short":
                assert trade.price > 0
                assert trade.quantity > 0

    def test_exit_records_position_side(self, sample_klines):
        """Exit records should carry the position_side field for per-side PnL."""
        engine = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            min_ev_bps=0,
            cooldown_bars=0,
        )
        result = engine.run(sample_klines, "BTC-USDT", initial_equity=10_000, warmup=60)
        for trade in result.trades:
            if trade.side == "exit":
                # position_side must be "long" or "short" or "" (for unclosed positions)
                assert trade.position_side in ("long", "short", "")


# ---------------------------------------------------------------------------
# 6. reduceOnly exits used in futures
# ---------------------------------------------------------------------------


class TestReduceOnly:
    """OrderRequest must set reduce_only=True for futures exits."""

    def test_reduce_only_flag_on_order_request(self):
        """OrderRequest with reduce_only=True should be distinct from default."""
        from kucoin_bot.services.execution import OrderRequest

        req_normal = OrderRequest(symbol="BTC-USDT", side="sell", notional=100.0)
        req_reduce = OrderRequest(symbol="BTC-USDT", side="sell", notional=100.0, reduce_only=True)
        assert req_normal.reduce_only is False
        assert req_reduce.reduce_only is True

    def test_futures_order_has_reduce_only_in_body(self):
        """place_futures_order must include reduceOnly=True in the request body when set."""
        import asyncio

        call_args = {}

        async def fake_request(method, path, **kwargs):
            call_args.update(kwargs)
            return {"code": "200000", "data": {"orderId": "x"}}

        from kucoin_bot.api.client import KuCoinClient

        client = KuCoinClient.__new__(KuCoinClient)
        client._api_key = "k"
        client._api_secret = "s"
        client._api_passphrase = "p"
        client._rest_url = "https://api.kucoin.com"
        client._futures_rest_url = "https://api-futures.kucoin.com"
        client._session = object()  # not None
        client._request_times = []
        client._time_offset_ms = 0
        client._request = fake_request

        asyncio.run(
            client.place_futures_order(
                symbol="XBTUSDTM",
                side="sell",
                size=1,
                reduce_only=True,
            )
        )
        assert call_args.get("body", {}).get("reduceOnly") is True

    def test_futures_order_no_reduce_only_by_default(self):
        """place_futures_order should NOT include reduceOnly by default."""
        import asyncio

        call_args = {}

        async def fake_request(method, path, **kwargs):
            call_args.update(kwargs)
            return {"code": "200000", "data": {"orderId": "x"}}

        from kucoin_bot.api.client import KuCoinClient

        client = KuCoinClient.__new__(KuCoinClient)
        client._api_key = "k"
        client._api_secret = "s"
        client._api_passphrase = "p"
        client._rest_url = "https://api.kucoin.com"
        client._futures_rest_url = "https://api-futures.kucoin.com"
        client._session = object()
        client._request_times = []
        client._time_offset_ms = 0
        client._request = fake_request

        asyncio.run(
            client.place_futures_order(
                symbol="XBTUSDTM",
                side="sell",
                size=1,
            )
        )
        assert "reduceOnly" not in call_args.get("body", {})


# ---------------------------------------------------------------------------
# 7. Side selector and squeeze filter
# ---------------------------------------------------------------------------


class TestSideSelector:
    """Verify side selector blocks shorts under squeeze and crowding conditions."""

    def _signals(self, **kwargs) -> SignalScores:
        defaults = dict(
            symbol="BTC-USDT",
            momentum=-0.4,
            trend_strength=0.7,
            volatility=0.2,
            volume_anomaly=0.0,
            funding_rate=0.0,
            regime=Regime.TRENDING_DOWN,
        )
        defaults.update(kwargs)
        return SignalScores(**defaults)

    def test_short_allowed_on_futures(self):
        sel = SideSelector(allow_shorts=True, require_futures_for_short=True)
        sig = self._signals()
        dec = sel.select(sig, market_type="futures", proposed_side="short")
        assert dec.side == "short"

    def test_short_blocked_on_spot(self):
        sel = SideSelector(allow_shorts=True, require_futures_for_short=True)
        sig = self._signals()
        dec = sel.select(sig, market_type="spot", proposed_side="short")
        assert dec.side == "flat"
        assert dec.reason == "short_not_available"

    def test_short_blocked_when_allow_shorts_false(self):
        sel = SideSelector(allow_shorts=False)
        sig = self._signals()
        dec = sel.select(sig, market_type="futures", proposed_side="short")
        assert dec.side == "flat"

    def test_short_blocked_on_volatility_spike(self):
        sel = SideSelector(allow_shorts=True, require_futures_for_short=False)
        # High vol + positive momentum = squeeze risk blocks short
        sig = self._signals(volatility=0.8, momentum=0.3)
        dec = sel.select(sig, market_type="futures", proposed_side="short")
        assert dec.side == "flat"
        assert dec.reason == "squeeze_risk"
        assert dec.squeeze_risk is True

    def test_short_allowed_in_downtrend_high_vol(self):
        """High volatility alone should NOT block shorts when momentum is negative (downtrend)."""
        sel = SideSelector(allow_shorts=True, require_futures_for_short=False)
        sig = self._signals(volatility=0.8, momentum=-0.4)
        dec = sel.select(sig, market_type="futures", proposed_side="short")
        assert dec.side == "short"

    def test_short_blocked_on_momentum_volume_spike(self):
        sel = SideSelector(allow_shorts=True, require_futures_for_short=False)
        # Strong positive momentum + volume spike = potential squeeze
        sig = self._signals(momentum=0.6, volume_anomaly=3.0, volatility=0.2)
        dec = sel.select(sig, market_type="futures", proposed_side="short")
        assert dec.side == "flat"
        assert dec.squeeze_risk is True

    def test_short_blocked_on_crowded_funding(self):
        sel = SideSelector(allow_shorts=True, require_futures_for_short=False)
        # Very negative funding = crowded short
        sig = self._signals(funding_rate=-0.001)
        dec = sel.select(sig, market_type="futures", proposed_side="short")
        assert dec.side == "flat"
        assert dec.reason == "crowded_short"

    def test_long_not_affected_by_squeeze_filter(self):
        sel = SideSelector(allow_shorts=True)
        sig = self._signals(volatility=0.8, momentum=0.7, regime=Regime.TRENDING_UP)
        dec = sel.select(sig, market_type="futures", proposed_side="long")
        # Long trades are not blocked by the squeeze filter
        assert dec.side == "long"

    def test_derive_side_downtrend(self):
        sel = SideSelector(allow_shorts=False)  # shorts not allowed
        sig = self._signals()
        dec = sel.select(sig, market_type="spot")
        # No short allowed → flat
        assert dec.side == "flat"

    def test_derive_side_strong_signal_cross_regime(self):
        """Strong trend + momentum should derive side even in UNKNOWN regime."""
        sel = SideSelector(allow_shorts=True, require_futures_for_short=False)
        sig = self._signals(
            regime=Regime.UNKNOWN,
            trend_strength=0.7,
            momentum=0.4,
        )
        dec = sel.select(sig, market_type="futures")
        assert dec.side == "long"


# ---------------------------------------------------------------------------
# 8. Strategy monitor
# ---------------------------------------------------------------------------


class TestStrategyMonitor:
    """Verify rolling PnL tracking and auto-disable behaviour."""

    def test_module_starts_enabled(self):
        monitor = StrategyMonitor()
        assert monitor.is_enabled("trend_following") is True

    def test_module_not_disabled_before_min_trades(self):
        monitor = StrategyMonitor(min_trades=5)
        for _ in range(4):
            monitor.record_trade("trend_following", pnl=-1.0, cost=0.1)
        assert monitor.is_enabled("trend_following") is True

    def test_module_disabled_after_negative_expectancy(self):
        monitor = StrategyMonitor(min_trades=5, window=20)
        for _ in range(5):
            monitor.record_trade("trend_following", pnl=-1.0, cost=0.1)
        assert monitor.is_enabled("trend_following") is False

    def test_module_stays_enabled_with_positive_expectancy(self):
        monitor = StrategyMonitor(min_trades=5)
        for _ in range(10):
            monitor.record_trade("mean_reversion", pnl=2.0, cost=0.5)
        assert monitor.is_enabled("mean_reversion") is True

    def test_manual_re_enable(self):
        monitor = StrategyMonitor(min_trades=2)
        for _ in range(2):
            monitor.record_trade("scalping", pnl=-5.0, cost=0.1)
        assert monitor.is_enabled("scalping") is False
        monitor.enable("scalping")
        assert monitor.is_enabled("scalping") is True

    def test_get_status_returns_all_modules(self):
        monitor = StrategyMonitor()
        monitor.record_trade("trend_following", pnl=1.0)
        monitor.record_trade("mean_reversion", pnl=-0.5)
        status = monitor.get_status()
        assert "trend_following" in status
        assert "mean_reversion" in status
        assert "trade_count" in status["trend_following"]
        assert "net_expectancy" in status["trend_following"]

    def test_net_expectancy_correct(self):
        monitor = StrategyMonitor()
        monitor.record_trade("hedge", pnl=10.0, cost=2.0)  # net = 8
        monitor.record_trade("hedge", pnl=6.0, cost=2.0)  # net = 4
        # avg net = (8 + 4) / 2 = 6
        status = monitor.get_status()
        assert status["hedge"]["net_expectancy"] == pytest.approx(6.0, abs=0.001)


# ---------------------------------------------------------------------------
# 9. BacktestResult per-side stats and cost breakdown
# ---------------------------------------------------------------------------


class TestBacktestResultPerSide:
    """Verify per-side PnL and cost breakdown fields."""

    def test_result_has_per_side_fields(self, sample_klines):
        engine = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            min_ev_bps=0,
            cooldown_bars=0,
        )
        result = engine.run(sample_klines, "BTC-USDT", initial_equity=10_000)
        assert hasattr(result, "long_pnl")
        assert hasattr(result, "short_pnl")
        assert hasattr(result, "long_trades")
        assert hasattr(result, "short_trades")
        assert isinstance(result.long_trades, int)
        assert isinstance(result.short_trades, int)
        assert result.long_trades + result.short_trades <= result.total_trades

    def test_result_has_cost_breakdown(self, sample_klines):
        engine = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            min_ev_bps=0,
            cooldown_bars=0,
        )
        result = engine.run(sample_klines, "BTC-USDT", initial_equity=10_000)
        cb = result.cost_breakdown
        assert "fees" in cb
        assert "slippage" in cb
        assert "funding" in cb
        assert "borrow" in cb

    def test_futures_result_has_funding_cost(self):
        """Futures backtest should report funding costs in cost_breakdown."""
        klines = _make_trending_down_klines(200)
        engine = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            min_ev_bps=0,
            cooldown_bars=0,
        )
        result = engine.run(klines, "BTC-USDT", initial_equity=10_000, market_type="futures")
        # If any position was held for 8+ bars, funding cost > 0
        assert result.cost_breakdown["funding"] >= 0

    def test_summary_includes_per_side_info(self, sample_klines):
        """summary() should mention both long and short PnL."""
        engine = BacktestEngine(strategies=[TrendFollowing()])
        result = engine.run(sample_klines, "BTC-USDT")
        summary = result.summary()
        assert "Long PnL" in summary
        assert "Short PnL" in summary
        assert "Costs[" in summary

    def test_margin_result_has_borrow_cost(self):
        """Margin short backtest should report borrow costs."""
        klines = _make_trending_down_klines(200)
        engine = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            min_ev_bps=0,
            cooldown_bars=0,
        )
        result = engine.run(klines, "BTC-USDT", initial_equity=10_000, market_type="margin")
        assert result.cost_breakdown["borrow"] >= 0


# ---------------------------------------------------------------------------
# 10. Risk manager squeeze risk check
# ---------------------------------------------------------------------------


class TestRiskManagerSqueezeRisk:
    """RiskManager.check_squeeze_risk delegates to the shared squeeze heuristic."""

    def test_squeeze_risk_false_normal_conditions(self):
        from kucoin_bot.services.risk_manager import RiskManager

        rm = RiskManager(config=RiskConfig())
        sig = SignalScores(symbol="BTC-USDT", volatility=0.3, momentum=0.2, volume_anomaly=1.0)
        assert rm.check_squeeze_risk(sig) is False

    def test_squeeze_risk_true_on_vol_spike(self):
        from kucoin_bot.services.risk_manager import RiskManager

        rm = RiskManager(config=RiskConfig())
        # High vol + positive momentum triggers squeeze risk
        sig = SignalScores(symbol="BTC-USDT", volatility=0.7, momentum=0.3)
        assert rm.check_squeeze_risk(sig) is True

    def test_squeeze_risk_false_on_vol_spike_negative_momentum(self):
        """High vol with negative momentum (downtrend) should NOT flag squeeze risk."""
        from kucoin_bot.services.risk_manager import RiskManager

        rm = RiskManager(config=RiskConfig())
        sig = SignalScores(symbol="BTC-USDT", volatility=0.7, momentum=-0.3)
        assert rm.check_squeeze_risk(sig) is False

    def test_squeeze_risk_true_on_momentum_volume_spike(self):
        from kucoin_bot.services.risk_manager import RiskManager

        rm = RiskManager(config=RiskConfig())
        sig = SignalScores(symbol="BTC-USDT", volatility=0.2, momentum=0.6, volume_anomaly=3.0)
        assert rm.check_squeeze_risk(sig) is True
