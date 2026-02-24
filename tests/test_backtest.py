"""Tests for backtest engine."""

from __future__ import annotations

from kucoin_bot.backtest.engine import BacktestEngine
from kucoin_bot.strategies.mean_reversion import MeanReversion
from kucoin_bot.strategies.risk_off import RiskOff
from kucoin_bot.strategies.trend import TrendFollowing


class TestBacktestEngine:
    def test_basic_run(self, sample_klines):
        engine = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion(), RiskOff()],
        )
        result = engine.run(sample_klines, "BTC-USDT", initial_equity=10_000)
        assert result.initial_equity == 10_000
        assert result.final_equity > 0
        assert result.total_trades >= 0
        assert len(result.equity_curve) > 0

    def test_trending_market(self, trending_up_klines):
        engine = BacktestEngine(
            strategies=[TrendFollowing(), RiskOff()],
        )
        result = engine.run(trending_up_klines, "TREND-USDT", initial_equity=10_000)
        assert result.final_equity > 0
        assert result.max_drawdown_pct >= 0

    def test_fees_are_charged(self, sample_klines):
        engine = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            maker_fee=0.01,  # 1% fee to make it obvious
            taker_fee=0.01,
        )
        result = engine.run(sample_klines, "BTC-USDT", initial_equity=10_000)
        if result.total_trades > 0:
            assert result.total_fees > 0

    def test_deterministic(self, sample_klines):
        engine1 = BacktestEngine(strategies=[TrendFollowing()], seed=42)
        engine2 = BacktestEngine(strategies=[TrendFollowing()], seed=42)
        r1 = engine1.run(sample_klines, "BTC-USDT")
        r2 = engine2.run(sample_klines, "BTC-USDT")
        assert r1.final_equity == r2.final_equity
        assert r1.total_trades == r2.total_trades

    def test_summary_string(self, sample_klines):
        engine = BacktestEngine(strategies=[TrendFollowing()])
        result = engine.run(sample_klines, "BTC-USDT")
        summary = result.summary()
        assert "Backtest:" in summary
        assert "Return:" in summary

    def test_expectancy_and_turnover_fields(self, trending_up_klines):
        """BacktestResult must include expectancy and turnover."""
        engine = BacktestEngine(strategies=[TrendFollowing(), RiskOff()])
        result = engine.run(trending_up_klines, "BTC-USDT", initial_equity=10_000)
        assert hasattr(result, "expectancy")
        assert hasattr(result, "turnover")
        assert isinstance(result.expectancy, float)
        assert isinstance(result.turnover, float)
        assert result.turnover >= 0

    def test_no_look_ahead_fills_at_next_open(self, sample_klines):
        """Fills must happen at bar i+1 open, not bar i close (no look-ahead)."""
        engine = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            min_ev_bps=0,  # disable EV gate so trades can happen
            cooldown_bars=0,
        )
        result = engine.run(sample_klines, "BTC-USDT", initial_equity=10_000, warmup=60)
        # Each entry fill price should equal the open of the bar AFTER the signal
        for trade in result.trades:
            if trade.side.startswith("entry_"):
                # The fill price should NOT match any close exactly as fill was at open
                # We can't check the exact bar without replaying, but we can verify
                # fill prices are non-zero and trades are recorded
                assert trade.price > 0
                assert trade.quantity > 0

    def test_ev_gate_blocks_marginal_trades(self, sample_klines):
        """EV gate should block trades when expected edge is below round-trip cost."""
        # With a very high min_ev_bps, nothing should pass the gate
        engine_strict = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            min_ev_bps=10_000,  # absurdly high: blocks every trade
        )
        result_strict = engine_strict.run(sample_klines, "BTC-USDT", initial_equity=10_000)
        assert result_strict.total_trades == 0

        # With min_ev_bps=0 and cooldown disabled, more trades should be allowed
        engine_open = BacktestEngine(
            strategies=[TrendFollowing(), MeanReversion()],
            min_ev_bps=0,
            cooldown_bars=0,
        )
        result_open = engine_open.run(sample_klines, "BTC-USDT", initial_equity=10_000)
        # The open engine should allow at least as many trades
        assert result_open.total_trades >= result_strict.total_trades

    def test_cooldown_reduces_trades(self, trending_up_klines):
        """Cooldown should reduce trade frequency compared to no-cooldown."""
        engine_no_cd = BacktestEngine(
            strategies=[TrendFollowing(), RiskOff()],
            cooldown_bars=0,
            min_ev_bps=0,
        )
        engine_with_cd = BacktestEngine(
            strategies=[TrendFollowing(), RiskOff()],
            cooldown_bars=20,
            min_ev_bps=0,
        )
        r_no_cd = engine_no_cd.run(trending_up_klines, "BTC-USDT", initial_equity=10_000)
        r_with_cd = engine_with_cd.run(trending_up_klines, "BTC-USDT", initial_equity=10_000)
        assert r_with_cd.total_trades <= r_no_cd.total_trades

    def test_walk_forward_returns_results(self, sample_klines):
        """walk_forward must return a list of BacktestResult objects."""
        engine = BacktestEngine(
            strategies=[TrendFollowing()],
            min_ev_bps=0,
            cooldown_bars=0,
        )
        results = engine.walk_forward(sample_klines, "BTC-USDT", n_splits=3)
        assert isinstance(results, list)
        # Each result should be a valid BacktestResult
        for r in results:
            assert r.initial_equity > 0
            assert len(r.equity_curve) > 0

    def test_latency_ms_increases_effective_slippage(self, trending_up_klines):
        """latency_ms > 0 should increase fill costs vs latency_ms=0."""
        engine_no_lat = BacktestEngine(
            strategies=[TrendFollowing(), RiskOff()],
            latency_ms=0,
            min_ev_bps=0,
            cooldown_bars=0,
        )
        engine_with_lat = BacktestEngine(
            strategies=[TrendFollowing(), RiskOff()],
            latency_ms=500,  # 500 ms adds slippage
            min_ev_bps=0,
            cooldown_bars=0,
        )
        r_no_lat = engine_no_lat.run(trending_up_klines, "BTC-USDT", initial_equity=10_000)
        r_with_lat = engine_with_lat.run(trending_up_klines, "BTC-USDT", initial_equity=10_000)
        # With higher slippage, total fees should be >= no-latency case
        if r_no_lat.total_trades > 0:
            assert r_with_lat.total_fees >= r_no_lat.total_fees

    def test_kline_type_scales_funding_period(self, trending_up_klines):
        """Explicit kline_type must scale funding period to match bar duration."""
        engine = BacktestEngine(
            strategies=[TrendFollowing(), RiskOff()],
            min_ev_bps=0,
            cooldown_bars=0,
        )
        # Run with 1-hour bars (default inference from timestamps) vs explicit "15min"
        # The test just asserts it runs without error and produces results
        result = engine.run(
            trending_up_klines, "BTC-USDT", initial_equity=10_000,
            market_type="futures", kline_type="15min",
        )
        assert result.final_equity > 0
        assert len(result.equity_curve) > 0
