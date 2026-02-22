"""Tests for backtest engine."""

from __future__ import annotations

import json
import pytest

from kucoin_bot.config import RiskConfig
from kucoin_bot.backtest.engine import BacktestEngine
from kucoin_bot.strategies.trend import TrendFollowing
from kucoin_bot.strategies.mean_reversion import MeanReversion
from kucoin_bot.strategies.risk_off import RiskOff
from kucoin_bot.reporting.cli import export_backtest_report


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

    def test_expectancy_and_turnover(self, sample_klines):
        engine = BacktestEngine(strategies=[TrendFollowing(), MeanReversion()])
        result = engine.run(sample_klines, "BTC-USDT")
        # turnover should be non-negative
        assert result.turnover >= 0.0
        # expectancy is defined (even if 0 when no trades)
        assert isinstance(result.expectancy, float)

    def test_latency_simulation(self, sample_klines):
        """Latency simulation should produce non-zero latency on filled trades."""
        engine = BacktestEngine(
            strategies=[TrendFollowing()],
            latency_ms=50,
            seed=42,
        )
        result = engine.run(sample_klines, "BTC-USDT")
        entry_trades = [t for t in result.trades if t.side != "exit"]
        if entry_trades:
            assert any(t.latency_ms > 0 for t in entry_trades)

    def test_walk_forward(self, sample_klines):
        engine = BacktestEngine(strategies=[TrendFollowing(), RiskOff()], seed=42)
        results = engine.walk_forward(sample_klines, "BTC-USDT", n_splits=3)
        assert len(results) == 3
        for r in results:
            assert r.initial_equity == 10_000.0
            assert r.final_equity > 0

    def test_walk_forward_insufficient_data(self):
        """Walk-forward falls back to single run when data is too short."""
        engine = BacktestEngine(strategies=[TrendFollowing(), RiskOff()], seed=42)
        tiny = [[i * 3600, "100", "101", "102", "99", "500", "50000"] for i in range(10)]
        results = engine.walk_forward(tiny, "X-USDT", n_splits=5, warmup=60)
        assert len(results) == 1

    def test_export_backtest_report(self, sample_klines, tmp_path):
        """export_backtest_report should write JSON with summary and daily/weekly PnL."""
        engine = BacktestEngine(strategies=[TrendFollowing(), MeanReversion(), RiskOff()])
        result = engine.run(sample_klines, "BTC-USDT")
        output_file = str(tmp_path / "report.json")
        report = export_backtest_report(result, filepath=output_file)
        assert "summary" in report
        assert "daily_pnl" in report
        assert "weekly_pnl" in report
        assert report["summary"]["total_trades"] == result.total_trades
        assert report["summary"]["expectancy"] == result.expectancy
        assert report["summary"]["turnover"] == result.turnover
        # File was written
        with open(output_file) as f:
            on_disk = json.load(f)
        assert on_disk["summary"]["sharpe_ratio"] == result.sharpe_ratio
