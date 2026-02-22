"""Tests for backtest engine."""

from __future__ import annotations

import pytest

from kucoin_bot.config import RiskConfig
from kucoin_bot.backtest.engine import BacktestEngine
from kucoin_bot.strategies.trend import TrendFollowing
from kucoin_bot.strategies.mean_reversion import MeanReversion
from kucoin_bot.strategies.risk_off import RiskOff


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
