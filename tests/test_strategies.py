"""Tests for strategies."""

from __future__ import annotations

import pytest

from kucoin_bot.services.signal_engine import SignalScores, Regime
from kucoin_bot.strategies.trend import TrendFollowing
from kucoin_bot.strategies.mean_reversion import MeanReversion
from kucoin_bot.strategies.volatility_breakout import VolatilityBreakout
from kucoin_bot.strategies.scalping import Scalping
from kucoin_bot.strategies.hedge import HedgeMode
from kucoin_bot.strategies.risk_off import RiskOff


class TestTrendFollowing:
    def test_preconditions_met(self):
        strat = TrendFollowing()
        sig = SignalScores(
            symbol="BTC-USDT", regime=Regime.TRENDING_UP,
            trend_strength=0.6, confidence=0.5, momentum=0.3,
        )
        assert strat.preconditions_met(sig) is True

    def test_entry_long(self):
        strat = TrendFollowing()
        sig = SignalScores(
            symbol="BTC-USDT", regime=Regime.TRENDING_UP,
            trend_strength=0.6, confidence=0.5, momentum=0.3,
        )
        decision = strat.evaluate(sig, None, None, 30000.0)
        assert decision.action == "entry_long"
        assert decision.stop_price is not None

    def test_entry_short(self):
        strat = TrendFollowing()
        sig = SignalScores(
            symbol="BTC-USDT", regime=Regime.TRENDING_DOWN,
            trend_strength=0.6, confidence=0.5, momentum=-0.3,
        )
        decision = strat.evaluate(sig, None, None, 30000.0)
        assert decision.action == "entry_short"

    def test_trailing_stop_exit(self):
        strat = TrendFollowing(trail_pct=0.02)
        sig = SignalScores(
            symbol="BTC-USDT", regime=Regime.TRENDING_UP,
            trend_strength=0.6, confidence=0.5, momentum=0.3,
        )
        # Price dropped below trailing stop
        decision = strat.evaluate(sig, "long", 30000.0, 29000.0)
        assert decision.action == "exit"


class TestMeanReversion:
    def test_preconditions(self):
        strat = MeanReversion()
        sig = SignalScores(
            symbol="BTC-USDT", regime=Regime.RANGING,
            volatility=0.2, mean_reversion=0.5,
        )
        assert strat.preconditions_met(sig) is True

    def test_entry_long_oversold(self):
        strat = MeanReversion()
        sig = SignalScores(
            symbol="BTC-USDT", regime=Regime.RANGING,
            mean_reversion=0.6, confidence=0.5,
        )
        decision = strat.evaluate(sig, None, None, 100.0)
        assert decision.action == "entry_long"

    def test_entry_short_overbought(self):
        strat = MeanReversion()
        sig = SignalScores(
            symbol="BTC-USDT", regime=Regime.RANGING,
            mean_reversion=-0.6, confidence=0.5,
        )
        decision = strat.evaluate(sig, None, None, 100.0)
        assert decision.action == "entry_short"


class TestVolatilityBreakout:
    def test_preconditions(self):
        strat = VolatilityBreakout()
        sig = SignalScores(
            symbol="BTC-USDT", regime=Regime.HIGH_VOLATILITY,
            momentum=0.5, volume_anomaly=2.0,
        )
        assert strat.preconditions_met(sig) is True

    def test_no_entry_low_volume(self):
        strat = VolatilityBreakout()
        sig = SignalScores(
            symbol="BTC-USDT", regime=Regime.HIGH_VOLATILITY,
            momentum=0.5, volume_anomaly=0.5,
        )
        assert strat.preconditions_met(sig) is False


class TestRiskOff:
    def test_always_applicable(self):
        strat = RiskOff()
        sig = SignalScores(symbol="BTC-USDT")
        assert strat.preconditions_met(sig) is True

    def test_exits_existing_position(self):
        strat = RiskOff()
        sig = SignalScores(symbol="BTC-USDT")
        decision = strat.evaluate(sig, "long", 100.0, 95.0)
        assert decision.action == "exit"

    def test_holds_when_flat(self):
        strat = RiskOff()
        sig = SignalScores(symbol="BTC-USDT")
        decision = strat.evaluate(sig, None, None, 100.0)
        assert decision.action == "hold"


class TestHedgeMode:
    def test_hedge_on_downtrend(self):
        strat = HedgeMode()
        sig = SignalScores(
            symbol="BTC-USDT", regime=Regime.TRENDING_DOWN,
            volatility=0.5, confidence=0.6,
        )
        assert strat.preconditions_met(sig) is True
        decision = strat.evaluate(sig, "long", 30000.0, 29000.0)
        assert decision.action == "entry_short"


class TestScalping:
    def test_preconditions(self):
        strat = Scalping()
        sig = SignalScores(
            symbol="BTC-USDT", regime=Regime.RANGING,
            volatility=0.1, confidence=0.3,
        )
        assert strat.preconditions_met(sig) is True

    def test_entry_on_imbalance(self):
        strat = Scalping()
        sig = SignalScores(
            symbol="BTC-USDT", regime=Regime.RANGING,
            orderbook_imbalance=0.5, mean_reversion=0.3,
            confidence=0.5,
        )
        decision = strat.evaluate(sig, None, None, 100.0)
        assert decision.action == "entry_long"
