"""Tests for risk manager."""

from __future__ import annotations

import pytest

from kucoin_bot.config import RiskConfig
from kucoin_bot.services.risk_manager import RiskManager, PositionInfo
from kucoin_bot.services.signal_engine import SignalScores, Regime


class TestRiskManager:
    def _make_risk_mgr(self, equity: float = 10_000.0) -> RiskManager:
        rm = RiskManager(config=RiskConfig())
        rm.update_equity(equity)
        return rm

    def test_daily_loss_check(self):
        rm = self._make_risk_mgr(10_000)
        rm.daily_pnl = -250  # 2.5% loss
        assert rm.check_daily_loss() is False  # under 3%
        rm.daily_pnl = -350  # 3.5% loss
        assert rm.check_daily_loss() is True

    def test_drawdown_check(self):
        rm = self._make_risk_mgr(10_000)
        rm.peak_equity = 10_000
        rm.current_equity = 9_100  # 9% drawdown
        assert rm.check_drawdown() is False
        rm.current_equity = 8_900  # 11% drawdown
        assert rm.check_drawdown() is True

    def test_circuit_breaker(self):
        rm = self._make_risk_mgr(10_000)
        rm.daily_pnl = -500  # 5% daily loss
        assert rm.check_circuit_breaker() is True
        assert rm.circuit_breaker_active is True

    def test_position_sizing_respects_caps(self):
        rm = self._make_risk_mgr(10_000)
        signals = SignalScores(symbol="BTC-USDT", confidence=0.8, volatility=0.3)
        size = rm.compute_position_size("BTC-USDT", 30000, 0.3, signals)
        # Should be > 0 but <= max risk per position
        assert size > 0
        max_risk = 10_000 * (2.0 / 100)  # 2% per position
        assert size <= max_risk * 1.01  # small tolerance

    def test_position_sizing_zero_on_circuit_breaker(self):
        rm = self._make_risk_mgr(10_000)
        rm.circuit_breaker_active = True
        signals = SignalScores(symbol="BTC-USDT", confidence=0.8)
        size = rm.compute_position_size("BTC-USDT", 30000, 0.3, signals)
        assert size == 0.0

    def test_leverage_computation(self):
        rm = self._make_risk_mgr(10_000)
        signals = SignalScores(symbol="BTC-USDT", confidence=0.3, volatility=0.2)
        lev = rm.compute_leverage(signals, 0.2)
        assert lev == 1.0  # low confidence => no leverage

        signals_high = SignalScores(symbol="BTC-USDT", confidence=0.9, volatility=0.2)
        lev_high = rm.compute_leverage(signals_high, 0.2)
        assert lev_high > 1.0
        assert lev_high <= 3.0  # max leverage cap

    def test_exposure_check(self):
        rm = self._make_risk_mgr(10_000)
        rm.positions["BTC-USDT"] = PositionInfo(
            symbol="BTC-USDT", side="long", size=0.3, current_price=30000, leverage=1.0
        )
        # 0.3 * 30000 = 9000 => 90% exposure
        assert rm.check_total_exposure() is True

    def test_update_position_removal(self):
        rm = self._make_risk_mgr()
        rm.positions["BTC-USDT"] = PositionInfo(symbol="BTC-USDT", side="long", size=1.0)
        rm.update_position("BTC-USDT", PositionInfo(symbol="BTC-USDT", side="long", size=0))
        assert "BTC-USDT" not in rm.positions

    def test_daily_reset(self):
        rm = self._make_risk_mgr(10_000)
        rm.daily_pnl = -500
        rm.circuit_breaker_active = True
        rm.reset_daily()
        assert rm.daily_pnl == 0.0
        # CB stays active if drawdown is still breached
