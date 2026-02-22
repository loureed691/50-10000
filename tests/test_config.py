"""Tests for config module."""

from __future__ import annotations

import os
import sys
import pytest
from unittest.mock import patch
from kucoin_bot.config import load_config, BotConfig, RiskConfig


class TestConfig:
    def test_default_config(self):
        cfg = load_config()
        assert cfg.mode == "BACKTEST"
        assert cfg.kill_switch is False
        assert cfg.risk.max_daily_loss_pct == 3.0
        assert cfg.risk.max_leverage == 3.0

    def test_is_live(self):
        cfg = BotConfig(mode="LIVE")
        assert cfg.is_live is True
        cfg2 = BotConfig(mode="BACKTEST")
        assert cfg2.is_live is False

    def test_is_paper(self):
        cfg = BotConfig(mode="PAPER")
        assert cfg.is_paper is True
        assert cfg.is_live is False
        assert cfg.is_shadow is False

    def test_is_shadow(self):
        cfg = BotConfig(mode="SHADOW")
        assert cfg.is_shadow is True
        assert cfg.is_live is False
        assert cfg.is_paper is False

    def test_risk_config_defaults(self):
        rc = RiskConfig()
        assert rc.max_drawdown_pct == 10.0
        assert rc.max_total_exposure_pct == 80.0
        assert rc.max_per_position_risk_pct == 2.0
        assert rc.min_ev_bps == 10.0
        assert rc.cooldown_bars == 5

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MAX_LEVERAGE", "5.0")
        monkeypatch.setenv("BOT_MODE", "LIVE")
        # Prevent YAML overlay from interfering
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.risk.max_leverage == 5.0
        assert cfg.mode == "LIVE"

    def test_live_trading_flag_loaded_from_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LIVE_TRADING", "true")
        monkeypatch.setenv("BOT_MODE", "LIVE")
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.live_trading is True

    def test_live_trading_defaults_false(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LIVE_TRADING", raising=False)
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.live_trading is False

    def test_ev_bps_and_cooldown_from_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MIN_EV_BPS", "25.0")
        monkeypatch.setenv("COOLDOWN_BARS", "10")
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.risk.min_ev_bps == 25.0
        assert cfg.risk.cooldown_bars == 10

    def test_live_mode_refused_without_live_trading(self, monkeypatch, tmp_path):
        """main() must exit(1) when BOT_MODE=LIVE but LIVE_TRADING!=true."""
        monkeypatch.setenv("BOT_MODE", "LIVE")
        monkeypatch.delenv("LIVE_TRADING", raising=False)
        monkeypatch.chdir(tmp_path)

        from kucoin_bot.__main__ import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
