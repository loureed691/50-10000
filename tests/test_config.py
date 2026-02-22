"""Tests for config module."""

from __future__ import annotations

import os
import pytest
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

    def test_risk_config_defaults(self):
        rc = RiskConfig()
        assert rc.max_drawdown_pct == 10.0
        assert rc.max_total_exposure_pct == 80.0
        assert rc.max_per_position_risk_pct == 2.0

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MAX_LEVERAGE", "5.0")
        monkeypatch.setenv("BOT_MODE", "LIVE")
        # Prevent YAML overlay from interfering
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.risk.max_leverage == 5.0
        assert cfg.mode == "LIVE"

    def test_internal_transfers_require_ack(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ALLOW_INTERNAL_TRANSFERS", "true")
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.allow_internal_transfers is False

        monkeypatch.setenv("INTERNAL_TRANSFERS_ACK", "I_UNDERSTAND_INTERNAL_TRANSFERS_RISK")
        cfg = load_config()
        assert cfg.allow_internal_transfers is True
