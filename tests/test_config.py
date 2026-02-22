"""Tests for config module."""

from __future__ import annotations

import os
import sys
import pytest
from unittest.mock import patch
from kucoin_bot.config import load_config, BotConfig, RiskConfig, parse_bool, Mode, resolve_mode


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
        monkeypatch.setattr(sys, "argv", ["kucoin-bot"])

        from kucoin_bot.__main__ import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


class TestParseBool:
    def test_truthy_variants(self):
        for val in ("true", "True", "TRUE", "1", "yes", "Yes", "YES", "on", "ON"):
            assert parse_bool(val) is True, f"Expected True for {val!r}"

    def test_falsy_variants(self):
        for val in ("false", "False", "FALSE", "0", "no", "No", "NO", "off", "OFF"):
            assert parse_bool(val) is False, f"Expected False for {val!r}"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_bool("maybe")

    def test_whitespace_stripped(self):
        assert parse_bool("  true  ") is True
        assert parse_bool("  false  ") is False


class TestResolveMode:
    def test_cli_overrides_all(self):
        mode, src = resolve_mode("live", {"BOT_MODE": "BACKTEST"}, "BACKTEST")
        assert mode == Mode.LIVE
        assert "CLI" in src

    def test_env_mode_overrides_yaml(self):
        mode, src = resolve_mode(None, {"MODE": "PAPER"}, "BACKTEST")
        assert mode == Mode.PAPER
        assert "env MODE" in src

    def test_env_bot_mode_overrides_yaml(self):
        mode, src = resolve_mode(None, {"BOT_MODE": "LIVE"}, "BACKTEST")
        assert mode == Mode.LIVE
        assert "env BOT_MODE" in src

    def test_mode_env_takes_precedence_over_bot_mode(self):
        """MODE env var shadows BOT_MODE when both are set."""
        mode, src = resolve_mode(None, {"MODE": "PAPER", "BOT_MODE": "LIVE"}, None)
        assert mode == Mode.PAPER

    def test_yaml_used_when_no_env(self):
        mode, src = resolve_mode(None, {}, "SHADOW")
        assert mode == Mode.SHADOW
        assert "config.yaml" in src

    def test_default_is_backtest(self):
        mode, src = resolve_mode(None, {}, None)
        assert mode == Mode.BACKTEST
        assert "default" in src

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            resolve_mode("bogus", {}, None)


class TestModeResolutionIntegration:
    """Integration tests for mode resolution via load_config()."""

    def test_env_bot_mode_live_overrides_yaml_backtest(self, monkeypatch, tmp_path):
        """BOT_MODE=LIVE env var must NOT be overridden by config.yaml mode=BACKTEST."""
        import yaml
        (tmp_path / "config.yaml").write_text(yaml.dump({"mode": "BACKTEST"}))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("BOT_MODE", "LIVE")

        cfg = load_config()
        assert cfg.mode == "LIVE"

    def test_cli_mode_overrides_env_and_yaml(self, monkeypatch, tmp_path):
        """CLI --mode live must win over both env BOT_MODE and config.yaml."""
        import yaml
        (tmp_path / "config.yaml").write_text(yaml.dump({"mode": "BACKTEST"}))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("BOT_MODE", "BACKTEST")

        cfg = load_config(cli_mode="live")
        assert cfg.mode == "LIVE"

    def test_live_trading_true_string(self, monkeypatch, tmp_path):
        """LIVE_TRADING=1 and =yes must be accepted as truthy."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LIVE_TRADING", "1")
        cfg = load_config()
        assert cfg.live_trading is True

        monkeypatch.setenv("LIVE_TRADING", "yes")
        cfg = load_config()
        assert cfg.live_trading is True

    def test_live_trading_false_string(self, monkeypatch, tmp_path):
        """LIVE_TRADING=0 and =no must be accepted as falsy."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LIVE_TRADING", "0")
        cfg = load_config()
        assert cfg.live_trading is False

        monkeypatch.setenv("LIVE_TRADING", "no")
        cfg = load_config()
        assert cfg.live_trading is False

    def test_live_mode_missing_api_keys_exits_nonzero(self, monkeypatch, tmp_path):
        """LIVE mode with LIVE_TRADING=true but no API keys → RuntimeError → exit(1)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("BOT_MODE", "LIVE")
        monkeypatch.setenv("LIVE_TRADING", "true")
        monkeypatch.delenv("KUCOIN_API_KEY", raising=False)
        monkeypatch.delenv("KUCOIN_API_SECRET", raising=False)
        monkeypatch.delenv("KUCOIN_API_PASSPHRASE", raising=False)
        monkeypatch.setattr(sys, "argv", ["kucoin-bot"])

        from kucoin_bot.__main__ import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

