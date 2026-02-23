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
        assert rc.min_ev_bps == 5.0
        assert rc.cooldown_bars == 5

    def test_blank_numeric_env_uses_default(self, monkeypatch):
        """Blank numeric env vars (e.g. MAX_LEVERAGE=) must fall back to defaults."""
        monkeypatch.setenv("MAX_LEVERAGE", "")
        monkeypatch.setenv("COOLDOWN_BARS", "")
        monkeypatch.setenv("FUNDING_RATE_PER_8H", "")
        cfg = load_config()
        assert cfg.risk.max_leverage == 3.0
        assert cfg.risk.cooldown_bars == 5
        assert cfg.short.funding_rate_per_8h == 0.0001

    def test_invalid_numeric_env_uses_default(self, monkeypatch):
        """Invalid numeric env vars must fall back to defaults with a warning."""
        monkeypatch.setenv("MAX_LEVERAGE", "not-a-number")
        cfg = load_config()
        assert cfg.risk.max_leverage == 3.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MAX_LEVERAGE", "5.0")
        monkeypatch.setenv("BOT_MODE", "LIVE")
        cfg = load_config()
        assert cfg.risk.max_leverage == 5.0
        assert cfg.mode == "LIVE"

    def test_live_trading_flag_loaded_from_env(self, monkeypatch):
        monkeypatch.setenv("LIVE_TRADING", "true")
        monkeypatch.setenv("BOT_MODE", "LIVE")
        cfg = load_config()
        assert cfg.live_trading is True

    def test_live_trading_defaults_false(self, monkeypatch):
        monkeypatch.delenv("LIVE_TRADING", raising=False)
        cfg = load_config()
        assert cfg.live_trading is False

    def test_ev_bps_and_cooldown_from_env(self, monkeypatch):
        monkeypatch.setenv("MIN_EV_BPS", "25.0")
        monkeypatch.setenv("COOLDOWN_BARS", "10")
        cfg = load_config()
        assert cfg.risk.min_ev_bps == 25.0
        assert cfg.risk.cooldown_bars == 10

    def test_live_mode_refused_without_live_trading(self, monkeypatch):
        """main() must exit(1) when BOT_MODE=LIVE but LIVE_TRADING!=true."""
        monkeypatch.setenv("BOT_MODE", "LIVE")
        monkeypatch.delenv("LIVE_TRADING", raising=False)
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
    def test_env_mode_overrides_bot_mode(self):
        mode, src = resolve_mode({"MODE": "PAPER", "BOT_MODE": "LIVE"})
        assert mode == Mode.PAPER
        assert "env MODE" in src

    def test_env_bot_mode(self):
        mode, src = resolve_mode({"BOT_MODE": "LIVE"})
        assert mode == Mode.LIVE
        assert "env BOT_MODE" in src

    def test_mode_env_takes_precedence_over_bot_mode(self):
        """MODE env var shadows BOT_MODE when both are set."""
        mode, src = resolve_mode({"MODE": "PAPER", "BOT_MODE": "LIVE"})
        assert mode == Mode.PAPER

    def test_default_is_backtest(self):
        mode, src = resolve_mode({})
        assert mode == Mode.BACKTEST
        assert "default" in src

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            resolve_mode({"BOT_MODE": "bogus"})


class TestDotenvLoading:
    """Verify that load_config() calls load_dotenv to read .env files."""

    def test_load_config_calls_load_dotenv(self, monkeypatch):
        """load_config() must call load_dotenv() so .env files are picked up."""
        from unittest.mock import MagicMock
        mock_ld = MagicMock()
        monkeypatch.setattr("kucoin_bot.config.load_dotenv", mock_ld)
        load_config()
        mock_ld.assert_called_once()

    def test_dotenv_file_sets_bot_mode(self, tmp_path, monkeypatch):
        """A .env file should be loaded so BOT_MODE is available."""
        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text("BOT_MODE=LIVE\nLIVE_TRADING=true\n")
        # Remove any pre-existing env vars so .env is the only source
        monkeypatch.delenv("BOT_MODE", raising=False)
        monkeypatch.delenv("MODE", raising=False)
        monkeypatch.delenv("LIVE_TRADING", raising=False)
        # Patch load_dotenv to load from our tmp .env file
        from functools import partial
        from dotenv import load_dotenv as real_load_dotenv
        monkeypatch.setattr(
            "kucoin_bot.config.load_dotenv",
            partial(real_load_dotenv, dotenv_path=str(dotenv_file)),
        )
        cfg = load_config()
        assert cfg.mode == "LIVE"
        assert cfg.live_trading is True


class TestModeResolutionIntegration:
    """Integration tests for mode resolution via load_config()."""

    def test_env_bot_mode_live(self, monkeypatch):
        """BOT_MODE=LIVE env var sets mode to LIVE."""
        monkeypatch.setenv("BOT_MODE", "LIVE")
        cfg = load_config()
        assert cfg.mode == "LIVE"

    def test_live_trading_true_string(self, monkeypatch):
        """LIVE_TRADING=1 and =yes must be accepted as truthy."""
        monkeypatch.setenv("LIVE_TRADING", "1")
        cfg = load_config()
        assert cfg.live_trading is True

        monkeypatch.setenv("LIVE_TRADING", "yes")
        cfg = load_config()
        assert cfg.live_trading is True

    def test_live_trading_false_string(self, monkeypatch):
        """LIVE_TRADING=0 and =no must be accepted as falsy."""
        monkeypatch.setenv("LIVE_TRADING", "0")
        cfg = load_config()
        assert cfg.live_trading is False

        monkeypatch.setenv("LIVE_TRADING", "no")
        cfg = load_config()
        assert cfg.live_trading is False

    def test_live_mode_missing_api_keys_exits_nonzero(self, monkeypatch):
        """LIVE mode with LIVE_TRADING=true but no API keys → RuntimeError → exit(1)."""
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

    def test_live_mode_unexpected_exception_exits_nonzero(self, monkeypatch):
        """LIVE mode with LIVE_TRADING=true but non-RuntimeError exception → exit(1)."""
        monkeypatch.setenv("BOT_MODE", "LIVE")
        monkeypatch.setenv("LIVE_TRADING", "true")
        monkeypatch.setenv("KUCOIN_API_KEY", "key")
        monkeypatch.setenv("KUCOIN_API_SECRET", "secret")
        monkeypatch.setenv("KUCOIN_API_PASSPHRASE", "passphrase")
        monkeypatch.setattr(sys, "argv", ["kucoin-bot"])

        async def _raise_value_error(cfg):
            raise ValueError("simulated unexpected error")

        from kucoin_bot import __main__ as main_module
        monkeypatch.setattr(main_module, "run_live", _raise_value_error)

        with pytest.raises(SystemExit) as exc_info:
            main_module.main()
        assert exc_info.value.code == 1

    def test_paper_mode_unexpected_exception_exits_nonzero(self, monkeypatch):
        """PAPER mode non-RuntimeError exception → exit(1) (no raw traceback)."""
        monkeypatch.setenv("BOT_MODE", "PAPER")
        monkeypatch.setenv("KUCOIN_API_KEY", "key")
        monkeypatch.setenv("KUCOIN_API_SECRET", "secret")
        monkeypatch.setenv("KUCOIN_API_PASSPHRASE", "passphrase")
        monkeypatch.setattr(sys, "argv", ["kucoin-bot"])

        async def _raise_connection_error(cfg):
            raise ConnectionError("simulated network failure")

        from kucoin_bot import __main__ as main_module
        monkeypatch.setattr(main_module, "run_live", _raise_connection_error)

        with pytest.raises(SystemExit) as exc_info:
            main_module.main()
        assert exc_info.value.code == 1

    def test_help_flag_exits_zero(self, monkeypatch):
        """--help and -h must print usage and exit 0."""
        from kucoin_bot import __main__ as main_module
        for flag in ("--help", "-h"):
            monkeypatch.setattr(sys, "argv", ["kucoin-bot", flag])
            with pytest.raises(SystemExit) as exc_info:
                main_module.main()
            assert exc_info.value.code == 0

    def test_unknown_cli_args_exit_two(self, monkeypatch):
        """Unknown CLI args must exit with code 2."""
        from kucoin_bot import __main__ as main_module
        monkeypatch.setattr(sys, "argv", ["kucoin-bot", "--mode", "live"])
        with pytest.raises(SystemExit) as exc_info:
            main_module.main()
        assert exc_info.value.code == 2


