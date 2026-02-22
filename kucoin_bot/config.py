"""Configuration management via environment variables and optional YAML."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class ShortConfig:
    """Short-trading parameters."""

    allow_shorts: bool = True
    # Prefer USDT-margined futures for shorts; fall back to margin if unavailable
    prefer_futures: bool = True
    # Require futures/margin account for shorts (spot-only = no short)
    require_futures_for_short: bool = True
    # Estimated perpetual funding rate per 8-hour period (decimal, e.g. 0.0001 = 0.01 %)
    funding_rate_per_8h: float = 0.0001
    # Margin borrow rate per hour (decimal, e.g. 0.00003 = 0.003 %/hr)
    borrow_rate_per_hour: float = 0.00003
    # Expected holding period used in the EV gate cost estimate (hours)
    expected_holding_hours: float = 24.0


@dataclass
class RiskConfig:
    """Portfolio-level risk parameters."""

    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 10.0
    max_total_exposure_pct: float = 80.0
    max_leverage: float = 3.0
    max_per_position_risk_pct: float = 2.0
    max_correlated_exposure_pct: float = 30.0
    # EV gate: minimum expected-value buffer above round-trip costs (bps)
    min_ev_bps: float = 10.0
    # Minimum bars required between entry signals per symbol
    cooldown_bars: int = 5


@dataclass
class BotConfig:
    """Top-level bot configuration.  All secrets come from env vars only."""

    # API credentials (never logged)
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""

    # Operating mode: LIVE, PAPER, SHADOW, or BACKTEST (default: BACKTEST)
    # LIVE  – real orders; requires LIVE_TRADING=true as a second explicit gate
    # PAPER – simulated orders with live market data (no real risk)
    # SHADOW – signal computation only; logs "would-trade" decisions, no orders
    # BACKTEST – historical simulation
    mode: str = "BACKTEST"
    kill_switch: bool = False
    live_trading: bool = False  # Must be explicitly true to allow LIVE mode

    # Transfers
    allow_internal_transfers: bool = False

    # Observability
    live_diagnostic: bool = False

    # Database
    db_type: str = "sqlite"
    db_url: str = "sqlite:///kucoin_bot.db"

    # Redis
    redis_url: Optional[str] = None

    # Risk
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Short trading
    short: ShortConfig = field(default_factory=ShortConfig)

    # Logging
    log_level: str = "INFO"

    # KuCoin endpoints
    rest_url: str = "https://api.kucoin.com"
    futures_rest_url: str = "https://api-futures.kucoin.com"

    @property
    def is_live(self) -> bool:
        return self.mode.upper() == "LIVE"

    @property
    def is_paper(self) -> bool:
        return self.mode.upper() == "PAPER"

    @property
    def is_shadow(self) -> bool:
        return self.mode.upper() == "SHADOW"


def load_config() -> BotConfig:
    """Load configuration from env vars, then optionally overlay a YAML file."""
    cfg = BotConfig(
        api_key=os.getenv("KUCOIN_API_KEY", ""),
        api_secret=os.getenv("KUCOIN_API_SECRET", ""),
        api_passphrase=os.getenv("KUCOIN_API_PASSPHRASE", ""),
        mode=os.getenv("BOT_MODE", "BACKTEST"),
        kill_switch=os.getenv("KILL_SWITCH", "false").lower() == "true",
        live_trading=os.getenv("LIVE_TRADING", "false").lower() == "true",
        allow_internal_transfers=os.getenv("ALLOW_INTERNAL_TRANSFERS", "false").lower() == "true",
        live_diagnostic=os.getenv("LIVE_DIAGNOSTIC", "false").lower() == "true",
        db_type=os.getenv("DB_TYPE", "sqlite"),
        db_url=os.getenv("DB_URL", "sqlite:///kucoin_bot.db"),
        redis_url=os.getenv("REDIS_URL"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        risk=RiskConfig(
            max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "3.0")),
            max_drawdown_pct=float(os.getenv("MAX_DRAWDOWN_PCT", "10.0")),
            max_total_exposure_pct=float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "80.0")),
            max_leverage=float(os.getenv("MAX_LEVERAGE", "3.0")),
            max_per_position_risk_pct=float(os.getenv("MAX_PER_POSITION_RISK_PCT", "2.0")),
            max_correlated_exposure_pct=float(os.getenv("MAX_CORRELATED_EXPOSURE_PCT", "30.0")),
            min_ev_bps=float(os.getenv("MIN_EV_BPS", "10.0")),
            cooldown_bars=int(os.getenv("COOLDOWN_BARS", "5")),
        ),
        short=ShortConfig(
            allow_shorts=os.getenv("ALLOW_SHORTS", "true").lower() == "true",
            prefer_futures=os.getenv("SHORT_PREFER_FUTURES", "true").lower() == "true",
            require_futures_for_short=os.getenv("REQUIRE_FUTURES_FOR_SHORT", "true").lower() == "true",
            funding_rate_per_8h=float(os.getenv("FUNDING_RATE_PER_8H", "0.0001")),
            borrow_rate_per_hour=float(os.getenv("BORROW_RATE_PER_HOUR", "0.00003")),
            expected_holding_hours=float(os.getenv("EXPECTED_HOLDING_HOURS", "24.0")),
        ),
    )

    # Overlay YAML if present
    if _DEFAULT_CONFIG_PATH.exists():
        try:
            with open(_DEFAULT_CONFIG_PATH) as fh:
                data = yaml.safe_load(fh) or {}
            if "risk" in data:
                for k, v in data["risk"].items():
                    if hasattr(cfg.risk, k):
                        setattr(cfg.risk, k, float(v))
            for k in ("mode", "db_type", "db_url", "log_level"):
                if k in data:
                    setattr(cfg, k, str(data[k]))
        except Exception:
            logger.warning("Failed to read config.yaml, using env/defaults", exc_info=True)
    else:
        _generate_default_yaml()

    _setup_logging(cfg.log_level)
    return cfg


def _generate_default_yaml() -> None:
    """Write a default config.yaml on first run."""
    defaults = {
        "mode": "BACKTEST",
        "log_level": "INFO",
        "risk": {
            "max_daily_loss_pct": 3.0,
            "max_drawdown_pct": 10.0,
            "max_total_exposure_pct": 80.0,
            "max_leverage": 3.0,
            "max_per_position_risk_pct": 2.0,
            "max_correlated_exposure_pct": 30.0,
            "min_ev_bps": 10.0,
            "cooldown_bars": 5,
        },
        "short": {
            "allow_shorts": True,
            "prefer_futures": True,
            "require_futures_for_short": True,
            "funding_rate_per_8h": 0.0001,
            "borrow_rate_per_hour": 0.00003,
            "expected_holding_hours": 24.0,
        },
    }
    try:
        with open(_DEFAULT_CONFIG_PATH, "w") as fh:
            yaml.dump(defaults, fh, default_flow_style=False)
        logger.info("Generated default config.yaml")
    except OSError:
        pass


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
