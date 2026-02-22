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
_INTERNAL_TRANSFER_ACK = "I_UNDERSTAND_INTERNAL_TRANSFERS_RISK"


@dataclass
class RiskConfig:
    """Portfolio-level risk parameters."""

    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 10.0
    max_total_exposure_pct: float = 80.0
    max_leverage: float = 3.0
    max_per_position_risk_pct: float = 2.0
    max_correlated_exposure_pct: float = 30.0


@dataclass
class BotConfig:
    """Top-level bot configuration.  All secrets come from env vars only."""

    # API credentials (never logged)
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""

    # Operating mode
    mode: str = "BACKTEST"  # LIVE or BACKTEST
    kill_switch: bool = False

    # Transfers
    allow_internal_transfers: bool = False

    # Database
    db_type: str = "sqlite"
    db_url: str = "sqlite:///kucoin_bot.db"

    # Redis
    redis_url: Optional[str] = None

    # Risk
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Logging
    log_level: str = "INFO"

    # KuCoin endpoints
    rest_url: str = "https://api.kucoin.com"
    futures_rest_url: str = "https://api-futures.kucoin.com"

    @property
    def is_live(self) -> bool:
        return self.mode.upper() == "LIVE"


def load_config() -> BotConfig:
    """Load configuration from env vars, then optionally overlay a YAML file."""
    transfers_enabled = os.getenv("ALLOW_INTERNAL_TRANSFERS", "false").lower() == "true"
    transfers_ack = os.getenv("INTERNAL_TRANSFERS_ACK", "")
    cfg = BotConfig(
        api_key=os.getenv("KUCOIN_API_KEY", ""),
        api_secret=os.getenv("KUCOIN_API_SECRET", ""),
        api_passphrase=os.getenv("KUCOIN_API_PASSPHRASE", ""),
        mode=os.getenv("BOT_MODE", "BACKTEST"),
        kill_switch=os.getenv("KILL_SWITCH", "false").lower() == "true",
        allow_internal_transfers=transfers_enabled and transfers_ack == _INTERNAL_TRANSFER_ACK,
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
    if transfers_enabled and not cfg.allow_internal_transfers:
        logger.warning(
            "Internal transfers requested but acknowledgment is missing or invalid. "
            "Set INTERNAL_TRANSFERS_ACK=%s to enable.",
            _INTERNAL_TRANSFER_ACK,
        )
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
