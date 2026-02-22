"""Configuration management via environment variables and optional YAML."""

from __future__ import annotations

import enum
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("config.yaml")

# Truthy strings accepted by parse_bool
_TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})
# Falsy strings accepted by parse_bool
_FALSE_STRINGS = frozenset({"0", "false", "no", "off"})


def parse_bool(value: str) -> bool:
    """Parse a string to bool, accepting common truthy/falsy variants.

    Raises ValueError for unrecognised values.
    """
    lower = value.strip().lower()
    if lower in _TRUE_STRINGS:
        return True
    if lower in _FALSE_STRINGS:
        return False
    raise ValueError(
        f"Cannot parse {value!r} as boolean. "
        f"Use one of: {sorted(_TRUE_STRINGS | _FALSE_STRINGS)}"
    )


class Mode(str, enum.Enum):
    """Supported operating modes."""

    LIVE = "LIVE"
    PAPER = "PAPER"
    SHADOW = "SHADOW"
    BACKTEST = "BACKTEST"

    @classmethod
    def from_str(cls, value: str) -> "Mode":
        try:
            return cls(value.strip().upper())
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(f"Unknown mode {value!r}. Valid modes: {valid}") from None


def resolve_mode(
    cli_mode: Optional[str],
    env: dict[str, str],
    yaml_mode: Optional[str],
) -> tuple[Mode, str]:
    """Return ``(Mode, source_description)`` with clear precedence.

    Precedence (highest to lowest):
    1. CLI ``--mode`` flag
    2. Environment variable ``MODE`` or ``BOT_MODE``
    3. YAML ``mode`` key
    4. Hardcoded default (BACKTEST)
    """
    if cli_mode:
        mode = Mode.from_str(cli_mode)
        return mode, f"CLI --mode {cli_mode}"

    env_mode = env.get("MODE") or env.get("BOT_MODE")
    if env_mode:
        mode = Mode.from_str(env_mode)
        source = "env MODE" if env.get("MODE") else "env BOT_MODE"
        return mode, f"{source}={env_mode}"

    if yaml_mode:
        mode = Mode.from_str(yaml_mode)
        return mode, f"config.yaml mode={yaml_mode}"

    return Mode.BACKTEST, "default (BACKTEST)"


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


def load_config(cli_mode: Optional[str] = None) -> BotConfig:
    """Load configuration with clear precedence: CLI > env vars > YAML > defaults.

    Args:
        cli_mode: Mode string from a CLI ``--mode`` flag, or ``None``.
    """
    # Step 1: load YAML as lowest-priority source (below env vars)
    yaml_data: dict = {}
    if _DEFAULT_CONFIG_PATH.exists():
        try:
            with open(_DEFAULT_CONFIG_PATH) as fh:
                yaml_data = yaml.safe_load(fh) or {}
        except Exception:
            logger.warning("Failed to read config.yaml, using env/defaults", exc_info=True)
    else:
        _generate_default_yaml()

    yaml_risk: dict = yaml_data.get("risk", {})

    # Step 2: resolve mode with explicit source logging
    mode_val, mode_source = resolve_mode(cli_mode, dict(os.environ), yaml_data.get("mode"))
    # Log at WARNING so it appears before _setup_logging() configures the level
    logger.warning("Mode resolved: %s (source: %s)", mode_val.value, mode_source)

    # Step 3: env vars take precedence over YAML for scalar settings
    def _env_or_yaml(env_key: str, yaml_key: str, default: str) -> str:
        env_val = os.getenv(env_key)
        if env_val is not None and env_val != "":
            return env_val
        yaml_val = yaml_data.get(yaml_key, default)
        # Treat explicit null/None or blank strings in YAML as missing
        if yaml_key in yaml_data:
            if yaml_val is None:
                yaml_val = default
            elif isinstance(yaml_val, str) and yaml_val.strip() == "":
                yaml_val = default
        return str(yaml_val)

    def _bool_env(env_key: str, default: str = "false") -> bool:
        raw = os.getenv(env_key, default)
        try:
            return parse_bool(raw)
        except ValueError:
            logger.warning("Invalid boolean value %r for %s, using %s", raw, env_key, default)
            return parse_bool(default)

    cfg = BotConfig(
        api_key=os.getenv("KUCOIN_API_KEY", ""),
        api_secret=os.getenv("KUCOIN_API_SECRET", ""),
        api_passphrase=os.getenv("KUCOIN_API_PASSPHRASE", ""),
        mode=mode_val.value,
        kill_switch=_bool_env("KILL_SWITCH"),
        live_trading=_bool_env("LIVE_TRADING"),
        allow_internal_transfers=_bool_env("ALLOW_INTERNAL_TRANSFERS"),
        live_diagnostic=_bool_env("LIVE_DIAGNOSTIC"),
        db_type=_env_or_yaml("DB_TYPE", "db_type", "sqlite"),
        db_url=_env_or_yaml("DB_URL", "db_url", "sqlite:///kucoin_bot.db"),
        redis_url=os.getenv("REDIS_URL") or yaml_data.get("redis_url"),
        log_level=_env_or_yaml("LOG_LEVEL", "log_level", "INFO"),
        risk=RiskConfig(
            max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT") or yaml_risk.get("max_daily_loss_pct", 3.0)),
            max_drawdown_pct=float(os.getenv("MAX_DRAWDOWN_PCT") or yaml_risk.get("max_drawdown_pct", 10.0)),
            max_total_exposure_pct=float(os.getenv("MAX_TOTAL_EXPOSURE_PCT") or yaml_risk.get("max_total_exposure_pct", 80.0)),
            max_leverage=float(os.getenv("MAX_LEVERAGE") or yaml_risk.get("max_leverage", 3.0)),
            max_per_position_risk_pct=float(os.getenv("MAX_PER_POSITION_RISK_PCT") or yaml_risk.get("max_per_position_risk_pct", 2.0)),
            max_correlated_exposure_pct=float(os.getenv("MAX_CORRELATED_EXPOSURE_PCT") or yaml_risk.get("max_correlated_exposure_pct", 30.0)),
            min_ev_bps=float(os.getenv("MIN_EV_BPS") or yaml_risk.get("min_ev_bps", 10.0)),
            cooldown_bars=int(os.getenv("COOLDOWN_BARS") or yaml_risk.get("cooldown_bars", 5)),
        ),
        short=ShortConfig(
            allow_shorts=_bool_env("ALLOW_SHORTS", "true"),
            prefer_futures=_bool_env("SHORT_PREFER_FUTURES", "true"),
            require_futures_for_short=_bool_env("REQUIRE_FUTURES_FOR_SHORT", "true"),
            funding_rate_per_8h=float(os.getenv("FUNDING_RATE_PER_8H", "0.0001")),
            borrow_rate_per_hour=float(os.getenv("BORROW_RATE_PER_HOUR", "0.00003")),
            expected_holding_hours=float(os.getenv("EXPECTED_HOLDING_HOURS", "24.0")),
        ),
    )

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
