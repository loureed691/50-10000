"""Configuration management via environment variables."""

from __future__ import annotations

import enum
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

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
    raise ValueError(f"Cannot parse {value!r} as boolean. " f"Use one of: {sorted(_TRUE_STRINGS | _FALSE_STRINGS)}")


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


def resolve_mode(env: dict[str, str]) -> tuple[Mode, str]:
    """Return ``(Mode, source_description)`` with env var precedence.

    Precedence (highest to lowest):
    1. Environment variable ``MODE`` or ``BOT_MODE``
    2. Hardcoded default (BACKTEST)
    """
    env_mode = env.get("MODE") or env.get("BOT_MODE")
    if env_mode:
        mode = Mode.from_str(env_mode)
        source = "env MODE" if env.get("MODE") else "env BOT_MODE"
        return mode, f"{source}={env_mode}"

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
    expected_holding_hours: float = 4.0


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
    min_ev_bps: float = 15.0
    # Minimum bars required between entry signals per symbol
    cooldown_bars: int = 3
    # Circuit breaker: disabled by default
    circuit_breaker_enabled: bool = False


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

    # Transfers (default True so futures entries can move margin automatically)
    allow_internal_transfers: bool = True

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

    # Maximum number of symbols to scan from the market universe per cycle.
    # 0 means no limit (scan all eligible pairs).
    # Symbols with open positions are always included regardless of this cap.
    max_symbols: int = 0

    # Kline interval used for the slow-path (signal + allocation) cadence.
    # Must match a key in ``_KLINE_PERIOD_SECONDS`` (e.g. "5min", "15min",
    # "1hour").  Shorter candles let the bot react faster to price moves.
    kline_type: str = "15min"

    # Fast-path interval (seconds): how often the fast loop runs for
    # order-polling, stop checks, circuit-breaker evaluation, and
    # cancel/flatten operations.  Slow path (klines + regime + allocation)
    # runs only when a new candle has closed.
    fast_interval: int = 30

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
    """Load configuration from environment variables (and ``.env`` file)."""
    load_dotenv()  # load .env file into os.environ so users don't have to source it
    # Resolve mode from env vars
    mode_val, mode_source = resolve_mode(dict(os.environ))
    # Log at WARNING so it appears before _setup_logging() configures the level
    logger.warning("Mode resolved: %s (source: %s)", mode_val.value, mode_source)

    def _bool_env(env_key: str, default: str = "false") -> bool:
        raw = os.getenv(env_key, default)
        try:
            return parse_bool(raw)
        except ValueError:
            logger.warning("Invalid boolean value %r for %s, using %s", raw, env_key, default)
            return parse_bool(default)

    def _float_env(env_key: str, default: float) -> float:
        raw = os.getenv(env_key, "")
        if not raw.strip():
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid float value %r for %s, using %s", raw, env_key, default)
            return default

    def _int_env(env_key: str, default: int) -> int:
        raw = os.getenv(env_key, "")
        if not raw.strip():
            return default
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid int value %r for %s, using %s", raw, env_key, default)
            return default

    cfg = BotConfig(
        api_key=os.getenv("KUCOIN_API_KEY", ""),
        api_secret=os.getenv("KUCOIN_API_SECRET", ""),
        api_passphrase=os.getenv("KUCOIN_API_PASSPHRASE", ""),
        mode=mode_val.value,
        kill_switch=_bool_env("KILL_SWITCH"),
        live_trading=_bool_env("LIVE_TRADING"),
        allow_internal_transfers=_bool_env("ALLOW_INTERNAL_TRANSFERS", "true"),
        live_diagnostic=_bool_env("LIVE_DIAGNOSTIC"),
        db_type=os.getenv("DB_TYPE", "sqlite"),
        db_url=os.getenv("DB_URL", "sqlite:///kucoin_bot.db"),
        redis_url=os.getenv("REDIS_URL") or None,
        max_symbols=_int_env("MAX_SYMBOLS", 0),
        kline_type=os.getenv("KLINE_TYPE", "15min"),
        fast_interval=_int_env("FAST_INTERVAL", 30),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        risk=RiskConfig(
            max_daily_loss_pct=_float_env("MAX_DAILY_LOSS_PCT", 3.0),
            max_drawdown_pct=_float_env("MAX_DRAWDOWN_PCT", 10.0),
            max_total_exposure_pct=_float_env("MAX_TOTAL_EXPOSURE_PCT", 80.0),
            max_leverage=_float_env("MAX_LEVERAGE", 3.0),
            max_per_position_risk_pct=_float_env("MAX_PER_POSITION_RISK_PCT", 2.0),
            max_correlated_exposure_pct=_float_env("MAX_CORRELATED_EXPOSURE_PCT", 30.0),
            min_ev_bps=_float_env("MIN_EV_BPS", 15.0),
            cooldown_bars=_int_env("COOLDOWN_BARS", 3),
            circuit_breaker_enabled=_bool_env("CIRCUIT_BREAKER_ENABLED"),
        ),
        short=ShortConfig(
            allow_shorts=_bool_env("ALLOW_SHORTS", "true"),
            prefer_futures=_bool_env("SHORT_PREFER_FUTURES", "true"),
            require_futures_for_short=_bool_env("REQUIRE_FUTURES_FOR_SHORT", "true"),
            funding_rate_per_8h=_float_env("FUNDING_RATE_PER_8H", 0.0001),
            borrow_rate_per_hour=_float_env("BORROW_RATE_PER_HOUR", 0.00003),
            expected_holding_hours=_float_env("EXPECTED_HOLDING_HOURS", 4.0),
        ),
    )

    _setup_logging(cfg.log_level)
    return cfg


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
