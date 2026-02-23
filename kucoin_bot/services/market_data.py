"""Market Data Service – discovers and filters USDT markets, streams data."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from kucoin_bot.api.client import KuCoinClient

logger = logging.getLogger(__name__)

MAX_SPREAD_BPS = 100  # 1 %
MIN_VOL_VALUE = 10_000.0  # minimum 24 h traded amount (USDT)
_KLINE_CACHE_MAX_ENTRIES = 5_000  # LRU cap
_STALE_CLEANUP_INTERVAL = 60.0  # seconds between stale-entry sweeps

# Map kline type strings to their period in seconds for correct time window calculation
_KLINE_PERIOD_SECONDS: Dict[str, int] = {
    "1min": 60,
    "3min": 180,
    "5min": 300,
    "15min": 900,
    "30min": 1800,
    "1hour": 3600,
    "2hour": 7200,
    "4hour": 14400,
    "6hour": 21600,
    "8hour": 28800,
    "12hour": 43200,
    "1day": 86400,
    "1week": 604800,
}

# Map kline type to futures granularity (minutes)
_FUTURES_GRANULARITY: Dict[str, int] = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1hour": 60,
    "2hour": 120,
    "4hour": 240,
    "8hour": 480,
    "12hour": 720,
    "1day": 1440,
    "1week": 10080,
}


def _dynamic_ttl(kline_type: str) -> float:
    """Derive cache TTL from candle period: ``min(300, period / 2)``."""
    period = _KLINE_PERIOD_SECONDS.get(kline_type, 3600)
    return min(300.0, period / 2)


@dataclass
class MarketInfo:
    """Parsed market metadata."""

    symbol: str
    base: str
    quote: str
    base_min_size: float = 0.0
    base_increment: float = 0.0
    price_increment: float = 0.0
    min_funds: float = 0.0
    is_trading: bool = True
    volume_24h: float = 0.0
    last_price: float = 0.0
    spread_bps: float = 0.0
    market_type: str = "spot"
    # Futures-specific fields
    contract_multiplier: float = 0.0
    lot_size: int = 0
    tick_size: float = 0.0
    max_leverage: float = 1.0


@dataclass
class MarketDataService:
    """Discovers and maintains the active USDT market universe."""

    client: KuCoinClient
    universe: Dict[str, MarketInfo] = field(default_factory=dict)
    _kline_cache: OrderedDict[str, tuple[list, float, str]] = field(default_factory=OrderedDict)
    _kline_cache_max_entries: int = _KLINE_CACHE_MAX_ENTRIES
    _refresh_interval: float = 300.0  # 5 min
    _last_cache_cleanup: float = 0.0

    # ── LRU cache helpers ──────────────────────────────────────────

    def _cache_get(self, key: str) -> Optional[tuple[list, float, str]]:
        """Return cached entry and promote it (LRU), or *None*."""
        entry = self._kline_cache.get(key)
        if entry is not None:
            self._kline_cache.move_to_end(key)
        return entry

    def _cache_put(self, key: str, data: list, kline_type: str) -> None:
        """Insert/update a cache entry, enforce LRU cap, run periodic cleanup."""
        now = time.time()
        if key in self._kline_cache:
            self._kline_cache.move_to_end(key)
        self._kline_cache[key] = (data, now, kline_type)
        # Evict least-recently-used entries when over cap
        while len(self._kline_cache) > self._kline_cache_max_entries:
            self._kline_cache.popitem(last=False)
        # Periodic stale-entry sweep
        if now - self._last_cache_cleanup > _STALE_CLEANUP_INTERVAL:
            self._evict_stale_entries()

    def _evict_stale_entries(self) -> int:
        """Remove entries whose candle has already closed. Returns count removed."""
        now = time.time()
        self._last_cache_cleanup = now
        stale_keys = [
            k
            for k, (data, _ts, kt) in self._kline_cache.items()
            if not self._candle_still_fresh(data, kt, now)
        ]
        for k in stale_keys:
            del self._kline_cache[k]
        if stale_keys:
            logger.debug("Evicted %d stale kline cache entries", len(stale_keys))
        return len(stale_keys)

    async def refresh_universe(self) -> None:
        """Fetch all USDT-quoted pairs and filter by liquidity."""
        symbols = await self.client.get_symbols()
        eligible: Dict[str, MarketInfo] = {}
        for s in symbols:
            if s.get("quoteCurrency") != "USDT":
                continue
            if not s.get("enableTrading", True):
                continue
            sym = s["symbol"]
            info = MarketInfo(
                symbol=sym,
                base=s.get("baseCurrency", ""),
                quote="USDT",
                base_min_size=float(s.get("baseMinSize", 0)),
                base_increment=float(s.get("baseIncrement", 0)),
                price_increment=float(s.get("priceIncrement", 0)),
                min_funds=float(s.get("minFunds", 0)),
            )
            eligible[sym] = info

        # Include eligible USDT futures contracts where available
        try:
            contracts = await self.client.get_futures_contracts()
            for contract in contracts:
                quote_currency = contract.get("quoteCurrency")
                quote = quote_currency if quote_currency is not None else contract.get("quoteCurrencyName")
                if quote != "USDT":
                    continue
                sym = contract.get("symbol")
                if not sym:
                    continue
                multiplier = float(contract.get("multiplier", 0) or 0)
                lot_size = int(contract.get("lotSize", 1) or 1)
                tick_size = float(contract.get("tickSize", 0) or 0)
                max_lev = float(contract.get("maxLeverage", 1) or 1)
                eligible[sym] = MarketInfo(
                    symbol=sym,
                    base=contract.get("baseCurrency", ""),
                    quote="USDT",
                    base_min_size=float(lot_size),
                    base_increment=multiplier,
                    price_increment=tick_size,
                    min_funds=0.0,
                    last_price=float(contract.get("markPrice", contract.get("lastTradePrice", 0)) or 0),
                    spread_bps=0.0,
                    market_type="futures",
                    contract_multiplier=multiplier,
                    lot_size=lot_size,
                    tick_size=tick_size,
                    max_leverage=max_lev,
                )
        except Exception:
            logger.warning("Failed to refresh futures universe (spot universe remains available)", exc_info=True)

        # Enrich with bulk ticker data (single request instead of per-symbol)
        try:
            all_tickers = await self.client.get_all_tickers()
            ticker_map = {t["symbol"]: t for t in all_tickers if "symbol" in t}
        except Exception:
            logger.warning("get_all_tickers failed, falling back to empty ticker map", exc_info=True)
            ticker_map = {}

        for sym, info in list(eligible.items()):
            if info.market_type == "futures":
                continue
            ticker = ticker_map.get(sym)
            if ticker is None:
                logger.debug("No ticker data for %s, skipping enrichment", sym)
                continue
            info.last_price = float(ticker.get("last", 0) or 0)
            info.volume_24h = float(ticker.get("volValue", 0) or 0)
            bid = float(ticker.get("buy", 0) or 0)
            ask = float(ticker.get("sell", 0) or 0)
            if bid > 0 and ask > 0:
                info.spread_bps = (ask - bid) / ((ask + bid) / 2) * 10_000

        # Filter: spread, price, and minimum 24h volume for spot pairs
        self.universe = {
            sym: info
            for sym, info in eligible.items()
            if info.spread_bps <= MAX_SPREAD_BPS
            and info.last_price > 0
            and (info.market_type == "futures" or info.volume_24h >= MIN_VOL_VALUE)
        }
        logger.info("Market universe: %d USDT pairs", len(self.universe))

    async def get_klines(self, symbol: str, kline_type: str = "1hour", bars: int = 200) -> List[list]:
        """Fetch klines with TTL-based caching. Routes to spot or futures endpoint."""
        info = self.universe.get(symbol)
        if info and info.market_type == "futures":
            return await self.get_klines_futures(symbol, kline_type, bars)
        return await self.get_klines_spot(symbol, kline_type, bars)

    def _candle_still_fresh(self, klines: List[list], kline_type: str, now: float) -> bool:
        """Return True if no new candle has closed since the last kline.

        Uses the last (most recent) timestamp in the ascending-sorted kline
        array and the candle period to decide whether a fresh API call is
        needed.  Falls back to ``False`` (= refetch) when the data is empty.
        """
        if not klines:
            return False
        period = _KLINE_PERIOD_SECONDS.get(kline_type, 3600)
        try:
            last_ts = int(klines[-1][0])
        except (IndexError, ValueError, TypeError):
            return False
        return now < last_ts + period

    async def get_klines_spot(self, symbol: str, kline_type: str = "1hour", bars: int = 200) -> List[list]:
        """Fetch spot klines with candle-aware caching.

        A new API call is made only when a new candle has closed (i.e. the
        current time exceeds the last candle timestamp plus one period).
        """
        cache_key = f"{symbol}:{kline_type}"
        now = time.time()

        cached = self._cache_get(cache_key)
        if cached is not None:
            cached_data, ts, _kt = cached
            if self._candle_still_fresh(cached_data, kline_type, now):
                return list(cached_data)

        now_int = int(now)
        period = _KLINE_PERIOD_SECONDS.get(kline_type, 3600)
        data = await self.client.get_klines(symbol, kline_type, start=now_int - bars * period, end=now_int)
        # KuCoin returns klines in descending time order (newest first).
        # The signal engine expects ascending order (oldest first), so sort
        # by the timestamp field (index 0).
        if data and len(data) > 1:
            data = sorted(data, key=lambda k: int(k[0]))
        self._cache_put(cache_key, data, kline_type)
        return data

    async def get_klines_futures(self, symbol: str, kline_type: str = "1hour", bars: int = 200) -> List[list]:
        """Fetch futures klines with candle-aware caching."""
        cache_key = f"{symbol}:{kline_type}:futures"
        now = time.time()

        cached = self._cache_get(cache_key)
        if cached is not None:
            cached_data, ts, _kt = cached
            if self._candle_still_fresh(cached_data, kline_type, now):
                return list(cached_data)

        now_int = int(now)
        granularity = _FUTURES_GRANULARITY.get(kline_type, 60)
        period = _KLINE_PERIOD_SECONDS.get(kline_type, 3600)
        start_ms = (now_int - bars * period) * 1000
        end_ms = now_int * 1000
        raw = await self.client.get_futures_klines(symbol, granularity, start=start_ms, end=end_ms)
        # Futures klines format: [time_ms, open, high, low, close, volume]
        # Convert to spot-compatible format: [time_s, open, close, high, low, volume, turnover]
        result: List[list] = []
        for k in raw:
            if len(k) >= 6:
                ts_s = int(k[0]) // 1000 if int(k[0]) > 1e12 else int(k[0])
                o, h, l_, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
                result.append([str(ts_s), str(o), str(c), str(h), str(l_), str(v), str(v * c)])
        if result and len(result) > 1:
            result = sorted(result, key=lambda k: int(k[0]))
        self._cache_put(cache_key, result, kline_type)
        return result

    def get_symbols(self) -> List[str]:
        return list(self.universe.keys())

    def get_info(self, symbol: str) -> Optional[MarketInfo]:
        return self.universe.get(symbol)
