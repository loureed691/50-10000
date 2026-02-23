"""Market Data Service – discovers and filters USDT markets, streams data."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from kucoin_bot.api.client import KuCoinClient

logger = logging.getLogger(__name__)

MAX_SPREAD_BPS = 100  # 1 %
_KLINE_CACHE_TTL = 60.0  # seconds


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


@dataclass
class MarketDataService:
    """Discovers and maintains the active USDT market universe."""

    client: KuCoinClient
    universe: Dict[str, MarketInfo] = field(default_factory=dict)
    _kline_cache: Dict[str, tuple] = field(default_factory=dict)
    _refresh_interval: float = 300.0  # 5 min

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
                eligible[sym] = MarketInfo(
                    symbol=sym,
                    base=contract.get("baseCurrency", ""),
                    quote="USDT",
                    base_min_size=float(contract.get("lotSize", 0) or 0),
                    base_increment=float(contract.get("multiplier", 0) or 0),
                    price_increment=float(contract.get("tickSize", 0) or 0),
                    min_funds=0.0,
                    last_price=float(contract.get("markPrice", contract.get("lastTradePrice", 0)) or 0),
                    spread_bps=0.0,
                    market_type="futures",
                )
        except Exception:
            logger.warning("Failed to refresh futures universe (spot universe remains available)", exc_info=True)

        # Enrich with ticker data for top candidates (batch)
        for sym, info in list(eligible.items()):
            if info.market_type == "futures":
                continue
            try:
                ticker = await self.client.get_ticker(sym)
                info.last_price = float(ticker.get("price", 0) or 0)
                bid = float(ticker.get("bestBid", 0) or 0)
                ask = float(ticker.get("bestAsk", 0) or 0)
                if bid > 0 and ask > 0:
                    info.spread_bps = (ask - bid) / ((ask + bid) / 2) * 10_000
            except Exception:
                pass

        # Filter
        self.universe = {
            sym: info for sym, info in eligible.items() if info.spread_bps <= MAX_SPREAD_BPS and info.last_price > 0
        }
        logger.info("Market universe: %d USDT pairs", len(self.universe))

    async def get_klines(self, symbol: str, kline_type: str = "1hour", bars: int = 200) -> List[list]:
        """Fetch klines with TTL-based caching to reduce API calls."""
        cache_key = f"{symbol}:{kline_type}"
        now = time.time()

        cached = self._kline_cache.get(cache_key)
        if cached is not None:
            data, ts = cached
            if now - ts < _KLINE_CACHE_TTL:
                return data

        now_int = int(now)
        data = await self.client.get_klines(symbol, kline_type, start=now_int - bars * 3600, end=now_int)
        # KuCoin returns klines in descending time order (newest first).
        # The signal engine expects ascending order (oldest first), so sort
        # by the timestamp field (index 0).
        if data and len(data) > 1:
            data = sorted(data, key=lambda k: int(k[0]))
        self._kline_cache[cache_key] = (data, now)
        return data

    def get_symbols(self) -> List[str]:
        return list(self.universe.keys())

    def get_info(self, symbol: str) -> Optional[MarketInfo]:
        return self.universe.get(symbol)
