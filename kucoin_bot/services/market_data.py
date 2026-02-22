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


@dataclass
class MarketDataService:
    """Discovers and maintains the active USDT market universe."""

    client: KuCoinClient
    universe: Dict[str, MarketInfo] = field(default_factory=dict)
    _kline_cache: Dict[str, list] = field(default_factory=dict)
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

        # Enrich with ticker data for top candidates (batch)
        for sym, info in list(eligible.items()):
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
            sym: info
            for sym, info in eligible.items()
            if info.spread_bps <= MAX_SPREAD_BPS and info.last_price > 0
        }
        logger.info("Market universe: %d USDT pairs", len(self.universe))

    async def get_klines(self, symbol: str, kline_type: str = "1hour", bars: int = 200) -> List[list]:
        """Fetch klines with caching."""
        cache_key = f"{symbol}:{kline_type}"
        now = int(time.time())
        data = await self.client.get_klines(symbol, kline_type, start=now - bars * 3600, end=now)
        self._kline_cache[cache_key] = data
        return data

    def get_symbols(self) -> List[str]:
        return list(self.universe.keys())

    def get_info(self, symbol: str) -> Optional[MarketInfo]:
        return self.universe.get(symbol)
