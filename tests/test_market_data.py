"""Tests for market universe discovery."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kucoin_bot.api.client import KuCoinClient
from kucoin_bot.services.market_data import (
    _KLINE_CACHE_MAX_ENTRIES,
    _KLINE_PERIOD_SECONDS,
    MarketDataService,
    _dynamic_ttl,
)


@pytest.mark.asyncio
async def test_refresh_universe_includes_spot_and_futures_usdt():
    client = MagicMock(spec=KuCoinClient)
    client.get_symbols = AsyncMock(
        return_value=[
            {"symbol": "BTC-USDT", "quoteCurrency": "USDT", "enableTrading": True},
            {"symbol": "ETH-BTC", "quoteCurrency": "BTC", "enableTrading": True},
        ]
    )
    client.get_futures_contracts = AsyncMock(
        return_value=[
            {"symbol": "XBTUSDTM", "quoteCurrency": "USDT", "markPrice": "30000"},
        ]
    )
    client.get_all_tickers = AsyncMock(
        return_value=[
            {"symbol": "BTC-USDT", "last": "30000", "buy": "29999", "sell": "30001", "volValue": "500000"},
        ]
    )

    service = MarketDataService(client=client)
    await service.refresh_universe()

    assert "BTC-USDT" in service.universe
    assert "XBTUSDTM" in service.universe
    assert service.universe["XBTUSDTM"].market_type == "futures"


@pytest.mark.asyncio
async def test_refresh_universe_filters_futures_with_zero_or_missing_price():
    client = MagicMock(spec=KuCoinClient)
    client.get_symbols = AsyncMock(
        return_value=[
            {"symbol": "BTC-USDT", "quoteCurrency": "USDT", "enableTrading": True},
        ]
    )
    client.get_futures_contracts = AsyncMock(
        return_value=[
            {"symbol": "ZERO_PRICE", "quoteCurrency": "USDT", "markPrice": "0"},
            {"symbol": "MISSING_MARK", "quoteCurrency": "USDT", "lastTradePrice": "0"},
            {"symbol": "NO_PRICE_FIELDS", "quoteCurrency": "USDT"},
        ]
    )
    client.get_all_tickers = AsyncMock(
        return_value=[
            {"symbol": "BTC-USDT", "last": "30000", "buy": "29999", "sell": "30001", "volValue": "500000"},
        ]
    )

    service = MarketDataService(client=client)
    await service.refresh_universe()

    assert "BTC-USDT" in service.universe
    assert "ZERO_PRICE" not in service.universe
    assert "MISSING_MARK" not in service.universe
    assert "NO_PRICE_FIELDS" not in service.universe


@pytest.mark.asyncio
async def test_refresh_universe_excludes_non_usdt_futures():
    client = MagicMock(spec=KuCoinClient)
    client.get_symbols = AsyncMock(
        return_value=[
            {"symbol": "BTC-USDT", "quoteCurrency": "USDT", "enableTrading": True},
        ]
    )
    client.get_futures_contracts = AsyncMock(
        return_value=[
            {"symbol": "XBTUSDTM", "quoteCurrency": "USDT", "markPrice": "30000"},
            {"symbol": "XBTBTCM", "quoteCurrency": "BTC", "markPrice": "30000"},
        ]
    )
    client.get_all_tickers = AsyncMock(
        return_value=[
            {"symbol": "BTC-USDT", "last": "30000", "buy": "29999", "sell": "30001", "volValue": "500000"},
        ]
    )

    service = MarketDataService(client=client)
    await service.refresh_universe()

    assert "XBTUSDTM" in service.universe
    assert "XBTBTCM" not in service.universe


@pytest.mark.asyncio
async def test_refresh_universe_continues_when_futures_fetch_fails():
    client = MagicMock(spec=KuCoinClient)
    client.get_symbols = AsyncMock(
        return_value=[
            {"symbol": "BTC-USDT", "quoteCurrency": "USDT", "enableTrading": True},
        ]
    )
    client.get_futures_contracts = AsyncMock(side_effect=Exception("futures API error"))
    client.get_all_tickers = AsyncMock(
        return_value=[
            {"symbol": "BTC-USDT", "last": "30000", "buy": "29999", "sell": "30001", "volValue": "500000"},
        ]
    )

    service = MarketDataService(client=client)
    await service.refresh_universe()

    assert "BTC-USDT" in service.universe
    assert len(service.universe) == 1


@pytest.mark.asyncio
async def test_refresh_universe_ignores_futures_with_missing_symbol():
    client = MagicMock(spec=KuCoinClient)
    client.get_symbols = AsyncMock(
        return_value=[
            {"symbol": "BTC-USDT", "quoteCurrency": "USDT", "enableTrading": True},
        ]
    )
    client.get_futures_contracts = AsyncMock(
        return_value=[
            {"quoteCurrency": "USDT", "markPrice": "30000"},
            {"symbol": "VALIDUSDTM", "quoteCurrency": "USDT", "markPrice": "30000"},
        ]
    )
    client.get_all_tickers = AsyncMock(
        return_value=[
            {"symbol": "BTC-USDT", "last": "30000", "buy": "29999", "sell": "30001", "volValue": "500000"},
        ]
    )

    service = MarketDataService(client=client)
    await service.refresh_universe()

    assert "VALIDUSDTM" in service.universe
    assert len(service.universe) == 2


@pytest.mark.asyncio
async def test_refresh_universe_filters_low_volume_spot():
    """Spot pairs with volValue below MIN_VOL_VALUE are excluded."""
    client = MagicMock(spec=KuCoinClient)
    client.get_symbols = AsyncMock(
        return_value=[
            {"symbol": "BTC-USDT", "quoteCurrency": "USDT", "enableTrading": True},
            {"symbol": "LOW-USDT", "quoteCurrency": "USDT", "enableTrading": True},
        ]
    )
    client.get_futures_contracts = AsyncMock(return_value=[])
    client.get_all_tickers = AsyncMock(
        return_value=[
            {"symbol": "BTC-USDT", "last": "30000", "buy": "29999", "sell": "30001", "volValue": "500000"},
            {"symbol": "LOW-USDT", "last": "1.5", "buy": "1.49", "sell": "1.51", "volValue": "50"},
        ]
    )

    service = MarketDataService(client=client)
    await service.refresh_universe()

    assert "BTC-USDT" in service.universe
    assert "LOW-USDT" not in service.universe


@pytest.mark.asyncio
async def test_refresh_universe_volume_filter_does_not_apply_to_futures():
    """Futures contracts bypass the MIN_VOL_VALUE filter."""
    client = MagicMock(spec=KuCoinClient)
    client.get_symbols = AsyncMock(return_value=[])
    client.get_futures_contracts = AsyncMock(
        return_value=[
            {"symbol": "XBTUSDTM", "quoteCurrency": "USDT", "markPrice": "30000"},
        ]
    )
    client.get_all_tickers = AsyncMock(return_value=[])

    service = MarketDataService(client=client)
    await service.refresh_universe()

    assert "XBTUSDTM" in service.universe


@pytest.mark.asyncio
async def test_refresh_universe_populates_volume_24h():
    """volume_24h is populated from allTickers volValue."""
    client = MagicMock(spec=KuCoinClient)
    client.get_symbols = AsyncMock(
        return_value=[
            {"symbol": "BTC-USDT", "quoteCurrency": "USDT", "enableTrading": True},
        ]
    )
    client.get_futures_contracts = AsyncMock(return_value=[])
    client.get_all_tickers = AsyncMock(
        return_value=[
            {"symbol": "BTC-USDT", "last": "30000", "buy": "29999", "sell": "30001", "volValue": "1234567.89"},
        ]
    )

    service = MarketDataService(client=client)
    await service.refresh_universe()

    assert "BTC-USDT" in service.universe
    assert service.universe["BTC-USDT"].volume_24h == pytest.approx(1234567.89)


@pytest.mark.asyncio
async def test_refresh_universe_continues_when_all_tickers_fails():
    """If get_all_tickers fails, spot pairs without enrichment are filtered out."""
    client = MagicMock(spec=KuCoinClient)
    client.get_symbols = AsyncMock(
        return_value=[
            {"symbol": "BTC-USDT", "quoteCurrency": "USDT", "enableTrading": True},
        ]
    )
    client.get_futures_contracts = AsyncMock(
        return_value=[
            {"symbol": "XBTUSDTM", "quoteCurrency": "USDT", "markPrice": "30000"},
        ]
    )
    client.get_all_tickers = AsyncMock(side_effect=Exception("API down"))

    service = MarketDataService(client=client)
    await service.refresh_universe()

    # Spot pair has no price/volume → filtered out; futures survive
    assert "BTC-USDT" not in service.universe
    assert "XBTUSDTM" in service.universe


# ── Candle-aware caching tests ──────────────────────────────────────


class TestCandleStillFresh:
    """Unit tests for MarketDataService._candle_still_fresh()."""

    def _make_service(self) -> MarketDataService:
        client = MagicMock(spec=KuCoinClient)
        return MarketDataService(client=client)

    def test_empty_klines_returns_false(self):
        svc = self._make_service()
        assert svc._candle_still_fresh([], "1hour", 1_700_000_000) is False

    def test_fresh_when_within_candle_period(self):
        svc = self._make_service()
        period = _KLINE_PERIOD_SECONDS["1hour"]  # 3600
        last_ts = 1_700_000_000
        klines = [["1699996400"], [str(last_ts)]]
        # now is within the current candle → still fresh
        assert svc._candle_still_fresh(klines, "1hour", last_ts + period - 1) is True

    def test_stale_when_candle_boundary_crossed(self):
        svc = self._make_service()
        period = _KLINE_PERIOD_SECONDS["1hour"]  # 3600
        last_ts = 1_700_000_000
        klines = [["1699996400"], [str(last_ts)]]
        # now is past the candle close → stale
        assert svc._candle_still_fresh(klines, "1hour", last_ts + period) is False

    def test_invalid_timestamp_returns_false(self):
        svc = self._make_service()
        klines = [["not_a_number"]]
        assert svc._candle_still_fresh(klines, "1hour", 1_700_000_000) is False

    def test_5min_candle_period(self):
        svc = self._make_service()
        period = _KLINE_PERIOD_SECONDS["5min"]  # 300
        last_ts = 1_700_000_000
        klines = [[str(last_ts)]]
        assert svc._candle_still_fresh(klines, "5min", last_ts + period - 1) is True
        assert svc._candle_still_fresh(klines, "5min", last_ts + period) is False


@pytest.mark.asyncio
async def test_get_klines_spot_candle_aware_cache():
    """Spot klines use candle-aware caching: no API call when candle is still fresh."""
    import time

    client = MagicMock(spec=KuCoinClient)
    now = int(time.time())
    period = _KLINE_PERIOD_SECONDS["1hour"]
    # Return klines whose last timestamp is within the current candle period
    klines = [[str(now - period), "1", "2", "3", "4", "5", "6"], [str(now - 10), "1", "2", "3", "4", "5", "6"]]
    client.get_klines = AsyncMock(return_value=klines)

    service = MarketDataService(client=client)
    service.universe["BTC-USDT"] = MagicMock(market_type="spot")

    # First call hits the API
    result1 = await service.get_klines_spot("BTC-USDT", "1hour")
    assert len(result1) == 2
    assert client.get_klines.call_count == 1

    # Second call should use cache (candle is still fresh)
    result2 = await service.get_klines_spot("BTC-USDT", "1hour")
    assert result2 == result1
    assert client.get_klines.call_count == 1  # no additional API call


@pytest.mark.asyncio
async def test_get_klines_futures_candle_aware_cache():
    """Futures klines use candle-aware caching: no API call when candle is still fresh."""
    import time

    client = MagicMock(spec=KuCoinClient)
    now = int(time.time())
    # Return futures klines (format: [time_ms, open, high, low, close, volume])
    klines_raw = [
        [now * 1000 - 3600000, 100, 105, 95, 102, 1000],
        [now * 1000 - 10000, 102, 108, 100, 106, 1200],
    ]
    client.get_futures_klines = AsyncMock(return_value=klines_raw)

    service = MarketDataService(client=client)

    # First call hits the API
    result1 = await service.get_klines_futures("BTC-USDT", "1hour")
    assert len(result1) == 2
    assert client.get_futures_klines.call_count == 1

    # Second call should use cache (candle is still fresh)
    result2 = await service.get_klines_futures("BTC-USDT", "1hour")
    assert result2 == result1
    assert client.get_futures_klines.call_count == 1  # no additional API call


# ── Dynamic TTL tests ───────────────────────────────────────────


class TestDynamicTTL:
    """Unit tests for _dynamic_ttl()."""

    def test_short_period_returns_half(self):
        # 1min → period=60 → min(300, 30) = 30
        assert _dynamic_ttl("1min") == 30.0

    def test_5min_period_returns_half(self):
        # 5min → period=300 → min(300, 150) = 150
        assert _dynamic_ttl("5min") == 150.0

    def test_long_period_capped_at_300(self):
        # 1hour → period=3600 → min(300, 1800) = 300
        assert _dynamic_ttl("1hour") == 300.0

    def test_1day_capped_at_300(self):
        assert _dynamic_ttl("1day") == 300.0

    def test_unknown_type_uses_fallback(self):
        # Unknown → default period=3600 → min(300, 1800) = 300
        assert _dynamic_ttl("unknown") == 300.0


# ── LRU cache tests ────────────────────────────────────────────


class TestLRUCache:
    """Unit tests for LRU cache helpers on MarketDataService."""

    def _make_service(self, max_entries: int = 5_000) -> MarketDataService:
        client = MagicMock(spec=KuCoinClient)
        svc = MarketDataService(client=client)
        svc._kline_cache_max_entries = max_entries
        svc._last_cache_cleanup = float("inf")  # disable auto-cleanup
        return svc

    def test_cache_put_and_get(self):
        svc = self._make_service()
        svc._cache_put("k1", [["100"]], "1hour")
        entry = svc._cache_get("k1")
        assert entry is not None
        data, ts, kt = entry
        assert data == [["100"]]
        assert kt == "1hour"

    def test_cache_get_missing_returns_none(self):
        svc = self._make_service()
        assert svc._cache_get("missing") is None

    def test_lru_eviction_removes_oldest(self):
        svc = self._make_service(max_entries=3)
        svc._cache_put("a", [["1"]], "1min")
        svc._cache_put("b", [["2"]], "1min")
        svc._cache_put("c", [["3"]], "1min")
        # Cache full: [a, b, c]
        svc._cache_put("d", [["4"]], "1min")
        # 'a' should be evicted
        assert svc._cache_get("a") is None
        assert svc._cache_get("d") is not None

    def test_lru_access_promotes_entry(self):
        svc = self._make_service(max_entries=3)
        svc._cache_put("a", [["1"]], "1min")
        svc._cache_put("b", [["2"]], "1min")
        svc._cache_put("c", [["3"]], "1min")
        # Access 'a' to promote it
        svc._cache_get("a")
        # Now insert 'd' → 'b' (oldest untouched) evicted
        svc._cache_put("d", [["4"]], "1min")
        assert svc._cache_get("b") is None
        assert svc._cache_get("a") is not None

    def test_cache_put_updates_existing(self):
        svc = self._make_service()
        svc._cache_put("k1", [["100"]], "1hour")
        svc._cache_put("k1", [["200"]], "1hour")
        entry = svc._cache_get("k1")
        assert entry is not None
        assert entry[0] == [["200"]]
        assert len(svc._kline_cache) == 1

    def test_default_max_entries(self):
        client = MagicMock(spec=KuCoinClient)
        svc = MarketDataService(client=client)
        assert svc._kline_cache_max_entries == _KLINE_CACHE_MAX_ENTRIES


# ── Stale-entry cleanup tests ──────────────────────────────────


class TestStaleEviction:
    """Unit tests for _evict_stale_entries()."""

    def _make_service(self) -> MarketDataService:
        client = MagicMock(spec=KuCoinClient)
        return MarketDataService(client=client)

    def test_evicts_stale_entries(self):
        svc = self._make_service()
        import time

        now = time.time()
        period = _KLINE_PERIOD_SECONDS["1hour"]
        # Fresh entry: last kline ts is recent
        fresh_data = [[str(int(now) - 10)]]
        # Stale entry: last kline ts is old (candle boundary crossed)
        stale_data = [[str(int(now) - period - 100)]]
        svc._kline_cache["fresh"] = (fresh_data, now, "1hour")
        svc._kline_cache["stale"] = (stale_data, now - 999, "1hour")

        removed = svc._evict_stale_entries()
        assert removed == 1
        assert "fresh" in svc._kline_cache
        assert "stale" not in svc._kline_cache

    def test_evict_empty_data(self):
        svc = self._make_service()
        svc._kline_cache["empty"] = ([], 0, "5min")
        removed = svc._evict_stale_entries()
        assert removed == 1
        assert len(svc._kline_cache) == 0

    def test_periodic_cleanup_triggered(self):
        """_cache_put triggers stale cleanup when interval elapsed."""
        svc = self._make_service()
        svc._last_cache_cleanup = 0.0  # long ago
        # Put a stale entry directly, then trigger cleanup via _cache_put
        period = _KLINE_PERIOD_SECONDS["5min"]
        import time

        now = time.time()
        stale_data = [[str(int(now) - period - 100)]]
        svc._kline_cache["stale"] = (stale_data, now - 999, "5min")
        # This _cache_put should trigger cleanup
        svc._cache_put("new", [[str(int(now) - 1)]], "5min")
        assert "stale" not in svc._kline_cache
        assert "new" in svc._kline_cache
