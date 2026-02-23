"""Tests for market universe discovery."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kucoin_bot.api.client import KuCoinClient
from kucoin_bot.services.market_data import _KLINE_PERIOD_SECONDS, MarketDataService


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
