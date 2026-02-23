"""Tests for market universe discovery."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kucoin_bot.api.client import KuCoinClient
from kucoin_bot.services.market_data import MarketDataService


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
    client.get_ticker = AsyncMock(return_value={"price": "30000", "bestBid": "29999", "bestAsk": "30001"})

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
    client.get_ticker = AsyncMock(return_value={"price": "30000", "bestBid": "29999", "bestAsk": "30001"})

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
    client.get_ticker = AsyncMock(return_value={"price": "30000", "bestBid": "29999", "bestAsk": "30001"})

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
    client.get_ticker = AsyncMock(return_value={"price": "30000", "bestBid": "29999", "bestAsk": "30001"})

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
    client.get_ticker = AsyncMock(return_value={"price": "30000", "bestBid": "29999", "bestAsk": "30001"})

    service = MarketDataService(client=client)
    await service.refresh_universe()

    assert "VALIDUSDTM" in service.universe
    assert len(service.universe) == 2
