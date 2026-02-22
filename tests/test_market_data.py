"""Tests for market universe discovery."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kucoin_bot.api.client import KuCoinClient
from kucoin_bot.services.market_data import MarketDataService


@pytest.mark.asyncio
async def test_refresh_universe_includes_spot_and_futures_usdt():
    client = MagicMock(spec=KuCoinClient)
    client.get_symbols = AsyncMock(return_value=[
        {"symbol": "BTC-USDT", "quoteCurrency": "USDT", "enableTrading": True},
        {"symbol": "ETH-BTC", "quoteCurrency": "BTC", "enableTrading": True},
    ])
    client.get_futures_contracts = AsyncMock(return_value=[
        {"symbol": "XBTUSDTM", "quoteCurrency": "USDT", "markPrice": "30000"},
    ])
    client.get_ticker = AsyncMock(return_value={"price": "30000", "bestBid": "29999", "bestAsk": "30001"})

    service = MarketDataService(client=client)
    await service.refresh_universe()

    assert "BTC-USDT" in service.universe
    assert "XBTUSDTM" in service.universe
    assert service.universe["XBTUSDTM"].market_type == "futures"
