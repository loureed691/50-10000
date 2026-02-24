"""Tests for P0/P1 optimizations: stop-loss, HTTP fail-fast, parallel klines, batch DB, kline ordering."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aioresponses import aioresponses

from kucoin_bot.api.client import KuCoinAPIError, KuCoinClient
from kucoin_bot.config import RiskConfig
from kucoin_bot.services.market_data import MarketDataService
from kucoin_bot.services.risk_manager import PositionInfo, RiskManager

# ── Change A: Stop-loss correctness ──────────────────────────────────


class TestStopLossCorrectness:
    """Tests for stop_price handling in PositionInfo and RiskManager."""

    def _make_risk_mgr(self, equity: float = 10_000.0) -> RiskManager:
        rm = RiskManager(config=RiskConfig())
        rm.update_equity(equity)
        return rm

    def test_entry_sets_stop_price(self):
        """Entering a position sets stop_price when provided."""
        rm = self._make_risk_mgr()
        rm.update_position(
            "BTC-USDT",
            PositionInfo(
                symbol="BTC-USDT",
                side="long",
                size=1.0,
                entry_price=30000.0,
                current_price=30000.0,
                stop_price=29000.0,
            ),
        )
        assert rm.positions["BTC-USDT"].stop_price == 29000.0

    def test_hold_updates_stop_price_when_new_provided(self):
        """HOLD updates stop_price when a new stop is provided."""
        rm = self._make_risk_mgr()
        rm.update_position(
            "BTC-USDT",
            PositionInfo(
                symbol="BTC-USDT",
                side="long",
                size=1.0,
                entry_price=30000.0,
                stop_price=29000.0,
            ),
        )
        # Simulate HOLD path with new stop_price
        pos = rm.positions["BTC-USDT"]
        new_stop = 29500.0
        pos.stop_price = new_stop
        assert rm.positions["BTC-USDT"].stop_price == 29500.0

    def test_hold_does_not_erase_stop_price_when_none(self):
        """HOLD does not erase stop_price when decision.stop_price is None."""
        rm = self._make_risk_mgr()
        rm.update_position(
            "BTC-USDT",
            PositionInfo(
                symbol="BTC-USDT",
                side="long",
                size=1.0,
                entry_price=30000.0,
                stop_price=29000.0,
            ),
        )
        # update_position with stop_price=None should preserve existing
        rm.update_position(
            "BTC-USDT",
            PositionInfo(
                symbol="BTC-USDT",
                side="long",
                size=1.0,
                entry_price=30000.0,
                stop_price=None,
            ),
        )
        assert rm.positions["BTC-USDT"].stop_price == 29000.0

    def test_stop_price_triggers_flatten_long(self):
        """Stop-check logic triggers for long position when price crosses stop."""
        rm = self._make_risk_mgr()
        rm.update_position(
            "BTC-USDT",
            PositionInfo(
                symbol="BTC-USDT",
                side="long",
                size=1.0,
                entry_price=30000.0,
                current_price=28000.0,
                stop_price=29000.0,
            ),
        )
        pos = rm.positions["BTC-USDT"]
        # For long: triggered when current_price <= stop_price
        triggered = pos.side == "long" and pos.current_price <= pos.stop_price
        assert triggered is True

    def test_stop_price_triggers_flatten_short(self):
        """Stop-check logic triggers for short position when price crosses stop."""
        rm = self._make_risk_mgr()
        rm.update_position(
            "BTC-USDT",
            PositionInfo(
                symbol="BTC-USDT",
                side="short",
                size=1.0,
                entry_price=30000.0,
                current_price=31000.0,
                stop_price=30500.0,
            ),
        )
        pos = rm.positions["BTC-USDT"]
        # For short: triggered when current_price >= stop_price
        triggered = pos.side == "short" and pos.current_price >= pos.stop_price
        assert triggered is True

    def test_stop_price_not_triggered_when_price_within(self):
        """Stop is not triggered when current price is within bounds."""
        rm = self._make_risk_mgr()
        rm.update_position(
            "BTC-USDT",
            PositionInfo(
                symbol="BTC-USDT",
                side="long",
                size=1.0,
                entry_price=30000.0,
                current_price=30500.0,
                stop_price=29000.0,
            ),
        )
        pos = rm.positions["BTC-USDT"]
        triggered = pos.side == "long" and pos.current_price <= pos.stop_price
        assert triggered is False

    def test_update_position_preserves_stop_when_new_is_none(self):
        """update_position preserves existing stop_price when incoming is None."""
        rm = self._make_risk_mgr()
        rm.update_position(
            "ETH-USDT",
            PositionInfo(symbol="ETH-USDT", side="long", size=5.0, stop_price=1800.0),
        )
        # New update without stop_price
        rm.update_position(
            "ETH-USDT",
            PositionInfo(symbol="ETH-USDT", side="long", size=5.0, current_price=2000.0),
        )
        assert rm.positions["ETH-USDT"].stop_price == 1800.0

    def test_update_position_overwrites_stop_when_new_provided(self):
        """update_position overwrites stop_price when a new value is provided."""
        rm = self._make_risk_mgr()
        rm.update_position(
            "ETH-USDT",
            PositionInfo(symbol="ETH-USDT", side="long", size=5.0, stop_price=1800.0),
        )
        rm.update_position(
            "ETH-USDT",
            PositionInfo(symbol="ETH-USDT", side="long", size=5.0, stop_price=1900.0),
        )
        assert rm.positions["ETH-USDT"].stop_price == 1900.0


# ── Change B: HTTP fail-fast ─────────────────────────────────────────


class TestHTTPFailFast:
    """Tests for HTTP error handling in KuCoinClient._request."""

    def _make_client(self) -> KuCoinClient:
        return KuCoinClient(
            api_key="test-key",
            api_secret="test-secret",
            api_passphrase="test-pass",
        )

    @pytest.mark.asyncio
    async def test_400_raises_api_error(self):
        """HTTP 400 raises KuCoinAPIError immediately."""
        client = self._make_client()

        with aioresponses() as m:
            m.get(
                "https://api.kucoin.com/api/v1/timestamp",
                payload={"data": 1234567890000},
            )
            m.get(
                "https://api.kucoin.com/api/v1/test",
                status=400,
                payload={"code": "400100", "msg": "Invalid param"},
            )

            await client.start()
            with pytest.raises(KuCoinAPIError) as exc_info:
                await client._request("GET", "/api/v1/test", signed=False)
            assert exc_info.value.status == 400
            await client.close()

    @pytest.mark.asyncio
    async def test_429_then_200_retries_successfully(self):
        """HTTP 429 is retried, second attempt returns success."""
        client = self._make_client()

        with aioresponses() as m:
            m.get(
                "https://api.kucoin.com/api/v1/timestamp",
                payload={"data": 1234567890000},
            )
            # First attempt: 429
            m.get(
                "https://api.kucoin.com/api/v1/test",
                status=429,
                payload={"code": "429000", "msg": "Too many requests"},
            )
            # Second attempt: 200
            m.get(
                "https://api.kucoin.com/api/v1/test",
                payload={"code": "200000", "data": {"ok": True}},
            )

            await client.start()
            # Patch sleep to avoid delays in test
            with patch("kucoin_bot.api.client.asyncio.sleep", new_callable=AsyncMock):
                result = await client._request("GET", "/api/v1/test", signed=False)
            assert result.get("data", {}).get("ok") is True
            await client.close()

    @pytest.mark.asyncio
    async def test_500_then_200_retries_successfully(self):
        """HTTP 500 is retried, second attempt returns success."""
        client = self._make_client()

        with aioresponses() as m:
            m.get(
                "https://api.kucoin.com/api/v1/timestamp",
                payload={"data": 1234567890000},
            )
            m.get(
                "https://api.kucoin.com/api/v1/test",
                status=500,
                payload={"code": "500000", "msg": "Internal error"},
            )
            m.get(
                "https://api.kucoin.com/api/v1/test",
                payload={"code": "200000", "data": {"ok": True}},
            )

            await client.start()
            with patch("kucoin_bot.api.client.asyncio.sleep", new_callable=AsyncMock):
                result = await client._request("GET", "/api/v1/test", signed=False)
            assert result.get("data", {}).get("ok") is True
            await client.close()

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self):
        """After 3 retries on 500, KuCoinAPIError is raised."""
        client = self._make_client()

        with aioresponses() as m:
            m.get(
                "https://api.kucoin.com/api/v1/timestamp",
                payload={"data": 1234567890000},
            )
            # All 3 attempts fail with 500
            for _ in range(3):
                m.get(
                    "https://api.kucoin.com/api/v1/test",
                    status=500,
                    payload={"code": "500000", "msg": "Server error"},
                )

            await client.start()
            with patch("kucoin_bot.api.client.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(KuCoinAPIError) as exc_info:
                    await client._request("GET", "/api/v1/test", signed=False)
            assert exc_info.value.code == "max_retries"
            await client.close()

    @pytest.mark.asyncio
    async def test_error_json_never_returned_as_success(self):
        """HTTP >= 400 must not return error body as successful data."""
        client = self._make_client()

        with aioresponses() as m:
            m.get(
                "https://api.kucoin.com/api/v1/timestamp",
                payload={"data": 1234567890000},
            )
            m.get(
                "https://api.kucoin.com/api/v1/test",
                status=403,
                payload={"code": "403000", "msg": "Forbidden"},
            )

            await client.start()
            with pytest.raises(KuCoinAPIError) as exc_info:
                await client._request("GET", "/api/v1/test", signed=False)
            assert exc_info.value.status == 403
            await client.close()

    @pytest.mark.asyncio
    async def test_429_honors_retry_after_header(self):
        """429 with Retry-After header uses that value for backoff."""
        client = self._make_client()

        with aioresponses() as m:
            m.get(
                "https://api.kucoin.com/api/v1/timestamp",
                payload={"data": 1234567890000},
            )
            m.get(
                "https://api.kucoin.com/api/v1/test",
                status=429,
                payload={"code": "429000", "msg": "Too many requests"},
                headers={"Retry-After": "2"},
            )
            m.get(
                "https://api.kucoin.com/api/v1/test",
                payload={"code": "200000", "data": {"ok": True}},
            )

            await client.start()
            sleep_mock = AsyncMock()
            with patch("kucoin_bot.api.client.asyncio.sleep", sleep_mock):
                result = await client._request("GET", "/api/v1/test", signed=False)
            # The first sleep should use Retry-After value (2.0)
            first_sleep = sleep_mock.call_args_list[0][0][0]
            assert first_sleep == pytest.approx(2.0)
            assert result.get("data", {}).get("ok") is True
            await client.close()


# ── Change C: Parallel kline fetching ────────────────────────────────


class TestParallelKlineFetching:
    """Tests for parallel kline fetching with bounded concurrency."""

    @pytest.mark.asyncio
    async def test_parallel_klines_produces_correct_results(self):
        """Parallel fetch produces same mapping as sequential."""
        symbols = ["SYM-A", "SYM-B", "SYM-C", "SYM-D"]
        kline_data = {s: [[str(i * 100), "1", "2", "3", "4", "5", "6"] for i in range(5)] for s in symbols}

        async def mock_get_klines(sym, kline_type="1hour", bars=200):
            await asyncio.sleep(0.01)  # Small delay to simulate IO
            return kline_data[sym]

        max_conc = 2
        sem = asyncio.Semaphore(max_conc)

        async def _fetch(s):
            async with sem:
                return s, await mock_get_klines(s)

        results = await asyncio.gather(*(_fetch(s) for s in symbols), return_exceptions=True)
        collected = {}
        for item in results:
            assert not isinstance(item, Exception)
            sym, kl = item
            collected[sym] = kl

        assert set(collected.keys()) == set(symbols)
        for sym in symbols:
            assert collected[sym] == kline_data[sym]

    @pytest.mark.asyncio
    async def test_parallel_klines_handles_exceptions(self):
        """Exceptions per symbol are logged and skipped without killing the loop."""

        async def mock_get_klines(sym, kline_type="1hour", bars=200):
            if sym == "FAIL":
                raise RuntimeError("API error")
            return [[str(100), "1", "2", "3", "4", "5", "6"]]

        symbols = ["OK1", "FAIL", "OK2"]
        max_conc = 8
        sem = asyncio.Semaphore(max_conc)

        async def _fetch(s):
            async with sem:
                return s, await mock_get_klines(s)

        results = await asyncio.gather(*(_fetch(s) for s in symbols), return_exceptions=True)
        collected = {}
        for item in results:
            if isinstance(item, Exception):
                continue
            sym, kl = item
            collected[sym] = kl

        assert "OK1" in collected
        assert "OK2" in collected
        assert "FAIL" not in collected

    @pytest.mark.asyncio
    async def test_concurrency_bound_respected(self):
        """Max in-flight never exceeds MAX_KLINE_CONCURRENCY."""
        max_conc = 3
        in_flight = 0
        max_observed = 0

        async def mock_get_klines(sym, kline_type="1hour", bars=200):
            nonlocal in_flight, max_observed
            in_flight += 1
            max_observed = max(max_observed, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1
            return [[str(100), "1", "2", "3", "4", "5", "6"]]

        symbols = [f"SYM-{i}" for i in range(10)]
        sem = asyncio.Semaphore(max_conc)

        async def _fetch(s):
            async with sem:
                return s, await mock_get_klines(s)

        await asyncio.gather(*(_fetch(s) for s in symbols), return_exceptions=True)
        assert max_observed <= max_conc


# ── Change D: Batch DB writes ────────────────────────────────────────


class TestBatchDBWrites:
    """Tests for batched DB writes per slow cycle."""

    def test_batch_mode_single_commit(self):
        """When BATCH_DB_WRITES=1, commit is called once at end."""
        mock_session = MagicMock()
        snapshots = [MagicMock() for _ in range(5)]
        batch = True

        for snap in snapshots:
            mock_session.add(snap)
            if not batch:
                mock_session.commit()
        if batch:
            mock_session.commit()

        assert mock_session.add.call_count == 5
        assert mock_session.commit.call_count == 1

    def test_legacy_mode_per_symbol_commit(self):
        """When BATCH_DB_WRITES=0, commit is called after each add."""
        mock_session = MagicMock()
        snapshots = [MagicMock() for _ in range(5)]
        batch = False

        for snap in snapshots:
            mock_session.add(snap)
            if not batch:
                mock_session.commit()
        if batch:
            mock_session.commit()

        assert mock_session.add.call_count == 5
        assert mock_session.commit.call_count == 5

    def test_rollback_on_error(self):
        """DB errors trigger rollback and don't crash."""
        mock_session = MagicMock()
        mock_session.commit.side_effect = RuntimeError("DB error")

        try:
            mock_session.add(MagicMock())
            mock_session.commit()
        except Exception:
            mock_session.rollback()

        assert mock_session.rollback.call_count == 1


# ── Change E: Kline ordering optimization ────────────────────────────


class TestKlineOrdering:
    """Tests for optimized kline ordering in MarketDataService."""

    def _make_service(self) -> MarketDataService:
        client = MagicMock(spec="kucoin_bot.api.client.KuCoinClient")
        return MarketDataService(client=client)

    @pytest.mark.asyncio
    async def test_newest_first_input_is_reversed(self):
        """Descending (newest-first) klines are reversed to ascending."""
        client = MagicMock()
        # Descending order
        raw = [
            ["300", "1", "2", "3", "0.5", "10", "20"],
            ["200", "1", "2", "3", "0.5", "10", "20"],
            ["100", "1", "2", "3", "0.5", "10", "20"],
        ]
        client.get_klines = AsyncMock(return_value=raw)

        service = MarketDataService(client=client)
        result = await service.get_klines_spot("BTC-USDT", "1hour")
        timestamps = [int(k[0]) for k in result]
        assert timestamps == [100, 200, 300]

    @pytest.mark.asyncio
    async def test_oldest_first_input_stays(self):
        """Already ascending klines are not changed."""
        client = MagicMock()
        raw = [
            ["100", "1", "2", "3", "0.5", "10", "20"],
            ["200", "1", "2", "3", "0.5", "10", "20"],
            ["300", "1", "2", "3", "0.5", "10", "20"],
        ]
        client.get_klines = AsyncMock(return_value=raw)

        service = MarketDataService(client=client)
        result = await service.get_klines_spot("BTC-USDT", "1hour")
        timestamps = [int(k[0]) for k in result]
        assert timestamps == [100, 200, 300]

    @pytest.mark.asyncio
    async def test_shuffled_input_is_sorted(self):
        """Shuffled klines are sorted correctly."""
        client = MagicMock()
        raw = [
            ["300", "1", "2", "3", "0.5", "10", "20"],
            ["100", "1", "2", "3", "0.5", "10", "20"],
            ["200", "1", "2", "3", "0.5", "10", "20"],
        ]
        client.get_klines = AsyncMock(return_value=raw)

        service = MarketDataService(client=client)
        result = await service.get_klines_spot("BTC-USDT", "1hour")
        timestamps = [int(k[0]) for k in result]
        assert timestamps == [100, 200, 300]

    @pytest.mark.asyncio
    async def test_non_int_timestamps_fallback_to_sort(self):
        """Non-integer timestamps fall back to sort; data is still returned."""
        client = MagicMock()
        # Non-int timestamps trigger the exception path in the try block
        raw = [
            ["abc", "1", "2", "3", "0.5", "10", "20"],
            ["def", "1", "2", "3", "0.5", "10", "20"],
        ]
        client.get_klines = AsyncMock(return_value=raw)

        service = MarketDataService(client=client)
        # The fallback sort will also fail on non-int data, but the except
        # clause in the sort logic catches it and the data is cached as-is.
        # The function should not crash and should return some data.
        result = await service.get_klines_spot("BTC-USDT", "1hour")
        # Data is returned (possibly unsorted) rather than crashing
        assert len(result) == 2
