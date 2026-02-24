"""Tests for the observability HTTP server, DB retention, and related features."""

from __future__ import annotations

import datetime as dt

import pytest
from aiohttp import ClientSession

from kucoin_bot.reporting.http_server import (
    _health,
    set_healthy,
    start_metrics_server,
)
from kucoin_bot.reporting.metrics import METRICS


class TestHealthFunction:
    """Unit tests for the health setter."""

    def test_set_healthy_ok(self):
        set_healthy(True)
        assert _health["status"] == "ok"
        assert _health["last_cycle"] > 0

    def test_set_healthy_degraded(self):
        set_healthy(False)
        assert _health["status"] == "degraded"


class TestMetricsHTTPServer:
    """Integration tests for /healthz and /metrics HTTP endpoints."""

    @pytest.fixture
    async def server(self):
        """Start the metrics server on a fixed test port and clean up after."""
        port = 19876  # high port unlikely to conflict
        runner = await start_metrics_server(host="127.0.0.1", port=port)
        yield port
        await runner.cleanup()

    @pytest.mark.asyncio
    async def test_healthz_returns_json(self, server):
        set_healthy(True)
        async with ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{server}/healthz") as resp:
                assert resp.status == 200
                body = await resp.text()
                assert '"status": "ok"' in body

    @pytest.mark.asyncio
    async def test_healthz_degraded_returns_503(self, server):
        set_healthy(False)
        async with ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{server}/healthz") as resp:
                assert resp.status == 503
                body = await resp.text()
                assert '"status": "degraded"' in body

    @pytest.mark.asyncio
    async def test_metrics_returns_prometheus_format(self, server):
        METRICS.inc("test_counter_total")
        METRICS.set("test_gauge", 42.0)
        async with ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{server}/metrics") as resp:
                assert resp.status == 200
                body = await resp.text()
                assert "test_counter_total" in body
                assert "test_gauge 42.0" in body
                # Check content type
                ct = resp.headers.get("Content-Type", "")
                assert "text/plain" in ct


class TestRetention:
    """Tests for DB retention / SignalSnapshot purging."""

    def test_purge_old_snapshots(self):
        from kucoin_bot.models import SignalSnapshot, init_db
        from kucoin_bot.reporting.retention import purge_old_snapshots

        session_factory = init_db("sqlite:///:memory:")

        # Insert snapshots: some old, some recent
        with session_factory() as sess:
            old_snap = SignalSnapshot(
                symbol="BTC-USDT",
                regime="trending",
                strategy_name="trend",
                decision="entry",
                timestamp=dt.datetime.utcnow() - dt.timedelta(days=10),
            )
            recent_snap = SignalSnapshot(
                symbol="ETH-USDT",
                regime="ranging",
                strategy_name="mean_rev",
                decision="hold",
                timestamp=dt.datetime.utcnow() - dt.timedelta(hours=1),
            )
            sess.add_all([old_snap, recent_snap])
            sess.commit()

        # Purge with 7-day retention
        deleted = purge_old_snapshots(session_factory, days=7)
        assert deleted == 1

        # Verify only the recent one remains
        with session_factory() as sess:
            remaining = sess.query(SignalSnapshot).all()
            assert len(remaining) == 1
            assert remaining[0].symbol == "ETH-USDT"

    def test_purge_returns_zero_when_nothing_to_delete(self):
        from kucoin_bot.models import init_db
        from kucoin_bot.reporting.retention import purge_old_snapshots

        session_factory = init_db("sqlite:///:memory:")
        deleted = purge_old_snapshots(session_factory, days=7)
        assert deleted == 0


class TestPrometheusTextFormat:
    """Validate Prometheus text exposition output."""

    def test_counters_in_output(self):
        METRICS.inc("prom_test_requests_total", 5.0)
        output = METRICS.to_prometheus()
        assert "prom_test_requests_total 5.0" in output

    def test_labeled_metric_in_output(self):
        METRICS.inc("prom_test_labeled", 1.0, labels={"method": "GET"})
        output = METRICS.to_prometheus()
        assert 'prom_test_labeled{method="GET"}' in output

    def test_histogram_sum_and_count(self):
        METRICS.observe("prom_test_latency", 0.1)
        METRICS.observe("prom_test_latency", 0.2)
        output = METRICS.to_prometheus()
        assert "prom_test_latency_count 2" in output
        assert "prom_test_latency_sum" in output
