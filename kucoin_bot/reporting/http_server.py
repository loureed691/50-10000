"""Lightweight HTTP server exposing /healthz and /metrics endpoints.

Runs as an ``asyncio`` background task inside the bot process so that
Prometheus (or any HTTP scraper) and container orchestrators can probe
the bot without additional dependencies.

Configuration via environment variables:

* ``METRICS_PORT`` – TCP port to listen on (default **9090**).
* ``METRICS_HOST`` – bind address (default **0.0.0.0**).

Usage inside the live loop::

    from kucoin_bot.reporting.http_server import start_metrics_server

    server = await start_metrics_server()
    # ... bot runs ...
    await server.cleanup()
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from aiohttp import web

from kucoin_bot.reporting.metrics import METRICS

logger = logging.getLogger(__name__)

# Module-level health status that the main loop can update.
_health: dict = {
    "status": "starting",
    "started_at": time.time(),
    "last_cycle": 0.0,
}


def set_healthy(healthy: bool = True) -> None:
    """Mark the bot as healthy (called from the live loop each cycle)."""
    _health["status"] = "ok" if healthy else "degraded"
    _health["last_cycle"] = time.time()


async def _handle_healthz(request: web.Request) -> web.Response:
    """Return 200 when healthy, 503 when degraded or stale."""
    status = _health.get("status", "unknown")
    last_cycle = _health.get("last_cycle", 0.0)
    uptime = time.time() - _health.get("started_at", time.time())

    # Consider bot stale if no cycle update in 5 minutes
    stale = last_cycle > 0 and (time.time() - last_cycle) > 300
    http_status = 200 if status == "ok" and not stale else 503

    body = (
        f'{{"status": "{status}", "stale": {str(stale).lower()}, '
        f'"uptime_seconds": {uptime:.0f}, "last_cycle": {last_cycle:.0f}}}\n'
    )
    return web.Response(text=body, status=http_status, content_type="application/json")


async def _handle_metrics(request: web.Request) -> web.Response:
    """Return all metrics in Prometheus text exposition format 0.0.4."""
    body = METRICS.to_prometheus()
    return web.Response(
        text=body,
        status=200,
        content_type="text/plain",
        charset="utf-8",
        headers={"X-Content-Type-Options": "nosniff"},
    )


async def start_metrics_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> web.AppRunner:
    """Create and start the metrics HTTP server.

    Returns the ``AppRunner`` so the caller can call ``await runner.cleanup()``
    on shutdown.
    """
    host = host or os.getenv("METRICS_HOST", "0.0.0.0")
    port = port or int(os.getenv("METRICS_PORT", "9090"))

    app = web.Application()
    app.router.add_get("/healthz", _handle_healthz)
    app.router.add_get("/metrics", _handle_metrics)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    logger.info("Metrics server listening on http://%s:%d", host, port)
    return runner
