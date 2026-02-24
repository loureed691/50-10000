"""Lightweight metrics collection and export (Prometheus-style counters/gauges).

Provides a simple in-process metrics store that can be exported as JSON or
Prometheus text format for external scraping.

Usage in live loop::

    from kucoin_bot.reporting.metrics import METRICS

    METRICS.inc("orders_placed_total", labels={"symbol": "BTC-USDT", "side": "buy"})
    METRICS.set("equity_usdt", 12345.67)
    METRICS.observe("order_latency_seconds", 0.45)

Export::

    print(METRICS.to_json())
    print(METRICS.to_prometheus())
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from typing import Dict, Optional


class _MetricStore:
    """Thread-safe metric store with counter, gauge, and histogram semantics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = defaultdict(list)
        self._last_update: float = 0.0

    # -- Counter --
    def inc(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] += value
            self._last_update = time.time()

    # -- Gauge --
    def set(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._gauges[key] = value
            self._last_update = time.time()

    # -- Histogram (simple: stores last N observations) --
    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            bucket = self._histograms[key]
            bucket.append(value)
            if len(bucket) > 1000:
                bucket[:] = bucket[-500:]
            self._last_update = time.time()

    # -- Export --
    def to_dict(self) -> dict:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: {"count": len(v), "sum": sum(v)} for k, v in self._histograms.items()},
                "last_update": self._last_update,
            }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_prometheus(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        lines: list[str] = []
        with self._lock:
            for k, v in sorted(self._counters.items()):
                lines.append(f"{k} {v}")
            for k, v in sorted(self._gauges.items()):
                lines.append(f"{k} {v}")
            for k, vals in sorted(self._histograms.items()):
                lines.append(f"{k}_count {len(vals)}")
                lines.append(f"{k}_sum {sum(vals)}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _key(name: str, labels: Optional[Dict[str, str]] = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Singleton instance
METRICS = _MetricStore()
