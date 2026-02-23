"""Shared test fixtures."""

from __future__ import annotations

import os

import numpy as np
import pytest

# Ensure we're in BACKTEST mode during tests
os.environ["BOT_MODE"] = "BACKTEST"
os.environ["KILL_SWITCH"] = "false"
os.environ["KUCOIN_API_KEY"] = "test_key"
os.environ["KUCOIN_API_SECRET"] = "test_secret"
os.environ["KUCOIN_API_PASSPHRASE"] = "test_pass"


@pytest.fixture
def sample_klines():
    """Generate deterministic sample klines for testing."""
    rng = np.random.RandomState(42)
    n = 200
    price = 30000.0
    klines = []
    for i in range(n):
        ret = rng.normal(0, 0.01)
        o = price
        c = price * (1 + ret)
        h = max(o, c) * (1 + abs(rng.normal(0, 0.003)))
        low = min(o, c) * (1 - abs(rng.normal(0, 0.003)))
        vol = rng.uniform(100, 1000)
        klines.append([i * 3600, str(o), str(c), str(h), str(low), str(vol), str(vol * c)])
        price = c
    return klines


@pytest.fixture
def trending_up_klines():
    """Klines with a clear uptrend."""
    klines = []
    price = 100.0
    for i in range(200):
        price *= 1.005  # consistent uptrend
        o = price * 0.998
        h = price * 1.003
        low = price * 0.996
        klines.append([i * 3600, str(o), str(price), str(h), str(low), "500", str(500 * price)])
    return klines


@pytest.fixture
def ranging_klines():
    """Klines that oscillate in a range."""
    rng = np.random.RandomState(99)
    klines = []
    price = 100.0
    for i in range(200):
        price = 100.0 + 2.0 * np.sin(i * 0.1) + rng.normal(0, 0.3)
        o = price - 0.1
        h = price + 0.5
        low = price - 0.5
        klines.append([i * 3600, str(o), str(price), str(h), str(low), "500", str(500 * price)])
    return klines
