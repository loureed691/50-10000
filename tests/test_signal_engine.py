"""Tests for signal engine."""

from __future__ import annotations

import pytest
import numpy as np

from kucoin_bot.services.signal_engine import SignalEngine, Regime


class TestSignalEngine:
    def test_compute_returns_scores(self, sample_klines):
        engine = SignalEngine()
        scores = engine.compute("BTC-USDT", sample_klines)
        assert scores.symbol == "BTC-USDT"
        assert -1 <= scores.momentum <= 1
        assert 0 <= scores.trend_strength <= 1
        assert -1 <= scores.mean_reversion <= 1
        assert 0 <= scores.volatility <= 1
        assert scores.regime in Regime

    def test_trending_up_detection(self, trending_up_klines):
        engine = SignalEngine()
        scores = engine.compute("TEST-USDT", trending_up_klines)
        assert scores.momentum > 0
        assert scores.trend_strength > 0.2

    def test_ranging_detection(self, ranging_klines):
        engine = SignalEngine()
        scores = engine.compute("TEST-USDT", ranging_klines)
        # In ranging market, trend strength should be lower
        assert scores.trend_strength < 0.8

    def test_insufficient_data(self):
        engine = SignalEngine()
        klines = [[0, "100", "101", "102", "99", "500", "50000"]] * 10
        scores = engine.compute("TEST-USDT", klines)
        assert scores.symbol == "TEST-USDT"
        # With insufficient data, defaults should be returned
        assert scores.confidence == 0.0

    def test_cache(self, sample_klines):
        engine = SignalEngine()
        engine.compute("BTC-USDT", sample_klines)
        cached = engine.get_cached("BTC-USDT")
        assert cached is not None
        assert cached.symbol == "BTC-USDT"

    def test_to_dict(self, sample_klines):
        engine = SignalEngine()
        scores = engine.compute("BTC-USDT", sample_klines)
        d = scores.to_dict()
        assert "symbol" in d
        assert "regime" in d
        assert "momentum" in d

    def test_orderbook_and_funding_inputs(self, sample_klines):
        engine = SignalEngine()
        scores = engine.compute(
            "BTC-USDT",
            sample_klines,
            orderbook={"bids": [["1", "10"]], "asks": [["1", "2"]]},
            funding_rate=0.0005,
        )
        assert scores.orderbook_imbalance > 0
        assert scores.funding_rate == pytest.approx(0.0005)

    def test_orderbook_imbalance_equal_volumes(self, sample_klines):
        engine = SignalEngine()
        scores = engine.compute(
            "BTC-USDT",
            sample_klines,
            orderbook={"bids": [["1", "10"]], "asks": [["1", "10"]]},
        )
        assert scores.orderbook_imbalance == pytest.approx(0.0)

    def test_orderbook_imbalance_empty_orderbook(self, sample_klines):
        engine = SignalEngine()
        scores = engine.compute(
            "BTC-USDT",
            sample_klines,
            orderbook={"bids": [], "asks": []},
        )
        assert scores.orderbook_imbalance == pytest.approx(0.0)

    def test_orderbook_imbalance_malformed_levels(self, sample_klines):
        engine = SignalEngine()
        scores = engine.compute(
            "BTC-USDT",
            sample_klines,
            orderbook={"bids": [["bad", "bad"]], "asks": [["bad", "bad"]]},
        )
        assert scores.orderbook_imbalance == pytest.approx(0.0)

    def test_news_spike_regime(self):
        engine = SignalEngine()
        klines = []
        for i in range(60):
            close = 100 + (i % 2)
            volume = 90 + (i % 3) * 10
            if i == 59:
                close = 160
                volume = 10000
            high = close + 5
            low = close - 5
            klines.append([i * 3600, "100", str(close), str(high), str(low), str(volume), str(volume * close)])
        scores = engine.compute("BTC-USDT", klines)
        assert scores.regime == Regime.NEWS_SPIKE
