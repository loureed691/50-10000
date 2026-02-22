"""Signal Engine – computes multi-timeframe features and regime classification."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    UNKNOWN = "unknown"


@dataclass
class SignalScores:
    """Normalized signal scores for one symbol."""

    symbol: str
    momentum: float = 0.0       # -1 to 1
    trend_strength: float = 0.0  # 0 to 1
    mean_reversion: float = 0.0  # -1 to 1 (positive = oversold bounce expected)
    volatility: float = 0.0      # normalized 0-1
    volume_anomaly: float = 0.0  # z-score
    orderbook_imbalance: float = 0.0  # -1 to 1
    funding_rate: float = 0.0
    regime: Regime = Regime.UNKNOWN
    confidence: float = 0.0      # 0 to 1

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "momentum": round(self.momentum, 4),
            "trend_strength": round(self.trend_strength, 4),
            "mean_reversion": round(self.mean_reversion, 4),
            "volatility": round(self.volatility, 4),
            "volume_anomaly": round(self.volume_anomaly, 4),
            "orderbook_imbalance": round(self.orderbook_imbalance, 4),
            "funding_rate": round(self.funding_rate, 4),
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class SignalEngine:
    """Compute feature signals from kline data and classify regime."""

    lookback: int = 50
    _cache: Dict[str, SignalScores] = field(default_factory=dict)

    def compute(self, symbol: str, klines: List[list]) -> SignalScores:
        """Compute signal scores from KuCoin klines.

        KuCoin kline format: [time, open, close, high, low, volume, turnover]
        """
        if len(klines) < self.lookback:
            return SignalScores(symbol=symbol)

        closes = np.array([float(k[2]) for k in klines[-self.lookback:]], dtype=np.float64)
        highs = np.array([float(k[3]) for k in klines[-self.lookback:]], dtype=np.float64)
        lows = np.array([float(k[4]) for k in klines[-self.lookback:]], dtype=np.float64)
        volumes = np.array([float(k[5]) for k in klines[-self.lookback:]], dtype=np.float64)

        scores = SignalScores(symbol=symbol)

        # --- Momentum (ROC) ---
        if closes[-1] != 0 and closes[0] != 0:
            roc = (closes[-1] - closes[0]) / closes[0]
            scores.momentum = float(np.clip(roc * 10, -1, 1))

        # --- Trend strength (ADX-like via directional movement) ---
        scores.trend_strength = self._trend_strength(closes, highs, lows)

        # --- Mean reversion (Bollinger %B) ---
        scores.mean_reversion = self._bollinger_signal(closes)

        # --- Volatility (normalized ATR) ---
        scores.volatility = self._normalized_volatility(closes, highs, lows)

        # --- Volume anomaly (z-score of recent volume) ---
        if len(volumes) > 20 and np.std(volumes[:-1]) > 0:
            scores.volume_anomaly = float(
                (volumes[-1] - np.mean(volumes[:-1])) / np.std(volumes[:-1])
            )

        # --- Regime ---
        scores.regime = self._classify_regime(scores)

        # --- Confidence ---
        scores.confidence = self._compute_confidence(scores)

        self._cache[symbol] = scores
        return scores

    def get_cached(self, symbol: str) -> Optional[SignalScores]:
        return self._cache.get(symbol)

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trend_strength(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> float:
        """Simplified ADX proxy using directional movement."""
        if len(closes) < 14:
            return 0.0
        plus_dm = np.maximum(np.diff(highs), 0)
        minus_dm = np.maximum(-np.diff(lows), 0)
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
        )
        period = 14
        if len(tr) < period:
            return 0.0
        atr = np.mean(tr[-period:])
        if atr == 0:
            return 0.0
        plus_di = np.mean(plus_dm[-period:]) / atr
        minus_di = np.mean(minus_dm[-period:]) / atr
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0
        dx = abs(plus_di - minus_di) / di_sum
        return float(np.clip(dx, 0, 1))

    @staticmethod
    def _bollinger_signal(closes: np.ndarray, period: int = 20) -> float:
        """Bollinger %B mapped to -1..1 (positive = oversold / bounce expected)."""
        if len(closes) < period:
            return 0.0
        sma = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        if std == 0:
            return 0.0
        pct_b = (closes[-1] - (sma - 2 * std)) / (4 * std)  # 0..1 range
        return float(np.clip(1 - 2 * pct_b, -1, 1))  # invert: low %B => positive signal

    @staticmethod
    def _normalized_volatility(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> float:
        if len(closes) < 14:
            return 0.0
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
        )
        atr = np.mean(tr[-14:])
        price = closes[-1]
        if price == 0:
            return 0.0
        normalized = atr / price  # as fraction of price
        return float(np.clip(normalized * 100, 0, 1))  # scale: 1% ATR -> 1.0

    @staticmethod
    def _classify_regime(scores: SignalScores) -> Regime:
        if scores.volatility > 0.7:
            return Regime.HIGH_VOLATILITY
        if scores.trend_strength > 0.5:
            if scores.momentum > 0.1:
                return Regime.TRENDING_UP
            elif scores.momentum < -0.1:
                return Regime.TRENDING_DOWN
        if scores.trend_strength < 0.25 and scores.volatility < 0.3:
            return Regime.RANGING
        return Regime.UNKNOWN

    @staticmethod
    def _compute_confidence(scores: SignalScores) -> float:
        """Composite confidence from signal agreement."""
        factors = []
        factors.append(scores.trend_strength)
        factors.append(min(abs(scores.momentum), 1.0))
        factors.append(1.0 - scores.volatility)  # prefer lower vol
        if factors:
            return float(np.clip(sum(factors) / len(factors), 0, 1))
        return 0.0
