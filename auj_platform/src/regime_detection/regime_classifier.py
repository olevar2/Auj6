#!/usr/bin/env python3
"""
Market Regime Classifier for AUJ Platform
==========================================

This module implements an intelligent market regime detection system that classifies
market conditions to optimize indicator and agent performance. It's designed to
prevent overfitting by identifying distinct market regimes.

Supported Market Regimes:
- TRENDING_BULLISH: Strong upward momentum
- TRENDING_BEARISH: Strong downward momentum
- SIDEWAYS_CONSOLIDATION: Range-bound market
- HIGH_VOLATILITY: Elevated volatility periods
- LOW_VOLATILITY: Quiet market conditions
- BREAKOUT: Emerging from consolidation
- REVERSAL: Potential trend change

Mission: Support sustainable profit generation for helping sick children and families.
"""

import asyncio
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Type definitions
from ..core.data_contracts import MarketDataPoint


class MarketRegime(Enum):
    """Market regime classifications for optimization."""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    SIDEWAYS_CONSOLIDATION = "sideways_consolidation"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"


@dataclass
class RegimeClassification:
    """Result of market regime classification."""
    regime: MarketRegime
    confidence: float  # 0.0 to 1.0
    characteristics: Dict[str, Any]
    timestamp: datetime
    timeframe: str


class RegimeClassifier:
    """
    Intelligent market regime detection system.

    Uses multiple technical indicators and statistical measures to classify
    current market conditions and adapt indicator/agent performance accordingly.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the regime classifier.

        Args:
            config: Configuration dictionary with classifier parameters
        """
        from ..core.unified_config import get_unified_config
        self.config_manager = get_unified_config()
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Classification parameters
        self.trend_threshold = self.config_manager.get_float('trend_threshold', 0.02)  # 2% for trend detection
        self.volatility_threshold = self.config_manager.get_float('volatility_threshold', 0.01)  # 1% for volatility
        self.lookback_periods = self.config_manager.get_int('lookback_periods', 50)
        self.confidence_threshold = self.config_manager.get_float('confidence_threshold', 0.7)

        # Regime history for stability
        self.regime_history: List[RegimeClassification] = []
        self.max_history = self.config_manager.get_int('max_history', 100)

        self.logger.info("RegimeClassifier initialized for humanitarian mission")

    async def classify_regime(self,
                            market_data: List[MarketDataPoint],
                            timeframe: str = "1H") -> RegimeClassification:
        """
        Classify the current market regime based on recent market data.

        Args:
            market_data: Recent market data points
            timeframe: Timeframe for classification

        Returns:
            RegimeClassification object with regime and confidence
        """
        if len(market_data) < self.lookback_periods:
            self.logger.warning("Insufficient data for regime classification")
            return RegimeClassification(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                characteristics={},
                timestamp=datetime.now(),
                timeframe=timeframe
            )

        try:
            # Convert to DataFrame for analysis
            df = self._prepare_dataframe(market_data)

            # Calculate regime indicators
            trend_strength = self._calculate_trend_strength(df)
            volatility_level = self._calculate_volatility_level(df)
            momentum_state = self._calculate_momentum_state(df)
            range_behavior = self._calculate_range_behavior(df)

            # Classify regime based on indicators
            regime, confidence = self._classify_based_on_indicators(
                trend_strength, volatility_level, momentum_state, range_behavior
            )

            # Build characteristics dictionary
            characteristics = {
                'trend_strength': trend_strength,
                'volatility_level': volatility_level,
                'momentum_state': momentum_state,
                'range_behavior': range_behavior,
                'price_change_pct': ((df.iloc[-1]['close'] - df.iloc[0]['close']) / df.iloc[0]['close']) * 100
            }

            classification = RegimeClassification(
                regime=regime,
                confidence=confidence,
                characteristics=characteristics,
                timestamp=datetime.now(),
                timeframe=timeframe
            )

            # Add to history
            self._add_to_history(classification)

            self.logger.info(f"Market regime classified as {regime.value} with {confidence:.2f} confidence")
            return classification

        except Exception as e:
            self.logger.error(f"Error in regime classification: {e}")
            return RegimeClassification(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                characteristics={'error': str(e)},
                timestamp=datetime.now(),
                timeframe=timeframe
            )

    def _prepare_dataframe(self, market_data: List[MarketDataPoint]) -> pd.DataFrame:
        """Convert market data to DataFrame for analysis."""
        data = []
        for point in market_data[-self.lookback_periods:]:
            data.append({
                'timestamp': point.timestamp,
                'open': point.open,
                'high': point.high,
                'low': point.low,
                'close': point.close,
                'volume': point.volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df.sort_index()

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength (-1 to 1, negative=bearish, positive=bullish)."""
        # Simple trend calculation using linear regression slope
        prices = df['close'].values
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]

        # Normalize slope relative to average price
        avg_price = np.mean(prices)
        normalized_slope = slope / avg_price if avg_price > 0 else 0

        # Clamp to -1 to 1 range
        return max(-1.0, min(1.0, normalized_slope * 100))

    def _calculate_volatility_level(self, df: pd.DataFrame) -> float:
        """Calculate volatility level (0 to 1, higher=more volatile)."""
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()

        # Normalize to 0-1 scale (assuming 5% daily volatility as high)
        normalized_vol = min(1.0, volatility / 0.05)
        return normalized_vol

    def _calculate_momentum_state(self, df: pd.DataFrame) -> float:
        """Calculate momentum state (-1 to 1)."""
        # Simple momentum using rate of change
        if len(df) < 10:
            return 0.0

        recent_close = df['close'].iloc[-1]
        past_close = df['close'].iloc[-10]

        momentum = (recent_close - past_close) / past_close if past_close > 0 else 0
        return max(-1.0, min(1.0, momentum * 10))  # Scale for reasonable range

    def _calculate_range_behavior(self, df: pd.DataFrame) -> float:
        """Calculate range behavior (0 to 1, higher=more range-bound)."""
        # Calculate if price is staying within a range
        high_max = df['high'].max()
        low_min = df['low'].min()
        current_close = df['close'].iloc[-1]

        # Check how much of recent action is within the range
        range_size = (high_max - low_min) / low_min if low_min > 0 else 0
        position_in_range = (current_close - low_min) / (high_max - low_min) if high_max > low_min else 0.5

        # Higher value means more range-bound behavior
        range_factor = 1.0 - min(1.0, range_size / 0.1)  # 10% range is considered full range behavior
        return range_factor

    def _classify_based_on_indicators(self,
                                    trend_strength: float,
                                    volatility_level: float,
                                    momentum_state: float,
                                    range_behavior: float) -> Tuple[MarketRegime, float]:
        """Classify regime based on calculated indicators."""

        # Strong trending conditions
        if abs(trend_strength) > 0.5 and volatility_level < 0.7:
            if trend_strength > 0:
                return MarketRegime.TRENDING_BULLISH, 0.8
            else:
                return MarketRegime.TRENDING_BEARISH, 0.8

        # High volatility conditions
        if volatility_level > 0.8:
            return MarketRegime.HIGH_VOLATILITY, 0.7

        # Low volatility conditions
        if volatility_level < 0.3 and range_behavior > 0.7:
            return MarketRegime.LOW_VOLATILITY, 0.7

        # Breakout conditions
        if abs(momentum_state) > 0.6 and volatility_level > 0.5:
            return MarketRegime.BREAKOUT, 0.6

        # Reversal conditions
        if (trend_strength > 0.3 and momentum_state < -0.3) or (trend_strength < -0.3 and momentum_state > 0.3):
            return MarketRegime.REVERSAL, 0.6

        # Sideways/consolidation (default for unclear conditions)
        if range_behavior > 0.5:
            return MarketRegime.SIDEWAYS_CONSOLIDATION, 0.5

        # Unknown if nothing matches
        return MarketRegime.UNKNOWN, 0.3

    def _add_to_history(self, classification: RegimeClassification):
        """Add classification to history and maintain size limit."""
        self.regime_history.append(classification)

        # Trim history if too long
        if len(self.regime_history) > self.max_history:
            self.regime_history = self.regime_history[-self.max_history:]

    def get_regime_stability(self) -> float:
        """
        Calculate regime stability based on recent classifications.

        Returns:
            Stability score (0.0 to 1.0, higher=more stable)
        """
        if len(self.regime_history) < 5:
            return 0.5  # Neutral stability for insufficient data

        recent_regimes = [r.regime for r in self.regime_history[-10:]]
        most_common = max(set(recent_regimes), key=recent_regimes.count)
        stability = recent_regimes.count(most_common) / len(recent_regimes)

        return stability

    def get_current_regime(self) -> Optional[MarketRegime]:
        """Get the most recent regime classification."""
        if self.regime_history:
            return self.regime_history[-1].regime
        return None

    def get_regime_characteristics(self) -> Dict[str, Any]:
        """Get characteristics of the current regime."""
        if self.regime_history:
            return self.regime_history[-1].characteristics
        return {}

    async def cleanup(self):
        """Clean up resources."""
        self.logger.info("RegimeClassifier cleanup completed")


# Helper functions for external use
def create_regime_classifier(config: Dict[str, Any]) -> RegimeClassifier:
    """Factory function to create a configured RegimeClassifier."""
    return RegimeClassifier(config)
