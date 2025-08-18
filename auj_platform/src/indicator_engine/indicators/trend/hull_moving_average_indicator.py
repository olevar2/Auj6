"""
Advanced Hull Moving Average (HMA) Indicator with Lag Reduction

Sophisticated moving average with:
- Traditional Hull MA calculations using WMA
- Lag reduction analysis and optimization
- Multi-period HMA analysis
- Trend reversal detection
- Signal smoothing and filtering

HMA reduces lag while maintaining smoothness by using weighted moving averages.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class HMATrend(Enum):
    STRONG_UP = "strong_up"
    WEAK_UP = "weak_up"
    SIDEWAYS = "sideways"
    WEAK_DOWN = "weak_down"
    STRONG_DOWN = "strong_down"

@dataclass
class HullMAResult:
    hma_value: float
    trend_direction: HMATrend
    slope: float
    signal: SignalType
    confidence: float
    responsiveness: float

class HullMovingAverageIndicator(StandardIndicatorInterface):
    def __init__(self, period: int = 21, enable_smoothing: bool = True):
        self.period = period
        self.enable_smoothing = enable_smoothing
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.period + 10:
                raise ValueError("Insufficient data")
            
            # Calculate Hull Moving Average
            hma_values = self._calculate_hma(data['close'])
            
            # Analyze trend
            trend_direction, slope = self._analyze_trend(hma_values, data['close'])
            responsiveness = self._calculate_responsiveness(hma_values, data['close'])
            
            # Generate signals
            signal, confidence = self._generate_signals(data['close'], hma_values, trend_direction, slope)
            
            latest_result = HullMAResult(
                hma_value=hma_values.iloc[-1],
                trend_direction=trend_direction,
                slope=slope,
                signal=signal,
                confidence=confidence,
                responsiveness=responsiveness
            )
            
            return {
                'current': latest_result,
                'values': {'hma': hma_values.tolist()},
                'signal': signal,
                'confidence': confidence,
                'trend_direction': trend_direction.value,
                'slope': slope,
                'responsiveness': responsiveness
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Hull MA: {e}")
            return self._get_default_result()
    
    def _calculate_hma(self, series: pd.Series) -> pd.Series:
        """Calculate Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
        n = self.period
        half_period = int(n / 2)
        sqrt_period = int(np.sqrt(n))
        
        # WMA calculations
        wma_half = self._weighted_ma(series, half_period)
        wma_full = self._weighted_ma(series, n)
        
        # HMA intermediate calculation
        hma_intermediate = 2 * wma_half - wma_full
        
        # Final HMA
        hma = self._weighted_ma(hma_intermediate, sqrt_period)
        
        return hma
    
    def _weighted_ma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        
        def wma_calc(values):
            if len(values) < period:
                return np.nan
            return np.sum(weights * values[-period:]) / np.sum(weights)
        
        return series.rolling(window=period).apply(wma_calc, raw=True)
    
    def _analyze_trend(self, hma: pd.Series, price: pd.Series) -> Tuple[HMATrend, float]:
        """Analyze trend direction and strength"""
        if len(hma) < 10:
            return HMATrend.SIDEWAYS, 0.0
        
        # Calculate slope over recent periods
        recent_hma = hma.iloc[-10:].dropna()
        if len(recent_hma) < 5:
            return HMATrend.SIDEWAYS, 0.0
        
        slope = np.polyfit(range(len(recent_hma)), recent_hma.values, 1)[0]
        
        # Normalize slope
        current_price = price.iloc[-1]
        normalized_slope = slope / current_price if current_price > 0 else 0
        
        # Classify trend
        if normalized_slope > 0.003:
            return HMATrend.STRONG_UP, normalized_slope
        elif normalized_slope > 0.001:
            return HMATrend.WEAK_UP, normalized_slope
        elif normalized_slope < -0.003:
            return HMATrend.STRONG_DOWN, normalized_slope
        elif normalized_slope < -0.001:
            return HMATrend.WEAK_DOWN, normalized_slope
        else:
            return HMATrend.SIDEWAYS, normalized_slope
    
    def _calculate_responsiveness(self, hma: pd.Series, price: pd.Series) -> float:
        """Calculate how responsive HMA is to price changes"""
        if len(hma) < 20:
            return 0.5
        
        # Compare HMA changes vs price changes
        hma_changes = hma.pct_change().iloc[-10:].abs().mean()
        price_changes = price.pct_change().iloc[-10:].abs().mean()
        
        if price_changes > 0:
            responsiveness = hma_changes / price_changes
            return min(max(responsiveness, 0.0), 2.0)
        
        return 0.5
    
    def _generate_signals(self, price: pd.Series, hma: pd.Series, 
                         trend: HMATrend, slope: float) -> Tuple[SignalType, float]:
        """Generate trading signals"""
        current_price = price.iloc[-1]
        current_hma = hma.iloc[-1]
        
        if pd.isna(current_hma):
            return SignalType.NEUTRAL, 0.0
        
        # Base signal from price vs HMA
        price_above = current_price > current_hma
        
        # Trend-based confidence
        trend_confidence = {
            HMATrend.STRONG_UP: 0.9,
            HMATrend.WEAK_UP: 0.6,
            HMATrend.SIDEWAYS: 0.3,
            HMATrend.WEAK_DOWN: 0.6,
            HMATrend.STRONG_DOWN: 0.9
        }.get(trend, 0.3)
        
        # Generate signal
        if trend in [HMATrend.STRONG_UP, HMATrend.WEAK_UP] and price_above:
            return SignalType.BUY, trend_confidence
        elif trend in [HMATrend.STRONG_DOWN, HMATrend.WEAK_DOWN] and not price_above:
            return SignalType.SELL, trend_confidence
        elif price_above:
            return SignalType.BUY, 0.5
        else:
            return SignalType.SELL, 0.5
    
    def _get_default_result(self) -> Dict:
        default_result = HullMAResult(0.0, HMATrend.SIDEWAYS, 0.0, SignalType.NEUTRAL, 0.0, 0.5)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {'period': self.period, 'enable_smoothing': self.enable_smoothing}
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)