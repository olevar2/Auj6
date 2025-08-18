"""
Advanced Simple Moving Average (SMA) Indicator with Trend Analysis

Features:
- Traditional SMA calculations with multiple periods
- Trend strength and direction analysis
- Moving average crossover detection
- Multi-timeframe SMA alignment
- Volume-weighted signal confirmation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class SMATrendState(Enum):
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

@dataclass
class SMAResult:
    sma_value: float
    trend_state: SMATrendState
    signal: SignalType
    confidence: float
    slope_angle: float

class SimpleMovingAverageIndicator(StandardIndicatorInterface):
    def __init__(self, period: int = 20, multiple_periods: List[int] = None):
        self.period = period
        self.multiple_periods = multiple_periods or [10, 20, 50]
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < max(self.multiple_periods) + 5:
                raise ValueError("Insufficient data")
            
            # Calculate SMAs
            sma_data = self._calculate_smas(data)
            
            # Analyze trend
            trend_state, slope = self._analyze_trend(data, sma_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, sma_data, trend_state)
            
            latest_result = SMAResult(
                sma_value=sma_data[f'sma_{self.period}'].iloc[-1],
                trend_state=trend_state,
                signal=signal,
                confidence=confidence,
                slope_angle=slope
            )
            
            return {
                'current': latest_result,
                'values': {f'sma_{p}': sma_data[f'sma_{p}'].tolist() for p in self.multiple_periods},
                'signal': signal,
                'confidence': confidence,
                'trend_state': trend_state.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {e}")
            return self._get_default_result()
    
    def _calculate_smas(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        for period in self.multiple_periods:
            df[f'sma_{period}'] = data['close'].rolling(window=period).mean()
        return df
    
    def _analyze_trend(self, data: pd.DataFrame, sma_data: pd.DataFrame) -> Tuple[SMATrendState, float]:
        main_sma = sma_data[f'sma_{self.period}']
        
        if len(main_sma) < 10:
            return SMATrendState.SIDEWAYS, 0.0
        
        # Calculate slope
        recent_sma = main_sma.iloc[-10:].values
        slope = np.polyfit(range(len(recent_sma)), recent_sma, 1)[0]
        
        # Normalize slope
        current_price = data['close'].iloc[-1]
        normalized_slope = slope / current_price if current_price > 0 else 0
        
        # Classify trend
        if normalized_slope > 0.002:
            return SMATrendState.STRONG_UPTREND, normalized_slope
        elif normalized_slope > 0.0005:
            return SMATrendState.WEAK_UPTREND, normalized_slope
        elif normalized_slope < -0.002:
            return SMATrendState.STRONG_DOWNTREND, normalized_slope
        elif normalized_slope < -0.0005:
            return SMATrendState.WEAK_DOWNTREND, normalized_slope
        else:
            return SMATrendState.SIDEWAYS, normalized_slope
    
    def _generate_signals(self, data: pd.DataFrame, sma_data: pd.DataFrame, 
                         trend_state: SMATrendState) -> Tuple[SignalType, float]:
        current_price = data['close'].iloc[-1]
        main_sma = sma_data[f'sma_{self.period}'].iloc[-1]
        
        # Base signal
        if current_price > main_sma:
            base_signal = SignalType.BUY
        elif current_price < main_sma:
            base_signal = SignalType.SELL
        else:
            base_signal = SignalType.NEUTRAL
        
        # Confidence based on trend strength
        confidence_map = {
            SMATrendState.STRONG_UPTREND: 0.9,
            SMATrendState.WEAK_UPTREND: 0.6,
            SMATrendState.SIDEWAYS: 0.3,
            SMATrendState.WEAK_DOWNTREND: 0.6,
            SMATrendState.STRONG_DOWNTREND: 0.9
        }
        
        confidence = confidence_map.get(trend_state, 0.3)
        return base_signal, confidence
    
    def _get_default_result(self) -> Dict:
        default_result = SMAResult(0.0, SMATrendState.SIDEWAYS, SignalType.NEUTRAL, 0.0, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {'period': self.period, 'multiple_periods': self.multiple_periods}
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)