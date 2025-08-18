"""
Advanced Weighted Moving Average (WMA) with Linearly Decreasing Weights

Features:
- Linear weight distribution favoring recent prices
- Trend analysis with slope calculation
- Multi-period comparison for strength assessment
- Signal generation with momentum confirmation
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class WMAState(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    FLAT = "flat"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

@dataclass
class WMAResult:
    wma_value: float
    slope: float
    momentum: float
    state: WMAState
    signal: SignalType
    confidence: float

class WeightedMovingAverageIndicator(StandardIndicatorInterface):
    def __init__(self, period: int = 14):
        self.period = period
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.period + 5:
                raise ValueError("Insufficient data")
            
            # Calculate WMA
            wma_data = self._calculate_wma(data)
            
            # Analyze state
            state = self._analyze_wma_state(data, wma_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, wma_data, state)
            
            latest_result = WMAResult(
                wma_value=wma_data['wma'].iloc[-1],
                slope=wma_data['slope'].iloc[-1],
                momentum=wma_data['momentum'].iloc[-1],
                state=state,
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'wma': wma_data['wma'].tolist(),
                    'slope': wma_data['slope'].tolist(),
                    'momentum': wma_data['momentum'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'state': state.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating WMA: {e}")
            return self._get_default_result()
    
    def _calculate_wma(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        
        # Calculate weights (linearly decreasing)
        weights = np.arange(1, self.period + 1)
        weights_sum = weights.sum()
        
        # Calculate WMA
        wma_values = []
        for i in range(len(data)):
            if i < self.period - 1:
                wma_values.append(np.nan)
            else:
                # Get price window
                price_window = data['close'].iloc[i - self.period + 1:i + 1].values
                # Calculate weighted average
                wma_value = np.sum(price_window * weights) / weights_sum
                wma_values.append(wma_value)
        
        df['wma'] = wma_values
        
        # Calculate slope (rate of change)
        df['slope'] = df['wma'].diff(periods=3)  # 3-period slope
        
        # Calculate momentum (rate of change over longer period)
        df['momentum'] = df['wma'].pct_change(periods=5) * 100  # 5-period momentum
        
        return df
    
    def _analyze_wma_state(self, data: pd.DataFrame, wma_data: pd.DataFrame) -> WMAState:
        current_slope = wma_data['slope'].iloc[-1]
        current_momentum = wma_data['momentum'].iloc[-1]
        current_wma = wma_data['wma'].iloc[-1]
        
        if pd.isna(current_slope) or pd.isna(current_momentum):
            return WMAState.FLAT
        
        # Normalize slope relative to price
        normalized_slope = abs(current_slope) / current_wma if current_wma != 0 else 0
        
        # Determine trend strength and direction
        if current_slope > 0:
            if normalized_slope > 0.005 and current_momentum > 0.5:
                return WMAState.STRONG_UPTREND
            else:
                return WMAState.UPTREND
        elif current_slope < 0:
            if normalized_slope > 0.005 and current_momentum < -0.5:
                return WMAState.STRONG_DOWNTREND
            else:
                return WMAState.DOWNTREND
        else:
            return WMAState.FLAT
    
    def _generate_signals(self, data: pd.DataFrame, wma_data: pd.DataFrame, 
                         state: WMAState) -> Tuple[SignalType, float]:
        current_price = data['close'].iloc[-1]
        current_wma = wma_data['wma'].iloc[-1]
        current_momentum = wma_data['momentum'].iloc[-1]
        
        if pd.isna(current_wma) or pd.isna(current_momentum):
            return SignalType.NEUTRAL, 0.0
        
        # Price position relative to WMA
        price_above_wma = current_price > current_wma
        price_distance = abs(current_price - current_wma) / current_price
        
        # Base confidence on momentum and price distance
        momentum_confidence = min(0.8, abs(current_momentum) / 2.0)
        distance_confidence = min(0.7, price_distance * 20)
        base_confidence = (momentum_confidence + distance_confidence) / 2
        
        # State-based confidence adjustment
        confidence_map = {
            WMAState.STRONG_UPTREND: 0.9,
            WMAState.UPTREND: 0.7,
            WMAState.FLAT: 0.3,
            WMAState.DOWNTREND: 0.7,
            WMAState.STRONG_DOWNTREND: 0.9
        }
        
        state_confidence = confidence_map.get(state, 0.5)
        final_confidence = (base_confidence + state_confidence) / 2
        
        # Generate signals
        if state in [WMAState.STRONG_UPTREND, WMAState.UPTREND] and price_above_wma:
            return SignalType.BUY, final_confidence
        elif state in [WMAState.STRONG_DOWNTREND, WMAState.DOWNTREND] and not price_above_wma:
            return SignalType.SELL, final_confidence
        elif price_above_wma and current_momentum > 0:
            return SignalType.BUY, final_confidence * 0.6
        elif not price_above_wma and current_momentum < 0:
            return SignalType.SELL, final_confidence * 0.6
        else:
            return SignalType.NEUTRAL, 0.3
    
    def _get_default_result(self) -> Dict:
        default_result = WMAResult(0.0, 0.0, 0.0, WMAState.FLAT, SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {'period': self.period}
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)