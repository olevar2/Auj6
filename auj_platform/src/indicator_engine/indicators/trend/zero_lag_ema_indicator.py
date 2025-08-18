"""
Advanced Zero Lag EMA with Phase-Shifted Signal Processing

Features:
- Zero lag calculation using price anticipation
- Phase-shifted signal processing
- Trend detection with minimal delay
- Adaptive smoothing based on volatility
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class ZeroLagEMAState(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    CONSOLIDATION = "consolidation"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

@dataclass
class ZeroLagEMAResult:
    zlema_value: float
    ema_value: float
    lag_difference: float
    velocity: float
    state: ZeroLagEMAState
    signal: SignalType
    confidence: float

class ZeroLagEMAIndicator(StandardIndicatorInterface):
    def __init__(self, period: int = 21):
        self.period = period
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.period * 2:
                raise ValueError("Insufficient data")
            
            # Calculate Zero Lag EMA
            zlema_data = self._calculate_zlema(data)
            
            # Analyze state
            state = self._analyze_zlema_state(data, zlema_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, zlema_data, state)
            
            latest_result = ZeroLagEMAResult(
                zlema_value=zlema_data['zlema'].iloc[-1],
                ema_value=zlema_data['ema'].iloc[-1],
                lag_difference=zlema_data['lag_difference'].iloc[-1],
                velocity=zlema_data['velocity'].iloc[-1],
                state=state,
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'zlema': zlema_data['zlema'].tolist(),
                    'ema': zlema_data['ema'].tolist(),
                    'velocity': zlema_data['velocity'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'state': state.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Zero Lag EMA: {e}")
            return self._get_default_result()
    
    def _calculate_zlema(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        
        # Calculate standard EMA
        alpha = 2.0 / (self.period + 1)
        df['ema'] = data['close'].ewm(alpha=alpha, adjust=False).mean()
        
        # Calculate lag period
        lag = int((self.period - 1) / 2)
        
        # Calculate Zero Lag EMA
        # ZLEMA = EMA(data + (data - data[lag]))
        df['lagged_close'] = data['close'].shift(lag)
        df['corrected_close'] = data['close'] + (data['close'] - df['lagged_close'])
        df['zlema'] = df['corrected_close'].ewm(alpha=alpha, adjust=False).mean()
        
        # Calculate lag difference
        df['lag_difference'] = df['zlema'] - df['ema']
        
        # Calculate velocity (rate of change)
        df['velocity'] = df['zlema'].diff(periods=3)
        
        return df
    
    def _analyze_zlema_state(self, data: pd.DataFrame, zlema_data: pd.DataFrame) -> ZeroLagEMAState:
        current_price = data['close'].iloc[-1]
        current_zlema = zlema_data['zlema'].iloc[-1]
        velocity = zlema_data['velocity'].iloc[-1]
        
        if pd.isna(current_zlema) or pd.isna(velocity):
            return ZeroLagEMAState.CONSOLIDATION
        
        # Calculate relative velocity
        relative_velocity = velocity / current_price if current_price != 0 else 0
        
        # Determine trend based on price position and velocity
        price_above_zlema = current_price > current_zlema
        
        if price_above_zlema and velocity > 0:
            if relative_velocity > 0.01:
                return ZeroLagEMAState.STRONG_UPTREND
            else:
                return ZeroLagEMAState.UPTREND
        elif not price_above_zlema and velocity < 0:
            if relative_velocity < -0.01:
                return ZeroLagEMAState.STRONG_DOWNTREND
            else:
                return ZeroLagEMAState.DOWNTREND
        else:
            return ZeroLagEMAState.CONSOLIDATION
    
    def _generate_signals(self, data: pd.DataFrame, zlema_data: pd.DataFrame, 
                         state: ZeroLagEMAState) -> Tuple[SignalType, float]:
        current_price = data['close'].iloc[-1]
        current_zlema = zlema_data['zlema'].iloc[-1]
        velocity = zlema_data['velocity'].iloc[-1]
        
        if pd.isna(current_zlema) or pd.isna(velocity):
            return SignalType.NEUTRAL, 0.0
        
        # Calculate distance for confidence
        distance = abs(current_price - current_zlema) / current_price
        velocity_strength = abs(velocity) / current_price if current_price != 0 else 0
        
        base_confidence = min(0.8, (distance + velocity_strength) * 10)
        
        # State-based confidence
        confidence_map = {
            ZeroLagEMAState.STRONG_UPTREND: 0.9,
            ZeroLagEMAState.UPTREND: 0.7,
            ZeroLagEMAState.CONSOLIDATION: 0.3,
            ZeroLagEMAState.DOWNTREND: 0.7,
            ZeroLagEMAState.STRONG_DOWNTREND: 0.9
        }
        
        state_confidence = confidence_map.get(state, 0.5)
        final_confidence = (base_confidence + state_confidence) / 2
        
        # Generate signals
        if state in [ZeroLagEMAState.STRONG_UPTREND, ZeroLagEMAState.UPTREND]:
            return SignalType.BUY, final_confidence
        elif state in [ZeroLagEMAState.STRONG_DOWNTREND, ZeroLagEMAState.DOWNTREND]:
            return SignalType.SELL, final_confidence
        else:
            return SignalType.NEUTRAL, final_confidence
    
    def _get_default_result(self) -> Dict:
        default_result = ZeroLagEMAResult(0.0, 0.0, 0.0, 0.0, ZeroLagEMAState.CONSOLIDATION, 
                                        SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {'period': self.period}
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)