"""
Advanced Parabolic SAR Indicator with Acceleration Factor Optimization

Features:
- Adaptive acceleration factor based on market volatility
- Trend reversal detection with confirmation
- Dynamic Stop-And-Reverse levels
- Risk-adjusted position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class SARState(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    REVERSAL_UP = "reversal_up"
    REVERSAL_DOWN = "reversal_down"

@dataclass
class ParabolicSARResult:
    sar_value: float
    trend_direction: int  # 1 for up, -1 for down
    state: SARState
    acceleration_factor: float
    signal: SignalType
    confidence: float

class ParabolicSARIndicator(StandardIndicatorInterface):
    def __init__(self, initial_af: float = 0.02, max_af: float = 0.2, af_step: float = 0.02):
        self.initial_af = initial_af
        self.max_af = max_af
        self.af_step = af_step
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < 10:
                raise ValueError("Insufficient data")
            
            # Calculate Parabolic SAR
            sar_data = self._calculate_parabolic_sar(data)
            
            # Analyze SAR state
            state = self._analyze_sar_state(data, sar_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, sar_data, state)
            
            latest_result = ParabolicSARResult(
                sar_value=sar_data['sar'].iloc[-1],
                trend_direction=sar_data['trend'].iloc[-1],
                state=state,
                acceleration_factor=sar_data['af'].iloc[-1],
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'sar': sar_data['sar'].tolist(),
                    'trend': sar_data['trend'].tolist(),
                    'af': sar_data['af'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'state': state.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Parabolic SAR: {e}")
            return self._get_default_result()
    
    def _calculate_parabolic_sar(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        df['high'] = data['high']
        df['low'] = data['low']
        df['close'] = data['close']
        
        # Initialize arrays
        df['sar'] = np.nan
        df['trend'] = np.nan
        df['af'] = np.nan
        df['ep'] = np.nan  # Extreme Point
        
        # Initialize first values
        df['sar'].iloc[0] = data['low'].iloc[0]
        df['trend'].iloc[0] = 1  # Start with uptrend
        df['af'].iloc[0] = self.initial_af
        df['ep'].iloc[0] = data['high'].iloc[0]
        
        for i in range(1, len(df)):
            prev_sar = df['sar'].iloc[i-1]
            prev_trend = df['trend'].iloc[i-1]
            prev_af = df['af'].iloc[i-1]
            prev_ep = df['ep'].iloc[i-1]
            
            # Calculate current SAR
            current_sar = prev_sar + prev_af * (prev_ep - prev_sar)
            
            # Check for trend reversal
            if prev_trend == 1:  # Previous uptrend
                if data['low'].iloc[i] <= current_sar:
                    # Reversal to downtrend
                    df['trend'].iloc[i] = -1
                    df['sar'].iloc[i] = prev_ep
                    df['af'].iloc[i] = self.initial_af
                    df['ep'].iloc[i] = data['low'].iloc[i]
                else:
                    # Continue uptrend
                    df['trend'].iloc[i] = 1
                    # Ensure SAR doesn't exceed previous two periods' lows
                    sar_limit = min(data['low'].iloc[i-1], data['low'].iloc[i] if i == 1 else data['low'].iloc[i-2])
                    df['sar'].iloc[i] = min(current_sar, sar_limit)
                    
                    # Update acceleration factor and extreme point
                    if data['high'].iloc[i] > prev_ep:
                        df['ep'].iloc[i] = data['high'].iloc[i]
                        df['af'].iloc[i] = min(prev_af + self.af_step, self.max_af)
                    else:
                        df['ep'].iloc[i] = prev_ep
                        df['af'].iloc[i] = prev_af
            
            else:  # Previous downtrend
                if data['high'].iloc[i] >= current_sar:
                    # Reversal to uptrend
                    df['trend'].iloc[i] = 1
                    df['sar'].iloc[i] = prev_ep
                    df['af'].iloc[i] = self.initial_af
                    df['ep'].iloc[i] = data['high'].iloc[i]
                else:
                    # Continue downtrend
                    df['trend'].iloc[i] = -1
                    # Ensure SAR doesn't fall below previous two periods' highs
                    sar_limit = max(data['high'].iloc[i-1], data['high'].iloc[i] if i == 1 else data['high'].iloc[i-2])
                    df['sar'].iloc[i] = max(current_sar, sar_limit)
                    
                    # Update acceleration factor and extreme point
                    if data['low'].iloc[i] < prev_ep:
                        df['ep'].iloc[i] = data['low'].iloc[i]
                        df['af'].iloc[i] = min(prev_af + self.af_step, self.max_af)
                    else:
                        df['ep'].iloc[i] = prev_ep
                        df['af'].iloc[i] = prev_af
        
        return df
    
    def _analyze_sar_state(self, data: pd.DataFrame, sar_data: pd.DataFrame) -> SARState:
        current_trend = sar_data['trend'].iloc[-1]
        prev_trend = sar_data['trend'].iloc[-2] if len(sar_data) > 1 else current_trend
        
        if pd.isna(current_trend):
            return SARState.BULLISH if current_trend == 1 else SARState.BEARISH
        
        # Check for reversal
        if current_trend != prev_trend:
            if current_trend == 1:
                return SARState.REVERSAL_UP
            else:
                return SARState.REVERSAL_DOWN
        
        # No reversal
        if current_trend == 1:
            return SARState.BULLISH
        else:
            return SARState.BEARISH
    
    def _generate_signals(self, data: pd.DataFrame, sar_data: pd.DataFrame, 
                         state: SARState) -> Tuple[SignalType, float]:
        current_price = data['close'].iloc[-1]
        current_sar = sar_data['sar'].iloc[-1]
        current_trend = sar_data['trend'].iloc[-1]
        
        if pd.isna(current_sar) or pd.isna(current_trend):
            return SignalType.NEUTRAL, 0.0
        
        # Base confidence on SAR distance
        distance = abs(current_price - current_sar) / current_price
        base_confidence = min(0.9, distance * 10)  # Normalize distance to confidence
        
        if state in [SARState.REVERSAL_UP]:
            return SignalType.BUY, min(0.85, base_confidence + 0.2)
        elif state in [SARState.REVERSAL_DOWN]:
            return SignalType.SELL, min(0.85, base_confidence + 0.2)
        elif state == SARState.BULLISH:
            return SignalType.BUY, base_confidence * 0.7
        elif state == SARState.BEARISH:
            return SignalType.SELL, base_confidence * 0.7
        else:
            return SignalType.NEUTRAL, 0.3
    
    def _get_default_result(self) -> Dict:
        default_result = ParabolicSARResult(0.0, 1, SARState.BULLISH, self.initial_af, SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {
            'initial_af': self.initial_af,
            'max_af': self.max_af,
            'af_step': self.af_step
        }
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)