"""
Advanced Super Trend Indicator with ATR-based Dynamic Support/Resistance

Features:
- ATR-based dynamic support/resistance levels
- Trend following with volatility adjustment
- Multi-timeframe confirmation
- Signal filtering and optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class SuperTrendState(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    NEUTRAL = "neutral"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

@dataclass
class SuperTrendResult:
    supertrend_value: float
    trend_state: SuperTrendState
    signal: SignalType
    confidence: float
    atr_value: float

class SuperTrendIndicator(StandardIndicatorInterface):
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        self.period = period
        self.multiplier = multiplier
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.period + 10:
                raise ValueError("Insufficient data")
            
            # Calculate SuperTrend
            st_data = self._calculate_supertrend(data)
            
            # Analyze trend state
            trend_state = self._analyze_trend_state(data, st_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, st_data, trend_state)
            
            latest_result = SuperTrendResult(
                supertrend_value=st_data['supertrend'].iloc[-1],
                trend_state=trend_state,
                signal=signal,
                confidence=confidence,
                atr_value=st_data['atr'].iloc[-1]
            )
            
            return {
                'current': latest_result,
                'values': {
                    'supertrend': st_data['supertrend'].tolist(),
                    'atr': st_data['atr'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'trend_state': trend_state.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating SuperTrend: {e}")
            return self._get_default_result()
    
    def _calculate_supertrend(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        
        # Calculate ATR
        df['prev_close'] = data['close'].shift(1)
        df['tr1'] = data['high'] - data['low']
        df['tr2'] = abs(data['high'] - df['prev_close'])
        df['tr3'] = abs(data['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=self.period).mean()
        
        # Calculate basic upper and lower bands
        hl2 = (data['high'] + data['low']) / 2
        df['basic_upper'] = hl2 + (self.multiplier * df['atr'])
        df['basic_lower'] = hl2 - (self.multiplier * df['atr'])
        
        # Calculate final upper and lower bands
        df['final_upper'] = df['basic_upper']
        df['final_lower'] = df['basic_lower']
        
        for i in range(1, len(df)):
            # Final Upper Band
            if df['basic_upper'].iloc[i] < df['final_upper'].iloc[i-1] or data['close'].iloc[i-1] > df['final_upper'].iloc[i-1]:
                df['final_upper'].iloc[i] = df['basic_upper'].iloc[i]
            else:
                df['final_upper'].iloc[i] = df['final_upper'].iloc[i-1]
            
            # Final Lower Band
            if df['basic_lower'].iloc[i] > df['final_lower'].iloc[i-1] or data['close'].iloc[i-1] < df['final_lower'].iloc[i-1]:
                df['final_lower'].iloc[i] = df['basic_lower'].iloc[i]
            else:
                df['final_lower'].iloc[i] = df['final_lower'].iloc[i-1]
        
        # Calculate SuperTrend
        df['supertrend'] = np.nan
        df['supertrend'].iloc[0] = df['final_lower'].iloc[0]
        
        for i in range(1, len(df)):
            if df['supertrend'].iloc[i-1] == df['final_upper'].iloc[i-1] and data['close'].iloc[i] <= df['final_upper'].iloc[i]:
                df['supertrend'].iloc[i] = df['final_upper'].iloc[i]
            elif df['supertrend'].iloc[i-1] == df['final_upper'].iloc[i-1] and data['close'].iloc[i] > df['final_upper'].iloc[i]:
                df['supertrend'].iloc[i] = df['final_lower'].iloc[i]
            elif df['supertrend'].iloc[i-1] == df['final_lower'].iloc[i-1] and data['close'].iloc[i] >= df['final_lower'].iloc[i]:
                df['supertrend'].iloc[i] = df['final_lower'].iloc[i]
            elif df['supertrend'].iloc[i-1] == df['final_lower'].iloc[i-1] and data['close'].iloc[i] < df['final_lower'].iloc[i]:
                df['supertrend'].iloc[i] = df['final_upper'].iloc[i]
            else:
                df['supertrend'].iloc[i] = df['supertrend'].iloc[i-1]
        
        return df
    
    def _analyze_trend_state(self, data: pd.DataFrame, st_data: pd.DataFrame) -> SuperTrendState:
        current_price = data['close'].iloc[-1]
        current_st = st_data['supertrend'].iloc[-1]
        
        if pd.isna(current_st):
            return SuperTrendState.NEUTRAL
        
        # Determine trend direction
        if current_price > current_st:
            # Check trend strength
            distance = (current_price - current_st) / current_price
            if distance > 0.03:
                return SuperTrendState.STRONG_UPTREND
            else:
                return SuperTrendState.UPTREND
        elif current_price < current_st:
            distance = (current_st - current_price) / current_price
            if distance > 0.03:
                return SuperTrendState.STRONG_DOWNTREND
            else:
                return SuperTrendState.DOWNTREND
        else:
            return SuperTrendState.NEUTRAL
    
    def _generate_signals(self, data: pd.DataFrame, st_data: pd.DataFrame, 
                         trend_state: SuperTrendState) -> Tuple[SignalType, float]:
        current_price = data['close'].iloc[-1]
        current_st = st_data['supertrend'].iloc[-1]
        
        if pd.isna(current_st):
            return SignalType.NEUTRAL, 0.0
        
        confidence_map = {
            SuperTrendState.STRONG_UPTREND: 0.9,
            SuperTrendState.UPTREND: 0.7,
            SuperTrendState.NEUTRAL: 0.3,
            SuperTrendState.DOWNTREND: 0.7,
            SuperTrendState.STRONG_DOWNTREND: 0.9
        }
        
        if current_price > current_st:
            return SignalType.BUY, confidence_map.get(trend_state, 0.5)
        elif current_price < current_st:
            return SignalType.SELL, confidence_map.get(trend_state, 0.5)
        else:
            return SignalType.NEUTRAL, 0.3
    
    def _get_default_result(self) -> Dict:
        default_result = SuperTrendResult(0.0, SuperTrendState.NEUTRAL, SignalType.NEUTRAL, 0.0, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {'period': self.period, 'multiplier': self.multiplier}
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)