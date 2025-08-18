"""
Advanced Kaufman's Adaptive Moving Average (KAMA) with Efficiency Ratio

Features:
- Efficiency Ratio calculation for market noise measurement
- Adaptive smoothing based on market conditions
- Directional movement analysis
- Signal optimization with confidence scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class KAMAState(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CHOPPY = "choppy"
    ACCELERATING_UP = "accelerating_up"
    ACCELERATING_DOWN = "accelerating_down"

@dataclass
class KAMAResult:
    kama_value: float
    efficiency_ratio: float
    smoothing_constant: float
    state: KAMAState
    signal: SignalType
    confidence: float

class KaufmanAdaptiveMovingAverageIndicator(StandardIndicatorInterface):
    def __init__(self, period: int = 10, fast_sc: int = 2, slow_sc: int = 30):
        self.period = period
        self.fast_sc = fast_sc
        self.slow_sc = slow_sc
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.period + 10:
                raise ValueError("Insufficient data")
            
            # Calculate KAMA
            kama_data = self._calculate_kama(data)
            
            # Analyze state
            state = self._analyze_kama_state(data, kama_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, kama_data, state)
            
            latest_result = KAMAResult(
                kama_value=kama_data['kama'].iloc[-1],
                efficiency_ratio=kama_data['efficiency_ratio'].iloc[-1],
                smoothing_constant=kama_data['smoothing_constant'].iloc[-1],
                state=state,
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'kama': kama_data['kama'].tolist(),
                    'efficiency_ratio': kama_data['efficiency_ratio'].tolist(),
                    'smoothing_constant': kama_data['smoothing_constant'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'state': state.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating KAMA: {e}")
            return self._get_default_result()
    
    def _calculate_kama(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        df['close'] = data['close']
        
        # Calculate Change (absolute price change over period)
        df['change'] = abs(data['close'] - data['close'].shift(self.period))
        
        # Calculate Volatility (sum of absolute price changes)
        df['volatility'] = abs(data['close'].diff()).rolling(window=self.period).sum()
        
        # Calculate Efficiency Ratio
        df['efficiency_ratio'] = df['change'] / df['volatility']
        df['efficiency_ratio'] = df['efficiency_ratio'].fillna(0)
        
        # Calculate Smoothing Constants
        fastest_sc = 2.0 / (self.fast_sc + 1)
        slowest_sc = 2.0 / (self.slow_sc + 1)
        
        df['smoothing_constant'] = (df['efficiency_ratio'] * (fastest_sc - slowest_sc) + slowest_sc) ** 2
        
        # Calculate KAMA
        df['kama'] = np.nan
        df['kama'].iloc[self.period] = data['close'].iloc[self.period]
        
        for i in range(self.period + 1, len(df)):
            if not pd.isna(df['smoothing_constant'].iloc[i]):
                df['kama'].iloc[i] = (df['kama'].iloc[i-1] + 
                                     df['smoothing_constant'].iloc[i] * 
                                     (data['close'].iloc[i] - df['kama'].iloc[i-1]))
        
        return df
    
    def _analyze_kama_state(self, data: pd.DataFrame, kama_data: pd.DataFrame) -> KAMAState:
        current_kama = kama_data['kama'].iloc[-1]
        prev_kama = kama_data['kama'].iloc[-2] if len(kama_data) > 1 else current_kama
        efficiency_ratio = kama_data['efficiency_ratio'].iloc[-1]
        
        if pd.isna(current_kama) or pd.isna(efficiency_ratio):
            return KAMAState.CHOPPY
        
        # Determine trend direction
        kama_slope = current_kama - prev_kama
        
        # Efficiency ratio threshold for trending vs choppy
        if efficiency_ratio < 0.3:
            return KAMAState.CHOPPY
        
        if kama_slope > 0:
            if efficiency_ratio > 0.7:
                return KAMAState.ACCELERATING_UP
            else:
                return KAMAState.TRENDING_UP
        elif kama_slope < 0:
            if efficiency_ratio > 0.7:
                return KAMAState.ACCELERATING_DOWN
            else:
                return KAMAState.TRENDING_DOWN
        else:
            return KAMAState.CHOPPY
    
    def _generate_signals(self, data: pd.DataFrame, kama_data: pd.DataFrame, 
                         state: KAMAState) -> Tuple[SignalType, float]:
        current_price = data['close'].iloc[-1]
        current_kama = kama_data['kama'].iloc[-1]
        efficiency_ratio = kama_data['efficiency_ratio'].iloc[-1]
        
        if pd.isna(current_kama) or pd.isna(efficiency_ratio):
            return SignalType.NEUTRAL, 0.0
        
        # Base confidence on efficiency ratio
        base_confidence = min(0.9, efficiency_ratio)
        
        # State-based signals
        if state == KAMAState.ACCELERATING_UP:
            return SignalType.BUY, base_confidence * 0.9
        elif state == KAMAState.ACCELERATING_DOWN:
            return SignalType.SELL, base_confidence * 0.9
        elif state == KAMAState.TRENDING_UP and current_price > current_kama:
            return SignalType.BUY, base_confidence * 0.7
        elif state == KAMAState.TRENDING_DOWN and current_price < current_kama:
            return SignalType.SELL, base_confidence * 0.7
        elif state == KAMAState.CHOPPY:
            return SignalType.NEUTRAL, 0.2
        else:
            # Price vs KAMA position
            if current_price > current_kama:
                return SignalType.BUY, base_confidence * 0.5
            else:
                return SignalType.SELL, base_confidence * 0.5
    
    def _get_default_result(self) -> Dict:
        default_result = KAMAResult(0.0, 0.0, 0.0, KAMAState.CHOPPY, SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {
            'period': self.period,
            'fast_sc': self.fast_sc,
            'slow_sc': self.slow_sc
        }
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)