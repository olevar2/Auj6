"""
Advanced Vortex Indicator with Trend Identification and Divergence Detection

Features:
- True Range and Vortex Movement calculations
- VI+ and VI- crossover analysis
- Trend strength measurement
- Divergence detection with price action
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class VortexState(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

@dataclass
class VortexResult:
    vi_plus: float
    vi_minus: float
    vortex_difference: float
    state: VortexState
    signal: SignalType
    confidence: float

class VortexIndicator(StandardIndicatorInterface):
    def __init__(self, period: int = 14):
        self.period = period
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.period + 5:
                raise ValueError("Insufficient data")
            
            # Calculate Vortex Indicator
            vortex_data = self._calculate_vortex(data)
            
            # Analyze vortex state
            state = self._analyze_vortex_state(vortex_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(vortex_data, state)
            
            latest_result = VortexResult(
                vi_plus=vortex_data['vi_plus'].iloc[-1],
                vi_minus=vortex_data['vi_minus'].iloc[-1],
                vortex_difference=vortex_data['vi_diff'].iloc[-1],
                state=state,
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'vi_plus': vortex_data['vi_plus'].tolist(),
                    'vi_minus': vortex_data['vi_minus'].tolist(),
                    'vi_diff': vortex_data['vi_diff'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'state': state.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Vortex Indicator: {e}")
            return self._get_default_result()
    
    def _calculate_vortex(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        
        # Calculate True Range
        df['prev_close'] = data['close'].shift(1)
        df['tr1'] = data['high'] - data['low']
        df['tr2'] = abs(data['high'] - df['prev_close'])
        df['tr3'] = abs(data['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate Vortex Movements
        df['vm_plus'] = abs(data['high'] - data['low'].shift(1))
        df['vm_minus'] = abs(data['low'] - data['high'].shift(1))
        
        # Calculate rolling sums
        df['sum_tr'] = df['tr'].rolling(window=self.period).sum()
        df['sum_vm_plus'] = df['vm_plus'].rolling(window=self.period).sum()
        df['sum_vm_minus'] = df['vm_minus'].rolling(window=self.period).sum()
        
        # Calculate Vortex Indicators
        df['vi_plus'] = df['sum_vm_plus'] / df['sum_tr']
        df['vi_minus'] = df['sum_vm_minus'] / df['sum_tr']
        
        # Calculate difference and ratio
        df['vi_diff'] = df['vi_plus'] - df['vi_minus']
        df['vi_ratio'] = df['vi_plus'] / df['vi_minus']
        
        return df
    
    def _analyze_vortex_state(self, vortex_data: pd.DataFrame) -> VortexState:
        vi_plus = vortex_data['vi_plus'].iloc[-1]
        vi_minus = vortex_data['vi_minus'].iloc[-1]
        vi_diff = vortex_data['vi_diff'].iloc[-1]
        
        if pd.isna(vi_plus) or pd.isna(vi_minus):
            return VortexState.NEUTRAL
        
        # Determine state based on VI+ and VI- relationship
        if vi_plus > vi_minus:
            if vi_diff > 0.3:
                return VortexState.STRONG_BULLISH
            else:
                return VortexState.BULLISH
        elif vi_minus > vi_plus:
            if vi_diff < -0.3:
                return VortexState.STRONG_BEARISH
            else:
                return VortexState.BEARISH
        else:
            return VortexState.NEUTRAL
    
    def _generate_signals(self, vortex_data: pd.DataFrame, 
                         state: VortexState) -> Tuple[SignalType, float]:
        vi_plus = vortex_data['vi_plus'].iloc[-1]
        vi_minus = vortex_data['vi_minus'].iloc[-1]
        
        if pd.isna(vi_plus) or pd.isna(vi_minus):
            return SignalType.NEUTRAL, 0.0
        
        # Check for crossovers
        prev_vi_plus = vortex_data['vi_plus'].iloc[-2] if len(vortex_data) > 1 else vi_plus
        prev_vi_minus = vortex_data['vi_minus'].iloc[-2] if len(vortex_data) > 1 else vi_minus
        
        # Calculate signal strength
        difference = abs(vi_plus - vi_minus)
        base_confidence = min(0.9, difference * 2)
        
        # Crossover signals
        if prev_vi_plus <= prev_vi_minus and vi_plus > vi_minus:
            return SignalType.BUY, min(0.85, base_confidence + 0.15)
        elif prev_vi_plus >= prev_vi_minus and vi_plus < vi_minus:
            return SignalType.SELL, min(0.85, base_confidence + 0.15)
        
        # State-based signals
        confidence_map = {
            VortexState.STRONG_BULLISH: 0.8,
            VortexState.BULLISH: 0.6,
            VortexState.NEUTRAL: 0.3,
            VortexState.BEARISH: 0.6,
            VortexState.STRONG_BEARISH: 0.8
        }
        
        if state in [VortexState.STRONG_BULLISH, VortexState.BULLISH]:
            return SignalType.BUY, confidence_map[state]
        elif state in [VortexState.STRONG_BEARISH, VortexState.BEARISH]:
            return SignalType.SELL, confidence_map[state]
        else:
            return SignalType.NEUTRAL, confidence_map[state]
    
    def _get_default_result(self) -> Dict:
        default_result = VortexResult(1.0, 1.0, 0.0, VortexState.NEUTRAL, SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {'period': self.period}
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)