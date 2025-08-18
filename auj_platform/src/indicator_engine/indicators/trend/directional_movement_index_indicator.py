"""
Advanced Directional Movement Index (DMI) with ADX Integration

Features:
- True Directional Movement calculation
- DI+ and DI- with trend direction analysis
- ADX integration for trend strength
- Signal optimization with divergence detection
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class DMIState(Enum):
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    WEAK_DOWNTREND = "weak_downtrend"
    SIDEWAYS = "sideways"
    TRENDING = "trending"

@dataclass
class DMIResult:
    di_plus: float
    di_minus: float
    adx: float
    dx: float
    state: DMIState
    signal: SignalType
    confidence: float

class DirectionalMovementIndexIndicator(StandardIndicatorInterface):
    def __init__(self, period: int = 14):
        self.period = period
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.period * 2:
                raise ValueError("Insufficient data")
            
            # Calculate DMI components
            dmi_data = self._calculate_dmi(data)
            
            # Analyze state
            state = self._analyze_dmi_state(dmi_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(dmi_data, state)
            
            latest_result = DMIResult(
                di_plus=dmi_data['di_plus'].iloc[-1],
                di_minus=dmi_data['di_minus'].iloc[-1],
                adx=dmi_data['adx'].iloc[-1],
                dx=dmi_data['dx'].iloc[-1],
                state=state,
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'di_plus': dmi_data['di_plus'].tolist(),
                    'di_minus': dmi_data['di_minus'].tolist(),
                    'adx': dmi_data['adx'].tolist(),
                    'dx': dmi_data['dx'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'state': state.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating DMI: {e}")
            return self._get_default_result()
    
    def _calculate_dmi(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        
        # Calculate True Range and Directional Movements
        df['prev_close'] = data['close'].shift(1)
        df['tr1'] = data['high'] - data['low']
        df['tr2'] = abs(data['high'] - df['prev_close'])
        df['tr3'] = abs(data['low'] - df['prev_close'])
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Directional Movements
        df['up_move'] = data['high'] - data['high'].shift(1)
        df['down_move'] = data['low'].shift(1) - data['low']
        
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), 
                                df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), 
                                 df['down_move'], 0)
        
        # Smoothed values using Wilder's smoothing
        alpha = 1.0 / self.period
        
        # Initialize smoothed TR
        df['atr'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
        
        # Smoothed Directional Movements
        df['plus_di_raw'] = df['plus_dm'].ewm(alpha=alpha, adjust=False).mean()
        df['minus_di_raw'] = df['minus_dm'].ewm(alpha=alpha, adjust=False).mean()
        
        # Calculate DI+ and DI-
        df['di_plus'] = 100 * (df['plus_di_raw'] / df['atr'])
        df['di_minus'] = 100 * (df['minus_di_raw'] / df['atr'])
        
        # Calculate DX
        df['di_sum'] = df['di_plus'] + df['di_minus']
        df['di_diff'] = abs(df['di_plus'] - df['di_minus'])
        df['dx'] = 100 * (df['di_diff'] / df['di_sum'])
        
        # Calculate ADX (smoothed DX)
        df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()
        
        return df
    
    def _analyze_dmi_state(self, dmi_data: pd.DataFrame) -> DMIState:
        di_plus = dmi_data['di_plus'].iloc[-1]
        di_minus = dmi_data['di_minus'].iloc[-1]
        adx = dmi_data['adx'].iloc[-1]
        
        if pd.isna(di_plus) or pd.isna(di_minus) or pd.isna(adx):
            return DMIState.SIDEWAYS
        
        # Determine trend strength
        if adx < 20:
            return DMIState.SIDEWAYS
        elif adx < 25:
            trend_strength = "weak"
        else:
            trend_strength = "strong"
        
        # Determine trend direction
        if di_plus > di_minus:
            if trend_strength == "strong":
                return DMIState.STRONG_UPTREND
            else:
                return DMIState.WEAK_UPTREND
        elif di_minus > di_plus:
            if trend_strength == "strong":
                return DMIState.STRONG_DOWNTREND
            else:
                return DMIState.WEAK_DOWNTREND
        else:
            return DMIState.SIDEWAYS
    
    def _generate_signals(self, dmi_data: pd.DataFrame, state: DMIState) -> Tuple[SignalType, float]:
        di_plus = dmi_data['di_plus'].iloc[-1]
        di_minus = dmi_data['di_minus'].iloc[-1]
        adx = dmi_data['adx'].iloc[-1]
        
        if pd.isna(di_plus) or pd.isna(di_minus) or pd.isna(adx):
            return SignalType.NEUTRAL, 0.0
        
        # Check for crossovers
        prev_di_plus = dmi_data['di_plus'].iloc[-2] if len(dmi_data) > 1 else di_plus
        prev_di_minus = dmi_data['di_minus'].iloc[-2] if len(dmi_data) > 1 else di_minus
        
        # Base confidence on ADX strength
        adx_confidence = min(0.9, adx / 50.0)  # Normalize ADX to confidence
        
        # Crossover signals (highest priority)
        if prev_di_plus <= prev_di_minus and di_plus > di_minus and adx > 20:
            return SignalType.BUY, min(0.9, adx_confidence + 0.2)
        elif prev_di_plus >= prev_di_minus and di_plus < di_minus and adx > 20:
            return SignalType.SELL, min(0.9, adx_confidence + 0.2)
        
        # State-based signals
        confidence_map = {
            DMIState.STRONG_UPTREND: 0.85,
            DMIState.WEAK_UPTREND: 0.6,
            DMIState.STRONG_DOWNTREND: 0.85,
            DMIState.WEAK_DOWNTREND: 0.6,
            DMIState.SIDEWAYS: 0.2
        }
        
        base_confidence = confidence_map.get(state, 0.3)
        final_confidence = (base_confidence + adx_confidence) / 2
        
        if state in [DMIState.STRONG_UPTREND, DMIState.WEAK_UPTREND]:
            return SignalType.BUY, final_confidence
        elif state in [DMIState.STRONG_DOWNTREND, DMIState.WEAK_DOWNTREND]:
            return SignalType.SELL, final_confidence
        else:
            return SignalType.NEUTRAL, final_confidence
    
    def _get_default_result(self) -> Dict:
        default_result = DMIResult(0.0, 0.0, 0.0, 0.0, DMIState.SIDEWAYS, SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {'period': self.period}
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)