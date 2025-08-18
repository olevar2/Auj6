"""
Advanced Triple EMA (TEMA) Indicator with Lag Reduction and Trend Analysis

Features:
- Triple exponential smoothing for reduced lag
- Trend direction and strength analysis
- Signal filtering with confidence levels
- Adaptive period optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class TEMAState(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    SIDEWAYS = "sideways"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

@dataclass
class TEMAResult:
    tema_value: float
    ema1: float
    ema2: float
    ema3: float
    trend_slope: float
    state: TEMAState
    signal: SignalType
    confidence: float

class TripleEMAIndicator(StandardIndicatorInterface):
    def __init__(self, period: int = 14):
        self.period = period
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.period * 3:
                raise ValueError("Insufficient data")
            
            # Calculate Triple EMA
            tema_data = self._calculate_tema(data)
            
            # Analyze trend state
            state = self._analyze_trend_state(tema_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, tema_data, state)
            
            latest_result = TEMAResult(
                tema_value=tema_data['tema'].iloc[-1],
                ema1=tema_data['ema1'].iloc[-1],
                ema2=tema_data['ema2'].iloc[-1],
                ema3=tema_data['ema3'].iloc[-1],
                trend_slope=tema_data['slope'].iloc[-1],
                state=state,
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'tema': tema_data['tema'].tolist(),
                    'ema1': tema_data['ema1'].tolist(),
                    'slope': tema_data['slope'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'state': state.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Triple EMA: {e}")
            return self._get_default_result()
    
    def _calculate_tema(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        
        # Calculate first EMA
        df['ema1'] = data['close'].ewm(span=self.period).mean()
        
        # Calculate second EMA (EMA of EMA1)
        df['ema2'] = df['ema1'].ewm(span=self.period).mean()
        
        # Calculate third EMA (EMA of EMA2)
        df['ema3'] = df['ema2'].ewm(span=self.period).mean()
        
        # Calculate TEMA
        df['tema'] = 3 * df['ema1'] - 3 * df['ema2'] + df['ema3']
        
        # Calculate slope (trend direction)
        df['slope'] = df['tema'].diff(5)  # 5-period slope
        
        return df
    
    def _analyze_trend_state(self, tema_data: pd.DataFrame) -> TEMAState:
        current_slope = tema_data['slope'].iloc[-1]
        
        if pd.isna(current_slope):
            return TEMAState.SIDEWAYS
        
        # Calculate slope magnitude relative to price
        tema_value = tema_data['tema'].iloc[-1]
        relative_slope = abs(current_slope) / tema_value if tema_value != 0 else 0
        
        if current_slope > 0:
            if relative_slope > 0.01:  # Strong uptrend
                return TEMAState.STRONG_UPTREND
            else:
                return TEMAState.UPTREND
        elif current_slope < 0:
            if relative_slope > 0.01:  # Strong downtrend
                return TEMAState.STRONG_DOWNTREND
            else:
                return TEMAState.DOWNTREND
        else:
            return TEMAState.SIDEWAYS
    
    def _generate_signals(self, data: pd.DataFrame, tema_data: pd.DataFrame, 
                         state: TEMAState) -> Tuple[SignalType, float]:
        current_price = data['close'].iloc[-1]
        current_tema = tema_data['tema'].iloc[-1]
        
        if pd.isna(current_tema):
            return SignalType.NEUTRAL, 0.0
        
        # Price vs TEMA position
        price_above_tema = current_price > current_tema
        
        # Calculate distance for confidence
        distance = abs(current_price - current_tema) / current_price
        base_confidence = min(0.8, distance * 20)
        
        # State-based confidence adjustment
        confidence_map = {
            TEMAState.STRONG_UPTREND: 0.9,
            TEMAState.UPTREND: 0.7,
            TEMAState.SIDEWAYS: 0.3,
            TEMAState.DOWNTREND: 0.7,
            TEMAState.STRONG_DOWNTREND: 0.9
        }
        
        state_confidence = confidence_map.get(state, 0.5)
        final_confidence = (base_confidence + state_confidence) / 2
        
        # Generate signals
        if price_above_tema and state in [TEMAState.STRONG_UPTREND, TEMAState.UPTREND]:
            return SignalType.BUY, final_confidence
        elif not price_above_tema and state in [TEMAState.STRONG_DOWNTREND, TEMAState.DOWNTREND]:
            return SignalType.SELL, final_confidence
        elif price_above_tema:
            return SignalType.BUY, final_confidence * 0.6
        elif not price_above_tema:
            return SignalType.SELL, final_confidence * 0.6
        else:
            return SignalType.NEUTRAL, 0.3
    
    def _get_default_result(self) -> Dict:
        default_result = TEMAResult(0.0, 0.0, 0.0, 0.0, 0.0, TEMAState.SIDEWAYS, SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {'period': self.period}
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)