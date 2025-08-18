"""
SMA-EMA Comparison Indicator with Divergence Analysis

Features:
- Simultaneous SMA and EMA calculation
- Convergence/divergence analysis between SMA and EMA
- Cross-over detection and trend confirmation
- Relative strength measurement
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class SMAEMAState(Enum):
    EMA_ABOVE_SMA_DIVERGING = "ema_above_sma_diverging"
    EMA_ABOVE_SMA_CONVERGING = "ema_above_sma_converging"
    SMA_ABOVE_EMA_DIVERGING = "sma_above_ema_diverging"
    SMA_ABOVE_EMA_CONVERGING = "sma_above_ema_converging"
    CROSSOVER_BULLISH = "crossover_bullish"
    CROSSOVER_BEARISH = "crossover_bearish"
    NEUTRAL = "neutral"

@dataclass
class SMAEMAResult:
    sma_value: float
    ema_value: float
    difference: float
    difference_rate: float
    state: SMAEMAState
    signal: SignalType
    confidence: float

class SMAEMAIndicator(StandardIndicatorInterface):
    def __init__(self, period: int = 20):
        self.period = period
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.period + 10:
                raise ValueError("Insufficient data")
            
            # Calculate SMA and EMA
            sma_ema_data = self._calculate_sma_ema(data)
            
            # Analyze state
            state = self._analyze_sma_ema_state(sma_ema_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, sma_ema_data, state)
            
            latest_result = SMAEMAResult(
                sma_value=sma_ema_data['sma'].iloc[-1],
                ema_value=sma_ema_data['ema'].iloc[-1],
                difference=sma_ema_data['difference'].iloc[-1],
                difference_rate=sma_ema_data['difference_rate'].iloc[-1],
                state=state,
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'sma': sma_ema_data['sma'].tolist(),
                    'ema': sma_ema_data['ema'].tolist(),
                    'difference': sma_ema_data['difference'].tolist(),
                    'difference_rate': sma_ema_data['difference_rate'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'state': state.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating SMA-EMA: {e}")
            return self._get_default_result()
    
    def _calculate_sma_ema(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        
        # Calculate SMA and EMA
        df['sma'] = data['close'].rolling(window=self.period).mean()
        df['ema'] = data['close'].ewm(span=self.period).mean()
        
        # Calculate difference (EMA - SMA)
        df['difference'] = df['ema'] - df['sma']
        
        # Calculate rate of change in difference
        df['difference_rate'] = df['difference'].diff(periods=3)
        
        return df
    
    def _analyze_sma_ema_state(self, sma_ema_data: pd.DataFrame) -> SMAEMAState:
        current_diff = sma_ema_data['difference'].iloc[-1]
        current_diff_rate = sma_ema_data['difference_rate'].iloc[-1]
        
        if pd.isna(current_diff) or pd.isna(current_diff_rate):
            return SMAEMAState.NEUTRAL
        
        # Check for crossovers
        if len(sma_ema_data) > 1:
            prev_diff = sma_ema_data['difference'].iloc[-2]
            
            if prev_diff <= 0 and current_diff > 0:
                return SMAEMAState.CROSSOVER_BULLISH
            elif prev_diff >= 0 and current_diff < 0:
                return SMAEMAState.CROSSOVER_BEARISH
        
        # Analyze divergence/convergence
        if current_diff > 0:  # EMA above SMA
            if current_diff_rate > 0:
                return SMAEMAState.EMA_ABOVE_SMA_DIVERGING
            else:
                return SMAEMAState.EMA_ABOVE_SMA_CONVERGING
        elif current_diff < 0:  # SMA above EMA
            if current_diff_rate < 0:
                return SMAEMAState.SMA_ABOVE_EMA_DIVERGING
            else:
                return SMAEMAState.SMA_ABOVE_EMA_CONVERGING
        else:
            return SMAEMAState.NEUTRAL
    
    def _generate_signals(self, data: pd.DataFrame, sma_ema_data: pd.DataFrame, 
                         state: SMAEMAState) -> Tuple[SignalType, float]:
        current_price = data['close'].iloc[-1]
        sma_value = sma_ema_data['sma'].iloc[-1]
        ema_value = sma_ema_data['ema'].iloc[-1]
        difference = abs(sma_ema_data['difference'].iloc[-1])
        
        if pd.isna(sma_value) or pd.isna(ema_value):
            return SignalType.NEUTRAL, 0.0
        
        # Base confidence on difference magnitude
        relative_diff = difference / current_price if current_price != 0 else 0
        base_confidence = min(0.8, relative_diff * 50)
        
        # State-based signals
        if state == SMAEMAState.CROSSOVER_BULLISH:
            return SignalType.BUY, min(0.85, base_confidence + 0.2)
        elif state == SMAEMAState.CROSSOVER_BEARISH:
            return SignalType.SELL, min(0.85, base_confidence + 0.2)
        elif state == SMAEMAState.EMA_ABOVE_SMA_DIVERGING:
            return SignalType.BUY, base_confidence * 0.8
        elif state == SMAEMAState.SMA_ABOVE_EMA_DIVERGING:
            return SignalType.SELL, base_confidence * 0.8
        elif state in [SMAEMAState.EMA_ABOVE_SMA_CONVERGING]:
            return SignalType.BUY, base_confidence * 0.5
        elif state in [SMAEMAState.SMA_ABOVE_EMA_CONVERGING]:
            return SignalType.SELL, base_confidence * 0.5
        else:
            return SignalType.NEUTRAL, 0.3
    
    def _get_default_result(self) -> Dict:
        default_result = SMAEMAResult(0.0, 0.0, 0.0, 0.0, SMAEMAState.NEUTRAL, 
                                    SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {'period': self.period}
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)