"""
Generic Moving Average Indicator with Multiple MA Types

Features:
- Support for SMA, EMA, WMA, and other MA types
- Configurable period and MA type
- Unified interface for all moving average calculations
- Trend analysis and signal generation
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class MAType(Enum):
    SMA = "sma"
    EMA = "ema"
    WMA = "wma"
    HULL = "hull"

class MAState(Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"

@dataclass
class MovingAverageResult:
    ma_value: float
    ma_type: MAType
    slope: float
    state: MAState
    signal: SignalType
    confidence: float

class MovingAverageIndicator(StandardIndicatorInterface):
    def __init__(self, period: int = 20, ma_type: str = "sma"):
        self.period = period
        self.ma_type = MAType(ma_type.lower())
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.period + 5:
                raise ValueError("Insufficient data")
            
            # Calculate moving average
            ma_data = self._calculate_ma(data)
            
            # Analyze state
            state = self._analyze_ma_state(data, ma_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, ma_data, state)
            
            latest_result = MovingAverageResult(
                ma_value=ma_data['ma'].iloc[-1],
                ma_type=self.ma_type,
                slope=ma_data['slope'].iloc[-1],
                state=state,
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'ma': ma_data['ma'].tolist(),
                    'slope': ma_data['slope'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'state': state.value,
                'ma_type': self.ma_type.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Moving Average: {e}")
            return self._get_default_result()
    
    def _calculate_ma(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        
        if self.ma_type == MAType.SMA:
            df['ma'] = data['close'].rolling(window=self.period).mean()
        elif self.ma_type == MAType.EMA:
            df['ma'] = data['close'].ewm(span=self.period).mean()
        elif self.ma_type == MAType.WMA:
            weights = np.arange(1, self.period + 1)
            weights_sum = weights.sum()
            ma_values = []
            for i in range(len(data)):
                if i < self.period - 1:
                    ma_values.append(np.nan)
                else:
                    price_window = data['close'].iloc[i - self.period + 1:i + 1].values
                    ma_value = np.sum(price_window * weights) / weights_sum
                    ma_values.append(ma_value)
            df['ma'] = ma_values
        elif self.ma_type == MAType.HULL:
            # Hull Moving Average calculation
            n = self.period
            wma_n = data['close'].rolling(window=n).apply(
                lambda x: np.sum(x * np.arange(1, n + 1)) / np.sum(np.arange(1, n + 1)), raw=False)
            wma_n2 = data['close'].rolling(window=n//2).apply(
                lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=False)
            raw_hma = 2 * wma_n2 - wma_n
            sqrt_n = int(np.sqrt(n))
            df['ma'] = raw_hma.rolling(window=sqrt_n).apply(
                lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=False)
        
        # Calculate slope
        df['slope'] = df['ma'].diff(periods=3)
        
        return df
    
    def _analyze_ma_state(self, data: pd.DataFrame, ma_data: pd.DataFrame) -> MAState:
        current_slope = ma_data['slope'].iloc[-1]
        
        if pd.isna(current_slope):
            return MAState.SIDEWAYS
        
        # Determine trend based on slope
        if current_slope > 0.001:
            return MAState.UPTREND
        elif current_slope < -0.001:
            return MAState.DOWNTREND
        else:
            return MAState.SIDEWAYS
    
    def _generate_signals(self, data: pd.DataFrame, ma_data: pd.DataFrame, 
                         state: MAState) -> Tuple[SignalType, float]:
        current_price = data['close'].iloc[-1]
        current_ma = ma_data['ma'].iloc[-1]
        
        if pd.isna(current_ma):
            return SignalType.NEUTRAL, 0.0
        
        # Price position relative to MA
        price_above_ma = current_price > current_ma
        distance = abs(current_price - current_ma) / current_price
        
        base_confidence = min(0.7, distance * 15)
        
        # State-based signals
        if state == MAState.UPTREND and price_above_ma:
            return SignalType.BUY, base_confidence
        elif state == MAState.DOWNTREND and not price_above_ma:
            return SignalType.SELL, base_confidence
        elif price_above_ma:
            return SignalType.BUY, base_confidence * 0.6
        elif not price_above_ma:
            return SignalType.SELL, base_confidence * 0.6
        else:
            return SignalType.NEUTRAL, 0.3
    
    def _get_default_result(self) -> Dict:
        default_result = MovingAverageResult(0.0, self.ma_type, 0.0, MAState.SIDEWAYS, 
                                           SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {'period': self.period, 'ma_type': self.ma_type.value}
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if param == 'ma_type':
                self.ma_type = MAType(value.lower())
            elif hasattr(self, param):
                setattr(self, param, value)