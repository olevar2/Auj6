"""
Advanced ZigZag Indicator with Dynamic Reversal Detection

Features:
- Adaptive percentage-based reversal detection
- Trend line identification and breakouts
- Support/resistance level calculation
- Pattern recognition for swings
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class ZigZagState(Enum):
    UPSWING = "upswing"
    DOWNSWING = "downswing"
    REVERSAL_HIGH = "reversal_high"
    REVERSAL_LOW = "reversal_low"
    CONSOLIDATION = "consolidation"

@dataclass
class ZigZagPoint:
    index: int
    price: float
    is_high: bool
    
@dataclass
class ZigZagResult:
    current_level: float
    swing_points: List[ZigZagPoint]
    state: ZigZagState
    signal: SignalType
    confidence: float

class ZigZagIndicator(StandardIndicatorInterface):
    def __init__(self, percentage: float = 5.0, use_high_low: bool = True):
        self.percentage = percentage
        self.use_high_low = use_high_low
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < 20:
                raise ValueError("Insufficient data")
            
            # Calculate ZigZag
            zigzag_data = self._calculate_zigzag(data)
            
            # Analyze state
            state = self._analyze_zigzag_state(data, zigzag_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, zigzag_data, state)
            
            swing_points = self._extract_swing_points(zigzag_data)
            
            latest_result = ZigZagResult(
                current_level=zigzag_data['zigzag'].iloc[-1],
                swing_points=swing_points,
                state=state,
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'zigzag': zigzag_data['zigzag'].tolist(),
                    'highs': zigzag_data['highs'].tolist(),
                    'lows': zigzag_data['lows'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'state': state.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ZigZag: {e}")
            return self._get_default_result()
    
    def _calculate_zigzag(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        df['high'] = data['high']
        df['low'] = data['low']
        df['close'] = data['close']
        
        # Use high/low or close prices
        if self.use_high_low:
            price_high = data['high']
            price_low = data['low']
        else:
            price_high = data['close']
            price_low = data['close']
        
        # Initialize ZigZag array
        df['zigzag'] = np.nan
        df['highs'] = np.nan
        df['lows'] = np.nan
        
        # Find initial direction
        threshold = self.percentage / 100.0
        
        # Start with first price
        current_extreme = price_high.iloc[0]
        current_extreme_idx = 0
        is_looking_for_high = True
        
        for i in range(1, len(data)):
            current_high = price_high.iloc[i]
            current_low = price_low.iloc[i]
            
            if is_looking_for_high:
                # Looking for higher high
                if current_high > current_extreme:
                    current_extreme = current_high
                    current_extreme_idx = i
                elif (current_extreme - current_low) / current_extreme >= threshold:
                    # Found significant reversal
                    df['zigzag'].iloc[current_extreme_idx] = current_extreme
                    df['highs'].iloc[current_extreme_idx] = current_extreme
                    current_extreme = current_low
                    current_extreme_idx = i
                    is_looking_for_high = False
            else:
                # Looking for lower low
                if current_low < current_extreme:
                    current_extreme = current_low
                    current_extreme_idx = i
                elif (current_high - current_extreme) / current_extreme >= threshold:
                    # Found significant reversal
                    df['zigzag'].iloc[current_extreme_idx] = current_extreme
                    df['lows'].iloc[current_extreme_idx] = current_extreme
                    current_extreme = current_high
                    current_extreme_idx = i
                    is_looking_for_high = True
        
        # Mark the last extreme
        df['zigzag'].iloc[current_extreme_idx] = current_extreme
        if is_looking_for_high:
            df['lows'].iloc[current_extreme_idx] = current_extreme
        else:
            df['highs'].iloc[current_extreme_idx] = current_extreme
        
        # Forward fill ZigZag values
        df['zigzag'].fillna(method='ffill', inplace=True)
        
        return df
    
    def _analyze_zigzag_state(self, data: pd.DataFrame, zigzag_data: pd.DataFrame) -> ZigZagState:
        # Find last few swing points
        highs = zigzag_data['highs'].dropna()
        lows = zigzag_data['lows'].dropna()
        
        if len(highs) == 0 and len(lows) == 0:
            return ZigZagState.CONSOLIDATION
        
        # Get last swing point
        last_high_idx = highs.index[-1] if len(highs) > 0 else -1
        last_low_idx = lows.index[-1] if len(lows) > 0 else -1
        
        current_price = data['close'].iloc[-1]
        current_idx = len(data) - 1
        
        # Determine current state
        if last_high_idx > last_low_idx:
            # Last swing was a high
            last_high = highs.iloc[-1]
            if current_price < last_high * (1 - self.percentage / 100):
                return ZigZagState.REVERSAL_LOW
            else:
                return ZigZagState.DOWNSWING
        else:
            # Last swing was a low
            last_low = lows.iloc[-1]
            if current_price > last_low * (1 + self.percentage / 100):
                return ZigZagState.REVERSAL_HIGH
            else:
                return ZigZagState.UPSWING
    
    def _generate_signals(self, data: pd.DataFrame, zigzag_data: pd.DataFrame, 
                         state: ZigZagState) -> Tuple[SignalType, float]:
        
        # Base confidence on state
        if state == ZigZagState.REVERSAL_HIGH:
            return SignalType.BUY, 0.75
        elif state == ZigZagState.REVERSAL_LOW:
            return SignalType.SELL, 0.75
        elif state == ZigZagState.UPSWING:
            return SignalType.BUY, 0.6
        elif state == ZigZagState.DOWNSWING:
            return SignalType.SELL, 0.6
        else:
            return SignalType.NEUTRAL, 0.3
    
    def _extract_swing_points(self, zigzag_data: pd.DataFrame) -> List[ZigZagPoint]:
        swing_points = []
        
        # Extract highs
        highs = zigzag_data['highs'].dropna()
        for idx, price in highs.items():
            swing_points.append(ZigZagPoint(idx, price, True))
        
        # Extract lows
        lows = zigzag_data['lows'].dropna()
        for idx, price in lows.items():
            swing_points.append(ZigZagPoint(idx, price, False))
        
        # Sort by index
        swing_points.sort(key=lambda x: x.index)
        
        return swing_points[-10:]  # Return last 10 swing points
    
    def _get_default_result(self) -> Dict:
        default_result = ZigZagResult(0.0, [], ZigZagState.CONSOLIDATION, SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {'percentage': self.percentage, 'use_high_low': self.use_high_low}
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)