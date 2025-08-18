"""
Advanced Trend Following System with Multi-Component Analysis

Features:
- Composite trend following signals
- Multiple timeframe analysis
- Risk-adjusted position sizing
- Dynamic stop-loss levels
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class TrendFollowingState(Enum):
    STRONG_BULLISH_TREND = "strong_bullish_trend"
    BULLISH_TREND = "bullish_trend"
    WEAK_BULLISH_TREND = "weak_bullish_trend"
    SIDEWAYS = "sideways"
    WEAK_BEARISH_TREND = "weak_bearish_trend"
    BEARISH_TREND = "bearish_trend"
    STRONG_BEARISH_TREND = "strong_bearish_trend"

@dataclass
class TrendFollowingResult:
    trend_score: float
    trend_strength: float
    trend_direction: int
    stop_loss_level: float
    position_size_factor: float
    state: TrendFollowingState
    signal: SignalType
    confidence: float

class TrendFollowingSystemIndicator(StandardIndicatorInterface):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.slow_period * 2:
                raise ValueError("Insufficient data")
            
            # Calculate trend following components
            trend_data = self._calculate_trend_following(data)
            
            # Determine state
            state = self._determine_trend_state(trend_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, trend_data, state)
            
            latest_result = TrendFollowingResult(
                trend_score=trend_data['trend_score'].iloc[-1],
                trend_strength=trend_data['trend_strength'].iloc[-1],
                trend_direction=trend_data['trend_direction'].iloc[-1],
                stop_loss_level=trend_data['stop_loss_level'].iloc[-1],
                position_size_factor=trend_data['position_size_factor'].iloc[-1],
                state=state,
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'trend_score': trend_data['trend_score'].tolist(),
                    'trend_strength': trend_data['trend_strength'].tolist(),
                    'stop_loss_level': trend_data['stop_loss_level'].tolist(),
                    'position_size_factor': trend_data['position_size_factor'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'state': state.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Trend Following System: {e}")
            return self._get_default_result()
    
    def _calculate_trend_following(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        
        # MACD-like calculation for trend
        ema_fast = data['close'].ewm(span=self.fast_period).mean()
        ema_slow = data['close'].ewm(span=self.slow_period).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal_period).mean()
        histogram = macd - signal_line
        
        # Trend Score (normalized MACD)
        df['trend_score'] = macd / data['close']
        
        # Trend Strength (based on histogram and volatility)
        volatility = data['close'].rolling(20).std()
        df['trend_strength'] = abs(histogram) / volatility
        
        # Trend Direction
        df['trend_direction'] = np.sign(macd)
        
        # Dynamic Stop Loss Level
        df['stop_loss_level'] = self._calculate_stop_loss(data, macd)
        
        # Position Size Factor (risk-adjusted)
        df['position_size_factor'] = self._calculate_position_size(data, df['trend_strength'])
        
        return df
    
    def _calculate_stop_loss(self, data: pd.DataFrame, macd: pd.Series) -> pd.Series:
        """Calculate dynamic stop-loss levels"""
        atr_period = 14
        atr_multiplier = 2.0
        
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift(1))
        low_close = abs(data['low'] - data['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()
        
        # Dynamic stop based on trend direction
        stop_distance = atr * atr_multiplier
        
        stop_levels = []
        for i in range(len(data)):
            if i < atr_period:
                stop_levels.append(np.nan)
            else:
                current_price = data['close'].iloc[i]
                trend_dir = np.sign(macd.iloc[i]) if not pd.isna(macd.iloc[i]) else 0
                
                if trend_dir > 0:  # Uptrend
                    stop_level = current_price - stop_distance.iloc[i]
                elif trend_dir < 0:  # Downtrend
                    stop_level = current_price + stop_distance.iloc[i]
                else:  # Neutral
                    stop_level = current_price
                
                stop_levels.append(stop_level)
        
        return pd.Series(stop_levels, index=data.index)
    
    def _calculate_position_size(self, data: pd.DataFrame, trend_strength: pd.Series) -> pd.Series:
        """Calculate risk-adjusted position sizing factor"""
        # Base position size on trend strength and volatility
        volatility = data['close'].rolling(20).std() / data['close'].rolling(20).mean()
        
        # Normalize trend strength
        normalized_strength = np.clip(trend_strength, 0, 2)
        
        # Calculate position size factor (0 to 1)
        base_size = 0.5  # Base position size
        volatility_adjustment = np.clip(1 / (1 + volatility * 10), 0.2, 1.0)
        strength_adjustment = normalized_strength / 2
        
        position_factor = base_size * volatility_adjustment * (1 + strength_adjustment)
        
        return np.clip(position_factor, 0.1, 1.0)
    
    def _determine_trend_state(self, trend_data: pd.DataFrame) -> TrendFollowingState:
        """Determine overall trend following state"""
        trend_score = trend_data['trend_score'].iloc[-1]
        trend_strength = trend_data['trend_strength'].iloc[-1]
        
        if pd.isna(trend_score) or pd.isna(trend_strength):
            return TrendFollowingState.SIDEWAYS
        
        # Classify based on score and strength
        if trend_score > 0:  # Bullish
            if trend_strength > 1.5:
                return TrendFollowingState.STRONG_BULLISH_TREND
            elif trend_strength > 0.8:
                return TrendFollowingState.BULLISH_TREND
            else:
                return TrendFollowingState.WEAK_BULLISH_TREND
        elif trend_score < 0:  # Bearish
            if trend_strength > 1.5:
                return TrendFollowingState.STRONG_BEARISH_TREND
            elif trend_strength > 0.8:
                return TrendFollowingState.BEARISH_TREND
            else:
                return TrendFollowingState.WEAK_BEARISH_TREND
        else:
            return TrendFollowingState.SIDEWAYS
    
    def _generate_signals(self, data: pd.DataFrame, trend_data: pd.DataFrame, 
                         state: TrendFollowingState) -> Tuple[SignalType, float]:
        trend_strength = trend_data['trend_strength'].iloc[-1]
        position_size_factor = trend_data['position_size_factor'].iloc[-1]
        
        if pd.isna(trend_strength):
            return SignalType.NEUTRAL, 0.0
        
        # Base confidence on trend strength and position sizing
        base_confidence = min(0.9, trend_strength * position_size_factor)
        
        # State-based signals
        confidence_map = {
            TrendFollowingState.STRONG_BULLISH_TREND: 0.9,
            TrendFollowingState.BULLISH_TREND: 0.75,
            TrendFollowingState.WEAK_BULLISH_TREND: 0.5,
            TrendFollowingState.SIDEWAYS: 0.2,
            TrendFollowingState.WEAK_BEARISH_TREND: 0.5,
            TrendFollowingState.BEARISH_TREND: 0.75,
            TrendFollowingState.STRONG_BEARISH_TREND: 0.9
        }
        
        state_confidence = confidence_map.get(state, 0.3)
        final_confidence = (base_confidence + state_confidence) / 2
        
        if state in [TrendFollowingState.STRONG_BULLISH_TREND, 
                    TrendFollowingState.BULLISH_TREND,
                    TrendFollowingState.WEAK_BULLISH_TREND]:
            return SignalType.BUY, final_confidence
        elif state in [TrendFollowingState.STRONG_BEARISH_TREND,
                      TrendFollowingState.BEARISH_TREND,
                      TrendFollowingState.WEAK_BEARISH_TREND]:
            return SignalType.SELL, final_confidence
        else:
            return SignalType.NEUTRAL, final_confidence
    
    def _get_default_result(self) -> Dict:
        default_result = TrendFollowingResult(0.0, 0.0, 0, 0.0, 0.5, 
                                            TrendFollowingState.SIDEWAYS, SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period
        }
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)