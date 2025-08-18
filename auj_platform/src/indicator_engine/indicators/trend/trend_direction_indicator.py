"""
Advanced Trend Direction Indicator with Multi-Signal Analysis

Features:
- Multiple trend direction algorithms
- Consensus-based trend determination
- Strength and persistence analysis
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

class TrendDirection(Enum):
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"  
    DOWN = "down"
    STRONG_DOWN = "strong_down"

@dataclass
class TrendDirectionResult:
    direction: TrendDirection
    direction_score: float
    ma_consensus: float
    slope_consensus: float
    momentum_consensus: float
    persistence_score: float
    signal: SignalType
    confidence: float

class TrendDirectionIndicator(StandardIndicatorInterface):
    def __init__(self, short_period: int = 10, medium_period: int = 20, long_period: int = 50):
        self.short_period = short_period
        self.medium_period = medium_period  
        self.long_period = long_period
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.long_period + 10:
                raise ValueError("Insufficient data")
            
            # Calculate trend direction components
            direction_data = self._calculate_trend_direction(data)
            
            # Determine overall direction
            direction = self._determine_overall_direction(direction_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(direction_data, direction)
            
            latest_result = TrendDirectionResult(
                direction=direction,
                direction_score=direction_data['direction_score'].iloc[-1],
                ma_consensus=direction_data['ma_consensus'].iloc[-1],
                slope_consensus=direction_data['slope_consensus'].iloc[-1],
                momentum_consensus=direction_data['momentum_consensus'].iloc[-1],
                persistence_score=direction_data['persistence_score'].iloc[-1],
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'direction_score': direction_data['direction_score'].tolist(),
                    'ma_consensus': direction_data['ma_consensus'].tolist(),
                    'slope_consensus': direction_data['slope_consensus'].tolist(),
                    'momentum_consensus': direction_data['momentum_consensus'].tolist(),
                    'persistence_score': direction_data['persistence_score'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'direction': direction.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Trend Direction: {e}")
            return self._get_default_result()
    
    def _calculate_trend_direction(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        
        # Moving Average Consensus
        df['ma_consensus'] = self._calculate_ma_consensus(data)
        
        # Slope Consensus
        df['slope_consensus'] = self._calculate_slope_consensus(data)
        
        # Momentum Consensus
        df['momentum_consensus'] = self._calculate_momentum_consensus(data)
        
        # Persistence Score
        df['persistence_score'] = self._calculate_persistence_score(data)
        
        # Overall Direction Score
        weights = {'ma': 0.3, 'slope': 0.3, 'momentum': 0.25, 'persistence': 0.15}
        df['direction_score'] = (
            df['ma_consensus'] * weights['ma'] +
            df['slope_consensus'] * weights['slope'] +
            df['momentum_consensus'] * weights['momentum'] +
            df['persistence_score'] * weights['persistence']
        )
        
        return df
    
    def _calculate_ma_consensus(self, data: pd.DataFrame) -> pd.Series:
        """Calculate moving average consensus"""
        # Calculate multiple MAs
        ma_short = data['close'].rolling(self.short_period).mean()
        ma_medium = data['close'].rolling(self.medium_period).mean()
        ma_long = data['close'].rolling(self.long_period).mean()
        
        # Price vs MA signals
        price = data['close']
        signals = []
        
        # Signal strength based on position
        signals.append(np.where(price > ma_short, 1, -1))
        signals.append(np.where(price > ma_medium, 1, -1))
        signals.append(np.where(price > ma_long, 1, -1))
        
        # MA alignment signals
        signals.append(np.where(ma_short > ma_medium, 1, -1))
        signals.append(np.where(ma_medium > ma_long, 1, -1))
        
        # Calculate consensus (-1 to 1)
        consensus = pd.DataFrame(signals).T.mean(axis=1)
        return consensus
    
    def _calculate_slope_consensus(self, data: pd.DataFrame) -> pd.Series:
        """Calculate slope consensus across timeframes"""
        slopes = []
        
        # Multiple EMA slopes
        for period in [self.short_period, self.medium_period, self.long_period]:
            ema = data['close'].ewm(span=period).mean()
            slope = ema.diff(5)  # 5-period slope
            normalized_slope = slope / data['close']  # Normalize
            # Convert to -1 to 1 scale
            slopes.append(np.tanh(normalized_slope * 100))
        
        # Average slopes
        consensus = pd.DataFrame(slopes).T.mean(axis=1)
        return consensus
    
    def _calculate_momentum_consensus(self, data: pd.DataFrame) -> pd.Series:
        """Calculate momentum consensus"""
        momentums = []
        
        # Multiple timeframe momentum
        for period in [3, 7, 14, 21]:
            mom = data['close'].pct_change(period)
            # Convert to -1 to 1 scale
            momentums.append(np.tanh(mom * 50))
        
        # Average momentum
        consensus = pd.DataFrame(momentums).T.mean(axis=1)
        return consensus
    
    def _calculate_persistence_score(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend persistence"""
        # Use price vs medium MA for persistence
        ma_medium = data['close'].rolling(self.medium_period).mean()
        above_ma = (data['close'] > ma_medium).astype(int)
        
        # Rolling persistence (consistency over time)
        persistence_window = 10
        persistence = above_ma.rolling(persistence_window).mean()
        
        # Convert to -1 to 1 scale (0.5 becomes 0)
        persistence_normalized = (persistence - 0.5) * 2
        
        return persistence_normalized
    
    def _determine_overall_direction(self, direction_data: pd.DataFrame) -> TrendDirection:
        """Determine overall trend direction"""
        direction_score = direction_data['direction_score'].iloc[-1]
        
        if pd.isna(direction_score):
            return TrendDirection.NEUTRAL
        
        if direction_score >= 0.6:
            return TrendDirection.STRONG_UP
        elif direction_score >= 0.2:
            return TrendDirection.UP
        elif direction_score <= -0.6:
            return TrendDirection.STRONG_DOWN
        elif direction_score <= -0.2:
            return TrendDirection.DOWN
        else:
            return TrendDirection.NEUTRAL
    
    def _generate_signals(self, direction_data: pd.DataFrame, 
                         direction: TrendDirection) -> Tuple[SignalType, float]:
        direction_score = direction_data['direction_score'].iloc[-1]
        
        if pd.isna(direction_score):
            return SignalType.NEUTRAL, 0.0
        
        # Base confidence on score magnitude
        base_confidence = abs(direction_score)
        
        # Direction-based signals
        if direction == TrendDirection.STRONG_UP:
            return SignalType.BUY, min(0.9, base_confidence + 0.1)
        elif direction == TrendDirection.UP:
            return SignalType.BUY, base_confidence * 0.8
        elif direction == TrendDirection.STRONG_DOWN:
            return SignalType.SELL, min(0.9, base_confidence + 0.1)
        elif direction == TrendDirection.DOWN:
            return SignalType.SELL, base_confidence * 0.8
        else:
            return SignalType.NEUTRAL, 0.3
    
    def _get_default_result(self) -> Dict:
        default_result = TrendDirectionResult(TrendDirection.NEUTRAL, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {
            'short_period': self.short_period,
            'medium_period': self.medium_period,
            'long_period': self.long_period
        }
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)