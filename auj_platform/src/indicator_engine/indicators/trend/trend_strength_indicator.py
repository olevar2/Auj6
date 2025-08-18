"""
Advanced Trend Strength Indicator with Multi-Factor Analysis

Features:
- Combination of multiple trend strength measures
- ADX, slope analysis, and momentum integration
- Volatility-adjusted trend strength
- Confidence scoring based on multiple factors
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class TrendStrengthLevel(Enum):
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"

@dataclass
class TrendStrengthResult:
    strength_score: float
    direction: int  # 1 for up, -1 for down, 0 for neutral
    strength_level: TrendStrengthLevel
    adx_component: float
    slope_component: float
    momentum_component: float
    signal: SignalType
    confidence: float

class TrendStrengthIndicator(StandardIndicatorInterface):
    def __init__(self, period: int = 14):
        self.period = period
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < self.period * 2:
                raise ValueError("Insufficient data")
            
            # Calculate trend strength components
            strength_data = self._calculate_trend_strength(data)
            
            # Determine strength level
            strength_level = self._determine_strength_level(strength_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(strength_data, strength_level)
            
            latest_result = TrendStrengthResult(
                strength_score=strength_data['strength_score'].iloc[-1],
                direction=strength_data['direction'].iloc[-1],
                strength_level=strength_level,
                adx_component=strength_data['adx_component'].iloc[-1],
                slope_component=strength_data['slope_component'].iloc[-1],
                momentum_component=strength_data['momentum_component'].iloc[-1],
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'strength_score': strength_data['strength_score'].tolist(),
                    'direction': strength_data['direction'].tolist(),
                    'adx_component': strength_data['adx_component'].tolist(),
                    'slope_component': strength_data['slope_component'].tolist(),
                    'momentum_component': strength_data['momentum_component'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'strength_level': strength_level.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Trend Strength: {e}")
            return self._get_default_result()
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        
        # Component 1: ADX-like calculation
        df['adx_component'] = self._calculate_adx_component(data)
        
        # Component 2: Slope analysis
        df['slope_component'] = self._calculate_slope_component(data)
        
        # Component 3: Momentum component
        df['momentum_component'] = self._calculate_momentum_component(data)
        
        # Combine components with weights
        weights = {'adx': 0.4, 'slope': 0.35, 'momentum': 0.25}
        
        df['strength_score'] = (
            df['adx_component'] * weights['adx'] +
            df['slope_component'] * weights['slope'] +
            df['momentum_component'] * weights['momentum']
        )
        
        # Determine trend direction
        df['direction'] = self._determine_trend_direction(data, df)
        
        return df
    
    def _calculate_adx_component(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ADX-like trend strength component"""
        # True Range calculation
        prev_close = data['close'].shift(1)
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - prev_close)
        tr3 = abs(data['low'] - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional movements
        up_move = data['high'] - data['high'].shift(1)
        down_move = data['low'].shift(1) - data['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth the values
        alpha = 1.0 / self.period
        atr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
        plus_di = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean() / atr * 100
        minus_di = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean() / atr * 100
        
        # Calculate DX and ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        
        # Normalize to 0-1 scale
        return adx / 100.0
    
    def _calculate_slope_component(self, data: pd.DataFrame) -> pd.Series:
        """Calculate slope-based trend strength"""
        # Use EMA for smoothing
        ema = data['close'].ewm(span=self.period).mean()
        
        # Calculate slope over different periods
        slope_short = ema.diff(3)
        slope_medium = ema.diff(7)
        slope_long = ema.diff(14)
        
        # Normalize slopes relative to price
        normalized_slopes = []
        for slope in [slope_short, slope_medium, slope_long]:
            normalized = abs(slope) / data['close']
            normalized_slopes.append(normalized)
        
        # Average the normalized slopes
        avg_slope = pd.concat(normalized_slopes, axis=1).mean(axis=1)
        
        # Scale to 0-1 range
        return np.clip(avg_slope * 100, 0, 1)
    
    def _calculate_momentum_component(self, data: pd.DataFrame) -> pd.Series:
        """Calculate momentum-based trend strength"""
        # Multiple timeframe momentum
        mom_3 = data['close'].pct_change(3)
        mom_7 = data['close'].pct_change(7)
        mom_14 = data['close'].pct_change(14)
        
        # Average absolute momentum
        avg_momentum = pd.concat([abs(mom_3), abs(mom_7), abs(mom_14)], axis=1).mean(axis=1)
        
        # Scale to 0-1 range
        return np.clip(avg_momentum * 20, 0, 1)
    
    def _determine_trend_direction(self, data: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        """Determine overall trend direction"""
        # Use EMA slope as primary direction indicator
        ema = data['close'].ewm(span=self.period).mean()
        slope = ema.diff(5)
        
        # Normalize and create direction signal
        direction = np.where(slope > 0, 1, np.where(slope < 0, -1, 0))
        
        return pd.Series(direction, index=data.index)
    
    def _determine_strength_level(self, strength_data: pd.DataFrame) -> TrendStrengthLevel:
        """Determine strength level based on score"""
        strength_score = strength_data['strength_score'].iloc[-1]
        
        if pd.isna(strength_score):
            return TrendStrengthLevel.VERY_WEAK
        
        if strength_score >= 0.8:
            return TrendStrengthLevel.VERY_STRONG
        elif strength_score >= 0.6:
            return TrendStrengthLevel.STRONG
        elif strength_score >= 0.4:
            return TrendStrengthLevel.MODERATE
        elif strength_score >= 0.2:
            return TrendStrengthLevel.WEAK
        else:
            return TrendStrengthLevel.VERY_WEAK
    
    def _generate_signals(self, strength_data: pd.DataFrame, 
                         strength_level: TrendStrengthLevel) -> Tuple[SignalType, float]:
        strength_score = strength_data['strength_score'].iloc[-1]
        direction = strength_data['direction'].iloc[-1]
        
        if pd.isna(strength_score) or pd.isna(direction):
            return SignalType.NEUTRAL, 0.0
        
        # Base confidence on strength score
        base_confidence = strength_score
        
        # Adjust confidence based on strength level
        level_multipliers = {
            TrendStrengthLevel.VERY_STRONG: 1.0,
            TrendStrengthLevel.STRONG: 0.8,
            TrendStrengthLevel.MODERATE: 0.6,
            TrendStrengthLevel.WEAK: 0.4,
            TrendStrengthLevel.VERY_WEAK: 0.2
        }
        
        final_confidence = base_confidence * level_multipliers.get(strength_level, 0.5)
        
        # Generate signals based on direction and strength
        if direction > 0 and strength_level in [TrendStrengthLevel.STRONG, TrendStrengthLevel.VERY_STRONG]:
            return SignalType.BUY, final_confidence
        elif direction < 0 and strength_level in [TrendStrengthLevel.STRONG, TrendStrengthLevel.VERY_STRONG]:
            return SignalType.SELL, final_confidence
        elif direction > 0:
            return SignalType.BUY, final_confidence * 0.7
        elif direction < 0:
            return SignalType.SELL, final_confidence * 0.7
        else:
            return SignalType.NEUTRAL, 0.3
    
    def _get_default_result(self) -> Dict:
        default_result = TrendStrengthResult(0.0, 0, TrendStrengthLevel.VERY_WEAK, 
                                           0.0, 0.0, 0.0, SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {'period': self.period}
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)