"""
Advanced Renko Trend Indicator with Brick Size Optimization

Features:
- Dynamic brick size calculation based on ATR
- Trend identification through brick color sequences
- Support/resistance level detection
- Trend strength measurement and confirmation
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class RenkoTrendState(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    CONSOLIDATION = "consolidation"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    REVERSAL_UP = "reversal_up"
    REVERSAL_DOWN = "reversal_down"

@dataclass
class RenkoBrick:
    open_price: float
    close_price: float
    is_up: bool
    timestamp: int

@dataclass
class RenkoTrendResult:
    current_brick: RenkoBrick
    brick_size: float
    trend_state: RenkoTrendState
    consecutive_bricks: int
    support_level: float
    resistance_level: float
    trend_strength: float
    signal: SignalType
    confidence: float

class RenkoTrendIndicator(StandardIndicatorInterface):
    def __init__(self, brick_size: float = None, atr_period: int = 14, atr_multiplier: float = 1.0):
        self.brick_size = brick_size  # If None, will calculate dynamically
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if len(data) < max(self.atr_period, 20):
                raise ValueError("Insufficient data")
            
            # Calculate dynamic brick size if not specified
            if self.brick_size is None:
                brick_size = self._calculate_dynamic_brick_size(data)
            else:
                brick_size = self.brick_size
            
            # Generate Renko bricks
            renko_data = self._generate_renko_bricks(data, brick_size)
            
            # Analyze trend state
            trend_state = self._analyze_renko_trend(renko_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, renko_data, trend_state)
            
            if not renko_data['bricks']:
                raise ValueError("No Renko bricks generated")
            
            latest_result = RenkoTrendResult(
                current_brick=renko_data['bricks'][-1],
                brick_size=brick_size,
                trend_state=trend_state,
                consecutive_bricks=renko_data['consecutive_count'],
                support_level=renko_data['support_level'],
                resistance_level=renko_data['resistance_level'],
                trend_strength=renko_data['trend_strength'],
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'brick_prices': [brick.close_price for brick in renko_data['bricks']],
                    'brick_colors': [brick.is_up for brick in renko_data['bricks']],
                    'support_levels': [renko_data['support_level']] * len(renko_data['bricks']),
                    'resistance_levels': [renko_data['resistance_level']] * len(renko_data['bricks'])
                },
                'signal': signal,
                'confidence': confidence,
                'trend_state': trend_state.value,
                'brick_size': brick_size
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Renko Trend: {e}")
            return self._get_default_result()
    
    def _calculate_dynamic_brick_size(self, data: pd.DataFrame) -> float:
        """Calculate dynamic brick size based on ATR"""
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift(1))
        low_close = abs(data['low'] - data['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean().iloc[-1]
        
        # Use ATR multiplier for brick size
        brick_size = atr * self.atr_multiplier
        
        # Ensure minimum brick size relative to price
        min_brick_size = data['close'].iloc[-1] * 0.001  # 0.1% of price
        
        return max(brick_size, min_brick_size)
    
    def _generate_renko_bricks(self, data: pd.DataFrame, brick_size: float) -> Dict:
        """Generate Renko bricks from price data"""
        bricks = []
        
        # Start with first price
        current_brick_base = data['close'].iloc[0]
        
        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            
            # Check for up brick
            while current_price >= current_brick_base + brick_size:
                # Create up brick
                brick = RenkoBrick(
                    open_price=current_brick_base,
                    close_price=current_brick_base + brick_size,
                    is_up=True,
                    timestamp=i
                )
                bricks.append(brick)
                current_brick_base += brick_size
            
            # Check for down brick
            while current_price <= current_brick_base - brick_size:
                # Create down brick
                brick = RenkoBrick(
                    open_price=current_brick_base,
                    close_price=current_brick_base - brick_size,
                    is_up=False,
                    timestamp=i
                )
                bricks.append(brick)
                current_brick_base -= brick_size
        
        # Analyze brick patterns
        consecutive_count = self._count_consecutive_bricks(bricks)
        support_level, resistance_level = self._find_support_resistance(bricks)
        trend_strength = self._calculate_trend_strength(bricks)
        
        return {
            'bricks': bricks,
            'consecutive_count': consecutive_count,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'trend_strength': trend_strength
        }
    
    def _count_consecutive_bricks(self, bricks: List[RenkoBrick]) -> int:
        """Count consecutive bricks of the same color"""
        if len(bricks) < 2:
            return 1
        
        last_color = bricks[-1].is_up
        count = 1
        
        for i in range(len(bricks) - 2, -1, -1):
            if bricks[i].is_up == last_color:
                count += 1
            else:
                break
        
        return count
    
    def _find_support_resistance(self, bricks: List[RenkoBrick]) -> Tuple[float, float]:
        """Find support and resistance levels from brick patterns"""
        if len(bricks) < 10:
            return 0.0, 0.0
        
        # Take last 20 bricks for analysis
        recent_bricks = bricks[-20:]
        prices = [brick.close_price for brick in recent_bricks]
        
        # Simple support/resistance based on price clusters
        sorted_prices = sorted(prices)
        support_level = np.percentile(sorted_prices, 25)
        resistance_level = np.percentile(sorted_prices, 75)
        
        return support_level, resistance_level
    
    def _calculate_trend_strength(self, bricks: List[RenkoBrick]) -> float:
        """Calculate trend strength based on brick patterns"""
        if len(bricks) < 5:
            return 0.5
        
        recent_bricks = bricks[-10:] if len(bricks) >= 10 else bricks
        
        # Count up vs down bricks
        up_count = sum(1 for brick in recent_bricks if brick.is_up)
        down_count = len(recent_bricks) - up_count
        
        # Calculate strength (0 = strong down, 1 = strong up)
        if len(recent_bricks) == 0:
            return 0.5
        
        strength = up_count / len(recent_bricks)
        
        # Adjust for consecutive brick patterns
        consecutive = self._count_consecutive_bricks(bricks)
        if consecutive >= 3:
            # Boost strength for strong trends
            if recent_bricks[-1].is_up:
                strength = min(1.0, strength + 0.2)
            else:
                strength = max(0.0, strength - 0.2)
        
        return strength
    
    def _analyze_renko_trend(self, renko_data: Dict) -> RenkoTrendState:
        """Analyze Renko trend state"""
        bricks = renko_data['bricks']
        consecutive_count = renko_data['consecutive_count']
        trend_strength = renko_data['trend_strength']
        
        if len(bricks) < 2:
            return RenkoTrendState.CONSOLIDATION
        
        current_brick = bricks[-1]
        prev_brick = bricks[-2] if len(bricks) > 1 else None
        
        # Check for reversals
        if prev_brick and current_brick.is_up != prev_brick.is_up:
            if current_brick.is_up:
                return RenkoTrendState.REVERSAL_UP
            else:
                return RenkoTrendState.REVERSAL_DOWN
        
        # Determine trend based on consecutive bricks and strength
        if current_brick.is_up:
            if consecutive_count >= 5 and trend_strength > 0.8:
                return RenkoTrendState.STRONG_UPTREND
            elif consecutive_count >= 3 or trend_strength > 0.6:
                return RenkoTrendState.UPTREND
            else:
                return RenkoTrendState.CONSOLIDATION
        else:
            if consecutive_count >= 5 and trend_strength < 0.2:
                return RenkoTrendState.STRONG_DOWNTREND
            elif consecutive_count >= 3 or trend_strength < 0.4:
                return RenkoTrendState.DOWNTREND
            else:
                return RenkoTrendState.CONSOLIDATION
    
    def _generate_signals(self, data: pd.DataFrame, renko_data: Dict, 
                         trend_state: RenkoTrendState) -> Tuple[SignalType, float]:
        consecutive_count = renko_data['consecutive_count']
        trend_strength = renko_data['trend_strength']
        
        # Base confidence on consecutive count and trend strength
        base_confidence = min(0.9, (consecutive_count * 0.15) + (abs(trend_strength - 0.5) * 2))
        
        # State-based signals
        if trend_state == RenkoTrendState.REVERSAL_UP:
            return SignalType.BUY, min(0.85, base_confidence + 0.2)
        elif trend_state == RenkoTrendState.REVERSAL_DOWN:
            return SignalType.SELL, min(0.85, base_confidence + 0.2)
        elif trend_state in [RenkoTrendState.STRONG_UPTREND, RenkoTrendState.UPTREND]:
            confidence = base_confidence * (0.9 if trend_state == RenkoTrendState.STRONG_UPTREND else 0.7)
            return SignalType.BUY, confidence
        elif trend_state in [RenkoTrendState.STRONG_DOWNTREND, RenkoTrendState.DOWNTREND]:
            confidence = base_confidence * (0.9 if trend_state == RenkoTrendState.STRONG_DOWNTREND else 0.7)
            return SignalType.SELL, confidence
        else:
            return SignalType.NEUTRAL, 0.3
    
    def _get_default_result(self) -> Dict:
        default_brick = RenkoBrick(0.0, 0.0, True, 0)
        default_result = RenkoTrendResult(default_brick, 0.0, RenkoTrendState.CONSOLIDATION, 
                                        0, 0.0, 0.0, 0.5, SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {
            'brick_size': self.brick_size,
            'atr_period': self.atr_period,
            'atr_multiplier': self.atr_multiplier
        }
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)