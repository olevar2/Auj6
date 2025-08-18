"""
Advanced Super Guppy Indicator with Multiple EMA Analysis

Features:
- Multiple EMA periods for comprehensive trend analysis
- Guppy alignment scoring
- Trend strength and direction measurement
- Fast and slow EMA group analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType

class SuperGuppyState(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    CONSOLIDATING_UP = "consolidating_up"
    NEUTRAL = "neutral"
    CONSOLIDATING_DOWN = "consolidating_down"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"

@dataclass
class SuperGuppyResult:
    fast_ema_alignment: float
    slow_ema_alignment: float
    overall_alignment: float
    trend_strength: float
    price_position: float
    state: SuperGuppyState
    signal: SignalType
    confidence: float

class SuperGuppyIndicator(StandardIndicatorInterface):
    def __init__(self, fast_periods: List[int] = None, slow_periods: List[int] = None):
        # Default Guppy periods
        self.fast_periods = fast_periods or [3, 5, 8, 10, 12, 15]
        self.slow_periods = slow_periods or [30, 35, 40, 45, 50, 60]
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            max_period = max(self.fast_periods + self.slow_periods)
            if len(data) < max_period + 10:
                raise ValueError("Insufficient data")
            
            # Calculate Super Guppy
            guppy_data = self._calculate_super_guppy(data)
            
            # Analyze state
            state = self._analyze_guppy_state(data, guppy_data)
            
            # Generate signals
            signal, confidence = self._generate_signals(data, guppy_data, state)
            
            latest_result = SuperGuppyResult(
                fast_ema_alignment=guppy_data['fast_alignment'].iloc[-1],
                slow_ema_alignment=guppy_data['slow_alignment'].iloc[-1],
                overall_alignment=guppy_data['overall_alignment'].iloc[-1],
                trend_strength=guppy_data['trend_strength'].iloc[-1],
                price_position=guppy_data['price_position'].iloc[-1],
                state=state,
                signal=signal,
                confidence=confidence
            )
            
            return {
                'current': latest_result,
                'values': {
                    'fast_alignment': guppy_data['fast_alignment'].tolist(),
                    'slow_alignment': guppy_data['slow_alignment'].tolist(),
                    'overall_alignment': guppy_data['overall_alignment'].tolist(),
                    'trend_strength': guppy_data['trend_strength'].tolist(),
                    'price_position': guppy_data['price_position'].tolist()
                },
                'signal': signal,
                'confidence': confidence,
                'state': state.value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Super Guppy: {e}")
            return self._get_default_result()
    
    def _calculate_super_guppy(self, data: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=data.index)
        
        # Calculate all EMAs
        fast_emas = {}
        slow_emas = {}
        
        for period in self.fast_periods:
            fast_emas[f'ema_{period}'] = data['close'].ewm(span=period).mean()
        
        for period in self.slow_periods:
            slow_emas[f'ema_{period}'] = data['close'].ewm(span=period).mean()
        
        # Calculate alignment scores
        df['fast_alignment'] = self._calculate_alignment_score(fast_emas)
        df['slow_alignment'] = self._calculate_alignment_score(slow_emas)
        df['overall_alignment'] = (df['fast_alignment'] + df['slow_alignment']) / 2
        
        # Calculate trend strength
        df['trend_strength'] = self._calculate_trend_strength(fast_emas, slow_emas)
        
        # Calculate price position relative to EMAs
        df['price_position'] = self._calculate_price_position(data, fast_emas, slow_emas)
        
        return df
    
    def _calculate_alignment_score(self, emas: Dict[str, pd.Series]) -> pd.Series:
        """Calculate EMA alignment score (-1 to 1)"""
        ema_values = list(emas.values())
        
        # Check if EMAs are properly ordered
        alignment_scores = []
        
        for i in range(len(ema_values[0])):
            if i < max(self.fast_periods + self.slow_periods):
                alignment_scores.append(0)
                continue
            
            current_values = [ema.iloc[i] for ema in ema_values if not pd.isna(ema.iloc[i])]
            
            if len(current_values) < 2:
                alignment_scores.append(0)
                continue
            
            # Check if sorted (ascending = bearish, descending = bullish)
            sorted_asc = sorted(current_values)
            sorted_desc = sorted(current_values, reverse=True)
            
            if current_values == sorted_desc:
                alignment_scores.append(1.0)  # Perfect bullish alignment
            elif current_values == sorted_asc:
                alignment_scores.append(-1.0)  # Perfect bearish alignment
            else:
                # Partial alignment - calculate how close to perfect order
                asc_diff = sum(abs(a - b) for a, b in zip(current_values, sorted_asc))
                desc_diff = sum(abs(a - b) for a, b in zip(current_values, sorted_desc))
                
                total_range = max(current_values) - min(current_values)
                if total_range == 0:
                    alignment_scores.append(0)
                else:
                    if desc_diff < asc_diff:
                        score = 1.0 - (desc_diff / (total_range * len(current_values)))
                    else:
                        score = -1.0 + (asc_diff / (total_range * len(current_values)))
                    alignment_scores.append(score)
        
        return pd.Series(alignment_scores, index=ema_values[0].index)
    
    def _calculate_trend_strength(self, fast_emas: Dict, slow_emas: Dict) -> pd.Series:
        """Calculate trend strength based on EMA separation"""
        all_emas = {**fast_emas, **slow_emas}
        ema_values = list(all_emas.values())
        
        strength_scores = []
        
        for i in range(len(ema_values[0])):
            if i < max(self.fast_periods + self.slow_periods):
                strength_scores.append(0)
                continue
            
            current_values = [ema.iloc[i] for ema in ema_values if not pd.isna(ema.iloc[i])]
            
            if len(current_values) < 2:
                strength_scores.append(0)
                continue
            
            # Calculate relative spread
            min_val = min(current_values)
            max_val = max(current_values)
            avg_val = sum(current_values) / len(current_values)
            
            if avg_val == 0:
                strength_scores.append(0)
            else:
                relative_spread = (max_val - min_val) / avg_val
                strength_scores.append(min(1.0, relative_spread * 10))
        
        return pd.Series(strength_scores, index=ema_values[0].index)
    
    def _calculate_price_position(self, data: pd.DataFrame, fast_emas: Dict, slow_emas: Dict) -> pd.Series:
        """Calculate price position relative to EMA groups"""
        all_emas = {**fast_emas, **slow_emas}
        
        position_scores = []
        
        for i in range(len(data)):
            if i < max(self.fast_periods + self.slow_periods):
                position_scores.append(0)
                continue
            
            current_price = data['close'].iloc[i]
            ema_values = [ema.iloc[i] for ema in all_emas.values() if not pd.isna(ema.iloc[i])]
            
            if not ema_values:
                position_scores.append(0)
                continue
            
            # Calculate how many EMAs price is above
            above_count = sum(1 for ema_val in ema_values if current_price > ema_val)
            total_count = len(ema_values)
            
            # Convert to -1 to 1 scale
            position_score = (above_count / total_count) * 2 - 1
            position_scores.append(position_score)
        
        return pd.Series(position_scores, index=data.index)
    
    def _analyze_guppy_state(self, data: pd.DataFrame, guppy_data: pd.DataFrame) -> SuperGuppyState:
        """Analyze Super Guppy state"""
        overall_alignment = guppy_data['overall_alignment'].iloc[-1]
        trend_strength = guppy_data['trend_strength'].iloc[-1]
        price_position = guppy_data['price_position'].iloc[-1]
        
        if pd.isna(overall_alignment) or pd.isna(trend_strength):
            return SuperGuppyState.NEUTRAL
        
        # Determine state based on alignment, strength, and price position
        if overall_alignment > 0.7 and trend_strength > 0.6 and price_position > 0.5:
            return SuperGuppyState.STRONG_UPTREND
        elif overall_alignment > 0.3 and price_position > 0.2:
            return SuperGuppyState.UPTREND
        elif overall_alignment > 0 and price_position > -0.2:
            return SuperGuppyState.CONSOLIDATING_UP
        elif overall_alignment < -0.7 and trend_strength > 0.6 and price_position < -0.5:
            return SuperGuppyState.STRONG_DOWNTREND
        elif overall_alignment < -0.3 and price_position < -0.2:
            return SuperGuppyState.DOWNTREND
        elif overall_alignment < 0 and price_position < 0.2:
            return SuperGuppyState.CONSOLIDATING_DOWN
        else:
            return SuperGuppyState.NEUTRAL
    
    def _generate_signals(self, data: pd.DataFrame, guppy_data: pd.DataFrame, 
                         state: SuperGuppyState) -> Tuple[SignalType, float]:
        overall_alignment = guppy_data['overall_alignment'].iloc[-1]
        trend_strength = guppy_data['trend_strength'].iloc[-1]
        
        if pd.isna(overall_alignment) or pd.isna(trend_strength):
            return SignalType.NEUTRAL, 0.0
        
        # Base confidence on alignment and strength
        base_confidence = (abs(overall_alignment) + trend_strength) / 2
        
        # State-based signals
        confidence_map = {
            SuperGuppyState.STRONG_UPTREND: 0.9,
            SuperGuppyState.UPTREND: 0.75,
            SuperGuppyState.CONSOLIDATING_UP: 0.5,
            SuperGuppyState.NEUTRAL: 0.3,
            SuperGuppyState.CONSOLIDATING_DOWN: 0.5,
            SuperGuppyState.DOWNTREND: 0.75,
            SuperGuppyState.STRONG_DOWNTREND: 0.9
        }
        
        state_confidence = confidence_map.get(state, 0.3)
        final_confidence = (base_confidence + state_confidence) / 2
        
        if state in [SuperGuppyState.STRONG_UPTREND, SuperGuppyState.UPTREND]:
            return SignalType.BUY, final_confidence
        elif state in [SuperGuppyState.STRONG_DOWNTREND, SuperGuppyState.DOWNTREND]:
            return SignalType.SELL, final_confidence
        elif state == SuperGuppyState.CONSOLIDATING_UP:
            return SignalType.BUY, final_confidence * 0.7
        elif state == SuperGuppyState.CONSOLIDATING_DOWN:
            return SignalType.SELL, final_confidence * 0.7
        else:
            return SignalType.NEUTRAL, final_confidence
    
    def _get_default_result(self) -> Dict:
        default_result = SuperGuppyResult(0.0, 0.0, 0.0, 0.0, 0.0, SuperGuppyState.NEUTRAL, 
                                        SignalType.NEUTRAL, 0.0)
        return {'current': default_result, 'values': {}, 'signal': SignalType.NEUTRAL, 'error': True}

    def get_parameters(self) -> Dict:
        return {
            'fast_periods': self.fast_periods,
            'slow_periods': self.slow_periods
        }
    
    def set_parameters(self, **kwargs):
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)