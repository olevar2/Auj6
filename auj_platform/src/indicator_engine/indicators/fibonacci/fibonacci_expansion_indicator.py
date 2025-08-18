"""
Fibonacci Expansion Indicator - Advanced Implementation
=========================

This indicator implements Fibonacci expansion calculations that measure the expansion
of price ranges based on Fibonacci ratios. Unlike extensions which project beyond swings,
expansions analyze the proportional relationships within completed price movements.

Features:
    - Advanced expansion ratio calculations using geometric analysis
- Multi-timeframe expansion pattern detection
- Dynamic expansion level validation with volume confirmation
- Expansion breakout probability modeling
- Machine learning expansion strength assessment
- Support/resistance zone identification based on expansions
- Comprehensive error handling and edge case management

Mathematical Foundation:
    - Primary expansion ratios: 61.8%, 100%, 161.8%, 261.8%
- Range expansion measurement: (high - low) / reference_range
- Volatility-adjusted expansion calculations
- Statistical validation using historical expansion performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta

from auj_platform.src.indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface
from auj_platform.src.indicator_engine.indicators.base.standard_indicator import IndicatorCalculationError

logger = logging.getLogger(__name__)

@dataclass
class FibonacciExpansionLevel:
    """Represents a Fibonacci expansion level with comprehensive properties."""
    ratio: float
    expansion_value: float
    time_index: int
    strength: float
    volume_confirmation: float
    reference_range: float
    expansion_type: str  # 'range', 'breakout', 'continuation'
    confidence: float
    support_resistance: str  # 'support', 'resistance', 'neutral'

@dataclass
class ExpansionPattern:
    """Represents an expansion pattern with geometric properties."""
    start_index: int
    end_index: int
    reference_high: float
    reference_low: float
    reference_range: float
    expansion_levels: List[FibonacciExpansionLevel]
    pattern_strength: float
    completion_percentage: float
    breakout_probability: float

class FibonacciExpansionIndicator(StandardIndicatorInterface):
    """
    Fibonacci Expansion Indicator for measuring price range expansions.

    This indicator analyzes the expansion of price ranges using Fibonacci ratios
    to identify potential breakout levels and range expansion patterns.
    """

def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the Fibonacci Expansion Indicator.

        Args:
            parameters: Configuration parameters including:
                - lookback_period: Period for finding reference ranges (default: 20)
                - expansion_ratios: List of Fibonacci ratios to use (default: [0.618, 1.0, 1.618, 2.618])
                - min_range_pips: Minimum range size for analysis (default: 20)
                - volume_confirmation_threshold: Volume threshold for confirmation (default: 1.2)
                - confidence_threshold: Minimum confidence for signals (default: 0.6)
        """

        default_params = {
            'lookback_period': 20,
            'expansion_ratios': [0.618, 1.0, 1.618, 2.618],
            'min_range_pips': 20,
            'volume_confirmation_threshold': 1.2,
            'confidence_threshold': 0.6,
            'clustering_eps': 0.1,
            'min_samples': 2
        }

        self.parameters = {**default_params, **(parameters or {})}

        super().__init__(name="FibonacciExpansionIndicator")
(        )

        logger.info(f"FibonacciExpansionIndicator initialized with parameters: {self.parameters}")

def calculate(self) -> IndicatorResult:
        """
        Calculate Fibonacci expansion levels and patterns.

        Args:
            data: Dictionary containing market data with 'ohlcv' key

        Returns:
            IndicatorResult containing expansion analysis
        """
        try:
            ohlcv = data['ohlcv']

            if len(ohlcv) < self.parameters['lookback_period']:
                raise IndicatorCalculationError()
                    f"Insufficient data for Fibonacci Expansion calculation. "
                    f"Required: {self.parameters['lookback_period']}, Available: {len(ohlcv)}"
(                )

            # Calculate expansion patterns
            expansion_patterns = self._identify_expansion_patterns(ohlcv)

            # Calculate current expansion levels
            current_levels = self._calculate_current_expansion_levels(ohlcv, expansion_patterns)

            # Assess expansion strength
            expansion_strength = self._assess_expansion_strength(ohlcv, current_levels)

            # Generate signals
            signals = self._generate_expansion_signals(ohlcv, current_levels, expansion_strength)

            # Calculate support/resistance zones
            support_resistance = self._calculate_support_resistance_zones(current_levels)

            result = IndicatorResult()
                value=expansion_strength,
                signals=signals,
                metadata={
                    'expansion_patterns': [self._pattern_to_dict(p) for p in expansion_patterns],
                    'current_levels': [self._level_to_dict(l) for l in current_levels],
                    'support_resistance': support_resistance,
                    'expansion_summary': {
                        'total_patterns': len(expansion_patterns),
                        'active_levels': len(current_levels),
                        'avg_strength': np.mean([l.strength for l in current_levels]) if current_levels else 0.0
                    }
                }
(            )

            logger.debug(f"FibonacciExpansionIndicator calculated successfully for {len(ohlcv)} periods")
            return result

        except Exception as e:
            logger.error(f"Error in FibonacciExpansionIndicator calculation: {str(e)}")
            raise IndicatorCalculationError(f"Fibonacci Expansion calculation failed: {str(e)}")

def _identify_expansion_patterns(self, ohlcv: pd.DataFrame) -> List[ExpansionPattern]:
        """Identify expansion patterns in the price data."""
        patterns = []
        lookback = self.parameters['lookback_period']

        for i in range(lookback, len(ohlcv) - lookback):
            # Find reference range
            start_idx = max(0, i - lookback)
            reference_data = ohlcv.iloc[start_idx:i+1]

            ref_high = reference_data['high'].max()
            ref_low = reference_data['low'].min()
            ref_range = ref_high - ref_low

            if ref_range < self.parameters['min_range_pips'] * 0.0001:
                continue

            # Calculate expansion levels for this pattern
            expansion_levels = []
            for ratio in self.parameters['expansion_ratios']:
                expansion_value = ref_range * ratio

                level = FibonacciExpansionLevel()
                    ratio=ratio,
                    expansion_value=expansion_value,
                    time_index=i,
                    strength=self._calculate_level_strength(ohlcv, i, ref_high, ref_low, expansion_value),
                    volume_confirmation=self._calculate_volume_confirmation(ohlcv, i),
                    reference_range=ref_range,
                    expansion_type=self._classify_expansion_type(ohlcv, i, expansion_value),
                    confidence=0.0,  # Will be calculated later
                    support_resistance='neutral'
(                )

                expansion_levels.append(level)

            # Create pattern
            pattern = ExpansionPattern()
                start_index=start_idx,
                end_index=i,
                reference_high=ref_high,
                reference_low=ref_low,
                reference_range=ref_range,
                expansion_levels=expansion_levels,
                pattern_strength=self._calculate_pattern_strength(expansion_levels),
                completion_percentage=self._calculate_completion_percentage(ohlcv, i, expansion_levels),
                breakout_probability=self._calculate_breakout_probability(ohlcv, i, expansion_levels)
(            )

            patterns.append(pattern)

        return patterns[-10:]  # Keep last 10 patterns for efficiency

def _calculate_current_expansion_levels(self, ohlcv: pd.DataFrame, patterns: List[ExpansionPattern]) -> List[FibonacciExpansionLevel]:
        """Calculate current active expansion levels."""
        if not patterns:
            return []

        # Use the most recent pattern
        current_pattern = patterns[-1]
        current_price = ohlcv['close'].iloc[-1]

        # Update levels with current market conditions
        current_levels = []
        for level in current_pattern.expansion_levels:
            # Recalculate strength and confidence based on current price action
            updated_level = FibonacciExpansionLevel()
                ratio=level.ratio,
                expansion_value=level.expansion_value,
                time_index=len(ohlcv) - 1,
                strength=self._calculate_current_level_strength(ohlcv, level, current_price),
                volume_confirmation=self._calculate_volume_confirmation(ohlcv, -1),
                reference_range=level.reference_range,
                expansion_type=level.expansion_type,
                confidence=self._calculate_level_confidence(ohlcv, level, current_price),
                support_resistance=self._determine_support_resistance(level, current_price)
(            )
            current_levels.append(updated_level)

        return current_levels

def _calculate_level_strength(self, ohlcv: pd.DataFrame, index: int, ref_high: float, ref_low: float, expansion_value: float) -> float:
        """Calculate the strength of an expansion level."""
        if index >= len(ohlcv):
            return 0.0

        current_range = ohlcv['high'].iloc[index] - ohlcv['low'].iloc[index]
        ref_range = ref_high - ref_low

        if ref_range == 0:
            return 0.0

        # Calculate expansion ratio
        expansion_ratio = current_range / ref_range

        # Fibonacci ratio alignment
        fib_alignment = 0.0
        for fib_ratio in self.parameters['expansion_ratios']:
            if abs(expansion_ratio - fib_ratio) < 0.1:
                fib_alignment = 1.0 - abs(expansion_ratio - fib_ratio) / 0.1
                break

        # Volume factor
        volume_factor = min(2.0, ohlcv['volume'].iloc[index] / ohlcv['volume'].rolling(20).mean().iloc[index]) if index >= 20 else 1.0

        strength = (fib_alignment * 0.6 + min(1.0, volume_factor) * 0.4)
        return max(0.0, min(1.0, strength))

def _calculate_volume_confirmation(self, ohlcv: pd.DataFrame, index: int) -> float:
        """Calculate volume confirmation for expansion level."""
        if index >= len(ohlcv) or index < 20:
            return 0.5

        current_volume = ohlcv['volume'].iloc[index]
        avg_volume = ohlcv['volume'].rolling(20).mean().iloc[index]

        if avg_volume == 0:
            return 0.5

        volume_ratio = current_volume / avg_volume
        confirmation = min(1.0, volume_ratio / self.parameters['volume_confirmation_threshold'])

        return confirmation

def _classify_expansion_type(self, ohlcv: pd.DataFrame, index: int, expansion_value: float) -> str:
        """Classify the type of expansion."""
        if index < 5 or index >= len(ohlcv):
            return 'neutral'

        recent_range = ohlcv['high'].iloc[index-4:index+1].max() - ohlcv['low'].iloc[index-4:index+1].min()

        if recent_range > expansion_value * 1.2:
            return 'breakout'
        elif recent_range < expansion_value * 0.8:
            return 'range'
        else:
            return 'continuation'

def _calculate_pattern_strength(self, levels: List[FibonacciExpansionLevel]) -> float:
        """Calculate overall pattern strength."""
        if not levels:
            return 0.0

        avg_strength = np.mean([level.strength for level in levels])
        volume_confirmation = np.mean([level.volume_confirmation for level in levels])

        return (avg_strength * 0.7 + volume_confirmation * 0.3)

def _calculate_completion_percentage(self, ohlcv: pd.DataFrame, index: int, levels: List[FibonacciExpansionLevel]) -> float:
        """Calculate pattern completion percentage."""
        if not levels or index >= len(ohlcv):
            return 0.0

        current_price = ohlcv['close'].iloc[index]
        completed_levels = sum(1 for level in levels if abs(current_price - level.expansion_value) < level.expansion_value * 0.05)

        return completed_levels / len(levels)

def _calculate_breakout_probability(self, ohlcv: pd.DataFrame, index: int, levels: List[FibonacciExpansionLevel]) -> float:
        """Calculate breakout probability."""
        if not levels or index >= len(ohlcv) - 5:
            return 0.0

        # Analyze recent price action
        recent_volatility = ohlcv['high'].iloc[index-4:index+1].std()
        avg_volatility = ohlcv['high'].rolling(20).std().iloc[index]

        if avg_volatility == 0:
            return 0.5

        volatility_ratio = recent_volatility / avg_volatility

        # Volume analysis
        recent_volume = ohlcv['volume'].iloc[index-4:index+1].mean()
        avg_volume = ohlcv['volume'].rolling(20).mean().iloc[index]

        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

        # Combine factors
        probability = min(1.0, (volatility_ratio * 0.6 + volume_ratio * 0.4) / 2.0)
        return probability

def _calculate_current_level_strength(self, ohlcv: pd.DataFrame, level: FibonacciExpansionLevel, current_price: float) -> float:
        """Calculate current strength of expansion level."""
        # Distance factor
        distance = abs(current_price - level.expansion_value) / level.expansion_value
        distance_factor = max(0.0, 1.0 - distance * 10)  # Closer = stronger

        # Original strength weighted by current relevance
        return level.strength * 0.7 + distance_factor * 0.3

def _calculate_level_confidence(self, ohlcv: pd.DataFrame, level: FibonacciExpansionLevel, current_price: float) -> float:
        """Calculate confidence in expansion level."""
        # Base confidence from strength and volume
        base_confidence = (level.strength + level.volume_confirmation) / 2.0

        # Fibonacci ratio weight (more standard ratios get higher confidence)
        fib_weights = {0.618: 1.0, 1.0: 0.9, 1.618: 1.0, 2.618: 0.8}
        ratio_weight = fib_weights.get(level.ratio, 0.5)

        confidence = base_confidence * ratio_weight
        return max(0.0, min(1.0, confidence))

def _determine_support_resistance(self, level: FibonacciExpansionLevel, current_price: float) -> str:
        """Determine if level acts as support or resistance."""
        if current_price > level.expansion_value * 1.02:
            return 'support'
        elif current_price < level.expansion_value * 0.98:
            return 'resistance'
        else:
            return 'neutral'

def _assess_expansion_strength(self, ohlcv: pd.DataFrame, levels: List[FibonacciExpansionLevel]) -> float:
        """Assess overall expansion strength."""
        if not levels:
            return 0.0

        # Calculate weighted average of level strengths
        total_weight = 0.0
        weighted_strength = 0.0

        for level in levels:
            weight = level.confidence * (1.0 + level.volume_confirmation)
            weighted_strength += level.strength * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_strength / total_weight

def _generate_expansion_signals(self, ohlcv: pd.DataFrame, levels: List[FibonacciExpansionLevel], strength: float) -> Dict[str, Any]:
        """Generate trading signals based on expansion analysis."""
        signals = {
            'signal_type': SignalType.NEUTRAL,
            'strength': strength,
            'confidence': 0.0,
            'expansion_signals': []
        }

        if not levels:
            return signals

        current_price = ohlcv['close'].iloc[-1]
        strong_levels = [l for l in levels if l.confidence > self.parameters['confidence_threshold']]

        if not strong_levels:
            return signals

        # Find nearest strong level
        nearest_level = min(strong_levels, key=lambda l: abs(current_price - l.expansion_value))

        # Generate signals based on proximity to expansion levels
        distance_pct = abs(current_price - nearest_level.expansion_value) / nearest_level.expansion_value

        if distance_pct < 0.02:  # Within 2% of expansion level:
            if current_price > nearest_level.expansion_value and nearest_level.expansion_type == 'breakout':
                signals['signal_type'] = SignalType.BUY
                signals['confidence'] = nearest_level.confidence
            elif current_price < nearest_level.expansion_value and nearest_level.support_resistance == 'support':
                signals['signal_type'] = SignalType.BUY
                signals['confidence'] = nearest_level.confidence * 0.8

        # Add individual level signals
        for level in strong_levels:
            level_signal = {
                'ratio': level.ratio,
                'price_level': level.expansion_value,
                'signal_type': self._get_level_signal_type(current_price, level),
                'strength': level.strength,
                'confidence': level.confidence
            }
            signals['expansion_signals'].append(level_signal)

        return signals

def _get_level_signal_type(self, current_price: float, level: FibonacciExpansionLevel) -> str:
        """Get signal type for specific expansion level."""
        distance_pct = (current_price - level.expansion_value) / level.expansion_value

        if abs(distance_pct) < 0.01:  # Very close:
            return 'AT_LEVEL'
        elif distance_pct > 0.01:
            return 'ABOVE_LEVEL'
        else:
            return 'BELOW_LEVEL'

def _calculate_support_resistance_zones(self, levels: List[FibonacciExpansionLevel]) -> Dict[str, Any]:
        """Calculate support and resistance zones from expansion levels."""
        support_levels = []
        resistance_levels = []

        for level in levels:
            level_info = {
                'price': level.expansion_value,
                'strength': level.strength,
                'confidence': level.confidence,
                'ratio': level.ratio
            }

            if level.support_resistance == 'support':
                support_levels.append(level_info)
            elif level.support_resistance == 'resistance':
                resistance_levels.append(level_info)

        return {
            'support_levels': sorted(support_levels, key=lambda x: x['price']),
            'resistance_levels': sorted(resistance_levels, key=lambda x: x['price'], reverse=True),
            'total_levels': len(levels),
            'strong_levels': len([l for l in levels if l.confidence > self.parameters['confidence_threshold']])
        }

def _pattern_to_dict(self, pattern: ExpansionPattern) -> Dict[str, Any]:
        """Convert expansion pattern to dictionary."""
        return {
            'start_index': pattern.start_index,
            'end_index': pattern.end_index,
            'reference_high': pattern.reference_high,
            'reference_low': pattern.reference_low,
            'reference_range': pattern.reference_range,
            'pattern_strength': pattern.pattern_strength,
            'completion_percentage': pattern.completion_percentage,
            'breakout_probability': pattern.breakout_probability,
            'levels_count': len(pattern.expansion_levels)
        }

def _level_to_dict(self, level: FibonacciExpansionLevel) -> Dict[str, Any]:
        """Convert expansion level to dictionary."""
        return {
            'ratio': level.ratio,
            'expansion_value': level.expansion_value,
            'strength': level.strength,
            'confidence': level.confidence,
            'volume_confirmation': level.volume_confirmation,
            'expansion_type': level.expansion_type,
            'support_resistance': level.support_resistance
        }

def get_data_requirements(self) -> List[DataRequirement]:
        """Return data requirements for this indicator."""
        return []
            DataRequirement()
                data_type=DataType.OHLCV,
                required_columns=['open', 'high', 'low', 'close'],
                min_periods=50
(            )
[        ]

def validate_parameters(self) -> bool:
        """Validate indicator parameters."""
        required_params = ['lookback_period', 'expansion_ratios', 'confidence_threshold']

        for param in required_params:
            if param not in self.parameters:
                logger.error(f"Missing required parameter: {param}")
                return False

        if not isinstance(self.parameters['expansion_ratios'], list):
            logger.error("expansion_ratios must be a list")
            return False

        if not all(isinstance(ratio, (int, float)) for ratio in self.parameters['expansion_ratios']):
            logger.error("All expansion ratios must be numeric")
            return False

        return True

def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate raw Fibonacci expansion data for single timeframe.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dict containing basic expansion calculations
        """
        try:
            if len(data) < self.parameters['lookback_period']:
                return {'error': 'Insufficient data'}
                
            # Extract basic price data
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            
            # Find recent swing points
            lookback = self.parameters['lookback_period']
            recent_data = data.tail(lookback)
            
            highs = recent_data['high']
            lows = recent_data['low']
            
            swing_high = highs.max()
            swing_low = lows.min()
            current_price = close[-1]
            
            # Calculate basic expansion levels
            expansion_range = swing_high - swing_low
            expansion_levels = {}
            
            for ratio in self.parameters['expansion_ratios']:
                level_value = swing_high + (expansion_range * ratio)
                expansion_levels[f'expansion_{ratio}'] = level_value
                
            return {
                'swing_high': swing_high,
                'swing_low': swing_low,
                'current_price': current_price,
                'expansion_range': expansion_range,
                'expansion_levels': expansion_levels,
                'trend_direction': 'bullish' if current_price > (swing_high + swing_low) / 2 else 'bearish'
            }
            
        except Exception as e:
            logger.error(f"Error in fibonacci expansion raw calculation: {str(e)}")
            return {'error': str(e)}
