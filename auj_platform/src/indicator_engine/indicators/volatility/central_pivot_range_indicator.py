"""
Advanced Central Pivot Range Indicator with Adaptive Calculations

This implementation features:
- Sophisticated pivot point calculations using multiple methods
- Dynamic support and resistance level detection
- Adaptive range calculations based on market volatility
- Multi-timeframe pivot analysis
- Volume-weighted pivot points
- Statistical significance testing
- Production-ready error handling

Author: AUJ Platform Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
from scipy.stats import norm, t
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import IndicatorCalculationError
from ....core.signal_type import SignalType


class PivotMethod(Enum):
    """Pivot calculation methods"""
    STANDARD = "standard"
    FIBONACCI = "fibonacci"
    WOODIE = "woodie"
    CAMARILLA = "camarilla"
    DEMARK = "demark"
    VOLUME_WEIGHTED = "volume_weighted"


@dataclass
class PivotConfig:
    """Configuration for Central Pivot Range calculation"""
    method: PivotMethod = PivotMethod.STANDARD
    adaptive_ranges: bool = True
    volume_weighting: bool = True
    multi_timeframe: bool = True
    statistical_validation: bool = True
    volatility_adjustment: bool = True
    min_periods: int = 20
    confidence_level: float = 0.95
    
    
@dataclass
class PivotLevels:
    """Pivot point levels structure"""
    pivot: float
    resistance_1: float
    resistance_2: float
    resistance_3: float
    support_1: float
    support_2: float
    support_3: float
    range_width: float
    volume_weight: float
    confidence: float


@dataclass
class PivotAnalysis:
    """Analysis results for pivot points"""
    current_position: str  # above_r3, between_r2_r3, etc.
    nearest_level: float
    distance_to_level: float
    probability_bounce: float
    probability_break: float
    strength_rating: float
    trend_bias: str


class CentralPivotRangeIndicator(StandardIndicatorInterface):
    """
    Advanced Central Pivot Range Indicator with Adaptive Calculations
    
    Features:
    - Multiple pivot calculation methods
    - Adaptive range calculations
    - Volume-weighted pivots
    - Statistical validation
    - Multi-timeframe analysis
    """
    
    def __init__(self, config: Optional[PivotConfig] = None):
        super().__init__()
        self.config = config or PivotConfig()
        self.logger = logging.getLogger(__name__)
        
        # Historical data storage
        self.pivot_history: List[PivotLevels] = []
        self.price_history: List[Tuple[float, float, float, float]] = []  # OHLC
        self.volume_history: List[float] = []
        
        # Statistical tracking
        self.bounce_accuracy: Dict[str, List[bool]] = {}
        self.break_accuracy: Dict[str, List[bool]] = {}
        
        # Performance metrics
        self.calculation_count = 0
        self.error_count = 0
        
    def get_required_data_types(self) -> List[str]:
        """Return required data types"""
        return ["ohlcv"]
    
    def get_required_columns(self) -> List[str]:
        """Return required columns"""
        return ["open", "high", "low", "close", "volume"]
    
    def calculate(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate advanced Central Pivot Range with adaptive features
        
        Args:
            data: Dictionary containing OHLCV data
            
        Returns:
            Dictionary containing pivot range results
        """
        try:
            self.calculation_count += 1
            self.logger.debug(f"Calculating Central Pivot Range (calculation #{self.calculation_count})")
            
            # Validate input data
            ohlcv_data = self._validate_input_data(data)
            
            if len(ohlcv_data) < self.config.min_periods:
                raise IndicatorCalculationError(
                    f"Insufficient data: need at least {self.config.min_periods} periods, got {len(ohlcv_data)}"
                )
            
            # Extract OHLCV data
            opens = ohlcv_data['open'].values
            highs = ohlcv_data['high'].values
            lows = ohlcv_data['low'].values
            closes = ohlcv_data['close'].values
            volumes = ohlcv_data['volume'].values
            
            # Calculate current pivot levels
            pivot_levels = self._calculate_pivot_levels(
                opens[-1], highs[-1], lows[-1], closes[-1], volumes[-1],
                highs, lows, closes, volumes
            )
            
            # Perform analysis
            analysis = self._analyze_pivot_position(
                closes[-1], pivot_levels, highs, lows, closes
            )
            
            # Update historical data
            self._update_history(opens[-1], highs[-1], lows[-1], closes[-1], volumes[-1], pivot_levels)
            
            # Validate levels statistically if enabled
            if self.config.statistical_validation:
                pivot_levels = self._validate_levels_statistically(pivot_levels, closes)
            
            return self._format_output(pivot_levels, analysis, closes[-1])
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error calculating Central Pivot Range: {str(e)}")
            raise IndicatorCalculationError(f"Central Pivot Range calculation failed: {str(e)}")
    
    def _validate_input_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Validate and extract OHLCV data"""
        if "ohlcv" not in data:
            raise IndicatorCalculationError("OHLCV data not found in input")
        
        ohlcv_data = data["ohlcv"]
        required_columns = self.get_required_columns()
        
        for col in required_columns:
            if col not in ohlcv_data.columns:
                raise IndicatorCalculationError(f"Required column '{col}' not found in data")
        
        # Check for NaN values
        if ohlcv_data[required_columns].isnull().any().any():
            self.logger.warning("NaN values detected in input data, forward filling...")
            ohlcv_data = ohlcv_data.fillna(method='ffill')
        
        return ohlcv_data
    
    def _calculate_pivot_levels(self, open_price: float, high: float, low: float, 
                              close: float, volume: float, hist_highs: np.ndarray, 
                              hist_lows: np.ndarray, hist_closes: np.ndarray, 
                              hist_volumes: np.ndarray) -> PivotLevels:
        """Calculate pivot point levels using selected method"""
        
        if self.config.method == PivotMethod.STANDARD:
            return self._calculate_standard_pivots(high, low, close, volume, hist_volumes)
        elif self.config.method == PivotMethod.FIBONACCI:
            return self._calculate_fibonacci_pivots(high, low, close, volume, hist_volumes)
        elif self.config.method == PivotMethod.WOODIE:
            return self._calculate_woodie_pivots(open_price, high, low, close, volume, hist_volumes)
        elif self.config.method == PivotMethod.CAMARILLA:
            return self._calculate_camarilla_pivots(high, low, close, volume, hist_volumes)
        elif self.config.method == PivotMethod.DEMARK:
            return self._calculate_demark_pivots(open_price, high, low, close, volume, hist_volumes)
        elif self.config.method == PivotMethod.VOLUME_WEIGHTED:
            return self._calculate_volume_weighted_pivots(high, low, close, volume, 
                                                        hist_highs, hist_lows, hist_closes, hist_volumes)
        else:
            return self._calculate_standard_pivots(high, low, close, volume, hist_volumes)
    
    def _calculate_standard_pivots(self, high: float, low: float, close: float, 
                                 volume: float, hist_volumes: np.ndarray) -> PivotLevels:
        """Calculate standard pivot points"""
        pivot = (high + low + close) / 3
        
        # Apply volume weighting if enabled
        volume_weight = 1.0
        if self.config.volume_weighting and len(hist_volumes) >= 10:
            avg_volume = np.mean(hist_volumes[-10:])
            volume_weight = min(2.0, max(0.5, volume / avg_volume)) if avg_volume > 0 else 1.0
        
        # Calculate adaptive range multiplier
        range_multiplier = self._calculate_adaptive_range_multiplier(high, low, close)
        
        # Standard calculations with adaptive ranges
        r1 = pivot + (range_multiplier * (pivot - low))
        s1 = pivot - (range_multiplier * (high - pivot))
        r2 = pivot + (range_multiplier * (high - low))
        s2 = pivot - (range_multiplier * (high - low))
        r3 = high + 2 * (range_multiplier * (pivot - low))
        s3 = low - 2 * (range_multiplier * (high - pivot))
        
        range_width = r1 - s1
        confidence = self._calculate_level_confidence(pivot, high, low, close)
        
        return PivotLevels(
            pivot=pivot, resistance_1=r1, resistance_2=r2, resistance_3=r3,
            support_1=s1, support_2=s2, support_3=s3,
            range_width=range_width, volume_weight=volume_weight, confidence=confidence
        )
    
    def _calculate_fibonacci_pivots(self, high: float, low: float, close: float,
                                  volume: float, hist_volumes: np.ndarray) -> PivotLevels:
        """Calculate Fibonacci-based pivot points"""
        pivot = (high + low + close) / 3
        
        # Fibonacci ratios
        fib_ratios = [0.382, 0.618, 1.0]
        range_val = high - low
        
        volume_weight = 1.0
        if self.config.volume_weighting and len(hist_volumes) >= 10:
            avg_volume = np.mean(hist_volumes[-10:])
            volume_weight = min(2.0, max(0.5, volume / avg_volume)) if avg_volume > 0 else 1.0
        
        # Apply adaptive range multiplier
        range_multiplier = self._calculate_adaptive_range_multiplier(high, low, close)
        
        r1 = pivot + (fib_ratios[0] * range_val * range_multiplier)
        r2 = pivot + (fib_ratios[1] * range_val * range_multiplier)
        r3 = pivot + (fib_ratios[2] * range_val * range_multiplier)
        
        s1 = pivot - (fib_ratios[0] * range_val * range_multiplier)
        s2 = pivot - (fib_ratios[1] * range_val * range_multiplier)
        s3 = pivot - (fib_ratios[2] * range_val * range_multiplier)
        
        range_width = r1 - s1
        confidence = self._calculate_level_confidence(pivot, high, low, close)
        
        return PivotLevels(
            pivot=pivot, resistance_1=r1, resistance_2=r2, resistance_3=r3,
            support_1=s1, support_2=s2, support_3=s3,
            range_width=range_width, volume_weight=volume_weight, confidence=confidence
        )
    
    def _calculate_woodie_pivots(self, open_price: float, high: float, low: float, 
                               close: float, volume: float, hist_volumes: np.ndarray) -> PivotLevels:
        """Calculate Woodie's pivot points"""
        if close > open_price:
            pivot = (high + low + 2 * close) / 4
        elif close < open_price:
            pivot = (high + low + 2 * open_price) / 4
        else:
            pivot = (high + low + close + open_price) / 4
        
        volume_weight = 1.0
        if self.config.volume_weighting and len(hist_volumes) >= 10:
            avg_volume = np.mean(hist_volumes[-10:])
            volume_weight = min(2.0, max(0.5, volume / avg_volume)) if avg_volume > 0 else 1.0
        
        range_multiplier = self._calculate_adaptive_range_multiplier(high, low, close)
        
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (range_multiplier * (high - low))
        s2 = pivot - (range_multiplier * (high - low))
        r3 = high + 2 * (range_multiplier * (pivot - low))
        s3 = low - 2 * (range_multiplier * (high - pivot))
        
        range_width = r1 - s1
        confidence = self._calculate_level_confidence(pivot, high, low, close)
        
        return PivotLevels(
            pivot=pivot, resistance_1=r1, resistance_2=r2, resistance_3=r3,
            support_1=s1, support_2=s2, support_3=s3,
            range_width=range_width, volume_weight=volume_weight, confidence=confidence
        )
    
    def _calculate_camarilla_pivots(self, high: float, low: float, close: float,
                                  volume: float, hist_volumes: np.ndarray) -> PivotLevels:
        """Calculate Camarilla pivot points"""
        pivot = (high + low + close) / 3
        
        # Camarilla multipliers
        multipliers = [1.1/12, 1.1/6, 1.1/4]
        range_val = high - low
        
        volume_weight = 1.0
        if self.config.volume_weighting and len(hist_volumes) >= 10:
            avg_volume = np.mean(hist_volumes[-10:])
            volume_weight = min(2.0, max(0.5, volume / avg_volume)) if avg_volume > 0 else 1.0
        
        range_multiplier = self._calculate_adaptive_range_multiplier(high, low, close)
        
        r1 = close + (multipliers[0] * range_val * range_multiplier)
        r2 = close + (multipliers[1] * range_val * range_multiplier)
        r3 = close + (multipliers[2] * range_val * range_multiplier)
        
        s1 = close - (multipliers[0] * range_val * range_multiplier)
        s2 = close - (multipliers[1] * range_val * range_multiplier)
        s3 = close - (multipliers[2] * range_val * range_multiplier)
        
        range_width = r1 - s1
        confidence = self._calculate_level_confidence(pivot, high, low, close)
        
        return PivotLevels(
            pivot=pivot, resistance_1=r1, resistance_2=r2, resistance_3=r3,
            support_1=s1, support_2=s2, support_3=s3,
            range_width=range_width, volume_weight=volume_weight, confidence=confidence
        )    
    def _calculate_demark_pivots(self, open_price: float, high: float, low: float, 
                               close: float, volume: float, hist_volumes: np.ndarray) -> PivotLevels:
        """Calculate Tom DeMark's pivot points"""
        # DeMark pivot calculation
        if close < open_price:
            x = high + 2 * low + close
        elif close > open_price:
            x = 2 * high + low + close
        else:
            x = high + low + 2 * close
        
        pivot = x / 4
        
        volume_weight = 1.0
        if self.config.volume_weighting and len(hist_volumes) >= 10:
            avg_volume = np.mean(hist_volumes[-10:])
            volume_weight = min(2.0, max(0.5, volume / avg_volume)) if avg_volume > 0 else 1.0
        
        range_multiplier = self._calculate_adaptive_range_multiplier(high, low, close)
        
        r1 = (x / 2) - low
        s1 = (x / 2) - high
        
        # Extended levels with range multiplier
        r2 = pivot + (range_multiplier * (high - low))
        s2 = pivot - (range_multiplier * (high - low))
        r3 = high + 2 * (range_multiplier * (pivot - low))
        s3 = low - 2 * (range_multiplier * (high - pivot))
        
        range_width = r1 - s1
        confidence = self._calculate_level_confidence(pivot, high, low, close)
        
        return PivotLevels(
            pivot=pivot, resistance_1=r1, resistance_2=r2, resistance_3=r3,
            support_1=s1, support_2=s2, support_3=s3,
            range_width=range_width, volume_weight=volume_weight, confidence=confidence
        )
    
    def _calculate_volume_weighted_pivots(self, high: float, low: float, close: float,
                                        volume: float, hist_highs: np.ndarray, 
                                        hist_lows: np.ndarray, hist_closes: np.ndarray,
                                        hist_volumes: np.ndarray) -> PivotLevels:
        """Calculate volume-weighted pivot points"""
        # Use last 10 periods for volume weighting
        periods = min(10, len(hist_volumes))
        if periods < 3:
            return self._calculate_standard_pivots(high, low, close, volume, hist_volumes)
        
        # Calculate VWAP-style pivot
        recent_volumes = hist_volumes[-periods:]
        recent_closes = hist_closes[-periods:]
        recent_highs = hist_highs[-periods:]
        recent_lows = hist_lows[-periods:]
        
        total_volume = np.sum(recent_volumes)
        if total_volume == 0:
            return self._calculate_standard_pivots(high, low, close, volume, hist_volumes)
        
        vw_close = np.sum(recent_closes * recent_volumes) / total_volume
        vw_high = np.sum(recent_highs * recent_volumes) / total_volume
        vw_low = np.sum(recent_lows * recent_volumes) / total_volume
        
        pivot = (vw_high + vw_low + vw_close) / 3
        
        volume_weight = volume / (total_volume / periods) if total_volume > 0 else 1.0
        volume_weight = min(2.0, max(0.5, volume_weight))
        
        range_multiplier = self._calculate_adaptive_range_multiplier(high, low, close)
        
        # Volume-adjusted ranges
        vol_range = vw_high - vw_low
        r1 = pivot + (0.382 * vol_range * range_multiplier * volume_weight)
        r2 = pivot + (0.618 * vol_range * range_multiplier * volume_weight)
        r3 = pivot + (1.0 * vol_range * range_multiplier * volume_weight)
        
        s1 = pivot - (0.382 * vol_range * range_multiplier * volume_weight)
        s2 = pivot - (0.618 * vol_range * range_multiplier * volume_weight)
        s3 = pivot - (1.0 * vol_range * range_multiplier * volume_weight)
        
        range_width = r1 - s1
        confidence = self._calculate_level_confidence(pivot, high, low, close)
        
        return PivotLevels(
            pivot=pivot, resistance_1=r1, resistance_2=r2, resistance_3=r3,
            support_1=s1, support_2=s2, support_3=s3,
            range_width=range_width, volume_weight=volume_weight, confidence=confidence
        )
    
    def _calculate_adaptive_range_multiplier(self, high: float, low: float, close: float) -> float:
        """Calculate adaptive range multiplier based on volatility"""
        if not self.config.adaptive_ranges:
            return 1.0
        
        try:
            if len(self.price_history) < 20:
                return 1.0
            
            # Calculate recent volatility
            recent_ranges = []
            for i, (o, h, l, c) in enumerate(self.price_history[-20:]):
                true_range = max(h - l, abs(h - c), abs(l - c))
                recent_ranges.append(true_range)
            
            if not recent_ranges:
                return 1.0
            
            avg_range = np.mean(recent_ranges)
            current_range = high - low
            
            if avg_range == 0:
                return 1.0
            
            volatility_ratio = current_range / avg_range
            
            # Adaptive multiplier: higher volatility = wider ranges
            multiplier = min(2.0, max(0.5, 0.8 + (volatility_ratio * 0.4)))
            
            return multiplier
            
        except Exception as e:
            self.logger.warning(f"Adaptive range calculation failed: {e}")
            return 1.0
    
    def _calculate_level_confidence(self, pivot: float, high: float, low: float, close: float) -> float:
        """Calculate confidence level for pivot points"""
        try:
            if len(self.pivot_history) < 10:
                return 0.7  # Default confidence
            
            # Analyze historical accuracy
            recent_accuracy = []
            for hist_pivot in self.pivot_history[-10:]:
                # Check if historical levels were respected
                level_accuracy = 0.0
                levels = [
                    hist_pivot.support_3, hist_pivot.support_2, hist_pivot.support_1,
                    hist_pivot.pivot,
                    hist_pivot.resistance_1, hist_pivot.resistance_2, hist_pivot.resistance_3
                ]
                
                # Simple accuracy check based on price action near levels
                for level in levels:
                    if abs(close - level) / close < 0.01:  # Within 1%
                        level_accuracy += 0.2
                
                recent_accuracy.append(min(1.0, level_accuracy))
            
            avg_accuracy = np.mean(recent_accuracy) if recent_accuracy else 0.7
            
            # Adjust for current market conditions
            range_factor = (high - low) / close if close > 0 else 0
            if range_factor > 0.03:  # High volatility
                avg_accuracy *= 0.8
            elif range_factor < 0.01:  # Low volatility
                avg_accuracy *= 1.1
            
            return min(0.95, max(0.2, avg_accuracy))
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.7
    
    def _analyze_pivot_position(self, current_price: float, pivot_levels: PivotLevels,
                              hist_highs: np.ndarray, hist_lows: np.ndarray,
                              hist_closes: np.ndarray) -> PivotAnalysis:
        """Analyze current price position relative to pivot levels"""
        
        # Determine current position
        position = self._determine_position(current_price, pivot_levels)
        
        # Find nearest level and distance
        nearest_level, distance = self._find_nearest_level(current_price, pivot_levels)
        
        # Calculate probabilities
        bounce_prob, break_prob = self._calculate_level_probabilities(
            current_price, nearest_level, hist_highs, hist_lows, hist_closes
        )
        
        # Calculate strength rating
        strength = self._calculate_level_strength(nearest_level, pivot_levels, hist_closes)
        
        # Determine trend bias
        trend_bias = self._determine_trend_bias(current_price, pivot_levels, hist_closes)
        
        return PivotAnalysis(
            current_position=position,
            nearest_level=nearest_level,
            distance_to_level=distance,
            probability_bounce=bounce_prob,
            probability_break=break_prob,
            strength_rating=strength,
            trend_bias=trend_bias
        )
    
    def _determine_position(self, price: float, levels: PivotLevels) -> str:
        """Determine price position relative to pivot levels"""
        if price > levels.resistance_3:
            return "above_r3"
        elif price > levels.resistance_2:
            return "between_r2_r3"
        elif price > levels.resistance_1:
            return "between_r1_r2"
        elif price > levels.pivot:
            return "between_pivot_r1"
        elif price > levels.support_1:
            return "between_s1_pivot"
        elif price > levels.support_2:
            return "between_s2_s1"
        elif price > levels.support_3:
            return "between_s3_s2"
        else:
            return "below_s3"
    
    def _find_nearest_level(self, price: float, levels: PivotLevels) -> Tuple[float, float]:
        """Find the nearest pivot level and distance to it"""
        all_levels = [
            levels.support_3, levels.support_2, levels.support_1,
            levels.pivot,
            levels.resistance_1, levels.resistance_2, levels.resistance_3
        ]
        
        distances = [abs(price - level) for level in all_levels]
        min_index = np.argmin(distances)
        
        return all_levels[min_index], distances[min_index]
    
    def _calculate_level_probabilities(self, current_price: float, nearest_level: float,
                                     hist_highs: np.ndarray, hist_lows: np.ndarray,
                                     hist_closes: np.ndarray) -> Tuple[float, float]:
        """Calculate bounce and break probabilities"""
        try:
            if len(hist_closes) < 20:
                return 0.5, 0.5
            
            # Analyze historical behavior near similar levels
            level_tolerance = 0.01  # 1% tolerance
            touches = []
            
            for i in range(len(hist_closes)):
                high, low, close = hist_highs[i], hist_lows[i], hist_closes[i]
                
                # Check if price touched the level
                if (abs(high - nearest_level) / nearest_level < level_tolerance or
                    abs(low - nearest_level) / nearest_level < level_tolerance or
                    abs(close - nearest_level) / nearest_level < level_tolerance):
                    
                    # Determine if it bounced or broke
                    if i < len(hist_closes) - 1:
                        next_close = hist_closes[i + 1]
                        
                        if current_price > nearest_level:  # Resistance level
                            bounced = next_close < nearest_level
                        else:  # Support level
                            bounced = next_close > nearest_level
                        
                        touches.append(bounced)
            
            if len(touches) < 3:
                return 0.5, 0.5
            
            bounce_rate = sum(touches) / len(touches)
            break_rate = 1 - bounce_rate
            
            # Adjust for current volatility
            recent_volatility = np.std(hist_closes[-10:]) / np.mean(hist_closes[-10:])
            if recent_volatility > 0.02:  # High volatility
                break_rate *= 1.2
                bounce_rate *= 0.8
            
            # Normalize
            total = bounce_rate + break_rate
            if total > 0:
                bounce_rate /= total
                break_rate /= total
            
            return min(0.9, max(0.1, bounce_rate)), min(0.9, max(0.1, break_rate))
            
        except Exception as e:
            self.logger.warning(f"Probability calculation failed: {e}")
            return 0.5, 0.5
    
    def _calculate_level_strength(self, level: float, pivot_levels: PivotLevels,
                                hist_closes: np.ndarray) -> float:
        """Calculate the strength rating of a level"""
        try:
            strength = 0.5  # Base strength
            
            # Major levels get higher strength
            if abs(level - pivot_levels.pivot) < 0.001:
                strength = 0.8
            elif (abs(level - pivot_levels.resistance_1) < 0.001 or 
                  abs(level - pivot_levels.support_1) < 0.001):
                strength = 0.7
            elif (abs(level - pivot_levels.resistance_2) < 0.001 or 
                  abs(level - pivot_levels.support_2) < 0.001):
                strength = 0.6
            
            # Adjust for volume weight
            strength *= pivot_levels.volume_weight
            
            # Adjust for confidence
            strength *= pivot_levels.confidence
            
            return min(1.0, max(0.1, strength))
            
        except Exception:
            return 0.5
    
    def _determine_trend_bias(self, current_price: float, pivot_levels: PivotLevels,
                            hist_closes: np.ndarray) -> str:
        """Determine the trend bias based on price position"""
        if current_price > pivot_levels.pivot:
            if current_price > pivot_levels.resistance_1:
                return "strong_bullish"
            else:
                return "bullish"
        else:
            if current_price < pivot_levels.support_1:
                return "strong_bearish"
            else:
                return "bearish"
    
    def _validate_levels_statistically(self, pivot_levels: PivotLevels,
                                     hist_closes: np.ndarray) -> PivotLevels:
        """Validate pivot levels using statistical methods"""
        try:
            if len(hist_closes) < 30:
                return pivot_levels
            
            # Perform t-test on historical price distribution
            recent_prices = hist_closes[-30:]
            mean_price = np.mean(recent_prices)
            std_price = np.std(recent_prices)
            
            # Calculate statistical significance of levels
            t_stat = abs(pivot_levels.pivot - mean_price) / (std_price / np.sqrt(len(recent_prices)))
            p_value = 2 * (1 - t.cdf(abs(t_stat), len(recent_prices) - 1))
            
            # Adjust confidence based on statistical significance
            if p_value < 0.05:  # Statistically significant
                pivot_levels.confidence *= 1.1
            elif p_value > 0.2:  # Not significant
                pivot_levels.confidence *= 0.9
            
            pivot_levels.confidence = min(0.95, max(0.2, pivot_levels.confidence))
            
            return pivot_levels
            
        except Exception as e:
            self.logger.warning(f"Statistical validation failed: {e}")
            return pivot_levels
    
    def _update_history(self, open_price: float, high: float, low: float, 
                       close: float, volume: float, pivot_levels: PivotLevels):
        """Update historical data"""
        self.price_history.append((open_price, high, low, close))
        self.volume_history.append(volume)
        self.pivot_history.append(pivot_levels)
        
        # Keep only recent history
        max_history = 200
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]
            self.pivot_history = self.pivot_history[-max_history:]
    
    def _format_output(self, pivot_levels: PivotLevels, analysis: PivotAnalysis,
                      current_price: float) -> Dict[str, Any]:
        """Format the output result"""
        
        # Determine signal type and strength
        signal_type = SignalType.NEUTRAL
        signal_strength = 0.5
        
        # Signal logic based on position and probabilities
        if analysis.trend_bias in ["strong_bullish", "bullish"]:
            if analysis.probability_bounce > 0.7 and current_price < analysis.nearest_level:
                signal_type = SignalType.BUY
                signal_strength = analysis.probability_bounce * analysis.strength_rating
        elif analysis.trend_bias in ["strong_bearish", "bearish"]:
            if analysis.probability_bounce > 0.7 and current_price > analysis.nearest_level:
                signal_type = SignalType.SELL
                signal_strength = analysis.probability_bounce * analysis.strength_rating
        
        # Adjust for confidence
        signal_strength *= pivot_levels.confidence
        signal_strength = min(1.0, signal_strength)
        
        return {
            "signal_type": signal_type,
            "signal_strength": signal_strength,
            "values": {
                "pivot": pivot_levels.pivot,
                "resistance_1": pivot_levels.resistance_1,
                "resistance_2": pivot_levels.resistance_2,
                "resistance_3": pivot_levels.resistance_3,
                "support_1": pivot_levels.support_1,
                "support_2": pivot_levels.support_2,
                "support_3": pivot_levels.support_3,
                "range_width": pivot_levels.range_width,
                "current_price": current_price
            },
            "analysis": {
                "current_position": analysis.current_position,
                "nearest_level": analysis.nearest_level,
                "distance_to_level": analysis.distance_to_level,
                "probability_bounce": analysis.probability_bounce,
                "probability_break": analysis.probability_break,
                "strength_rating": analysis.strength_rating,
                "trend_bias": analysis.trend_bias
            },
            "metadata": {
                "method": self.config.method.value,
                "volume_weight": pivot_levels.volume_weight,
                "confidence": pivot_levels.confidence,
                "adaptive_ranges": self.config.adaptive_ranges,
                "calculation_count": self.calculation_count,
                "error_rate": self.error_count / max(1, self.calculation_count)
            }
        }
    
    def get_signal_type(self, data: Dict[str, pd.DataFrame]) -> SignalType:
        """Get signal type based on pivot analysis"""
        try:
            result = self.calculate(data)
            return result["signal_type"]
        except Exception:
            return SignalType.NEUTRAL
    
    def get_signal_strength(self, data: Dict[str, pd.DataFrame]) -> float:
        """Get signal strength"""
        try:
            result = self.calculate(data)
            return result["signal_strength"]
        except Exception:
            return 0.0