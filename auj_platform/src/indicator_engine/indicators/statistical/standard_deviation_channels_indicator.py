"""
Advanced Standard Deviation Channels Indicator with Dynamic Channel Analysis

This indicator implements comprehensive standard deviation channel analysis including:
- Multi-timeframe channel construction
- Dynamic channel width adjustment
- Breakout and reversal detection
- Channel momentum and trend analysis
- Volatility-adjusted channel boundaries
- Mean reversion and trend continuation signals
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
import logging
from dataclasses import dataclass

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError


@dataclass
class StandardDeviationChannelsResult:
    """Result container for standard deviation channels analysis"""
    upper_channel: np.ndarray
    lower_channel: np.ndarray
    middle_line: np.ndarray
    channel_width: np.ndarray
    price_position: np.ndarray
    breakout_signals: np.ndarray
    channel_slope: np.ndarray
    volatility_regime: np.ndarray
    mean_reversion_strength: np.ndarray
    trend_strength: np.ndarray
    channel_efficiency: np.ndarray
    support_resistance_levels: Dict[str, List[float]]
    adaptive_multiplier: np.ndarray
    volume_weighted_position: np.ndarray


class StandardDeviationChannelsIndicator(StandardIndicatorInterface):
    """
    Advanced Standard Deviation Channels Indicator
    
    Implements sophisticated channel analysis with dynamic boundaries,
    breakout detection, and comprehensive trend analysis.
    """
    
    def __init__(self, 
                 period: int = 20,
                 std_multiplier: float = 2.0,
                 adaptive_period: bool = True,
                 min_period: int = 10,
                 max_period: int = 50):
        """
        Initialize Standard Deviation Channels Indicator
        
        Args:
            period: Base period for channel calculation
            std_multiplier: Standard deviation multiplier for channel width
            adaptive_period: Whether to use adaptive period based on volatility
            min_period: Minimum period for adaptive calculation
            max_period: Maximum period for adaptive calculation
        """
        super().__init__()
        self.period = period
        self.std_multiplier = std_multiplier
        self.adaptive_period = adaptive_period
        self.min_period = min_period
        self.max_period = max_period
        self.logger = logging.getLogger(__name__)
        
    def calculate(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Calculate standard deviation channels
        
        Args:
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing channels analysis results
        """
        try:
            if data.empty or len(data) < self.min_period:
                raise IndicatorCalculationError("Insufficient data for channels analysis")
            
            prices = data['close'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            volumes = data['volume'].values if 'volume' in data.columns else np.ones(len(prices))
            
            # Calculate channels
            channels_result = self._calculate_channels_analysis(prices, high_prices, low_prices, volumes)
            
            # Generate trading signal
            signal = self._generate_signal(channels_result, prices)
            
            return {
                'signal': signal,
                'upper_channel': channels_result.upper_channel.tolist(),
                'lower_channel': channels_result.lower_channel.tolist(),
                'middle_line': channels_result.middle_line.tolist(),
                'channel_width': channels_result.channel_width.tolist(),
                'price_position': channels_result.price_position.tolist(),
                'breakout_signals': channels_result.breakout_signals.tolist(),
                'channel_slope': channels_result.channel_slope.tolist(),
                'volatility_regime': channels_result.volatility_regime.tolist(),
                'mean_reversion_strength': channels_result.mean_reversion_strength.tolist(),
                'trend_strength': channels_result.trend_strength.tolist(),
                'channel_efficiency': channels_result.channel_efficiency.tolist(),
                'support_resistance_levels': channels_result.support_resistance_levels,
                'adaptive_multiplier': channels_result.adaptive_multiplier.tolist(),
                'volume_weighted_position': channels_result.volume_weighted_position.tolist(),
                'strength': self._calculate_signal_strength(channels_result, prices),
                'confidence': self._calculate_confidence(channels_result, prices)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating standard deviation channels: {str(e)}")
            raise IndicatorCalculationError(f"Channels calculation failed: {str(e)}")
    
    def _calculate_channels_analysis(self, prices: np.ndarray, highs: np.ndarray, 
                                   lows: np.ndarray, volumes: np.ndarray) -> StandardDeviationChannelsResult:
        """Perform comprehensive channels analysis"""
        n = len(prices)
        
        # Adaptive period calculation
        adaptive_periods = self._calculate_adaptive_periods(prices) if self.adaptive_period else np.full(n, self.period)
        
        # Initialize arrays
        upper_channel = np.zeros(n)
        lower_channel = np.zeros(n)
        middle_line = np.zeros(n)
        channel_width = np.zeros(n)
        price_position = np.zeros(n)
        breakout_signals = np.zeros(n)
        channel_slope = np.zeros(n)
        volatility_regime = np.zeros(n)
        mean_reversion_strength = np.zeros(n)
        trend_strength = np.zeros(n)
        channel_efficiency = np.zeros(n)
        adaptive_multiplier = np.zeros(n)
        volume_weighted_position = np.zeros(n)
        
        # Calculate channels for each point
        for i in range(n):
            current_period = int(adaptive_periods[i])
            start_idx = max(0, i - current_period + 1)
            end_idx = i + 1
            
            if end_idx - start_idx < 3:
                continue
                
            # Current window data
            window_prices = prices[start_idx:end_idx]
            window_highs = highs[start_idx:end_idx]
            window_lows = lows[start_idx:end_idx]
            window_volumes = volumes[start_idx:end_idx]
            
            # Calculate middle line (linear regression)
            middle_line[i] = self._calculate_middle_line(window_prices, i)
            
            # Calculate standard deviation
            residuals = window_prices - np.mean(window_prices)
            std_dev = np.std(residuals, ddof=1)
            
            # Adaptive multiplier based on volatility regime
            adaptive_multiplier[i] = self._calculate_adaptive_multiplier(window_prices, std_dev)
            
            # Channel boundaries
            upper_channel[i] = middle_line[i] + adaptive_multiplier[i] * std_dev
            lower_channel[i] = middle_line[i] - adaptive_multiplier[i] * std_dev
            
            # Channel metrics
            channel_width[i] = upper_channel[i] - lower_channel[i]
            price_position[i] = self._calculate_price_position(prices[i], upper_channel[i], lower_channel[i])
            breakout_signals[i] = self._detect_breakout(prices, i, upper_channel[i], lower_channel[i])
            channel_slope[i] = self._calculate_channel_slope(middle_line, i)
            volatility_regime[i] = self._classify_volatility_regime(window_prices)
            mean_reversion_strength[i] = self._calculate_mean_reversion_strength(window_prices, middle_line[i])
            trend_strength[i] = self._calculate_trend_strength(window_prices)
            channel_efficiency[i] = self._calculate_channel_efficiency(window_prices, upper_channel[i], lower_channel[i])
            volume_weighted_position[i] = self._calculate_volume_weighted_position(
                window_prices, window_volumes, upper_channel[i], lower_channel[i]
            )
        
        # Support and resistance levels
        support_resistance = self._identify_support_resistance_levels(prices, upper_channel, lower_channel)
        
        return StandardDeviationChannelsResult(
            upper_channel=upper_channel,
            lower_channel=lower_channel,
            middle_line=middle_line,
            channel_width=channel_width,
            price_position=price_position,
            breakout_signals=breakout_signals,
            channel_slope=channel_slope,
            volatility_regime=volatility_regime,
            mean_reversion_strength=mean_reversion_strength,
            trend_strength=trend_strength,
            channel_efficiency=channel_efficiency,
            support_resistance_levels=support_resistance,
            adaptive_multiplier=adaptive_multiplier,
            volume_weighted_position=volume_weighted_position
        )
    
    def _calculate_adaptive_periods(self, prices: np.ndarray) -> np.ndarray:
        """Calculate adaptive periods based on volatility"""
        n = len(prices)
        periods = np.zeros(n)
        
        # Base volatility calculation
        returns = np.diff(np.log(prices + 1e-10))
        
        for i in range(20, n):
            # Recent volatility
            recent_vol = np.std(returns[i-20:i])
            
            # Historical volatility (longer window)
            if i >= 60:
                hist_vol = np.std(returns[i-60:i-20])
            else:
                hist_vol = recent_vol
            
            # Volatility ratio
            vol_ratio = recent_vol / (hist_vol + 1e-10)
            
            # Adaptive period: shorter in high volatility, longer in low volatility
            if vol_ratio > 1.5:  # High volatility
                periods[i] = self.min_period
            elif vol_ratio < 0.7:  # Low volatility
                periods[i] = self.max_period
            else:  # Normal volatility
                periods[i] = self.period
        
        # Fill initial values
        periods[:20] = self.period
        
        return periods
    
    def _calculate_middle_line(self, window_prices: np.ndarray, current_idx: int) -> float:
        """Calculate middle line using linear regression"""
        n = len(window_prices)
        if n < 2:
            return window_prices[-1]
        
        # Time index for regression
        x = np.arange(n).reshape(-1, 1)
        y = window_prices
        
        try:
            # Linear regression
            reg = LinearRegression().fit(x, y)
            
            # Predict current value
            middle_value = reg.predict([[n-1]])[0]
            
            return middle_value
        except:
            return np.mean(window_prices)
    
    def _calculate_adaptive_multiplier(self, window_prices: np.ndarray, base_std: float) -> float:
        """Calculate adaptive multiplier based on market conditions"""
        # Base multiplier
        multiplier = self.std_multiplier
        
        # Volatility adjustment
        returns = np.diff(np.log(window_prices + 1e-10))
        if len(returns) > 1:
            vol = np.std(returns)
            
            # Higher multiplier in high volatility periods
            if vol > 0.02:  # High volatility
                multiplier *= 1.2
            elif vol < 0.005:  # Low volatility
                multiplier *= 0.8
        
        # Trend strength adjustment
        if len(window_prices) >= 5:
            trend_slope = (window_prices[-1] - window_prices[0]) / len(window_prices)
            trend_strength = abs(trend_slope) / (np.mean(window_prices) + 1e-10)
            
            # Wider channels in strong trends
            if trend_strength > 0.001:
                multiplier *= (1 + trend_strength * 100)
        
        return max(0.5, min(multiplier, 4.0))  # Reasonable bounds
    
    def _calculate_price_position(self, price: float, upper: float, lower: float) -> float:
        """Calculate normalized price position within channel (0 to 1)"""
        if upper <= lower:
            return 0.5
        
        position = (price - lower) / (upper - lower)
        return max(0, min(position, 1))
    
    def _detect_breakout(self, prices: np.ndarray, idx: int, upper: float, lower: float) -> float:
        """Detect breakout signals"""
        if idx < 2:
            return 0
        
        current_price = prices[idx]
        prev_price = prices[idx-1]
        
        # Upward breakout
        if prev_price <= upper and current_price > upper:
            return 1  # Bullish breakout
        
        # Downward breakout
        elif prev_price >= lower and current_price < lower:
            return -1  # Bearish breakout
        
        return 0  # No breakout
    
    def _calculate_channel_slope(self, middle_line: np.ndarray, idx: int) -> float:
        """Calculate channel slope (trend direction)"""
        if idx < 5:
            return 0
        
        # Linear regression on recent middle line values
        recent_values = middle_line[max(0, idx-9):idx+1]
        n = len(recent_values)
        
        if n < 2:
            return 0
        
        x = np.arange(n)
        try:
            slope = np.polyfit(x, recent_values, 1)[0]
            return slope
        except:
            return 0
    
    def _classify_volatility_regime(self, window_prices: np.ndarray) -> float:
        """Classify volatility regime"""
        if len(window_prices) < 5:
            return 1  # Normal
        
        returns = np.diff(np.log(window_prices + 1e-10))
        current_vol = np.std(returns)
        
        # Thresholds for classification
        low_vol_threshold = 0.005
        high_vol_threshold = 0.025
        
        if current_vol < low_vol_threshold:
            return 0  # Low volatility
        elif current_vol > high_vol_threshold:
            return 2  # High volatility
        else:
            return 1  # Normal volatility
    
    def _calculate_mean_reversion_strength(self, window_prices: np.ndarray, middle_value: float) -> float:
        """Calculate mean reversion strength"""
        if len(window_prices) < 3:
            return 0
        
        # Distance from middle line
        current_distance = abs(window_prices[-1] - middle_value)
        avg_distance = np.mean(np.abs(window_prices - middle_value))
        
        if avg_distance == 0:
            return 0
        
        # Reversion strength based on relative distance
        reversion_strength = current_distance / avg_distance
        
        # Consider recent price movement direction
        if len(window_prices) >= 3:
            recent_move = window_prices[-1] - window_prices[-3]
            distance_move = (window_prices[-1] - middle_value) - (window_prices[-3] - middle_value)
            
            # If moving towards middle, increase reversion strength
            if (recent_move > 0 and distance_move < 0) or (recent_move < 0 and distance_move > 0):
                reversion_strength *= 1.5
        
        return min(reversion_strength, 3.0)
    
    def _calculate_trend_strength(self, window_prices: np.ndarray) -> float:
        """Calculate trend strength"""
        if len(window_prices) < 5:
            return 0
        
        # Linear trend correlation
        x = np.arange(len(window_prices))
        correlation = np.corrcoef(x, window_prices)[0, 1]
        
        # Trend strength as absolute correlation
        trend_strength = abs(correlation) if not np.isnan(correlation) else 0
        
        return trend_strength
    
    def _calculate_channel_efficiency(self, window_prices: np.ndarray, upper: float, lower: float) -> float:
        """Calculate channel efficiency (how well prices stay within bounds)"""
        if len(window_prices) < 3:
            return 0.5
        
        # Count prices within channel
        within_channel = np.sum((window_prices >= lower) & (window_prices <= upper))
        total_prices = len(window_prices)
        
        efficiency = within_channel / total_prices
        return efficiency
    
    def _calculate_volume_weighted_position(self, window_prices: np.ndarray, window_volumes: np.ndarray,
                                          upper: float, lower: float) -> float:
        """Calculate volume-weighted price position within channel"""
        if len(window_prices) != len(window_volumes) or len(window_prices) == 0:
            return 0.5
        
        if upper <= lower:
            return 0.5
        
        # Normalize positions
        positions = (window_prices - lower) / (upper - lower)
        positions = np.clip(positions, 0, 1)
        
        # Volume-weighted average position
        total_volume = np.sum(window_volumes)
        if total_volume == 0:
            return np.mean(positions)
        
        weighted_position = np.sum(positions * window_volumes) / total_volume
        return weighted_position
    
    def _identify_support_resistance_levels(self, prices: np.ndarray, upper_channel: np.ndarray,
                                          lower_channel: np.ndarray) -> Dict[str, List[float]]:
        """Identify key support and resistance levels from channels"""
        # Recent channel levels
        recent_window = 50
        start_idx = max(0, len(prices) - recent_window)
        
        recent_upper = upper_channel[start_idx:]
        recent_lower = lower_channel[start_idx:]
        recent_prices = prices[start_idx:]
        
        # Find local maxima and minima in channels
        support_levels = []
        resistance_levels = []
        
        # Channel boundaries that acted as support/resistance
        for i in range(2, len(recent_prices) - 2):
            price = recent_prices[i]
            upper = recent_upper[i]
            lower = recent_lower[i]
            
            # Check if price bounced off upper channel (resistance)
            if (price >= upper * 0.98 and 
                recent_prices[i-1] < price and recent_prices[i+1] < price):
                resistance_levels.append(upper)
            
            # Check if price bounced off lower channel (support)
            if (price <= lower * 1.02 and 
                recent_prices[i-1] > price and recent_prices[i+1] > price):
                support_levels.append(lower)
        
        # Remove duplicates and sort
        support_levels = sorted(list(set([round(level, 4) for level in support_levels])))
        resistance_levels = sorted(list(set([round(level, 4) for level in resistance_levels])))
        
        return {
            'support_levels': support_levels[-5:],  # Keep last 5
            'resistance_levels': resistance_levels[-5:]  # Keep last 5
        }
    
    def _generate_signal(self, result: StandardDeviationChannelsResult, prices: np.ndarray) -> SignalType:
        """Generate trading signal based on channel analysis"""
        if len(prices) < 3:
            return SignalType.HOLD
        
        current_price = prices[-1]
        current_position = result.price_position[-1]
        current_breakout = result.breakout_signals[-1]
        current_slope = result.channel_slope[-1]
        trend_strength = result.trend_strength[-1]
        mean_reversion_strength = result.mean_reversion_strength[-1]
        volatility_regime = result.volatility_regime[-1]
        
        # Breakout signals
        if current_breakout == 1:  # Upward breakout
            # Confirm with trend strength and volatility
            if trend_strength > 0.6 and volatility_regime != 0:  # Not in low volatility
                return SignalType.BUY
        elif current_breakout == -1:  # Downward breakout
            if trend_strength > 0.6 and volatility_regime != 0:
                return SignalType.SELL
        
        # Mean reversion signals
        if mean_reversion_strength > 1.5:
            # Price at extreme levels with high reversion probability
            if current_position >= 0.9:  # Near upper channel
                return SignalType.SELL
            elif current_position <= 0.1:  # Near lower channel
                return SignalType.BUY
        
        # Trend following signals
        if trend_strength > 0.7:
            if current_slope > 0 and current_position > 0.3:  # Uptrend, price above middle
                return SignalType.BUY
            elif current_slope < 0 and current_position < 0.7:  # Downtrend, price below middle
                return SignalType.SELL
        
        # Channel position signals
        if volatility_regime == 1:  # Normal volatility
            # Buy near support in uptrend
            if current_position < 0.3 and current_slope > 0:
                return SignalType.BUY
            # Sell near resistance in downtrend
            elif current_position > 0.7 and current_slope < 0:
                return SignalType.SELL
        
        return SignalType.HOLD
    
    def _calculate_signal_strength(self, result: StandardDeviationChannelsResult, prices: np.ndarray) -> float:
        """Calculate signal strength based on channel characteristics"""
        if len(prices) < 2:
            return 0.0
        
        current_position = result.price_position[-1]
        trend_strength = result.trend_strength[-1]
        channel_efficiency = result.channel_efficiency[-1]
        volatility_regime = result.volatility_regime[-1]
        
        # Position strength (extreme positions are stronger)
        position_strength = max(current_position, 1 - current_position) * 2 - 1
        position_strength = max(0, position_strength)
        
        # Trend strength
        trend_component = trend_strength
        
        # Channel quality
        efficiency_component = channel_efficiency
        
        # Volatility adjustment
        vol_adjustment = 1.0
        if volatility_regime == 0:  # Low volatility
            vol_adjustment = 0.7
        elif volatility_regime == 2:  # High volatility
            vol_adjustment = 1.2
        
        strength = (position_strength + trend_component + efficiency_component) / 3
        strength *= vol_adjustment
        
        return min(strength, 1.0)
    
    def _calculate_confidence(self, result: StandardDeviationChannelsResult, prices: np.ndarray) -> float:
        """Calculate confidence based on channel stability and consistency"""
        if len(prices) < 10:
            return 0.5
        
        # Channel width stability
        recent_widths = result.channel_width[-10:]
        width_stability = 1 / (1 + np.std(recent_widths) / (np.mean(recent_widths) + 1e-10))
        
        # Channel efficiency
        efficiency = np.mean(result.channel_efficiency[-10:])
        
        # Trend consistency
        recent_slopes = result.channel_slope[-10:]
        trend_consistency = 1 / (1 + np.std(recent_slopes))
        
        # Breakout confirmation
        recent_breakouts = result.breakout_signals[-5:]
        breakout_clarity = 1 if np.any(recent_breakouts != 0) else 0.7
        
        confidence = (width_stability + efficiency + trend_consistency + breakout_clarity) / 4
        return min(confidence, 1.0)