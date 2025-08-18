"""
Fractal Channel Indicator - Advanced Implementation

This indicator constructs dynamic channels using fractal highs and lows with
sophisticated mathematical models for adaptive width calculation and trend analysis.
Features include:
- Dynamic fractal channel construction
- Adaptive width calculation using volatility and fractal dimension
- Channel break detection and validation
- Trend strength analysis within channels
- Support/resistance level identification from channel boundaries

Mission: Supporting humanitarian trading platform for poor and sick children through
maximum profitability via advanced fractal channel analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RANSACRegressor
import warnings

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import IndicatorCalculationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChannelBoundary:
    """Channel boundary definition"""
    upper_line: np.ndarray
    lower_line: np.ndarray
    center_line: np.ndarray
    timestamps: np.ndarray
    slope: float
    r_squared: float
    channel_width: float

@dataclass
class FractalChannelResult:
    """Results container for Fractal Channel analysis"""
    channel_boundary: ChannelBoundary
    current_position: float  # Position within channel (0=bottom, 1=top)
    channel_strength: float
    trend_direction: str
    channel_break_signal: str
    support_level: float
    resistance_level: float
    channel_width_percentile: float
    trend_consistency: float
    fractal_dimension: float

class FractalChannelIndicator(StandardIndicatorInterface):
    """
    Advanced Fractal Channel Indicator
    
    Constructs dynamic channels using fractal highs and lows with sophisticated
    mathematical models for trend analysis and breakout detection.
    """
    
    def __init__(self, 
                 fractal_period: int = 5,
                 channel_lookback: int = 100,
                 min_fractal_points: int = 4,
                 width_multiplier: float = 2.0,
                 trend_threshold: float = 0.02,
                 break_threshold: float = 0.01):
        """
        Initialize the Fractal Channel Indicator
        
        Args:
            fractal_period: Period for fractal detection
            channel_lookback: Lookback period for channel construction
            min_fractal_points: Minimum fractal points for channel
            width_multiplier: Channel width multiplier
            trend_threshold: Minimum slope for trend detection
            break_threshold: Threshold for channel break detection
        """
        super().__init__()
        self.fractal_period = fractal_period
        self.channel_lookback = channel_lookback
        self.min_fractal_points = min_fractal_points
        self.width_multiplier = width_multiplier
        self.trend_threshold = trend_threshold
        self.break_threshold = break_threshold
        
        # Initialize caches
        self._channel_history = []
        self._fractal_cache = {}
        
        logger.info(f"Initialized FractalChannelIndicator with lookback={channel_lookback}")

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate fractal channel with comprehensive analysis
        
        Args:
            data: OHLCV DataFrame with required columns
            
        Returns:
            Dictionary containing fractal channel analysis
        """
        try:
            # Validate input data
            self._validate_data(data)
            
            if len(data) < self.channel_lookback:
                logger.warning(f"Insufficient data: {len(data)} < {self.channel_lookback}")
                return self._create_default_result()
            
            # Use recent data window
            recent_data = data[-self.channel_lookback:].copy()
            
            # Detect fractal points
            fractal_highs, fractal_lows = self._detect_fractal_points(recent_data)
            
            # Construct channel boundaries
            channel_boundary = self._construct_channel(recent_data, fractal_highs, fractal_lows)
            
            if channel_boundary is None:
                logger.warning("Failed to construct valid channel")
                return self._create_default_result()
            
            # Analyze current market position
            current_price = recent_data['close'].iloc[-1]
            position_analysis = self._analyze_position_in_channel(
                current_price, channel_boundary, recent_data
            )
            
            # Calculate channel characteristics
            channel_metrics = self._calculate_channel_metrics(
                channel_boundary, recent_data, fractal_highs, fractal_lows
            )
            
            # Generate signals
            signal_analysis = self._generate_channel_signals(
                recent_data, channel_boundary, position_analysis, channel_metrics
            )
            
            # Create comprehensive result
            result = FractalChannelResult(
                channel_boundary=channel_boundary,
                current_position=position_analysis['relative_position'],
                channel_strength=channel_metrics['strength'],
                trend_direction=channel_metrics['trend_direction'],
                channel_break_signal=signal_analysis['break_signal'],
                support_level=position_analysis['support_level'],
                resistance_level=position_analysis['resistance_level'],
                channel_width_percentile=channel_metrics['width_percentile'],
                trend_consistency=channel_metrics['trend_consistency'],
                fractal_dimension=channel_metrics['fractal_dimension']
            )
            
            # Update history
            self._update_channel_history(result)
            
            return self._format_output(result, data.index[-1])
            
        except Exception as e:
            logger.error(f"Error in fractal channel calculation: {e}")
            raise IndicatorCalculationError(f"FractalChannelIndicator calculation failed: {e}")

    def _detect_fractal_points(self, data: pd.DataFrame) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        Detect fractal high and low points
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Tuple of (fractal_highs, fractal_lows) as (index, price) pairs
        """
        try:
            highs = data['high'].values
            lows = data['low'].values
            fractal_highs = []
            fractal_lows = []
            
            half_period = self.fractal_period // 2
            
            for i in range(half_period, len(data) - half_period):
                # Fractal high detection
                is_fractal_high = True
                for j in range(i - half_period, i + half_period + 1):
                    if j != i and highs[j] >= highs[i]:
                        is_fractal_high = False
                        break
                
                if is_fractal_high:
                    fractal_highs.append((i, highs[i]))
                
                # Fractal low detection
                is_fractal_low = True
                for j in range(i - half_period, i + half_period + 1):
                    if j != i and lows[j] <= lows[i]:
                        is_fractal_low = False
                        break
                
                if is_fractal_low:
                    fractal_lows.append((i, lows[i]))
            
            return fractal_highs, fractal_lows
            
        except Exception as e:
            logger.warning(f"Fractal detection error: {e}")
            \1\n\n    \3construct_channel(self, data: pd.DataFrame, 
                          fractal_highs: List[Tuple[int, float]], 
                          fractal_lows: List[Tuple[int, float]]) -> Optional[ChannelBoundary]:
        """
        Construct channel using fractal points with robust regression
        
        Args:
            data: OHLCV DataFrame
            fractal_highs: List of fractal high points
            fractal_lows: List of fractal low points
            
        Returns:
            ChannelBoundary object or None if construction fails
        """
        try:
            if len(fractal_highs) < self.min_fractal_points or len(fractal_lows) < self.min_fractal_points:
                return None
            
            # Use recent fractal points for channel construction
            recent_highs = fractal_highs[-self.min_fractal_points*2:]
            recent_lows = fractal_lows[-self.min_fractal_points*2:]
            
            # Fit trend lines using RANSAC for robustness
            upper_line_params = self._fit_robust_trendline(recent_highs)
            lower_line_params = self._fit_robust_trendline(recent_lows)
            
            if upper_line_params is None or lower_line_params is None:
                return None
            
            # Generate channel lines for the entire data range
            x_indices = np.arange(len(data))
            
            upper_line = upper_line_params[0] * x_indices + upper_line_params[1]
            lower_line = lower_line_params[0] * x_indices + lower_line_params[1]
            center_line = (upper_line + lower_line) / 2
            
            # Adjust channel width based on volatility and fractal dimension
            adjusted_width = self._calculate_adaptive_width(data, upper_line, lower_line)
            width_adjustment = adjusted_width / np.mean(upper_line - lower_line)
            
            # Apply width adjustment
            center = center_line
            half_width = (upper_line - lower_line) / 2 * width_adjustment
            
            upper_line = center + half_width
            lower_line = center - half_width
            
            # Calculate channel quality metrics
            channel_slope = (upper_line_params[0] + lower_line_params[0]) / 2
            r_squared = self._calculate_channel_r_squared(data, upper_line, lower_line)
            channel_width = np.mean(upper_line - lower_line)
            
            return ChannelBoundary(
                upper_line=upper_line,
                lower_line=lower_line,
                center_line=center_line,
                timestamps=data.index.values,
                slope=channel_slope,
                r_squared=r_squared,
                channel_width=channel_width
            )
            
        except Exception as e:
            logger.warning(f"Channel construction error: {e}")
            return None

    def _fit_robust_trendline(self, points: List[Tuple[int, float]]) -> Optional[Tuple[float, float]]:
        """
        Fit robust trendline using RANSAC regression
        
        Args:
            points: List of (index, price) tuples
            
        Returns:
            Tuple of (slope, intercept) or None if fitting fails
        """
        try:
            if len(points) < 2:
                return None
            
            X = np.array([point[0] for point in points]).reshape(-1, 1)
            y = np.array([point[1] for point in points])
            
            # Use RANSAC for robust fitting
            ransac = RANSACRegressor(
                random_state=42,
                residual_threshold=np.std(y) * 0.5,
                max_trials=100
            )
            
            ransac.fit(X, y)
            
            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_
            
            return (slope, intercept)
            
        except Exception as e:
            logger.warning(f"Trendline fitting error: {e}")
            return None

    def _calculate_adaptive_width(self, data: pd.DataFrame, 
                                 upper_line: np.ndarray, 
                                 lower_line: np.ndarray) -> float:
        """
        Calculate adaptive channel width based on market conditions
        
        Args:
            data: OHLCV DataFrame
            upper_line: Upper channel line
            lower_line: Lower channel line
            
        Returns:
            Adaptive channel width
        """
        try:
            # Base width from initial channel
            base_width = np.mean(upper_line - lower_line)
            
            # Volatility adjustment
            returns = data['close'].pct_change().dropna()
            current_volatility = returns.std()
            historical_volatility = returns[-50:].std() if len(returns) >= 50 else current_volatility
            
            volatility_factor = current_volatility / (historical_volatility + 1e-10)
            volatility_adjustment = 1.0 + (volatility_factor - 1.0) * 0.5
            
            # Fractal dimension adjustment
            fractal_dim = self._calculate_price_fractal_dimension(data['close'].values)
            fractal_adjustment = fractal_dim / 1.5  # Normalize around 1.5
            
            # Combine adjustments
            adaptive_width = base_width * volatility_adjustment * fractal_adjustment * self.width_multiplier
            
            return adaptive_width
            
        except Exception as e:
            logger.warning(f"Adaptive width calculation error: {e}")
            return np.mean(upper_line - lower_line) * self.width_multiplier

    def _calculate_price_fractal_dimension(self, prices: np.ndarray) -> float:
        """
        Calculate fractal dimension of price series
        
        Args:
            prices: Price array
            
        Returns:
            Fractal dimension estimate
        """
        try:
            if len(prices) < 10:
                return 1.5
            
            # Normalize prices
            normalized_prices = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)
            
            # Box-counting method
            box_sizes = [2, 4, 8, 16]
            box_counts = []
            
            for box_size in box_sizes:
                if box_size >= len(prices):
                    continue
                
                # Create 2D embedding
                x_coords = np.arange(len(normalized_prices))
                y_coords = normalized_prices
                
                # Count occupied boxes
                x_boxes = (x_coords * box_size / len(x_coords)).astype(int)
                y_boxes = (y_coords * box_size).astype(int)
                
                unique_boxes = len(set(zip(x_boxes, y_boxes)))
                box_counts.append(unique_boxes)
            
            if len(box_counts) >= 2:
                log_sizes = np.log(box_sizes[:len(box_counts)])
                log_counts = np.log(box_counts)
                
                slope, _, _, _, _ = stats.linregress(log_sizes, log_counts)
                fractal_dim = -slope
                
                return np.clip(fractal_dim, 1.0, 2.0)
            
            return 1.5
            
        except Exception as e:
            logger.warning(f"Fractal dimension calculation error: {e}")
            \1\n\n    \3calculate_channel_r_squared(self, data: pd.DataFrame, 
                                   upper_line: np.ndarray, 
                                   lower_line: np.ndarray) -> float:
        """
        Calculate R-squared for channel fit quality
        
        Args:
            data: OHLCV DataFrame
            upper_line: Upper channel line
            lower_line: Lower channel line
            
        Returns:
            R-squared value
        """
        try:
            highs = data['high'].values
            lows = data['low'].values
            
            # Calculate deviations from channel boundaries
            upper_deviations = np.abs(highs - upper_line)
            lower_deviations = np.abs(lows - lower_line)
            
            # Combined fit quality
            total_variance = np.var(highs) + np.var(lows)
            residual_variance = np.var(upper_deviations) + np.var(lower_deviations)
            
            if total_variance > 0:
                r_squared = 1.0 - (residual_variance / total_variance)
                return np.clip(r_squared, 0.0, 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"R-squared calculation error: {e}")
            \1\n\n    \3analyze_position_in_channel(self, current_price: float, 
                                   channel: ChannelBoundary, 
                                   data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze current price position within the channel
        
        Args:
            current_price: Current market price
            channel: Channel boundary object
            data: OHLCV DataFrame
            
        Returns:
            Position analysis results
        """
        try:
            current_idx = len(data) - 1
            
            current_upper = channel.upper_line[current_idx]
            current_lower = channel.lower_line[current_idx]
            current_center = channel.center_line[current_idx]
            
            # Calculate relative position (0 = bottom, 1 = top)
            if current_upper != current_lower:
                relative_position = (current_price - current_lower) / (current_upper - current_lower)
                relative_position = np.clip(relative_position, 0.0, 1.0)
            else:
                relative_position = 0.5
            
            # Determine support and resistance levels
            support_level = current_lower
            resistance_level = current_upper
            
            # Distance from center
            center_distance = abs(current_price - current_center) / current_center
            
            return {
                'relative_position': relative_position,
                'support_level': support_level,
                'resistance_level': resistance_level,
                'center_distance': center_distance,
                'current_upper': current_upper,
                'current_lower': current_lower,
                'current_center': current_center
            }
            
        except Exception as e:
            logger.warning(f"Position analysis error: {e}")
            return {
                'relative_position': 0.5,
                'support_level': current_price * 0.98,
                'resistance_level': current_price * 1.02,
                'center_distance': 0.0,
                'current_upper': current_price * 1.02,
                'current_lower': current_price * 0.98,
                'current_center': current_price
            }

    def _calculate_channel_metrics(self, channel: ChannelBoundary, 
                                 data: pd.DataFrame,
                                 fractal_highs: List[Tuple[int, float]], 
                                 fractal_lows: List[Tuple[int, float]]) -> Dict[str, Any]:
        """
        Calculate comprehensive channel metrics
        
        Args:
            channel: Channel boundary object
            data: OHLCV DataFrame
            fractal_highs: Fractal high points
            fractal_lows: Fractal low points
            
        Returns:
            Channel metrics dictionary
        """
        try:
            # Channel strength based on R-squared and fractal alignment
            strength = channel.r_squared
            
            # Trend direction based on channel slope
            if abs(channel.slope) > self.trend_threshold:
                trend_direction = "UPTREND" if channel.slope > 0 else "DOWNTREND"
            else:
                trend_direction = "SIDEWAYS"
            
            # Channel width percentile
            recent_widths = []
            for i in range(max(0, len(data) - 50), len(data)):
                width = channel.upper_line[i] - channel.lower_line[i]
                recent_widths.append(width)
            
            if recent_widths:
                current_width = recent_widths[-1]
                width_percentile = stats.percentileofscore(recent_widths, current_width) / 100.0
            else:
                width_percentile = 0.5
            
            # Trend consistency
            price_touches_upper = 0
            price_touches_lower = 0
            total_periods = len(data)
            
            for i in range(total_periods):
                if data['high'].iloc[i] >= channel.upper_line[i] * 0.995:  # Within 0.5%
                    price_touches_upper += 1
                if data['low'].iloc[i] <= channel.lower_line[i] * 1.005:  # Within 0.5%
                    price_touches_lower += 1
            
            touch_balance = 1.0 - abs((price_touches_upper - price_touches_lower) / total_periods)
            trend_consistency = channel.r_squared * touch_balance
            
            # Fractal dimension of channel
            channel_fractal_dim = self._calculate_price_fractal_dimension(data['close'].values)
            
            return {
                'strength': strength,
                'trend_direction': trend_direction,
                'width_percentile': width_percentile,
                'trend_consistency': trend_consistency,
                'fractal_dimension': channel_fractal_dim,
                'slope_magnitude': abs(channel.slope),
                'touches_upper': price_touches_upper,
                'touches_lower': price_touches_lower
            }
            
        except Exception as e:
            logger.warning(f"Channel metrics calculation error: {e}")
            return {
                'strength': 0.5,
                'trend_direction': 'UNKNOWN',
                'width_percentile': 0.5,
                'trend_consistency': 0.5,
                'fractal_dimension': 1.5,
                'slope_magnitude': 0.0,
                'touches_upper': 0,
                'touches_lower': 0
            }

    def _generate_channel_signals(self, data: pd.DataFrame, 
                                channel: ChannelBoundary,
                                position_analysis: Dict[str, Any],
                                channel_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on channel analysis
        
        Args:
            data: OHLCV DataFrame
            channel: Channel boundary object
            position_analysis: Position analysis results
            channel_metrics: Channel metrics
            
        Returns:
            Signal analysis results
        """
        try:
            current_price = data['close'].iloc[-1]
            relative_position = position_analysis['relative_position']
            
            # Channel break detection
            break_signal = "NO_BREAK"
            
            # Check for breaks above upper channel
            if current_price > position_analysis['current_upper'] * (1 + self.break_threshold):
                break_signal = "UPWARD_BREAK"
            # Check for breaks below lower channel
            elif current_price < position_analysis['current_lower'] * (1 - self.break_threshold):
                break_signal = "DOWNWARD_BREAK"
            # Check for position-based signals
            elif relative_position > 0.8:
                break_signal = "APPROACHING_UPPER"
            elif relative_position < 0.2:
                break_signal = "APPROACHING_LOWER"
            
            return {
                'break_signal': break_signal,
                'signal_strength': channel_metrics['strength'],
                'trend_alignment': channel_metrics['trend_direction'],
                'position_extreme': relative_position > 0.8 or relative_position < 0.2
            }
            
        except Exception as e:
            logger.warning(f"Signal generation error: {e}")
            return {
                'break_signal': 'ERROR',
                'signal_strength': 0.0,
                'trend_alignment': 'UNKNOWN',
                'position_extreme': False
            }    def _update_channel_history(self, result: FractalChannelResult) -> None:
        """
        Update channel history for trend analysis
        
        Args:
            result: Current channel result
        """
        try:
            self._channel_history.append({
                'timestamp': pd.Timestamp.now(),
                'channel_strength': result.channel_strength,
                'trend_direction': result.trend_direction,
                'channel_width': result.channel_boundary.channel_width,
                'position': result.current_position
            })
            
            # Keep last 100 records
            if len(self._channel_history) > 100:
                self._channel_history.pop(0)
                
        except Exception as e:
            logger.warning(f"Channel history update error: {e}")

    def _format_output(self, result: FractalChannelResult, timestamp) -> Dict[str, Any]:
        """
        Format the calculation results for output
        
        Args:
            result: Fractal channel calculation results
            timestamp: Current timestamp
            
        Returns:
            Formatted output dictionary
        """
        return {
            'timestamp': timestamp,
            'indicator_name': 'FractalChannel',
            
            # Channel boundaries
            'upper_channel': round(result.resistance_level, 6),
            'lower_channel': round(result.support_level, 6),
            'center_channel': round((result.resistance_level + result.support_level) / 2, 6),
            'channel_width': round(result.channel_boundary.channel_width, 6),
            
            # Position analysis
            'current_position': round(result.current_position, 4),
            'position_description': self._describe_position(result.current_position),
            
            # Channel characteristics
            'channel_strength': round(result.channel_strength, 4),
            'trend_direction': result.trend_direction,
            'channel_slope': round(result.channel_boundary.slope, 8),
            'r_squared': round(result.channel_boundary.r_squared, 4),
            
            # Signals
            'channel_break_signal': result.channel_break_signal,
            'support_level': round(result.support_level, 6),
            'resistance_level': round(result.resistance_level, 6),
            
            # Additional metrics
            'channel_width_percentile': round(result.channel_width_percentile, 4),
            'trend_consistency': round(result.trend_consistency, 4),
            'fractal_dimension': round(result.fractal_dimension, 4),
            
            # Trading insights
            'channel_quality': self._assess_channel_quality(result),
            'breakout_probability': self._estimate_breakout_probability(result),
            'position_risk': self._assess_position_risk(result),
            
            # Historical context
            'channel_stability': self._assess_channel_stability(),
            'trend_persistence': self._assess_trend_persistence()
        }

    def _describe_position(self, position: float) -> str:
        """
        Describe current position within channel
        
        Args:
            position: Relative position (0-1)
            
        Returns:
            Position description string
        """
        if position < 0.2:
            return "NEAR_LOWER_BOUNDARY"
        elif position < 0.4:
            return "LOWER_HALF"
        elif position < 0.6:
            return "CENTER_REGION"
        elif position < 0.8:
            return "UPPER_HALF"
        else:
            \1\n\n    \3assess_channel_quality(self, result: FractalChannelResult) -> str:
        """
        Assess overall channel quality
        
        Args:
            result: Channel analysis result
            
        Returns:
            Quality assessment string
        """
        try:
            quality_score = (result.channel_strength + result.trend_consistency) / 2
            
            if quality_score > 0.7:
                return "HIGH_QUALITY"
            elif quality_score > 0.5:
                return "MEDIUM_QUALITY"
            elif quality_score > 0.3:
                return "LOW_QUALITY"
            else:
                return "POOR_QUALITY"
                
        except Exception as e:
            logger.warning(f"Channel quality assessment error: {e}")
            \1\n\n    \3estimate_breakout_probability(self, result: FractalChannelResult) -> float:
        """
        Estimate probability of channel breakout
        
        Args:
            result: Channel analysis result
            
        Returns:
            Breakout probability (0-1)
        """
        try:
            # Higher probability near boundaries with strong trend
            position_factor = 0.0
            if result.current_position > 0.8:
                position_factor = (result.current_position - 0.8) / 0.2
            elif result.current_position < 0.2:
                position_factor = (0.2 - result.current_position) / 0.2
            
            # Trend strength factor
            trend_factor = 0.0
            if result.trend_direction in ['UPTREND', 'DOWNTREND']:
                trend_factor = result.trend_consistency
            
            # Channel width factor (wider channels = higher breakout probability)
            width_factor = result.channel_width_percentile
            
            # Combine factors
            breakout_probability = (position_factor * 0.4 + 
                                  trend_factor * 0.4 + 
                                  width_factor * 0.2)
            
            return np.clip(breakout_probability, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Breakout probability estimation error: {e}")
            \1\n\n    \3assess_position_risk(self, result: FractalChannelResult) -> str:
        """
        Assess risk based on current position
        
        Args:
            result: Channel analysis result
            
        Returns:
            Risk assessment string
        """
        try:
            # Risk is higher near boundaries, especially with weak channels
            boundary_risk = max(result.current_position, 1 - result.current_position)
            channel_risk = 1 - result.channel_strength
            
            combined_risk = (boundary_risk + channel_risk) / 2
            
            if combined_risk > 0.7:
                return "HIGH_RISK"
            elif combined_risk > 0.5:
                return "MEDIUM_RISK"
            else:
                return "LOW_RISK"
                
        except Exception as e:
            logger.warning(f"Position risk assessment error: {e}")
            \1\n\n    \3assess_channel_stability(self) -> str:
        """
        Assess channel stability over time
        
        Returns:
            Stability assessment string
        """
        try:
            if len(self._channel_history) < 10:
                return "INSUFFICIENT_DATA"
            
            recent_strengths = [record['channel_strength'] for record in self._channel_history[-10:]]
            strength_stability = 1.0 - np.std(recent_strengths)
            
            if strength_stability > 0.8:
                return "VERY_STABLE"
            elif strength_stability > 0.6:
                return "STABLE"
            elif strength_stability > 0.4:
                return "MODERATELY_STABLE"
            else:
                return "UNSTABLE"
                
        except Exception as e:
            logger.warning(f"Channel stability assessment error: {e}")
            \1\n\n    \3assess_trend_persistence(self) -> str:
        """
        Assess trend persistence over time
        
        Returns:
            Trend persistence assessment string
        """
        try:
            if len(self._channel_history) < 10:
                return "INSUFFICIENT_DATA"
            
            recent_trends = [record['trend_direction'] for record in self._channel_history[-10:]]
            
            # Count trend changes
            trend_changes = 0
            for i in range(1, len(recent_trends)):
                if recent_trends[i] != recent_trends[i-1]:
                    trend_changes += 1
            
            change_rate = trend_changes / len(recent_trends)
            
            if change_rate < 0.2:
                return "VERY_PERSISTENT"
            elif change_rate < 0.4:
                return "PERSISTENT"
            elif change_rate < 0.6:
                return "MODERATELY_PERSISTENT"
            else:
                return "VOLATILE"
                
        except Exception as e:
            logger.warning(f"Trend persistence assessment error: {e}")
            \1\n\n    \3validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data for required columns and quality
        
        Args:
            data: Input DataFrame to validate
            
        Raises:
            IndicatorCalculationError: If data validation fails
        """
        required_columns = ['high', 'low', 'close', 'open']
        
        if not all(col in data.columns for col in required_columns):
            raise IndicatorCalculationError(f"Missing required columns: {required_columns}")
        
        if len(data) < self.channel_lookback:
            raise IndicatorCalculationError(f"Insufficient data: minimum {self.channel_lookback} periods required")
        
        # Check for invalid values
        for col in required_columns:
            if data[col].isnull().any():
                raise IndicatorCalculationError(f"Null values found in {col}")
            if (data[col] <= 0).any():
                raise IndicatorCalculationError(f"Non-positive values found in {col}")

    def _create_default_result(self) -> Dict[str, Any]:
        """
        Create default result for insufficient data cases
        
        Returns:
            Default result dictionary
        """
        return {
            'timestamp': pd.Timestamp.now(),
            'indicator_name': 'FractalChannel',
            'upper_channel': 0.0,
            'lower_channel': 0.0,
            'center_channel': 0.0,
            'channel_width': 0.0,
            'current_position': 0.5,
            'position_description': 'INSUFFICIENT_DATA',
            'channel_strength': 0.0,
            'trend_direction': 'UNKNOWN',
            'channel_slope': 0.0,
            'r_squared': 0.0,
            'channel_break_signal': 'INSUFFICIENT_DATA',
            'support_level': 0.0,
            'resistance_level': 0.0,
            'channel_width_percentile': 0.5,
            'trend_consistency': 0.0,
            'fractal_dimension': 1.5,
            'channel_quality': 'INSUFFICIENT_DATA',
            'breakout_probability': 0.5,
            'position_risk': 'UNKNOWN',
            'channel_stability': 'INSUFFICIENT_DATA',
            'trend_persistence': 'INSUFFICIENT_DATA'
        }

    def get_required_columns(self) -> List[str]:
        """
        Get list of required data columns
        
        Returns:
            List of required column names
        """
        return ['high', 'low', 'close', 'open']

    def get_indicator_name(self) -> str:
        """
        Get the indicator name
        
        Returns:
            Indicator name string
        """
        return "FractalChannel"