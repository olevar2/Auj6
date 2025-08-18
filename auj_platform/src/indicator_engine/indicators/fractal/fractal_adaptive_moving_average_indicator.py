"""
Fractal Adaptive Moving Average Indicator - Advanced Implementation

This indicator implements a sophisticated adaptive moving average that adjusts its
smoothing parameters based on fractal efficiency and Hurst exponent analysis.
Features include:
- Fractal efficiency ratio calculation
- Hurst exponent estimation for memory analysis
- Dynamic smoothing factor adaptation
- Fractal dimension-based trend detection
- Advanced noise filtering using fractal geometry

Mission: Supporting humanitarian trading platform for poor and sick children through
maximum profitability via advanced fractal adaptive analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import MinMaxScaler
import warnings

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import IndicatorCalculationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FractalAMAResult:
    """Results container for Fractal Adaptive Moving Average calculations"""
    adaptive_ma: float
    efficiency_ratio: float
    hurst_exponent: float
    fractal_dimension: float
    smoothing_factor: float
    trend_strength: float
    noise_level: float
    adaptation_speed: float
    signal_quality: float
    regime_classification: str

class FractalAdaptiveMovingAverageIndicator(StandardIndicatorInterface):
    """
    Advanced Fractal Adaptive Moving Average Indicator
    
    Combines Kaufman's Adaptive Moving Average concepts with fractal geometry
    and Hurst exponent analysis for superior trend following and noise reduction.
    """
    
    def __init__(self, 
                 period: int = 30,
                 fast_sc: float = 2.0,
                 slow_sc: float = 30.0,
                 hurst_window: int = 100,
                 efficiency_window: int = 20,
                 fractal_window: int = 50,
                 min_smoothing: float = 0.1,
                 max_smoothing: float = 1.0,
                 adaptation_sensitivity: float = 2.0):
        """
        Initialize the Fractal Adaptive Moving Average Indicator
        
        Args:
            period: Base period for calculations
            fast_sc: Fast smoothing constant
            slow_sc: Slow smoothing constant
            hurst_window: Window for Hurst exponent calculation
            efficiency_window: Window for efficiency ratio calculation
            fractal_window: Window for fractal dimension analysis
            min_smoothing: Minimum smoothing factor
            max_smoothing: Maximum smoothing factor
            adaptation_sensitivity: Sensitivity of adaptation mechanism
        """
        super().__init__()
        self.period = period
        self.fast_sc = fast_sc
        self.slow_sc = slow_sc
        self.hurst_window = hurst_window
        self.efficiency_window = efficiency_window
        self.fractal_window = fractal_window
        self.min_smoothing = min_smoothing
        self.max_smoothing = max_smoothing
        self.adaptation_sensitivity = adaptation_sensitivity
        
        # Calculate smoothing constants
        self.fast_alpha = 2.0 / (fast_sc + 1.0)
        self.slow_alpha = 2.0 / (slow_sc + 1.0)
        
        # Initialize calculation cache
        self._previous_ama = None
        self._efficiency_history = []
        self._hurst_history = []
        self._fractal_history = []
        
        logger.info(f"Initialized FractalAdaptiveMovingAverageIndicator with period={period}")

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate fractal adaptive moving average with comprehensive analysis
        
        Args:
            data: OHLCV DataFrame with required columns
            
        Returns:
            Dictionary containing adaptive MA and fractal analysis
        """
        try:
            # Validate input data
            self._validate_data(data)
            
            # Prepare price series
            prices = self._prepare_price_series(data)
            
            if len(prices) < max(self.period, self.hurst_window):
                logger.warning(f"Insufficient data: {len(prices)} < {max(self.period, self.hurst_window)}")
                return self._create_default_result()
            
            # Calculate efficiency ratio
            efficiency_ratio = self._calculate_efficiency_ratio(prices)
            
            # Calculate Hurst exponent
            hurst_exponent = self._calculate_hurst_exponent(prices)
            
            # Calculate fractal dimension
            fractal_dimension = self._calculate_fractal_dimension(prices)
            
            # Calculate adaptive smoothing factor
            smoothing_factor = self._calculate_adaptive_smoothing(
                efficiency_ratio, hurst_exponent, fractal_dimension
            )
            
            # Calculate adaptive moving average
            current_price = prices.iloc[-1]
            adaptive_ma = self._calculate_adaptive_ma(current_price, smoothing_factor)
            
            # Analyze trend and signal quality
            trend_analysis = self._analyze_trend_characteristics(
                prices, adaptive_ma, efficiency_ratio, hurst_exponent
            )
            
            # Create comprehensive result
            result = FractalAMAResult(
                adaptive_ma=adaptive_ma,
                efficiency_ratio=efficiency_ratio,
                hurst_exponent=hurst_exponent,
                fractal_dimension=fractal_dimension,
                smoothing_factor=smoothing_factor,
                trend_strength=trend_analysis['trend_strength'],
                noise_level=trend_analysis['noise_level'],
                adaptation_speed=trend_analysis['adaptation_speed'],
                signal_quality=trend_analysis['signal_quality'],
                regime_classification=trend_analysis['regime_classification']
            )
            
            # Update historical data
            self._update_history(result)
            
            return self._format_output(result, data.index[-1])
            
        except Exception as e:
            logger.error(f"Error in fractal adaptive MA calculation: {e}")
            raise IndicatorCalculationError(f"FractalAdaptiveMovingAverageIndicator calculation failed: {e}")

    def _calculate_efficiency_ratio(self, prices: pd.Series) -> float:
        """
        Calculate enhanced efficiency ratio with fractal considerations
        
        Args:
            prices: Price time series
            
        Returns:
            Fractal-enhanced efficiency ratio
        """
        try:
            if len(prices) < self.efficiency_window:
                return 0.5
            
            recent_prices = prices[-self.efficiency_window:]
            
            # Direction (net price change)
            direction = abs(recent_prices.iloc[-1] - recent_prices.iloc[0])
            
            # Volatility (sum of absolute changes)
            volatility = recent_prices.diff().abs().sum()
            
            if volatility == 0:
                return 0.0
            
            # Basic efficiency ratio
            basic_er = direction / volatility
            
            # Fractal enhancement: consider path complexity
            path_complexity = self._calculate_path_complexity(recent_prices)
            fractal_adjustment = 1.0 / (1.0 + path_complexity)
            
            # Enhanced efficiency ratio
            enhanced_er = basic_er * fractal_adjustment
            
            return np.clip(enhanced_er, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Efficiency ratio calculation error: {e}")
            return 0.5

    def _calculate_hurst_exponent(self, prices: pd.Series) -> float:
        """
        Calculate Hurst exponent using R/S analysis
        
        Args:
            prices: Price time series
            
        Returns:
            Hurst exponent (0.5 = random walk, >0.5 = persistent, <0.5 = anti-persistent)
        """
        try:
            if len(prices) < self.hurst_window:
                return 0.5
            
            recent_prices = prices[-self.hurst_window:]
            log_returns = np.log(recent_prices / recent_prices.shift(1)).dropna()
            
            if len(log_returns) < 20:
                return 0.5
            
            # Calculate mean-adjusted cumulative sum
            mean_return = log_returns.mean()
            cumulative_deviance = np.cumsum(log_returns - mean_return)
            
            # Define time scales for R/S analysis
            scales = np.unique(np.logspace(1, np.log10(len(log_returns)//4), num=15).astype(int))
            rs_values = []
            
            for scale in scales:
                if scale >= len(cumulative_deviance):
                    continue
                
                # Partition into non-overlapping segments
                n_segments = len(cumulative_deviance) // scale
                rs_segment_values = []
                
                for i in range(n_segments):
                    start_idx = i * scale
                    end_idx = (i + 1) * scale
                    
                    segment_cumdev = cumulative_deviance[start_idx:end_idx]
                    segment_returns = log_returns[start_idx:end_idx]
                    
                    if len(segment_cumdev) == 0:
                        continue
                    
                    # Range of cumulative deviance
                    R = segment_cumdev.max() - segment_cumdev.min()
                    
                    # Standard deviation of returns
                    S = segment_returns.std()
                    
                    if S > 0:
                        rs_segment_values.append(R / S)
                
                if rs_segment_values:
                    rs_values.append(np.mean(rs_segment_values))
            
            if len(rs_values) < 3:
                return 0.5
            
            # Linear regression in log-log space
            log_scales = np.log(scales[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Remove any infinite or NaN values
            valid_mask = np.isfinite(log_scales) & np.isfinite(log_rs)
            if not np.any(valid_mask):
                return 0.5
            
            slope, _, _, _, _ = stats.linregress(log_scales[valid_mask], log_rs[valid_mask])
            
            # Hurst exponent is the slope
            hurst = slope
            
            # Ensure reasonable bounds
            hurst = np.clip(hurst, 0.1, 0.9)
            
            return hurst
            
        except Exception as e:
            logger.warning(f"Hurst exponent calculation error: {e}")
            return 0.5

    def _calculate_fractal_dimension(self, prices: pd.Series) -> float:
        """
        Calculate fractal dimension using box-counting method
        
        Args:
            prices: Price time series
            
        Returns:
            Fractal dimension
        """
        try:
            if len(prices) < self.fractal_window:
                return 1.5
            
            recent_prices = prices[-self.fractal_window:].values
            
            # Normalize prices to [0,1] range
            normalized_prices = (recent_prices - recent_prices.min()) / (recent_prices.max() - recent_prices.min() + 1e-10)
            
            # Define box sizes
            min_box_size = 2
            max_box_size = min(20, len(normalized_prices) // 4)
            box_sizes = np.unique(np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=10).astype(int))
            
            box_counts = []
            
            for box_size in box_sizes:
                # Create 2D embedding
                if len(normalized_prices) < box_size * 2:
                    continue
                
                # Simple 2D embedding: (price[i], price[i+1])
                embedding_x = normalized_prices[:-1]
                embedding_y = normalized_prices[1:]
                
                # Count occupied boxes
                x_boxes = np.floor(embedding_x * box_size).astype(int)
                y_boxes = np.floor(embedding_y * box_size).astype(int)
                
                # Clip to valid range
                x_boxes = np.clip(x_boxes, 0, box_size - 1)
                y_boxes = np.clip(y_boxes, 0, box_size - 1)
                
                # Count unique boxes
                unique_boxes = set(zip(x_boxes, y_boxes))
                box_counts.append(len(unique_boxes))
            
            if len(box_counts) < 3:
                return 1.5
            
            # Linear regression in log-log space
            log_box_sizes = np.log(box_sizes[:len(box_counts)])
            log_box_counts = np.log(box_counts)
            
            valid_mask = np.isfinite(log_box_sizes) & np.isfinite(log_box_counts)
            if not np.any(valid_mask):
                return 1.5
            
            slope, _, _, _, _ = stats.linregress(log_box_sizes[valid_mask], log_box_counts[valid_mask])
            
            # Fractal dimension is negative slope
            fractal_dimension = -slope
            
            # Ensure reasonable bounds
            fractal_dimension = np.clip(fractal_dimension, 1.0, 2.0)
            
            return fractal_dimension
            
        except Exception as e:
            logger.warning(f"Fractal dimension calculation error: {e}")
            return 1.5

    def _calculate_path_complexity(self, prices: pd.Series) -> float:
        """
        Calculate path complexity using fractal geometry concepts
        
        Args:
            prices: Price time series
            
        Returns:
            Path complexity measure
        """
        try:
            if len(prices) < 3:
                return 1.0
            
            # Calculate path length (sum of absolute changes)
            path_length = prices.diff().abs().sum()
            
            # Calculate Euclidean distance (direct path)
            euclidean_distance = abs(prices.iloc[-1] - prices.iloc[0])
            
            if euclidean_distance == 0:
                return 10.0  # Maximum complexity for flat price
            
            # Complexity ratio
            complexity = path_length / euclidean_distance
            
            return complexity
            
        except Exception as e:
            logger.warning(f"Path complexity calculation error: {e}")
            return 1.0

    def _calculate_adaptive_smoothing(self, efficiency_ratio: float, 
                                    hurst_exponent: float, 
                                    fractal_dimension: float) -> float:
        """
        Calculate adaptive smoothing factor based on multiple fractal metrics
        
        Args:
            efficiency_ratio: Market efficiency measure
            hurst_exponent: Memory persistence measure
            fractal_dimension: Complexity measure
            
        Returns:
            Adaptive smoothing factor
        """
        try:
            # Base smoothing from efficiency ratio (Kaufman's method)
            sc_base = efficiency_ratio * (self.fast_alpha - self.slow_alpha) + self.slow_alpha
            base_smoothing = sc_base * sc_base
            
            # Hurst adjustment: stronger smoothing for anti-persistent markets
            if hurst_exponent < 0.5:
                # Anti-persistent (mean-reverting) - use more smoothing
                hurst_adjustment = 1.0 + (0.5 - hurst_exponent) * 2.0
            else:
                # Persistent (trending) - use less smoothing
                hurst_adjustment = 1.0 - (hurst_exponent - 0.5) * 0.5
            
            # Fractal dimension adjustment: more complex markets need more smoothing
            fractal_adjustment = fractal_dimension / 1.5  # Normalize around 1.5
            
            # Combine adjustments
            adaptive_smoothing = base_smoothing * hurst_adjustment * fractal_adjustment
            
            # Apply sensitivity scaling
            adaptive_smoothing = adaptive_smoothing ** (1.0 / self.adaptation_sensitivity)
            
            # Ensure bounds
            adaptive_smoothing = np.clip(adaptive_smoothing, self.min_smoothing, self.max_smoothing)
            
            return adaptive_smoothing
            
        except Exception as e:
            logger.warning(f"Adaptive smoothing calculation error: {e}")
            return 0.5

    def _calculate_adaptive_ma(self, current_price: float, smoothing_factor: float) -> float:
        """
        Calculate the adaptive moving average value
        
        Args:
            current_price: Current price value
            smoothing_factor: Adaptive smoothing factor
            
        Returns:
            New adaptive moving average value
        """
        try:
            if self._previous_ama is None:
                # Initialize with current price
                self._previous_ama = current_price
                return current_price
            
            # Calculate new AMA value
            new_ama = self._previous_ama + smoothing_factor * (current_price - self._previous_ama)
            
            # Update stored value
            self._previous_ama = new_ama
            
            return new_ama
            
        except Exception as e:
            logger.warning(f"Adaptive MA calculation error: {e}")
            return current_price

    def _analyze_trend_characteristics(self, prices: pd.Series, ama_value: float,
                                     efficiency_ratio: float, hurst_exponent: float) -> Dict[str, Any]:
        """
        Analyze trend characteristics and signal quality
        
        Args:
            prices: Price time series
            ama_value: Current adaptive moving average value
            efficiency_ratio: Current efficiency ratio
            hurst_exponent: Current Hurst exponent
            
        Returns:
            Dictionary containing trend analysis results
        """
        try:
            current_price = prices.iloc[-1]
            
            # Trend strength based on price vs AMA and efficiency
            price_distance = abs(current_price - ama_value) / ama_value
            trend_strength = efficiency_ratio * (1.0 + price_distance)
            trend_strength = np.clip(trend_strength, 0.0, 1.0)
            
            # Noise level based on recent volatility and fractal characteristics
            recent_returns = prices.pct_change().dropna()[-20:]
            if len(recent_returns) > 0:
                noise_level = recent_returns.std() / abs(recent_returns.mean() + 1e-10)
                noise_level = np.clip(noise_level / 10.0, 0.0, 1.0)  # Normalize
            else:
                noise_level = 0.5
            
            # Adaptation speed based on efficiency and Hurst
            adaptation_speed = efficiency_ratio * (2.0 - hurst_exponent)
            adaptation_speed = np.clip(adaptation_speed, 0.0, 1.0)
            
            # Signal quality combining multiple factors
            signal_quality = (efficiency_ratio + (1.0 - noise_level) + trend_strength) / 3.0
            signal_quality = np.clip(signal_quality, 0.0, 1.0)
            
            # Regime classification
            regime_classification = self._classify_market_regime(
                efficiency_ratio, hurst_exponent, trend_strength, noise_level
            )
            
            return {
                'trend_strength': trend_strength,
                'noise_level': noise_level,
                'adaptation_speed': adaptation_speed,
                'signal_quality': signal_quality,
                'regime_classification': regime_classification
            }
            
        except Exception as e:
            logger.warning(f"Trend analysis error: {e}")
            return {
                'trend_strength': 0.5,
                'noise_level': 0.5,
                'adaptation_speed': 0.5,
                'signal_quality': 0.5,
                'regime_classification': 'UNKNOWN'
            }

    def _classify_market_regime(self, efficiency_ratio: float, hurst_exponent: float,
                              trend_strength: float, noise_level: float) -> str:
        """
        Classify market regime based on fractal characteristics
        
        Args:
            efficiency_ratio: Market efficiency measure
            hurst_exponent: Memory persistence measure
            trend_strength: Trend strength measure
            noise_level: Market noise level
            
        Returns:
            Market regime classification string
        """
        try:
            # Define regime classification rules
            if efficiency_ratio > 0.7 and trend_strength > 0.6:
                if hurst_exponent > 0.6:
                    return "STRONG_TRENDING"
                else:
                    return "MODERATE_TRENDING"
            elif efficiency_ratio < 0.3 or noise_level > 0.7:
                if hurst_exponent < 0.4:
                    return "MEAN_REVERTING"
                else:
                    return "CHOPPY_RANGING"
            elif 0.45 <= hurst_exponent <= 0.55:
                return "RANDOM_WALK"
            elif hurst_exponent > 0.6:
                return "PERSISTENT_DRIFT"
            elif hurst_exponent < 0.4:
                return "ANTI_PERSISTENT"
            else:
                return "TRANSITIONAL"
                
        except Exception as e:
            logger.warning(f"Regime classification error: {e}")
            return "UNKNOWN"    def _update_history(self, result: FractalAMAResult) -> None:
        """
        Update historical data for trend analysis
        
        Args:
            result: Current calculation result
        """
        # Update efficiency history
        self._efficiency_history.append(result.efficiency_ratio)
        if len(self._efficiency_history) > 100:
            self._efficiency_history.pop(0)
        
        # Update Hurst history
        self._hurst_history.append(result.hurst_exponent)
        if len(self._hurst_history) > 100:
            self._hurst_history.pop(0)
        
        # Update fractal history
        self._fractal_history.append(result.fractal_dimension)
        if len(self._fractal_history) > 100:
            self._fractal_history.pop(0)

    def _format_output(self, result: FractalAMAResult, timestamp) -> Dict[str, Any]:
        """
        Format the calculation results for output
        
        Args:
            result: Fractal AMA calculation results
            timestamp: Current timestamp
            
        Returns:
            Formatted output dictionary
        """
        return {
            'timestamp': timestamp,
            'indicator_name': 'FractalAdaptiveMovingAverage',
            
            # Primary values
            'adaptive_ma': round(result.adaptive_ma, 6),
            'efficiency_ratio': round(result.efficiency_ratio, 4),
            'hurst_exponent': round(result.hurst_exponent, 4),
            'fractal_dimension': round(result.fractal_dimension, 4),
            'smoothing_factor': round(result.smoothing_factor, 4),
            
            # Analysis metrics
            'trend_strength': round(result.trend_strength, 4),
            'noise_level': round(result.noise_level, 4),
            'adaptation_speed': round(result.adaptation_speed, 4),
            'signal_quality': round(result.signal_quality, 4),
            'regime_classification': result.regime_classification,
            
            # Additional insights
            'market_memory': 'PERSISTENT' if result.hurst_exponent > 0.55 else 'ANTI_PERSISTENT' if result.hurst_exponent < 0.45 else 'RANDOM',
            'efficiency_level': 'HIGH' if result.efficiency_ratio > 0.7 else 'MEDIUM' if result.efficiency_ratio > 0.4 else 'LOW',
            'complexity_level': 'HIGH' if result.fractal_dimension > 1.7 else 'MEDIUM' if result.fractal_dimension > 1.3 else 'LOW',
            
            # Trading signals
            'trend_signal': self._generate_trend_signal(result),
            'adaptation_signal': self._generate_adaptation_signal(result),
            'quality_signal': 'HIGH_QUALITY' if result.signal_quality > 0.7 else 'MEDIUM_QUALITY' if result.signal_quality > 0.4 else 'LOW_QUALITY',
            
            # Historical analysis
            'efficiency_trend': self._analyze_efficiency_trend(),
            'hurst_stability': self._analyze_hurst_stability(),
            'fractal_consistency': self._analyze_fractal_consistency()
        }

    def _generate_trend_signal(self, result: FractalAMAResult) -> str:
        """
        Generate trend signal based on fractal analysis
        
        Args:
            result: Current calculation result
            
        Returns:
            Trend signal string
        """
        try:
            if result.trend_strength > 0.7 and result.efficiency_ratio > 0.6:
                if result.hurst_exponent > 0.6:
                    return "STRONG_TREND"
                else:
                    return "MODERATE_TREND"
            elif result.trend_strength < 0.3 or result.efficiency_ratio < 0.3:
                return "NO_TREND"
            else:
                return "WEAK_TREND"
                
        except Exception as e:
            logger.warning(f"Trend signal generation error: {e}")
            return "UNKNOWN"

    def _generate_adaptation_signal(self, result: FractalAMAResult) -> str:
        """
        Generate adaptation signal based on smoothing characteristics
        
        Args:
            result: Current calculation result
            
        Returns:
            Adaptation signal string
        """
        try:
            if result.adaptation_speed > 0.7:
                return "FAST_ADAPTATION"
            elif result.adaptation_speed > 0.4:
                return "MODERATE_ADAPTATION"
            else:
                return "SLOW_ADAPTATION"
                
        except Exception as e:
            logger.warning(f"Adaptation signal generation error: {e}")
            return "UNKNOWN"

    def _analyze_efficiency_trend(self) -> str:
        """
        Analyze trend in efficiency ratio over time
        
        Returns:
            Efficiency trend description
        """
        try:
            if len(self._efficiency_history) < 10:
                return "INSUFFICIENT_DATA"
            
            recent_efficiency = self._efficiency_history[-10:]
            slope, _, _, _, _ = stats.linregress(range(len(recent_efficiency)), recent_efficiency)
            
            if slope > 0.01:
                return "IMPROVING"
            elif slope < -0.01:
                return "DETERIORATING"
            else:
                return "STABLE"
                
        except Exception as e:
            logger.warning(f"Efficiency trend analysis error: {e}")
            return "UNKNOWN"

    def _analyze_hurst_stability(self) -> str:
        """
        Analyze stability of Hurst exponent over time
        
        Returns:
            Hurst stability description
        """
        try:
            if len(self._hurst_history) < 10:
                return "INSUFFICIENT_DATA"
            
            recent_hurst = self._hurst_history[-10:]
            stability = np.std(recent_hurst)
            
            if stability < 0.05:
                return "VERY_STABLE"
            elif stability < 0.1:
                return "STABLE"
            elif stability < 0.15:
                return "MODERATELY_STABLE"
            else:
                return "UNSTABLE"
                
        except Exception as e:
            logger.warning(f"Hurst stability analysis error: {e}")
            return "UNKNOWN"

    def _analyze_fractal_consistency(self) -> str:
        """
        Analyze consistency of fractal dimension over time
        
        Returns:
            Fractal consistency description
        """
        try:
            if len(self._fractal_history) < 10:
                return "INSUFFICIENT_DATA"
            
            recent_fractal = self._fractal_history[-10:]
            consistency = 1.0 / (1.0 + np.std(recent_fractal))
            
            if consistency > 0.8:
                return "HIGHLY_CONSISTENT"
            elif consistency > 0.6:
                return "CONSISTENT"
            elif consistency > 0.4:
                return "MODERATELY_CONSISTENT"
            else:
                return "INCONSISTENT"
                
        except Exception as e:
            logger.warning(f"Fractal consistency analysis error: {e}")
            return "UNKNOWN"

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data for required columns and quality
        
        Args:
            data: Input DataFrame to validate
            
        Raises:
            IndicatorCalculationError: If data validation fails
        """
        required_columns = ['high', 'low', 'close']
        
        if not all(col in data.columns for col in required_columns):
            raise IndicatorCalculationError(f"Missing required columns: {required_columns}")
        
        if len(data) < self.period:
            raise IndicatorCalculationError(f"Insufficient data: minimum {self.period} periods required")
        
        # Check for invalid values
        for col in required_columns:
            if data[col].isnull().any():
                raise IndicatorCalculationError(f"Null values found in {col}")
            if (data[col] <= 0).any():
                raise IndicatorCalculationError(f"Non-positive values found in {col}")

    def _prepare_price_series(self, data: pd.DataFrame) -> pd.Series:
        """
        Prepare price series for analysis
        
        Args:
            data: Input OHLCV data
            
        Returns:
            Processed price series (typically closing prices)
        """
        return data['close']

    def _create_default_result(self) -> Dict[str, Any]:
        """
        Create default result for insufficient data cases
        
        Returns:
            Default result dictionary
        """
        return {
            'timestamp': pd.Timestamp.now(),
            'indicator_name': 'FractalAdaptiveMovingAverage',
            'adaptive_ma': 0.0,
            'efficiency_ratio': 0.5,
            'hurst_exponent': 0.5,
            'fractal_dimension': 1.5,
            'smoothing_factor': 0.5,
            'trend_strength': 0.5,
            'noise_level': 0.5,
            'adaptation_speed': 0.5,
            'signal_quality': 0.0,
            'regime_classification': 'INSUFFICIENT_DATA',
            'market_memory': 'UNKNOWN',
            'efficiency_level': 'UNKNOWN',
            'complexity_level': 'UNKNOWN',
            'trend_signal': 'UNKNOWN',
            'adaptation_signal': 'UNKNOWN',
            'quality_signal': 'INSUFFICIENT_DATA',
            'efficiency_trend': 'INSUFFICIENT_DATA',
            'hurst_stability': 'INSUFFICIENT_DATA',
            'fractal_consistency': 'INSUFFICIENT_DATA'
        }

    def get_required_columns(self) -> List[str]:
        """
        Get list of required data columns
        
        Returns:
            List of required column names
        """
        return ['high', 'low', 'close']

    def get_indicator_name(self) -> str:
        """
        Get the indicator name
        
        Returns:
            Indicator name string
        """
        return "FractalAdaptiveMovingAverage"

    def reset(self) -> None:
        """
        Reset the indicator state
        """
        self._previous_ama = None
        self._efficiency_history = []
        self._hurst_history = []
        self._fractal_history = []