"""
Fractal Breakout Indicator - Advanced Implementation

This indicator implements sophisticated fractal-based support/resistance detection
and breakout validation using advanced mathematical models. Features include:
- Multi-timeframe fractal pattern recognition
- Dynamic support/resistance level calculation
- Breakout strength validation using fractal geometry
- Momentum confirmation with fractal analysis
- Advanced false breakout filtering

Mission: Supporting humanitarian trading platform for poor and sick children through
maximum profitability via advanced fractal breakout analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.signal import find_peaks, find_peaks_cwt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import warnings

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import IndicatorCalculationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FractalLevel:
    """Fractal support/resistance level"""
    price: float
    timestamp: pd.Timestamp
    level_type: str  # 'SUPPORT' or 'RESISTANCE'
    strength: float
    test_count: int
    fractal_dimension: float
    volume_confirmation: float

@dataclass
class BreakoutEvent:
    """Breakout event details"""
    timestamp: pd.Timestamp
    level_broken: FractalLevel
    breakout_strength: float
    volume_surge: float
    momentum_confirmation: float
    false_breakout_probability: float
    target_projection: float

@dataclass
class FractalBreakoutResult:
    """Results container for Fractal Breakout analysis"""
    current_fractals: List[FractalLevel]
    recent_breakouts: List[BreakoutEvent]
    support_levels: List[float]
    resistance_levels: List[float]
    breakout_signal: str
    signal_strength: float
    next_target: float
    risk_level: float
    fractal_pattern_quality: float
    market_structure: str

class FractalBreakoutIndicator(StandardIndicatorInterface):
    """
    Advanced Fractal Breakout Indicator
    
    Implements sophisticated fractal analysis for support/resistance detection
    and breakout validation using mathematical models from fractal geometry
    and chaos theory.
    """
    
    def __init__(self, 
                 fractal_period: int = 5,
                 lookback_window: int = 200,
                 min_fractal_strength: float = 0.3,
                 breakout_threshold: float = 0.02,
                 volume_confirmation_factor: float = 1.5,
                 false_breakout_threshold: float = 0.7,
                 clustering_eps: float = 0.01,
                 min_samples: int = 2):
        """
        Initialize the Fractal Breakout Indicator
        
        Args:
            fractal_period: Period for fractal calculation (standard is 5)
            lookback_window: Historical data window for analysis
            min_fractal_strength: Minimum strength for valid fractals
            breakout_threshold: Minimum price movement to confirm breakout
            volume_confirmation_factor: Volume surge factor for confirmation
            false_breakout_threshold: Threshold for false breakout detection
            clustering_eps: DBSCAN epsilon for level clustering
            min_samples: Minimum samples for DBSCAN clustering
        """
        super().__init__()
        self.fractal_period = fractal_period
        self.lookback_window = lookback_window
        self.min_fractal_strength = min_fractal_strength
        self.breakout_threshold = breakout_threshold
        self.volume_confirmation_factor = volume_confirmation_factor
        self.false_breakout_threshold = false_breakout_threshold
        self.clustering_eps = clustering_eps
        self.min_samples = min_samples
        
        # Initialize storage
        self._fractal_levels = []
        self._breakout_history = []
        self._pattern_cache = {}
        
        logger.info(f"Initialized FractalBreakoutIndicator with period={fractal_period}")

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate fractal breakout analysis with comprehensive validation
        
        Args:
            data: OHLCV DataFrame with required columns
            
        Returns:
            Dictionary containing fractal breakout analysis results
        """
        try:
            # Validate input data
            self._validate_data(data)
            
            if len(data) < self.lookback_window:
                logger.warning(f"Insufficient data: {len(data)} < {self.lookback_window}")
                return self._create_default_result()
            
            # Use recent data window
            recent_data = data[-self.lookback_window:].copy()
            
            # Detect fractal patterns
            fractal_highs, fractal_lows = self._detect_fractal_patterns(recent_data)
            
            # Calculate fractal levels with strength analysis
            current_fractals = self._calculate_fractal_levels(
                recent_data, fractal_highs, fractal_lows
            )
            
            # Cluster and filter fractal levels
            clustered_levels = self._cluster_fractal_levels(current_fractals)
            
            # Detect breakout events
            recent_breakouts = self._detect_breakout_events(recent_data, clustered_levels)
            
            # Analyze current market structure
            market_analysis = self._analyze_market_structure(
                recent_data, clustered_levels, recent_breakouts
            )
            
            # Generate trading signals
            signal_analysis = self._generate_breakout_signals(
                recent_data, clustered_levels, recent_breakouts, market_analysis
            )
            
            # Separate support and resistance levels
            support_levels = [level.price for level in clustered_levels if level.level_type == 'SUPPORT']
            resistance_levels = [level.price for level in clustered_levels if level.level_type == 'RESISTANCE']
            
            # Create comprehensive result
            result = FractalBreakoutResult(
                current_fractals=clustered_levels,
                recent_breakouts=recent_breakouts,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                breakout_signal=signal_analysis['breakout_signal'],
                signal_strength=signal_analysis['signal_strength'],
                next_target=signal_analysis['next_target'],
                risk_level=signal_analysis['risk_level'],
                fractal_pattern_quality=market_analysis['pattern_quality'],
                market_structure=market_analysis['structure_type']
            )
            
            # Update historical data
            self._update_historical_data(clustered_levels, recent_breakouts)
            
            return self._format_output(result, data.index[-1])
            
        except Exception as e:
            logger.error(f"Error in fractal breakout calculation: {e}")
            raise IndicatorCalculationError(f"FractalBreakoutIndicator calculation failed: {e}")

    def _detect_fractal_patterns(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect fractal high and low patterns using advanced pattern recognition
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Tuple of (fractal_highs_indices, fractal_lows_indices)
        """
        try:
            highs = data['high'].values
            lows = data['low'].values
            
            # Enhanced fractal detection with multiple confirmation methods
            
            # Method 1: Traditional fractal pattern (Williams)
            traditional_highs = self._detect_traditional_fractals(highs, pattern_type='high')
            traditional_lows = self._detect_traditional_fractals(lows, pattern_type='low')
            
            # Method 2: Peak detection with prominence
            prominence_threshold = np.std(highs) * 0.5
            peak_highs, _ = find_peaks(highs, prominence=prominence_threshold, distance=self.fractal_period)
            peak_lows, _ = find_peaks(-lows, prominence=prominence_threshold, distance=self.fractal_period)
            
            # Method 3: Local extrema with statistical validation
            statistical_highs = self._detect_statistical_extrema(highs, pattern_type='high')
            statistical_lows = self._detect_statistical_extrema(lows, pattern_type='low')
            
            # Combine and validate patterns
            confirmed_highs = self._confirm_fractal_patterns(
                [traditional_highs, peak_highs, statistical_highs], highs
            )
            confirmed_lows = self._confirm_fractal_patterns(
                [traditional_lows, peak_lows, statistical_lows], lows
            )
            
            return confirmed_highs, confirmed_lows
            
        except Exception as e:
            logger.warning(f"Fractal pattern detection error: {e}")
            return np.array([]), np.array([])

    def _detect_traditional_fractals(self, prices: np.ndarray, pattern_type: str) -> np.ndarray:
        """
        Detect traditional Williams fractal patterns
        
        Args:
            prices: Price array (highs or lows)
            pattern_type: 'high' or 'low'
            
        Returns:
            Array of fractal indices
        """
        try:
            fractals = []
            half_period = self.fractal_period // 2
            
            for i in range(half_period, len(prices) - half_period):
                if pattern_type == 'high':
                    # Fractal high: middle value is highest
                    if prices[i] == max(prices[i-half_period:i+half_period+1]):
                        # Additional confirmation: must be strict maximum
                        if all(prices[i] > prices[j] for j in range(i-half_period, i+half_period+1) if j != i):
                            fractals.append(i)
                else:
                    # Fractal low: middle value is lowest
                    if prices[i] == min(prices[i-half_period:i+half_period+1]):
                        # Additional confirmation: must be strict minimum
                        if all(prices[i] < prices[j] for j in range(i-half_period, i+half_period+1) if j != i):
                            fractals.append(i)
            
            return np.array(fractals)
            
        except Exception as e:
            logger.warning(f"Traditional fractal detection error: {e}")
            return np.array([])

    def _detect_statistical_extrema(self, prices: np.ndarray, pattern_type: str) -> np.ndarray:
        """
        Detect statistical extrema using z-score analysis
        
        Args:
            prices: Price array
            pattern_type: 'high' or 'low'
            
        Returns:
            Array of extrema indices
        """
        try:
            if len(prices) < 20:
                return np.array([])
            
            # Rolling z-score calculation
            window = min(20, len(prices) // 4)
            z_scores = []
            
            for i in range(window, len(prices)):
                window_data = prices[i-window:i]
                if len(window_data) > 1:
                    mean_val = np.mean(window_data)
                    std_val = np.std(window_data)
                    if std_val > 0:
                        z_score = (prices[i] - mean_val) / std_val
                        z_scores.append(abs(z_score))
                    else:
                        z_scores.append(0)
                else:
                    z_scores.append(0)
            
            # Find significant extrema
            threshold = 2.0  # 2 standard deviations
            extrema = []
            
            for i, z_score in enumerate(z_scores):
                actual_index = i + window
                if z_score > threshold:
                    if pattern_type == 'high':
                        # Check if it's a local maximum
                        local_window = 5
                        start_idx = max(0, actual_index - local_window)
                        end_idx = min(len(prices), actual_index + local_window + 1)
                        if prices[actual_index] == max(prices[start_idx:end_idx]):
                            extrema.append(actual_index)
                    else:
                        # Check if it's a local minimum
                        local_window = 5
                        start_idx = max(0, actual_index - local_window)
                        end_idx = min(len(prices), actual_index + local_window + 1)
                        if prices[actual_index] == min(prices[start_idx:end_idx]):
                            extrema.append(actual_index)
            
            return np.array(extrema)
            
        except Exception as e:
            logger.warning(f"Statistical extrema detection error: {e}")
            return np.array([])

    def _confirm_fractal_patterns(self, pattern_lists: List[np.ndarray], 
                                 prices: np.ndarray) -> np.ndarray:
        """
        Confirm fractal patterns using multiple detection methods
        
        Args:
            pattern_lists: List of pattern arrays from different methods
            prices: Price array for validation
            
        Returns:
            Confirmed fractal indices
        """
        try:
            # Combine all detected patterns
            all_patterns = []
            for patterns in pattern_lists:
                all_patterns.extend(patterns.tolist())
            
            if not all_patterns:
                return np.array([])
            
            # Count occurrences (patterns confirmed by multiple methods)
            pattern_counts = {}
            for pattern in all_patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Filter patterns with minimum confirmation threshold
            min_confirmations = 2  # Require at least 2 methods to agree
            confirmed_patterns = []
            
            for pattern, count in pattern_counts.items():
                if count >= min_confirmations:
                    confirmed_patterns.append(pattern)
            
            # Remove patterns too close to each other (keep strongest)
            if confirmed_patterns:
                confirmed_patterns = self._filter_close_patterns(
                    confirmed_patterns, prices, min_distance=self.fractal_period
                )
            
            return np.array(sorted(confirmed_patterns))
            
        except Exception as e:
            logger.warning(f"Fractal confirmation error: {e}")
            return np.array([])

    def _filter_close_patterns(self, patterns: List[int], prices: np.ndarray, 
                              min_distance: int) -> List[int]:
        """
        Filter patterns that are too close to each other
        
        Args:
            patterns: List of pattern indices
            prices: Price array
            min_distance: Minimum distance between patterns
            
        Returns:
            Filtered pattern list
        """
        try:
            if len(patterns) <= 1:
                return patterns
            
            sorted_patterns = sorted(patterns)
            filtered_patterns = [sorted_patterns[0]]
            
            for i in range(1, len(sorted_patterns)):
                current_pattern = sorted_patterns[i]
                last_accepted = filtered_patterns[-1]
                
                if current_pattern - last_accepted >= min_distance:
                    filtered_patterns.append(current_pattern)
                else:
                    # Keep the pattern with more extreme price
                    if abs(prices[current_pattern] - np.mean(prices)) > abs(prices[last_accepted] - np.mean(prices)):
                        filtered_patterns[-1] = current_pattern
            
            return filtered_patterns
            
        except Exception as e:
            logger.warning(f"Pattern filtering error: {e}")
            return patterns

    def _calculate_fractal_levels(self, data: pd.DataFrame, 
                                 fractal_highs: np.ndarray, 
                                 fractal_lows: np.ndarray) -> List[FractalLevel]:
        """
        Calculate fractal levels with strength and validation analysis
        
        Args:
            data: OHLCV DataFrame
            fractal_highs: Indices of fractal highs
            fractal_lows: Indices of fractal lows
            
        Returns:
            List of validated fractal levels
        """
        try:
            fractal_levels = []
            
            # Process fractal highs (resistance levels)
            for idx in fractal_highs:
                if idx >= len(data):
                    continue
                    
                price = data['high'].iloc[idx]
                timestamp = data.index[idx]
                
                # Calculate fractal strength
                strength = self._calculate_fractal_strength(
                    data, idx, price, level_type='RESISTANCE'
                )
                
                if strength >= self.min_fractal_strength:
                    # Calculate additional metrics
                    test_count = self._count_level_tests(data, price, tolerance=0.01)
                    fractal_dim = self._calculate_local_fractal_dimension(data, idx)
                    volume_conf = self._calculate_volume_confirmation(data, idx)
                    
                    fractal_level = FractalLevel(
                        price=price,
                        timestamp=timestamp,
                        level_type='RESISTANCE',
                        strength=strength,
                        test_count=test_count,
                        fractal_dimension=fractal_dim,
                        volume_confirmation=volume_conf
                    )
                    fractal_levels.append(fractal_level)
            
            # Process fractal lows (support levels)
            for idx in fractal_lows:
                if idx >= len(data):
                    continue
                    
                price = data['low'].iloc[idx]
                timestamp = data.index[idx]
                
                # Calculate fractal strength
                strength = self._calculate_fractal_strength(
                    data, idx, price, level_type='SUPPORT'
                )
                
                if strength >= self.min_fractal_strength:
                    # Calculate additional metrics
                    test_count = self._count_level_tests(data, price, tolerance=0.01)
                    fractal_dim = self._calculate_local_fractal_dimension(data, idx)
                    volume_conf = self._calculate_volume_confirmation(data, idx)
                    
                    fractal_level = FractalLevel(
                        price=price,
                        timestamp=timestamp,
                        level_type='SUPPORT',
                        strength=strength,
                        test_count=test_count,
                        fractal_dimension=fractal_dim,
                        volume_confirmation=volume_conf
                    )
                    fractal_levels.append(fractal_level)
            
            return fractal_levels
            
        except Exception as e:
            logger.warning(f"Fractal level calculation error: {e}")
            \1\n\n    \3calculate_fractal_strength(self, data: pd.DataFrame, idx: int, 
                                   price: float, level_type: str) -> float:
        """
        Calculate the strength of a fractal level
        
        Args:
            data: OHLCV DataFrame
            idx: Index of the fractal
            price: Price of the fractal level
            level_type: 'SUPPORT' or 'RESISTANCE'
            
        Returns:
            Fractal strength score (0-1)
        """
        try:
            strength_factors = []
            
            # Factor 1: Time since formation (older = stronger)
            time_factor = min(1.0, (len(data) - idx) / 50.0)
            strength_factors.append(time_factor * 0.2)
            
            # Factor 2: Number of price tests
            tolerance = abs(price * 0.01)  # 1% tolerance
            test_count = 0
            
            for i in range(idx + 1, len(data)):
                if level_type == 'RESISTANCE':
                    if abs(data['high'].iloc[i] - price) <= tolerance:
                        test_count += 1
                else:
                    if abs(data['low'].iloc[i] - price) <= tolerance:
                        test_count += 1
            
            test_factor = min(1.0, test_count / 5.0)
            strength_factors.append(test_factor * 0.3)
            
            # Factor 3: Volume at fractal formation
            if 'volume' in data.columns:
                fractal_volume = data['volume'].iloc[idx]
                avg_volume = data['volume'].iloc[max(0, idx-20):idx+1].mean()
                volume_factor = min(1.0, fractal_volume / (avg_volume + 1e-10))
                strength_factors.append(volume_factor * 0.2)
            else:
                strength_factors.append(0.2)  # Default volume factor
            
            # Factor 4: Price extremity (how extreme the fractal is)
            window_size = min(50, len(data))
            start_idx = max(0, idx - window_size // 2)
            end_idx = min(len(data), idx + window_size // 2)
            
            if level_type == 'RESISTANCE':
                window_max = data['high'].iloc[start_idx:end_idx].max()
                extremity_factor = price / (window_max + 1e-10)
            else:
                window_min = data['low'].iloc[start_idx:end_idx].min()
                extremity_factor = (window_min + 1e-10) / price if price > 0 else 0
            
            strength_factors.append(min(1.0, extremity_factor) * 0.3)
            
            return sum(strength_factors)
            
        except Exception as e:
            logger.warning(f"Fractal strength calculation error: {e}")
            \1\n\n    \3count_level_tests(self, data: pd.DataFrame, level_price: float, 
                          tolerance: float = 0.01) -> int:
        """
        Count how many times a price level has been tested
        
        Args:
            data: OHLCV DataFrame
            level_price: Price level to test
            tolerance: Tolerance for level testing (as fraction)
            
        Returns:
            Number of times the level was tested
        """
        try:
            tolerance_amount = abs(level_price * tolerance)
            test_count = 0
            
            # Check highs and lows for tests
            for i in range(len(data)):
                high_test = abs(data['high'].iloc[i] - level_price) <= tolerance_amount
                low_test = abs(data['low'].iloc[i] - level_price) <= tolerance_amount
                
                if high_test or low_test:
                    test_count += 1
            
            return test_count
            
        except Exception as e:
            logger.warning(f"Level test counting error: {e}")
            \1\n\n    \3calculate_local_fractal_dimension(self, data: pd.DataFrame, idx: int) -> float:
        """
        Calculate local fractal dimension around a fractal point
        
        Args:
            data: OHLCV DataFrame
            idx: Index of the fractal
            
        Returns:
            Local fractal dimension
        """
        try:
            # Use a local window around the fractal
            window_size = min(20, len(data) // 4)
            start_idx = max(0, idx - window_size // 2)
            end_idx = min(len(data), idx + window_size // 2)
            
            prices = data['close'].iloc[start_idx:end_idx].values
            
            if len(prices) < 5:
                return 1.5
            
            # Simple box-counting approximation
            normalized_prices = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)
            
            box_sizes = [2, 4, 8]
            box_counts = []
            
            for box_size in box_sizes:
                # Count boxes
                boxes = set()
                for i, price in enumerate(normalized_prices):
                    x_box = int(i * box_size / len(normalized_prices))
                    y_box = int(price * box_size)
                    boxes.add((x_box, y_box))
                
                box_counts.append(len(boxes))
            
            if len(box_counts) >= 2:
                # Simple slope calculation
                log_sizes = np.log(box_sizes[:len(box_counts)])
                log_counts = np.log(box_counts)
                
                if len(log_sizes) >= 2:
                    slope = (log_counts[-1] - log_counts[0]) / (log_sizes[-1] - log_sizes[0])
                    fractal_dim = -slope
                    return np.clip(fractal_dim, 1.0, 2.0)
            
            return 1.5
            
        except Exception as e:
            logger.warning(f"Local fractal dimension calculation error: {e}")
            \1\n\n    \3calculate_volume_confirmation(self, data: pd.DataFrame, idx: int) -> float:
        """
        Calculate volume confirmation for a fractal level
        
        Args:
            data: OHLCV DataFrame
            idx: Index of the fractal
            
        Returns:
            Volume confirmation score (0-1)
        """
        try:
            if 'volume' not in data.columns:
                return 0.5  # Default score if no volume data
            
            # Compare fractal volume to average
            window_size = min(20, len(data))
            start_idx = max(0, idx - window_size)
            
            fractal_volume = data['volume'].iloc[idx]
            avg_volume = data['volume'].iloc[start_idx:idx+1].mean()
            
            if avg_volume > 0:
                volume_ratio = fractal_volume / avg_volume
                # Normalize to 0-1 scale
                confirmation = min(1.0, volume_ratio / 2.0)
                return confirmation
            
            return 0.5
            
        except Exception as e:
            logger.warning(f"Volume confirmation calculation error: {e}")
            \1\n\n    \3cluster_fractal_levels(self, fractal_levels: List[FractalLevel]) -> List[FractalLevel]:
        """
        Cluster nearby fractal levels and select the strongest
        
        Args:
            fractal_levels: List of fractal levels
            
        Returns:
            Clustered and filtered fractal levels
        """
        try:
            if len(fractal_levels) < 2:
                return fractal_levels
            
            # Separate support and resistance levels
            support_levels = [level for level in fractal_levels if level.level_type == 'SUPPORT']
            resistance_levels = [level for level in fractal_levels if level.level_type == 'RESISTANCE']
            
            clustered_levels = []
            
            # Cluster support levels
            if support_levels:
                clustered_support = self._cluster_levels_by_price(support_levels)
                clustered_levels.extend(clustered_support)
            
            # Cluster resistance levels
            if resistance_levels:
                clustered_resistance = self._cluster_levels_by_price(resistance_levels)
                clustered_levels.extend(clustered_resistance)
            
            return clustered_levels
            
        except Exception as e:
            logger.warning(f"Fractal clustering error: {e}")
            \1\n\n    \3cluster_levels_by_price(self, levels: List[FractalLevel]) -> List[FractalLevel]:
        """
        Cluster levels by price proximity using DBSCAN
        
        Args:
            levels: List of fractal levels of same type
            
        Returns:
            Clustered levels (strongest from each cluster)
        """
        try:
            if len(levels) < 2:
                return levels
            
            # Prepare data for clustering
            prices = np.array([level.price for level in levels]).reshape(-1, 1)
            
            # Normalize prices for clustering
            scaler = MinMaxScaler()
            normalized_prices = scaler.fit_transform(prices)
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=self.clustering_eps, min_samples=self.min_samples)
            cluster_labels = clustering.fit_predict(normalized_prices)
            
            # Select strongest level from each cluster
            clustered_levels = []
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    # Add all noise points (they are unique)
                    noise_indices = np.where(cluster_labels == -1)[0]
                    for idx in noise_indices:
                        clustered_levels.append(levels[idx])
                else:
                    # Find strongest level in cluster
                    cluster_indices = np.where(cluster_labels == label)[0]
                    cluster_levels = [levels[idx] for idx in cluster_indices]
                    
                    # Select level with highest strength
                    strongest_level = max(cluster_levels, key=lambda x: x.strength)
                    clustered_levels.append(strongest_level)
            
            return clustered_levels
            
        except Exception as e:
            logger.warning(f"Price clustering error: {e}")
            \1\n\n    \3detect_breakout_events(self, data: pd.DataFrame, 
                               fractal_levels: List[FractalLevel]) -> List[BreakoutEvent]:
        """
        Detect recent breakout events with comprehensive validation
        
        Args:
            data: OHLCV DataFrame
            fractal_levels: List of fractal levels
            
        Returns:
            List of detected breakout events
        """
        try:
            breakout_events = []
            recent_periods = min(50, len(data))  # Look at recent periods
            
            for level in fractal_levels:
                # Check for breakouts in recent data
                for i in range(len(data) - recent_periods, len(data)):
                    if i < 0:
                        continue
                    
                    current_high = data['high'].iloc[i]
                    current_low = data['low'].iloc[i]
                    current_close = data['close'].iloc[i]
                    timestamp = data.index[i]
                    
                    breakout_detected = False
                    
                    if level.level_type == 'RESISTANCE':
                        # Check for resistance breakout
                        if current_high > level.price * (1 + self.breakout_threshold):
                            breakout_detected = True
                    else:
                        # Check for support breakout
                        if current_low < level.price * (1 - self.breakout_threshold):
                            breakout_detected = True
                    
                    if breakout_detected:
                        # Validate breakout with additional criteria
                        breakout_strength = self._calculate_breakout_strength(data, i, level)
                        volume_surge = self._calculate_volume_surge(data, i)
                        momentum_conf = self._calculate_momentum_confirmation(data, i, level)
                        false_breakout_prob = self._calculate_false_breakout_probability(
                            data, i, level, breakout_strength
                        )
                        target_projection = self._calculate_target_projection(data, level)
                        
                        breakout_event = BreakoutEvent(
                            timestamp=timestamp,
                            level_broken=level,
                            breakout_strength=breakout_strength,
                            volume_surge=volume_surge,
                            momentum_confirmation=momentum_conf,
                            false_breakout_probability=false_breakout_prob,
                            target_projection=target_projection
                        )
                        
                        breakout_events.append(breakout_event)
                        break  # Only one breakout per level
            
            return breakout_events
            
        except Exception as e:
            logger.warning(f"Breakout detection error: {e}")
            \1\n\n    \3calculate_breakout_strength(self, data: pd.DataFrame, breakout_idx: int, 
                                   level: FractalLevel) -> float:
        """
        Calculate the strength of a breakout
        
        Args:
            data: OHLCV DataFrame
            breakout_idx: Index of breakout candle
            level: Fractal level that was broken
            
        Returns:
            Breakout strength score (0-1)
        """
        try:
            strength_factors = []
            
            # Factor 1: Price penetration distance
            if level.level_type == 'RESISTANCE':
                penetration = (data['close'].iloc[breakout_idx] - level.price) / level.price
            else:
                penetration = (level.price - data['close'].iloc[breakout_idx]) / level.price
            
            penetration_factor = min(1.0, penetration / 0.05)  # Normalize to 5% max
            strength_factors.append(penetration_factor * 0.4)
            
            # Factor 2: Candle body size (strong breakout = large body)
            open_price = data['open'].iloc[breakout_idx]
            close_price = data['close'].iloc[breakout_idx]
            high_price = data['high'].iloc[breakout_idx]
            low_price = data['low'].iloc[breakout_idx]
            
            body_size = abs(close_price - open_price)
            candle_range = high_price - low_price
            
            if candle_range > 0:
                body_ratio = body_size / candle_range
                strength_factors.append(body_ratio * 0.3)
            else:
                strength_factors.append(0.0)
            
            # Factor 3: Follow-through (next few candles continue the move)
            follow_through_score = 0
            for i in range(1, min(4, len(data) - breakout_idx)):
                next_idx = breakout_idx + i
                if level.level_type == 'RESISTANCE':
                    if data['close'].iloc[next_idx] > level.price:
                        follow_through_score += 0.25
                else:
                    if data['close'].iloc[next_idx] < level.price:
                        follow_through_score += 0.25
            
            strength_factors.append(follow_through_score * 0.3)
            
            return sum(strength_factors)
            
        except Exception as e:
            logger.warning(f"Breakout strength calculation error: {e}")
            \1\n\n    \3calculate_volume_surge(self, data: pd.DataFrame, breakout_idx: int) -> float:
        """
        Calculate volume surge during breakout
        
        Args:
            data: OHLCV DataFrame
            breakout_idx: Index of breakout candle
            
        Returns:
            Volume surge ratio
        """
        try:
            if 'volume' not in data.columns:
                return 1.0  # Default if no volume data
            
            breakout_volume = data['volume'].iloc[breakout_idx]
            
            # Calculate average volume over previous periods
            lookback = min(20, breakout_idx)
            if lookback > 0:
                avg_volume = data['volume'].iloc[breakout_idx-lookback:breakout_idx].mean()
                if avg_volume > 0:
                    volume_surge = breakout_volume / avg_volume
                    return min(5.0, volume_surge)  # Cap at 5x
            
            return 1.0
            
        except Exception as e:
            logger.warning(f"Volume surge calculation error: {e}")
            \1\n\n    \3calculate_momentum_confirmation(self, data: pd.DataFrame, breakout_idx: int, 
                                       level: FractalLevel) -> float:
        """
        Calculate momentum confirmation for breakout
        
        Args:
            data: OHLCV DataFrame
            breakout_idx: Index of breakout candle
            level: Fractal level that was broken
            
        Returns:
            Momentum confirmation score (0-1)
        """
        try:
            momentum_factors = []
            
            # Factor 1: RSI momentum (simplified calculation)
            if breakout_idx >= 14:  # Need at least 14 periods for RSI
                closes = data['close'].iloc[breakout_idx-13:breakout_idx+1]
                gains = closes.diff().clip(lower=0)
                losses = (-closes.diff()).clip(lower=0)
                
                avg_gain = gains.mean()
                avg_loss = losses.mean()
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    if level.level_type == 'RESISTANCE':
                        rsi_factor = min(1.0, (rsi - 50) / 30)  # Bullish momentum
                    else:
                        rsi_factor = min(1.0, (50 - rsi) / 30)  # Bearish momentum
                    
                    momentum_factors.append(max(0, rsi_factor) * 0.5)
                else:
                    momentum_factors.append(0.25)
            else:
                momentum_factors.append(0.25)
            
            # Factor 2: Price momentum (rate of change)
            if breakout_idx >= 5:
                price_change = data['close'].iloc[breakout_idx] / data['close'].iloc[breakout_idx-5]
                
                if level.level_type == 'RESISTANCE':
                    momentum_score = min(1.0, (price_change - 1.0) / 0.05)  # 5% max
                else:
                    momentum_score = min(1.0, (1.0 - price_change) / 0.05)  # 5% max
                
                momentum_factors.append(max(0, momentum_score) * 0.5)
            else:
                momentum_factors.append(0.25)
            
            return sum(momentum_factors)
            
        except Exception as e:
            logger.warning(f"Momentum confirmation calculation error: {e}")
            \1\n\n    \3calculate_false_breakout_probability(self, data: pd.DataFrame, breakout_idx: int,
                                            level: FractalLevel, breakout_strength: float) -> float:
        """
        Calculate probability that breakout is false
        
        Args:
            data: OHLCV DataFrame
            breakout_idx: Index of breakout candle
            level: Fractal level that was broken
            breakout_strength: Calculated breakout strength
            
        Returns:
            False breakout probability (0-1)
        """
        try:
            false_signals = []
            
            # Factor 1: Weak breakout strength indicates higher false probability
            strength_factor = 1.0 - breakout_strength
            false_signals.append(strength_factor * 0.4)
            
            # Factor 2: Immediate reversal (next candle reverses)
            if breakout_idx < len(data) - 1:
                next_close = data['close'].iloc[breakout_idx + 1]
                current_close = data['close'].iloc[breakout_idx]
                
                if level.level_type == 'RESISTANCE':
                    if next_close < level.price:  # Immediate reversal below resistance
                        false_signals.append(0.3)
                    else:
                        false_signals.append(0.0)
                else:
                    if next_close > level.price:  # Immediate reversal above support
                        false_signals.append(0.3)
                    else:
                        false_signals.append(0.0)
            else:
                false_signals.append(0.15)  # Default uncertainty
            
            # Factor 3: Low volume suggests false breakout
            volume_surge = self._calculate_volume_surge(data, breakout_idx)
            if volume_surge < self.volume_confirmation_factor:
                volume_false_factor = (self.volume_confirmation_factor - volume_surge) / self.volume_confirmation_factor
                false_signals.append(volume_false_factor * 0.3)
            else:
                false_signals.append(0.0)
            
            total_false_probability = sum(false_signals)
            return min(1.0, total_false_probability)
            
        except Exception as e:
            logger.warning(f"False breakout probability calculation error: {e}")
            \1\n\n    \3calculate_target_projection(self, data: pd.DataFrame, level: FractalLevel) -> float:
        """
        Calculate price target projection for breakout
        
        Args:
            data: OHLCV DataFrame
            level: Fractal level that was broken
            
        Returns:
            Projected target price
        """
        try:
            # Method: Use the height of the fractal formation
            current_price = data['close'].iloc[-1]
            
            # Find recent swing high and low
            recent_periods = min(50, len(data))
            recent_data = data[-recent_periods:]
            
            recent_high = recent_data['high'].max()
            recent_low = recent_data['low'].min()
            formation_height = recent_high - recent_low
            
            if level.level_type == 'RESISTANCE':
                # Project upward
                target = level.price + formation_height
            else:
                # Project downward
                target = level.price - formation_height
            
            return max(0, target)  # Ensure non-negative price
            
        except Exception as e:
            logger.warning(f"Target projection calculation error: {e}")
            return data['close'].iloc[-1]  # Return current price as fallback    def _analyze_market_structure(self, data: pd.DataFrame, 
                                 fractal_levels: List[FractalLevel],
                                 breakout_events: List[BreakoutEvent]) -> Dict[str, Any]:
        """
        Analyze overall market structure using fractal analysis
        
        Args:
            data: OHLCV DataFrame
            fractal_levels: Current fractal levels
            breakout_events: Recent breakout events
            
        Returns:
            Market structure analysis results
        """
        try:
            # Analyze pattern quality
            pattern_quality = self._assess_fractal_pattern_quality(fractal_levels)
            
            # Determine market structure type
            structure_type = self._classify_market_structure(data, fractal_levels)
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(data, fractal_levels)
            
            # Assess volatility regime
            volatility_regime = self._assess_volatility_regime(data)
            
            return {
                'pattern_quality': pattern_quality,
                'structure_type': structure_type,
                'trend_strength': trend_strength,
                'volatility_regime': volatility_regime,
                'support_strength': self._calculate_support_strength(fractal_levels),
                'resistance_strength': self._calculate_resistance_strength(fractal_levels)
            }
            
        except Exception as e:
            logger.warning(f"Market structure analysis error: {e}")
            return {
                'pattern_quality': 0.5,
                'structure_type': 'UNKNOWN',
                'trend_strength': 0.5,
                'volatility_regime': 'NORMAL',
                'support_strength': 0.5,
                'resistance_strength': 0.5
            }

    def _assess_fractal_pattern_quality(self, fractal_levels: List[FractalLevel]) -> float:
        """
        Assess the quality of fractal patterns
        
        Args:
            fractal_levels: List of fractal levels
            
        Returns:
            Pattern quality score (0-1)
        """
        try:
            if not fractal_levels:
                return 0.0
            
            quality_factors = []
            
            # Factor 1: Average strength of fractals
            avg_strength = np.mean([level.strength for level in fractal_levels])
            quality_factors.append(avg_strength * 0.4)
            
            # Factor 2: Number of well-tested levels
            well_tested_count = sum(1 for level in fractal_levels if level.test_count >= 3)
            well_tested_ratio = well_tested_count / len(fractal_levels)
            quality_factors.append(well_tested_ratio * 0.3)
            
            # Factor 3: Fractal dimension consistency
            fractal_dims = [level.fractal_dimension for level in fractal_levels]
            if len(fractal_dims) > 1:
                dim_consistency = 1.0 / (1.0 + np.std(fractal_dims))
                quality_factors.append(dim_consistency * 0.3)
            else:
                quality_factors.append(0.15)
            
            return sum(quality_factors)
            
        except Exception as e:
            logger.warning(f"Pattern quality assessment error: {e}")
            \1\n\n    \3classify_market_structure(self, data: pd.DataFrame, 
                                  fractal_levels: List[FractalLevel]) -> str:
        """
        Classify the current market structure
        
        Args:
            data: OHLCV DataFrame
            fractal_levels: Current fractal levels
            
        Returns:
            Market structure classification
        """
        try:
            if len(data) < 50:
                return "INSUFFICIENT_DATA"
            
            # Analyze recent price action
            recent_data = data[-50:]
            recent_high = recent_data['high'].max()
            recent_low = recent_data['low'].min()
            current_price = data['close'].iloc[-1]
            
            # Count support and resistance levels
            support_levels = [level for level in fractal_levels if level.level_type == 'SUPPORT']
            resistance_levels = [level for level in fractal_levels if level.level_type == 'RESISTANCE']
            
            # Analyze price position relative to levels
            price_position = (current_price - recent_low) / (recent_high - recent_low + 1e-10)
            
            # Classification logic
            if len(support_levels) > len(resistance_levels) and price_position > 0.7:
                return "UPTREND_STRUCTURE"
            elif len(resistance_levels) > len(support_levels) and price_position < 0.3:
                return "DOWNTREND_STRUCTURE"
            elif len(support_levels) > 2 and len(resistance_levels) > 2:
                if 0.3 <= price_position <= 0.7:
                    return "RANGE_BOUND"
                else:
                    return "COMPLEX_STRUCTURE"
            elif len(fractal_levels) < 3:
                return "DEVELOPING_STRUCTURE"
            else:
                return "TRANSITIONAL"
                
        except Exception as e:
            logger.warning(f"Market structure classification error: {e}")
            \1\n\n    \3calculate_trend_strength(self, data: pd.DataFrame, 
                                 fractal_levels: List[FractalLevel]) -> float:
        """
        Calculate overall trend strength
        
        Args:
            data: OHLCV DataFrame
            fractal_levels: Current fractal levels
            
        Returns:
            Trend strength score (0-1)
        """
        try:
            if len(data) < 20:
                return 0.5
            
            # Calculate directional movement
            recent_data = data[-20:]
            price_change = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1
            
            # Calculate trend consistency
            close_prices = recent_data['close'].values
            trend_direction = 1 if price_change > 0 else -1
            
            consistent_moves = 0
            total_moves = len(close_prices) - 1
            
            for i in range(1, len(close_prices)):
                daily_change = close_prices[i] - close_prices[i-1]
                if (daily_change > 0 and trend_direction > 0) or (daily_change < 0 and trend_direction < 0):
                    consistent_moves += 1
            
            consistency_ratio = consistent_moves / total_moves if total_moves > 0 else 0
            
            # Combine factors
            magnitude_strength = min(1.0, abs(price_change) / 0.1)  # 10% max for full strength
            trend_strength = (magnitude_strength + consistency_ratio) / 2
            
            return np.clip(trend_strength, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Trend strength calculation error: {e}")
            \1\n\n    \3assess_volatility_regime(self, data: pd.DataFrame) -> str:
        """
        Assess current volatility regime
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Volatility regime classification
        """
        try:
            if len(data) < 20:
                return "UNKNOWN"
            
            # Calculate recent volatility
            recent_returns = data['close'].pct_change().dropna()[-20:]
            current_vol = recent_returns.std()
            
            # Calculate longer-term volatility for comparison
            if len(data) >= 60:
                long_term_returns = data['close'].pct_change().dropna()[-60:]
                long_term_vol = long_term_returns.std()
                
                vol_ratio = current_vol / (long_term_vol + 1e-10)
                
                if vol_ratio > 1.5:
                    return "HIGH_VOLATILITY"
                elif vol_ratio < 0.7:
                    return "LOW_VOLATILITY"
                else:
                    return "NORMAL_VOLATILITY"
            else:
                # Use absolute thresholds if insufficient long-term data
                if current_vol > 0.03:  # 3% daily volatility
                    return "HIGH_VOLATILITY"
                elif current_vol < 0.01:  # 1% daily volatility
                    return "LOW_VOLATILITY"
                else:
                    return "NORMAL_VOLATILITY"
                    
        except Exception as e:
            logger.warning(f"Volatility regime assessment error: {e}")
            \1\n\n    \3calculate_support_strength(self, fractal_levels: List[FractalLevel]) -> float:
        """
        Calculate overall support strength
        
        Args:
            fractal_levels: Current fractal levels
            
        Returns:
            Support strength score (0-1)
        """
        try:
            support_levels = [level for level in fractal_levels if level.level_type == 'SUPPORT']
            
            if not support_levels:
                return 0.0
            
            # Average strength weighted by test count
            total_weight = 0
            weighted_strength = 0
            
            for level in support_levels:
                weight = 1 + level.test_count
                weighted_strength += level.strength * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_strength / total_weight
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Support strength calculation error: {e}")
            \1\n\n    \3calculate_resistance_strength(self, fractal_levels: List[FractalLevel]) -> float:
        """
        Calculate overall resistance strength
        
        Args:
            fractal_levels: Current fractal levels
            
        Returns:
            Resistance strength score (0-1)
        """
        try:
            resistance_levels = [level for level in fractal_levels if level.level_type == 'RESISTANCE']
            
            if not resistance_levels:
                return 0.0
            
            # Average strength weighted by test count
            total_weight = 0
            weighted_strength = 0
            
            for level in resistance_levels:
                weight = 1 + level.test_count
                weighted_strength += level.strength * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_strength / total_weight
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Resistance strength calculation error: {e}")
            \1\n\n    \3generate_breakout_signals(self, data: pd.DataFrame,
                                  fractal_levels: List[FractalLevel],
                                  breakout_events: List[BreakoutEvent],
                                  market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive breakout trading signals
        
        Args:
            data: OHLCV DataFrame
            fractal_levels: Current fractal levels
            breakout_events: Recent breakout events
            market_analysis: Market structure analysis
            
        Returns:
            Signal analysis results
        """
        try:
            # Analyze recent breakouts for signal generation
            if breakout_events:
                latest_breakout = max(breakout_events, key=lambda x: x.timestamp)
                
                # Check if breakout is valid (low false breakout probability)
                if latest_breakout.false_breakout_probability < self.false_breakout_threshold:
                    if latest_breakout.level_broken.level_type == 'RESISTANCE':
                        breakout_signal = "BULLISH_BREAKOUT"
                    else:
                        breakout_signal = "BEARISH_BREAKOUT"
                    
                    signal_strength = (latest_breakout.breakout_strength + 
                                     latest_breakout.momentum_confirmation) / 2
                    next_target = latest_breakout.target_projection
                    risk_level = latest_breakout.false_breakout_probability
                else:
                    breakout_signal = "FALSE_BREAKOUT_RISK"
                    signal_strength = 1.0 - latest_breakout.false_breakout_probability
                    next_target = latest_breakout.level_broken.price
                    risk_level = latest_breakout.false_breakout_probability
            else:
                # No recent breakouts - analyze setup conditions
                current_price = data['close'].iloc[-1]
                
                # Find nearest levels
                nearest_resistance = self._find_nearest_level(
                    current_price, fractal_levels, 'RESISTANCE'
                )
                nearest_support = self._find_nearest_level(
                    current_price, fractal_levels, 'SUPPORT'
                )
                
                if nearest_resistance and nearest_support:
                    resistance_distance = abs(current_price - nearest_resistance.price) / current_price
                    support_distance = abs(current_price - nearest_support.price) / current_price
                    
                    if resistance_distance < 0.02:  # Within 2% of resistance
                        breakout_signal = "APPROACHING_RESISTANCE"
                        signal_strength = nearest_resistance.strength
                        next_target = nearest_resistance.price * 1.05  # 5% above resistance
                        risk_level = 1.0 - nearest_resistance.strength
                    elif support_distance < 0.02:  # Within 2% of support
                        breakout_signal = "APPROACHING_SUPPORT"
                        signal_strength = nearest_support.strength
                        next_target = nearest_support.price * 0.95  # 5% below support
                        risk_level = 1.0 - nearest_support.strength
                    else:
                        breakout_signal = "NO_IMMEDIATE_SETUP"
                        signal_strength = 0.5
                        next_target = current_price
                        risk_level = 0.5
                else:
                    breakout_signal = "INSUFFICIENT_LEVELS"
                    signal_strength = 0.0
                    next_target = current_price
                    risk_level = 1.0
            
            return {
                'breakout_signal': breakout_signal,
                'signal_strength': np.clip(signal_strength, 0.0, 1.0),
                'next_target': next_target,
                'risk_level': np.clip(risk_level, 0.0, 1.0)
            }
            
        except Exception as e:
            logger.warning(f"Signal generation error: {e}")
            return {
                'breakout_signal': 'ERROR',
                'signal_strength': 0.0,
                'next_target': data['close'].iloc[-1],
                'risk_level': 1.0
            }

    def _find_nearest_level(self, current_price: float, 
                           fractal_levels: List[FractalLevel],
                           level_type: str) -> Optional[FractalLevel]:
        """
        Find the nearest fractal level of specified type
        
        Args:
            current_price: Current market price
            fractal_levels: List of fractal levels
            level_type: 'SUPPORT' or 'RESISTANCE'
            
        Returns:
            Nearest fractal level or None
        """
        try:
            relevant_levels = [level for level in fractal_levels if level.level_type == level_type]
            
            if not relevant_levels:
                return None
            
            # Find closest level
            distances = [abs(level.price - current_price) for level in relevant_levels]
            min_distance_idx = distances.index(min(distances))
            
            return relevant_levels[min_distance_idx]
            
        except Exception as e:
            logger.warning(f"Nearest level finding error: {e}")
            return None

    def _update_historical_data(self, fractal_levels: List[FractalLevel],
                               breakout_events: List[BreakoutEvent]) -> None:
        """
        Update historical data storage
        
        Args:
            fractal_levels: Current fractal levels
            breakout_events: Recent breakout events
        """
        try:
            # Update fractal levels (keep recent ones)
            self._fractal_levels.extend(fractal_levels)
            if len(self._fractal_levels) > 200:  # Keep last 200
                self._fractal_levels = self._fractal_levels[-200:]
            
            # Update breakout history
            self._breakout_history.extend(breakout_events)
            if len(self._breakout_history) > 100:  # Keep last 100
                self._breakout_history = self._breakout_history[-100:]
                
        except Exception as e:
            logger.warning(f"Historical data update error: {e}")

    def _format_output(self, result: FractalBreakoutResult, timestamp) -> Dict[str, Any]:
        """
        Format the calculation results for output
        
        Args:
            result: Fractal breakout calculation results
            timestamp: Current timestamp
            
        Returns:
            Formatted output dictionary
        """
        return {
            'timestamp': timestamp,
            'indicator_name': 'FractalBreakout',
            
            # Primary signals
            'breakout_signal': result.breakout_signal,
            'signal_strength': round(result.signal_strength, 4),
            'next_target': round(result.next_target, 6),
            'risk_level': round(result.risk_level, 4),
            
            # Fractal levels
            'support_levels': [round(level, 6) for level in result.support_levels[-5:]],  # Last 5
            'resistance_levels': [round(level, 6) for level in result.resistance_levels[-5:]],  # Last 5
            'total_support_levels': len(result.support_levels),
            'total_resistance_levels': len(result.resistance_levels),
            
            # Market structure
            'market_structure': result.market_structure,
            'fractal_pattern_quality': round(result.fractal_pattern_quality, 4),
            
            # Recent breakouts
            'recent_breakout_count': len(result.recent_breakouts),
            'latest_breakout_type': result.recent_breakouts[-1].level_broken.level_type if result.recent_breakouts else 'NONE',
            
            # Additional analysis
            'level_density': len(result.current_fractals),
            'support_resistance_ratio': len(result.support_levels) / max(1, len(result.resistance_levels)),
            
            # Trading insights
            'setup_quality': 'HIGH' if result.fractal_pattern_quality > 0.7 and result.signal_strength > 0.6 else 'MEDIUM' if result.fractal_pattern_quality > 0.4 else 'LOW',
            'risk_assessment': 'LOW' if result.risk_level < 0.3 else 'MEDIUM' if result.risk_level < 0.7 else 'HIGH',
            'signal_reliability': 'HIGH' if result.signal_strength > 0.7 and result.fractal_pattern_quality > 0.6 else 'MEDIUM' if result.signal_strength > 0.4 else 'LOW'
        }

    def _validate_data(self, data: pd.DataFrame) -> None:
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
        
        if len(data) < self.fractal_period * 4:
            raise IndicatorCalculationError(f"Insufficient data: minimum {self.fractal_period * 4} periods required")
        
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
            'indicator_name': 'FractalBreakout',
            'breakout_signal': 'INSUFFICIENT_DATA',
            'signal_strength': 0.0,
            'next_target': 0.0,
            'risk_level': 1.0,
            'support_levels': [],
            'resistance_levels': [],
            'total_support_levels': 0,
            'total_resistance_levels': 0,
            'market_structure': 'UNKNOWN',
            'fractal_pattern_quality': 0.0,
            'recent_breakout_count': 0,
            'latest_breakout_type': 'NONE',
            'level_density': 0,
            'support_resistance_ratio': 1.0,
            'setup_quality': 'INSUFFICIENT_DATA',
            'risk_assessment': 'HIGH',
            'signal_reliability': 'INSUFFICIENT_DATA'
        }

    def get_required_columns(self) -> List[str]:
        """
        Get list of required data columns
        
        Returns:
            List of required column names
        """
        return ['high', 'low', 'close', 'open', 'volume']

    def get_indicator_name(self) -> str:
        """
        Get the indicator name
        
        Returns:
            Indicator name string
        """
        return "FractalBreakout"