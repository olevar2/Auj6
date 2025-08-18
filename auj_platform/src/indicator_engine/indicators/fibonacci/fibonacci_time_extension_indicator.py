"""
Fibonacci Time Extension Indicator - Advanced Time-Based Reversal Point Projections
===================================================================================

This module implements a sophisticated Fibonacci time extension indicator that projects
future time-based reversal points using Fibonacci number sequences and advanced
mathematical algorithms. It includes machine learning integration, cycle analysis,
and sophisticated pattern recognition for market timing precision.

Features:
- Mathematical Fibonacci sequence time projections (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89...)
- Advanced swing point detection and time analysis
- Machine learning time pattern recognition and validation
- Multi-timeframe time extension confluence analysis
- Cycle harmonics and rhythm detection
- Time-based momentum and volatility correlation
- Dynamic time zone strength calculation
- Real-time time extension adjustment and validation
- Advanced time clustering and confluence detection
- Natural market timing rhythm analysis

The indicator helps traders identify precise timing for market reversals and
continuations based on the mathematical properties of Fibonacci time sequences,
providing insights into natural market timing patterns and cyclical behavior.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats, signal
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, IndicatorResult, DataRequirement, DataType, SignalType
from ...core.exceptions import IndicatorCalculationException

# Configure logging
logger = logging.getLogger(__name__)

# Fibonacci sequence constants
FIBONACCI_NUMBERS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
FIBONACCI_RATIOS = [0.618, 1.0, 1.618, 2.618, 4.236, 6.854]


@dataclass
class TimeExtensionPoint:
    """Represents a Fibonacci time extension point"""
    base_swing_index: int
    base_swing_timestamp: datetime
    extension_index: int
    extension_timestamp: datetime
    fibonacci_number: int
    fibonacci_ratio: float
    time_periods: int
    strength: float         # 0.0 to 1.0
    price_at_point: float
    volume_at_point: float
    volatility_score: float
    ml_confidence: float
    

@dataclass
class TimeCluster:
    """Represents a cluster of time extension points"""
    cluster_id: str
    center_index: int
    center_timestamp: datetime
    points: List[TimeExtensionPoint]
    cluster_strength: float
    confluence_count: int
    price_range: Tuple[float, float]
    dominant_fibonacci: int
    

@dataclass
class TimePattern:
    """Represents a detected time pattern"""
    pattern_id: str
    pattern_type: str       # 'cycle', 'harmonic', 'sequence'
    start_index: int
    end_index: int
    period_length: int
    strength: float
    fibonacci_basis: List[int]
    predicted_next: List[int]
    

class FibonacciTimeExtensionIndicator(StandardIndicatorInterface):
    """
    Advanced Fibonacci Time Extension Indicator with machine learning integration
    and sophisticated time-based analysis capabilities.
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'swing_detection_period': 15,
            'min_swing_strength': 0.3,
            'fibonacci_numbers': FIBONACCI_NUMBERS[:12],  # Use first 12 Fibonacci numbers
            'fibonacci_ratios': FIBONACCI_RATIOS,
            'max_extension_periods': 500,
            'time_tolerance': 3,  # Periods tolerance for clustering
            'min_cluster_size': 2,
            'confluence_weight': 0.4,
            'ml_lookback': 100,
            'cycle_detection': True,
            'harmonic_analysis': True,
            'volume_weight': 0.3,
            'volatility_weight': 0.2,
            'max_time_extensions': 20,
            'strength_threshold': 0.25,
            'pattern_lookback': 200,
            'future_projection_periods': 100
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name="FibonacciTimeExtension", parameters=default_params)
        
        # Initialize internal state
        self.time_extensions: List[TimeExtensionPoint] = []
        self.time_clusters: List[TimeCluster] = []
        self.time_patterns: List[TimePattern] = []
        self.ml_model = None
        self.pattern_classifier = None
        self.scaler = StandardScaler()
        self.last_calculation = None
        
        # Precompute Fibonacci-based time intervals
        self.fibonacci_intervals = self._generate_fibonacci_intervals()
        
        logger.info(f"FibonacciTimeExtensionIndicator initialized with {len(self.parameters['fibonacci_numbers'])} Fibonacci numbers")
    
    def get_data_requirements(self) -> DataRequirement:
        """Define the data requirements for Fibonacci time extension calculation"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(150, self.parameters['pattern_lookback']),
            lookback_periods=400
        )
    
    def validate_parameters(self) -> bool:
        """Validate the indicator parameters"""
        try:
            required_params = ['fibonacci_numbers', 'swing_detection_period', 'max_extension_periods']
            for param in required_params:
                if param not in self.parameters:
                    raise ValueError(f"Missing required parameter: {param}")
            
            if self.parameters['swing_detection_period'] < 3:
                raise ValueError("swing_detection_period must be at least 3")
            
            if not self.parameters['fibonacci_numbers']:
                raise ValueError("fibonacci_numbers cannot be empty")
            
            if self.parameters['max_extension_periods'] < 10:
                raise ValueError("max_extension_periods must be at least 10")
            
            return True
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            return False
    
    def _generate_fibonacci_intervals(self) -> List[int]:
        """Generate Fibonacci-based time intervals for analysis"""
        intervals = []
        
        # Pure Fibonacci numbers
        intervals.extend(self.parameters['fibonacci_numbers'])
        
        # Fibonacci ratios applied to base periods
        base_periods = [10, 20, 30, 50]
        for base in base_periods:
            for ratio in self.parameters['fibonacci_ratios']:
                interval = int(base * ratio)
                if interval not in intervals and interval <= self.parameters['max_extension_periods']:
                    intervals.append(interval)
        
        # Fibonacci number combinations
        for i, fib1 in enumerate(self.parameters['fibonacci_numbers'][:6]):
            for j, fib2 in enumerate(self.parameters['fibonacci_numbers'][:6]):
                if i != j:
                    combined = fib1 + fib2
                    if combined not in intervals and combined <= self.parameters['max_extension_periods']:
                        intervals.append(combined)
        
        return sorted(list(set(intervals)))
    
    def _detect_significant_swings(self, data: pd.DataFrame) -> List[Tuple[int, float, str]]:
        """
        Detect significant swing points for time extension analysis
        """
        try:
            period = self.parameters['swing_detection_period']
            min_strength = self.parameters['min_swing_strength']
            
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            volumes = data['volume'].values
            
            significant_swings = []
            
            # Detect swing highs
            high_peaks, high_properties = signal.find_peaks(
                highs, 
                distance=period,
                prominence=np.std(highs) * min_strength
            )
            
            # Detect swing lows
            low_peaks, low_properties = signal.find_peaks(
                -lows,
                distance=period,
                prominence=np.std(lows) * min_strength
            )
            
            # Process swing highs
            for i, peak_idx in enumerate(high_peaks):
                if peak_idx >= period and peak_idx < len(data) - period:
                    strength = high_properties['prominences'][i] / np.std(highs)
                    if strength >= min_strength:
                        significant_swings.append((peak_idx, highs[peak_idx], 'high'))
            
            # Process swing lows
            for i, peak_idx in enumerate(low_peaks):
                if peak_idx >= period and peak_idx < len(data) - period:
                    strength = low_properties['prominences'][i] / np.std(lows)
                    if strength >= min_strength:
                        significant_swings.append((peak_idx, lows[peak_idx], 'low'))
            
            # Sort by time and return recent swings
            significant_swings.sort(key=lambda x: x[0])
            return significant_swings[-15:]  # Keep most recent swings
            
        except Exception as e:
            logger.error(f"Error detecting significant swings: {str(e)}")
            return []
    
    def _calculate_time_extensions(self, swings: List[Tuple[int, float, str]], 
                                  data: pd.DataFrame) -> List[TimeExtensionPoint]:
        """
        Calculate Fibonacci time extensions from swing points
        """
        try:
            extensions = []
            current_index = len(data) - 1
            
            for swing_idx, swing_price, swing_type in swings:
                # Calculate time extensions for each Fibonacci interval
                for fib_number in self.parameters['fibonacci_numbers']:
                    # Direct Fibonacci number extension
                    extension_idx = swing_idx + fib_number
                    
                    if extension_idx <= current_index + self.parameters['future_projection_periods']:
                        # Calculate extension point properties
                        extension_point = self._create_time_extension_point(
                            swing_idx, swing_price, extension_idx, fib_number, 1.0, data
                        )
                        
                        if extension_point and extension_point.strength >= self.parameters['strength_threshold']:
                            extensions.append(extension_point)
                
                # Calculate ratio-based extensions
                for ratio in self.parameters['fibonacci_ratios']:
                    extension_periods = int(fib_number * ratio)
                    extension_idx = swing_idx + extension_periods
                    
                    if extension_idx <= current_index + self.parameters['future_projection_periods']:
                        extension_point = self._create_time_extension_point(
                            swing_idx, swing_price, extension_idx, fib_number, ratio, data
                        )
                        
                        if extension_point and extension_point.strength >= self.parameters['strength_threshold']:
                            extensions.append(extension_point)
            
            # Sort by extension index and limit count
            extensions.sort(key=lambda x: x.extension_index)
            return extensions[:self.parameters['max_time_extensions']]
            
        except Exception as e:
            logger.error(f"Error calculating time extensions: {str(e)}")
            return []
    
    def _create_time_extension_point(self, base_idx: int, base_price: float, 
                                   extension_idx: int, fib_number: int, ratio: float,
                                   data: pd.DataFrame) -> Optional[TimeExtensionPoint]:
        """
        Create a time extension point with calculated properties
        """
        try:
            # Get base swing timestamp
            base_timestamp = data.index[base_idx]
            
            # Calculate extension timestamp
            if extension_idx < len(data):
                extension_timestamp = data.index[extension_idx]
                price_at_point = data['close'].iloc[extension_idx]
                volume_at_point = data['volume'].iloc[extension_idx]
            else:
                # Future projection
                time_delta = (data.index[1] - data.index[0]) * (extension_idx - len(data) + 1)
                extension_timestamp = data.index[-1] + time_delta
                price_at_point = data['close'].iloc[-1]  # Use last known price
                volume_at_point = data['volume'].mean()   # Use average volume
            
            # Calculate time periods
            time_periods = extension_idx - base_idx
            
            # Calculate volatility score
            volatility_score = self._calculate_volatility_score(extension_idx, data)
            
            # Calculate strength based on multiple factors
            strength = self._calculate_time_extension_strength(
                base_idx, extension_idx, fib_number, ratio, data
            )
            
            return TimeExtensionPoint(
                base_swing_index=base_idx,
                base_swing_timestamp=base_timestamp,
                extension_index=extension_idx,
                extension_timestamp=extension_timestamp,
                fibonacci_number=fib_number,
                fibonacci_ratio=ratio,
                time_periods=time_periods,
                strength=strength,
                price_at_point=price_at_point,
                volume_at_point=volume_at_point,
                volatility_score=volatility_score,
                ml_confidence=0.0
            )
            
        except Exception as e:
            logger.error(f"Error creating time extension point: {str(e)}")
            return None
    
    def _calculate_volatility_score(self, index: int, data: pd.DataFrame) -> float:
        """
        Calculate volatility score around a time extension point
        """
        try:
            if index >= len(data):
                # For future points, use recent volatility
                volatility_window = data['close'].pct_change().rolling(20).std().iloc[-20:]
            else:
                # For historical points, use local volatility
                start_idx = max(0, index - 10)
                end_idx = min(len(data), index + 10)
                volatility_window = data['close'].iloc[start_idx:end_idx].pct_change().rolling(5).std()
            
            if volatility_window.empty:
                return 0.0
            
            return float(volatility_window.mean()) if not pd.isna(volatility_window.mean()) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volatility score: {str(e)}")
            return 0.0
    
    def _calculate_time_extension_strength(self, base_idx: int, extension_idx: int, 
                                         fib_number: int, ratio: float, 
                                         data: pd.DataFrame) -> float:
        """
        Calculate the strength of a time extension point
        """
        try:
            strength_factors = []
            
            # Fibonacci number significance (lower numbers are stronger)
            fib_significance = 1.0 / (1 + np.log(fib_number))
            strength_factors.append(fib_significance)
            
            # Ratio significance (pure ratios like 1.0, 1.618 are stronger)
            ratio_significance = 1.0
            if abs(ratio - 1.0) < 0.01:
                ratio_significance = 1.0
            elif abs(ratio - 1.618) < 0.01:
                ratio_significance = 0.9
            elif abs(ratio - 0.618) < 0.01:
                ratio_significance = 0.9
            else:
                ratio_significance = 0.5
            strength_factors.append(ratio_significance)
            
            # Historical price action strength
            if extension_idx < len(data):
                price_action_strength = self._analyze_price_action_at_time(extension_idx, data)
                strength_factors.append(price_action_strength)
            else:
                strength_factors.append(0.5)  # Neutral for future points
            
            # Volume confirmation
            volume_strength = self._analyze_volume_at_time(extension_idx, data)
            strength_factors.append(volume_strength * self.parameters['volume_weight'])
            
            # Time period significance (shorter periods can be stronger)
            time_periods = extension_idx - base_idx
            period_strength = 1.0 / (1 + time_periods / 100.0)
            strength_factors.append(period_strength)
            
            return min(np.mean(strength_factors), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating time extension strength: {str(e)}")
            return 0.0
    
    def _analyze_price_action_at_time(self, index: int, data: pd.DataFrame) -> float:
        """
        Analyze price action strength at a specific time point
        """
        try:
            if index >= len(data) or index < 5:
                return 0.5
            
            # Look for reversal patterns around the time point
            window_start = max(0, index - 3)
            window_end = min(len(data), index + 3)
            
            price_window = data['close'].iloc[window_start:window_end]
            
            if len(price_window) < 3:
                return 0.5
            
            # Check for local extremes (reversals)
            center_idx = len(price_window) // 2
            center_price = price_window.iloc[center_idx]
            
            # Check if it's a local high
            is_local_high = all(center_price >= price for price in price_window)
            # Check if it's a local low
            is_local_low = all(center_price <= price for price in price_window)
            
            if is_local_high or is_local_low:
                return 0.8
            
            # Check for significant price movement
            price_range = price_window.max() - price_window.min()
            avg_range = data['high'].subtract(data['low']).rolling(20).mean().iloc[index]
            
            if not pd.isna(avg_range) and avg_range > 0:
                relative_range = price_range / avg_range
                return min(relative_range / 2.0, 1.0)
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error analyzing price action: {str(e)}")
            return 0.5
    
    def _analyze_volume_at_time(self, index: int, data: pd.DataFrame) -> float:
        """
        Analyze volume strength at a specific time point
        """
        try:
            if index >= len(data):
                return 0.5  # Neutral for future points
            
            current_volume = data['volume'].iloc[index]
            avg_volume = data['volume'].rolling(20).mean().iloc[index]
            
            if pd.isna(avg_volume) or avg_volume == 0:
                return 0.5
            
            volume_ratio = current_volume / avg_volume
            return min(volume_ratio / 3.0, 1.0)  # Normalize to 0-1 range
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {str(e)}")
            return 0.5
    
    def _detect_time_clusters(self, extensions: List[TimeExtensionPoint]) -> List[TimeCluster]:
        """
        Detect clusters of time extension points
        """
        try:
            if len(extensions) < 2:
                return []
            
            # Prepare data for clustering
            time_indices = np.array([ext.extension_index for ext in extensions]).reshape(-1, 1)
            
            # Use DBSCAN for clustering
            tolerance = self.parameters['time_tolerance']
            min_samples = self.parameters['min_cluster_size']
            
            clustering = DBSCAN(eps=tolerance, min_samples=min_samples)
            cluster_labels = clustering.fit_predict(time_indices)
            
            clusters = []
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                
                # Get points in this cluster
                cluster_points = [ext for i, ext in enumerate(extensions) if cluster_labels[i] == label]
                
                if len(cluster_points) >= min_samples:
                    cluster = self._create_time_cluster(cluster_points, label)
                    if cluster:
                        clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error detecting time clusters: {str(e)}")
            return []
    
    def _create_time_cluster(self, points: List[TimeExtensionPoint], cluster_id: int) -> Optional[TimeCluster]:
        """
        Create a time cluster from a group of extension points
        """
        try:
            if not points:
                return None
            
            # Calculate cluster center
            center_index = int(np.mean([p.extension_index for p in points]))
            center_timestamp = points[0].extension_timestamp  # Approximate
            
            # Calculate cluster strength
            cluster_strength = np.mean([p.strength for p in points])
            
            # Apply confluence weighting
            confluence_bonus = (len(points) - 1) * self.parameters['confluence_weight']
            cluster_strength = min(cluster_strength + confluence_bonus, 1.0)
            
            # Calculate price range
            prices = [p.price_at_point for p in points]
            price_range = (min(prices), max(prices))
            
            # Find dominant Fibonacci number
            fib_numbers = [p.fibonacci_number for p in points]
            dominant_fibonacci = max(set(fib_numbers), key=fib_numbers.count)
            
            return TimeCluster(
                cluster_id=f"cluster_{cluster_id}",
                center_index=center_index,
                center_timestamp=center_timestamp,
                points=points,
                cluster_strength=cluster_strength,
                confluence_count=len(points),
                price_range=price_range,
                dominant_fibonacci=dominant_fibonacci
            )
            
        except Exception as e:
            logger.error(f"Error creating time cluster: {str(e)}")
            return None
    
    def _detect_time_patterns(self, data: pd.DataFrame, extensions: List[TimeExtensionPoint]) -> List[TimePattern]:
        """
        Detect recurring time patterns in the data
        """
        try:
            patterns = []
            lookback = self.parameters['pattern_lookback']
            
            if len(data) < lookback:
                return patterns
            
            # Detect cycle patterns
            if self.parameters['cycle_detection']:
                cycle_patterns = self._detect_cycle_patterns(data, extensions)
                patterns.extend(cycle_patterns)
            
            # Detect harmonic patterns
            if self.parameters['harmonic_analysis']:
                harmonic_patterns = self._detect_harmonic_patterns(data, extensions)
                patterns.extend(harmonic_patterns)
            
            # Detect Fibonacci sequence patterns
            sequence_patterns = self._detect_sequence_patterns(extensions)
            patterns.extend(sequence_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting time patterns: {str(e)}")
            return []
    
    def _detect_cycle_patterns(self, data: pd.DataFrame, extensions: List[TimeExtensionPoint]) -> List[TimePattern]:
        """
        Detect cyclical time patterns
        """
        try:
            patterns = []
            
            # Use FFT to detect dominant cycles
            prices = data['close'].values
            if len(prices) < 50:
                return patterns
            
            # Detrend the data
            detrended = signal.detrend(prices)
            
            # Apply FFT
            fft_values = np.fft.fft(detrended)
            freqs = np.fft.fftfreq(len(detrended))
            
            # Find dominant frequencies
            power_spectrum = np.abs(fft_values) ** 2
            dominant_freq_indices = signal.find_peaks(power_spectrum[:len(power_spectrum)//2])[0]
            
            for freq_idx in dominant_freq_indices[:5]:  # Top 5 cycles
                if freqs[freq_idx] > 0:
                    cycle_length = int(1 / freqs[freq_idx])
                    
                    # Check if cycle length matches Fibonacci numbers
                    fib_match = min(self.parameters['fibonacci_numbers'], 
                                  key=lambda x: abs(x - cycle_length))
                    
                    if abs(fib_match - cycle_length) <= 3:  # Close match
                        pattern = TimePattern(
                            pattern_id=f"cycle_{cycle_length}",
                            pattern_type='cycle',
                            start_index=0,
                            end_index=len(data) - 1,
                            period_length=cycle_length,
                            strength=power_spectrum[freq_idx] / np.max(power_spectrum),
                            fibonacci_basis=[fib_match],
                            predicted_next=[len(data) + cycle_length]
                        )
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting cycle patterns: {str(e)}")
            return []
    
    def _detect_harmonic_patterns(self, data: pd.DataFrame, extensions: List[TimeExtensionPoint]) -> List[TimePattern]:
        """
        Detect harmonic time patterns based on Fibonacci ratios
        """
        try:
            patterns = []
            
            # Look for harmonic relationships in extension points
            for i, ext1 in enumerate(extensions):
                for j, ext2 in enumerate(extensions[i+1:], i+1):
                    time_diff = abs(ext2.extension_index - ext1.extension_index)
                    
                    # Check if time difference matches Fibonacci harmonic ratios
                    for ratio in self.parameters['fibonacci_ratios']:
                        expected_diff = int(ext1.time_periods * ratio)
                        
                        if abs(time_diff - expected_diff) <= self.parameters['time_tolerance']:
                            pattern = TimePattern(
                                pattern_id=f"harmonic_{i}_{j}",
                                pattern_type='harmonic',
                                start_index=ext1.extension_index,
                                end_index=ext2.extension_index,
                                period_length=time_diff,
                                strength=(ext1.strength + ext2.strength) / 2,
                                fibonacci_basis=[ext1.fibonacci_number, ext2.fibonacci_number],
                                predicted_next=[ext2.extension_index + time_diff]
                            )
                            patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting harmonic patterns: {str(e)}")
            return []
    
    def _detect_sequence_patterns(self, extensions: List[TimeExtensionPoint]) -> List[TimePattern]:
        """
        Detect Fibonacci sequence patterns in time extensions
        """
        try:
            patterns = []
            
            # Group extensions by base swing
            swing_groups = {}
            for ext in extensions:
                if ext.base_swing_index not in swing_groups:
                    swing_groups[ext.base_swing_index] = []
                swing_groups[ext.base_swing_index].append(ext)
            
            # Look for Fibonacci sequences within each group
            for base_idx, group in swing_groups.items():
                if len(group) >= 3:
                    # Sort by Fibonacci number
                    sorted_group = sorted(group, key=lambda x: x.fibonacci_number)
                    
                    # Check if we have a Fibonacci sequence
                    fib_numbers = [ext.fibonacci_number for ext in sorted_group]
                    
                    for i in range(len(fib_numbers) - 2):
                        if (fib_numbers[i] + fib_numbers[i+1] == fib_numbers[i+2] or
                            abs((fib_numbers[i] + fib_numbers[i+1]) - fib_numbers[i+2]) <= 1):
                            
                            pattern = TimePattern(
                                pattern_id=f"sequence_{base_idx}_{i}",
                                pattern_type='sequence',
                                start_index=sorted_group[i].extension_index,
                                end_index=sorted_group[i+2].extension_index,
                                period_length=sorted_group[i+2].extension_index - sorted_group[i].extension_index,
                                strength=np.mean([ext.strength for ext in sorted_group[i:i+3]]),
                                fibonacci_basis=fib_numbers[i:i+3],
                                predicted_next=[sorted_group[i+2].extension_index + fib_numbers[i+2]]
                            )
                            patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting sequence patterns: {str(e)}")
            return []
    
    def _prepare_ml_features(self, data: pd.DataFrame, extension: TimeExtensionPoint) -> np.ndarray:
        """
        Prepare features for machine learning validation
        """
        try:
            features = []
            
            # Time-based features
            features.append(extension.fibonacci_number / 100.0)  # Normalized
            features.append(extension.fibonacci_ratio)
            features.append(extension.time_periods / 100.0)  # Normalized
            features.append(extension.strength)
            
            # Market context features
            if extension.extension_index < len(data):
                # Historical data available
                current_price = data['close'].iloc[extension.extension_index]
                
                # Volatility
                volatility = data['close'].pct_change().rolling(10).std().iloc[extension.extension_index]
                features.append(volatility if not pd.isna(volatility) else 0.0)
                
                # Volume ratio
                volume_ratio = (data['volume'].iloc[extension.extension_index] / 
                              data['volume'].rolling(20).mean().iloc[extension.extension_index])
                features.append(volume_ratio if not pd.isna(volume_ratio) else 1.0)
                
                # Price momentum
                momentum = data['close'].pct_change(5).iloc[extension.extension_index]
                features.append(momentum if not pd.isna(momentum) else 0.0)
            else:
                # Future projection - use recent data
                features.extend([
                    data['close'].pct_change().rolling(10).std().iloc[-1] or 0.0,
                    1.0,  # Neutral volume ratio
                    data['close'].pct_change(5).iloc[-1] or 0.0
                ])
            
            # Seasonal features (time of day, day of week if applicable)
            if hasattr(extension.extension_timestamp, 'hour'):
                features.append(extension.extension_timestamp.hour / 24.0)
                features.append(extension.extension_timestamp.weekday() / 7.0)
            else:
                features.extend([0.5, 0.5])  # Neutral values
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {str(e)}")
            return np.array([[0.0] * 9])
    
    def _train_ml_models(self, data: pd.DataFrame, extensions: List[TimeExtensionPoint]):
        """
        Train machine learning models for time extension validation
        """
        try:
            if len(extensions) < 10 or len(data) < self.parameters['ml_lookback']:
                return
            
            X, y = [], []
            
            # Prepare training data
            for ext in extensions:
                if ext.extension_index < len(data) - 10:  # Ensure we have future data
                    features = self._prepare_ml_features(data, ext)
                    
                    # Calculate target (effectiveness of the time extension)
                    effectiveness = self._calculate_extension_effectiveness(ext, data)
                    
                    X.append(features[0])
                    y.append(effectiveness)
            
            if len(X) > 5:
                X = np.array(X)
                y = np.array(y)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train regression model for strength prediction
                self.ml_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1
                )
                self.ml_model.fit(X_scaled, y)
                
                # Train classifier for pattern type prediction
                pattern_labels = [self._classify_extension_pattern(ext, data) for ext in extensions[:len(X)]]
                
                self.pattern_classifier = GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=5,
                    random_state=42
                )
                self.pattern_classifier.fit(X_scaled, pattern_labels)
                
                logger.debug(f"ML models trained with {len(X)} time extension samples")
                
        except Exception as e:
            logger.error(f"Error training ML models: {str(e)}")
    
    def _calculate_extension_effectiveness(self, extension: TimeExtensionPoint, data: pd.DataFrame) -> float:
        """
        Calculate the effectiveness of a time extension point
        """
        try:
            ext_idx = extension.extension_index
            
            if ext_idx >= len(data) - 5:
                return 0.5  # Neutral for points too close to end
            
            # Look for price reaction around the extension point
            window_start = max(0, ext_idx - 2)
            window_end = min(len(data), ext_idx + 5)
            
            price_window = data['close'].iloc[window_start:window_end]
            volume_window = data['volume'].iloc[window_start:window_end]
            
            if len(price_window) < 3:
                return 0.5
            
            # Calculate price volatility around the point
            price_std = price_window.std()
            avg_std = data['close'].rolling(20).std().iloc[ext_idx]
            
            if pd.isna(avg_std) or avg_std == 0:
                volatility_ratio = 1.0
            else:
                volatility_ratio = price_std / avg_std
            
            # Calculate volume spike
            avg_volume = volume_window.mean()
            normal_volume = data['volume'].rolling(20).mean().iloc[ext_idx]
            
            if pd.isna(normal_volume) or normal_volume == 0:
                volume_ratio = 1.0
            else:
                volume_ratio = avg_volume / normal_volume
            
            # Combine factors
            effectiveness = (volatility_ratio * 0.6 + volume_ratio * 0.4) / 2
            return min(effectiveness, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating extension effectiveness: {str(e)}")
            return 0.5
    
    def _classify_extension_pattern(self, extension: TimeExtensionPoint, data: pd.DataFrame) -> int:
        """
        Classify the pattern type of a time extension (for ML training)
        """
        try:
            ext_idx = extension.extension_index
            
            if ext_idx >= len(data) - 3:
                return 0  # Neutral
            
            # Analyze price action pattern
            before_price = data['close'].iloc[max(0, ext_idx - 2)]
            at_price = data['close'].iloc[ext_idx]
            after_price = data['close'].iloc[min(len(data) - 1, ext_idx + 2)]
            
            # Classify patterns
            if at_price > before_price and at_price > after_price:
                return 1  # Reversal high
            elif at_price < before_price and at_price < after_price:
                return 2  # Reversal low
            elif (at_price - before_price) * (after_price - at_price) > 0:
                return 3  # Continuation
            else:
                return 0  # Neutral
            
        except Exception as e:
            logger.error(f"Error classifying extension pattern: {str(e)}")
            return 0
    
    def _predict_extension_confidence(self, data: pd.DataFrame, extension: TimeExtensionPoint) -> float:
        """
        Use machine learning to predict extension confidence
        """
        try:
            if self.ml_model is None:
                return extension.strength
            
            features = self._prepare_ml_features(data, extension)
            features_scaled = self.scaler.transform(features)
            prediction = self.ml_model.predict(features_scaled)[0]
            
            return min(prediction, 1.0)
            
        except Exception as e:
            logger.error(f"Error predicting extension confidence: {str(e)}")
            return extension.strength
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Fibonacci time extensions with advanced analysis
        """
        try:
            # Detect significant swing points
            significant_swings = self._detect_significant_swings(data)
            
            if len(significant_swings) < 2:
                return {
                    'time_extensions': [],
                    'time_clusters': [],
                    'time_patterns': [],
                    'future_projections': [],
                    'current_time_index': len(data) - 1,
                    'signal_strength': 0.0,
                    'dominant_cycle': None,
                    'next_significant_time': None
                }
            
            # Calculate time extensions
            self.time_extensions = self._calculate_time_extensions(significant_swings, data)
            
            if not self.time_extensions:
                return {
                    'time_extensions': [],
                    'time_clusters': [],
                    'time_patterns': [],
                    'future_projections': [],
                    'current_time_index': len(data) - 1,
                    'signal_strength': 0.0,
                    'dominant_cycle': None,
                    'next_significant_time': None
                }
            
            # Train ML models
            self._train_ml_models(data, self.time_extensions)
            
            # Update ML confidence for all extensions
            for extension in self.time_extensions:
                extension.ml_confidence = self._predict_extension_confidence(data, extension)
                # Combine original strength with ML prediction
                extension.strength = (extension.strength + extension.ml_confidence) / 2
            
            # Detect time clusters
            self.time_clusters = self._detect_time_clusters(self.time_extensions)
            
            # Detect time patterns
            self.time_patterns = self._detect_time_patterns(data, self.time_extensions)
            
            # Generate future projections
            future_projections = self._generate_future_projections(data)
            
            # Find next significant time
            current_index = len(data) - 1
            next_significant_time = self._find_next_significant_time(current_index)
            
            # Identify dominant cycle
            dominant_cycle = self._identify_dominant_cycle()
            
            # Calculate overall signal strength
            signal_strength = self._calculate_overall_signal_strength(current_index)
            
            # Prepare result
            result = {
                'time_extensions': [self._extension_to_dict(ext) for ext in self.time_extensions],
                'time_clusters': [self._cluster_to_dict(cluster) for cluster in self.time_clusters],
                'time_patterns': [self._pattern_to_dict(pattern) for pattern in self.time_patterns],
                'future_projections': future_projections,
                'current_time_index': current_index,
                'signal_strength': signal_strength,
                'dominant_cycle': dominant_cycle,
                'next_significant_time': next_significant_time,
                'ml_models_active': self.ml_model is not None,
                'fibonacci_numbers_used': self.parameters['fibonacci_numbers'],
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            self.last_calculation = result
            return result
            
        except Exception as e:
            logger.error(f"Error in Fibonacci time extension calculation: {str(e)}")
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="fibonacci_time_extension_calculation",
                message=str(e)
            )
    
    def _generate_future_projections(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate future time extension projections
        """
        try:
            projections = []
            current_index = len(data) - 1
            
            # Project based on strongest patterns
            for pattern in sorted(self.time_patterns, key=lambda p: p.strength, reverse=True)[:3]:
                if pattern.predicted_next:
                    for next_time in pattern.predicted_next:
                        if next_time > current_index:
                            projection = {
                                'projected_time_index': next_time,
                                'pattern_basis': pattern.pattern_type,
                                'fibonacci_basis': pattern.fibonacci_basis,
                                'confidence': pattern.strength,
                                'pattern_id': pattern.pattern_id
                            }
                            projections.append(projection)
            
            # Add projections from strongest clusters
            for cluster in sorted(self.time_clusters, key=lambda c: c.cluster_strength, reverse=True)[:2]:
                # Project next occurrence based on dominant Fibonacci
                next_projection = current_index + cluster.dominant_fibonacci
                projection = {
                    'projected_time_index': next_projection,
                    'pattern_basis': 'cluster_extension',
                    'fibonacci_basis': [cluster.dominant_fibonacci],
                    'confidence': cluster.cluster_strength,
                    'pattern_id': cluster.cluster_id
                }
                projections.append(projection)
            
            return sorted(projections, key=lambda p: p['projected_time_index'])[:10]
            
        except Exception as e:
            logger.error(f"Error generating future projections: {str(e)}")
            return []
    
    def _find_next_significant_time(self, current_index: int) -> Optional[Dict[str, Any]]:
        """
        Find the next most significant time point
        """
        try:
            future_extensions = [ext for ext in self.time_extensions 
                               if ext.extension_index > current_index]
            
            if not future_extensions:
                return None
            
            # Find the strongest future extension
            strongest_extension = max(future_extensions, key=lambda ext: ext.strength)
            
            return {
                'time_index': strongest_extension.extension_index,
                'timestamp': strongest_extension.extension_timestamp.isoformat(),
                'strength': strongest_extension.strength,
                'fibonacci_number': strongest_extension.fibonacci_number,
                'fibonacci_ratio': strongest_extension.fibonacci_ratio,
                'periods_ahead': strongest_extension.extension_index - current_index
            }
            
        except Exception as e:
            logger.error(f"Error finding next significant time: {str(e)}")
            return None
    
    def _identify_dominant_cycle(self) -> Optional[Dict[str, Any]]:
        """
        Identify the dominant cycle from patterns
        """
        try:
            cycle_patterns = [p for p in self.time_patterns if p.pattern_type == 'cycle']
            
            if not cycle_patterns:
                return None
            
            strongest_cycle = max(cycle_patterns, key=lambda p: p.strength)
            
            return {
                'cycle_length': strongest_cycle.period_length,
                'strength': strongest_cycle.strength,
                'fibonacci_basis': strongest_cycle.fibonacci_basis,
                'pattern_id': strongest_cycle.pattern_id
            }
            
        except Exception as e:
            logger.error(f"Error identifying dominant cycle: {str(e)}")
            return None
    
    def _calculate_overall_signal_strength(self, current_index: int) -> float:
        """
        Calculate overall signal strength based on nearby time extensions
        """
        try:
            if not self.time_extensions:
                return 0.0
            
            # Find extensions near current time
            nearby_extensions = [
                ext for ext in self.time_extensions
                if abs(ext.extension_index - current_index) <= 5
            ]
            
            if not nearby_extensions:
                return 0.0
            
            # Calculate weighted strength
            total_strength = sum(ext.strength for ext in nearby_extensions)
            confluence_bonus = len(nearby_extensions) * 0.1
            
            return min(total_strength + confluence_bonus, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating overall signal strength: {str(e)}")
            return 0.0
    
    def _extension_to_dict(self, extension: TimeExtensionPoint) -> Dict[str, Any]:
        """Convert TimeExtensionPoint to dictionary"""
        return {
            'base_swing_index': extension.base_swing_index,
            'base_swing_timestamp': extension.base_swing_timestamp.isoformat(),
            'extension_index': extension.extension_index,
            'extension_timestamp': extension.extension_timestamp.isoformat(),
            'fibonacci_number': extension.fibonacci_number,
            'fibonacci_ratio': extension.fibonacci_ratio,
            'time_periods': extension.time_periods,
            'strength': extension.strength,
            'price_at_point': extension.price_at_point,
            'volume_at_point': extension.volume_at_point,
            'volatility_score': extension.volatility_score,
            'ml_confidence': extension.ml_confidence
        }
    
    def _cluster_to_dict(self, cluster: TimeCluster) -> Dict[str, Any]:
        """Convert TimeCluster to dictionary"""
        return {
            'cluster_id': cluster.cluster_id,
            'center_index': cluster.center_index,
            'center_timestamp': cluster.center_timestamp.isoformat(),
            'cluster_strength': cluster.cluster_strength,
            'confluence_count': cluster.confluence_count,
            'price_range': cluster.price_range,
            'dominant_fibonacci': cluster.dominant_fibonacci,
            'point_count': len(cluster.points)
        }
    
    def _pattern_to_dict(self, pattern: TimePattern) -> Dict[str, Any]:
        """Convert TimePattern to dictionary"""
        return {
            'pattern_id': pattern.pattern_id,
            'pattern_type': pattern.pattern_type,
            'start_index': pattern.start_index,
            'end_index': pattern.end_index,
            'period_length': pattern.period_length,
            'strength': pattern.strength,
            'fibonacci_basis': pattern.fibonacci_basis,
            'predicted_next': pattern.predicted_next
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """
        Generate trading signals based on Fibonacci time extension analysis
        """
        try:
            if value['signal_strength'] < 0.3:
                return SignalType.NEUTRAL, 0.0
            
            current_index = value['current_time_index']
            signal_strength = value['signal_strength']
            
            # Check for nearby time extensions
            nearby_extensions = []
            for ext_dict in value['time_extensions']:
                if abs(ext_dict['extension_index'] - current_index) <= 3:
                    nearby_extensions.append(ext_dict)
            
            if not nearby_extensions:
                return SignalType.NEUTRAL, 0.0
            
            # Analyze time extension signals
            buy_signals = 0
            sell_signals = 0
            total_confidence = 0.0
            
            for ext in nearby_extensions:
                confidence = ext['strength'] * ext['ml_confidence']
                total_confidence += confidence
                
                # Check if we're approaching a time extension
                if ext['extension_index'] > current_index:
                    # Time extension ahead - potential reversal point
                    if ext['extension_index'] - current_index <= 2:
                        # Very close to time extension
                        recent_momentum = data['close'].pct_change(3).iloc[-1]
                        if recent_momentum > 0:
                            sell_signals += 1  # Uptrend might reverse
                        else:
                            buy_signals += 1   # Downtrend might reverse
                
                # Check if we're at a time extension
                elif abs(ext['extension_index'] - current_index) <= 1:
                    # At time extension - reversal likely
                    recent_momentum = data['close'].pct_change(5).iloc[-1]
                    if recent_momentum > 0:
                        sell_signals += 1  # Reversal from uptrend
                    else:
                        buy_signals += 1   # Reversal from downtrend
            
            if total_confidence == 0:
                return SignalType.NEUTRAL, 0.0
            
            avg_confidence = total_confidence / len(nearby_extensions)
            
            # Determine signal direction
            if buy_signals > sell_signals:
                return SignalType.BUY, min(signal_strength * avg_confidence, 1.0)
            elif sell_signals > buy_signals:
                return SignalType.SELL, min(signal_strength * avg_confidence, 1.0)
            else:
                # Check dominant cycle for trend bias
                if value.get('dominant_cycle') and value['dominant_cycle']['strength'] > 0.6:
                    # Use recent price momentum as tiebreaker
                    recent_momentum = data['close'].pct_change(3).iloc[-1]
                    if recent_momentum > 0:
                        return SignalType.BUY, signal_strength * 0.6
                    else:
                        return SignalType.SELL, signal_strength * 0.6
            
            return SignalType.NEUTRAL, 0.0
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        
        time_metadata = {
            'time_extensions_count': len(self.time_extensions),
            'time_clusters_count': len(self.time_clusters),
            'time_patterns_count': len(self.time_patterns),
            'ml_models_trained': self.ml_model is not None,
            'fibonacci_numbers_used': len(self.parameters['fibonacci_numbers']),
            'fibonacci_intervals_generated': len(self.fibonacci_intervals)
        }
        
        base_metadata.update(time_metadata)
        return base_metadata


def create_fibonacci_time_extension_indicator(parameters: Optional[Dict[str, Any]] = None) -> FibonacciTimeExtensionIndicator:
    """
    Factory function to create a FibonacciTimeExtensionIndicator instance
    
    Args:
        parameters: Optional dictionary of parameters to customize the indicator
        
    Returns:
        Configured FibonacciTimeExtensionIndicator instance
    """
    return FibonacciTimeExtensionIndicator(parameters=parameters)


# Example usage
if __name__ == "__main__":
    # Create sample data with Fibonacci-based time patterns
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1h')
    
    # Generate cyclical price data with Fibonacci time influences
    base_trend = np.linspace(100, 130, len(dates))
    
    # Add Fibonacci-based cycles
    fib_cycle_1 = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 55)  # 55-period cycle
    fib_cycle_2 = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 89)  # 89-period cycle
    noise = np.random.randn(len(dates)) * 0.8
    
    prices = base_trend + fib_cycle_1 + fib_cycle_2 + noise
    
    sample_data = pd.DataFrame({
        'high': prices + np.random.uniform(0, 1.5, len(dates)),
        'low': prices - np.random.uniform(0, 1.5, len(dates)),
        'close': prices,
        'volume': np.random.uniform(1000, 15000, len(dates))
    }, index=dates)
    
    # Test the indicator
    indicator = create_fibonacci_time_extension_indicator({
        'fibonacci_numbers': FIBONACCI_NUMBERS[:10],
        'cycle_detection': True,
        'harmonic_analysis': True,
        'max_time_extensions': 15
    })
    
    try:
        result = indicator.calculate(sample_data)
        print("Fibonacci Time Extension Calculation Result:")
        print(f"Signal: {result.signal}, Confidence: {result.confidence:.3f}")
        print(f"Time extensions: {len(result.value.get('time_extensions', []))}")
        print(f"Signal strength: {result.value.get('signal_strength', 0):.3f}")
        
        if result.value.get('next_significant_time'):
            next_time = result.value['next_significant_time']
            print(f"Next significant time: {next_time['periods_ahead']} periods ahead")
            print(f"Strength: {next_time['strength']:.3f}")
        
        if result.value.get('dominant_cycle'):
            cycle = result.value['dominant_cycle']
            print(f"Dominant cycle: {cycle['cycle_length']} periods, strength: {cycle['strength']:.3f}")
            
    except Exception as e:
        print(f"Error testing indicator: {str(e)}")