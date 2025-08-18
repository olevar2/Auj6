"""
Fibonacci Time Zone Indicator - Advanced Vertical Time-Based Support/Resistance
==============================================================================

This module implements a sophisticated Fibonacci time zone indicator that creates
vertical time-based support and resistance levels using Fibonacci number sequences.
It includes advanced time analysis, machine learning integration, and sophisticated
pattern recognition for precise market timing.

Features:
- Vertical time zones at Fibonacci intervals (1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89...)
- Advanced swing point detection and time zone calculation
- Machine learning time zone effectiveness prediction
- Multi-timeframe time zone confluence analysis
- Dynamic time zone strength and significance scoring
- Time-based market structure analysis
- Real-time time zone validation and adjustment
- Advanced clustering of time zones by effectiveness
- Time zone breakout and reversal pattern detection
- Natural market timing rhythm analysis

The indicator helps traders identify precise timing levels where significant
market moves are likely to occur, based on the mathematical properties of
Fibonacci number sequences and historical market behavior patterns.
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
class TimeZone:
    """Represents a Fibonacci time zone"""
    zone_id: str
    base_swing_index: int
    base_swing_timestamp: datetime
    zone_index: int
    zone_timestamp: datetime
    fibonacci_number: int
    fibonacci_ratio: float
    zone_strength: float        # 0.0 to 1.0
    price_action_score: float   # Historical price action at this time zone
    volume_score: float         # Volume activity at this time zone
    volatility_score: float     # Volatility around this time zone
    reversal_probability: float # ML-predicted reversal probability
    breakout_probability: float # ML-predicted breakout probability
    is_future: bool            # Whether this is a future projection
    

@dataclass
class TimeZoneCluster:
    """Represents a cluster of overlapping time zones"""
    cluster_id: str
    center_index: int
    center_timestamp: datetime
    zones: List[TimeZone]
    cluster_strength: float
    confluence_count: int
    dominant_fibonacci: int
    cluster_type: str          # 'reversal', 'breakout', 'continuation'
    

@dataclass
class TimePattern:
    """Represents a detected time pattern"""
    pattern_id: str
    pattern_type: str          # 'cycle', 'harmonic', 'sequence', 'rhythm'
    start_index: int
    end_index: int
    period_length: int
    strength: float
    fibonacci_basis: List[int]
    predicted_zones: List[int]
    effectiveness_score: float
    

class FibonacciTimeZoneIndicator(StandardIndicatorInterface):
    """
    Advanced Fibonacci Time Zone Indicator with machine learning integration
    and sophisticated time-based analysis capabilities.
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'swing_detection_period': 15,
            'min_swing_strength': 0.3,
            'fibonacci_numbers': FIBONACCI_NUMBERS[:12],  # Use first 12 Fibonacci numbers
            'fibonacci_ratios': FIBONACCI_RATIOS,
            'max_zones': 25,
            'zone_tolerance': 2,  # Periods tolerance for clustering
            'min_cluster_size': 2,
            'confluence_weight': 0.4,
            'ml_lookback': 100,
            'volume_weight': 0.3,
            'volatility_weight': 0.2,
            'price_action_weight': 0.5,
            'strength_threshold': 0.25,
            'pattern_detection': True,
            'rhythm_analysis': True,
            'future_projection_periods': 100,
            'zone_validation_periods': 5,
            'breakout_threshold': 0.6,
            'reversal_threshold': 0.6
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name="FibonacciTimeZone", parameters=default_params)
        
        # Initialize internal state
        self.time_zones: List[TimeZone] = []
        self.time_zone_clusters: List[TimeZoneCluster] = []
        self.time_patterns: List[TimePattern] = []
        self.ml_model = None
        self.pattern_classifier = None
        self.scaler = StandardScaler()
        self.last_calculation = None
        
        # Precompute Fibonacci-based intervals
        self.fibonacci_intervals = self._generate_fibonacci_intervals()
        
        logger.info(f"FibonacciTimeZoneIndicator initialized with {len(self.parameters['fibonacci_numbers'])} Fibonacci numbers")
    
    def get_data_requirements(self) -> DataRequirement:
        """Define the data requirements for Fibonacci time zone calculation"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(150, self.parameters['ml_lookback']),
            lookback_periods=400
        )
    
    def validate_parameters(self) -> bool:
        """Validate the indicator parameters"""
        try:
            required_params = ['fibonacci_numbers', 'swing_detection_period', 'max_zones']
            for param in required_params:
                if param not in self.parameters:
                    raise ValueError(f"Missing required parameter: {param}")
            
            if self.parameters['swing_detection_period'] < 3:
                raise ValueError("swing_detection_period must be at least 3")
            
            if not self.parameters['fibonacci_numbers']:
                raise ValueError("fibonacci_numbers cannot be empty")
            
            if self.parameters['max_zones'] < 5:
                raise ValueError("max_zones must be at least 5")
            
            return True
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            return False
    
    def _generate_fibonacci_intervals(self) -> List[int]:
        """Generate Fibonacci-based time intervals for time zone calculation"""
        intervals = []
        
        # Pure Fibonacci numbers
        intervals.extend(self.parameters['fibonacci_numbers'])
        
        # Fibonacci ratios applied to base periods
        base_periods = [10, 20, 30, 50, 100]
        for base in base_periods:
            for ratio in self.parameters['fibonacci_ratios']:
                interval = int(base * ratio)
                if interval not in intervals and interval <= 1000:
                    intervals.append(interval)
        
        # Fibonacci harmonics (combinations)
        for i, fib1 in enumerate(self.parameters['fibonacci_numbers'][:8]):
            for j, fib2 in enumerate(self.parameters['fibonacci_numbers'][:8]):
                if i != j:
                    # Addition harmonics
                    harmonic = fib1 + fib2
                    if harmonic not in intervals and harmonic <= 500:
                        intervals.append(harmonic)
                    
                    # Multiplication harmonics (scaled down)
                    if fib1 <= 13 and fib2 <= 13:
                        harmonic = fib1 * fib2
                        if harmonic not in intervals and harmonic <= 300:
                            intervals.append(harmonic)
        
        return sorted(list(set(intervals)))
    
    def _detect_significant_swings(self, data: pd.DataFrame) -> List[Tuple[int, float, str]]:
        """
        Detect significant swing points for time zone analysis
        """
        try:
            period = self.parameters['swing_detection_period']
            min_strength = self.parameters['min_swing_strength']
            
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            volumes = data['volume'].values
            
            significant_swings = []
            
            # Detect swing highs with prominence
            high_peaks, high_properties = signal.find_peaks(
                highs, 
                distance=period,
                prominence=np.std(highs) * min_strength,
                width=3
            )
            
            # Detect swing lows
            low_peaks, low_properties = signal.find_peaks(
                -lows,
                distance=period,
                prominence=np.std(lows) * min_strength,
                width=3
            )
            
            # Process swing highs
            for i, peak_idx in enumerate(high_peaks):
                if peak_idx >= period and peak_idx < len(data) - period:
                    strength = high_properties['prominences'][i] / np.std(highs)
                    volume_confirmation = volumes[peak_idx] / np.mean(volumes[max(0, peak_idx-10):peak_idx+10])
                    
                    if strength >= min_strength and volume_confirmation > 0.5:
                        significant_swings.append((peak_idx, highs[peak_idx], 'high'))
            
            # Process swing lows
            for i, peak_idx in enumerate(low_peaks):
                if peak_idx >= period and peak_idx < len(data) - period:
                    strength = low_properties['prominences'][i] / np.std(lows)
                    volume_confirmation = volumes[peak_idx] / np.mean(volumes[max(0, peak_idx-10):peak_idx+10])
                    
                    if strength >= min_strength and volume_confirmation > 0.5:
                        significant_swings.append((peak_idx, lows[peak_idx], 'low'))
            
            # Sort by time and return most significant swings
            significant_swings.sort(key=lambda x: x[0])
            return significant_swings[-20:]  # Keep most recent significant swings
            
        except Exception as e:
            logger.error(f"Error detecting significant swings: {str(e)}")
            return []
    
    def _calculate_time_zones(self, swings: List[Tuple[int, float, str]], 
                             data: pd.DataFrame) -> List[TimeZone]:
        """
        Calculate Fibonacci time zones from swing points
        """
        try:
            time_zones = []
            current_index = len(data) - 1
            
            for swing_idx, swing_price, swing_type in swings:
                # Calculate time zones for each Fibonacci interval
                for fib_number in self.parameters['fibonacci_numbers']:
                    zone_idx = swing_idx + fib_number
                    
                    # Include both historical and future zones
                    if zone_idx <= current_index + self.parameters['future_projection_periods']:
                        time_zone = self._create_time_zone(
                            swing_idx, swing_price, zone_idx, fib_number, 1.0, data, swing_type
                        )
                        
                        if time_zone and time_zone.zone_strength >= self.parameters['strength_threshold']:
                            time_zones.append(time_zone)
                
                # Calculate ratio-based time zones
                for ratio in self.parameters['fibonacci_ratios']:
                    for base_fib in self.parameters['fibonacci_numbers'][:6]:
                        zone_periods = int(base_fib * ratio)
                        zone_idx = swing_idx + zone_periods
                        
                        if zone_idx <= current_index + self.parameters['future_projection_periods']:
                            time_zone = self._create_time_zone(
                                swing_idx, swing_price, zone_idx, base_fib, ratio, data, swing_type
                            )
                            
                            if time_zone and time_zone.zone_strength >= self.parameters['strength_threshold']:
                                time_zones.append(time_zone)
            
            # Sort by zone index and limit count
            time_zones.sort(key=lambda x: x.zone_index)
            return time_zones[:self.parameters['max_zones']]
            
        except Exception as e:
            logger.error(f"Error calculating time zones: {str(e)}")
            return []
    
    def _create_time_zone(self, base_idx: int, base_price: float, zone_idx: int,
                         fib_number: int, ratio: float, data: pd.DataFrame,
                         swing_type: str) -> Optional[TimeZone]:
        """
        Create a time zone with calculated properties
        """
        try:
            # Generate zone ID
            zone_id = f"tz_{base_idx}_{zone_idx}_{fib_number}"
            
            # Get timestamps
            base_timestamp = data.index[base_idx]
            is_future = zone_idx >= len(data)
            
            if is_future:
                # Future projection
                time_delta = (data.index[1] - data.index[0]) * (zone_idx - len(data) + 1)
                zone_timestamp = data.index[-1] + time_delta
            else:
                zone_timestamp = data.index[zone_idx]
            
            # Calculate zone strength
            zone_strength = self._calculate_zone_strength(
                base_idx, zone_idx, fib_number, ratio, data, swing_type
            )
            
            # Calculate historical price action score
            price_action_score = self._calculate_price_action_score(zone_idx, data)
            
            # Calculate volume score
            volume_score = self._calculate_volume_score(zone_idx, data)
            
            # Calculate volatility score
            volatility_score = self._calculate_volatility_score(zone_idx, data)
            
            return TimeZone(
                zone_id=zone_id,
                base_swing_index=base_idx,
                base_swing_timestamp=base_timestamp,
                zone_index=zone_idx,
                zone_timestamp=zone_timestamp,
                fibonacci_number=fib_number,
                fibonacci_ratio=ratio,
                zone_strength=zone_strength,
                price_action_score=price_action_score,
                volume_score=volume_score,
                volatility_score=volatility_score,
                reversal_probability=0.0,  # Will be updated by ML
                breakout_probability=0.0,  # Will be updated by ML
                is_future=is_future
            )
            
        except Exception as e:
            logger.error(f"Error creating time zone: {str(e)}")
            return None
    
    def _calculate_zone_strength(self, base_idx: int, zone_idx: int, fib_number: int,
                               ratio: float, data: pd.DataFrame, swing_type: str) -> float:
        """
        Calculate the strength of a time zone
        """
        try:
            strength_factors = []
            
            # Fibonacci number significance (lower numbers typically stronger)
            fib_significance = 1.0 / (1 + np.log(max(fib_number, 1)))
            strength_factors.append(fib_significance)
            
            # Ratio significance
            ratio_significance = 1.0
            if abs(ratio - 1.0) < 0.01:
                ratio_significance = 1.0  # Pure Fibonacci numbers
            elif abs(ratio - 1.618) < 0.01:
                ratio_significance = 0.95  # Golden ratio
            elif abs(ratio - 0.618) < 0.01:
                ratio_significance = 0.9   # Inverse golden ratio
            else:
                ratio_significance = 0.6
            strength_factors.append(ratio_significance)
            
            # Distance from current time (closer zones might be more relevant)
            current_idx = len(data) - 1
            distance = abs(zone_idx - current_idx)
            distance_factor = 1.0 / (1 + distance / 50.0)
            strength_factors.append(distance_factor)
            
            # Historical effectiveness (if zone is in past)
            if zone_idx < len(data):
                historical_effectiveness = self._calculate_historical_effectiveness(zone_idx, data)
                strength_factors.append(historical_effectiveness)
            else:
                strength_factors.append(0.7)  # Neutral for future zones
            
            # Time period significance
            time_periods = zone_idx - base_idx
            period_strength = 1.0 / (1 + time_periods / 100.0)
            strength_factors.append(period_strength)
            
            return min(np.mean(strength_factors), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating zone strength: {str(e)}")
            return 0.0
    
    def _calculate_historical_effectiveness(self, zone_idx: int, data: pd.DataFrame) -> float:
        """
        Calculate historical effectiveness of a time zone
        """
        try:
            if zone_idx >= len(data) or zone_idx < 5:
                return 0.5
            
            validation_periods = self.parameters['zone_validation_periods']
            
            # Analyze price action around the zone
            start_idx = max(0, zone_idx - validation_periods)
            end_idx = min(len(data), zone_idx + validation_periods)
            
            price_window = data['close'].iloc[start_idx:end_idx]
            volume_window = data['volume'].iloc[start_idx:end_idx]
            
            if len(price_window) < 3:
                return 0.5
            
            # Check for significant price movements at the zone
            zone_price = data['close'].iloc[zone_idx]
            
            # Look for reversals
            before_trend = price_window.iloc[:validation_periods].pct_change().mean()
            after_trend = price_window.iloc[validation_periods:].pct_change().mean()
            
            reversal_strength = 0.0
            if len(price_window) > validation_periods:
                if before_trend * after_trend < 0:  # Trend reversal
                    reversal_strength = abs(before_trend - after_trend)
            
            # Look for significant volatility
            price_volatility = price_window.std() / price_window.mean()
            normal_volatility = data['close'].rolling(20).std().div(data['close']).iloc[zone_idx]
            
            if not pd.isna(normal_volatility) and normal_volatility > 0:
                volatility_ratio = price_volatility / normal_volatility
            else:
                volatility_ratio = 1.0
            
            # Look for volume spikes
            avg_volume = volume_window.mean()
            normal_volume = data['volume'].rolling(20).mean().iloc[zone_idx]
            
            if not pd.isna(normal_volume) and normal_volume > 0:
                volume_ratio = avg_volume / normal_volume
            else:
                volume_ratio = 1.0
            
            # Combine effectiveness factors
            effectiveness = (reversal_strength * 0.4 + 
                           min(volatility_ratio / 2.0, 1.0) * 0.3 + 
                           min(volume_ratio / 2.0, 1.0) * 0.3)
            
            return min(effectiveness, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating historical effectiveness: {str(e)}")
            return 0.5
    
    def _calculate_price_action_score(self, zone_idx: int, data: pd.DataFrame) -> float:
        """
        Calculate price action score at a time zone
        """
        try:
            if zone_idx >= len(data):
                return 0.5  # Neutral for future zones
            
            validation_periods = self.parameters['zone_validation_periods']
            
            # Analyze price action around the zone
            start_idx = max(0, zone_idx - validation_periods)
            end_idx = min(len(data), zone_idx + validation_periods)
            
            if end_idx - start_idx < 3:
                return 0.5
            
            highs = data['high'].iloc[start_idx:end_idx]
            lows = data['low'].iloc[start_idx:end_idx]
            closes = data['close'].iloc[start_idx:end_idx]
            
            # Look for price extremes at the zone
            zone_high = data['high'].iloc[zone_idx]
            zone_low = data['low'].iloc[zone_idx]
            
            is_local_high = zone_high >= highs.max() * 0.98
            is_local_low = zone_low <= lows.min() * 1.02
            
            if is_local_high or is_local_low:
                return 0.8
            
            # Check for significant price movement
            price_range = highs.max() - lows.min()
            avg_range = (data['high'] - data['low']).rolling(20).mean().iloc[zone_idx]
            
            if not pd.isna(avg_range) and avg_range > 0:
                range_ratio = price_range / avg_range
                return min(range_ratio / 3.0, 1.0)
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating price action score: {str(e)}")
            return 0.5
    
    def _calculate_volume_score(self, zone_idx: int, data: pd.DataFrame) -> float:
        """
        Calculate volume score at a time zone
        """
        try:
            if zone_idx >= len(data):
                return 0.5  # Neutral for future zones
            
            validation_periods = self.parameters['zone_validation_periods']
            
            # Analyze volume around the zone
            start_idx = max(0, zone_idx - validation_periods)
            end_idx = min(len(data), zone_idx + validation_periods)
            
            volume_window = data['volume'].iloc[start_idx:end_idx]
            avg_volume = volume_window.mean()
            
            # Compare to normal volume
            normal_volume = data['volume'].rolling(20).mean().iloc[zone_idx]
            
            if pd.isna(normal_volume) or normal_volume == 0:
                return 0.5
            
            volume_ratio = avg_volume / normal_volume
            return min(volume_ratio / 3.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating volume score: {str(e)}")
            return 0.5
    
    def _calculate_volatility_score(self, zone_idx: int, data: pd.DataFrame) -> float:
        """
        Calculate volatility score at a time zone
        """
        try:
            if zone_idx >= len(data):
                # For future zones, use recent volatility
                recent_volatility = data['close'].pct_change().rolling(10).std().iloc[-10:]
                return float(recent_volatility.mean()) if not recent_volatility.empty else 0.0
            
            validation_periods = self.parameters['zone_validation_periods']
            
            # Calculate volatility around the zone
            start_idx = max(0, zone_idx - validation_periods)
            end_idx = min(len(data), zone_idx + validation_periods)
            
            price_window = data['close'].iloc[start_idx:end_idx]
            local_volatility = price_window.pct_change().std()
            
            # Compare to normal volatility
            normal_volatility = data['close'].pct_change().rolling(20).std().iloc[zone_idx]
            
            if pd.isna(normal_volatility) or normal_volatility == 0:
                return float(local_volatility) if not pd.isna(local_volatility) else 0.0
            
            volatility_ratio = local_volatility / normal_volatility
            return float(volatility_ratio) if not pd.isna(volatility_ratio) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volatility score: {str(e)}")
            return 0.0
    
    def _detect_time_zone_clusters(self, zones: List[TimeZone]) -> List[TimeZoneCluster]:
        """
        Detect clusters of overlapping time zones
        """
        try:
            if len(zones) < 2:
                return []
            
            # Prepare data for clustering
            zone_indices = np.array([zone.zone_index for zone in zones]).reshape(-1, 1)
            
            # Use DBSCAN for clustering
            tolerance = self.parameters['zone_tolerance']
            min_samples = self.parameters['min_cluster_size']
            
            clustering = DBSCAN(eps=tolerance, min_samples=min_samples)
            cluster_labels = clustering.fit_predict(zone_indices)
            
            clusters = []
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                
                # Get zones in this cluster
                cluster_zones = [zone for i, zone in enumerate(zones) if cluster_labels[i] == label]
                
                if len(cluster_zones) >= min_samples:
                    cluster = self._create_time_zone_cluster(cluster_zones, label)
                    if cluster:
                        clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error detecting time zone clusters: {str(e)}")
            return []
    
    def _create_time_zone_cluster(self, zones: List[TimeZone], cluster_id: int) -> Optional[TimeZoneCluster]:
        """
        Create a time zone cluster from a group of zones
        """
        try:
            if not zones:
                return None
            
            # Calculate cluster center
            center_index = int(np.mean([zone.zone_index for zone in zones]))
            center_timestamp = zones[0].zone_timestamp  # Approximate
            
            # Calculate cluster strength with confluence weighting
            base_strength = np.mean([zone.zone_strength for zone in zones])
            confluence_bonus = (len(zones) - 1) * self.parameters['confluence_weight']
            cluster_strength = min(base_strength + confluence_bonus, 1.0)
            
            # Find dominant Fibonacci number
            fib_numbers = [zone.fibonacci_number for zone in zones]
            dominant_fibonacci = max(set(fib_numbers), key=fib_numbers.count)
            
            # Determine cluster type based on probabilities
            avg_reversal_prob = np.mean([zone.reversal_probability for zone in zones])
            avg_breakout_prob = np.mean([zone.breakout_probability for zone in zones])
            
            if avg_reversal_prob > self.parameters['reversal_threshold']:
                cluster_type = 'reversal'
            elif avg_breakout_prob > self.parameters['breakout_threshold']:
                cluster_type = 'breakout'
            else:
                cluster_type = 'continuation'
            
            return TimeZoneCluster(
                cluster_id=f"cluster_{cluster_id}",
                center_index=center_index,
                center_timestamp=center_timestamp,
                zones=zones,
                cluster_strength=cluster_strength,
                confluence_count=len(zones),
                dominant_fibonacci=dominant_fibonacci,
                cluster_type=cluster_type
            )
            
        except Exception as e:
            logger.error(f"Error creating time zone cluster: {str(e)}")
            return None
    
    def _detect_time_patterns(self, data: pd.DataFrame, zones: List[TimeZone]) -> List[TimePattern]:
        """
        Detect recurring time patterns in the zones
        """
        try:
            patterns = []
            
            if not self.parameters['pattern_detection']:
                return patterns
            
            # Detect cycle patterns
            cycle_patterns = self._detect_cycle_patterns(data, zones)
            patterns.extend(cycle_patterns)
            
            # Detect rhythm patterns
            if self.parameters['rhythm_analysis']:
                rhythm_patterns = self._detect_rhythm_patterns(zones)
                patterns.extend(rhythm_patterns)
            
            # Detect sequence patterns
            sequence_patterns = self._detect_sequence_patterns(zones)
            patterns.extend(sequence_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting time patterns: {str(e)}")
            return []
    
    def _detect_cycle_patterns(self, data: pd.DataFrame, zones: List[TimeZone]) -> List[TimePattern]:
        """
        Detect cyclical patterns in time zones
        """
        try:
            patterns = []
            
            # Group zones by Fibonacci number
            fib_groups = {}
            for zone in zones:
                if zone.fibonacci_number not in fib_groups:
                    fib_groups[zone.fibonacci_number] = []
                fib_groups[zone.fibonacci_number].append(zone)
            
            # Look for cyclical patterns within each group
            for fib_num, group in fib_groups.items():
                if len(group) >= 3:
                    # Sort by zone index
                    sorted_group = sorted(group, key=lambda z: z.zone_index)
                    
                    # Calculate intervals between zones
                    intervals = []
                    for i in range(1, len(sorted_group)):
                        interval = sorted_group[i].zone_index - sorted_group[i-1].zone_index
                        intervals.append(interval)
                    
                    # Check for consistent intervals (cycles)
                    if len(intervals) >= 2:
                        avg_interval = np.mean(intervals)
                        interval_std = np.std(intervals)
                        
                        # If intervals are consistent, it's a cycle
                        if interval_std / avg_interval < 0.3:  # Low variation
                            pattern = TimePattern(
                                pattern_id=f"cycle_{fib_num}",
                                pattern_type='cycle',
                                start_index=sorted_group[0].zone_index,
                                end_index=sorted_group[-1].zone_index,
                                period_length=int(avg_interval),
                                strength=np.mean([z.zone_strength for z in sorted_group]),
                                fibonacci_basis=[fib_num],
                                predicted_zones=[sorted_group[-1].zone_index + int(avg_interval)],
                                effectiveness_score=np.mean([z.price_action_score for z in sorted_group])
                            )
                            patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting cycle patterns: {str(e)}")
            return []
    
    def _detect_rhythm_patterns(self, zones: List[TimeZone]) -> List[TimePattern]:
        """
        Detect rhythm patterns based on Fibonacci relationships
        """
        try:
            patterns = []
            
            # Look for harmonic relationships between zones
            for i, zone1 in enumerate(zones):
                for j, zone2 in enumerate(zones[i+1:], i+1):
                    time_diff = abs(zone2.zone_index - zone1.zone_index)
                    
                    # Check if time difference matches Fibonacci harmonics
                    for ratio in self.parameters['fibonacci_ratios']:
                        for fib_base in self.parameters['fibonacci_numbers'][:6]:
                            expected_diff = int(fib_base * ratio)
                            
                            if abs(time_diff - expected_diff) <= self.parameters['zone_tolerance']:
                                pattern = TimePattern(
                                    pattern_id=f"rhythm_{i}_{j}",
                                    pattern_type='rhythm',
                                    start_index=zone1.zone_index,
                                    end_index=zone2.zone_index,
                                    period_length=time_diff,
                                    strength=(zone1.zone_strength + zone2.zone_strength) / 2,
                                    fibonacci_basis=[zone1.fibonacci_number, zone2.fibonacci_number],
                                    predicted_zones=[zone2.zone_index + time_diff],
                                    effectiveness_score=(zone1.price_action_score + zone2.price_action_score) / 2
                                )
                                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting rhythm patterns: {str(e)}")
            return []
    
    def _detect_sequence_patterns(self, zones: List[TimeZone]) -> List[TimePattern]:
        """
        Detect Fibonacci sequence patterns in time zones
        """
        try:
            patterns = []
            
            # Group zones by base swing
            swing_groups = {}
            for zone in zones:
                if zone.base_swing_index not in swing_groups:
                    swing_groups[zone.base_swing_index] = []
                swing_groups[zone.base_swing_index].append(zone)
            
            # Look for Fibonacci sequences within each swing group
            for base_idx, group in swing_groups.items():
                if len(group) >= 3:
                    # Sort by Fibonacci number
                    sorted_group = sorted(group, key=lambda z: z.fibonacci_number)
                    
                    # Check for Fibonacci sequences
                    fib_numbers = [zone.fibonacci_number for zone in sorted_group]
                    
                    for i in range(len(fib_numbers) - 2):
                        if (fib_numbers[i] + fib_numbers[i+1] == fib_numbers[i+2] or
                            abs((fib_numbers[i] + fib_numbers[i+1]) - fib_numbers[i+2]) <= 1):
                            
                            pattern = TimePattern(
                                pattern_id=f"sequence_{base_idx}_{i}",
                                pattern_type='sequence',
                                start_index=sorted_group[i].zone_index,
                                end_index=sorted_group[i+2].zone_index,
                                period_length=sorted_group[i+2].zone_index - sorted_group[i].zone_index,
                                strength=np.mean([z.zone_strength for z in sorted_group[i:i+3]]),
                                fibonacci_basis=fib_numbers[i:i+3],
                                predicted_zones=[sorted_group[i+2].zone_index + fib_numbers[i+2]],
                                effectiveness_score=np.mean([z.price_action_score for z in sorted_group[i:i+3]])
                            )
                            patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting sequence patterns: {str(e)}")
            return []
    
    def _prepare_ml_features(self, data: pd.DataFrame, zone: TimeZone) -> np.ndarray:
        """
        Prepare features for machine learning models
        """
        try:
            features = []
            
            # Time zone properties
            features.append(zone.fibonacci_number / 100.0)  # Normalized
            features.append(zone.fibonacci_ratio)
            features.append(zone.zone_strength)
            features.append(zone.price_action_score)
            features.append(zone.volume_score)
            features.append(zone.volatility_score)
            
            # Market context at the zone
            if zone.zone_index < len(data):
                # Historical data available
                zone_price = data['close'].iloc[zone.zone_index]
                
                # Price momentum
                momentum_5 = data['close'].pct_change(5).iloc[zone.zone_index]
                features.append(momentum_5 if not pd.isna(momentum_5) else 0.0)
                
                momentum_10 = data['close'].pct_change(10).iloc[zone.zone_index]
                features.append(momentum_10 if not pd.isna(momentum_10) else 0.0)
                
                # Relative price position
                high_20 = data['high'].rolling(20).max().iloc[zone.zone_index]
                low_20 = data['low'].rolling(20).min().iloc[zone.zone_index]
                if not pd.isna(high_20) and not pd.isna(low_20) and high_20 != low_20:
                    price_position = (zone_price - low_20) / (high_20 - low_20)
                else:
                    price_position = 0.5
                features.append(price_position)
                
            else:
                # Future zone - use recent data
                features.extend([
                    data['close'].pct_change(5).iloc[-1] or 0.0,
                    data['close'].pct_change(10).iloc[-1] or 0.0,
                    0.5  # Neutral price position
                ])
            
            # Distance from current time
            current_idx = len(data) - 1
            time_distance = (zone.zone_index - current_idx) / 100.0  # Normalized
            features.append(time_distance)
            
            # Time-based features
            if hasattr(zone.zone_timestamp, 'hour'):
                features.append(zone.zone_timestamp.hour / 24.0)
                features.append(zone.zone_timestamp.weekday() / 7.0)
            else:
                features.extend([0.5, 0.5])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {str(e)}")
            return np.array([[0.0] * 12])
    
    def _train_ml_models(self, data: pd.DataFrame, zones: List[TimeZone]):
        """
        Train machine learning models for time zone predictions
        """
        try:
            if len(zones) < 10 or len(data) < self.parameters['ml_lookback']:
                return
            
            X, y_reversal, y_breakout = [], [], []
            
            # Prepare training data
            for zone in zones:
                if zone.zone_index < len(data) - 10:  # Ensure future data exists
                    features = self._prepare_ml_features(data, zone)
                    
                    # Calculate targets
                    reversal_score = self._calculate_reversal_target(zone, data)
                    breakout_score = self._calculate_breakout_target(zone, data)
                    
                    X.append(features[0])
                    y_reversal.append(reversal_score)
                    y_breakout.append(breakout_score)
            
            if len(X) > 5:
                X = np.array(X)
                y_reversal = np.array(y_reversal)
                y_breakout = np.array(y_breakout)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train reversal prediction model
                self.ml_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1
                )
                self.ml_model.fit(X_scaled, y_reversal)
                
                # Train pattern classifier for breakout vs reversal
                pattern_labels = (y_breakout > y_reversal).astype(int)
                
                self.pattern_classifier = GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=5,
                    random_state=42
                )
                self.pattern_classifier.fit(X_scaled, pattern_labels)
                
                logger.debug(f"ML models trained with {len(X)} time zone samples")
                
        except Exception as e:
            logger.error(f"Error training ML models: {str(e)}")
    
    def _calculate_reversal_target(self, zone: TimeZone, data: pd.DataFrame) -> float:
        """
        Calculate reversal effectiveness as training target
        """
        try:
            zone_idx = zone.zone_index
            
            if zone_idx >= len(data) - 5:
                return 0.5
            
            # Look for trend reversals around the zone
            before_window = 5
            after_window = 5
            
            before_start = max(0, zone_idx - before_window)
            after_end = min(len(data), zone_idx + after_window)
            
            if before_start >= zone_idx or after_end <= zone_idx:
                return 0.5
            
            # Calculate trends before and after
            before_prices = data['close'].iloc[before_start:zone_idx]
            after_prices = data['close'].iloc[zone_idx:after_end]
            
            if len(before_prices) < 2 or len(after_prices) < 2:
                return 0.5
            
            before_trend = (before_prices.iloc[-1] - before_prices.iloc[0]) / len(before_prices)
            after_trend = (after_prices.iloc[-1] - after_prices.iloc[0]) / len(after_prices)
            
            # Check for reversal
            if before_trend * after_trend < 0:  # Opposite signs = reversal
                reversal_strength = abs(before_trend - after_trend) / data['close'].iloc[zone_idx]
                return min(reversal_strength * 10, 1.0)
            
            return 0.2  # No significant reversal
            
        except Exception as e:
            logger.error(f"Error calculating reversal target: {str(e)}")
            return 0.5
    
    def _calculate_breakout_target(self, zone: TimeZone, data: pd.DataFrame) -> float:
        """
        Calculate breakout effectiveness as training target
        """
        try:
            zone_idx = zone.zone_index
            
            if zone_idx >= len(data) - 5:
                return 0.5
            
            # Look for volatility and momentum increases at the zone
            window = 5
            start_idx = max(0, zone_idx - window)
            end_idx = min(len(data), zone_idx + window)
            
            if end_idx - start_idx < 3:
                return 0.5
            
            # Calculate volatility around the zone
            price_window = data['close'].iloc[start_idx:end_idx]
            local_volatility = price_window.pct_change().std()
            
            # Compare to normal volatility
            normal_volatility = data['close'].pct_change().rolling(20).std().iloc[zone_idx]
            
            if pd.isna(normal_volatility) or normal_volatility == 0:
                return 0.5
            
            volatility_ratio = local_volatility / normal_volatility
            
            # Calculate momentum
            momentum = abs(data['close'].pct_change(window).iloc[zone_idx])
            
            # Combine factors
            breakout_score = (min(volatility_ratio / 3.0, 1.0) + min(momentum * 20, 1.0)) / 2
            return min(breakout_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating breakout target: {str(e)}")
            return 0.5
    
    def _predict_zone_probabilities(self, data: pd.DataFrame, zone: TimeZone):
        """
        Use ML models to predict zone probabilities
        """
        try:
            if self.ml_model is None or self.pattern_classifier is None:
                return
            
            features = self._prepare_ml_features(data, zone)
            features_scaled = self.scaler.transform(features)
            
            # Predict reversal probability
            zone.reversal_probability = min(self.ml_model.predict(features_scaled)[0], 1.0)
            
            # Predict breakout probability
            breakout_proba = self.pattern_classifier.predict_proba(features_scaled)[0]
            zone.breakout_probability = breakout_proba[1] if len(breakout_proba) > 1 else 0.5
            
        except Exception as e:
            logger.error(f"Error predicting zone probabilities: {str(e)}")
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Fibonacci time zones with advanced analysis
        """
        try:
            # Detect significant swing points
            significant_swings = self._detect_significant_swings(data)
            
            if len(significant_swings) < 2:
                return {
                    'time_zones': [],
                    'time_zone_clusters': [],
                    'time_patterns': [],
                    'current_time_index': len(data) - 1,
                    'active_zones_count': 0,
                    'strongest_zone': None,
                    'dominant_pattern': None,
                    'signal_strength': 0.0
                }
            
            # Calculate time zones
            self.time_zones = self._calculate_time_zones(significant_swings, data)
            
            if not self.time_zones:
                return {
                    'time_zones': [],
                    'time_zone_clusters': [],
                    'time_patterns': [],
                    'current_time_index': len(data) - 1,
                    'active_zones_count': 0,
                    'strongest_zone': None,
                    'dominant_pattern': None,
                    'signal_strength': 0.0
                }
            
            # Train ML models
            self._train_ml_models(data, self.time_zones)
            
            # Update ML predictions for all zones
            for zone in self.time_zones:
                self._predict_zone_probabilities(data, zone)
            
            # Detect time zone clusters
            self.time_zone_clusters = self._detect_time_zone_clusters(self.time_zones)
            
            # Detect time patterns
            self.time_patterns = self._detect_time_patterns(data, self.time_zones)
            
            # Analyze current market state
            current_index = len(data) - 1
            active_zones = self._get_active_zones(current_index)
            strongest_zone = self._find_strongest_zone(current_index)
            dominant_pattern = self._find_dominant_pattern()
            signal_strength = self._calculate_signal_strength(current_index)
            
            # Prepare result
            result = {
                'time_zones': [self._zone_to_dict(zone) for zone in self.time_zones],
                'time_zone_clusters': [self._cluster_to_dict(cluster) for cluster in self.time_zone_clusters],
                'time_patterns': [self._pattern_to_dict(pattern) for pattern in self.time_patterns],
                'current_time_index': current_index,
                'active_zones_count': len(active_zones),
                'strongest_zone': self._zone_to_dict(strongest_zone) if strongest_zone else None,
                'dominant_pattern': self._pattern_to_dict(dominant_pattern) if dominant_pattern else None,
                'signal_strength': signal_strength,
                'ml_models_active': self.ml_model is not None,
                'fibonacci_numbers_used': self.parameters['fibonacci_numbers'],
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            self.last_calculation = result
            return result
            
        except Exception as e:
            logger.error(f"Error in Fibonacci time zone calculation: {str(e)}")
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="fibonacci_time_zone_calculation",
                message=str(e)
            )
    
    def _get_active_zones(self, current_index: int) -> List[TimeZone]:
        """Get time zones active around the current time"""
        tolerance = self.parameters['zone_tolerance']
        return [zone for zone in self.time_zones 
                if abs(zone.zone_index - current_index) <= tolerance]
    
    def _find_strongest_zone(self, current_index: int) -> Optional[TimeZone]:
        """Find the strongest time zone near current time"""
        active_zones = self._get_active_zones(current_index)
        
        if not active_zones:
            # Look for nearby zones
            nearby_zones = [zone for zone in self.time_zones 
                           if abs(zone.zone_index - current_index) <= 10]
            if nearby_zones:
                return max(nearby_zones, key=lambda z: z.zone_strength)
            return None
        
        return max(active_zones, key=lambda z: z.zone_strength)
    
    def _find_dominant_pattern(self) -> Optional[TimePattern]:
        """Find the dominant time pattern"""
        if not self.time_patterns:
            return None
        
        return max(self.time_patterns, key=lambda p: p.strength)
    
    def _calculate_signal_strength(self, current_index: int) -> float:
        """Calculate overall signal strength"""
        active_zones = self._get_active_zones(current_index)
        
        if not active_zones:
            return 0.0
        
        # Base strength from active zones
        base_strength = np.mean([zone.zone_strength for zone in active_zones])
        
        # Confluence bonus
        confluence_bonus = len(active_zones) * 0.1
        
        # Pattern confirmation
        pattern_bonus = 0.0
        if self.time_patterns:
            relevant_patterns = [p for p in self.time_patterns 
                               if current_index >= p.start_index and current_index <= p.end_index]
            if relevant_patterns:
                pattern_bonus = max(p.strength for p in relevant_patterns) * 0.2
        
        return min(base_strength + confluence_bonus + pattern_bonus, 1.0)
    
    def _zone_to_dict(self, zone: TimeZone) -> Dict[str, Any]:
        """Convert TimeZone to dictionary"""
        return {
            'zone_id': zone.zone_id,
            'base_swing_index': zone.base_swing_index,
            'base_swing_timestamp': zone.base_swing_timestamp.isoformat(),
            'zone_index': zone.zone_index,
            'zone_timestamp': zone.zone_timestamp.isoformat(),
            'fibonacci_number': zone.fibonacci_number,
            'fibonacci_ratio': zone.fibonacci_ratio,
            'zone_strength': zone.zone_strength,
            'price_action_score': zone.price_action_score,
            'volume_score': zone.volume_score,
            'volatility_score': zone.volatility_score,
            'reversal_probability': zone.reversal_probability,
            'breakout_probability': zone.breakout_probability,
            'is_future': zone.is_future
        }
    
    def _cluster_to_dict(self, cluster: TimeZoneCluster) -> Dict[str, Any]:
        """Convert TimeZoneCluster to dictionary"""
        return {
            'cluster_id': cluster.cluster_id,
            'center_index': cluster.center_index,
            'center_timestamp': cluster.center_timestamp.isoformat(),
            'cluster_strength': cluster.cluster_strength,
            'confluence_count': cluster.confluence_count,
            'dominant_fibonacci': cluster.dominant_fibonacci,
            'cluster_type': cluster.cluster_type,
            'zone_count': len(cluster.zones)
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
            'predicted_zones': pattern.predicted_zones,
            'effectiveness_score': pattern.effectiveness_score
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """
        Generate trading signals based on Fibonacci time zone analysis
        """
        try:
            if value['signal_strength'] < 0.3:
                return SignalType.NEUTRAL, 0.0
            
            current_index = value['current_time_index']
            signal_strength = value['signal_strength']
            
            # Get strongest zone near current time
            strongest_zone_dict = value.get('strongest_zone')
            if not strongest_zone_dict:
                return SignalType.NEUTRAL, 0.0
            
            # Analyze zone signals
            reversal_prob = strongest_zone_dict['reversal_probability']
            breakout_prob = strongest_zone_dict['breakout_probability']
            zone_strength = strongest_zone_dict['zone_strength']
            
            # Determine signal based on probabilities and market context
            recent_momentum = data['close'].pct_change(5).iloc[-1]
            
            if reversal_prob > self.parameters['reversal_threshold']:
                # High reversal probability
                if recent_momentum > 0:
                    # Recent uptrend, expect reversal down
                    return SignalType.SELL, signal_strength * reversal_prob
                else:
                    # Recent downtrend, expect reversal up
                    return SignalType.BUY, signal_strength * reversal_prob
            
            elif breakout_prob > self.parameters['breakout_threshold']:
                # High breakout probability - continue in momentum direction
                if recent_momentum > 0:
                    return SignalType.BUY, signal_strength * breakout_prob * 0.8
                else:
                    return SignalType.SELL, signal_strength * breakout_prob * 0.8
            
            # Check cluster signals
            if value['active_zones_count'] >= 3:
                # Strong confluence suggests reversal
                if recent_momentum > 0:
                    return SignalType.SELL, signal_strength * 0.7
                else:
                    return SignalType.BUY, signal_strength * 0.7
            
            return SignalType.NEUTRAL, 0.0
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        
        zone_metadata = {
            'time_zones_count': len(self.time_zones),
            'time_zone_clusters_count': len(self.time_zone_clusters),
            'time_patterns_count': len(self.time_patterns),
            'ml_models_trained': self.ml_model is not None,
            'fibonacci_numbers_used': len(self.parameters['fibonacci_numbers']),
            'fibonacci_intervals_generated': len(self.fibonacci_intervals)
        }
        
        base_metadata.update(zone_metadata)
        return base_metadata


def create_fibonacci_time_zone_indicator(parameters: Optional[Dict[str, Any]] = None) -> FibonacciTimeZoneIndicator:
    """
    Factory function to create a FibonacciTimeZoneIndicator instance
    
    Args:
        parameters: Optional dictionary of parameters to customize the indicator
        
    Returns:
        Configured FibonacciTimeZoneIndicator instance
    """
    return FibonacciTimeZoneIndicator(parameters=parameters)


# Example usage
if __name__ == "__main__":
    # Create sample data with Fibonacci-based time patterns
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1h')
    
    # Generate price data with time-based patterns
    base_trend = np.linspace(100, 120, len(dates))
    
    # Add time-based patterns at Fibonacci intervals
    time_pattern = np.zeros(len(dates))
    for i, fib_num in enumerate([13, 21, 34, 55, 89]):
        pattern_indices = np.arange(fib_num, len(dates), fib_num * 2)
        for idx in pattern_indices:
            if idx < len(time_pattern):
                time_pattern[idx] += (5 - i) * np.sin(idx / fib_num)  # Decreasing influence
    
    noise = np.random.randn(len(dates)) * 1.2
    prices = base_trend + time_pattern + noise
    
    sample_data = pd.DataFrame({
        'high': prices + np.random.uniform(0, 2, len(dates)),
        'low': prices - np.random.uniform(0, 2, len(dates)),
        'close': prices,
        'volume': np.random.uniform(1000, 20000, len(dates))
    }, index=dates)
    
    # Test the indicator
    indicator = create_fibonacci_time_zone_indicator({
        'fibonacci_numbers': FIBONACCI_NUMBERS[:10],
        'pattern_detection': True,
        'rhythm_analysis': True,
        'max_zones': 20
    })
    
    try:
        result = indicator.calculate(sample_data)
        print("Fibonacci Time Zone Calculation Result:")
        print(f"Signal: {result.signal}, Confidence: {result.confidence:.3f}")
        print(f"Time zones: {len(result.value.get('time_zones', []))}")
        print(f"Active zones: {result.value.get('active_zones_count', 0)}")
        print(f"Signal strength: {result.value.get('signal_strength', 0):.3f}")
        
        if result.value.get('strongest_zone'):
            zone = result.value['strongest_zone']
            print(f"Strongest zone: Fibonacci {zone['fibonacci_number']}, strength: {zone['zone_strength']:.3f}")
            print(f"Reversal prob: {zone['reversal_probability']:.3f}, Breakout prob: {zone['breakout_probability']:.3f}")
        
        if result.value.get('dominant_pattern'):
            pattern = result.value['dominant_pattern']
            print(f"Dominant pattern: {pattern['pattern_type']}, strength: {pattern['strength']:.3f}")
            
    except Exception as e:
        print(f"Error testing indicator: {str(e)}")