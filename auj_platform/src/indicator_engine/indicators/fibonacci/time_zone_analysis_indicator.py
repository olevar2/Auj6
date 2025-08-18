"""
Time Zone Analysis Indicator

This comprehensive indicator combines multiple time-based Fibonacci techniques for advanced market timing
analysis. It integrates Fibonacci time zones, time extensions, cycles, and rhythms to provide sophisticated
temporal analysis for market turning points and trend changes.

Author: Trading Platform Team
Date: 2025-06-22
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import ValidationError, CalculationError

logger = logging.getLogger(__name__)


class TimeAnalysisType(Enum):
    """Types of time-based analysis methods."""
    FIBONACCI_TIME_ZONES = "fibonacci_time_zones"
    FIBONACCI_TIME_EXTENSIONS = "fibonacci_time_extensions"
    FIBONACCI_CYCLES = "fibonacci_cycles"
    MARKET_RHYTHMS = "market_rhythms"
    HARMONIC_TIME_CYCLES = "harmonic_time_cycles"
    SEASONAL_PATTERNS = "seasonal_patterns"
    LUNAR_CYCLES = "lunar_cycles"


class TimeSignalType(Enum):
    """Types of time-based signals."""
    REVERSAL_ZONE = "reversal_zone"
    CONTINUATION_ZONE = "continuation_zone"
    ACCELERATION_ZONE = "acceleration_zone"
    CONSOLIDATION_ZONE = "consolidation_zone"
    BREAKOUT_ZONE = "breakout_zone"


@dataclass
class TimeZone:
    """Container for time zone analysis data."""
    zone_type: TimeAnalysisType
    signal_type: TimeSignalType
    start_time: int
    end_time: int
    center_time: int
    intensity: float
    confidence: float
    fibonacci_ratio: Optional[float]
    cycle_length: Optional[int]
    harmonic_frequency: Optional[float]
    supporting_evidence: List[str]


@dataclass
class TimeConfluence:
    """Container for time confluence analysis."""
    confluence_time: int
    zone_count: int
    total_intensity: float
    average_confidence: float
    dominant_signal: TimeSignalType
    contributing_zones: List[TimeZone]
    strength_score: float


class TimeZoneAnalysisIndicator(StandardIndicatorInterface):
    """
    Comprehensive Time Zone Analysis Indicator

    This sophisticated indicator combines multiple time-based Fibonacci techniques to provide
    advanced market timing analysis. It integrates various temporal patterns including:
    - Fibonacci time zones and extensions
    - Market cycle analysis
    - Harmonic time relationships
    - Seasonal and lunar patterns
    - Machine learning-enhanced pattern recognition

    Key Features:
    - Multi-dimensional time analysis
    - Confluence zone detection
    - Cycle rhythm analysis
    - Harmonic frequency detection
    - Machine learning pattern enhancement
    - Real-time timing signals
    - Historical pattern validation
    """

    def __init__(self,
                 lookback_period: int = 200,
                 fibonacci_sequence: Optional[List[int]] = None,
                 fibonacci_ratios: Optional[List[float]] = None,
                 min_cycle_length: int = 5,
                 max_cycle_length: int = 89,
                 confluence_threshold: int = 3,
                 intensity_threshold: float = 0.6,
                 seasonal_analysis: bool = True,
                 lunar_analysis: bool = True,
                 harmonic_analysis: bool = True,
                 ml_enhanced: bool = True):
        """
        Initialize the Time Zone Analysis Indicator.

        Args:
            lookback_period: Number of periods to analyze for pattern detection
            fibonacci_sequence: Custom Fibonacci sequence for time calculations
            fibonacci_ratios: Custom Fibonacci ratios for time analysis
            min_cycle_length: Minimum cycle length to detect
            max_cycle_length: Maximum cycle length to detect
            confluence_threshold: Minimum zones for confluence detection
            intensity_threshold: Minimum intensity for significant zones
            seasonal_analysis: Whether to include seasonal pattern analysis
            lunar_analysis: Whether to include lunar cycle analysis
            harmonic_analysis: Whether to include harmonic frequency analysis
            ml_enhanced: Whether to use machine learning enhancements
        """
        super().__init__()

        self.lookback_period = max(50, lookback_period)
        self.min_cycle_length = max(3, min_cycle_length)
        self.max_cycle_length = min(200, max_cycle_length)
        self.confluence_threshold = max(2, confluence_threshold)
        self.intensity_threshold = max(0.1, intensity_threshold)
        self.seasonal_analysis = seasonal_analysis
        self.lunar_analysis = lunar_analysis
        self.harmonic_analysis = harmonic_analysis
        self.ml_enhanced = ml_enhanced

        # Fibonacci sequence for time calculations
        if fibonacci_sequence is None:
            self.fibonacci_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
        else:
            self.fibonacci_sequence = fibonacci_sequence

        # Fibonacci ratios for time analysis
        if fibonacci_ratios is None:
            self.fibonacci_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618, 4.236]
        else:
            self.fibonacci_ratios = fibonacci_ratios

        # Initialize ML components if enabled
        if self.ml_enhanced:
            self.scaler = StandardScaler()
            self.pattern_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            self.cluster_model = KMeans(n_clusters=5, random_state=42, n_init=10)

        # Storage for analysis
        self.time_zones_cache = []
        self.cycle_patterns_cache = {}
        self.historical_accuracy = {}

        logger.info(f"TimeZoneAnalysisIndicator initialized with {len(self.fibonacci_sequence)} Fibonacci numbers")

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive time zone analysis.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary containing time zone analysis results
        """
        try:
            # Validate input data
            self._validate_data(data)

            if len(data) < self.lookback_period:
                raise ValidationError(f"Insufficient data: {len(data)} < {self.lookback_period}")

            # Extract recent data for analysis
            recent_data = data.tail(self.lookback_period).copy()

            # Calculate base timing metrics
            timing_metrics = self._calculate_timing_metrics(recent_data)

            # Detect significant price swing points for time analysis
            swing_points = self._detect_time_swing_points(recent_data)

            # Perform different types of time analysis
            all_time_zones = []

            # Fibonacci time zone analysis
            fib_zones = self._analyze_fibonacci_time_zones(recent_data, swing_points, timing_metrics)
            all_time_zones.extend(fib_zones)

            # Fibonacci time extension analysis
            ext_zones = self._analyze_fibonacci_time_extensions(recent_data, swing_points, timing_metrics)
            all_time_zones.extend(ext_zones)

            # Cycle analysis
            cycle_zones = self._analyze_fibonacci_cycles(recent_data, timing_metrics)
            all_time_zones.extend(cycle_zones)

            # Market rhythm analysis
            rhythm_zones = self._analyze_market_rhythms(recent_data, timing_metrics)
            all_time_zones.extend(rhythm_zones)

            # Harmonic time cycle analysis
            if self.harmonic_analysis:
                harmonic_zones = self._analyze_harmonic_time_cycles(recent_data, timing_metrics)
                all_time_zones.extend(harmonic_zones)

            # Seasonal pattern analysis
            if self.seasonal_analysis:
                seasonal_zones = self._analyze_seasonal_patterns(recent_data, timing_metrics)
                all_time_zones.extend(seasonal_zones)

            # Lunar cycle analysis
            if self.lunar_analysis:
                lunar_zones = self._analyze_lunar_cycles(recent_data, timing_metrics)
                all_time_zones.extend(lunar_zones)

            # Apply machine learning enhancements
            if self.ml_enhanced and len(all_time_zones) > 10:
                all_time_zones = self._enhance_with_ml(recent_data, all_time_zones, timing_metrics)

            # Detect confluence zones
            confluence_zones = self._detect_time_confluence(all_time_zones)

            # Generate timing signals
            timing_signals = self._generate_timing_signals(
                all_time_zones, confluence_zones, timing_metrics, len(recent_data)
            )

            # Calculate future time projections
            future_projections = self._calculate_future_time_projections(
                recent_data, all_time_zones, timing_metrics
            )

            # Create comprehensive result
            result = {
                'time_zones': {
                    'all_zones': self._format_time_zones(all_time_zones),
                    'fibonacci_time_zones': [z for z in all_time_zones
                                           if z.zone_type == TimeAnalysisType.FIBONACCI_TIME_ZONES],
                    'fibonacci_time_extensions': [z for z in all_time_zones
                                                if z.zone_type == TimeAnalysisType.FIBONACCI_TIME_EXTENSIONS],
                    'fibonacci_cycles': [z for z in all_time_zones
                                       if z.zone_type == TimeAnalysisType.FIBONACCI_CYCLES],
                    'market_rhythms': [z for z in all_time_zones
                                     if z.zone_type == TimeAnalysisType.MARKET_RHYTHMS]
                },
                'confluence_analysis': {
                    'confluence_zones': self._format_confluence_zones(confluence_zones),
                    'high_confluence_count': len([c for c in confluence_zones if c.zone_count >= 4]),
                    'next_confluence_time': self._find_next_confluence_time(confluence_zones, len(recent_data))
                },
                'timing_signals': timing_signals,
                'future_projections': future_projections,
                'cycle_analysis': {
                    'dominant_cycles': self._identify_dominant_cycles(all_time_zones),
                    'cycle_synchronization': self._analyze_cycle_synchronization(all_time_zones),
                    'next_cycle_events': self._predict_next_cycle_events(all_time_zones, len(recent_data))
                },
                'metrics': {
                    'total_zones': len(all_time_zones),
                    'high_intensity_zones': len([z for z in all_time_zones if z.intensity > 0.8]),
                    'confluence_zones_count': len(confluence_zones),
                    'average_confidence': np.mean([z.confidence for z in all_time_zones]) if all_time_zones else 0,
                    'timing_accuracy': timing_metrics.get('timing_accuracy', 0.5)
                }
            }

            logger.info(f"Analyzed {len(all_time_zones)} time zones with "
                       f"{len(confluence_zones)} confluence areas")

            return result

        except Exception as e:
            logger.error(f"Error in TimeZoneAnalysisIndicator calculation: {str(e)}")
            raise CalculationError(f"Time zone analysis failed: {str(e)}")

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data structure and content."""
        required_columns = ['high', 'low', 'close', 'volume']

        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            raise ValidationError(f"Missing required columns: {missing}")

        if data[required_columns].isnull().any().any():
            raise ValidationError("Data contains null values")

        if (data[required_columns] <= 0).any().any():
            raise ValidationError("Data contains non-positive values")

    def _calculate_timing_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate base timing metrics for analysis."""
        try:
            # Price volatility metrics
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std()

            # Volume timing metrics
            volume_sma = data['volume'].rolling(20).mean()
            volume_ratio = data['volume'] / volume_sma

            # Trend persistence metrics
            sma_short = data['close'].rolling(10).mean()
            sma_long = data['close'].rolling(30).mean()
            trend_persistence = (sma_short > sma_long).rolling(10).sum() / 10

            # Market rhythm metrics
            high_low_range = (data['high'] - data['low']) / data['close']
            rhythm_consistency = 1.0 - high_low_range.rolling(20).std()

            # Timing accuracy from historical patterns
            timing_accuracy = self._calculate_historical_timing_accuracy(data)

            return {
                'volatility': volatility.iloc[-1] if not np.isnan(volatility.iloc[-1]) else 0.01,
                'volume_rhythm': volume_ratio.iloc[-10:].mean(),
                'trend_persistence': trend_persistence.iloc[-1] if not np.isnan(trend_persistence.iloc[-1]) else 0.5,
                'rhythm_consistency': rhythm_consistency.iloc[-1] if not np.isnan(rhythm_consistency.iloc[-1]) else 0.5,
                'market_tempo': returns.rolling(5).std().iloc[-1] if not np.isnan(returns.rolling(5).std().iloc[-1]) else 0.01,
                'timing_accuracy': timing_accuracy,
                'current_time': len(data) - 1
            }

        except Exception as e:
            logger.warning(f"Error calculating timing metrics: {str(e)}")
            return {
                'volatility': 0.01, 'volume_rhythm': 1.0, 'trend_persistence': 0.5,
                'rhythm_consistency': 0.5, 'market_tempo': 0.01, 'timing_accuracy': 0.5,
                'current_time': len(data) - 1
            }

    def _calculate_historical_timing_accuracy(self, data: pd.DataFrame) -> float:
        """Calculate historical timing accuracy for patterns."""
        try:
            # Simple historical validation based on volatility patterns
            recent_vol = data['close'].pct_change().tail(20).std()
            historical_vol = data['close'].pct_change().head(-20).std()

            # Higher accuracy for consistent volatility patterns
            vol_consistency = 1.0 - abs(recent_vol - historical_vol) / max(historical_vol, 0.001)

            return max(0.3, min(0.9, vol_consistency))

        except Exception as e:
            logger.warning(f"Error calculating historical timing accuracy: {str(e)}")
            return 0.6

    def _detect_time_swing_points(self, data: pd.DataFrame) -> List[Tuple[int, float, str]]:
        """Detect significant time-based swing points."""
        swing_points = []

        try:
            highs = data['high'].values
            lows = data['low'].values
            volumes = data['volume'].values

            # Use adaptive window based on market rhythm
            avg_volume = np.mean(volumes)
            window = max(3, min(21, int(10 + 5 * (np.std(volumes) / avg_volume))))

            # Detect time-significant swing highs
            for i in range(window, len(highs) - window):
                if highs[i] == max(highs[i-window:i+window+1]):
                    # Verify time significance with volume
                    volume_support = volumes[i] / avg_volume
                    if volume_support > 0.8:  # Volume confirmation
                        swing_points.append((i, highs[i], 'high'))

            # Detect time-significant swing lows
            for i in range(window, len(lows) - window):
                if lows[i] == min(lows[i-window:i+window+1]):
                    # Verify time significance with volume
                    volume_support = volumes[i] / avg_volume
                    if volume_support > 0.8:  # Volume confirmation
                        swing_points.append((i, lows[i], 'low'))

            # Sort by time and keep most significant
            swing_points.sort(key=lambda x: x[0])

            # Keep only most recent and significant swings
            if len(swing_points) > 30:
                swing_points = swing_points[-30:]

            logger.debug(f"Detected {len(swing_points)} time-significant swing points")
            return swing_points

        except Exception as e:
            logger.warning(f"Error detecting time swing points: {str(e)}")
            return []

    def _analyze_fibonacci_time_zones(self, data: pd.DataFrame, swing_points: List[Tuple],
                                    metrics: Dict) -> List[TimeZone]:
        """Analyze Fibonacci time zones from swing points."""
        time_zones = []

        try:
            current_time = len(data) - 1

            # Analyze time zones from each significant swing
            for swing_time, swing_price, swing_type in swing_points[-10:]:
                try:
                    time_distance = current_time - swing_time

                    # Calculate Fibonacci time projections
                    for fib_num in self.fibonacci_sequence:
                        if fib_num > time_distance:  # Future projections only
                            projected_time = swing_time + fib_num

                            if projected_time <= current_time + 50:  # Reasonable future range
                                # Calculate zone intensity based on multiple factors
                                intensity = self._calculate_time_zone_intensity(
                                    swing_time, projected_time, fib_num, data, metrics
                                )

                                # Calculate confidence based on historical patterns
                                confidence = self._calculate_time_zone_confidence(
                                    fib_num, swing_type, data, metrics
                                )

                                if intensity > self.intensity_threshold:
                                    # Determine signal type
                                    signal_type = self._determine_time_signal_type(
                                        swing_time, projected_time, swing_type, data
                                    )

                                    zone = TimeZone(
                                        zone_type=TimeAnalysisType.FIBONACCI_TIME_ZONES,
                                        signal_type=signal_type,
                                        start_time=projected_time - 2,
                                        end_time=projected_time + 2,
                                        center_time=projected_time,
                                        intensity=intensity,
                                        confidence=confidence,
                                        fibonacci_ratio=None,
                                        cycle_length=fib_num,
                                        harmonic_frequency=None,
                                        supporting_evidence=[f"Fibonacci sequence: {fib_num}",
                                                           f"Swing from: {swing_time}"]
                                    )

                                    time_zones.append(zone)

                except Exception as e:
                    logger.warning(f"Error analyzing Fibonacci time zone: {str(e)}")
                    continue

            return time_zones

        except Exception as e:
            logger.warning(f"Error in Fibonacci time zone analysis: {str(e)}")
            return []

    def _analyze_fibonacci_time_extensions(self, data: pd.DataFrame, swing_points: List[Tuple],
                                         metrics: Dict) -> List[TimeZone]:
        """Analyze Fibonacci time extensions between swing points."""
        time_zones = []

        try:
            # Analyze extensions between consecutive swings
            for i in range(len(swing_points) - 1):
                swing1_time, swing1_price, swing1_type = swing_points[i]
                swing2_time, swing2_price, swing2_type = swing_points[i + 1]

                time_distance = swing2_time - swing1_time

                # Calculate Fibonacci extensions
                for ratio in self.fibonacci_ratios:
                    if ratio >= 1.0:  # Extensions only
                        extension_distance = int(time_distance * ratio)
                        projected_time = swing2_time + extension_distance

                        if projected_time <= len(data) + 50:  # Reasonable range
                            # Calculate extension intensity
                            intensity = self._calculate_extension_intensity(
                                swing1_time, swing2_time, projected_time, ratio, data, metrics
                            )

                            # Calculate confidence
                            confidence = self._calculate_extension_confidence(
                                ratio, swing1_type, swing2_type, data, metrics
                            )

                            if intensity > self.intensity_threshold:
                                # Determine signal type based on swing pattern
                                signal_type = self._determine_extension_signal_type(
                                    swing1_type, swing2_type, ratio
                                )

                                zone = TimeZone(
                                    zone_type=TimeAnalysisType.FIBONACCI_TIME_EXTENSIONS,
                                    signal_type=signal_type,
                                    start_time=projected_time - 3,
                                    end_time=projected_time + 3,
                                    center_time=projected_time,
                                    intensity=intensity,
                                    confidence=confidence,
                                    fibonacci_ratio=ratio,
                                    cycle_length=extension_distance,
                                    harmonic_frequency=None,
                                    supporting_evidence=[f"Extension ratio: {ratio}",
                                                       f"Base cycle: {time_distance} periods"]
                                )

                                time_zones.append(zone)

            return time_zones

        except Exception as e:
            logger.warning(f"Error in Fibonacci time extension analysis: {str(e)}")
            return []

    def _analyze_fibonacci_cycles(self, data: pd.DataFrame, metrics: Dict) -> List[TimeZone]:
        """Analyze Fibonacci-based market cycles."""
        time_zones = []

        try:
            current_time = len(data) - 1

            # Detect cycles using Fibonacci numbers
            for fib_cycle in self.fibonacci_sequence:
                if fib_cycle < self.min_cycle_length or fib_cycle > self.max_cycle_length:
                    continue

                # Look for cycle patterns in the data
                cycle_strength = self._calculate_cycle_strength(data, fib_cycle)

                if cycle_strength > 0.6:
                    # Project next cycle occurrences
                    last_cycle_time = self._find_last_cycle_occurrence(data, fib_cycle)

                    if last_cycle_time is not None:
                        next_cycle_times = []
                        for multiplier in [1, 2, 3]:
                            next_time = last_cycle_time + (fib_cycle * multiplier)
                            if next_time <= current_time + 50:
                                next_cycle_times.append(next_time)

                        for next_time in next_cycle_times:
                            intensity = cycle_strength * (1.0 - (next_time - current_time) / 100.0)
                            intensity = max(0.1, intensity)

                            confidence = self._calculate_cycle_confidence(
                                fib_cycle, cycle_strength, metrics
                            )

                            signal_type = self._determine_cycle_signal_type(
                                fib_cycle, cycle_strength, metrics
                            )

                            zone = TimeZone(
                                zone_type=TimeAnalysisType.FIBONACCI_CYCLES,
                                signal_type=signal_type,
                                start_time=next_time - (fib_cycle // 4),
                                end_time=next_time + (fib_cycle // 4),
                                center_time=next_time,
                                intensity=intensity,
                                confidence=confidence,
                                fibonacci_ratio=None,
                                cycle_length=fib_cycle,
                                harmonic_frequency=1.0 / fib_cycle,
                                supporting_evidence=[f"Fibonacci cycle: {fib_cycle}",
                                                   f"Cycle strength: {cycle_strength:.3f}"]
                            )

                            time_zones.append(zone)

            return time_zones

        except Exception as e:
            logger.warning(f"Error in Fibonacci cycle analysis: {str(e)}")
            return []

    def _analyze_market_rhythms(self, data: pd.DataFrame, metrics: Dict) -> List[TimeZone]:
        """Analyze market rhythms and tempo patterns."""
        time_zones = []

        try:
            # Calculate market rhythm indicators
            price_ranges = (data['high'] - data['low']) / data['close']
            volume_rhythm = data['volume'] / data['volume'].rolling(20).mean()

            # Detect rhythm patterns
            rhythm_cycles = self._detect_rhythm_cycles(price_ranges, volume_rhythm)

            current_time = len(data) - 1

            for cycle_length, rhythm_strength in rhythm_cycles.items():
                if rhythm_strength > 0.7:
                    # Project rhythm-based time zones
                    last_rhythm_peak = self._find_last_rhythm_peak(
                        price_ranges, volume_rhythm, cycle_length
                    )

                    if last_rhythm_peak is not None:
                        next_rhythm_times = []
                        for i in range(1, 4):
                            next_time = last_rhythm_peak + (cycle_length * i)
                            if next_time <= current_time + 30:
                                next_rhythm_times.append(next_time)

                        for next_time in next_rhythm_times:
                            intensity = rhythm_strength * metrics['rhythm_consistency']
                            confidence = self._calculate_rhythm_confidence(
                                cycle_length, rhythm_strength, metrics
                            )

                            signal_type = TimeSignalType.ACCELERATION_ZONE  # Rhythm often indicates acceleration

                            zone = TimeZone(
                                zone_type=TimeAnalysisType.MARKET_RHYTHMS,
                                signal_type=signal_type,
                                start_time=next_time - 2,
                                end_time=next_time + 2,
                                center_time=next_time,
                                intensity=intensity,
                                confidence=confidence,
                                fibonacci_ratio=None,
                                cycle_length=cycle_length,
                                harmonic_frequency=1.0 / cycle_length,
                                supporting_evidence=[f"Market rhythm cycle: {cycle_length}",
                                                   f"Rhythm strength: {rhythm_strength:.3f}"]
                            )

                            time_zones.append(zone)

            return time_zones

        except Exception as e:
            logger.warning(f"Error in market rhythm analysis: {str(e)}")
            return []

    def _analyze_harmonic_time_cycles(self, data: pd.DataFrame, metrics: Dict) -> List[TimeZone]:
        """Analyze harmonic time cycles and frequency patterns."""
        time_zones = []

        try:
            if not self.harmonic_analysis:
                return time_zones

            # Calculate harmonic frequencies
            price_data = data['close'].values
            harmonic_frequencies = self._calculate_harmonic_frequencies(price_data)

            current_time = len(data) - 1

            for frequency, amplitude in harmonic_frequencies.items():
                if amplitude > 0.6:
                    cycle_length = int(1.0 / frequency) if frequency > 0 else None

                    if cycle_length and self.min_cycle_length <= cycle_length <= self.max_cycle_length:
                        # Project harmonic time zones
                        phase = self._calculate_harmonic_phase(price_data, frequency)

                        # Calculate next harmonic peaks
                        next_peaks = self._project_harmonic_peaks(
                            frequency, phase, current_time, amplitude
                        )

                        for peak_time in next_peaks:
                            if peak_time <= current_time + 40:
                                intensity = amplitude * metrics['rhythm_consistency']
                                confidence = self._calculate_harmonic_confidence(
                                    frequency, amplitude, metrics
                                )

                                signal_type = TimeSignalType.REVERSAL_ZONE  # Harmonic peaks often indicate reversals

                                zone = TimeZone(
                                    zone_type=TimeAnalysisType.HARMONIC_TIME_CYCLES,
                                    signal_type=signal_type,
                                    start_time=peak_time - 1,
                                    end_time=peak_time + 1,
                                    center_time=peak_time,
                                    intensity=intensity,
                                    confidence=confidence,
                                    fibonacci_ratio=None,
                                    cycle_length=cycle_length,
                                    harmonic_frequency=frequency,
                                    supporting_evidence=[f"Harmonic frequency: {frequency:.4f}",
                                                       f"Amplitude: {amplitude:.3f}"]
                                )

                                time_zones.append(zone)

            return time_zones

        except Exception as e:
            logger.warning(f"Error in harmonic time cycle analysis: {str(e)}")
            return []

    def _analyze_seasonal_patterns(self, data: pd.DataFrame, metrics: Dict) -> List[TimeZone]:
        """Analyze seasonal timing patterns."""
        time_zones = []

        try:
            if not self.seasonal_analysis:
                return time_zones

            # Simplified seasonal analysis (would need actual date information for full implementation)
            # This is a basic pattern recognition approach

            # Look for weekly patterns (assuming business days)
            weekly_pattern = self._detect_weekly_patterns(data)
            monthly_pattern = self._detect_monthly_patterns(data)

            current_time = len(data) - 1

            # Weekly seasonal zones
            for day_offset, pattern_strength in weekly_pattern.items():
                if pattern_strength > 0.7:
                    next_occurrences = []
                    for week in range(1, 5):  # Next 4 weeks
                        next_time = current_time + day_offset + (week * 5)  # Assuming 5-day weeks
                        if next_time <= current_time + 30:
                            next_occurrences.append(next_time)

                    for next_time in next_occurrences:
                        intensity = pattern_strength * 0.8  # Moderate intensity for seasonal
                        confidence = min(0.8, pattern_strength + 0.1)

                        zone = TimeZone(
                            zone_type=TimeAnalysisType.SEASONAL_PATTERNS,
                            signal_type=TimeSignalType.CONTINUATION_ZONE,
                            start_time=next_time,
                            end_time=next_time,
                            center_time=next_time,
                            intensity=intensity,
                            confidence=confidence,
                            fibonacci_ratio=None,
                            cycle_length=7,  # Weekly cycle
                            harmonic_frequency=None,
                            supporting_evidence=[f"Weekly pattern strength: {pattern_strength:.3f}"]
                        )

                        time_zones.append(zone)

            return time_zones

        except Exception as e:
            logger.warning(f"Error in seasonal pattern analysis: {str(e)}")
            return []

    def _analyze_lunar_cycles(self, data: pd.DataFrame, metrics: Dict) -> List[TimeZone]:
        """Analyze lunar cycle timing patterns."""
        time_zones = []

        try:
            if not self.lunar_analysis:
                return time_zones

            # Lunar cycle is approximately 29.5 days
            lunar_cycle_length = 30  # Approximation for trading days

            # Detect lunar influence patterns
            lunar_correlation = self._calculate_lunar_correlation(data, lunar_cycle_length)

            if lunar_correlation > 0.6:
                current_time = len(data) - 1

                # Find last lunar cycle occurrence
                last_lunar_peak = self._find_last_lunar_occurrence(data, lunar_cycle_length)

                if last_lunar_peak is not None:
                    # Project next lunar zones
                    next_lunar_times = []
                    for cycle in range(1, 3):  # Next 2 lunar cycles
                        next_time = last_lunar_peak + (lunar_cycle_length * cycle)
                        if next_time <= current_time + 60:
                            next_lunar_times.append(next_time)

                    for next_time in next_lunar_times:
                        intensity = lunar_correlation * 0.7  # Moderate intensity
                        confidence = min(0.7, lunar_correlation)

                        zone = TimeZone(
                            zone_type=TimeAnalysisType.LUNAR_CYCLES,
                            signal_type=TimeSignalType.REVERSAL_ZONE,  # Lunar often correlates with reversals
                            start_time=next_time - 2,
                            end_time=next_time + 2,
                            center_time=next_time,
                            intensity=intensity,
                            confidence=confidence,
                            fibonacci_ratio=None,
                            cycle_length=lunar_cycle_length,
                            harmonic_frequency=1.0 / lunar_cycle_length,
                            supporting_evidence=[f"Lunar correlation: {lunar_correlation:.3f}"]
                        )

                        time_zones.append(zone)

            return time_zones

        except Exception as e:
            logger.warning(f"Error in lunar cycle analysis: {str(e)}")
            return []    # Helper methods for calculations and analysis

    def _calculate_time_zone_intensity(self, swing_time: int, projected_time: int,
                                     fib_num: int, data: pd.DataFrame, metrics: Dict) -> float:
        """Calculate intensity score for a time zone."""
        try:
            base_intensity = 0.5

            # Adjust based on Fibonacci number significance
            if fib_num in [8, 13, 21, 34, 55, 89]:  # Key Fibonacci numbers
                base_intensity += 0.2

            # Adjust based on market conditions
            volatility_factor = min(0.2, metrics['volatility'] * 2)
            base_intensity += volatility_factor

            # Adjust based on volume rhythm
            volume_factor = (metrics['volume_rhythm'] - 1.0) * 0.1
            base_intensity += volume_factor

            # Distance decay factor
            distance = abs(projected_time - len(data))
            decay_factor = max(0.1, 1.0 - (distance / 50.0))
            base_intensity *= decay_factor

            return max(0.1, min(1.0, base_intensity))

        except Exception as e:
            logger.warning(f"Error calculating time zone intensity: {str(e)}")
            return 0.5

    def _calculate_time_zone_confidence(self, fib_num: int, swing_type: str,
                                      data: pd.DataFrame, metrics: Dict) -> float:
        """Calculate confidence score for a time zone."""
        try:
            base_confidence = 0.6

            # Adjust based on historical timing accuracy
            base_confidence += (metrics['timing_accuracy'] - 0.5) * 0.4

            # Adjust based on trend persistence
            base_confidence += (metrics['trend_persistence'] - 0.5) * 0.2

            # Adjust based on rhythm consistency
            base_confidence += (metrics['rhythm_consistency'] - 0.5) * 0.2

            # Fibonacci number significance
            if fib_num in [13, 21, 34, 55]:  # Most reliable Fibonacci numbers
                base_confidence += 0.1

            return max(0.2, min(0.9, base_confidence))

        except Exception as e:
            logger.warning(f"Error calculating time zone confidence: {str(e)}")
            return 0.6

    def _determine_time_signal_type(self, swing_time: int, projected_time: int,
                                  swing_type: str, data: pd.DataFrame) -> TimeSignalType:
        """Determine the signal type for a time zone."""
        try:
            # Basic signal type determination
            if swing_type == 'high':
                return TimeSignalType.REVERSAL_ZONE
            elif swing_type == 'low':
                return TimeSignalType.REVERSAL_ZONE
            else:
                return TimeSignalType.CONTINUATION_ZONE

        except Exception as e:
            logger.warning(f"Error determining time signal type: {str(e)}")
            return TimeSignalType.CONTINUATION_ZONE

    def _calculate_extension_intensity(self, swing1_time: int, swing2_time: int,
                                     projected_time: int, ratio: float,
                                     data: pd.DataFrame, metrics: Dict) -> float:
        """Calculate intensity for time extensions."""
        try:
            base_intensity = 0.6

            # Adjust based on Fibonacci ratio significance
            if ratio in [1.272, 1.618, 2.618]:  # Key extension ratios
                base_intensity += 0.2

            # Adjust based on base cycle strength
            base_cycle = swing2_time - swing1_time
            if 8 <= base_cycle <= 34:  # Optimal cycle range
                base_intensity += 0.1

            # Volume rhythm factor
            volume_factor = (metrics['volume_rhythm'] - 1.0) * 0.1
            base_intensity += volume_factor

            return max(0.2, min(1.0, base_intensity))

        except Exception as e:
            logger.warning(f"Error calculating extension intensity: {str(e)}")
            return 0.5

    def _calculate_extension_confidence(self, ratio: float, swing1_type: str,
                                      swing2_type: str, data: pd.DataFrame, metrics: Dict) -> float:
        """Calculate confidence for time extensions."""
        try:
            base_confidence = 0.7

            # Pattern significance
            if swing1_type != swing2_type:  # Alternating swings
                base_confidence += 0.1

            # Ratio significance
            if ratio == 1.618:  # Golden ratio
                base_confidence += 0.1

            # Market condition factor
            base_confidence += (metrics['timing_accuracy'] - 0.5) * 0.2

            return max(0.3, min(0.9, base_confidence))

        except Exception as e:
            logger.warning(f"Error calculating extension confidence: {str(e)}")
            return 0.7

    def _determine_extension_signal_type(self, swing1_type: str, swing2_type: str, ratio: float) -> TimeSignalType:
        """Determine signal type for extensions."""
        try:
            if ratio >= 2.0:
                return TimeSignalType.BREAKOUT_ZONE
            elif ratio >= 1.5:
                return TimeSignalType.ACCELERATION_ZONE
            else:
                return TimeSignalType.CONTINUATION_ZONE

        except Exception as e:
            logger.warning(f"Error determining extension signal type: {str(e)}")
            return TimeSignalType.CONTINUATION_ZONE

    def _calculate_cycle_strength(self, data: pd.DataFrame, cycle_length: int) -> float:
        """Calculate strength of a specific cycle length."""
        try:
            if len(data) < cycle_length * 3:
                return 0.0

            prices = data['close'].values

            # Calculate correlation with shifted data
            correlations = []
            for shift in [cycle_length, cycle_length * 2]:
                if len(prices) > shift:
                    shifted_prices = prices[:-shift]
                    current_prices = prices[shift:]
                    min_len = min(len(shifted_prices), len(current_prices))

                    if min_len > 10:
                        correlation = np.corrcoef(
                            shifted_prices[:min_len], current_prices[:min_len]
                        )[0, 1]
                        if not np.isnan(correlation):
                            correlations.append(abs(correlation))

            return np.mean(correlations) if correlations else 0.0

        except Exception as e:
            logger.warning(f"Error calculating cycle strength: {str(e)}")
            return 0.0

    def _find_last_cycle_occurrence(self, data: pd.DataFrame, cycle_length: int) -> Optional[int]:
        """Find the last occurrence of a specific cycle."""
        try:
            # Simplified cycle detection
            highs = data['high'].values
            lows = data['low'].values

            # Look for peaks that match the cycle
            for i in range(len(highs) - 1, cycle_length, -1):
                if (highs[i] > highs[i-1] and highs[i] > highs[i+1] if i < len(highs)-1 else True):
                    # Check if this could be part of the cycle
                    if i >= cycle_length:
                        prev_peak_area = i - cycle_length
                        if any(highs[prev_peak_area:prev_peak_area+5] > highs[i] * 0.95):
                            return i

            return None

        except Exception as e:
            logger.warning(f"Error finding last cycle occurrence: {str(e)}")
            return None

    def _calculate_cycle_confidence(self, cycle_length: int, cycle_strength: float, metrics: Dict) -> float:
        """Calculate confidence for cycle projections."""
        try:
            base_confidence = cycle_strength * 0.8

            # Adjust based on timing accuracy
            base_confidence += (metrics['timing_accuracy'] - 0.5) * 0.2

            # Fibonacci cycle bonus
            if cycle_length in self.fibonacci_sequence:
                base_confidence += 0.1

            return max(0.2, min(0.9, base_confidence))

        except Exception as e:
            logger.warning(f"Error calculating cycle confidence: {str(e)}")
            return 0.6

    def _determine_cycle_signal_type(self, cycle_length: int, cycle_strength: float, metrics: Dict) -> TimeSignalType:
        """Determine signal type for cycle projections."""
        try:
            if cycle_strength > 0.8:
                return TimeSignalType.REVERSAL_ZONE
            elif cycle_strength > 0.7:
                return TimeSignalType.ACCELERATION_ZONE
            else:
                return TimeSignalType.CONTINUATION_ZONE

        except Exception as e:
            logger.warning(f"Error determining cycle signal type: {str(e)}")
            return TimeSignalType.CONTINUATION_ZONE

    def _detect_rhythm_cycles(self, price_ranges: pd.Series, volume_rhythm: pd.Series) -> Dict[int, float]:
        """Detect market rhythm cycles."""
        try:
            rhythm_cycles = {}

            # Test different cycle lengths
            for cycle_len in range(self.min_cycle_length, min(30, self.max_cycle_length)):
                try:
                    # Calculate rhythm strength for this cycle
                    range_correlation = self._calculate_series_cycle_correlation(price_ranges, cycle_len)
                    volume_correlation = self._calculate_series_cycle_correlation(volume_rhythm, cycle_len)

                    combined_strength = (range_correlation + volume_correlation) / 2

                    if combined_strength > 0.5:
                        rhythm_cycles[cycle_len] = combined_strength

                except Exception as e:
                    logger.warning(f"Error detecting rhythm cycle {cycle_len}: {str(e)}")
                    continue

            return rhythm_cycles

        except Exception as e:
            logger.warning(f"Error detecting rhythm cycles: {str(e)}")
            return {}

    def _calculate_series_cycle_correlation(self, series: pd.Series, cycle_length: int) -> float:
        """Calculate correlation of a series with its shifted version."""
        try:
            if len(series) <= cycle_length:
                return 0.0

            shifted = series.shift(cycle_length)
            correlation = series.corr(shifted)

            return abs(correlation) if not np.isnan(correlation) else 0.0

        except Exception as e:
            logger.warning(f"Error calculating series cycle correlation: {str(e)}")
            return 0.0

    def _find_last_rhythm_peak(self, price_ranges: pd.Series, volume_rhythm: pd.Series, cycle_length: int) -> Optional[int]:
        """Find the last rhythm peak for a given cycle."""
        try:
            # Combine price and volume rhythm
            combined_rhythm = (price_ranges + volume_rhythm) / 2

            # Find peaks
            for i in range(len(combined_rhythm) - 1, cycle_length, -1):
                if (combined_rhythm.iloc[i] > combined_rhythm.iloc[i-1] and
                    (i == len(combined_rhythm) - 1 or combined_rhythm.iloc[i] > combined_rhythm.iloc[i+1])):
                    return i

            return None

        except Exception as e:
            logger.warning(f"Error finding last rhythm peak: {str(e)}")
            return None

    def _calculate_rhythm_confidence(self, cycle_length: int, rhythm_strength: float, metrics: Dict) -> float:
        """Calculate confidence for rhythm projections."""
        try:
            base_confidence = rhythm_strength * 0.9

            # Adjust based on rhythm consistency
            base_confidence += (metrics['rhythm_consistency'] - 0.5) * 0.2

            return max(0.3, min(0.9, base_confidence))

        except Exception as e:
            logger.warning(f"Error calculating rhythm confidence: {str(e)}")
            return 0.6

    def _calculate_harmonic_frequencies(self, price_data: np.ndarray) -> Dict[float, float]:
        """Calculate harmonic frequencies using FFT."""
        try:
            # Simple FFT-based frequency analysis
            fft_result = np.fft.fft(price_data)
            frequencies = np.fft.fftfreq(len(price_data))
            amplitudes = np.abs(fft_result)

            # Find dominant frequencies
            harmonic_frequencies = {}

            # Get top frequencies
            top_indices = np.argsort(amplitudes)[-10:]  # Top 10 frequencies

            for idx in top_indices:
                freq = abs(frequencies[idx])
                amplitude = amplitudes[idx] / np.max(amplitudes)  # Normalize

                if 0.01 <= freq <= 0.5 and amplitude > 0.1:  # Valid range
                    harmonic_frequencies[freq] = amplitude

            return harmonic_frequencies

        except Exception as e:
            logger.warning(f"Error calculating harmonic frequencies: {str(e)}")
            return {}

    def _calculate_harmonic_phase(self, price_data: np.ndarray, frequency: float) -> float:
        """Calculate phase for a harmonic frequency."""
        try:
            # Simplified phase calculation
            t = np.arange(len(price_data))
            sine_wave = np.sin(2 * np.pi * frequency * t)

            # Calculate correlation to find phase
            correlation = np.corrcoef(price_data, sine_wave)[0, 1]

            # Convert correlation to phase (simplified)
            phase = np.arccos(abs(correlation)) if abs(correlation) <= 1 else 0

            return phase

        except Exception as e:
            logger.warning(f"Error calculating harmonic phase: {str(e)}")
            return 0.0

    def _project_harmonic_peaks(self, frequency: float, phase: float, current_time: int, amplitude: float) -> List[int]:
        """Project future harmonic peaks."""
        try:
            peaks = []
            period = 1.0 / frequency if frequency > 0 else 20

            # Project next few peaks
            for i in range(1, 4):
                next_peak_time = current_time + (period * i) - phase
                if next_peak_time > current_time and next_peak_time <= current_time + 50:
                    peaks.append(int(next_peak_time))

            return peaks

        except Exception as e:
            logger.warning(f"Error projecting harmonic peaks: {str(e)}")
            return []

    def _calculate_harmonic_confidence(self, frequency: float, amplitude: float, metrics: Dict) -> float:
        """Calculate confidence for harmonic projections."""
        try:
            base_confidence = amplitude * 0.8

            # Adjust based on rhythm consistency
            base_confidence += (metrics['rhythm_consistency'] - 0.5) * 0.2

            return max(0.2, min(0.8, base_confidence))

        except Exception as e:
            logger.warning(f"Error calculating harmonic confidence: {str(e)}")
            return 0.5

    def _detect_weekly_patterns(self, data: pd.DataFrame) -> Dict[int, float]:
        """Detect weekly seasonal patterns."""
        try:
            weekly_patterns = {}

            # Analyze patterns for each day of week (0-4 for business days)
            for day in range(5):
                day_returns = []

                # Sample every 5th day starting from day offset
                for i in range(day, len(data), 5):
                    if i > 0:
                        day_return = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
                        day_returns.append(day_return)

                if len(day_returns) > 10:
                    # Calculate pattern strength based on consistency
                    avg_return = np.mean(day_returns)
                    std_return = np.std(day_returns)

                    if std_return > 0:
                        pattern_strength = abs(avg_return) / std_return
                        pattern_strength = min(1.0, pattern_strength)

                        if pattern_strength > 0.3:
                            weekly_patterns[day] = pattern_strength

            return weekly_patterns

        except Exception as e:
            logger.warning(f"Error detecting weekly patterns: {str(e)}")
            return {}

    def _detect_monthly_patterns(self, data: pd.DataFrame) -> Dict[int, float]:
        """Detect monthly seasonal patterns."""
        try:
            monthly_patterns = {}

            # Simplified monthly pattern detection
            month_length = 22  # Approximate trading days in a month

            for offset in range(month_length):
                month_returns = []

                for i in range(offset, len(data), month_length):
                    if i > 0:
                        month_return = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
                        month_returns.append(month_return)

                if len(month_returns) > 5:
                    avg_return = np.mean(month_returns)
                    std_return = np.std(month_returns)

                    if std_return > 0:
                        pattern_strength = abs(avg_return) / std_return
                        pattern_strength = min(1.0, pattern_strength * 0.8)  # Lower weight for monthly

                        if pattern_strength > 0.4:
                            monthly_patterns[offset] = pattern_strength

            return monthly_patterns

        except Exception as e:
            logger.warning(f"Error detecting monthly patterns: {str(e)}")
            return {}

    def _calculate_lunar_correlation(self, data: pd.DataFrame, cycle_length: int) -> float:
        """Calculate correlation with lunar cycles."""
        try:
            # Simplified lunar correlation
            returns = data['close'].pct_change()

            # Create lunar cycle proxy
            lunar_cycle = []
            for i in range(len(returns)):
                lunar_phase = (i % cycle_length) / cycle_length * 2 * np.pi
                lunar_value = np.sin(lunar_phase)
                lunar_cycle.append(lunar_value)

            # Calculate correlation
            lunar_series = pd.Series(lunar_cycle)
            correlation = returns.corr(lunar_series)

            return abs(correlation) if not np.isnan(correlation) else 0.0

        except Exception as e:
            logger.warning(f"Error calculating lunar correlation: {str(e)}")
            return 0.0

    def _find_last_lunar_occurrence(self, data: pd.DataFrame, cycle_length: int) -> Optional[int]:
        """Find the last lunar cycle occurrence."""
        try:
            # Simplified lunar occurrence detection
            highs = data['high'].values

            # Look for peaks that align with lunar cycle
            for i in range(len(highs) - 1, cycle_length, -1):
                lunar_phase = (i % cycle_length) / cycle_length

                # Check if this is near a lunar peak (around 0.25 or 0.75 of cycle)
                if 0.2 <= lunar_phase <= 0.3 or 0.7 <= lunar_phase <= 0.8:
                    if i > 0 and i < len(highs) - 1:
                        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                            return i

            return None

        except Exception as e:
            logger.warning(f"Error finding last lunar occurrence: {str(e)}")
            return None

    def _enhance_with_ml(self, data: pd.DataFrame, time_zones: List[TimeZone], metrics: Dict) -> List[TimeZone]:
        """Enhance time zones using machine learning."""
        try:
            if not self.ml_enhanced or len(time_zones) < 10:
                return time_zones

            # Prepare features for ML
            features = []
            targets = []

            for zone in time_zones:
                try:
                    # Feature engineering
                    feature_vector = [
                        zone.intensity,
                        zone.confidence,
                        zone.cycle_length or 0,
                        zone.fibonacci_ratio or 0,
                        zone.harmonic_frequency or 0,
                        zone.end_time - zone.start_time,
                        metrics['volatility'],
                        metrics['volume_rhythm'],
                        metrics['trend_persistence'],
                        metrics['rhythm_consistency']
                    ]

                    # Target: composite score
                    target = (zone.intensity * zone.confidence +
                            len(zone.supporting_evidence) * 0.1)

                    features.append(feature_vector)
                    targets.append(target)

                except Exception as e:
                    logger.warning(f"Error preparing ML features: {str(e)}")
                    continue

            if len(features) < 5:
                return time_zones

            try:
                # Train and enhance
                features_array = np.array(features)
                targets_array = np.array(targets)

                # Scale features
                features_scaled = self.scaler.fit_transform(features_array)

                # Train model
                self.pattern_model.fit(features_scaled, targets_array)

                # Get predictions
                predictions = self.pattern_model.predict(features_scaled)

                # Update time zones with ML-enhanced scores
                for i, zone in enumerate(time_zones):
                    if i < len(predictions):
                        ml_enhancement = max(0.1, min(1.0, predictions[i]))
                        zone.confidence = (zone.confidence + ml_enhancement) / 2
                        zone.intensity = (zone.intensity + ml_enhancement) / 2

            except Exception as e:
                logger.warning(f"Error in ML enhancement: {str(e)}")

            return time_zones

        except Exception as e:
            logger.warning(f"Error enhancing with ML: {str(e)}")
            return time_zones

    def _detect_time_confluence(self, time_zones: List[TimeZone]) -> List[TimeConfluence]:
        """Detect confluence zones where multiple time zones overlap."""
        try:
            confluence_zones = []

            if len(time_zones) < self.confluence_threshold:
                return confluence_zones

            # Group zones by proximity
            time_groups = {}

            for zone in time_zones:
                center_time = zone.center_time

                # Find existing group or create new one
                assigned = False
                for group_center, group_zones in time_groups.items():
                    if abs(center_time - group_center) <= 5:  # Within 5 periods
                        group_zones.append(zone)
                        assigned = True
                        break

                if not assigned:
                    time_groups[center_time] = [zone]

            # Create confluence zones from groups
            for group_center, group_zones in time_groups.items():
                if len(group_zones) >= self.confluence_threshold:
                    total_intensity = sum(zone.intensity for zone in group_zones)
                    avg_confidence = np.mean([zone.confidence for zone in group_zones])

                    # Determine dominant signal type
                    signal_counts = {}
                    for zone in group_zones:
                        signal_type = zone.signal_type
                        signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1

                    dominant_signal = max(signal_counts.keys(), key=lambda k: signal_counts[k])

                    # Calculate strength score
                    strength_score = (total_intensity / len(group_zones)) * avg_confidence

                    confluence = TimeConfluence(
                        confluence_time=group_center,
                        zone_count=len(group_zones),
                        total_intensity=total_intensity,
                        average_confidence=avg_confidence,
                        dominant_signal=dominant_signal,
                        contributing_zones=group_zones,
                        strength_score=strength_score
                    )

                    confluence_zones.append(confluence)

            # Sort by strength
            confluence_zones.sort(key=lambda c: c.strength_score, reverse=True)

            return confluence_zones

        except Exception as e:
            logger.warning(f"Error detecting time confluence: {str(e)}")
            return []

    def _generate_timing_signals(self, time_zones: List[TimeZone], confluence_zones: List[TimeConfluence],
                               metrics: Dict, current_time: int) -> Dict[str, Any]:
        """Generate timing signals based on time zone analysis."""
        try:
            signals = {
                'primary_timing_signal': 'NEUTRAL',
                'signal_strength': 0.0,
                'confidence': 0.0,
                'next_significant_time': None,
                'time_targets': [],
                'reversal_zones': [],
                'acceleration_zones': [],
                'confluence_alerts': []
            }

            # Analyze upcoming time zones
            upcoming_zones = [zone for zone in time_zones
                            if zone.center_time > current_time and zone.center_time <= current_time + 20]

            if not upcoming_zones:
                return signals

            # Find most significant upcoming zone
            strongest_zone = max(upcoming_zones, key=lambda z: z.intensity * z.confidence)

            # Determine primary signal
            if strongest_zone.signal_type == TimeSignalType.REVERSAL_ZONE:
                signals['primary_timing_signal'] = 'REVERSAL_EXPECTED'
            elif strongest_zone.signal_type == TimeSignalType.ACCELERATION_ZONE:
                signals['primary_timing_signal'] = 'ACCELERATION_EXPECTED'
            elif strongest_zone.signal_type == TimeSignalType.BREAKOUT_ZONE:
                signals['primary_timing_signal'] = 'BREAKOUT_EXPECTED'
            else:
                signals['primary_timing_signal'] = 'CONTINUATION_EXPECTED'

            signals['signal_strength'] = strongest_zone.intensity * strongest_zone.confidence
            signals['confidence'] = strongest_zone.confidence
            signals['next_significant_time'] = strongest_zone.center_time

            # Add time targets
            for zone in upcoming_zones:
                if zone.confidence > 0.6:
                    signals['time_targets'].append({
                        'time': zone.center_time,
                        'signal_type': zone.signal_type.value,
                        'intensity': zone.intensity,
                        'confidence': zone.confidence,
                        'analysis_type': zone.zone_type.value
                    })

            # Add confluence alerts
            upcoming_confluence = [conf for conf in confluence_zones
                                 if conf.confluence_time > current_time and conf.confluence_time <= current_time + 30]

            for confluence in upcoming_confluence:
                if confluence.strength_score > 0.7:
                    signals['confluence_alerts'].append({
                        'time': confluence.confluence_time,
                        'zone_count': confluence.zone_count,
                        'strength': confluence.strength_score,
                        'signal_type': confluence.dominant_signal.value
                    })

            return signals

        except Exception as e:
            logger.warning(f"Error generating timing signals: {str(e)}")
            return {
                'primary_timing_signal': 'NEUTRAL', 'signal_strength': 0.0,
                'confidence': 0.0, 'next_significant_time': None,
                'time_targets': [], 'reversal_zones': [], 'acceleration_zones': [],
                'confluence_alerts': []
            }

    def _calculate_future_time_projections(self, data: pd.DataFrame, time_zones: List[TimeZone],
                                         metrics: Dict) -> Dict[str, Any]:
        """Calculate future time projections."""
        try:
            current_time = len(data) - 1

            projections = {
                'next_fibonacci_times': [],
                'next_cycle_completions': [],
                'projected_reversals': [],
                'confluence_windows': []
            }

            # Project Fibonacci times
            fibonacci_zones = [z for z in time_zones if z.zone_type == TimeAnalysisType.FIBONACCI_TIME_ZONES]
            for zone in fibonacci_zones:
                if zone.center_time > current_time and zone.confidence > 0.6:
                    projections['next_fibonacci_times'].append({
                        'time': zone.center_time,
                        'fibonacci_number': zone.cycle_length,
                        'confidence': zone.confidence
                    })

            # Project cycle completions
            cycle_zones = [z for z in time_zones if z.zone_type == TimeAnalysisType.FIBONACCI_CYCLES]
            for zone in cycle_zones:
                if zone.center_time > current_time:
                    projections['next_cycle_completions'].append({
                        'time': zone.center_time,
                        'cycle_length': zone.cycle_length,
                        'strength': zone.intensity
                    })

            # Sort projections by time
            for key in projections:
                if projections[key]:
                    projections[key] = sorted(projections[key], key=lambda x: x['time'])

            return projections

        except Exception as e:
            logger.warning(f"Error calculating future projections: {str(e)}")
            return {
                'next_fibonacci_times': [], 'next_cycle_completions': [],
                'projected_reversals': [], 'confluence_windows': []
            }

    def _identify_dominant_cycles(self, time_zones: List[TimeZone]) -> List[Dict]:
        """Identify dominant cycles from the analysis."""
        try:
            cycle_strength = {}

            for zone in time_zones:
                if zone.cycle_length:
                    cycle_len = zone.cycle_length
                    if cycle_len not in cycle_strength:
                        cycle_strength[cycle_len] = []
                    cycle_strength[cycle_len].append(zone.intensity * zone.confidence)

            # Calculate average strength for each cycle
            dominant_cycles = []
            for cycle_len, strengths in cycle_strength.items():
                avg_strength = np.mean(strengths)
                if avg_strength > 0.6:
                    dominant_cycles.append({
                        'cycle_length': cycle_len,
                        'strength': avg_strength,
                        'occurrences': len(strengths)
                    })

            # Sort by strength
            dominant_cycles.sort(key=lambda x: x['strength'], reverse=True)

            return dominant_cycles[:5]  # Top 5 cycles

        except Exception as e:
            logger.warning(f"Error identifying dominant cycles: {str(e)}")
            return []

    def _analyze_cycle_synchronization(self, time_zones: List[TimeZone]) -> Dict[str, Any]:
        """Analyze synchronization between different cycles."""
        try:
            sync_analysis = {
                'synchronized_cycles': [],
                'synchronization_strength': 0.0,
                'next_sync_time': None
            }

            # Group zones by time proximity
            time_groups = {}
            for zone in time_zones:
                if zone.cycle_length:
                    time_key = zone.center_time // 5 * 5  # Group by 5-period windows
                    if time_key not in time_groups:
                        time_groups[time_key] = []
                    time_groups[time_key].append(zone)

            # Find synchronized cycles
            max_sync_count = 0
            best_sync_time = None

            for time_key, zones in time_groups.items():
                if len(zones) >= 3:  # At least 3 cycles
                    cycle_lengths = [z.cycle_length for z in zones]
                    avg_strength = np.mean([z.intensity * z.confidence for z in zones])

                    if len(zones) > max_sync_count:
                        max_sync_count = len(zones)
                        best_sync_time = time_key
                        sync_analysis['synchronized_cycles'] = cycle_lengths
                        sync_analysis['synchronization_strength'] = avg_strength

            sync_analysis['next_sync_time'] = best_sync_time

            return sync_analysis

        except Exception as e:
            logger.warning(f"Error analyzing cycle synchronization: {str(e)}")
            return {'synchronized_cycles': [], 'synchronization_strength': 0.0, 'next_sync_time': None}

    def _predict_next_cycle_events(self, time_zones: List[TimeZone], current_time: int) -> List[Dict]:
        """Predict next significant cycle events."""
        try:
            future_events = []

            # Find cycles with strong patterns
            strong_cycles = [z for z in time_zones
                           if z.cycle_length and z.intensity > 0.7 and z.confidence > 0.7]

            for zone in strong_cycles:
                if zone.center_time > current_time:
                    future_events.append({
                        'time': zone.center_time,
                        'cycle_type': zone.zone_type.value,
                        'cycle_length': zone.cycle_length,
                        'expected_signal': zone.signal_type.value,
                        'strength': zone.intensity * zone.confidence
                    })

            # Sort by time
            future_events.sort(key=lambda x: x['time'])

            return future_events[:10]  # Next 10 events

        except Exception as e:
            logger.warning(f"Error predicting next cycle events: {str(e)}")
            return []

    def _find_next_confluence_time(self, confluence_zones: List[TimeConfluence], current_time: int) -> Optional[int]:
        """Find the next significant confluence time."""
        try:
            future_confluences = [c for c in confluence_zones
                                if c.confluence_time > current_time and c.strength_score > 0.7]

            if future_confluences:
                next_confluence = min(future_confluences, key=lambda c: c.confluence_time)
                return next_confluence.confluence_time

            return None

        except Exception as e:
            logger.warning(f"Error finding next confluence time: {str(e)}")
            return None

    def _format_time_zones(self, time_zones: List[TimeZone]) -> List[Dict]:
        """Format time zones for output."""
        try:
            formatted = []
            for zone in time_zones:
                formatted.append({
                    'zone_type': zone.zone_type.value,
                    'signal_type': zone.signal_type.value,
                    'start_time': zone.start_time,
                    'end_time': zone.end_time,
                    'center_time': zone.center_time,
                    'intensity': zone.intensity,
                    'confidence': zone.confidence,
                    'fibonacci_ratio': zone.fibonacci_ratio,
                    'cycle_length': zone.cycle_length,
                    'harmonic_frequency': zone.harmonic_frequency,
                    'supporting_evidence': zone.supporting_evidence
                })
            return formatted
        except Exception as e:
            logger.warning(f"Error formatting time zones: {str(e)}")
            return []

    def _format_confluence_zones(self, confluence_zones: List[TimeConfluence]) -> List[Dict]:
        """Format confluence zones for output."""
        try:
            formatted = []
            for confluence in confluence_zones:
                formatted.append({
                    'confluence_time': confluence.confluence_time,
                    'zone_count': confluence.zone_count,
                    'total_intensity': confluence.total_intensity,
                    'average_confidence': confluence.average_confidence,
                    'dominant_signal': confluence.dominant_signal.value,
                    'strength_score': confluence.strength_score,
                    'contributing_zone_types': [z.zone_type.value for z in confluence.contributing_zones]
                })
            return formatted
        except Exception as e:
            logger.warning(f"Error formatting confluence zones: {str(e)}")
            return []

    def get_indicator_name(self) -> str:
        """Return the indicator name."""
        return "Time Zone Analysis"

    def get_parameters(self) -> Dict[str, Any]:
        """Return current parameters."""
        return {
            'lookback_period': self.lookback_period,
            'fibonacci_sequence': self.fibonacci_sequence,
            'fibonacci_ratios': self.fibonacci_ratios,
            'min_cycle_length': self.min_cycle_length,
            'max_cycle_length': self.max_cycle_length,
            'confluence_threshold': self.confluence_threshold,
            'intensity_threshold': self.intensity_threshold,
            'seasonal_analysis': self.seasonal_analysis,
            'lunar_analysis': self.lunar_analysis,
            'harmonic_analysis': self.harmonic_analysis,
            'ml_enhanced': self.ml_enhanced
        }
