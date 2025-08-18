"""
Gann Price-Time Indicator

Advanced implementation of W.D. Gann's Price-Time relationship analysis for the humanitarian trading platform.
This indicator performs sophisticated analysis of price-time relationships, cycle detection, mathematical modeling,
and predictive analysis based on Gann's principles.

Features:
- Advanced price-time relationship analysis
- Cycle detection and harmonic analysis
- Mathematical modeling of price-time interactions
- Support and resistance level identification
- Time-based projection and forecasting
- ML-enhanced pattern recognition
- Multi-timeframe analysis
- Risk assessment and trading signals

Author: Assistant
Date: 2025-06-22
Version: 1.0
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import time
import math
from collections import defaultdict, deque
import warnings

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Optional ML imports
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import scipy.stats as stats
    from scipy.signal import find_peaks, periodogram
    from scipy.optimize import minimize
    SKLEARN_AVAILABLE = True
    SCIPY_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SCIPY_AVAILABLE = False


class CycleType(Enum):
    """Types of cycles detected by the indicator"""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    MAJOR = "major"
    GRAND = "grand"


class RelationshipType(Enum):
    """Types of price-time relationships"""
    SQUARE = "square"
    RATIO = "ratio"
    HARMONIC = "harmonic"
    FIBONACCI = "fibonacci"
    GEOMETRIC = "geometric"


@dataclass
class GannPriceTimeConfig:
    """Configuration for Gann Price-Time Indicator"""
    
    # Analysis parameters
    lookback_period: int = 252  # Trading days (1 year)
    min_cycle_length: int = 5   # Minimum cycle length in periods
    max_cycle_length: int = 100 # Maximum cycle length in periods
    
    # Price-time relationship parameters
    square_tolerance: float = 0.05      # Tolerance for square relationships
    ratio_tolerance: float = 0.02       # Tolerance for ratio relationships
    harmonic_precision: float = 0.03    # Precision for harmonic analysis
    
    # Cycle detection parameters
    cycle_strength_threshold: float = 0.3  # Minimum cycle strength
    peak_prominence: float = 0.5           # Peak prominence for cycle detection
    min_cycle_amplitude: float = 0.01      # Minimum cycle amplitude (1%)
    
    # Machine learning parameters
    enable_ml: bool = True
    ml_lookback: int = 500
    cluster_count: int = 5
    feature_scaling: bool = True
    
    # Time analysis parameters
    time_unit: str = "days"  # days, hours, minutes
    harmonic_numbers: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 7, 8, 9, 12, 16, 24])
    fibonacci_ratios: List[float] = field(default_factory=lambda: [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.414, 1.618, 2.0, 2.618])
    
    # Trading parameters
    signal_strength_threshold: float = 0.6
    risk_factor: float = 0.02
    max_projection_periods: int = 50
    
    # Performance parameters
    enable_caching: bool = True
    max_cache_size: int = 1000
    parallel_processing: bool = False


@dataclass
class PriceTimePoint:
    """Represents a significant price-time point"""
    timestamp: float
    price: float
    point_type: str  # high, low, turning_point, projection
    significance: float
    cycle_phase: Optional[float] = None
    harmonic_level: Optional[int] = None
    
    def __post_init__(self):
        self.significance = max(0.0, min(1.0, self.significance))


@dataclass
class PriceTimeRelationship:
    """Represents a price-time relationship"""
    point1: PriceTimePoint
    point2: PriceTimePoint
    relationship_type: RelationshipType
    ratio: float
    strength: float
    time_span: float
    price_span: float
    harmonic_number: Optional[int] = None
    fibonacci_level: Optional[float] = None
    
    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))


@dataclass
class CycleAnalysis:
    """Represents a detected cycle"""
    cycle_type: CycleType
    period: float
    amplitude: float
    phase: float
    strength: float
    start_time: float
    end_time: float
    turning_points: List[PriceTimePoint]
    harmonic_components: List[Tuple[float, float, float]]  # (period, amplitude, phase)
    
    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        self.amplitude = max(0.0, self.amplitude)


@dataclass
class TimeProjection:
    """Represents a time-based price projection"""
    target_time: float
    projected_price: float
    confidence: float
    projection_type: str
    source_relationship: Optional[PriceTimeRelationship] = None
    supporting_cycles: List[CycleAnalysis] = field(default_factory=list)
    
    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))


class GannPriceTimeSignal(NamedTuple):
    """Trading signal from price-time analysis"""
    signal_type: str
    direction: str  # bullish, bearish, neutral
    strength: float
    entry_time: float
    entry_price: float
    target_price: float
    stop_loss: float
    time_target: float
    confidence: float
    cycle_support: bool
    relationship_support: bool


class GannPriceTimeIndicator:
    """
    Advanced Gann Price-Time Indicator
    
    Implements sophisticated price-time relationship analysis based on W.D. Gann's principles.
    Provides cycle detection, harmonic analysis, mathematical modeling, and predictive analysis.
    """
    
    def __init__(self, config: Optional[GannPriceTimeConfig] = None):
        """Initialize the Gann Price-Time Indicator"""
        
        self.config = config or GannPriceTimeConfig()
        self.logger = logging.getLogger(__name__)
        
        # Analysis state
        self.price_data: Optional[np.ndarray] = None
        self.significant_points: List[PriceTimePoint] = []
        self.relationships: List[PriceTimeRelationship] = []
        self.cycles: List[CycleAnalysis] = []
        self.projections: List[TimeProjection] = []
        
        # Analysis results
        self.current_phase: Optional[str] = None
        self.trend_strength: float = 0.0
        self.cycle_position: float = 0.0
        self.next_reversal_time: Optional[float] = None
        
        # ML components
        if self.config.enable_ml and SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            self.cycle_clusterer = None
            self.relationship_model = None
            
        # Caching
        if self.config.enable_caching:
            self._cache: Dict = {}
            self._cache_timestamps: deque = deque(maxlen=self.config.max_cache_size)
            
        self.logger.info("Gann Price-Time Indicator initialized")
        
    def analyze(self, price_data: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive price-time analysis
        
        Args:
            price_data: OHLCV data array [timestamp, open, high, low, close, volume]
            
        Returns:
            Dictionary containing analysis results
        """
        
        if price_data is None or len(price_data) < self.config.min_cycle_length:
            raise ValueError("Insufficient price data for analysis")
            
        self.price_data = price_data.copy()
        
        try:
            # Clear previous analysis
            self._reset_analysis()
            
            self.logger.info(f"Starting price-time analysis on {len(price_data)} data points")
            
            # Step 1: Identify significant points
            self._identify_significant_points()
            
            # Step 2: Detect cycles
            self._detect_cycles()
            
            # Step 3: Analyze price-time relationships
            self._analyze_relationships()
            
            # Step 4: Perform harmonic analysis
            self._harmonic_analysis()
            
            # Step 5: Apply ML analysis if enabled
            if self.config.enable_ml:
                self._ml_enhanced_analysis()
                
            # Step 6: Generate time projections
            self._generate_projections()
            
            # Step 7: Assess current market phase
            self._assess_current_phase()
            
            # Compile results
            results = self._compile_results()
            
            self.logger.info("Price-time analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Price-time analysis failed: {e}")
            return self._get_default_results()
            
    def _reset_analysis(self):
        """Reset analysis state"""
        
        self.significant_points.clear()
        self.relationships.clear()
        self.cycles.clear()
        self.projections.clear()
        
        self.current_phase = None
        self.trend_strength = 0.0
        self.cycle_position = 0.0
        self.next_reversal_time = None
        
    def _identify_significant_points(self):
        """Identify significant highs, lows, and turning points"""
        
        if self.price_data is None or len(self.price_data) < 5:
            return
            
        timestamps = self.price_data[:, 0]
        highs = self.price_data[:, 2]
        lows = self.price_data[:, 3]
        closes = self.price_data[:, 4]
        
        # Find local maxima and minima
        high_peaks, high_properties = self._find_peaks_with_properties(
            highs, prominence=np.std(highs) * 0.5
        )
        
        low_peaks, low_properties = self._find_peaks_with_properties(
            -lows, prominence=np.std(lows) * 0.5
        )
        
        # Convert to significant points
        for idx in high_peaks:
            if 0 <= idx < len(timestamps):
                point = PriceTimePoint(
                    timestamp=timestamps[idx],
                    price=highs[idx],
                    point_type="high",
                    significance=self._calculate_point_significance(idx, highs, high_properties)
                )
                self.significant_points.append(point)
                
        for idx in low_peaks:
            if 0 <= idx < len(timestamps):
                point = PriceTimePoint(
                    timestamp=timestamps[idx],
                    price=lows[idx],
                    point_type="low",
                    significance=self._calculate_point_significance(idx, lows, low_properties)
                )
                self.significant_points.append(point)
                
        # Add turning points based on price action
        turning_points = self._detect_turning_points(closes, timestamps)
        self.significant_points.extend(turning_points)
        
        # Sort by timestamp
        self.significant_points.sort(key=lambda p: p.timestamp)
        
        self.logger.info(f"Identified {len(self.significant_points)} significant points")
        
    def _find_peaks_with_properties(self, data: np.ndarray, prominence: float) -> Tuple[np.ndarray, Dict]:
        """Find peaks with properties using scipy or custom implementation"""
        
        if SCIPY_AVAILABLE:
            try:
                peaks, properties = find_peaks(data, prominence=prominence)
                return peaks, properties
            except Exception:
                pass
                
        # Custom peak detection fallback
        peaks = []
        properties = {'prominences': []}
        
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                # Calculate prominence manually
                left_min = min(data[max(0, i-10):i])
                right_min = min(data[i+1:min(len(data), i+11)])
                prominence_val = data[i] - max(left_min, right_min)
                
                if prominence_val >= prominence:
                    peaks.append(i)
                    properties['prominences'].append(prominence_val)
                    
        return np.array(peaks), properties
        
    def _calculate_point_significance(self, index: int, data: np.ndarray, 
                                    properties: Dict) -> float:
        """Calculate significance of a point"""
        
        if not properties.get('prominences'):
            return 0.5
            
        try:
            # Find the prominence for this index
            prominences = properties['prominences']
            if index < len(prominences):
                prominence = prominences[index]
                max_prominence = max(prominences) if prominences else 1.0
                
                # Normalize significance (0.3 to 1.0)
                significance = 0.3 + 0.7 * (prominence / max_prominence) if max_prominence > 0 else 0.5
                return min(1.0, significance)
                
        except (IndexError, KeyError):
            pass
            
        return 0.5
        
    def _detect_turning_points(self, prices: np.ndarray, timestamps: np.ndarray) -> List[PriceTimePoint]:
        """Detect additional turning points based on price action"""
        
        turning_points = []
        
        if len(prices) < 10:
            return turning_points
            
        # Use rate of change and momentum to detect turning points
        for i in range(5, len(prices) - 5):
            
            # Calculate rate of change
            roc_before = (prices[i] - prices[i-5]) / prices[i-5] if prices[i-5] != 0 else 0
            roc_after = (prices[i+5] - prices[i]) / prices[i] if prices[i] != 0 else 0
            
            # Detect momentum change
            if abs(roc_before) > 0.02 and abs(roc_after) > 0.02:  # Significant moves
                if (roc_before > 0 > roc_after) or (roc_before < 0 < roc_after):
                    
                    significance = min(abs(roc_before) + abs(roc_after), 1.0)
                    
                    point = PriceTimePoint(
                        timestamp=timestamps[i],
                        price=prices[i],
                        point_type="turning_point",
                        significance=significance
                    )
                    turning_points.append(point)
                    
        return turning_points
        
    def _detect_cycles(self):
        """Detect price cycles using advanced analysis"""
        
        if self.price_data is None or len(self.price_data) < self.config.min_cycle_length * 2:
            return
            
        closes = self.price_data[:, 4]
        timestamps = self.price_data[:, 0]
        
        # Method 1: Spectral analysis
        if SCIPY_AVAILABLE:
            self._spectral_cycle_analysis(closes, timestamps)
            
        # Method 2: Peak-to-peak analysis
        self._peak_cycle_analysis()
        
        # Method 3: Harmonic analysis
        self._harmonic_cycle_analysis(closes, timestamps)
        
        # Method 4: ML-based cycle detection
        if self.config.enable_ml:
            self._ml_cycle_detection(closes, timestamps)
            
        # Filter and validate cycles
        self._validate_cycles()
        
        self.logger.info(f"Detected {len(self.cycles)} cycles")
        
    def _spectral_cycle_analysis(self, prices: np.ndarray, timestamps: np.ndarray):
        """Perform spectral analysis to detect cycles"""
        
        try:
            # Detrend the data
            detrended = prices - np.linspace(prices[0], prices[-1], len(prices))
            
            # Compute periodogram
            frequencies, power = periodogram(detrended)
            
            # Find dominant frequencies
            peak_indices = find_peaks(power, height=np.mean(power))[0]
            
            for peak_idx in peak_indices:
                if frequencies[peak_idx] > 0:
                    period = 1.0 / frequencies[peak_idx]
                    
                    if self.config.min_cycle_length <= period <= self.config.max_cycle_length:
                        
                        # Calculate cycle properties
                        amplitude = np.sqrt(power[peak_idx])
                        strength = power[peak_idx] / np.max(power)
                        
                        if strength >= self.config.cycle_strength_threshold:
                            
                            cycle_type = self._classify_cycle_by_period(period)
                            
                            cycle = CycleAnalysis(
                                cycle_type=cycle_type,
                                period=period,
                                amplitude=amplitude,
                                phase=0.0,  # Will be calculated later
                                strength=strength,
                                start_time=timestamps[0],
                                end_time=timestamps[-1],
                                turning_points=[],
                                harmonic_components=[]
                            )
                            
                            self.cycles.append(cycle)
                            
        except Exception as e:
            self.logger.debug(f"Spectral cycle analysis failed: {e}")
            
    def _peak_cycle_analysis(self):
        """Analyze cycles based on peak-to-peak intervals"""
        
        if len(self.significant_points) < 4:
            return
            
        # Group points by type
        highs = [p for p in self.significant_points if p.point_type == "high"]
        lows = [p for p in self.significant_points if p.point_type == "low"]
        
        # Analyze high-to-high cycles
        self._analyze_point_cycles(highs, "high")
        
        # Analyze low-to-low cycles
        self._analyze_point_cycles(lows, "low")
        
        # Analyze high-to-low cycles
        self._analyze_alternating_cycles(highs, lows)
        
    def _analyze_point_cycles(self, points: List[PriceTimePoint], point_type: str):
        """Analyze cycles between similar points"""
        
        if len(points) < 3:
            return
            
        for i in range(len(points) - 2):
            for j in range(i + 2, min(i + 6, len(points))):  # Look ahead 2-5 cycles
                
                p1, p2 = points[i], points[j]
                time_span = p2.timestamp - p1.timestamp
                price_span = abs(p2.price - p1.price)
                
                # Calculate cycle period
                num_cycles = j - i
                period = time_span / num_cycles
                
                if self.config.min_cycle_length <= period <= self.config.max_cycle_length:
                    
                    # Calculate amplitude and strength
                    intermediate_points = points[i+1:j]
                    if intermediate_points:
                        price_range = max(p.price for p in intermediate_points) - min(p.price for p in intermediate_points)
                        amplitude = price_range / 2
                    else:
                        amplitude = price_span / 2
                        
                    # Strength based on consistency and significance
                    avg_significance = np.mean([p.significance for p in points[i:j+1]])
                    strength = avg_significance * min(1.0, amplitude / (p1.price * 0.05))  # Normalize by 5%
                    
                    if strength >= self.config.cycle_strength_threshold:
                        
                        cycle_type = self._classify_cycle_by_period(period)
                        
                        cycle = CycleAnalysis(
                            cycle_type=cycle_type,
                            period=period,
                            amplitude=amplitude,
                            phase=0.0,
                            strength=strength,
                            start_time=p1.timestamp,
                            end_time=p2.timestamp,
                            turning_points=points[i:j+1],
                            harmonic_components=[]
                        )
                        
                        self.cycles.append(cycle)
                        
    def _analyze_alternating_cycles(self, highs: List[PriceTimePoint], lows: List[PriceTimePoint]):
        """Analyze cycles between alternating highs and lows"""
        
        if not highs or not lows:
            return
            
        # Merge and sort all points
        all_points = highs + lows
        all_points.sort(key=lambda p: p.timestamp)
        
        # Find alternating patterns
        for i in range(len(all_points) - 3):
            
            # Look for high-low-high or low-high-low patterns
            p1, p2, p3, p4 = all_points[i:i+4]
            
            if ((p1.point_type == "high" and p2.point_type == "low" and 
                 p3.point_type == "high" and p4.point_type == "low") or
                (p1.point_type == "low" and p2.point_type == "high" and 
                 p3.point_type == "low" and p4.point_type == "high")):
                
                # Full cycle from p1 to p4
                time_span = p4.timestamp - p1.timestamp
                period = time_span  # One complete cycle
                
                if self.config.min_cycle_length <= period <= self.config.max_cycle_length:
                    
                    # Calculate amplitude
                    prices = [p.price for p in [p1, p2, p3, p4]]
                    amplitude = (max(prices) - min(prices)) / 2
                    
                    # Calculate strength
                    avg_significance = np.mean([p.significance for p in [p1, p2, p3, p4]])
                    strength = avg_significance
                    
                    if strength >= self.config.cycle_strength_threshold:
                        
                        cycle_type = self._classify_cycle_by_period(period)
                        
                        cycle = CycleAnalysis(
                            cycle_type=cycle_type,
                            period=period,
                            amplitude=amplitude,
                            phase=0.0,
                            strength=strength,
                            start_time=p1.timestamp,
                            end_time=p4.timestamp,
                            turning_points=[p1, p2, p3, p4],
                            harmonic_components=[]
                        )
                        
                        self.cycles.append(cycle)
                        
    def _classify_cycle_by_period(self, period: float) -> CycleType:
        """Classify cycle based on its period"""
        
        # Convert period to days if needed
        if self.config.time_unit == "hours":
            period_days = period / 24
        elif self.config.time_unit == "minutes":
            period_days = period / (24 * 60)
        else:
            period_days = period
            
        if period_days <= 7:
            return CycleType.SHORT
        elif period_days <= 30:
            return CycleType.MEDIUM
        elif period_days <= 90:
            return CycleType.LONG
        elif period_days <= 365:
            return CycleType.MAJOR
        else:
            return CycleType.GRAND        
    def _harmonic_cycle_analysis(self, prices: np.ndarray, timestamps: np.ndarray):
        """Perform harmonic analysis to detect cycles"""
        
        if len(prices) < 20:
            return
            
        try:
            # Analyze harmonic components using Fourier-like analysis
            for harmonic_num in self.config.harmonic_numbers:
                
                # Test for harmonic periods
                base_period = len(prices) / harmonic_num
                
                if self.config.min_cycle_length <= base_period <= self.config.max_cycle_length:
                    
                    # Calculate harmonic strength
                    harmonic_strength = self._calculate_harmonic_strength(
                        prices, base_period, harmonic_num
                    )
                    
                    if harmonic_strength >= self.config.cycle_strength_threshold:
                        
                        # Calculate amplitude
                        window_size = int(base_period)
                        windowed_prices = []
                        
                        for i in range(0, len(prices) - window_size, window_size):
                            window = prices[i:i + window_size]
                            windowed_prices.append(max(window) - min(window))
                            
                        amplitude = np.mean(windowed_prices) / 2 if windowed_prices else 0
                        
                        cycle_type = self._classify_cycle_by_period(base_period)
                        
                        cycle = CycleAnalysis(
                            cycle_type=cycle_type,
                            period=base_period,
                            amplitude=amplitude,
                            phase=0.0,
                            strength=harmonic_strength,
                            start_time=timestamps[0],
                            end_time=timestamps[-1],
                            turning_points=[],
                            harmonic_components=[(base_period, amplitude, 0.0)]
                        )
                        
                        self.cycles.append(cycle)
                        
        except Exception as e:
            self.logger.debug(f"Harmonic cycle analysis failed: {e}")
            
    def _calculate_harmonic_strength(self, prices: np.ndarray, period: float, harmonic_num: int) -> float:
        """Calculate the strength of a harmonic component"""
        
        try:
            # Simple harmonic analysis
            n = len(prices)
            period_int = int(period)
            
            if period_int <= 1 or period_int >= n:
                return 0.0
                
            # Calculate correlation with idealized harmonic
            ideal_harmonic = np.sin(2 * np.pi * np.arange(n) / period_int)
            
            # Normalize prices
            normalized_prices = (prices - np.mean(prices)) / np.std(prices) if np.std(prices) > 0 else prices
            
            # Calculate correlation
            if SCIPY_AVAILABLE:
                correlation, _ = stats.pearsonr(normalized_prices, ideal_harmonic)
                return abs(correlation)
            else:
                # Manual correlation calculation
                correlation = np.corrcoef(normalized_prices, ideal_harmonic)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0
                
        except Exception:
            return 0.0
            
    def _ml_cycle_detection(self, prices: np.ndarray, timestamps: np.ndarray):
        """Use ML to detect complex cycles"""
        
        if not SKLEARN_AVAILABLE or len(prices) < 50:
            return
            
        try:
            # Prepare features for ML analysis
            features = self._extract_cycle_features(prices)
            
            if len(features) < 10:
                return
                
            X = np.array(features)
            
            # Apply dimensionality reduction
            pca = PCA(n_components=min(5, X.shape[1]))
            X_reduced = pca.fit_transform(X)
            
            # Cluster to identify cycle patterns
            n_clusters = min(self.config.cluster_count, len(features) // 5)
            if n_clusters >= 2:
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(X_reduced)
                
                # Analyze each cluster
                for cluster_id in range(n_clusters):
                    cluster_indices = np.where(cluster_labels == cluster_id)[0]
                    
                    if len(cluster_indices) >= 3:
                        cycle = self._analyze_cluster_cycle(cluster_indices, prices, timestamps)
                        if cycle:
                            self.cycles.append(cycle)
                            
        except Exception as e:
            self.logger.debug(f"ML cycle detection failed: {e}")
            
    def _extract_cycle_features(self, prices: np.ndarray) -> List[List[float]]:
        """Extract features for cycle analysis"""
        
        features = []
        window_size = 20
        
        for i in range(window_size, len(prices) - window_size):
            window = prices[i - window_size:i + window_size]
            
            # Basic statistical features
            feature_vector = [
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                window[-1] - window[0],  # Total change
            ]
            
            # Momentum features
            momentum_5 = prices[i] - prices[i-5] if i >= 5 else 0
            momentum_10 = prices[i] - prices[i-10] if i >= 10 else 0
            
            feature_vector.extend([momentum_5, momentum_10])
            
            # Rate of change features
            roc_short = (prices[i] - prices[i-3]) / prices[i-3] if i >= 3 and prices[i-3] != 0 else 0
            roc_long = (prices[i] - prices[i-10]) / prices[i-10] if i >= 10 and prices[i-10] != 0 else 0
            
            feature_vector.extend([roc_short, roc_long])
            
            # Volatility features
            volatility = np.std(window[-10:]) if len(window) >= 10 else 0
            feature_vector.append(volatility)
            
            features.append(feature_vector)
            
        return features
        
    def _analyze_cluster_cycle(self, indices: np.ndarray, prices: np.ndarray, 
                              timestamps: np.ndarray) -> Optional[CycleAnalysis]:
        """Analyze a cluster to extract cycle information"""
        
        if len(indices) < 3:
            return None
            
        try:
            # Calculate average interval between cluster points
            intervals = []
            for i in range(len(indices) - 1):
                interval = indices[i + 1] - indices[i]
                intervals.append(interval)
                
            if not intervals:
                return None
                
            period = np.mean(intervals)
            
            if not (self.config.min_cycle_length <= period <= self.config.max_cycle_length):
                return None
                
            # Calculate amplitude
            cluster_prices = [prices[idx] for idx in indices]
            amplitude = (max(cluster_prices) - min(cluster_prices)) / 2
            
            # Calculate strength based on regularity
            interval_std = np.std(intervals)
            regularity = 1.0 - (interval_std / period) if period > 0 else 0.0
            strength = min(1.0, regularity * len(indices) / 10)  # Scale by number of occurrences
            
            if strength < self.config.cycle_strength_threshold:
                return None
                
            cycle_type = self._classify_cycle_by_period(period)
            
            cycle = CycleAnalysis(
                cycle_type=cycle_type,
                period=period,
                amplitude=amplitude,
                phase=0.0,
                strength=strength,
                start_time=timestamps[indices[0]],
                end_time=timestamps[indices[-1]],
                turning_points=[],
                harmonic_components=[]
            )
            
            return cycle
            
        except Exception:
            return None
            
    def _validate_cycles(self):
        """Validate and filter detected cycles"""
        
        if not self.cycles:
            return
            
        validated_cycles = []
        
        for cycle in self.cycles:
            # Validation criteria
            is_valid = (
                cycle.strength >= self.config.cycle_strength_threshold and
                cycle.amplitude >= self.config.min_cycle_amplitude and
                self.config.min_cycle_length <= cycle.period <= self.config.max_cycle_length
            )
            
            if is_valid:
                validated_cycles.append(cycle)
                
        # Remove duplicate cycles (similar periods)
        unique_cycles = []
        for cycle in validated_cycles:
            is_duplicate = False
            
            for existing in unique_cycles:
                period_similarity = abs(cycle.period - existing.period) / max(cycle.period, existing.period)
                
                if period_similarity < 0.1:  # 10% similarity threshold
                    # Keep the stronger cycle
                    if cycle.strength > existing.strength:
                        unique_cycles.remove(existing)
                        unique_cycles.append(cycle)
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_cycles.append(cycle)
                
        self.cycles = unique_cycles
        
    def _analyze_relationships(self):
        """Analyze price-time relationships between significant points"""
        
        if len(self.significant_points) < 2:
            return
            
        # Analyze all pairs of significant points
        for i in range(len(self.significant_points)):
            for j in range(i + 1, min(i + 20, len(self.significant_points))):  # Limit to nearby points
                
                point1, point2 = self.significant_points[i], self.significant_points[j]
                relationship = self._analyze_point_relationship(point1, point2)
                
                if relationship:
                    self.relationships.append(relationship)
                    
        # Sort relationships by strength
        self.relationships.sort(key=lambda r: r.strength, reverse=True)
        
        self.logger.info(f"Analyzed {len(self.relationships)} price-time relationships")
        
    def _analyze_point_relationship(self, point1: PriceTimePoint, 
                                   point2: PriceTimePoint) -> Optional[PriceTimeRelationship]:
        """Analyze relationship between two points"""
        
        time_span = point2.timestamp - point1.timestamp
        price_span = abs(point2.price - point1.price)
        
        if time_span <= 0 or price_span <= 0:
            return None
            
        # Test different relationship types
        
        # 1. Square relationship (time equals price in normalized units)
        square_relationship = self._test_square_relationship(point1, point2, time_span, price_span)
        if square_relationship:
            return square_relationship
            
        # 2. Ratio relationships
        ratio_relationship = self._test_ratio_relationship(point1, point2, time_span, price_span)
        if ratio_relationship:
            return ratio_relationship
            
        # 3. Fibonacci relationships
        fibonacci_relationship = self._test_fibonacci_relationship(point1, point2, time_span, price_span)
        if fibonacci_relationship:
            return fibonacci_relationship
            
        # 4. Harmonic relationships
        harmonic_relationship = self._test_harmonic_relationship(point1, point2, time_span, price_span)
        if harmonic_relationship:
            return harmonic_relationship
            
        return None
        
    def _test_square_relationship(self, point1: PriceTimePoint, point2: PriceTimePoint,
                                 time_span: float, price_span: float) -> Optional[PriceTimeRelationship]:
        """Test for square relationship (time = price in normalized units)"""
        
        # Normalize time to days
        if self.config.time_unit == "hours":
            time_days = time_span / 24
        elif self.config.time_unit == "minutes":
            time_days = time_span / (24 * 60)
        else:
            time_days = time_span
            
        # Normalize price to percentage
        price_percent = (price_span / point1.price) * 100 if point1.price > 0 else 0
        
        # Test if time (days) approximately equals price (percent)
        if abs(time_days - price_percent) / max(time_days, price_percent, 1) <= self.config.square_tolerance:
            
            strength = self._calculate_relationship_strength(point1, point2, "square")
            
            return PriceTimeRelationship(
                point1=point1,
                point2=point2,
                relationship_type=RelationshipType.SQUARE,
                ratio=price_percent / time_days if time_days > 0 else 1.0,
                strength=strength,
                time_span=time_span,
                price_span=price_span
            )
            
        return None
        
    def _test_ratio_relationship(self, point1: PriceTimePoint, point2: PriceTimePoint,
                                time_span: float, price_span: float) -> Optional[PriceTimeRelationship]:
        """Test for simple ratio relationships"""
        
        # Test common ratios: 1:1, 1:2, 2:1, 1:3, 3:1, etc.
        test_ratios = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        
        # Calculate actual ratio
        actual_ratio = price_span / time_span if time_span > 0 else 0
        
        for target_ratio in test_ratios:
            if abs(actual_ratio - target_ratio) / target_ratio <= self.config.ratio_tolerance:
                
                strength = self._calculate_relationship_strength(point1, point2, "ratio")
                
                return PriceTimeRelationship(
                    point1=point1,
                    point2=point2,
                    relationship_type=RelationshipType.RATIO,
                    ratio=actual_ratio,
                    strength=strength,
                    time_span=time_span,
                    price_span=price_span
                )
                
        return None
        
    def _test_fibonacci_relationship(self, point1: PriceTimePoint, point2: PriceTimePoint,
                                    time_span: float, price_span: float) -> Optional[PriceTimeRelationship]:
        """Test for Fibonacci relationships"""
        
        # Calculate ratio
        actual_ratio = price_span / time_span if time_span > 0 else 0
        
        for fib_ratio in self.config.fibonacci_ratios:
            if abs(actual_ratio - fib_ratio) / fib_ratio <= self.config.ratio_tolerance:
                
                strength = self._calculate_relationship_strength(point1, point2, "fibonacci")
                
                return PriceTimeRelationship(
                    point1=point1,
                    point2=point2,
                    relationship_type=RelationshipType.FIBONACCI,
                    ratio=actual_ratio,
                    strength=strength,
                    time_span=time_span,
                    price_span=price_span,
                    fibonacci_level=fib_ratio
                )
                
        return None
        
    def _test_harmonic_relationship(self, point1: PriceTimePoint, point2: PriceTimePoint,
                                   time_span: float, price_span: float) -> Optional[PriceTimeRelationship]:
        """Test for harmonic relationships"""
        
        for harmonic_num in self.config.harmonic_numbers:
            # Test if time or price span is a harmonic of a base unit
            
            # Test time harmonics
            for base_time in [1, 7, 30, 90]:  # 1 day, 1 week, 1 month, 1 quarter
                if abs(time_span - base_time * harmonic_num) / (base_time * harmonic_num) <= self.config.harmonic_precision:
                    
                    strength = self._calculate_relationship_strength(point1, point2, "harmonic")
                    
                    return PriceTimeRelationship(
                        point1=point1,
                        point2=point2,
                        relationship_type=RelationshipType.HARMONIC,
                        ratio=price_span / time_span if time_span > 0 else 0,
                        strength=strength,
                        time_span=time_span,
                        price_span=price_span,
                        harmonic_number=harmonic_num
                    )
                    
            # Test price harmonics (as percentage of base price)
            base_percentages = [1, 2, 5, 10]  # 1%, 2%, 5%, 10%
            price_percent = (price_span / point1.price) * 100 if point1.price > 0 else 0
            
            for base_pct in base_percentages:
                if abs(price_percent - base_pct * harmonic_num) / (base_pct * harmonic_num) <= self.config.harmonic_precision:
                    
                    strength = self._calculate_relationship_strength(point1, point2, "harmonic")
                    
                    return PriceTimeRelationship(
                        point1=point1,
                        point2=point2,
                        relationship_type=RelationshipType.HARMONIC,
                        ratio=price_span / time_span if time_span > 0 else 0,
                        strength=strength,
                        time_span=time_span,
                        price_span=price_span,
                        harmonic_number=harmonic_num
                    )
                    
        return None
        
    def _calculate_relationship_strength(self, point1: PriceTimePoint, point2: PriceTimePoint,
                                       relationship_type: str) -> float:
        """Calculate strength of a relationship"""
        
        # Base strength from point significance
        base_strength = (point1.significance + point2.significance) / 2
        
        # Type-specific bonuses
        type_bonus = {
            "square": 0.3,
            "ratio": 0.2,
            "fibonacci": 0.25,
            "harmonic": 0.2,
            "geometric": 0.15
        }.get(relationship_type, 0.1)
        
        # Time span factor (longer relationships are potentially more significant)
        time_factor = min(1.0, (point2.timestamp - point1.timestamp) / (30 * 24 * 3600))  # Normalize to 30 days
        
        # Calculate combined strength
        strength = base_strength * 0.6 + type_bonus + time_factor * 0.1
        
        return max(0.0, min(1.0, strength))
        
    def _harmonic_analysis(self):
        """Perform harmonic analysis on detected relationships and cycles"""
        
        if not self.relationships and not self.cycles:
            return
            
        # Analyze harmonic relationships between cycles
        self._analyze_cycle_harmonics()
        
        # Analyze harmonic patterns in price-time relationships
        self._analyze_relationship_harmonics()
        
        # Update cycle phases based on harmonic analysis
        self._update_cycle_phases()
        
        self.logger.info("Harmonic analysis completed")
        
    def _analyze_cycle_harmonics(self):
        """Analyze harmonic relationships between cycles"""
        
        if len(self.cycles) < 2:
            return
            
        for i in range(len(self.cycles)):
            for j in range(i + 1, len(self.cycles)):
                
                cycle1, cycle2 = self.cycles[i], self.cycles[j]
                
                # Check if cycles are harmonically related
                ratio = cycle1.period / cycle2.period if cycle2.period > 0 else 0
                
                for harmonic_num in self.config.harmonic_numbers:
                    if abs(ratio - harmonic_num) / harmonic_num <= self.config.harmonic_precision:
                        
                        # Add harmonic component to the longer cycle
                        longer_cycle = cycle1 if cycle1.period > cycle2.period else cycle2
                        shorter_cycle = cycle2 if cycle1.period > cycle2.period else cycle1
                        
                        harmonic_component = (
                            shorter_cycle.period,
                            shorter_cycle.amplitude,
                            shorter_cycle.phase
                        )
                        
                        if harmonic_component not in longer_cycle.harmonic_components:
                            longer_cycle.harmonic_components.append(harmonic_component)
                            
    def _analyze_relationship_harmonics(self):
        """Analyze harmonic patterns in price-time relationships"""
        
        harmonic_relationships = [r for r in self.relationships 
                                if r.relationship_type == RelationshipType.HARMONIC]
        
        if len(harmonic_relationships) < 2:
            return
            
        # Group relationships by harmonic number
        harmonic_groups = defaultdict(list)
        for rel in harmonic_relationships:
            if rel.harmonic_number:
                harmonic_groups[rel.harmonic_number].append(rel)
                
        # Analyze each harmonic group
        for harmonic_num, relationships in harmonic_groups.items():
            if len(relationships) >= 2:
                
                # Calculate average strength for this harmonic
                avg_strength = np.mean([r.strength for r in relationships])
                
                # Update point significance based on harmonic participation
                for rel in relationships:
                    if rel.point1.harmonic_level is None:
                        rel.point1.harmonic_level = harmonic_num
                    if rel.point2.harmonic_level is None:
                        rel.point2.harmonic_level = harmonic_num
                        
    def _update_cycle_phases(self):
        """Update cycle phases based on current market position"""
        
        if not self.cycles or not self.price_data:
            return
            
        current_time = self.price_data[-1, 0]
        current_price = self.price_data[-1, 4]
        
        for cycle in self.cycles:
            if cycle.period > 0:
                
                # Calculate phase based on time position in cycle
                time_since_start = current_time - cycle.start_time
                phase = (time_since_start % cycle.period) / cycle.period
                cycle.phase = phase * 2 * np.pi  # Convert to radians
                
                # Update turning points with cycle phases
                for point in cycle.turning_points:
                    point_time_offset = point.timestamp - cycle.start_time
                    point_phase = (point_time_offset % cycle.period) / cycle.period if cycle.period > 0 else 0
                    point.cycle_phase = point_phase * 2 * np.pi
                    
    def _ml_enhanced_analysis(self):
        """Apply ML enhancement to the analysis"""
        
        if not SKLEARN_AVAILABLE or not self.cycles:
            return
            
        try:
            # ML-enhanced cycle analysis
            self._ml_cycle_enhancement()
            
            # ML-enhanced relationship analysis
            self._ml_relationship_enhancement()
            
            # Predictive modeling
            self._ml_predictive_modeling()
            
            self.logger.info("ML enhancement completed")
            
        except Exception as e:
            self.logger.debug(f"ML enhancement failed: {e}")
            
    def _ml_cycle_enhancement(self):
        """Use ML to enhance cycle analysis"""
        
        if len(self.cycles) < 2:
            return
            
        try:
            # Extract cycle features
            cycle_features = []
            for cycle in self.cycles:
                features = [
                    cycle.period,
                    cycle.amplitude,
                    cycle.strength,
                    len(cycle.turning_points),
                    len(cycle.harmonic_components)
                ]
                cycle_features.append(features)
                
            X = np.array(cycle_features)
            
            # Normalize features
            if self.config.feature_scaling:
                X = self.scaler.fit_transform(X)
                
            # Cluster cycles by similarity
            n_clusters = min(3, len(self.cycles))
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(X)
            
            # Update cycle strengths based on cluster membership
            for i, cycle in enumerate(self.cycles):
                cluster_size = np.sum(cluster_labels == cluster_labels[i])
                cluster_bonus = min(0.2, cluster_size / len(self.cycles))
                cycle.strength = min(1.0, cycle.strength + cluster_bonus)
                
        except Exception as e:
            self.logger.debug(f"ML cycle enhancement failed: {e}")
            
    def _ml_relationship_enhancement(self):
        """Use ML to enhance relationship analysis"""
        
        if len(self.relationships) < 5:
            return
            
        try:
            # Extract relationship features
            rel_features = []
            for rel in self.relationships:
                features = [
                    rel.ratio,
                    rel.strength,
                    rel.time_span,
                    rel.price_span,
                    rel.point1.significance,
                    rel.point2.significance,
                    1.0 if rel.relationship_type == RelationshipType.FIBONACCI else 0.0,
                    1.0 if rel.relationship_type == RelationshipType.HARMONIC else 0.0,
                    1.0 if rel.relationship_type == RelationshipType.SQUARE else 0.0
                ]
                rel_features.append(features)
                
            X = np.array(rel_features)
            
            # Normalize features
            if self.config.feature_scaling:
                X = self.scaler.fit_transform(X)
                
            # Use PCA for dimensionality reduction
            pca = PCA(n_components=min(5, X.shape[1]))
            X_reduced = pca.fit_transform(X)
            
            # Cluster relationships
            n_clusters = min(3, len(self.relationships) // 2)
            if n_clusters >= 2:
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(X_reduced)
                
                # Enhance relationship strengths based on clustering
                for i, rel in enumerate(self.relationships):
                    cluster_size = np.sum(cluster_labels == cluster_labels[i])
                    cluster_bonus = min(0.15, cluster_size / len(self.relationships))
                    rel.strength = min(1.0, rel.strength + cluster_bonus)
                    
        except Exception as e:
            self.logger.debug(f"ML relationship enhancement failed: {e}")
            
    def _ml_predictive_modeling(self):
        """Build ML models for price prediction"""
        
        if not self.price_data or len(self.price_data) < 50:
            return
            
        try:
            # Prepare features for prediction
            features, targets = self._prepare_prediction_data()
            
            if len(features) < 20:
                return
                
            X = np.array(features)
            y = np.array(targets)
            
            # Split data for training/validation
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train linear regression model
            self.relationship_model = LinearRegression()
            self.relationship_model.fit(X_train, y_train)
            
            # Evaluate model
            val_predictions = self.relationship_model.predict(X_val)
            r2 = r2_score(y_val, val_predictions)
            
            if r2 > 0.1:  # Minimum acceptable R²
                self.logger.info(f"ML prediction model trained with R² = {r2:.3f}")
            else:
                self.relationship_model = None
                
        except Exception as e:
            self.logger.debug(f"ML predictive modeling failed: {e}")
            
    def _prepare_prediction_data(self) -> Tuple[List[List[float]], List[float]]:
        """Prepare data for ML prediction"""
        
        features = []
        targets = []
        
        lookback = 10
        forecast_horizon = 5
        
        prices = self.price_data[:, 4]  # Close prices
        
        for i in range(lookback, len(prices) - forecast_horizon):
            
            # Feature: price history, cycles, relationships
            feature_vector = []
            
            # Price features
            recent_prices = prices[i-lookback:i]
            feature_vector.extend([
                np.mean(recent_prices),
                np.std(recent_prices),
                recent_prices[-1] - recent_prices[0],  # Total change
                (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2] if recent_prices[-2] != 0 else 0  # Last change
            ])
            
            # Cycle features (simplified)
            cycle_strength = np.mean([c.strength for c in self.cycles]) if self.cycles else 0.0
            cycle_phase = np.mean([c.phase for c in self.cycles]) if self.cycles else 0.0
            
            feature_vector.extend([cycle_strength, cycle_phase])
            
            # Relationship features (simplified)
            rel_strength = np.mean([r.strength for r in self.relationships]) if self.relationships else 0.0
            feature_vector.append(rel_strength)
            
            # Target: future price change
            future_price = prices[i + forecast_horizon]
            current_price = prices[i]
            target = (future_price - current_price) / current_price if current_price != 0 else 0
            
            features.append(feature_vector)
            targets.append(target)
            
        return features, targets        
    def _generate_projections(self):
        """Generate time-based price projections"""
        
        if not self.relationships and not self.cycles:
            return
            
        current_time = self.price_data[-1, 0] if self.price_data is not None else time.time()
        current_price = self.price_data[-1, 4] if self.price_data is not None else 0
        
        # Generate projections from relationships
        self._generate_relationship_projections(current_time, current_price)
        
        # Generate projections from cycles
        self._generate_cycle_projections(current_time, current_price)
        
        # Generate ML-based projections
        if self.config.enable_ml and self.relationship_model:
            self._generate_ml_projections(current_time, current_price)
            
        # Sort projections by confidence
        self.projections.sort(key=lambda p: p.confidence, reverse=True)
        
        # Limit number of projections
        max_projections = 20
        if len(self.projections) > max_projections:
            self.projections = self.projections[:max_projections]
            
        self.logger.info(f"Generated {len(self.projections)} time projections")
        
    def _generate_relationship_projections(self, current_time: float, current_price: float):
        """Generate projections based on price-time relationships"""
        
        for relationship in self.relationships[:10]:  # Top 10 relationships
            
            if relationship.strength < 0.5:
                continue
                
            # Project forward based on relationship pattern
            projection = self._project_from_relationship(relationship, current_time, current_price)
            
            if projection:
                self.projections.append(projection)
                
    def _project_from_relationship(self, relationship: PriceTimeRelationship,
                                  current_time: float, current_price: float) -> Optional[TimeProjection]:
        """Project future price-time point from a relationship"""
        
        try:
            # Calculate time projection based on relationship type
            if relationship.relationship_type == RelationshipType.SQUARE:
                # For square relationships, project equal time and price moves
                time_advance = relationship.time_span
                price_advance = relationship.price_span
                
                # Apply direction based on last movement
                price_direction = 1 if relationship.point2.price > relationship.point1.price else -1
                
                target_time = current_time + time_advance
                projected_price = current_price + (price_advance * price_direction)
                
            elif relationship.relationship_type == RelationshipType.FIBONACCI:
                # Project using Fibonacci extensions
                time_advance = relationship.time_span * relationship.fibonacci_level
                price_advance = relationship.price_span * relationship.fibonacci_level
                
                price_direction = 1 if relationship.point2.price > relationship.point1.price else -1
                
                target_time = current_time + time_advance
                projected_price = current_price + (price_advance * price_direction)
                
            elif relationship.relationship_type == RelationshipType.HARMONIC:
                # Project using harmonic multiples
                harmonic_mult = relationship.harmonic_number or 2
                time_advance = relationship.time_span / harmonic_mult
                price_advance = relationship.price_span / harmonic_mult
                
                price_direction = 1 if relationship.point2.price > relationship.point1.price else -1
                
                target_time = current_time + time_advance
                projected_price = current_price + (price_advance * price_direction)
                
            else:
                # Default ratio-based projection
                time_advance = relationship.time_span
                price_advance = relationship.price_span
                
                price_direction = 1 if relationship.point2.price > relationship.point1.price else -1
                
                target_time = current_time + time_advance
                projected_price = current_price + (price_advance * price_direction)
                
            # Ensure target time is not too far in the future
            max_time_advance = self.config.max_projection_periods * 24 * 3600  # Convert to seconds
            if target_time - current_time > max_time_advance:
                return None
                
            # Calculate confidence based on relationship strength and recency
            base_confidence = relationship.strength
            
            # Reduce confidence for older relationships
            relationship_age = current_time - relationship.point2.timestamp
            age_factor = max(0.1, 1.0 - relationship_age / (30 * 24 * 3600))  # 30 days decay
            
            confidence = base_confidence * age_factor
            
            if confidence < 0.3:
                return None
                
            projection = TimeProjection(
                target_time=target_time,
                projected_price=projected_price,
                confidence=confidence,
                projection_type=f"relationship_{relationship.relationship_type.value}",
                source_relationship=relationship
            )
            
            return projection
            
        except Exception as e:
            self.logger.debug(f"Relationship projection failed: {e}")
            return None
            
    def _generate_cycle_projections(self, current_time: float, current_price: float):
        """Generate projections based on cycle analysis"""
        
        for cycle in self.cycles[:5]:  # Top 5 cycles
            
            if cycle.strength < 0.4:
                continue
                
            projection = self._project_from_cycle(cycle, current_time, current_price)
            
            if projection:
                self.projections.append(projection)
                
    def _project_from_cycle(self, cycle: CycleAnalysis,
                           current_time: float, current_price: float) -> Optional[TimeProjection]:
        """Project future price point from a cycle"""
        
        try:
            # Calculate next turning point based on cycle
            phase_to_next_turn = np.pi - (cycle.phase % np.pi)  # Next 180° phase
            time_to_next_turn = (phase_to_next_turn / (2 * np.pi)) * cycle.period
            
            target_time = current_time + time_to_next_turn
            
            # Project price based on cycle amplitude and current phase
            cycle_direction = 1 if np.sin(cycle.phase) > 0 else -1
            projected_price_change = cycle.amplitude * cycle_direction
            projected_price = current_price + projected_price_change
            
            # Ensure reasonable projection time
            max_time_advance = self.config.max_projection_periods * 24 * 3600
            if target_time - current_time > max_time_advance:
                return None
                
            # Calculate confidence
            confidence = cycle.strength * 0.8  # Slightly lower confidence for cycle projections
            
            if confidence < 0.3:
                return None
                
            projection = TimeProjection(
                target_time=target_time,
                projected_price=projected_price,
                confidence=confidence,
                projection_type=f"cycle_{cycle.cycle_type.value}",
                supporting_cycles=[cycle]
            )
            
            return projection
            
        except Exception as e:
            self.logger.debug(f"Cycle projection failed: {e}")
            return None
            
    def _generate_ml_projections(self, current_time: float, current_price: float):
        """Generate ML-based projections"""
        
        try:
            if not self.price_data or len(self.price_data) < 20:
                return
                
            # Prepare current features
            current_features = self._extract_current_features()
            
            if not current_features:
                return
                
            # Generate multiple projections for different time horizons
            time_horizons = [1, 3, 5, 7, 10]  # Days
            
            for horizon in time_horizons:
                
                # Predict price change
                feature_vector = np.array(current_features).reshape(1, -1)
                predicted_change = self.relationship_model.predict(feature_vector)[0]
                
                # Calculate target price
                projected_price = current_price * (1 + predicted_change)
                target_time = current_time + horizon * 24 * 3600
                
                # Calculate confidence based on model performance and horizon
                base_confidence = 0.6  # ML model base confidence
                horizon_penalty = 1.0 - (horizon / 20)  # Reduce confidence for longer horizons
                confidence = base_confidence * horizon_penalty
                
                if confidence >= 0.3:
                    projection = TimeProjection(
                        target_time=target_time,
                        projected_price=projected_price,
                        confidence=confidence,
                        projection_type=f"ml_prediction_{horizon}d"
                    )
                    
                    self.projections.append(projection)
                    
        except Exception as e:
            self.logger.debug(f"ML projection generation failed: {e}")
            
    def _extract_current_features(self) -> Optional[List[float]]:
        """Extract current features for ML prediction"""
        
        try:
            if not self.price_data or len(self.price_data) < 10:
                return None
                
            # Use same feature extraction as training
            prices = self.price_data[:, 4]
            recent_prices = prices[-10:]
            
            features = [
                np.mean(recent_prices),
                np.std(recent_prices),
                recent_prices[-1] - recent_prices[0],
                (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2] if recent_prices[-2] != 0 else 0
            ]
            
            # Cycle features
            cycle_strength = np.mean([c.strength for c in self.cycles]) if self.cycles else 0.0
            cycle_phase = np.mean([c.phase for c in self.cycles]) if self.cycles else 0.0
            features.extend([cycle_strength, cycle_phase])
            
            # Relationship features
            rel_strength = np.mean([r.strength for r in self.relationships]) if self.relationships else 0.0
            features.append(rel_strength)
            
            return features
            
        except Exception:
            return None
            
    def _assess_current_phase(self):
        """Assess current market phase based on analysis"""
        
        if not self.cycles and not self.relationships:
            self.current_phase = "uncertain"
            return
            
        # Analyze cycle positions
        cycle_phases = []
        cycle_strengths = []
        
        for cycle in self.cycles:
            if cycle.strength >= 0.4:
                # Determine cycle phase
                phase_normalized = (cycle.phase % (2 * np.pi)) / (2 * np.pi)
                
                if 0 <= phase_normalized < 0.25:
                    phase_name = "accumulation"
                elif 0.25 <= phase_normalized < 0.75:
                    phase_name = "trending"
                else:
                    phase_name = "distribution"
                    
                cycle_phases.append(phase_name)
                cycle_strengths.append(cycle.strength)
                
        # Determine dominant phase
        if cycle_phases:
            # Weight phases by cycle strength
            phase_weights = {}
            for phase, strength in zip(cycle_phases, cycle_strengths):
                phase_weights[phase] = phase_weights.get(phase, 0) + strength
                
            self.current_phase = max(phase_weights, key=phase_weights.get)
        else:
            self.current_phase = "neutral"
            
        # Calculate trend strength
        if self.relationships:
            recent_relationships = [r for r in self.relationships 
                                  if self.price_data and r.point2.timestamp > self.price_data[-1, 0] - 7*24*3600]
            
            if recent_relationships:
                bullish_strength = sum(r.strength for r in recent_relationships 
                                     if r.point2.price > r.point1.price)
                bearish_strength = sum(r.strength for r in recent_relationships 
                                     if r.point2.price < r.point1.price)
                
                total_strength = bullish_strength + bearish_strength
                if total_strength > 0:
                    self.trend_strength = (bullish_strength - bearish_strength) / total_strength
                else:
                    self.trend_strength = 0.0
            else:
                self.trend_strength = 0.0
        else:
            self.trend_strength = 0.0
            
        # Calculate cycle position (0 = trough, 1 = peak)
        if self.cycles:
            weighted_position = 0.0
            total_weight = 0.0
            
            for cycle in self.cycles:
                if cycle.strength >= 0.3:
                    phase_position = np.sin(cycle.phase)  # -1 to 1
                    position = (phase_position + 1) / 2  # 0 to 1
                    
                    weighted_position += position * cycle.strength
                    total_weight += cycle.strength
                    
            if total_weight > 0:
                self.cycle_position = weighted_position / total_weight
            else:
                self.cycle_position = 0.5
        else:
            self.cycle_position = 0.5
            
        # Estimate next reversal time
        self._estimate_next_reversal()
        
    def _estimate_next_reversal(self):
        """Estimate time of next market reversal"""
        
        if not self.cycles:
            return
            
        current_time = self.price_data[-1, 0] if self.price_data is not None else time.time()
        
        reversal_times = []
        weights = []
        
        for cycle in self.cycles:
            if cycle.strength >= 0.4:
                
                # Calculate time to next reversal (half cycle)
                current_phase = cycle.phase % (2 * np.pi)
                
                if current_phase < np.pi:
                    # Moving towards peak
                    time_to_reversal = ((np.pi - current_phase) / (2 * np.pi)) * cycle.period
                else:
                    # Moving towards trough
                    time_to_reversal = ((2 * np.pi - current_phase) / (2 * np.pi)) * cycle.period
                    
                reversal_time = current_time + time_to_reversal
                
                reversal_times.append(reversal_time)
                weights.append(cycle.strength)
                
        if reversal_times:
            # Calculate weighted average of reversal times
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_reversal = sum(t * w for t, w in zip(reversal_times, weights)) / total_weight
                self.next_reversal_time = weighted_reversal
                
    def _compile_results(self) -> Dict[str, Any]:
        """Compile analysis results"""
        
        # Generate trading signals
        signals = self._generate_trading_signals()
        
        results = {
            # Analysis summary
            'timestamp': time.time(),
            'data_points': len(self.price_data) if self.price_data is not None else 0,
            
            # Market phase assessment
            'current_phase': self.current_phase,
            'trend_strength': self.trend_strength,
            'cycle_position': self.cycle_position,
            'next_reversal_time': self.next_reversal_time,
            
            # Detected patterns
            'significant_points_count': len(self.significant_points),
            'relationships_count': len(self.relationships),
            'cycles_count': len(self.cycles),
            'projections_count': len(self.projections),
            
            # Detailed analysis
            'significant_points': [self._point_to_dict(p) for p in self.significant_points[:10]],
            'relationships': [self._relationship_to_dict(r) for r in self.relationships[:5]],
            'cycles': [self._cycle_to_dict(c) for c in self.cycles[:5]],
            'projections': [self._projection_to_dict(p) for p in self.projections[:5]],
            
            # Trading signals
            'signals': [self._signal_to_dict(s) for s in signals],
            
            # Quality metrics
            'analysis_quality': self._calculate_analysis_quality(),
            'confidence': self._calculate_overall_confidence()
        }
        
        return results
        
    def _generate_trading_signals(self) -> List[GannPriceTimeSignal]:
        """Generate trading signals from analysis"""
        
        signals = []
        
        if not self.price_data:
            return signals
            
        current_time = self.price_data[-1, 0]
        current_price = self.price_data[-1, 4]
        
        # Signal from cycle analysis
        cycle_signal = self._generate_cycle_signal(current_time, current_price)
        if cycle_signal:
            signals.append(cycle_signal)
            
        # Signal from relationship analysis
        relationship_signal = self._generate_relationship_signal(current_time, current_price)
        if relationship_signal:
            signals.append(relationship_signal)
            
        # Signal from projections
        projection_signal = self._generate_projection_signal(current_time, current_price)
        if projection_signal:
            signals.append(projection_signal)
            
        # Combined signal
        if len(signals) >= 2:
            combined_signal = self._generate_combined_signal(signals, current_time, current_price)
            if combined_signal:
                signals.append(combined_signal)
                
        return signals
        
    def _generate_cycle_signal(self, current_time: float, current_price: float) -> Optional[GannPriceTimeSignal]:
        """Generate signal based on cycle analysis"""
        
        if not self.cycles:
            return None
            
        # Find strongest cycle near reversal point
        best_cycle = None
        best_score = 0.0
        
        for cycle in self.cycles:
            if cycle.strength >= 0.5:
                
                # Calculate distance to reversal point
                current_phase = cycle.phase % (2 * np.pi)
                distance_to_reversal = min(abs(current_phase - np.pi), abs(current_phase - 2*np.pi), current_phase)
                
                # Score based on strength and proximity to reversal
                proximity_score = 1.0 - (distance_to_reversal / np.pi)
                score = cycle.strength * proximity_score
                
                if score > best_score:
                    best_score = score
                    best_cycle = cycle
                    
        if not best_cycle or best_score < 0.3:
            return None
            
        # Determine signal direction
        current_phase = best_cycle.phase % (2 * np.pi)
        
        if current_phase < np.pi:
            direction = "bearish"  # Approaching peak
        else:
            direction = "bullish"  # Approaching trough
            
        # Calculate targets
        if direction == "bullish":
            target_price = current_price + best_cycle.amplitude
            stop_loss = current_price - best_cycle.amplitude * 0.5
        else:
            target_price = current_price - best_cycle.amplitude
            stop_loss = current_price + best_cycle.amplitude * 0.5
            
        # Calculate time target
        time_to_reversal = ((np.pi - (current_phase % np.pi)) / (2 * np.pi)) * best_cycle.period
        time_target = current_time + time_to_reversal
        
        signal = GannPriceTimeSignal(
            signal_type="cycle_reversal",
            direction=direction,
            strength=best_score,
            entry_time=current_time,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            time_target=time_target,
            confidence=best_cycle.strength,
            cycle_support=True,
            relationship_support=False
        )
        
        return signal
        
    def _generate_relationship_signal(self, current_time: float, current_price: float) -> Optional[GannPriceTimeSignal]:
        """Generate signal based on relationship analysis"""
        
        if not self.relationships:
            return None
            
        # Find strongest recent relationship
        recent_relationships = [r for r in self.relationships 
                              if r.point2.timestamp > current_time - 7*24*3600]  # Last 7 days
        
        if not recent_relationships:
            return None
            
        best_relationship = max(recent_relationships, key=lambda r: r.strength)
        
        if best_relationship.strength < 0.6:
            return None
            
        # Determine direction based on relationship trend
        if best_relationship.point2.price > best_relationship.point1.price:
            direction = "bullish"
        else:
            direction = "bearish"
            
        # Calculate targets based on relationship projection
        price_span = abs(best_relationship.point2.price - best_relationship.point1.price)
        
        if direction == "bullish":
            target_price = current_price + price_span
            stop_loss = current_price - price_span * 0.5
        else:
            target_price = current_price - price_span
            stop_loss = current_price + price_span * 0.5
            
        # Time target based on relationship timespan
        time_target = current_time + best_relationship.time_span
        
        signal = GannPriceTimeSignal(
            signal_type="relationship_continuation",
            direction=direction,
            strength=best_relationship.strength,
            entry_time=current_time,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            time_target=time_target,
            confidence=best_relationship.strength,
            cycle_support=False,
            relationship_support=True
        )
        
        return signal
        
    def _generate_projection_signal(self, current_time: float, current_price: float) -> Optional[GannPriceTimeSignal]:
        """Generate signal based on projections"""
        
        if not self.projections:
            return None
            
        # Find best near-term projection
        near_projections = [p for p in self.projections 
                           if p.target_time <= current_time + 7*24*3600]  # Next 7 days
        
        if not near_projections:
            return None
            
        best_projection = max(near_projections, key=lambda p: p.confidence)
        
        if best_projection.confidence < 0.5:
            return None
            
        # Determine direction
        if best_projection.projected_price > current_price:
            direction = "bullish"
        else:
            direction = "bearish"
            
        # Calculate targets
        price_move = abs(best_projection.projected_price - current_price)
        
        if direction == "bullish":
            target_price = best_projection.projected_price
            stop_loss = current_price - price_move * 0.5
        else:
            target_price = best_projection.projected_price
            stop_loss = current_price + price_move * 0.5
            
        signal = GannPriceTimeSignal(
            signal_type="projection_target",
            direction=direction,
            strength=best_projection.confidence,
            entry_time=current_time,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            time_target=best_projection.target_time,
            confidence=best_projection.confidence,
            cycle_support=bool(best_projection.supporting_cycles),
            relationship_support=bool(best_projection.source_relationship)
        )
        
        return signal
        
    def _generate_combined_signal(self, individual_signals: List[GannPriceTimeSignal],
                                 current_time: float, current_price: float) -> Optional[GannPriceTimeSignal]:
        """Generate combined signal from multiple individual signals"""
        
        if len(individual_signals) < 2:
            return None
            
        # Check signal agreement
        bullish_signals = [s for s in individual_signals if s.direction == "bullish"]
        bearish_signals = [s for s in individual_signals if s.direction == "bearish"]
        
        if len(bullish_signals) > len(bearish_signals):
            direction = "bullish"
            supporting_signals = bullish_signals
        elif len(bearish_signals) > len(bullish_signals):
            direction = "bearish"
            supporting_signals = bearish_signals
        else:
            return None  # No clear consensus
            
        if len(supporting_signals) < 2:
            return None
            
        # Calculate combined metrics
        combined_strength = np.mean([s.strength for s in supporting_signals])
        combined_confidence = np.mean([s.confidence for s in supporting_signals])
        
        # Calculate average targets
        avg_target = np.mean([s.target_price for s in supporting_signals])
        avg_stop = np.mean([s.stop_loss for s in supporting_signals])
        avg_time_target = np.mean([s.time_target for s in supporting_signals])
        
        # Check for cycle and relationship support
        cycle_support = any(s.cycle_support for s in supporting_signals)
        relationship_support = any(s.relationship_support for s in supporting_signals)
        
        # Boost confidence for multi-signal agreement
        agreement_bonus = min(0.2, len(supporting_signals) * 0.05)
        combined_confidence = min(1.0, combined_confidence + agreement_bonus)
        
        signal = GannPriceTimeSignal(
            signal_type="combined_analysis",
            direction=direction,
            strength=combined_strength,
            entry_time=current_time,
            entry_price=current_price,
            target_price=avg_target,
            stop_loss=avg_stop,
            time_target=avg_time_target,
            confidence=combined_confidence,
            cycle_support=cycle_support,
            relationship_support=relationship_support
        )
        
        return signal        
    def _calculate_analysis_quality(self) -> float:
        """Calculate overall quality of the analysis"""
        
        quality_factors = []
        
        # Data quality factor
        if self.price_data is not None:
            data_quality = min(1.0, len(self.price_data) / 100)  # Normalize to 100 points
            quality_factors.append(data_quality)
            
        # Significant points quality
        if self.significant_points:
            avg_significance = np.mean([p.significance for p in self.significant_points])
            quality_factors.append(avg_significance)
        else:
            quality_factors.append(0.0)
            
        # Relationships quality
        if self.relationships:
            avg_rel_strength = np.mean([r.strength for r in self.relationships])
            quality_factors.append(avg_rel_strength)
        else:
            quality_factors.append(0.0)
            
        # Cycles quality
        if self.cycles:
            avg_cycle_strength = np.mean([c.strength for c in self.cycles])
            quality_factors.append(avg_cycle_strength)
        else:
            quality_factors.append(0.0)
            
        # Projections quality
        if self.projections:
            avg_proj_confidence = np.mean([p.confidence for p in self.projections])
            quality_factors.append(avg_proj_confidence)
        else:
            quality_factors.append(0.0)
            
        return np.mean(quality_factors) if quality_factors else 0.0
        
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence in the analysis"""
        
        confidence_factors = []
        
        # Phase assessment confidence
        if self.current_phase and self.current_phase != "uncertain":
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
            
        # Trend strength confidence
        if abs(self.trend_strength) > 0.5:
            confidence_factors.append(0.9)
        elif abs(self.trend_strength) > 0.2:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
            
        # Pattern consistency confidence
        strong_patterns = (
            len([p for p in self.significant_points if p.significance > 0.7]) +
            len([r for r in self.relationships if r.strength > 0.7]) +
            len([c for c in self.cycles if c.strength > 0.7])
        )
        
        pattern_confidence = min(1.0, strong_patterns / 5)  # Normalize to 5 strong patterns
        confidence_factors.append(pattern_confidence)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
        
    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results when analysis fails"""
        
        return {
            'timestamp': time.time(),
            'data_points': 0,
            'current_phase': 'uncertain',
            'trend_strength': 0.0,
            'cycle_position': 0.5,
            'next_reversal_time': None,
            'significant_points_count': 0,
            'relationships_count': 0,
            'cycles_count': 0,
            'projections_count': 0,
            'significant_points': [],
            'relationships': [],
            'cycles': [],
            'projections': [],
            'signals': [],
            'analysis_quality': 0.0,
            'confidence': 0.0
        }
        
    def _point_to_dict(self, point: PriceTimePoint) -> Dict[str, Any]:
        """Convert PriceTimePoint to dictionary"""
        
        return {
            'timestamp': point.timestamp,
            'price': point.price,
            'point_type': point.point_type,
            'significance': point.significance,
            'cycle_phase': point.cycle_phase,
            'harmonic_level': point.harmonic_level
        }
        
    def _relationship_to_dict(self, relationship: PriceTimeRelationship) -> Dict[str, Any]:
        """Convert PriceTimeRelationship to dictionary"""
        
        return {
            'point1': self._point_to_dict(relationship.point1),
            'point2': self._point_to_dict(relationship.point2),
            'relationship_type': relationship.relationship_type.value,
            'ratio': relationship.ratio,
            'strength': relationship.strength,
            'time_span': relationship.time_span,
            'price_span': relationship.price_span,
            'harmonic_number': relationship.harmonic_number,
            'fibonacci_level': relationship.fibonacci_level
        }
        
    def _cycle_to_dict(self, cycle: CycleAnalysis) -> Dict[str, Any]:
        """Convert CycleAnalysis to dictionary"""
        
        return {
            'cycle_type': cycle.cycle_type.value,
            'period': cycle.period,
            'amplitude': cycle.amplitude,
            'phase': cycle.phase,
            'strength': cycle.strength,
            'start_time': cycle.start_time,
            'end_time': cycle.end_time,
            'turning_points_count': len(cycle.turning_points),
            'harmonic_components_count': len(cycle.harmonic_components)
        }
        
    def _projection_to_dict(self, projection: TimeProjection) -> Dict[str, Any]:
        """Convert TimeProjection to dictionary"""
        
        return {
            'target_time': projection.target_time,
            'projected_price': projection.projected_price,
            'confidence': projection.confidence,
            'projection_type': projection.projection_type,
            'has_source_relationship': projection.source_relationship is not None,
            'supporting_cycles_count': len(projection.supporting_cycles)
        }
        
    def _signal_to_dict(self, signal: GannPriceTimeSignal) -> Dict[str, Any]:
        """Convert GannPriceTimeSignal to dictionary"""
        
        return {
            'signal_type': signal.signal_type,
            'direction': signal.direction,
            'strength': signal.strength,
            'entry_time': signal.entry_time,
            'entry_price': signal.entry_price,
            'target_price': signal.target_price,
            'stop_loss': signal.stop_loss,
            'time_target': signal.time_target,
            'confidence': signal.confidence,
            'cycle_support': signal.cycle_support,
            'relationship_support': signal.relationship_support,
            'risk_reward_ratio': abs(signal.target_price - signal.entry_price) / abs(signal.entry_price - signal.stop_loss) if signal.entry_price != signal.stop_loss else 0.0
        }
        
    def get_next_time_targets(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get next time-based price targets"""
        
        targets = []
        
        # Sort projections by time
        sorted_projections = sorted(self.projections, key=lambda p: p.target_time)
        
        for projection in sorted_projections[:count]:
            target = {
                'target_time': projection.target_time,
                'projected_price': projection.projected_price,
                'confidence': projection.confidence,
                'projection_type': projection.projection_type,
                'time_until_target': projection.target_time - time.time()
            }
            targets.append(target)
            
        return targets
        
    def get_cycle_summary(self) -> Dict[str, Any]:
        """Get summary of detected cycles"""
        
        if not self.cycles:
            return {'cycle_count': 0, 'cycles': []}
            
        cycle_summary = {
            'cycle_count': len(self.cycles),
            'average_strength': np.mean([c.strength for c in self.cycles]),
            'dominant_cycle_type': max(set(c.cycle_type.value for c in self.cycles), 
                                     key=lambda x: sum(1 for c in self.cycles if c.cycle_type.value == x)),
            'cycles': [self._cycle_to_dict(c) for c in self.cycles]
        }
        
        return cycle_summary
        
    def get_relationship_summary(self) -> Dict[str, Any]:
        """Get summary of price-time relationships"""
        
        if not self.relationships:
            return {'relationship_count': 0, 'relationships': []}
            
        rel_summary = {
            'relationship_count': len(self.relationships),
            'average_strength': np.mean([r.strength for r in self.relationships]),
            'dominant_relationship_type': max(set(r.relationship_type.value for r in self.relationships),
                                            key=lambda x: sum(1 for r in self.relationships if r.relationship_type.value == x)),
            'relationships': [self._relationship_to_dict(r) for r in self.relationships[:10]]
        }
        
        return rel_summary


# Advanced Analysis Classes

class GannTimeAnalyzer:
    """Advanced time analysis component"""
    
    def __init__(self, config: GannPriceTimeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_time_cycles(self, timestamps: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze time-based cycles"""
        
        if len(timestamps) < 10:
            return []
            
        time_cycles = []
        
        # Convert timestamps to time units
        if self.config.time_unit == "days":
            time_values = timestamps / (24 * 3600)
        elif self.config.time_unit == "hours":
            time_values = timestamps / 3600
        else:
            time_values = timestamps / 60
            
        # Analyze for natural time cycles
        natural_cycles = [7, 14, 30, 90, 365]  # Week, biweekly, month, quarter, year
        
        for cycle_length in natural_cycles:
            cycle_strength = self._analyze_cycle_strength(time_values, cycle_length)
            
            if cycle_strength > 0.3:
                time_cycles.append({
                    'cycle_length': cycle_length,
                    'cycle_unit': self.config.time_unit,
                    'strength': cycle_strength,
                    'type': 'natural'
                })
                
        return time_cycles
        
    def _analyze_cycle_strength(self, time_values: np.ndarray, cycle_length: float) -> float:
        """Analyze strength of a specific cycle"""
        
        try:
            # Create idealized cycle
            ideal_cycle = np.sin(2 * np.pi * time_values / cycle_length)
            
            # Calculate correlation with price movements (if available)
            # For now, return a simple periodicity measure
            
            # Check for periodic patterns in time intervals
            intervals = np.diff(time_values)
            
            if len(intervals) < 5:
                return 0.0
                
            # Look for repeating patterns
            pattern_strength = 0.0
            
            for i in range(len(intervals) - cycle_length):
                if i + cycle_length < len(intervals):
                    current_pattern = intervals[i:i+int(cycle_length)]
                    next_pattern = intervals[i+int(cycle_length):i+2*int(cycle_length)]
                    
                    if len(current_pattern) == len(next_pattern) and len(current_pattern) > 0:
                        correlation = np.corrcoef(current_pattern, next_pattern)[0, 1]
                        if not np.isnan(correlation):
                            pattern_strength = max(pattern_strength, abs(correlation))
                            
            return pattern_strength
            
        except Exception:
            return 0.0


# Demo and Testing Functions

def demo_gann_price_time_indicator():
    """Demonstrate the Gann Price-Time Indicator with sample data"""
    
    print("=== Gann Price-Time Indicator Demo ===")
    
    # Create sample price data with embedded cycles and relationships
    np.random.seed(42)
    
    # Generate 500 days of data
    days = 500
    timestamps = np.array([time.time() - (days - i) * 24 * 3600 for i in range(days)])
    
    # Create base price with multiple cycles
    base_price = 100
    
    # Major cycle (90 days)
    major_cycle = 10 * np.sin(2 * np.pi * np.arange(days) / 90)
    
    # Medium cycle (30 days)
    medium_cycle = 5 * np.sin(2 * np.pi * np.arange(days) / 30)
    
    # Short cycle (7 days)
    short_cycle = 2 * np.sin(2 * np.pi * np.arange(days) / 7)
    
    # Trend
    trend = 0.05 * np.arange(days)
    
    # Noise
    noise = np.random.normal(0, 1, days)
    
    # Combine all components
    prices = base_price + major_cycle + medium_cycle + short_cycle + trend + noise
    
    # Create OHLCV data
    highs = prices + np.abs(np.random.normal(0, 0.5, days))
    lows = prices - np.abs(np.random.normal(0, 0.5, days))
    opens = prices + np.random.normal(0, 0.3, days)
    volumes = np.random.randint(1000, 10000, days)
    
    price_data = np.column_stack([timestamps, opens, highs, lows, prices, volumes])
    
    # Initialize indicator
    config = GannPriceTimeConfig(
        lookback_period=252,
        enable_ml=True,
        cycle_strength_threshold=0.3,
        signal_strength_threshold=0.5
    )
    
    indicator = GannPriceTimeIndicator(config)
    
    print(f"Analyzing {len(price_data)} data points...")
    
    # Perform analysis
    start_time = time.time()
    results = indicator.analyze(price_data)
    analysis_time = time.time() - start_time
    
    print(f"Analysis completed in {analysis_time:.2f} seconds")
    print(f"Analysis quality: {results['analysis_quality']:.3f}")
    print(f"Overall confidence: {results['confidence']:.3f}")
    
    # Display results
    print(f"\n=== Market Phase Assessment ===")
    print(f"Current phase: {results['current_phase']}")
    print(f"Trend strength: {results['trend_strength']:.3f}")
    print(f"Cycle position: {results['cycle_position']:.3f}")
    
    if results['next_reversal_time']:
        reversal_hours = (results['next_reversal_time'] - time.time()) / 3600
        print(f"Next reversal in: {reversal_hours:.1f} hours")
    
    print(f"\n=== Pattern Detection ===")
    print(f"Significant points: {results['significant_points_count']}")
    print(f"Price-time relationships: {results['relationships_count']}")
    print(f"Cycles detected: {results['cycles_count']}")
    print(f"Time projections: {results['projections_count']}")
    
    # Show top cycles
    if results['cycles']:
        print(f"\n=== Top Cycles ===")
        for i, cycle in enumerate(results['cycles'][:3], 1):
            print(f"{i}. {cycle['cycle_type'].title()} cycle")
            print(f"   Period: {cycle['period']:.1f} {config.time_unit}")
            print(f"   Strength: {cycle['strength']:.3f}")
            print(f"   Amplitude: {cycle['amplitude']:.2f}")
            
    # Show top relationships
    if results['relationships']:
        print(f"\n=== Top Relationships ===")
        for i, rel in enumerate(results['relationships'][:3], 1):
            print(f"{i}. {rel['relationship_type'].title()} relationship")
            print(f"   Strength: {rel['strength']:.3f}")
            print(f"   Ratio: {rel['ratio']:.3f}")
            print(f"   Time span: {rel['time_span']/3600:.1f} hours")
            
    # Show projections
    if results['projections']:
        print(f"\n=== Time Projections ===")
        for i, proj in enumerate(results['projections'][:3], 1):
            hours_ahead = (proj['target_time'] - time.time()) / 3600
            print(f"{i}. {proj['projection_type']}")
            print(f"   Target time: {hours_ahead:.1f} hours ahead")
            print(f"   Projected price: {proj['projected_price']:.2f}")
            print(f"   Confidence: {proj['confidence']:.3f}")
            
    # Show trading signals
    if results['signals']:
        print(f"\n=== Trading Signals ===")
        for i, signal in enumerate(results['signals'], 1):
            print(f"{i}. {signal['signal_type']} - {signal['direction'].upper()}")
            print(f"   Strength: {signal['strength']:.3f}")
            print(f"   Entry: {signal['entry_price']:.2f}")
            print(f"   Target: {signal['target_price']:.2f}")
            print(f"   Stop: {signal['stop_loss']:.2f}")
            print(f"   R/R: {signal['risk_reward_ratio']:.2f}")
            print(f"   Confidence: {signal['confidence']:.3f}")
            print(f"   Support: Cycle={signal['cycle_support']}, Relationship={signal['relationship_support']}")
            
    else:
        print("\nNo trading signals generated")
        
    # Additional analysis
    print(f"\n=== Additional Analysis ===")
    
    # Get cycle summary
    cycle_summary = indicator.get_cycle_summary()
    print(f"Cycle analysis: {cycle_summary['cycle_count']} cycles detected")
    if cycle_summary['cycle_count'] > 0:
        print(f"Average cycle strength: {cycle_summary['average_strength']:.3f}")
        print(f"Dominant cycle type: {cycle_summary['dominant_cycle_type']}")
        
    # Get relationship summary
    rel_summary = indicator.get_relationship_summary()
    print(f"Relationship analysis: {rel_summary['relationship_count']} relationships detected")
    if rel_summary['relationship_count'] > 0:
        print(f"Average relationship strength: {rel_summary['average_strength']:.3f}")
        print(f"Dominant relationship type: {rel_summary['dominant_relationship_type']}")
        
    # Get next time targets
    time_targets = indicator.get_next_time_targets(3)
    if time_targets:
        print(f"\n=== Next Time Targets ===")
        for i, target in enumerate(time_targets, 1):
            hours_ahead = target['time_until_target'] / 3600
            print(f"{i}. {hours_ahead:.1f}h: {target['projected_price']:.2f} ({target['confidence']:.2f})")
            
    print(f"\n=== Demo Complete ===")


def test_gann_price_time_relationships():
    """Test specific price-time relationship detection"""
    
    print("=== Testing Price-Time Relationships ===")
    
    # Create data with known relationships
    timestamps = np.array([i * 3600 for i in range(100)])  # Hourly data
    prices = np.array([100 + i * 0.1 + 2 * np.sin(i / 10) for i in range(100)])
    
    # Create price data
    price_data = np.column_stack([
        timestamps,
        prices,  # open
        prices + 0.5,  # high
        prices - 0.5,  # low
        prices,  # close
        np.ones(100) * 1000  # volume
    ])
    
    config = GannPriceTimeConfig(square_tolerance=0.1, ratio_tolerance=0.1)
    indicator = GannPriceTimeIndicator(config)
    
    results = indicator.analyze(price_data)
    
    print(f"Detected {results['relationships_count']} relationships")
    
    for rel in results['relationships']:
        print(f"- {rel['relationship_type']}: strength={rel['strength']:.3f}, ratio={rel['ratio']:.3f}")
        
    print("Test complete")


if __name__ == "__main__":
    # Run demo
    demo_gann_price_time_indicator()
    
    print("\n" + "="*50 + "\n")
    
    # Run relationship test
    test_gann_price_time_relationships()