"""
Gann Pattern Detector Implementation
Humanitarian Trading Platform - Advanced Indicator Suite

This module implements a sophisticated Gann Pattern Detector with advanced pattern recognition 
algorithms, ML-based pattern classification, predictive pattern analysis, multi-dimensional 
pattern matching, geometric pattern analysis, and comprehensive trading signal generation.

Pattern Features:
- Advanced Gann pattern recognition (squares, triangles, circles, spirals)
- ML-based pattern classification and validation
- Predictive pattern analysis and completion forecasting
- Multi-timeframe pattern synchronization
- Geometric harmony and proportion analysis
- Pattern strength assessment and reliability scoring
- Dynamic pattern evolution tracking
- Comprehensive trading signal generation

Author: AUJ Platform Development Team
Mission: To help poor families and sick children through advanced trading technology
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import scipy.stats as stats
    from scipy.signal import find_peaks, savgol_filter
    from scipy.optimize import minimize, differential_evolution
    from scipy.spatial.distance import euclidean, cosine
    from scipy.interpolate import interp1d, UnivariateSpline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GannPatternConfig:
    """Configuration for Gann Pattern Detection"""
    
    # Pattern detection parameters
    min_pattern_length: int = 10
    max_pattern_length: int = 100
    pattern_tolerance: float = 0.05
    geometric_precision: float = 0.02
    
    # Pattern types to detect
    detect_squares: bool = True
    detect_triangles: bool = True
    detect_circles: bool = True
    detect_spirals: bool = True
    detect_custom_patterns: bool = True
    
    # ML classification parameters
    ml_classification: bool = ML_AVAILABLE
    pattern_prediction: bool = True
    prediction_horizon: int = 20
    confidence_threshold: float = 0.7
    
    # Pattern validation
    min_pattern_strength: float = 0.6
    max_patterns_per_timeframe: int = 20
    pattern_overlap_tolerance: float = 0.3
    
    # Geometric analysis
    golden_ratio_analysis: bool = True
    harmonic_proportion_analysis: bool = True
    fractal_dimension_analysis: bool = True
    
    # Signal generation
    signal_threshold: float = 0.65
    pattern_completion_signals: bool = True
    breakout_signals: bool = True
    reversal_signals: bool = True
    
    # Performance optimization
    max_lookback_periods: int = 1000
    calculation_precision: int = 6
    memory_optimization: bool = True


@dataclass
class GannPatternPoint:
    """Represents a point in a Gann pattern"""
    
    time: float
    price: float
    point_type: str  # high, low, pivot, intersection
    significance: float = 0.0
    
    # Geometric properties
    angle_from_origin: float = 0.0
    distance_from_origin: float = 0.0
    harmonic_ratio: float = 0.0
    
    # Market context
    volume: float = 0.0
    volatility: float = 0.0
    trend_strength: float = 0.0


@dataclass
class GannPattern:
    """Represents a detected Gann pattern"""
    
    id: str
    pattern_type: str  # square, triangle, circle, spiral, custom
    pattern_subtype: str  # equilateral, isosceles, right, fibonacci_spiral, etc.
    
    # Pattern geometry
    points: List[GannPatternPoint] = field(default_factory=list)
    center_point: Optional[GannPatternPoint] = None
    bounding_box: Dict[str, float] = field(default_factory=dict)
    
    # Pattern properties
    completion_percentage: float = 0.0
    geometric_accuracy: float = 0.0
    harmonic_resonance: float = 0.0
    pattern_strength: float = 0.0
    
    # Time and price dimensions
    time_span: float = 0.0
    price_span: float = 0.0
    aspect_ratio: float = 0.0
    
    # ML classification
    ml_confidence: float = 0.0
    predicted_completion: Optional[GannPatternPoint] = None
    pattern_evolution: List[float] = field(default_factory=list)
    
    # Market interaction
    support_resistance_levels: List[float] = field(default_factory=list)
    breakout_levels: List[float] = field(default_factory=list)
    price_touches: int = 0
    
    # Pattern validation
    statistical_significance: float = 0.0
    fractal_dimension: float = 0.0
    golden_ratio_alignment: float = 0.0


class GannPatternDetector:
    """
    Advanced Gann Pattern Detector with sophisticated pattern recognition algorithms,
    ML-based pattern classification, and predictive pattern analysis.
    """
    
    def __init__(self, config: Optional[GannPatternConfig] = None):
        self.config = config or GannPatternConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models if available
        self.ml_models = {}
        self.scalers = {}
        self.pattern_templates = {}
        
        if self.config.ml_classification and ML_AVAILABLE:
            self._initialize_ml_models()
            
        # Initialize pattern templates
        self._initialize_pattern_templates()
        
        # Pattern detection state
        self.detected_patterns: List[GannPattern] = []
        self.pattern_history: List[GannPattern] = []
        self.performance_metrics = {}
        
        self.logger.info("Advanced Gann Pattern Detector initialized")
        
    def _initialize_ml_models(self):
        """Initialize machine learning models for pattern classification"""
        
        try:
            # Pattern classifier
            self.ml_models['pattern_classifier'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42
            )
            
            # Pattern strength predictor
            self.ml_models['strength_predictor'] = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42
            )
            
            # Completion predictor
            if self.config.pattern_prediction:
                self.ml_models['completion_predictor'] = RandomForestClassifier(
                    n_estimators=75,
                    max_depth=12,
                    random_state=42
                )
            
            # Anomaly detection for pattern validation
            self.ml_models['anomaly_detector'] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Feature scalers
            self.scalers['standard'] = StandardScaler()
            self.scalers['minmax'] = MinMaxScaler()
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"ML model initialization failed: {e}")
            self.config.ml_classification = False
            
    def _initialize_pattern_templates(self):
        """Initialize geometric pattern templates for matching"""
        
        # Square pattern template
        self.pattern_templates['square'] = {
            'points': 4,
            'angles': [90, 90, 90, 90],
            'side_ratios': [1.0, 1.0, 1.0, 1.0],
            'tolerance': self.config.geometric_precision
        }
        
        # Triangle pattern templates
        self.pattern_templates['equilateral_triangle'] = {
            'points': 3,
            'angles': [60, 60, 60],
            'side_ratios': [1.0, 1.0, 1.0],
            'tolerance': self.config.geometric_precision
        }
        
        self.pattern_templates['right_triangle'] = {
            'points': 3,
            'angles': [90, 45, 45],
            'side_ratios': [1.0, 1.414, 1.0],  # Pythagorean ratios
            'tolerance': self.config.geometric_precision * 2
        }
        
        # Golden ratio triangle
        self.pattern_templates['golden_triangle'] = {
            'points': 3,
            'angles': [36, 72, 72],
            'side_ratios': [1.0, 1.618, 1.618],  # Golden ratio
            'tolerance': self.config.geometric_precision
        }
        
        # Circle pattern template
        self.pattern_templates['circle'] = {
            'points': 8,  # Octagon approximation
            'radius_consistency': 0.95,  # Minimum radius consistency
            'tolerance': self.config.geometric_precision * 3
        }
        
        # Spiral pattern templates
        self.pattern_templates['fibonacci_spiral'] = {
            'points': 12,
            'growth_ratio': 1.618,  # Golden ratio
            'angle_increment': 137.5,  # Golden angle
            'tolerance': self.config.geometric_precision * 2
        }
        
        self.pattern_templates['logarithmic_spiral'] = {
            'points': 10,
            'growth_factor': 1.5,
            'tolerance': self.config.geometric_precision * 2
        }
        
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Gann Pattern detection and analysis
        
        Args:
            df: DataFrame with OHLCV data and timestamp
            
        Returns:
            DataFrame with Gann Pattern analysis results
        """
        
        try:
            # Validate input data
            df = self._validate_input_data(df)
            
            # Prepare data for pattern detection
            prepared_data = self._prepare_pattern_data(df)
            
            # Detect significant points for pattern formation
            significant_points = self._detect_significant_points(prepared_data)
            
            # Detect geometric patterns
            detected_patterns = self._detect_geometric_patterns(prepared_data, significant_points)
            
            # Classify patterns using ML if enabled
            if self.config.ml_classification:
                classified_patterns = self._ml_classify_patterns(prepared_data, detected_patterns)
            else:
                classified_patterns = detected_patterns
                
            # Analyze pattern completion and prediction
            completion_analysis = self._analyze_pattern_completion(prepared_data, classified_patterns)
            
            # Validate patterns with geometric and statistical tests
            validated_patterns = self._validate_patterns(prepared_data, classified_patterns)
            
            # Analyze pattern harmonics and proportions
            harmonic_analysis = self._analyze_pattern_harmonics(prepared_data, validated_patterns)
            
            # Track pattern evolution
            evolution_analysis = self._track_pattern_evolution(prepared_data, validated_patterns)
            
            # Generate trading signals from patterns
            signals = self._generate_pattern_signals(prepared_data, validated_patterns, completion_analysis)
            
            # Combine all results
            result = self._combine_pattern_results(
                df, validated_patterns, completion_analysis, harmonic_analysis,
                evolution_analysis, signals
            )
            
            # Update state
            self.detected_patterns = validated_patterns
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Gann Pattern detection: {e}")
            return self._create_error_result(df)
            
    def _validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare input data"""
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Ensure timestamp is available
        if 'timestamp' not in df.columns:
            if df.index.name == 'timestamp' or 'datetime' in str(type(df.index[0])):
                df = df.reset_index()
                df['timestamp'] = df['index'] if 'index' in df.columns else df.index
            else:
                df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='1H')
                
        # Convert timestamp to numeric for calculations
        if not pd.api.types.is_numeric_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['time_numeric'] = df['timestamp'].astype('int64') // 10**9
        else:
            df['time_numeric'] = df['timestamp']
            
        # Calculate technical indicators needed for pattern detection
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_range'] = df['high'] - df['low']
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        # Volatility and trend measures
        df['price_volatility'] = df['close'].pct_change().rolling(window=20).std()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['trend_strength'] = self._calculate_trend_strength(df)
        
        return df.dropna()
        
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength for pattern context"""
        
        # Multiple moving averages for trend analysis
        ma_short = df['close'].rolling(window=10).mean()
        ma_medium = df['close'].rolling(window=20).mean()
        ma_long = df['close'].rolling(window=50).mean()
        
        # Trend alignment
        bullish_alignment = (ma_short > ma_medium) & (ma_medium > ma_long)
        bearish_alignment = (ma_short < ma_medium) & (ma_medium < ma_long)
        
        # Price momentum
        momentum = df['close'].pct_change(10)
        
        # Combine trend indicators
        trend_strength = np.where(
            bullish_alignment, 0.5 + momentum.clip(0, 0.5),
            np.where(bearish_alignment, -0.5 + momentum.clip(-0.5, 0), momentum)
        )
        
        return pd.Series(trend_strength, index=df.index).fillna(0)
        
    def _prepare_pattern_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data specifically for pattern detection"""
        
        data = df.copy()
        
        # Calculate pivot points for pattern anchors
        data['pivot_high'] = data['high'].rolling(window=5, center=True).max() == data['high']
        data['pivot_low'] = data['low'].rolling(window=5, center=True).min() == data['low']
        
        # Calculate pattern-relevant indicators
        data['price_acceleration'] = data['close'].diff().diff()
        data['volume_acceleration'] = data['volume'].diff().diff()
        
        # Geometric indicators
        data['price_to_time_ratio'] = data['close'] / data['time_numeric'] * 1e6  # Normalize
        
        # Pattern formation indicators
        data['local_volatility'] = data['price_range'].rolling(window=10).std()
        data['volume_profile'] = data['volume'] / data['volume_ma']
        
        # Harmonic indicators
        if len(data) >= 50:
            data['harmonic_component'] = self._calculate_harmonic_component(data['close'])
        else:
            data['harmonic_component'] = 0.0
            
        return data
        
    def _calculate_harmonic_component(self, price_series: pd.Series) -> pd.Series:
        """Calculate harmonic component for pattern analysis"""
        
        if not SCIPY_AVAILABLE or len(price_series) < 50:
            return pd.Series(0.0, index=price_series.index)
            
        try:
            # Detrend the price series
            x = np.arange(len(price_series))
            detrended = price_series - np.polyval(np.polyfit(x, price_series, 1), x)
            
            # FFT analysis
            fft = np.fft.fft(detrended.values)
            frequencies = np.fft.fftfreq(len(detrended))
            
            # Find dominant harmonics
            magnitude = np.abs(fft)
            dominant_indices = np.argsort(magnitude)[-5:]  # Top 5 harmonics
            
            # Reconstruct signal from dominant harmonics
            reconstructed = np.zeros(len(detrended))
            for idx in dominant_indices:
                if idx > 0:  # Skip DC component
                    harmonic = magnitude[idx] * np.cos(2 * np.pi * frequencies[idx] * x + np.angle(fft[idx]))
                    reconstructed += harmonic
                    
            return pd.Series(reconstructed, index=price_series.index)
            
        except Exception as e:
            self.logger.warning(f"Harmonic component calculation failed: {e}")
            return pd.Series(0.0, index=price_series.index)
            
    def _detect_significant_points(self, df: pd.DataFrame) -> List[GannPatternPoint]:
        """Detect significant points for pattern formation"""
        
        significant_points = []
        
        # Pivot highs and lows
        pivot_highs = df[df['pivot_high']].copy()
        pivot_lows = df[df['pivot_low']].copy()
        
        # Process pivot highs
        for idx, row in pivot_highs.iterrows():
            point = GannPatternPoint(
                time=row['time_numeric'],
                price=row['high'],
                point_type='high',
                significance=self._calculate_point_significance(df, idx, 'high'),
                volume=row['volume'],
                volatility=row['price_volatility'],
                trend_strength=row['trend_strength']
            )
            significant_points.append(point)
            
        # Process pivot lows
        for idx, row in pivot_lows.iterrows():
            point = GannPatternPoint(
                time=row['time_numeric'],
                price=row['low'],
                point_type='low',
                significance=self._calculate_point_significance(df, idx, 'low'),
                volume=row['volume'],
                volatility=row['price_volatility'],
                trend_strength=row['trend_strength']
            )
            significant_points.append(point)
            
        # Add intersection points (simplified)
        intersection_points = self._find_intersection_points(df)
        significant_points.extend(intersection_points)
        
        # Sort by time
        significant_points.sort(key=lambda x: x.time)
        
        # Filter by significance
        min_significance = 0.3
        filtered_points = [p for p in significant_points if p.significance > min_significance]
        
        self.logger.info(f"Detected {len(filtered_points)} significant points")
        return filtered_points
        
    def _calculate_point_significance(self, df: pd.DataFrame, idx: int, point_type: str) -> float:
        """Calculate significance of a pivot point"""
        
        significance = 0.0
        
        if idx not in df.index:
            return significance
            
        row = df.loc[idx]
        
        # Volume significance
        volume_ratio = row['volume'] / row['volume_ma'] if row['volume_ma'] > 0 else 1.0
        volume_significance = min(volume_ratio / 2.0, 1.0)
        significance += volume_significance * 0.3
        
        # Price movement significance
        if point_type == 'high':
            price_movement = (row['high'] - row['low']) / row['atr'] if row['atr'] > 0 else 0
        else:
            price_movement = (row['high'] - row['low']) / row['atr'] if row['atr'] > 0 else 0
            
        movement_significance = min(price_movement / 3.0, 1.0)
        significance += movement_significance * 0.4
        
        # Trend context significance
        trend_significance = abs(row['trend_strength'])
        significance += trend_significance * 0.3
        
        return min(significance, 1.0)
        
    def _find_intersection_points(self, df: pd.DataFrame) -> List[GannPatternPoint]:
        """Find intersection points of trend lines and support/resistance"""
        
        intersection_points = []
        
        # Simplified intersection detection
        # In a full implementation, this would detect actual line intersections
        
        # Find points where price crosses moving averages with high volume
        ma20 = df['close'].rolling(window=20).mean()
        
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Check for MA crossover with high volume
            if (prev_row['close'] < ma20.iloc[i-1] < current_row['close'] or
                prev_row['close'] > ma20.iloc[i-1] > current_row['close']):
                
                if current_row['volume'] > current_row['volume_ma'] * 1.5:
                    point = GannPatternPoint(
                        time=current_row['time_numeric'],
                        price=ma20.iloc[i],
                        point_type='intersection',
                        significance=0.6,
                        volume=current_row['volume'],
                        volatility=current_row['price_volatility'],
                        trend_strength=current_row['trend_strength']
                    )
                    intersection_points.append(point)
                    
        return intersection_points
        
    def _detect_geometric_patterns(self, df: pd.DataFrame, 
                                 points: List[GannPatternPoint]) -> List[GannPattern]:
        """Detect geometric patterns from significant points"""
        
        patterns = []
        
        if len(points) < 3:
            return patterns
            
        # Detect different pattern types
        if self.config.detect_squares:
            square_patterns = self._detect_square_patterns(points)
            patterns.extend(square_patterns)
            
        if self.config.detect_triangles:
            triangle_patterns = self._detect_triangle_patterns(points)
            patterns.extend(triangle_patterns)
            
        if self.config.detect_circles:
            circle_patterns = self._detect_circle_patterns(points)
            patterns.extend(circle_patterns)
            
        if self.config.detect_spirals:
            spiral_patterns = self._detect_spiral_patterns(points)
            patterns.extend(spiral_patterns)
            
        if self.config.detect_custom_patterns:
            custom_patterns = self._detect_custom_patterns(points)
            patterns.extend(custom_patterns)
            
        # Calculate geometric properties for all patterns
        for pattern in patterns:
            pattern.geometric_accuracy = self._calculate_geometric_accuracy(pattern)
            pattern.harmonic_resonance = self._calculate_pattern_harmonic_resonance(pattern)
            pattern.pattern_strength = self._calculate_pattern_strength(df, pattern)
            
        self.logger.info(f"Detected {len(patterns)} geometric patterns")
        return patterns        
    def _detect_square_patterns(self, points: List[GannPatternPoint]) -> List[GannPattern]:
        """Detect square patterns from significant points"""
        
        patterns = []
        template = self.pattern_templates['square']
        
        # Try all combinations of 4 points
        for i in range(len(points) - 3):
            for j in range(i + 1, len(points) - 2):
                for k in range(j + 1, len(points) - 1):
                    for l in range(k + 1, len(points)):
                        
                        four_points = [points[i], points[j], points[k], points[l]]
                        square_pattern = self._test_square_pattern(four_points, template)
                        
                        if square_pattern:
                            patterns.append(square_pattern)
                            
        return patterns
        
    def _test_square_pattern(self, four_points: List[GannPatternPoint], 
                           template: Dict) -> Optional[GannPattern]:
        """Test if four points form a square pattern"""
        
        # Calculate all distances between points
        distances = []
        for i in range(len(four_points)):
            for j in range(i + 1, len(four_points)):
                p1, p2 = four_points[i], four_points[j]
                distance = np.sqrt((p1.time - p2.time)**2 + (p1.price - p2.price)**2)
                distances.append(distance)
                
        distances.sort()
        
        # For a square, we should have 4 equal sides and 2 equal diagonals
        # distances[0:4] should be sides, distances[4:6] should be diagonals
        
        if len(distances) != 6:
            return None
            
        # Check side equality
        side_tolerance = template['tolerance']
        sides = distances[:4]
        diagonals = distances[4:]
        
        side_mean = np.mean(sides)
        diagonal_mean = np.mean(diagonals)
        
        # Sides should be equal
        side_variation = max(abs(s - side_mean) for s in sides) / side_mean if side_mean > 0 else 1.0
        
        # Diagonals should be equal and sqrt(2) * side length
        diagonal_variation = max(abs(d - diagonal_mean) for d in diagonals) / diagonal_mean if diagonal_mean > 0 else 1.0
        expected_diagonal = side_mean * np.sqrt(2)
        diagonal_ratio_error = abs(diagonal_mean - expected_diagonal) / expected_diagonal if expected_diagonal > 0 else 1.0
        
        if (side_variation < side_tolerance and 
            diagonal_variation < side_tolerance and 
            diagonal_ratio_error < side_tolerance * 2):
            
            # Create square pattern
            pattern = GannPattern(
                id=f"square_{four_points[0].time:.0f}",
                pattern_type="square",
                pattern_subtype="regular",
                points=four_points
            )
            
            # Calculate pattern properties
            pattern.time_span = max(p.time for p in four_points) - min(p.time for p in four_points)
            pattern.price_span = max(p.price for p in four_points) - min(p.price for p in four_points)
            pattern.aspect_ratio = pattern.price_span / pattern.time_span if pattern.time_span > 0 else 1.0
            
            # Calculate center point
            center_time = np.mean([p.time for p in four_points])
            center_price = np.mean([p.price for p in four_points])
            pattern.center_point = GannPatternPoint(center_time, center_price, "center")
            
            return pattern
            
        return None
        
    def _detect_triangle_patterns(self, points: List[GannPatternPoint]) -> List[GannPattern]:
        """Detect triangle patterns from significant points"""
        
        patterns = []
        
        # Test different triangle types
        triangle_types = ['equilateral_triangle', 'right_triangle', 'golden_triangle']
        
        for triangle_type in triangle_types:
            if triangle_type not in self.pattern_templates:
                continue
                
            template = self.pattern_templates[triangle_type]
            
            # Try all combinations of 3 points
            for i in range(len(points) - 2):
                for j in range(i + 1, len(points) - 1):
                    for k in range(j + 1, len(points)):
                        
                        three_points = [points[i], points[j], points[k]]
                        triangle_pattern = self._test_triangle_pattern(three_points, template, triangle_type)
                        
                        if triangle_pattern:
                            patterns.append(triangle_pattern)
                            
        return patterns
        
    def _test_triangle_pattern(self, three_points: List[GannPatternPoint], 
                             template: Dict, triangle_type: str) -> Optional[GannPattern]:
        """Test if three points form a specific triangle pattern"""
        
        p1, p2, p3 = three_points
        
        # Calculate side lengths
        side1 = np.sqrt((p1.time - p2.time)**2 + (p1.price - p2.price)**2)
        side2 = np.sqrt((p2.time - p3.time)**2 + (p2.price - p3.price)**2)
        side3 = np.sqrt((p3.time - p1.time)**2 + (p3.price - p1.price)**2)
        
        sides = [side1, side2, side3]
        sides.sort()
        
        # Calculate angles using law of cosines
        angles = []
        for i in range(3):
            a, b, c = sides[i], sides[(i+1)%3], sides[(i+2)%3]
            if a > 0 and b > 0:
                cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
                cos_angle = np.clip(cos_angle, -1, 1)  # Ensure valid range
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
                
        if len(angles) != 3:
            return None
            
        angles.sort()
        
        # Test against template
        tolerance = template['tolerance'] * 100  # Convert to degrees
        expected_angles = sorted(template['angles'])
        expected_ratios = sorted(template['side_ratios'])
        
        # Check angle accuracy
        angle_errors = [abs(a - e) for a, e in zip(angles, expected_angles)]
        max_angle_error = max(angle_errors)
        
        # Check side ratio accuracy
        if sides[0] > 0:
            actual_ratios = [s / sides[0] for s in sides]
            actual_ratios.sort()
            ratio_errors = [abs(a - e) for a, e in zip(actual_ratios, expected_ratios)]
            max_ratio_error = max(ratio_errors)
        else:
            max_ratio_error = 1.0
            
        if max_angle_error < tolerance and max_ratio_error < template['tolerance']:
            
            # Create triangle pattern
            pattern = GannPattern(
                id=f"triangle_{triangle_type}_{three_points[0].time:.0f}",
                pattern_type="triangle",
                pattern_subtype=triangle_type.replace('_triangle', ''),
                points=three_points
            )
            
            # Calculate pattern properties
            pattern.time_span = max(p.time for p in three_points) - min(p.time for p in three_points)
            pattern.price_span = max(p.price for p in three_points) - min(p.price for p in three_points)
            pattern.aspect_ratio = pattern.price_span / pattern.time_span if pattern.time_span > 0 else 1.0
            
            # Calculate center point (centroid)
            center_time = np.mean([p.time for p in three_points])
            center_price = np.mean([p.price for p in three_points])
            pattern.center_point = GannPatternPoint(center_time, center_price, "center")
            
            return pattern
            
        return None
        
    def _detect_circle_patterns(self, points: List[GannPatternPoint]) -> List[GannPattern]:
        """Detect circular patterns from significant points"""
        
        patterns = []
        template = self.pattern_templates['circle']
        min_points = template['points']
        
        if len(points) < min_points:
            return patterns
            
        # Try different center points and radii
        for center_idx in range(len(points) - min_points + 1):
            center_point = points[center_idx]
            
            # Find points that could form a circle around this center
            circle_candidates = []
            
            for point in points:
                if point != center_point:
                    distance = np.sqrt((point.time - center_point.time)**2 + 
                                     (point.price - center_point.price)**2)
                    circle_candidates.append((point, distance))
                    
            if len(circle_candidates) < min_points - 1:
                continue
                
            # Sort by distance from center
            circle_candidates.sort(key=lambda x: x[1])
            
            # Test if points form a circle
            circle_pattern = self._test_circle_pattern(center_point, circle_candidates, template)
            if circle_pattern:
                patterns.append(circle_pattern)
                
        return patterns
        
    def _test_circle_pattern(self, center: GannPatternPoint, 
                           candidates: List[Tuple[GannPatternPoint, float]], 
                           template: Dict) -> Optional[GannPattern]:
        """Test if points form a circular pattern"""
        
        if len(candidates) < template['points'] - 1:
            return None
            
        # Take the closest points that could form a circle
        circle_points = [center] + [c[0] for c in candidates[:template['points']-1]]
        distances = [c[1] for c in candidates[:template['points']-1]]
        
        # Check radius consistency
        mean_radius = np.mean(distances)
        radius_variations = [abs(d - mean_radius) / mean_radius for d in distances if mean_radius > 0]
        
        if not radius_variations:
            return None
            
        max_variation = max(radius_variations)
        required_consistency = template['radius_consistency']
        
        if max_variation > (1.0 - required_consistency):
            return None
            
        # Check angular distribution (points should be roughly evenly distributed)
        angles = []
        for point in circle_points[1:]:  # Exclude center
            angle = np.arctan2(point.price - center.price, point.time - center.time)
            angles.append(angle)
            
        angles.sort()
        
        # Check angular spacing consistency
        expected_spacing = 2 * np.pi / len(angles)
        actual_spacings = []
        
        for i in range(len(angles)):
            spacing = angles[(i+1) % len(angles)] - angles[i]
            if spacing < 0:
                spacing += 2 * np.pi
            actual_spacings.append(spacing)
            
        spacing_variations = [abs(s - expected_spacing) / expected_spacing 
                            for s in actual_spacings if expected_spacing > 0]
        
        if spacing_variations and max(spacing_variations) < template['tolerance']:
            
            # Create circle pattern
            pattern = GannPattern(
                id=f"circle_{center.time:.0f}",
                pattern_type="circle",
                pattern_subtype="regular",
                points=circle_points,
                center_point=center
            )
            
            # Calculate pattern properties
            pattern.time_span = 2 * max(abs(p.time - center.time) for p in circle_points[1:])
            pattern.price_span = 2 * max(abs(p.price - center.price) for p in circle_points[1:])
            pattern.aspect_ratio = pattern.price_span / pattern.time_span if pattern.time_span > 0 else 1.0
            
            return pattern
            
        return None
        
    def _detect_spiral_patterns(self, points: List[GannPatternPoint]) -> List[GannPattern]:
        """Detect spiral patterns from significant points"""
        
        patterns = []
        
        # Test different spiral types
        spiral_types = ['fibonacci_spiral', 'logarithmic_spiral']
        
        for spiral_type in spiral_types:
            if spiral_type not in self.pattern_templates:
                continue
                
            template = self.pattern_templates[spiral_type]
            min_points = template['points']
            
            if len(points) < min_points:
                continue
                
            # Try different starting points for spiral
            for start_idx in range(len(points) - min_points + 1):
                spiral_candidates = points[start_idx:start_idx + min_points]
                spiral_pattern = self._test_spiral_pattern(spiral_candidates, template, spiral_type)
                
                if spiral_pattern:
                    patterns.append(spiral_pattern)
                    
        return patterns
        
    def _test_spiral_pattern(self, candidates: List[GannPatternPoint], 
                           template: Dict, spiral_type: str) -> Optional[GannPattern]:
        """Test if points form a spiral pattern"""
        
        if len(candidates) < template['points']:
            return None
            
        # Calculate center of spiral (first point or geometric center)
        center_time = candidates[0].time
        center_price = candidates[0].price
        
        # Calculate distances and angles from center
        distances = []
        angles = []
        
        for point in candidates[1:]:
            distance = np.sqrt((point.time - center_time)**2 + (point.price - center_price)**2)
            angle = np.arctan2(point.price - center_price, point.time - center_time)
            distances.append(distance)
            angles.append(angle)
            
        if len(distances) < 2:
            return None
            
        # Test spiral properties based on type
        if spiral_type == 'fibonacci_spiral':
            growth_ratio = template['growth_ratio']
            
            # Check if distances follow golden ratio growth
            ratio_errors = []
            for i in range(len(distances) - 1):
                if distances[i] > 0:
                    actual_ratio = distances[i + 1] / distances[i]
                    ratio_error = abs(actual_ratio - growth_ratio) / growth_ratio
                    ratio_errors.append(ratio_error)
                    
            if ratio_errors and max(ratio_errors) < template['tolerance']:
                # Check golden angle progression
                angle_increment = np.radians(template['angle_increment'])
                angle_errors = []
                
                for i in range(len(angles) - 1):
                    expected_angle = angles[0] + i * angle_increment
                    actual_angle = angles[i]
                    
                    # Normalize angles
                    angle_diff = abs(actual_angle - expected_angle)
                    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                    
                    angle_errors.append(angle_diff)
                    
                if angle_errors and max(angle_errors) < np.radians(template['tolerance'] * 180):
                    
                    # Create fibonacci spiral pattern
                    pattern = GannPattern(
                        id=f"fibonacci_spiral_{candidates[0].time:.0f}",
                        pattern_type="spiral",
                        pattern_subtype="fibonacci",
                        points=candidates
                    )
                    
                    # Calculate pattern properties
                    pattern.time_span = max(p.time for p in candidates) - min(p.time for p in candidates)
                    pattern.price_span = max(p.price for p in candidates) - min(p.price for p in candidates)
                    pattern.aspect_ratio = pattern.price_span / pattern.time_span if pattern.time_span > 0 else 1.0
                    pattern.center_point = GannPatternPoint(center_time, center_price, "center")
                    
                    return pattern
                    
        elif spiral_type == 'logarithmic_spiral':
            growth_factor = template['growth_factor']
            
            # For logarithmic spiral: r = a * e^(b*θ)
            # Check if log(distance) is linear with angle
            if len(distances) >= 3 and all(d > 0 for d in distances):
                log_distances = [np.log(d) for d in distances]
                
                # Linear regression to check log-linear relationship
                if SCIPY_AVAILABLE:
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(angles[1:], log_distances)
                        
                        # Check goodness of fit
                        if r_value**2 > 0.8:  # Good linear correlation
                            
                            # Create logarithmic spiral pattern
                            pattern = GannPattern(
                                id=f"log_spiral_{candidates[0].time:.0f}",
                                pattern_type="spiral",
                                pattern_subtype="logarithmic",
                                points=candidates
                            )
                            
                            # Calculate pattern properties
                            pattern.time_span = max(p.time for p in candidates) - min(p.time for p in candidates)
                            pattern.price_span = max(p.price for p in candidates) - min(p.price for p in candidates)
                            pattern.aspect_ratio = pattern.price_span / pattern.time_span if pattern.time_span > 0 else 1.0
                            pattern.center_point = GannPatternPoint(center_time, center_price, "center")
                            
                            return pattern
                            
                    except Exception as e:
                        self.logger.debug(f"Logarithmic spiral test failed: {e}")
                        
        return None
        
    def _detect_custom_patterns(self, points: List[GannPatternPoint]) -> List[GannPattern]:
        """Detect custom Gann patterns"""
        
        patterns = []
        
        # Detect W.D. Gann specific patterns
        
        # 1. Gann Fan convergence patterns
        fan_patterns = self._detect_fan_convergence_patterns(points)
        patterns.extend(fan_patterns)
        
        # 2. Time-price square patterns
        square_patterns = self._detect_time_price_square_patterns(points)
        patterns.extend(square_patterns)
        
        # 3. Harmonic division patterns
        harmonic_patterns = self._detect_harmonic_division_patterns(points)
        patterns.extend(harmonic_patterns)
        
        return patterns
        
    def _detect_fan_convergence_patterns(self, points: List[GannPatternPoint]) -> List[GannPattern]:
        """Detect Gann fan convergence patterns"""
        
        patterns = []
        
        if len(points) < 4:
            return patterns
            
        # Look for points that form converging trend lines
        for i in range(len(points) - 3):
            for j in range(i + 2, len(points) - 1):
                
                # Two trend lines: points[i] to points[j] and points[i+1] to points[j+1]
                if j + 1 < len(points):
                    
                    line1_start, line1_end = points[i], points[j]
                    line2_start, line2_end = points[i + 1], points[j + 1]
                    
                    # Calculate convergence point
                    convergence = self._calculate_line_convergence(
                        line1_start, line1_end, line2_start, line2_end
                    )
                    
                    if convergence:
                        # Create fan convergence pattern
                        pattern_points = [line1_start, line1_end, line2_start, line2_end, convergence]
                        
                        pattern = GannPattern(
                            id=f"fan_convergence_{line1_start.time:.0f}",
                            pattern_type="custom",
                            pattern_subtype="fan_convergence",
                            points=pattern_points
                        )
                        
                        # Calculate pattern properties
                        pattern.time_span = max(p.time for p in pattern_points) - min(p.time for p in pattern_points)
                        pattern.price_span = max(p.price for p in pattern_points) - min(p.price for p in pattern_points)
                        pattern.aspect_ratio = pattern.price_span / pattern.time_span if pattern.time_span > 0 else 1.0
                        pattern.center_point = convergence
                        
                        patterns.append(pattern)
                        
        return patterns
        
    def _calculate_line_convergence(self, p1: GannPatternPoint, p2: GannPatternPoint,
                                  p3: GannPatternPoint, p4: GannPatternPoint) -> Optional[GannPatternPoint]:
        """Calculate convergence point of two lines"""
        
        # Line 1: p1 to p2, Line 2: p3 to p4
        
        # Calculate slopes
        if p2.time == p1.time or p4.time == p3.time:
            return None  # Vertical lines
            
        slope1 = (p2.price - p1.price) / (p2.time - p1.time)
        slope2 = (p4.price - p3.price) / (p4.time - p3.time)
        
        # Check if lines are parallel
        if abs(slope1 - slope2) < 1e-10:
            return None
            
        # Calculate intersection
        # Line 1: y - p1.price = slope1 * (x - p1.time)
        # Line 2: y - p3.price = slope2 * (x - p3.time)
        
        # Solve for x (time)
        x = (p3.price - p1.price + slope1 * p1.time - slope2 * p3.time) / (slope1 - slope2)
        y = p1.price + slope1 * (x - p1.time)
        
        convergence_point = GannPatternPoint(
            time=x,
            price=y,
            point_type="convergence",
            significance=0.7
        )
        
        return convergence_point
        
    def _detect_time_price_square_patterns(self, points: List[GannPatternPoint]) -> List[GannPattern]:
        """Detect time-price square patterns"""
        
        patterns = []
        
        # Look for points that form time-price squares (where time span equals price span)
        for i in range(len(points) - 1):
            for j in range(i + 1, len(points)):
                
                p1, p2 = points[i], points[j]
                
                time_span = abs(p2.time - p1.time)
                price_span = abs(p2.price - p1.price)
                
                # Normalize for comparison (simplified)
                normalized_time = time_span / 86400  # Convert to days
                normalized_price = price_span / p1.price if p1.price > 0 else 0  # Percentage
                
                # Check if time and price spans are proportional
                if normalized_time > 0 and normalized_price > 0:
                    ratio = normalized_price / normalized_time
                    
                    # Golden ratio or simple integer ratios
                    target_ratios = [1.0, 1.618, 0.618, 2.0, 0.5]
                    
                    for target_ratio in target_ratios:
                        if abs(ratio - target_ratio) / target_ratio < self.config.geometric_precision:
                            
                            # Create time-price square pattern
                            pattern = GannPattern(
                                id=f"time_price_square_{p1.time:.0f}",
                                pattern_type="custom",
                                pattern_subtype="time_price_square",
                                points=[p1, p2]
                            )
                            
                            pattern.time_span = time_span
                            pattern.price_span = price_span
                            pattern.aspect_ratio = ratio
                            
                            center_time = (p1.time + p2.time) / 2
                            center_price = (p1.price + p2.price) / 2
                            pattern.center_point = GannPatternPoint(center_time, center_price, "center")
                            
                            patterns.append(pattern)
                            break
                            
        return patterns
        
    def _detect_harmonic_division_patterns(self, points: List[GannPatternPoint]) -> List[GannPattern]:
        """Detect harmonic division patterns"""
        
        patterns = []
        
        # Look for points that divide time or price spans in harmonic ratios
        if len(points) < 3:
            return patterns
            
        harmonic_ratios = [0.382, 0.5, 0.618, 0.786]  # Common Fibonacci/harmonic ratios
        
        for i in range(len(points) - 2):
            for j in range(i + 2, len(points)):
                
                # Points i and j define the span, look for points in between
                span_start, span_end = points[i], points[j]
                
                for k in range(i + 1, j):
                    division_point = points[k]
                    
                    # Check time division
                    total_time = span_end.time - span_start.time
                    division_time = division_point.time - span_start.time
                    
                    if total_time > 0:
                        time_ratio = division_time / total_time
                        
                        for harmonic_ratio in harmonic_ratios:
                            if abs(time_ratio - harmonic_ratio) < self.config.geometric_precision:
                                
                                # Check if price also follows harmonic division
                                total_price = abs(span_end.price - span_start.price)
                                division_price = abs(division_point.price - span_start.price)
                                
                                if total_price > 0:
                                    price_ratio = division_price / total_price
                                    
                                    if abs(price_ratio - harmonic_ratio) < self.config.geometric_precision * 2:
                                        
                                        # Create harmonic division pattern
                                        pattern = GannPattern(
                                            id=f"harmonic_division_{span_start.time:.0f}",
                                            pattern_type="custom",
                                            pattern_subtype="harmonic_division",
                                            points=[span_start, division_point, span_end]
                                        )
                                        
                                        pattern.time_span = total_time
                                        pattern.price_span = total_price
                                        pattern.aspect_ratio = price_ratio / time_ratio if time_ratio > 0 else 1.0
                                        pattern.center_point = division_point
                                        
                                        patterns.append(pattern)
                                        
        return patterns
        
    def _calculate_geometric_accuracy(self, pattern: GannPattern) -> float:
        """Calculate geometric accuracy of a pattern"""
        
        if not pattern.points or len(pattern.points) < 3:
            return 0.0
            
        accuracy = 0.0
        
        if pattern.pattern_type == "square":
            accuracy = self._calculate_square_accuracy(pattern)
        elif pattern.pattern_type == "triangle":
            accuracy = self._calculate_triangle_accuracy(pattern)
        elif pattern.pattern_type == "circle":
            accuracy = self._calculate_circle_accuracy(pattern)
        elif pattern.pattern_type == "spiral":
            accuracy = self._calculate_spiral_accuracy(pattern)
        elif pattern.pattern_type == "custom":
            accuracy = self._calculate_custom_pattern_accuracy(pattern)
        else:
            accuracy = 0.5  # Default for unknown patterns
            
        return min(accuracy, 1.0)
        
    def _calculate_square_accuracy(self, pattern: GannPattern) -> float:
        """Calculate accuracy for square patterns"""
        
        if len(pattern.points) != 4:
            return 0.0
            
        points = pattern.points
        
        # Calculate all distances
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p1, p2 = points[i], points[j]
                distance = np.sqrt((p1.time - p2.time)**2 + (p1.price - p2.price)**2)
                distances.append(distance)
                
        distances.sort()
        
        # For a perfect square: 4 equal sides + 2 equal diagonals
        if len(distances) != 6:
            return 0.0
            
        sides = distances[:4]
        diagonals = distances[4:]
        
        # Calculate side consistency
        side_mean = np.mean(sides)
        side_std = np.std(sides)
        side_consistency = 1.0 - (side_std / side_mean) if side_mean > 0 else 0.0
        
        # Calculate diagonal consistency
        diagonal_mean = np.mean(diagonals)
        diagonal_std = np.std(diagonals)
        diagonal_consistency = 1.0 - (diagonal_std / diagonal_mean) if diagonal_mean > 0 else 0.0
        
        # Check diagonal to side ratio (should be sqrt(2))
        expected_diagonal = side_mean * np.sqrt(2)
        ratio_accuracy = 1.0 - abs(diagonal_mean - expected_diagonal) / expected_diagonal if expected_diagonal > 0 else 0.0
        
        accuracy = (side_consistency * 0.4 + diagonal_consistency * 0.3 + ratio_accuracy * 0.3)
        return max(0.0, accuracy)
        
    def _calculate_triangle_accuracy(self, pattern: GannPattern) -> float:
        """Calculate accuracy for triangle patterns"""
        
        if len(pattern.points) != 3:
            return 0.0
            
        points = pattern.points
        p1, p2, p3 = points
        
        # Calculate side lengths
        side1 = np.sqrt((p1.time - p2.time)**2 + (p1.price - p2.price)**2)
        side2 = np.sqrt((p2.time - p3.time)**2 + (p2.price - p3.price)**2)
        side3 = np.sqrt((p3.time - p1.time)**2 + (p3.price - p1.price)**2)
        
        sides = [side1, side2, side3]
        
        # Calculate angles
        angles = []
        for i in range(3):
            a, b, c = sides[i], sides[(i+1)%3], sides[(i+2)%3]
            if a > 0 and b > 0:
                cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
                
        if len(angles) != 3:
            return 0.0
            
        # Check angle sum (should be 180 degrees)
        angle_sum = sum(angles)
        angle_sum_accuracy = 1.0 - abs(angle_sum - 180) / 180
        
        # Check specific triangle type accuracy
        subtype_accuracy = 0.5  # Default
        
        if pattern.pattern_subtype == "equilateral":
            # All angles should be 60 degrees
            angle_deviations = [abs(angle - 60) for angle in angles]
            max_deviation = max(angle_deviations)
            subtype_accuracy = 1.0 - max_deviation / 60 if max_deviation < 60 else 0.0
            
        elif pattern.pattern_subtype == "right":
            # One angle should be 90 degrees
            right_angle_found = any(abs(angle - 90) < 5 for angle in angles)
            subtype_accuracy = 1.0 if right_angle_found else 0.0
            
        elif pattern.pattern_subtype == "golden":
            # Check for golden ratio in sides or angles
            sides.sort()
            if sides[0] > 0:
                ratio1 = sides[1] / sides[0]
                ratio2 = sides[2] / sides[1]
                
                golden_ratio = 1.618
                ratio_accuracy1 = 1.0 - abs(ratio1 - golden_ratio) / golden_ratio
                ratio_accuracy2 = 1.0 - abs(ratio2 - golden_ratio) / golden_ratio
                
                subtype_accuracy = max(ratio_accuracy1, ratio_accuracy2)
                
        accuracy = (angle_sum_accuracy * 0.3 + subtype_accuracy * 0.7)
        return max(0.0, accuracy)        
    def _calculate_circle_accuracy(self, pattern: GannPattern) -> float:
        """Calculate accuracy for circle patterns"""
        
        if not pattern.center_point or len(pattern.points) < 3:
            return 0.0
            
        center = pattern.center_point
        
        # Calculate distances from center to all points (excluding center itself)
        distances = []
        for point in pattern.points:
            if point != center:
                distance = np.sqrt((point.time - center.time)**2 + (point.price - center.price)**2)
                distances.append(distance)
                
        if not distances:
            return 0.0
            
        # Calculate radius consistency
        mean_radius = np.mean(distances)
        std_radius = np.std(distances)
        radius_consistency = 1.0 - (std_radius / mean_radius) if mean_radius > 0 else 0.0
        
        # Check angular distribution
        angles = []
        for point in pattern.points:
            if point != center:
                angle = np.arctan2(point.price - center.price, point.time - center.time)
                angles.append(angle)
                
        angles.sort()
        
        if len(angles) < 2:
            return radius_consistency
            
        # Check angular spacing consistency
        expected_spacing = 2 * np.pi / len(angles)
        actual_spacings = []
        
        for i in range(len(angles)):
            spacing = angles[(i+1) % len(angles)] - angles[i]
            if spacing < 0:
                spacing += 2 * np.pi
            actual_spacings.append(spacing)
            
        spacing_std = np.std(actual_spacings)
        spacing_consistency = 1.0 - (spacing_std / expected_spacing) if expected_spacing > 0 else 0.0
        
        accuracy = (radius_consistency * 0.6 + spacing_consistency * 0.4)
        return max(0.0, accuracy)
        
    def _calculate_spiral_accuracy(self, pattern: GannPattern) -> float:
        """Calculate accuracy for spiral patterns"""
        
        if not pattern.center_point or len(pattern.points) < 4:
            return 0.0
            
        center = pattern.center_point
        
        # Calculate distances and angles
        points_data = []
        for point in pattern.points:
            if point != center:
                distance = np.sqrt((point.time - center.time)**2 + (point.price - center.price)**2)
                angle = np.arctan2(point.price - center.price, point.time - center.time)
                points_data.append((angle, distance, point))
                
        if len(points_data) < 3:
            return 0.0
            
        # Sort by angle
        points_data.sort(key=lambda x: x[0])
        angles = [p[0] for p in points_data]
        distances = [p[1] for p in points_data]
        
        accuracy = 0.0
        
        if pattern.pattern_subtype == "fibonacci":
            # Check golden ratio growth in distances
            growth_ratios = []
            for i in range(len(distances) - 1):
                if distances[i] > 0:
                    ratio = distances[i + 1] / distances[i]
                    growth_ratios.append(ratio)
                    
            if growth_ratios:
                golden_ratio = 1.618
                ratio_errors = [abs(r - golden_ratio) / golden_ratio for r in growth_ratios]
                ratio_accuracy = 1.0 - np.mean(ratio_errors)
                
                # Check golden angle progression
                golden_angle = np.radians(137.5)  # Golden angle in radians
                angle_errors = []
                
                for i in range(len(angles) - 1):
                    expected_increment = golden_angle
                    actual_increment = angles[i + 1] - angles[i]
                    if actual_increment < 0:
                        actual_increment += 2 * np.pi
                    
                    angle_error = abs(actual_increment - expected_increment) / expected_increment
                    angle_errors.append(angle_error)
                    
                if angle_errors:
                    angle_accuracy = 1.0 - np.mean(angle_errors)
                    accuracy = (ratio_accuracy * 0.6 + angle_accuracy * 0.4)
                    
        elif pattern.pattern_subtype == "logarithmic":
            # Check if log(distance) is linear with angle
            if SCIPY_AVAILABLE and all(d > 0 for d in distances):
                try:
                    log_distances = [np.log(d) for d in distances]
                    slope, intercept, r_value, p_value, std_err = stats.linregress(angles, log_distances)
                    accuracy = r_value**2  # R-squared as accuracy measure
                except Exception:
                    accuracy = 0.0
                    
        return max(0.0, min(1.0, accuracy))
        
    def _calculate_custom_pattern_accuracy(self, pattern: GannPattern) -> float:
        """Calculate accuracy for custom patterns"""
        
        accuracy = 0.5  # Default
        
        if pattern.pattern_subtype == "fan_convergence":
            # Check convergence accuracy
            if len(pattern.points) >= 5:
                convergence_point = pattern.points[-1]  # Last point is convergence
                line_points = pattern.points[:-1]
                
                # Check how well the lines actually converge
                if len(line_points) >= 4:
                    line1_start, line1_end = line_points[0], line_points[1]
                    line2_start, line2_end = line_points[2], line_points[3]
                    
                    # Calculate actual convergence
                    actual_convergence = self._calculate_line_convergence(
                        line1_start, line1_end, line2_start, line2_end
                    )
                    
                    if actual_convergence:
                        time_error = abs(actual_convergence.time - convergence_point.time)
                        price_error = abs(actual_convergence.price - convergence_point.price)
                        
                        # Normalize errors
                        time_span = max(p.time for p in line_points) - min(p.time for p in line_points)
                        price_span = max(p.price for p in line_points) - min(p.price for p in line_points)
                        
                        if time_span > 0 and price_span > 0:
                            normalized_time_error = time_error / time_span
                            normalized_price_error = price_error / price_span
                            
                            accuracy = 1.0 - (normalized_time_error + normalized_price_error) / 2
                            
        elif pattern.pattern_subtype == "time_price_square":
            # Check time-price ratio accuracy
            if len(pattern.points) == 2:
                p1, p2 = pattern.points
                
                time_span = abs(p2.time - p1.time)
                price_span = abs(p2.price - p1.price)
                
                # Normalize for comparison
                normalized_time = time_span / 86400  # Days
                normalized_price = price_span / p1.price if p1.price > 0 else 0
                
                if normalized_time > 0 and normalized_price > 0:
                    ratio = normalized_price / normalized_time
                    
                    # Check against target ratios
                    target_ratios = [1.0, 1.618, 0.618, 2.0, 0.5]
                    min_error = float('inf')
                    
                    for target in target_ratios:
                        error = abs(ratio - target) / target
                        min_error = min(min_error, error)
                        
                    accuracy = 1.0 - min_error if min_error < 1.0 else 0.0
                    
        elif pattern.pattern_subtype == "harmonic_division":
            # Check harmonic ratio accuracy
            if len(pattern.points) == 3:
                span_start, division_point, span_end = pattern.points
                
                total_time = span_end.time - span_start.time
                division_time = division_point.time - span_start.time
                
                if total_time > 0:
                    time_ratio = division_time / total_time
                    
                    harmonic_ratios = [0.382, 0.5, 0.618, 0.786]
                    min_error = float('inf')
                    
                    for harmonic in harmonic_ratios:
                        error = abs(time_ratio - harmonic)
                        min_error = min(min_error, error)
                        
                    accuracy = 1.0 - min_error * 2  # Scale error
                    
        return max(0.0, min(1.0, accuracy))
        
    def _classify_patterns_with_ml(self, patterns: List[GannPattern]) -> List[GannPattern]:
        """Classify and validate patterns using ML if available"""
        
        if not self.config.enable_ml or not SKLEARN_AVAILABLE:
            return patterns
            
        if not patterns:
            return patterns
            
        try:
            # Extract features for ML classification
            features = []
            for pattern in patterns:
                feature_vector = self._extract_pattern_features(pattern)
                features.append(feature_vector)
                
            if not features:
                return patterns
                
            X = np.array(features)
            
            # Perform clustering to identify pattern types
            if hasattr(self, 'pattern_clusterer') and self.pattern_clusterer:
                cluster_labels = self.pattern_clusterer.fit_predict(X)
            else:
                # Initialize clusterer if not exists
                n_clusters = min(len(patterns) // 2 + 1, 8)  # Adaptive number of clusters
                self.pattern_clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = self.pattern_clusterer.fit_predict(X)
                
            # Update pattern classifications based on clustering
            for i, pattern in enumerate(patterns):
                pattern.ml_cluster = int(cluster_labels[i])
                pattern.ml_confidence = self._calculate_ml_confidence(pattern, X[i])
                
            # Filter patterns based on ML confidence
            confidence_threshold = 0.3
            validated_patterns = [p for p in patterns if p.ml_confidence >= confidence_threshold]
            
            self.logger.info(f"ML validation: {len(validated_patterns)}/{len(patterns)} patterns passed confidence threshold")
            
            return validated_patterns
            
        except Exception as e:
            self.logger.error(f"ML pattern classification failed: {e}")
            return patterns
            
    def _extract_pattern_features(self, pattern: GannPattern) -> List[float]:
        """Extract numerical features from a pattern for ML processing"""
        
        features = []
        
        # Basic pattern properties
        features.append(len(pattern.points))
        features.append(pattern.time_span)
        features.append(pattern.price_span)
        features.append(pattern.aspect_ratio)
        features.append(pattern.geometric_accuracy)
        
        # Pattern type encoding (one-hot style)
        pattern_types = ["square", "triangle", "circle", "spiral", "custom"]
        for ptype in pattern_types:
            features.append(1.0 if pattern.pattern_type == ptype else 0.0)
            
        # Subtype encoding
        subtypes = ["regular", "equilateral", "right", "golden", "fibonacci", "logarithmic", 
                   "fan_convergence", "time_price_square", "harmonic_division"]
        for subtype in subtypes:
            features.append(1.0 if pattern.pattern_subtype == subtype else 0.0)
            
        # Statistical features of point positions
        if pattern.points:
            times = [p.time for p in pattern.points]
            prices = [p.price for p in pattern.points]
            
            # Time statistics
            features.extend([
                np.mean(times),
                np.std(times),
                np.min(times),
                np.max(times)
            ])
            
            # Price statistics
            features.extend([
                np.mean(prices),
                np.std(prices),
                np.min(prices),
                np.max(prices)
            ])
            
            # Distance statistics (from center)
            if pattern.center_point:
                distances = [
                    np.sqrt((p.time - pattern.center_point.time)**2 + 
                           (p.price - pattern.center_point.price)**2)
                    for p in pattern.points
                ]
                features.extend([
                    np.mean(distances),
                    np.std(distances),
                    np.min(distances),
                    np.max(distances)
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
                
        else:
            # Fill with zeros if no points
            features.extend([0.0] * 12)
            
        # Market context features
        if hasattr(pattern, 'market_trend'):
            features.append(pattern.market_trend)
        else:
            features.append(0.0)
            
        if hasattr(pattern, 'volatility_context'):
            features.append(pattern.volatility_context)
        else:
            features.append(0.0)
            
        # Pad or truncate to fixed size
        target_size = 50
        while len(features) < target_size:
            features.append(0.0)
        features = features[:target_size]
        
        return features
        
    def _calculate_ml_confidence(self, pattern: GannPattern, feature_vector: np.ndarray) -> float:
        """Calculate ML confidence score for a pattern"""
        
        # Base confidence from geometric accuracy
        base_confidence = pattern.geometric_accuracy
        
        # Additional confidence factors
        
        # 1. Feature consistency (how well features match typical patterns)
        feature_consistency = 0.5  # Default
        
        if hasattr(self, 'pattern_clusterer') and self.pattern_clusterer:
            try:
                # Distance to cluster center
                cluster_center = self.pattern_clusterer.cluster_centers_[pattern.ml_cluster]
                distance_to_center = np.linalg.norm(feature_vector - cluster_center)
                
                # Normalize distance (closer to center = higher confidence)
                max_distance = np.linalg.norm(cluster_center)
                if max_distance > 0:
                    normalized_distance = distance_to_center / max_distance
                    feature_consistency = 1.0 - min(normalized_distance, 1.0)
                    
            except Exception:
                pass
                
        # 2. Pattern completeness
        completeness = min(len(pattern.points) / 5.0, 1.0)  # Normalize to 5 points max
        
        # 3. Market context relevance
        market_relevance = 0.7  # Default, could be enhanced with market data
        
        # Combine confidence factors
        ml_confidence = (
            base_confidence * 0.4 +
            feature_consistency * 0.3 +
            completeness * 0.2 +
            market_relevance * 0.1
        )
        
        return max(0.0, min(1.0, ml_confidence))
        
    def _generate_trading_signals(self, patterns: List[GannPattern]) -> List[GannTradingSignal]:
        """Generate trading signals from detected patterns"""
        
        signals = []
        
        for pattern in patterns:
            signal = self._generate_pattern_signal(pattern)
            if signal:
                signals.append(signal)
                
        # Sort signals by strength
        signals.sort(key=lambda s: s.strength, reverse=True)
        
        return signals
        
    def _generate_pattern_signal(self, pattern: GannPattern) -> Optional[GannTradingSignal]:
        """Generate trading signal from a single pattern"""
        
        if not pattern.points:
            return None
            
        # Determine signal direction based on pattern characteristics
        signal_direction = self._determine_signal_direction(pattern)
        
        if signal_direction == "neutral":
            return None
            
        # Calculate signal strength
        strength = self._calculate_signal_strength(pattern)
        
        if strength < 0.3:  # Minimum strength threshold
            return None
            
        # Calculate entry point
        entry_point = self._calculate_entry_point(pattern)
        
        # Calculate target and stop loss
        target_price, stop_loss = self._calculate_targets_and_stops(pattern, signal_direction)
        
        # Create trading signal
        signal = GannTradingSignal(
            pattern_id=pattern.id,
            signal_type="pattern_breakout",
            direction=signal_direction,
            strength=strength,
            entry_price=entry_point.price,
            entry_time=entry_point.time,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=pattern.geometric_accuracy * pattern.ml_confidence if hasattr(pattern, 'ml_confidence') else pattern.geometric_accuracy
        )
        
        # Add pattern context
        signal.pattern_type = pattern.pattern_type
        signal.pattern_subtype = pattern.pattern_subtype
        signal.risk_reward_ratio = self._calculate_risk_reward_ratio(signal)
        
        return signal
        
    def _determine_signal_direction(self, pattern: GannPattern) -> str:
        """Determine trading signal direction from pattern"""
        
        if not pattern.points or len(pattern.points) < 2:
            return "neutral"
            
        # Sort points by time
        sorted_points = sorted(pattern.points, key=lambda p: p.time)
        
        # Analyze price trend in pattern
        first_point = sorted_points[0]
        last_point = sorted_points[-1]
        
        price_change = last_point.price - first_point.price
        time_span = last_point.time - first_point.time
        
        if time_span <= 0:
            return "neutral"
            
        # Pattern-specific direction logic
        if pattern.pattern_type == "square":
            # Square patterns often indicate consolidation, look for breakout direction
            if pattern.center_point:
                recent_point = sorted_points[-1]
                if recent_point.price > pattern.center_point.price:
                    return "bullish"
                else:
                    return "bearish"
                    
        elif pattern.pattern_type == "triangle":
            # Triangle patterns indicate continuation or reversal
            if pattern.pattern_subtype == "ascending":
                return "bullish"
            elif pattern.pattern_subtype == "descending":
                return "bearish"
            else:
                # Use overall trend
                return "bullish" if price_change > 0 else "bearish"
                
        elif pattern.pattern_type == "spiral":
            # Spiral patterns indicate acceleration
            if pattern.pattern_subtype == "fibonacci":
                return "bullish" if price_change > 0 else "bearish"
            else:
                return "neutral"
                
        elif pattern.pattern_type == "custom":
            if pattern.pattern_subtype == "fan_convergence":
                # Convergence often precedes major moves
                return "bullish" if price_change > 0 else "bearish"
            elif pattern.pattern_subtype == "time_price_square":
                # Time-price squares indicate timing points
                return "bullish" if price_change > 0 else "bearish"
                
        # Default: use overall price trend
        if abs(price_change) < first_point.price * 0.01:  # Less than 1% change
            return "neutral"
        else:
            return "bullish" if price_change > 0 else "bearish"
            
    def _calculate_signal_strength(self, pattern: GannPattern) -> float:
        """Calculate trading signal strength"""
        
        # Base strength from geometric accuracy
        base_strength = pattern.geometric_accuracy
        
        # Pattern complexity factor
        complexity_factor = min(len(pattern.points) / 5.0, 1.0)
        
        # Time span factor (longer patterns are more significant)
        if hasattr(pattern, 'time_span') and pattern.time_span > 0:
            time_factor = min(pattern.time_span / (7 * 24 * 3600), 1.0)  # Normalize to week
        else:
            time_factor = 0.5
            
        # Price span factor (larger price movements are more significant)
        if (hasattr(pattern, 'price_span') and pattern.price_span > 0 and 
            pattern.points and pattern.points[0].price > 0):
            price_factor = min(pattern.price_span / (pattern.points[0].price * 0.1), 1.0)  # Normalize to 10%
        else:
            price_factor = 0.5
            
        # ML confidence factor
        ml_factor = getattr(pattern, 'ml_confidence', 0.5)
        
        # Pattern type factor
        type_factors = {
            "square": 0.8,
            "triangle": 0.9,
            "circle": 0.7,
            "spiral": 0.6,
            "custom": 0.7
        }
        type_factor = type_factors.get(pattern.pattern_type, 0.5)
        
        # Combine factors
        strength = (
            base_strength * 0.3 +
            complexity_factor * 0.15 +
            time_factor * 0.15 +
            price_factor * 0.15 +
            ml_factor * 0.15 +
            type_factor * 0.1
        )
        
        return max(0.0, min(1.0, strength))
        
    def _calculate_entry_point(self, pattern: GannPattern) -> GannPatternPoint:
        """Calculate optimal entry point for pattern"""
        
        if not pattern.points:
            return GannPatternPoint(0, 0, "entry")
            
        # Sort points by time
        sorted_points = sorted(pattern.points, key=lambda p: p.time)
        
        # Default: use the most recent point
        entry_point = sorted_points[-1]
        
        # Pattern-specific entry logic
        if pattern.pattern_type == "square":
            # Enter at pattern completion (corner breakout)
            if len(sorted_points) >= 4:
                entry_point = sorted_points[-1]  # Last corner
                
        elif pattern.pattern_type == "triangle":
            # Enter at apex or breakout point
            if pattern.center_point:
                entry_point = pattern.center_point
            else:
                entry_point = sorted_points[-1]
                
        elif pattern.pattern_type == "circle":
            # Enter at circle completion
            if pattern.center_point:
                # Project entry point based on circle direction
                last_point = sorted_points[-1]
                if pattern.center_point:
                    # Calculate next point on circle
                    angle = np.arctan2(last_point.price - pattern.center_point.price,
                                     last_point.time - pattern.center_point.time)
                    next_angle = angle + np.pi / 4  # 45 degrees ahead
                    
                    radius = np.sqrt((last_point.time - pattern.center_point.time)**2 +
                                   (last_point.price - pattern.center_point.price)**2)
                    
                    entry_time = pattern.center_point.time + radius * np.cos(next_angle)
                    entry_price = pattern.center_point.price + radius * np.sin(next_angle)
                    
                    entry_point = GannPatternPoint(entry_time, entry_price, "entry")
                    
        elif pattern.pattern_type == "spiral":
            # Enter at spiral completion or next turn
            if len(sorted_points) >= 3:
                # Project next point based on spiral growth
                entry_point = self._project_spiral_next_point(pattern)
                
        # Create entry point with current timestamp if projection failed
        if not entry_point or entry_point.time < sorted_points[-1].time:
            current_time = time.time()
            entry_point = GannPatternPoint(
                time=current_time,
                price=sorted_points[-1].price,
                point_type="entry"
            )
            
        return entry_point
        
    def _project_spiral_next_point(self, pattern: GannPattern) -> GannPatternPoint:
        """Project next point in spiral pattern"""
        
        if not pattern.center_point or len(pattern.points) < 3:
            return pattern.points[-1] if pattern.points else GannPatternPoint(0, 0, "entry")
            
        center = pattern.center_point
        sorted_points = sorted([p for p in pattern.points if p != center], key=lambda p: p.time)
        
        if len(sorted_points) < 2:
            return sorted_points[-1] if sorted_points else GannPatternPoint(0, 0, "entry")
            
        # Calculate spiral parameters from last few points
        last_point = sorted_points[-1]
        prev_point = sorted_points[-2]
        
        # Calculate distance growth and angle increment
        last_distance = np.sqrt((last_point.time - center.time)**2 + (last_point.price - center.price)**2)
        prev_distance = np.sqrt((prev_point.time - center.time)**2 + (prev_point.price - center.price)**2)
        
        if prev_distance > 0:
            growth_ratio = last_distance / prev_distance
        else:
            growth_ratio = 1.618  # Default to golden ratio
            
        last_angle = np.arctan2(last_point.price - center.price, last_point.time - center.time)
        prev_angle = np.arctan2(prev_point.price - center.price, prev_point.time - center.time)
        
        angle_increment = last_angle - prev_angle
        if angle_increment < 0:
            angle_increment += 2 * np.pi
            
        # Project next point
        next_distance = last_distance * growth_ratio
        next_angle = last_angle + angle_increment
        
        next_time = center.time + next_distance * np.cos(next_angle)
        next_price = center.price + next_distance * np.sin(next_angle)
        
        return GannPatternPoint(next_time, next_price, "entry")
        
    def _calculate_targets_and_stops(self, pattern: GannPattern, direction: str) -> Tuple[float, float]:
        """Calculate target price and stop loss for pattern"""
        
        if not pattern.points:
            return 0.0, 0.0
            
        # Get price range of pattern
        prices = [p.price for p in pattern.points]
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        
        # Pattern-specific targets
        if pattern.pattern_type == "square":
            # Square patterns: target is pattern height
            target_distance = price_range
            stop_distance = price_range * 0.5
            
        elif pattern.pattern_type == "triangle":
            # Triangle patterns: target is triangle height
            target_distance = price_range * 1.5
            stop_distance = price_range * 0.3
            
        elif pattern.pattern_type == "circle":
            # Circle patterns: target is diameter
            target_distance = price_range
            stop_distance = price_range * 0.4
            
        elif pattern.pattern_type == "spiral":
            # Spiral patterns: target based on spiral growth
            if pattern.pattern_subtype == "fibonacci":
                target_distance = price_range * 1.618
                stop_distance = price_range * 0.382
            else:
                target_distance = price_range * 1.2
                stop_distance = price_range * 0.5
                
        else:  # Custom patterns
            target_distance = price_range
            stop_distance = price_range * 0.4
            
        # Apply direction
        last_price = pattern.points[-1].price
        
        if direction == "bullish":
            target_price = last_price + target_distance
            stop_loss = last_price - stop_distance
        else:  # bearish
            target_price = last_price - target_distance
            stop_loss = last_price + stop_distance
            
        return target_price, stop_loss
        
    def _calculate_risk_reward_ratio(self, signal: GannTradingSignal) -> float:
        """Calculate risk-reward ratio for signal"""
        
        if signal.entry_price == 0:
            return 0.0
            
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.target_price - signal.entry_price)
        
        if risk > 0:
            return reward / risk
        else:
            return 0.0


# Pattern Analysis and Optimization Classes

class PatternAnalyzer:
    """Advanced pattern analysis and optimization"""
    
    def __init__(self, config: GannPatternConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_pattern_performance(self, patterns: List[GannPattern], 
                                   price_data: np.ndarray) -> Dict[str, Any]:
        """Analyze historical performance of detected patterns"""
        
        performance_metrics = {
            'pattern_count': len(patterns),
            'pattern_types': {},
            'accuracy_distribution': [],
            'success_rate': 0.0,
            'avg_signal_strength': 0.0
        }
        
        if not patterns:
            return performance_metrics
            
        # Count pattern types
        for pattern in patterns:
            pattern_key = f"{pattern.pattern_type}_{pattern.pattern_subtype}"
            if pattern_key not in performance_metrics['pattern_types']:
                performance_metrics['pattern_types'][pattern_key] = 0
            performance_metrics['pattern_types'][pattern_key] += 1
            
        # Analyze accuracy distribution
        accuracies = [p.geometric_accuracy for p in patterns]
        performance_metrics['accuracy_distribution'] = {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'min': np.min(accuracies),
            'max': np.max(accuracies),
            'median': np.median(accuracies)
        }
        
        # Calculate success rate (patterns with accuracy > threshold)
        successful_patterns = sum(1 for acc in accuracies if acc >= self.config.min_accuracy)
        performance_metrics['success_rate'] = successful_patterns / len(patterns)
        
        # Calculate average signal strength if available
        strengths = []
        for pattern in patterns:
            if hasattr(pattern, 'signal_strength'):
                strengths.append(pattern.signal_strength)
                
        if strengths:
            performance_metrics['avg_signal_strength'] = np.mean(strengths)
            
        return performance_metrics
        
    def optimize_pattern_parameters(self, historical_data: np.ndarray, 
                                   validation_data: np.ndarray) -> Dict[str, Any]:
        """Optimize pattern detection parameters using historical data"""
        
        optimization_results = {
            'optimal_params': {},
            'performance_improvement': 0.0,
            'validation_accuracy': 0.0
        }
        
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available for parameter optimization")
            return optimization_results
            
        try:
            # Define parameter ranges to optimize
            param_ranges = {
                'min_accuracy': [0.3, 0.4, 0.5, 0.6, 0.7],
                'geometric_precision': [0.05, 0.1, 0.15, 0.2, 0.25],
                'min_pattern_points': [3, 4, 5, 6, 7],
                'max_time_span': [7, 14, 30, 60, 90]  # days
            }
            
            best_score = 0.0
            best_params = {}
            
            # Grid search over parameter combinations
            for min_acc in param_ranges['min_accuracy']:
                for geom_prec in param_ranges['geometric_precision']:
                    for min_points in param_ranges['min_pattern_points']:
                        for max_span in param_ranges['max_time_span']:
                            
                            # Create test configuration
                            test_config = GannPatternConfig(
                                min_accuracy=min_acc,
                                geometric_precision=geom_prec,
                                min_pattern_points=min_points,
                                max_time_span_days=max_span
                            )
                            
                            # Test configuration on historical data
                            score = self._evaluate_configuration(test_config, historical_data, validation_data)
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'min_accuracy': min_acc,
                                    'geometric_precision': geom_prec,
                                    'min_pattern_points': min_points,
                                    'max_time_span_days': max_span
                                }
                                
            optimization_results['optimal_params'] = best_params
            optimization_results['validation_accuracy'] = best_score
            
            # Calculate improvement
            baseline_score = self._evaluate_configuration(self.config, historical_data, validation_data)
            optimization_results['performance_improvement'] = best_score - baseline_score
            
            self.logger.info(f"Parameter optimization completed. Best score: {best_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Parameter optimization failed: {e}")
            
        return optimization_results
        
    def _evaluate_configuration(self, config: GannPatternConfig, 
                               historical_data: np.ndarray, 
                               validation_data: np.ndarray) -> float:
        """Evaluate a configuration's performance"""
        
        try:
            # Create detector with test configuration
            detector = GannPatternDetector(config)
            
            # Detect patterns in historical data
            patterns = detector.detect_patterns(historical_data)
            
            if not patterns:
                return 0.0
                
            # Evaluate pattern quality
            quality_score = np.mean([p.geometric_accuracy for p in patterns])
            
            # Evaluate pattern predictiveness (simplified)
            # In practice, this would involve forward-testing signals
            predictive_score = 0.5  # Placeholder
            
            # Combine scores
            total_score = quality_score * 0.7 + predictive_score * 0.3
            
            return total_score
            
        except Exception as e:
            self.logger.debug(f"Configuration evaluation failed: {e}")
            return 0.0


# Demo and Testing Functions

def demo_pattern_detector():
    """Demonstrate the Gann Pattern Detector with sample data"""
    
    print("=== Gann Pattern Detector Demo ===")
    
    # Create sample price data (sine wave + trend + noise)
    np.random.seed(42)
    time_points = np.linspace(0, 100, 1000)
    
    # Generate synthetic price data with patterns
    base_price = 100
    trend = 0.02 * time_points
    cyclical = 5 * np.sin(0.5 * time_points) + 3 * np.sin(1.2 * time_points)
    noise = np.random.normal(0, 1, len(time_points))
    
    prices = base_price + trend + cyclical + noise
    
    # Create price data array
    timestamps = time.time() - (1000 - np.arange(1000)) * 3600  # Hourly data
    
    price_data = np.column_stack([
        timestamps,  # timestamp
        prices,      # close
        prices * 1.01,  # high
        prices * 0.99,  # low
        prices * 0.995,  # open
        np.random.randint(1000, 10000, 1000)  # volume
    ])
    
    # Initialize detector
    config = GannPatternConfig(
        min_accuracy=0.4,
        enable_ml=True,
        geometric_precision=0.1
    )
    
    detector = GannPatternDetector(config)
    
    print(f"Detecting patterns in {len(price_data)} data points...")
    
    # Detect patterns
    start_time = time.time()
    patterns = detector.detect_patterns(price_data)
    detection_time = time.time() - start_time
    
    print(f"Detection completed in {detection_time:.2f} seconds")
    print(f"Found {len(patterns)} patterns")
    
    # Display pattern summary
    if patterns:
        print("\n=== Pattern Summary ===")
        
        pattern_counts = {}
        for pattern in patterns:
            key = f"{pattern.pattern_type}_{pattern.pattern_subtype}"
            pattern_counts[key] = pattern_counts.get(key, 0) + 1
            
        for pattern_type, count in pattern_counts.items():
            print(f"{pattern_type}: {count}")
            
        # Show top patterns
        print("\n=== Top 5 Patterns ===")
        top_patterns = sorted(patterns, key=lambda p: p.geometric_accuracy, reverse=True)[:5]
        
        for i, pattern in enumerate(top_patterns, 1):
            print(f"{i}. {pattern.pattern_type}_{pattern.pattern_subtype}")
            print(f"   Accuracy: {pattern.geometric_accuracy:.3f}")
            print(f"   Points: {len(pattern.points)}")
            print(f"   Time span: {pattern.time_span/3600:.1f} hours")
            print(f"   Price span: {pattern.price_span:.2f}")
            
        # Generate trading signals
        print("\n=== Trading Signals ===")
        signals = detector._generate_trading_signals(patterns)
        
        if signals:
            print(f"Generated {len(signals)} trading signals")
            
            for i, signal in enumerate(signals[:3], 1):  # Show top 3
                print(f"{i}. {signal.direction.upper()} signal")
                print(f"   Strength: {signal.strength:.3f}")
                print(f"   Confidence: {signal.confidence:.3f}")
                print(f"   Entry: {signal.entry_price:.2f}")
                print(f"   Target: {signal.target_price:.2f}")
                print(f"   Stop: {signal.stop_loss:.2f}")
                print(f"   R/R: {signal.risk_reward_ratio:.2f}")
                
        else:
            print("No trading signals generated")
            
    else:
        print("No patterns detected")
        
    # Performance analysis
    if patterns:
        print("\n=== Performance Analysis ===")
        analyzer = PatternAnalyzer(config)
        performance = analyzer.analyze_pattern_performance(patterns, price_data)
        
        print(f"Success rate: {performance['success_rate']:.1%}")
        print(f"Average accuracy: {performance['accuracy_distribution']['mean']:.3f}")
        print(f"Accuracy std: {performance['accuracy_distribution']['std']:.3f}")
        
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    # Run demo
    demo_pattern_detector()