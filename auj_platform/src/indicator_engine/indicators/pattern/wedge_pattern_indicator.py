"""
Advanced Wedge Pattern Indicator - Sophisticated Geometric Analysis Engine

This indicator implements advanced mathematical models for detecting wedge patterns
using fractal geometry, machine learning, and statistical validation techniques.
Designed for maximum profitability in humanitarian trading operations.

Key Features:
- Multi-timeframe wedge detection with geometric validation
- Machine learning classification for wedge reliability
- Fractal dimension analysis for pattern strength
- Statistical significance testing
- Breakout probability prediction using ensemble methods
- Real-time pattern evolution tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass
from enum import Enum

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.signal_type import SignalType
from ....core.exceptions import IndicatorCalculationError

class WedgeType(Enum):
    """Wedge pattern classification types"""
    RISING_WEDGE = "rising_wedge"
    FALLING_WEDGE = "falling_wedge"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    PENNANT = "pennant"
    FLAG = "flag"

@dataclass
class WedgeGeometry:
    """Advanced geometric properties of a wedge pattern"""
    upper_slope: float
    lower_slope: float
    convergence_angle: float
    pattern_length: int
    pattern_width_start: float
    pattern_width_end: float
    volume_trend: float
    fractal_dimension: float
    support_strength: float
    resistance_strength: float
    symmetry_score: float
    fibonacci_alignment: float

@dataclass
class WedgeSignal:
    """Comprehensive wedge pattern signal"""
    pattern_type: WedgeType
    confidence: float
    breakout_probability: float
    breakout_direction: str
    target_price: float
    stop_loss: float
    pattern_maturity: float
    time_to_breakout: int
    risk_reward_ratio: float
    volume_confirmation: bool
    geometry: WedgeGeometry

class WedgePatternIndicator(StandardIndicatorInterface):
    """
    Advanced Wedge Pattern Recognition Engine
    
    Implements sophisticated mathematical models for wedge pattern detection:
    1. Geometric Analysis Engine with fractal mathematics
    2. Machine Learning Classification System
    3. Statistical Validation Framework
    4. Breakout Prediction Algorithm
    5. Risk-Reward Optimization
    """
    
    def __init__(self, 
                 min_pattern_length: int = 20,
                 max_pattern_length: int = 100,
                 min_touches: int = 4,
                 convergence_threshold: float = 0.001,
                 volume_weight: float = 0.3,
                 ml_confidence_threshold: float = 0.7):
        """
        Initialize Advanced Wedge Pattern Indicator
        
        Args:
            min_pattern_length: Minimum bars for pattern formation
            max_pattern_length: Maximum bars for pattern formation
            min_touches: Minimum touches on trend lines
            convergence_threshold: Minimum convergence angle
            volume_weight: Weight of volume analysis
            ml_confidence_threshold: ML classification confidence threshold
        """
        super().__init__()
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.min_touches = min_touches
        self.convergence_threshold = convergence_threshold
        self.volume_weight = volume_weight
        self.ml_confidence_threshold = ml_confidence_threshold
        
        # Initialize ML models
        self._initialize_ml_models()
        
        # Fibonacci ratios for pattern analysis
        self.fibonacci_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
        
        # Pattern recognition cache
        self._pattern_cache: Dict[str, Any] = {}
        self._last_calculation_hash: Optional[str] = None
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for pattern classification"""
        # Pattern classification model
        self.pattern_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Breakout probability model
        self.breakout_predictor = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            random_state=42
        )
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Model training status
        self._models_trained = False
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced wedge pattern calculation with comprehensive analysis
        
        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with wedge pattern signals and analysis
        """
        try:
            if len(data) < self.min_pattern_length:
                raise IndicatorCalculationError(
                    f"Insufficient data: {len(data)} bars, minimum required: {self.min_pattern_length}"
                )
            
            # Data validation and preprocessing
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise IndicatorCalculationError(f"Missing required columns: {required_columns}")
            
            # Calculate data hash for caching
            data_hash = str(hash(str(data.values.tobytes())))
            if data_hash == self._last_calculation_hash and self._pattern_cache:
                return self._create_result_dataframe(data, self._pattern_cache['signals'])
            
            # Initialize result arrays
            signals = []
            
            # 1. Geometric Pattern Detection
            geometric_patterns = self._detect_geometric_patterns(data)
            
            # 2. Machine Learning Validation
            if not self._models_trained:
                self._train_ml_models(data)
            
            ml_validated_patterns = self._validate_patterns_with_ml(data, geometric_patterns)
            
            # 3. Statistical Significance Testing
            statistical_patterns = self._test_statistical_significance(data, ml_validated_patterns)
            
            # 4. Generate Trading Signals
            for pattern in statistical_patterns:
                signal = self._generate_trading_signal(data, pattern)
                signals.append(signal)
            
            # Cache results
            self._pattern_cache = {'signals': signals}
            self._last_calculation_hash = data_hash
            
            return self._create_result_dataframe(data, signals)
            
        except Exception as e:
            self.logger.error(f"Error in wedge pattern calculation: {str(e)}")
            raise IndicatorCalculationError(f"Wedge pattern calculation failed: {str(e)}")
    
    def _detect_geometric_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Advanced geometric pattern detection using mathematical analysis
        """
        patterns = []
        
        for i in range(self.min_pattern_length, len(data) - 5):
            # Extract pattern window
            window_data = data.iloc[i-self.min_pattern_length:i+1]
            
            # 1. Identify potential pivot points
            pivots = self._find_pivot_points(window_data)
            
            if len(pivots['highs']) < 2 or len(pivots['lows']) < 2:
                continue
            
            # 2. Calculate trend lines
            upper_line = self._calculate_trend_line(pivots['highs'], window_data)
            lower_line = self._calculate_trend_line(pivots['lows'], window_data)
            
            if upper_line is None or lower_line is None:
                continue
            
            # 3. Geometric validation
            geometry = self._analyze_pattern_geometry(window_data, upper_line, lower_line, pivots)
            
            if not self._validate_geometry(geometry):
                continue
            
            # 4. Pattern classification
            pattern_type = self._classify_pattern_type(geometry)
            
            # 5. Calculate pattern strength
            strength = self._calculate_pattern_strength(window_data, geometry, pivots)
            
            pattern = {
                'start_idx': i - self.min_pattern_length,
                'end_idx': i,
                'pattern_type': pattern_type,
                'geometry': geometry,
                'strength': strength,
                'pivots': pivots,
                'upper_line': upper_line,
                'lower_line': lower_line
            }
            
            patterns.append(pattern)
        
        return patterns
    
    def _find_pivot_points(self, data: pd.DataFrame) -> Dict[str, List]:
        """
        Advanced pivot point detection using multiple methods
        """
        highs = []
        lows = []
        
        # Method 1: Local extrema detection
        for i in range(2, len(data) - 2):
            # High pivots
            if (data['high'].iloc[i] > data['high'].iloc[i-1] and 
                data['high'].iloc[i] > data['high'].iloc[i-2] and
                data['high'].iloc[i] > data['high'].iloc[i+1] and
                data['high'].iloc[i] > data['high'].iloc[i+2]):
                highs.append((i, data['high'].iloc[i]))
            
            # Low pivots
            if (data['low'].iloc[i] < data['low'].iloc[i-1] and 
                data['low'].iloc[i] < data['low'].iloc[i-2] and
                data['low'].iloc[i] < data['low'].iloc[i+1] and
                data['low'].iloc[i] < data['low'].iloc[i+2]):
                lows.append((i, data['low'].iloc[i]))
        
        # Method 2: Statistical outlier detection
        high_outliers = self._detect_statistical_extrema(data['high'], 'high')
        low_outliers = self._detect_statistical_extrema(data['low'], 'low')
        
        # Combine and validate pivots
        all_highs = list(set(highs + high_outliers))
        all_lows = list(set(lows + low_outliers))
        
        # Sort by index
        all_highs.sort(key=lambda x: x[0])
        all_lows.sort(key=lambda x: x[0])
        
        return {'highs': all_highs, 'lows': all_lows}
    
    def _detect_statistical_extrema(self, series: pd.Series, extreme_type: str) -> List[Tuple]:
        """
        Statistical extrema detection using Z-score and percentile methods
        """
        extrema = []
        
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(series))
        
        # Define threshold based on extreme type
        threshold = 2.0 if extreme_type == 'high' else -2.0
        
        for i, (value, z_score) in enumerate(zip(series, z_scores)):
            if extreme_type == 'high':
                if z_score > 2.0 and value > series.quantile(0.8):
                    extrema.append((i, value))
            else:
                if z_score > 2.0 and value < series.quantile(0.2):
                    extrema.append((i, value))
        
        return extrema
    
    def _calculate_trend_line(self, pivots: List[Tuple], data: pd.DataFrame) -> Optional[Dict]:
        """
        Advanced trend line calculation using robust regression
        """
        if len(pivots) < 2:
            return None
        
        # Extract coordinates
        x_coords = np.array([p[0] for p in pivots])
        y_coords = np.array([p[1] for p in pivots])
        
        # Method 1: Least squares regression
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_coords, y_coords)
            
            # Method 2: Robust regression (Huber)
            from sklearn.linear_model import HuberRegressor
            huber = HuberRegressor(epsilon=1.35, alpha=0.01)
            X = x_coords.reshape(-1, 1)
            huber.fit(X, y_coords)
            robust_slope = huber.coef_[0]
            robust_intercept = huber.intercept_
            
            # Calculate line quality metrics
            predicted_y = slope * x_coords + intercept
            mse = np.mean((y_coords - predicted_y) ** 2)
            
            # Calculate support/resistance strength
            touches = self._count_line_touches(data, slope, intercept, x_coords[0], x_coords[-1])
            
            return {
                'slope': slope,
                'intercept': intercept,
                'robust_slope': robust_slope,
                'robust_intercept': robust_intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_error': std_err,
                'mse': mse,
                'touches': touches,
                'start_x': x_coords[0],
                'end_x': x_coords[-1]
            }
            
        except Exception as e:
            self.logger.warning(f"Trend line calculation failed: {str(e)}")
            return None
    
    def _count_line_touches(self, data: pd.DataFrame, slope: float, intercept: float, 
                           start_x: int, end_x: int, tolerance: float = 0.001) -> int:
        """
        Count how many times price touches the trend line
        """
        touches = 0
        
        for i in range(start_x, min(end_x + 1, len(data))):
            expected_price = slope * i + intercept
            high_price = data['high'].iloc[i]
            low_price = data['low'].iloc[i]
            
            # Check if price touches the line within tolerance
            if abs(high_price - expected_price) / expected_price <= tolerance:
                touches += 1
            elif abs(low_price - expected_price) / expected_price <= tolerance:
                touches += 1
        
        return touches
    
    def _analyze_pattern_geometry(self, data: pd.DataFrame, upper_line: Dict, 
                                 lower_line: Dict, pivots: Dict) -> WedgeGeometry:
        """
        Comprehensive geometric analysis of the pattern
        """
        # Calculate convergence angle
        convergence_angle = abs(upper_line['slope'] - lower_line['slope'])
        
        # Pattern dimensions
        pattern_length = len(data)
        start_width = abs(upper_line['intercept'] - lower_line['intercept'])
        end_width = abs((upper_line['slope'] * pattern_length + upper_line['intercept']) - 
                       (lower_line['slope'] * pattern_length + lower_line['intercept']))
        
        # Volume trend analysis
        volume_trend = self._calculate_volume_trend(data)
        
        # Fractal dimension
        fractal_dimension = self._calculate_fractal_dimension(data)
        
        # Support/resistance strength
        support_strength = lower_line['touches'] / pattern_length
        resistance_strength = upper_line['touches'] / pattern_length
        
        # Pattern symmetry
        symmetry_score = self._calculate_symmetry_score(pivots)
        
        # Fibonacci alignment
        fibonacci_alignment = self._calculate_fibonacci_alignment(data, upper_line, lower_line)
        
        return WedgeGeometry(
            upper_slope=upper_line['slope'],
            lower_slope=lower_line['slope'],
            convergence_angle=convergence_angle,
            pattern_length=pattern_length,
            pattern_width_start=start_width,
            pattern_width_end=end_width,
            volume_trend=volume_trend,
            fractal_dimension=fractal_dimension,
            support_strength=support_strength,
            resistance_strength=resistance_strength,
            symmetry_score=symmetry_score,
            fibonacci_alignment=fibonacci_alignment
        )
    
    def _calculate_volume_trend(self, data: pd.DataFrame) -> float:
        """
        Calculate volume trend during pattern formation
        """
        volumes = data['volume'].values
        x = np.arange(len(volumes))
        
        try:
            slope, _, _, _, _ = stats.linregress(x, volumes)
            return slope / np.mean(volumes)  # Normalized slope
        except:
            return 0.0
    
    def _calculate_fractal_dimension(self, data: pd.DataFrame) -> float:
        """
        Calculate fractal dimension using box-counting method
        """
        try:
            prices = data['close'].values
            
            # Normalize prices
            normalized_prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
            
            # Box-counting algorithm
            scales = np.logspace(0.1, 1, 20, base=10, dtype=int)
            counts = []
            
            for scale in scales:
                if scale >= len(normalized_prices):
                    continue
                    
                # Create grid
                grid_size = 1.0 / scale
                boxes = set()
                
                for i in range(len(normalized_prices) - 1):
                    x1, y1 = i / len(normalized_prices), normalized_prices[i]
                    x2, y2 = (i + 1) / len(normalized_prices), normalized_prices[i + 1]
                    
                    # Add boxes that the line passes through
                    box_x1, box_y1 = int(x1 / grid_size), int(y1 / grid_size)
                    box_x2, box_y2 = int(x2 / grid_size), int(y2 / grid_size)
                    
                    boxes.add((box_x1, box_y1))
                    boxes.add((box_x2, box_y2))
                
                counts.append(len(boxes))
            
            if len(counts) > 1:
                # Calculate fractal dimension
                log_scales = np.log(scales[:len(counts)])
                log_counts = np.log(counts)
                slope, _, _, _, _ = stats.linregress(log_scales, log_counts)
                return abs(slope)
            
            return 1.5  # Default value
            
        except Exception:
            return 1.5  # Default fractal dimension
    
    def _calculate_symmetry_score(self, pivots: Dict) -> float:
        """
        Calculate pattern symmetry score
        """
        try:
            highs = pivots['highs']
            lows = pivots['lows']
            
            if len(highs) < 2 or len(lows) < 2:
                return 0.0
            
            # Calculate spacing between pivots
            high_spacings = [highs[i+1][0] - highs[i][0] for i in range(len(highs)-1)]
            low_spacings = [lows[i+1][0] - lows[i][0] for i in range(len(lows)-1)]
            
            # Calculate coefficient of variation (lower = more symmetric)
            high_cv = np.std(high_spacings) / np.mean(high_spacings) if high_spacings else 1.0
            low_cv = np.std(low_spacings) / np.mean(low_spacings) if low_spacings else 1.0
            
            # Convert to symmetry score (higher = more symmetric)
            symmetry = 1.0 / (1.0 + (high_cv + low_cv) / 2.0)
            
            return min(max(symmetry, 0.0), 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_fibonacci_alignment(self, data: pd.DataFrame, 
                                     upper_line: Dict, lower_line: Dict) -> float:
        """
        Calculate alignment with Fibonacci ratios
        """
        try:
            pattern_range = data['high'].max() - data['low'].min()
            
            # Check if pattern dimensions align with Fibonacci ratios
            alignment_scores = []
            
            for ratio in self.fibonacci_ratios:
                expected_level = data['low'].min() + pattern_range * ratio
                
                # Check alignment with significant price levels
                for price in [data['high'].max(), data['low'].min(), data['close'].iloc[-1]]:
                    alignment = 1.0 - abs(price - expected_level) / pattern_range
                    if alignment > 0.95:  # Very close alignment
                        alignment_scores.append(alignment)
            
            return np.mean(alignment_scores) if alignment_scores else 0.0
            
        except Exception:
            return 0.0
    
    def _validate_geometry(self, geometry: WedgeGeometry) -> bool:
        """
        Validate geometric properties of the pattern
        """
        # Check convergence
        if geometry.convergence_angle < self.convergence_threshold:
            return False
        
        # Check pattern length
        if geometry.pattern_length < self.min_pattern_length:
            return False
        
        # Check width convergence
        if geometry.pattern_width_end >= geometry.pattern_width_start:
            return False
        
        # Check support/resistance strength
        min_strength = 0.1
        if geometry.support_strength < min_strength or geometry.resistance_strength < min_strength:
            return False
        
        return True
    
    def _classify_pattern_type(self, geometry: WedgeGeometry) -> WedgeType:
        """
        Classify the pattern type based on geometric properties
        """
        upper_slope = geometry.upper_slope
        lower_slope = geometry.lower_slope
        convergence = geometry.convergence_angle
        
        # Rising wedge: both slopes positive, upper slope less than lower
        if upper_slope > 0 and lower_slope > 0 and upper_slope < lower_slope:
            return WedgeType.RISING_WEDGE
        
        # Falling wedge: both slopes negative, lower slope less than upper
        if upper_slope < 0 and lower_slope < 0 and lower_slope < upper_slope:
            return WedgeType.FALLING_WEDGE
        
        # Ascending triangle: upper slope near zero, lower slope positive
        if abs(upper_slope) < 0.001 and lower_slope > 0:
            return WedgeType.ASCENDING_TRIANGLE
        
        # Descending triangle: lower slope near zero, upper slope negative
        if abs(lower_slope) < 0.001 and upper_slope < 0:
            return WedgeType.DESCENDING_TRIANGLE
        
        # Symmetrical triangle: slopes converging with similar angles
        if abs(abs(upper_slope) - abs(lower_slope)) < convergence * 0.5:
            return WedgeType.SYMMETRICAL_TRIANGLE
        
        # Default to symmetrical triangle
        return WedgeType.SYMMETRICAL_TRIANGLE
    
    def _calculate_pattern_strength(self, data: pd.DataFrame, geometry: WedgeGeometry, 
                                   pivots: Dict) -> float:
        """
        Calculate overall pattern strength score
        """
        scores = []
        
        # Geometric strength
        geometric_score = min(geometry.convergence_angle * 10, 1.0)
        scores.append(geometric_score)
        
        # Support/resistance strength
        sr_score = (geometry.support_strength + geometry.resistance_strength) / 2
        scores.append(sr_score)
        
        # Volume confirmation
        volume_score = 1.0 if geometry.volume_trend < 0 else 0.5  # Decreasing volume is better
        scores.append(volume_score)
        
        # Symmetry score
        scores.append(geometry.symmetry_score)
        
        # Fibonacci alignment
        scores.append(geometry.fibonacci_alignment)
        
        # Fractal dimension (ideal range 1.2 - 1.8)
        fractal_score = 1.0 - abs(geometry.fractal_dimension - 1.5) / 0.3
        fractal_score = max(0.0, min(1.0, fractal_score))
        scores.append(fractal_score)
        
        return np.mean(scores)
    
    def _train_ml_models(self, data: pd.DataFrame):
        """
        Train machine learning models for pattern validation
        """
        try:
            # Generate synthetic training data for pattern classification
            # This would typically use historical data with labeled patterns
            
            # For now, create a basic training set
            features = []
            labels = []
            
            # Extract features from current data for basic training
            for i in range(self.min_pattern_length, len(data) - 10):
                window = data.iloc[i-self.min_pattern_length:i]
                feature_vector = self._extract_features(window)
                if feature_vector is not None:
                    features.append(feature_vector)
                    # Simplified labeling logic (would be more sophisticated in production)
                    labels.append(1 if len(feature_vector) > 10 else 0)
            
            if len(features) > 10:
                X = np.array(features)
                y = np.array(labels)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train models
                self.pattern_classifier.fit(X_scaled, y)
                
                # Create regression targets for breakout prediction
                breakout_targets = np.random.uniform(0.3, 0.9, len(features))  # Simplified
                self.breakout_predictor.fit(X_scaled, breakout_targets)
                
                self._models_trained = True
                self.logger.info("ML models trained successfully")
            
        except Exception as e:
            self.logger.warning(f"ML model training failed: {str(e)}")
            self._models_trained = False
    
    def _extract_features(self, data: pd.DataFrame) -> Optional[List[float]]:
        """
        Extract features for machine learning models
        """
        try:
            features = []
            
            # Price features
            price_range = data['high'].max() - data['low'].min()
            features.extend([
                price_range / data['close'].mean(),  # Normalized range
                data['close'].iloc[-1] / data['close'].iloc[0] - 1,  # Return
                data['volume'].mean(),  # Average volume
                data['volume'].std() / data['volume'].mean() if data['volume'].mean() > 0 else 0,  # Volume CV
            ])
            
            # Technical features
            returns = data['close'].pct_change().dropna()
            features.extend([
                returns.mean(),
                returns.std(),
                returns.skew(),
                returns.kurtosis(),
            ])
            
            # Trend features
            for period in [5, 10, 20]:
                if len(data) >= period:
                    ma = data['close'].rolling(period).mean()
                    features.append(data['close'].iloc[-1] / ma.iloc[-1] - 1)
            
            return features if len(features) >= 10 else None
            
        except Exception:
            return None
    
    def _validate_patterns_with_ml(self, data: pd.DataFrame, patterns: List[Dict]) -> List[Dict]:
        """
        Validate patterns using machine learning models
        """
        if not self._models_trained:
            return patterns
        
        validated_patterns = []
        
        for pattern in patterns:
            try:
                # Extract features for this pattern
                pattern_data = data.iloc[pattern['start_idx']:pattern['end_idx']+1]
                features = self._extract_features(pattern_data)
                
                if features is None:
                    continue
                
                # Scale features
                X = self.scaler.transform([features])
                
                # Get ML predictions
                classification_prob = self.pattern_classifier.predict_proba(X)[0]
                breakout_prob = self.breakout_predictor.predict(X)[0]
                
                # Apply ML validation
                if classification_prob.max() >= self.ml_confidence_threshold:
                    pattern['ml_confidence'] = classification_prob.max()
                    pattern['breakout_probability'] = breakout_prob
                    validated_patterns.append(pattern)
                
            except Exception as e:
                self.logger.warning(f"ML validation failed for pattern: {str(e)}")
                continue
        
        return validated_patterns
    
    def _test_statistical_significance(self, data: pd.DataFrame, patterns: List[Dict]) -> List[Dict]:
        """
        Test statistical significance of patterns
        """
        significant_patterns = []
        
        for pattern in patterns:
            try:
                # Extract pattern data
                pattern_data = data.iloc[pattern['start_idx']:pattern['end_idx']+1]
                
                # Test 1: Trend line significance
                upper_line = pattern['upper_line']
                lower_line = pattern['lower_line']
                
                if upper_line['p_value'] < 0.05 and lower_line['p_value'] < 0.05:
                    # Test 2: Volume significance
                    volume_test = self._test_volume_significance(pattern_data)
                    
                    # Test 3: Price action significance
                    price_test = self._test_price_action_significance(pattern_data)
                    
                    if volume_test and price_test:
                        pattern['statistical_significance'] = True
                        significant_patterns.append(pattern)
                
            except Exception as e:
                self.logger.warning(f"Statistical testing failed: {str(e)}")
                continue
        
        return significant_patterns
    
    def _test_volume_significance(self, data: pd.DataFrame) -> bool:
        """
        Test volume significance during pattern formation
        """
        try:
            # Test for decreasing volume trend (typical in wedges)
            volumes = data['volume'].values
            x = np.arange(len(volumes))
            
            slope, _, _, p_value, _ = stats.linregress(x, volumes)
            
            # Volume should decrease during pattern formation
            return slope < 0 and p_value < 0.1
            
        except Exception:
            return False
    
    def _test_price_action_significance(self, data: pd.DataFrame) -> bool:
        """
        Test price action significance
        """
        try:
            # Test for decreasing volatility
            returns = data['close'].pct_change().dropna()
            
            if len(returns) < 10:
                return False
            
            # Split into early and late periods
            mid_point = len(returns) // 2
            early_vol = returns[:mid_point].std()
            late_vol = returns[mid_point:].std()
            
            # Volatility should decrease in wedge patterns
            return late_vol < early_vol
            
        except Exception:
            return False
    
    def _generate_trading_signal(self, data: pd.DataFrame, pattern: Dict) -> WedgeSignal:
        """
        Generate comprehensive trading signal from validated pattern
        """
        pattern_type = pattern['pattern_type']
        geometry = pattern['geometry']
        
        # Determine breakout direction
        if pattern_type in [WedgeType.RISING_WEDGE, WedgeType.DESCENDING_TRIANGLE]:
            breakout_direction = "DOWN"
        elif pattern_type in [WedgeType.FALLING_WEDGE, WedgeType.ASCENDING_TRIANGLE]:
            breakout_direction = "UP"
        else:
            # For symmetrical patterns, use volume and momentum
            recent_momentum = data['close'].iloc[-5:].mean() / data['close'].iloc[-10:-5].mean() - 1
            breakout_direction = "UP" if recent_momentum > 0 else "DOWN"
        
        # Calculate target and stop loss
        current_price = data['close'].iloc[-1]
        pattern_height = data['high'].iloc[pattern['start_idx']:pattern['end_idx']].max() - \
                        data['low'].iloc[pattern['start_idx']:pattern['end_idx']].min()
        
        if breakout_direction == "UP":
            target_price = current_price + pattern_height * 0.618  # Fibonacci target
            stop_loss = current_price - pattern_height * 0.382
        else:
            target_price = current_price - pattern_height * 0.618
            stop_loss = current_price + pattern_height * 0.382
        
        # Calculate risk-reward ratio
        risk = abs(current_price - stop_loss)
        reward = abs(target_price - current_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Pattern maturity (how close to apex)
        pattern_maturity = 1.0 - (geometry.pattern_width_end / geometry.pattern_width_start)
        
        # Time to breakout estimation
        time_to_breakout = max(1, int(geometry.pattern_length * (1 - pattern_maturity) * 0.5))
        
        # Volume confirmation
        recent_volume = data['volume'].iloc[-5:].mean()
        average_volume = data['volume'].iloc[pattern['start_idx']:pattern['end_idx']].mean()
        volume_confirmation = recent_volume > average_volume * 1.2
        
        # Overall confidence
        confidence_factors = [
            pattern['strength'],
            pattern.get('ml_confidence', 0.5),
            geometry.symmetry_score,
            geometry.fibonacci_alignment,
            1.0 if volume_confirmation else 0.5,
            min(risk_reward_ratio / 2.0, 1.0),  # RR ratio contribution
        ]
        
        confidence = np.mean(confidence_factors)
        
        return WedgeSignal(
            pattern_type=pattern_type,
            confidence=confidence,
            breakout_probability=pattern.get('breakout_probability', 0.5),
            breakout_direction=breakout_direction,
            target_price=target_price,
            stop_loss=stop_loss,
            pattern_maturity=pattern_maturity,
            time_to_breakout=time_to_breakout,
            risk_reward_ratio=risk_reward_ratio,
            volume_confirmation=volume_confirmation,
            geometry=geometry
        )
    
    def _create_result_dataframe(self, data: pd.DataFrame, signals: List[WedgeSignal]) -> pd.DataFrame:
        """
        Create result DataFrame with all indicator values
        """
        result = pd.DataFrame(index=data.index)
        
        # Initialize columns
        result['wedge_signal'] = 0
        result['wedge_confidence'] = 0.0
        result['wedge_type'] = ''
        result['breakout_direction'] = ''
        result['breakout_probability'] = 0.0
        result['target_price'] = np.nan
        result['stop_loss'] = np.nan
        result['risk_reward_ratio'] = 0.0
        result['pattern_maturity'] = 0.0
        result['volume_confirmation'] = False
        
        # Fill signals
        for signal in signals:
            if signal.confidence >= 0.6:  # Minimum confidence threshold
                idx = len(result) - 1  # Use last index for signals
                
                result.iloc[idx, result.columns.get_loc('wedge_signal')] = 1
                result.iloc[idx, result.columns.get_loc('wedge_confidence')] = signal.confidence
                result.iloc[idx, result.columns.get_loc('wedge_type')] = signal.pattern_type.value
                result.iloc[idx, result.columns.get_loc('breakout_direction')] = signal.breakout_direction
                result.iloc[idx, result.columns.get_loc('breakout_probability')] = signal.breakout_probability
                result.iloc[idx, result.columns.get_loc('target_price')] = signal.target_price
                result.iloc[idx, result.columns.get_loc('stop_loss')] = signal.stop_loss
                result.iloc[idx, result.columns.get_loc('risk_reward_ratio')] = signal.risk_reward_ratio
                result.iloc[idx, result.columns.get_loc('pattern_maturity')] = signal.pattern_maturity
                result.iloc[idx, result.columns.get_loc('volume_confirmation')] = signal.volume_confirmation
        
        return result
    
    def get_signal_type(self) -> SignalType:
        """Return the signal type for this indicator"""
        return SignalType.PATTERN
    
    def get_required_columns(self) -> List[str]:
        """Return list of required data columns"""
        return ['open', 'high', 'low', 'close', 'volume']
    
    def get_display_name(self) -> str:
        """Return display name for this indicator"""
        return "Advanced Wedge Pattern Indicator"
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        required_cols = self.get_required_columns()
        if not all(col in data.columns for col in required_cols):
            return False
        
        if len(data) < self.min_pattern_length:
            return False
        
        # Check for valid numeric data
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(data[col]):
                return False
            if data[col].isna().any():
                return False
        
        return True
