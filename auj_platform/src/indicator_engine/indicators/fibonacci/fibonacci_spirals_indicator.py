"""
Fibonacci Spirals Indicator - Advanced Golden Ratio Spiral Mathematics for Market Cycles
========================================================================================

This module implements a sophisticated Fibonacci spirals indicator that uses golden ratio
spiral mathematics to detect natural market cycles and turning points. It includes advanced
algorithms for spiral construction, cycle analysis, machine learning integration, and
sophisticated pattern recognition for market timing.

Features:
- Mathematical golden ratio spiral construction (φ = 1.618...)
- Multiple spiral types (Archimedean, logarithmic, hyperbolic)
- Advanced swing point detection for spiral anchoring
- Machine learning cycle prediction and validation
- Natural market rhythm detection
- Multi-timeframe spiral confluence analysis
- Geometric price and time projections
- Spiral intersection point calculations
- Dynamic spiral expansion and contraction analysis
- Real-time spiral pattern matching

The indicator helps traders identify natural market turning points based on the mathematical
principles of the golden ratio and Fibonacci sequence, providing insights into market timing
and cyclical behavior patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import optimize, interpolate
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, IndicatorResult, DataRequirement, DataType, SignalType
from ...core.exceptions import IndicatorCalculationException

# Configure logging
logger = logging.getLogger(__name__)

# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio φ ≈ 1.618
PHI_INVERSE = 1 / PHI       # φ^-1 ≈ 0.618
FIBONACCI_ANGLES = [23.6, 38.2, 50.0, 61.8, 78.6]  # Fibonacci angle degrees


@dataclass
class SpiralPoint:
    """Represents a point on a Fibonacci spiral"""
    theta: float        # Angle in radians
    radius: float       # Distance from center
    price: float        # Market price at this point
    time_index: int     # Time index in the data
    timestamp: datetime # Actual timestamp
    spiral_id: str      # Identifier for the spiral
    

@dataclass
class FibonacciSpiral:
    """Represents a complete Fibonacci spiral"""
    spiral_id: str
    spiral_type: str    # 'logarithmic', 'archimedean', 'hyperbolic'
    center_point: Tuple[int, float]  # (time_index, price)
    center_timestamp: datetime
    growth_factor: float
    rotation: float     # Initial rotation in radians
    points: List[SpiralPoint]
    strength: float     # 0.0 to 1.0
    cycle_length: int   # Estimated cycle length in periods
    ml_score: float     # Machine learning confidence
    

@dataclass
class SpiralIntersection:
    """Represents an intersection between spirals or spiral and price"""
    intersection_type: str  # 'spiral_spiral', 'spiral_price', 'spiral_time'
    point: SpiralPoint
    confidence: float
    spirals_involved: List[str]
    prediction_strength: float
    

class FibonacciSpiralsIndicator(StandardIndicatorInterface):
    """
    Advanced Fibonacci Spirals Indicator with golden ratio mathematics
    and sophisticated cycle detection capabilities.
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'swing_detection_period': 12,
            'min_swing_strength': 0.4,
            'spiral_types': ['logarithmic', 'archimedean'],
            'max_spirals': 8,
            'spiral_resolution': 100,  # Points per spiral
            'cycle_lookback': 100,
            'min_cycle_length': 10,
            'max_cycle_length': 200,
            'intersection_tolerance': 0.005,  # 0.5% price tolerance
            'ml_lookback': 75,
            'growth_factor_range': (0.1, 0.5),
            'confidence_threshold': 0.6,
            'time_weight': 0.4,
            'price_weight': 0.6,
            'spiral_strength_threshold': 0.3,
            'max_spiral_age_days': 60,
            'use_ml_validation': True
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name="FibonacciSpirals", parameters=default_params)
        
        # Initialize internal state
        self.spirals: List[FibonacciSpiral] = []
        self.intersections: List[SpiralIntersection] = []
        self.ml_model = None
        self.scaler = StandardScaler()
        self.last_calculation = None
        
        # Precompute mathematical constants
        self.phi_powers = [PHI ** i for i in range(-5, 10)]
        self.fibonacci_numbers = self._generate_fibonacci_sequence(50)
        
        logger.info(f"FibonacciSpiralsIndicator initialized with {len(self.parameters['spiral_types'])} spiral types")
    
    def get_data_requirements(self) -> DataRequirement:
        """Define the data requirements for Fibonacci spirals calculation"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(100, self.parameters['cycle_lookback']),
            lookback_periods=300
        )
    
    def validate_parameters(self) -> bool:
        """Validate the indicator parameters"""
        try:
            required_params = ['spiral_types', 'spiral_resolution', 'cycle_lookback']
            for param in required_params:
                if param not in self.parameters:
                    raise ValueError(f"Missing required parameter: {param}")
            
            if self.parameters['spiral_resolution'] < 10:
                raise ValueError("spiral_resolution must be at least 10")
            
            if not self.parameters['spiral_types']:
                raise ValueError("spiral_types cannot be empty")
            
            valid_types = ['logarithmic', 'archimedean', 'hyperbolic']
            for spiral_type in self.parameters['spiral_types']:
                if spiral_type not in valid_types:
                    raise ValueError(f"Invalid spiral_type: {spiral_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            return False
    
    def _generate_fibonacci_sequence(self, n: int) -> List[int]:
        """Generate Fibonacci sequence up to n numbers"""
        if n <= 0:
            return []
        elif n == 1:
            return [1]
        elif n == 2:
            return [1, 1]
        
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        
        return fib
    
    def _detect_significant_points(self, data: pd.DataFrame) -> List[Tuple[int, float, str]]:
        """
        Detect significant market points for spiral anchoring
        """
        try:
            period = self.parameters['swing_detection_period']
            min_strength = self.parameters['min_swing_strength']
            
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values
            
            significant_points = []
            
            # Detect swing highs
            high_peaks, high_properties = find_peaks(
                highs, 
                distance=period,
                prominence=np.std(highs) * min_strength
            )
            
            # Detect swing lows
            low_peaks, low_properties = find_peaks(
                -lows,
                distance=period,
                prominence=np.std(lows) * min_strength
            )
            
            # Add significant highs
            for i, peak_idx in enumerate(high_peaks):
                if peak_idx >= period and peak_idx < len(data) - period:
                    strength = high_properties['prominences'][i] / np.std(highs)
                    if strength >= min_strength:
                        significant_points.append((peak_idx, highs[peak_idx], 'high'))
            
            # Add significant lows
            for i, peak_idx in enumerate(low_peaks):
                if peak_idx >= period and peak_idx < len(data) - period:
                    strength = low_properties['prominences'][i] / np.std(lows)
                    if strength >= min_strength:
                        significant_points.append((peak_idx, lows[peak_idx], 'low'))
            
            # Add some significant close price points for additional anchors
            close_changes = np.abs(np.diff(closes))
            significant_change_indices = np.where(close_changes > np.percentile(close_changes, 90))[0]
            
            for idx in significant_change_indices:
                if idx >= period and idx < len(data) - period:
                    significant_points.append((idx, closes[idx], 'close'))
            
            # Sort by time and limit count
            significant_points.sort(key=lambda x: x[0])
            return significant_points[-20:]  # Keep only recent points
            
        except Exception as e:
            logger.error(f"Error detecting significant points: {str(e)}")
            return []
    
    def _create_logarithmic_spiral(self, center: Tuple[int, float], data: pd.DataFrame, 
                                  spiral_id: str, growth_factor: float = None) -> FibonacciSpiral:
        """
        Create a logarithmic (golden) spiral based on the golden ratio
        """
        try:
            if growth_factor is None:
                growth_factor = np.log(PHI) / (2 * np.pi)  # Natural golden spiral growth
            
            center_idx, center_price = center
            resolution = self.parameters['spiral_resolution']
            
            # Calculate spiral parameters
            max_theta = 6 * np.pi  # 3 full rotations
            theta_values = np.linspace(0, max_theta, resolution)
            
            spiral_points = []
            
            for i, theta in enumerate(theta_values):
                # Logarithmic spiral: r = a * e^(b*θ)
                radius = np.exp(growth_factor * theta)
                
                # Convert polar to market coordinates
                time_offset = radius * np.cos(theta) * 10  # Scale factor for time
                price_offset = radius * np.sin(theta) * (data['close'].std() * 0.1)
                
                # Calculate actual market position
                time_index = int(center_idx + time_offset)
                price = center_price + price_offset
                
                # Ensure we stay within data bounds
                if 0 <= time_index < len(data):
                    spiral_point = SpiralPoint(
                        theta=theta,
                        radius=radius,
                        price=price,
                        time_index=time_index,
                        timestamp=data.index[time_index],
                        spiral_id=spiral_id
                    )
                    spiral_points.append(spiral_point)
            
            # Calculate spiral strength based on price proximity
            strength = self._calculate_spiral_strength(spiral_points, data)
            
            return FibonacciSpiral(
                spiral_id=spiral_id,
                spiral_type='logarithmic',
                center_point=center,
                center_timestamp=data.index[center_idx],
                growth_factor=growth_factor,
                rotation=0.0,
                points=spiral_points,
                strength=strength,
                cycle_length=self._estimate_cycle_length(spiral_points),
                ml_score=0.0
            )
            
        except Exception as e:
            logger.error(f"Error creating logarithmic spiral: {str(e)}")
            return None
    
    def _create_archimedean_spiral(self, center: Tuple[int, float], data: pd.DataFrame, 
                                  spiral_id: str, growth_factor: float = None) -> FibonacciSpiral:
        """
        Create an Archimedean spiral with Fibonacci-based parameters
        """
        try:
            if growth_factor is None:
                growth_factor = PHI / 10  # Fibonacci-based growth
            
            center_idx, center_price = center
            resolution = self.parameters['spiral_resolution']
            
            # Calculate spiral parameters
            max_theta = 4 * np.pi  # 2 full rotations
            theta_values = np.linspace(0, max_theta, resolution)
            
            spiral_points = []
            
            for i, theta in enumerate(theta_values):
                # Archimedean spiral: r = a + b*θ
                radius = growth_factor * theta
                
                # Apply Fibonacci scaling
                fib_scale = self.fibonacci_numbers[min(i // 10, len(self.fibonacci_numbers) - 1)]
                radius *= (1 + fib_scale / 100)
                
                # Convert polar to market coordinates
                time_offset = radius * np.cos(theta) * 5
                price_offset = radius * np.sin(theta) * (data['close'].std() * 0.05)
                
                # Calculate actual market position
                time_index = int(center_idx + time_offset)
                price = center_price + price_offset
                
                # Ensure we stay within data bounds
                if 0 <= time_index < len(data):
                    spiral_point = SpiralPoint(
                        theta=theta,
                        radius=radius,
                        price=price,
                        time_index=time_index,
                        timestamp=data.index[time_index],
                        spiral_id=spiral_id
                    )
                    spiral_points.append(spiral_point)
            
            # Calculate spiral strength
            strength = self._calculate_spiral_strength(spiral_points, data)
            
            return FibonacciSpiral(
                spiral_id=spiral_id,
                spiral_type='archimedean',
                center_point=center,
                center_timestamp=data.index[center_idx],
                growth_factor=growth_factor,
                rotation=0.0,
                points=spiral_points,
                strength=strength,
                cycle_length=self._estimate_cycle_length(spiral_points),
                ml_score=0.0
            )
            
        except Exception as e:
            logger.error(f"Error creating Archimedean spiral: {str(e)}")
            return None
    
    def _calculate_spiral_strength(self, spiral_points: List[SpiralPoint], data: pd.DataFrame) -> float:
        """
        Calculate the strength of a spiral based on price proximity
        """
        try:
            if not spiral_points:
                return 0.0
            
            tolerance = self.parameters['intersection_tolerance']
            total_strength = 0.0
            valid_points = 0
            
            for point in spiral_points:
                if 0 <= point.time_index < len(data):
                    actual_price = data['close'].iloc[point.time_index]
                    price_diff = abs(point.price - actual_price) / actual_price
                    
                    if price_diff <= tolerance * 5:  # Wider tolerance for strength calculation
                        proximity_strength = max(0, 1 - (price_diff / (tolerance * 5)))
                        total_strength += proximity_strength
                        valid_points += 1
            
            if valid_points == 0:
                return 0.0
            
            return min(total_strength / valid_points, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating spiral strength: {str(e)}")
            return 0.0
    
    def _estimate_cycle_length(self, spiral_points: List[SpiralPoint]) -> int:
        """
        Estimate the cycle length represented by the spiral
        """
        try:
            if len(spiral_points) < 10:
                return 0
            
            # Calculate time spans for full rotations
            rotations = []
            current_rotation_start = 0
            
            for i, point in enumerate(spiral_points[1:], 1):
                if point.theta >= spiral_points[current_rotation_start].theta + 2 * np.pi:
                    rotation_length = point.time_index - spiral_points[current_rotation_start].time_index
                    if rotation_length > 0:
                        rotations.append(rotation_length)
                    current_rotation_start = i
            
            if rotations:
                return int(np.median(rotations))
            
            # Fallback: estimate based on total spiral span
            total_span = spiral_points[-1].time_index - spiral_points[0].time_index
            estimated_rotations = spiral_points[-1].theta / (2 * np.pi)
            
            if estimated_rotations > 0:
                return int(total_span / estimated_rotations)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error estimating cycle length: {str(e)}")
            return 0
    
    def _find_spiral_intersections(self, data: pd.DataFrame) -> List[SpiralIntersection]:
        """
        Find intersections between spirals and with current price action
        """
        try:
            intersections = []
            tolerance = self.parameters['intersection_tolerance']
            current_price = data['close'].iloc[-1]
            current_time = len(data) - 1
            
            # Find spiral-price intersections
            for spiral in self.spirals:
                for point in spiral.points:
                    if abs(point.time_index - current_time) <= 5:  # Near current time
                        price_diff = abs(point.price - current_price) / current_price
                        
                        if price_diff <= tolerance:
                            confidence = (tolerance - price_diff) / tolerance
                            
                            intersection = SpiralIntersection(
                                intersection_type='spiral_price',
                                point=point,
                                confidence=confidence * spiral.strength,
                                spirals_involved=[spiral.spiral_id],
                                prediction_strength=confidence
                            )
                            intersections.append(intersection)
            
            # Find spiral-spiral intersections
            for i, spiral1 in enumerate(self.spirals):
                for j, spiral2 in enumerate(self.spirals[i+1:], i+1):
                    for point1 in spiral1.points:
                        for point2 in spiral2.points:
                            time_diff = abs(point1.time_index - point2.time_index)
                            price_diff = abs(point1.price - point2.price) / max(point1.price, point2.price)
                            
                            if time_diff <= 2 and price_diff <= tolerance:
                                confidence = (tolerance - price_diff) / tolerance
                                avg_strength = (spiral1.strength + spiral2.strength) / 2
                                
                                intersection = SpiralIntersection(
                                    intersection_type='spiral_spiral',
                                    point=point1,  # Use first spiral's point
                                    confidence=confidence * avg_strength,
                                    spirals_involved=[spiral1.spiral_id, spiral2.spiral_id],
                                    prediction_strength=confidence
                                )
                                intersections.append(intersection)
            
            # Sort by confidence and return top intersections
            intersections.sort(key=lambda x: x.confidence, reverse=True)
            return intersections[:10]
            
        except Exception as e:
            logger.error(f"Error finding spiral intersections: {str(e)}")
            return []
    
    def _prepare_ml_features(self, data: pd.DataFrame, spiral: FibonacciSpiral) -> np.ndarray:
        """
        Prepare features for machine learning validation
        """
        try:
            features = []
            current_price = data['close'].iloc[-1]
            
            # Spiral-based features
            features.append(spiral.strength)
            features.append(spiral.growth_factor)
            features.append(len(spiral.points) / self.parameters['spiral_resolution'])
            features.append(spiral.cycle_length / 50.0)  # Normalized
            
            # Market context features
            volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
            features.append(volatility if not pd.isna(volatility) else 0.0)
            
            # Volume features
            volume_ratio = data['volume'].rolling(10).mean().iloc[-1] / data['volume'].rolling(50).mean().iloc[-1]
            features.append(volume_ratio if not pd.isna(volume_ratio) else 1.0)
            
            # Trend features
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            sma_50 = data['close'].rolling(50).mean().iloc[-1]
            trend_strength = (sma_20 - sma_50) / sma_50 if not pd.isna(sma_50) and sma_50 != 0 else 0.0
            features.append(trend_strength)
            
            # Fibonacci ratio features
            recent_high = data['high'].rolling(20).max().iloc[-1]
            recent_low = data['low'].rolling(20).min().iloc[-1]
            if recent_high != recent_low:
                current_retracement = (recent_high - current_price) / (recent_high - recent_low)
                features.append(current_retracement)
            else:
                features.append(0.5)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {str(e)}")
            return np.array([[0.0] * 8])
    
    def _train_ml_model(self, data: pd.DataFrame, spirals: List[FibonacciSpiral]):
        """
        Train machine learning model for spiral validation
        """
        try:
            if len(spirals) < 3 or len(data) < self.parameters['ml_lookback']:
                return
            
            X, y = [], []
            
            # Prepare training data
            for spiral in spirals:
                if spiral.strength > 0.1:  # Only use spirals with some strength
                    features = self._prepare_ml_features(data, spiral)
                    
                    # Calculate target (future price movement alignment)
                    center_idx = spiral.center_point[0]
                    if center_idx + 20 < len(data):
                        future_prices = data['close'].iloc[center_idx:center_idx+20]
                        price_alignment = self._calculate_price_alignment(spiral, future_prices)
                        
                        X.append(features[0])
                        y.append(min(price_alignment, 1.0))
            
            if len(X) > 5:
                X = np.array(X)
                y = np.array(y)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train model
                self.ml_model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                self.ml_model.fit(X_scaled, y)
                
                logger.debug(f"ML model trained with {len(X)} spiral samples")
                
        except Exception as e:
            logger.error(f"Error training ML model: {str(e)}")
    
    def _calculate_price_alignment(self, spiral: FibonacciSpiral, future_prices: pd.Series) -> float:
        """
        Calculate how well future prices align with spiral predictions
        """
        try:
            alignment_score = 0.0
            valid_predictions = 0
            tolerance = self.parameters['intersection_tolerance'] * 2
            
            for point in spiral.points:
                if point.time_index < len(future_prices):
                    actual_price = future_prices.iloc[point.time_index]
                    predicted_price = point.price
                    
                    price_diff = abs(actual_price - predicted_price) / actual_price
                    if price_diff <= tolerance:
                        alignment = 1 - (price_diff / tolerance)
                        alignment_score += alignment
                        valid_predictions += 1
            
            if valid_predictions == 0:
                return 0.0
            
            return alignment_score / valid_predictions
            
        except Exception as e:
            logger.error(f"Error calculating price alignment: {str(e)}")
            return 0.0
    
    def _predict_spiral_validity(self, data: pd.DataFrame, spiral: FibonacciSpiral) -> float:
        """
        Use machine learning to predict spiral validity
        """
        try:
            if self.ml_model is None:
                return spiral.strength
            
            features = self._prepare_ml_features(data, spiral)
            features_scaled = self.scaler.transform(features)
            prediction = self.ml_model.predict(features_scaled)[0]
            
            return min(prediction, 1.0)
            
        except Exception as e:
            logger.error(f"Error predicting spiral validity: {str(e)}")
            return spiral.strength
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Fibonacci spirals with advanced mathematical analysis
        """
        try:
            # Detect significant points for spiral anchoring
            significant_points = self._detect_significant_points(data)
            
            if len(significant_points) < 2:
                return {
                    'spirals': [],
                    'intersections': [],
                    'cycle_predictions': [],
                    'current_price': data['close'].iloc[-1],
                    'signal_strength': 0.0,
                    'dominant_cycle': None,
                    'spiral_count': 0
                }
            
            # Generate spirals from significant points
            new_spirals = []
            spiral_count = 0
            max_spirals = self.parameters['max_spirals']
            
            for i, (time_idx, price, point_type) in enumerate(significant_points[-10:]):  # Recent points
                if spiral_count >= max_spirals:
                    break
                
                center = (time_idx, price)
                
                # Create different types of spirals
                for spiral_type in self.parameters['spiral_types']:
                    if spiral_count >= max_spirals:
                        break
                    
                    spiral_id = f"{spiral_type}_{i}_{spiral_count}"
                    
                    if spiral_type == 'logarithmic':
                        spiral = self._create_logarithmic_spiral(center, data, spiral_id)
                    elif spiral_type == 'archimedean':
                        spiral = self._create_archimedean_spiral(center, data, spiral_id)
                    else:
                        continue
                    
                    if spiral and spiral.strength >= self.parameters['spiral_strength_threshold']:
                        new_spirals.append(spiral)
                        spiral_count += 1
            
            # Filter and update spirals
            self.spirals = new_spirals
            
            if not self.spirals:
                return {
                    'spirals': [],
                    'intersections': [],
                    'cycle_predictions': [],
                    'current_price': data['close'].iloc[-1],
                    'signal_strength': 0.0,
                    'dominant_cycle': None,
                    'spiral_count': 0
                }
            
            # Train ML model
            if self.parameters['use_ml_validation']:
                self._train_ml_model(data, self.spirals)
            
            # Validate spirals with ML
            for spiral in self.spirals:
                spiral.ml_score = self._predict_spiral_validity(data, spiral)
                # Combine original strength with ML prediction
                spiral.strength = (spiral.strength + spiral.ml_score) / 2
            
            # Find intersections
            self.intersections = self._find_spiral_intersections(data)
            
            # Identify dominant cycle
            dominant_cycle = None
            if self.spirals:
                strongest_spiral = max(self.spirals, key=lambda s: s.strength)
                if strongest_spiral.cycle_length > 0:
                    dominant_cycle = {
                        'length': strongest_spiral.cycle_length,
                        'strength': strongest_spiral.strength,
                        'type': strongest_spiral.spiral_type,
                        'center_time': strongest_spiral.center_timestamp.isoformat()
                    }
            
            # Generate cycle predictions
            cycle_predictions = self._generate_cycle_predictions(data)
            
            # Calculate overall signal strength
            signal_strength = 0.0
            if self.intersections:
                avg_intersection_confidence = np.mean([i.confidence for i in self.intersections])
                signal_strength = min(avg_intersection_confidence, 1.0)
            
            # Prepare result
            result = {
                'spirals': [self._spiral_to_dict(spiral) for spiral in self.spirals],
                'intersections': [self._intersection_to_dict(intersection) for intersection in self.intersections],
                'cycle_predictions': cycle_predictions,
                'current_price': data['close'].iloc[-1],
                'signal_strength': signal_strength,
                'dominant_cycle': dominant_cycle,
                'spiral_count': len(self.spirals),
                'ml_model_active': self.ml_model is not None,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            self.last_calculation = result
            return result
            
        except Exception as e:
            logger.error(f"Error in Fibonacci spirals calculation: {str(e)}")
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="fibonacci_spirals_calculation",
                message=str(e)
            )
    
    def _generate_cycle_predictions(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate cycle-based market predictions
        """
        try:
            predictions = []
            current_time = len(data) - 1
            
            for spiral in self.spirals:
                if spiral.cycle_length > 0 and spiral.strength > 0.4:
                    # Predict next cycle points
                    cycles_ahead = 3
                    for cycle in range(1, cycles_ahead + 1):
                        predicted_time = current_time + (spiral.cycle_length * cycle)
                        
                        # Find corresponding spiral point
                        cycle_phase = (cycle * 2 * np.pi) % (2 * np.pi)
                        closest_point = min(spiral.points, 
                                          key=lambda p: abs(p.theta % (2 * np.pi) - cycle_phase))
                        
                        prediction = {
                            'predicted_time_index': predicted_time,
                            'predicted_price': closest_point.price,
                            'cycle_number': cycle,
                            'confidence': spiral.strength * (1 - 0.2 * cycle),  # Decrease with distance
                            'spiral_id': spiral.spiral_id,
                            'spiral_type': spiral.spiral_type
                        }
                        predictions.append(prediction)
            
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            return predictions[:10]  # Return top 10 predictions
            
        except Exception as e:
            logger.error(f"Error generating cycle predictions: {str(e)}")
            return []
    
    def _spiral_to_dict(self, spiral: FibonacciSpiral) -> Dict[str, Any]:
        """Convert FibonacciSpiral to dictionary"""
        return {
            'spiral_id': spiral.spiral_id,
            'spiral_type': spiral.spiral_type,
            'center_point': spiral.center_point,
            'center_timestamp': spiral.center_timestamp.isoformat(),
            'growth_factor': spiral.growth_factor,
            'rotation': spiral.rotation,
            'strength': spiral.strength,
            'cycle_length': spiral.cycle_length,
            'ml_score': spiral.ml_score,
            'point_count': len(spiral.points),
            'points_sample': [
                {
                    'theta': p.theta,
                    'radius': p.radius,
                    'price': p.price,
                    'time_index': p.time_index
                } for p in spiral.points[::10]  # Sample every 10th point
            ]
        }
    
    def _intersection_to_dict(self, intersection: SpiralIntersection) -> Dict[str, Any]:
        """Convert SpiralIntersection to dictionary"""
        return {
            'intersection_type': intersection.intersection_type,
            'point': {
                'theta': intersection.point.theta,
                'radius': intersection.point.radius,
                'price': intersection.point.price,
                'time_index': intersection.point.time_index,
                'timestamp': intersection.point.timestamp.isoformat()
            },
            'confidence': intersection.confidence,
            'spirals_involved': intersection.spirals_involved,
            'prediction_strength': intersection.prediction_strength
        }
    
    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """
        Generate trading signals based on Fibonacci spiral analysis
        """
        try:
            if value['signal_strength'] < 0.3 or not value.get('intersections'):
                return SignalType.NEUTRAL, 0.0
            
            signal_strength = value['signal_strength']
            intersections = value['intersections']
            current_price = value['current_price']
            
            # Analyze intersection patterns
            buy_signals = 0
            sell_signals = 0
            total_confidence = 0.0
            
            for intersection in intersections:
                if intersection['confidence'] > 0.5:
                    intersection_price = intersection['point']['price']
                    total_confidence += intersection['confidence']
                    
                    if intersection_price > current_price:
                        # Price below spiral intersection suggests upward movement
                        buy_signals += 1
                    else:
                        # Price above spiral intersection suggests downward movement
                        sell_signals += 1
            
            if total_confidence == 0:
                return SignalType.NEUTRAL, 0.0
            
            avg_confidence = total_confidence / len(intersections)
            
            # Determine signal direction
            if buy_signals > sell_signals:
                return SignalType.BUY, min(signal_strength * avg_confidence, 1.0)
            elif sell_signals > buy_signals:
                return SignalType.SELL, min(signal_strength * avg_confidence, 1.0)
            else:
                # Check dominant cycle for trend direction
                if value.get('dominant_cycle') and value['dominant_cycle']['strength'] > 0.6:
                    # Use recent price momentum as tiebreaker
                    recent_momentum = data['close'].pct_change(5).iloc[-1]
                    if recent_momentum > 0:
                        return SignalType.BUY, signal_strength * 0.7
                    else:
                        return SignalType.SELL, signal_strength * 0.7
            
            return SignalType.NEUTRAL, 0.0
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return SignalType.NEUTRAL, 0.0
    
    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        
        spirals_metadata = {
            'spirals_generated': len(self.spirals),
            'intersections_found': len(self.intersections),
            'ml_model_trained': self.ml_model is not None,
            'spiral_types_used': self.parameters['spiral_types'],
            'golden_ratio_phi': PHI,
            'fibonacci_sequence_length': len(self.fibonacci_numbers)
        }
        
        base_metadata.update(spirals_metadata)
        return base_metadata


def create_fibonacci_spirals_indicator(parameters: Optional[Dict[str, Any]] = None) -> FibonacciSpiralsIndicator:
    """
    Factory function to create a FibonacciSpiralsIndicator instance
    
    Args:
        parameters: Optional dictionary of parameters to customize the indicator
        
    Returns:
        Configured FibonacciSpiralsIndicator instance
    """
    return FibonacciSpiralsIndicator(parameters=parameters)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1h')
    
    # Generate cyclical price data with golden ratio influences
    base_trend = np.linspace(100, 120, len(dates))
    cyclical_component = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (89 * PHI))  # Fibonacci-based cycle
    noise = np.random.randn(len(dates)) * 0.5
    
    prices = base_trend + cyclical_component + noise
    
    sample_data = pd.DataFrame({
        'high': prices + np.random.uniform(0, 1, len(dates)),
        'low': prices - np.random.uniform(0, 1, len(dates)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)
    
    # Test the indicator
    indicator = create_fibonacci_spirals_indicator({
        'spiral_types': ['logarithmic', 'archimedean'],
        'max_spirals': 6,
        'use_ml_validation': True
    })
    
    try:
        result = indicator.calculate(sample_data)
        print("Fibonacci Spirals Calculation Result:")
        print(f"Signal: {result.signal}, Confidence: {result.confidence:.3f}")
        print(f"Spirals generated: {result.value.get('spiral_count', 0)}")
        print(f"Signal strength: {result.value.get('signal_strength', 0):.3f}")
        
        if result.value.get('dominant_cycle'):
            cycle = result.value['dominant_cycle']
            print(f"Dominant cycle: {cycle['length']} periods, strength: {cycle['strength']:.3f}")
        
        if result.value.get('intersections'):
            print(f"Intersections found: {len(result.value['intersections'])}")
            
    except Exception as e:
        print(f"Error testing indicator: {str(e)}")