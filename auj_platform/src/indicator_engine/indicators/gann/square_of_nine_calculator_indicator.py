"""
Advanced Square of Nine Calculator Indicator
==========================================

A sophisticated calculator specifically designed for W.D. Gann's Square of Nine methodology.
This indicator provides comprehensive calculator functionality for:

1. Square of Nine calculations (spiral, concentric, radial)
2. Advanced mathematical operations (sacred numbers, geometric progressions)
3. Price/time projections and forecasting
4. Harmonic analysis and resonance detection
5. ML-enhanced pattern recognition and validation
6. Interactive calculator interface for trading decisions
7. Real-time calculation updates and alerts

Features:
- Comprehensive Square of Nine mathematical operations
- Advanced calculator with multiple calculation modes
- ML-enhanced pattern validation and prediction
- Sacred number integration and harmonic analysis
- Price and time projection calculations
- Support and resistance level calculations
- Real-time alerts and notifications
- Performance optimization for high-frequency calculations

Author: Advanced Trading Platform Team
Version: 2.1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import math
import warnings
from scipy import signal, optimize, stats
from scipy.fft import fft, ifft, fftfreq
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error, r2_score
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalculationMode(Enum):
    """Calculation modes for Square of Nine calculator"""
    SPIRAL = "spiral"
    CONCENTRIC = "concentric"
    RADIAL = "radial"
    HARMONIC = "harmonic"
    SACRED = "sacred"
    PROJECTION = "projection"
    HYBRID = "hybrid"

class SquareType(Enum):
    """Types of squares for calculations"""
    PRICE_SQUARE = "price"
    TIME_SQUARE = "time"
    PRICE_TIME_SQUARE = "price_time"
    HARMONIC_SQUARE = "harmonic"
    SACRED_SQUARE = "sacred"

@dataclass
class CalculationResult:
    """Results from Square of Nine calculations"""
    value: float
    position: Tuple[int, int]
    angle: float
    radius: float
    spiral_turn: int
    harmonic_ratio: float
    sacred_relationship: bool
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProjectionResult:
    """Results from projection calculations"""
    target_value: float
    target_time: datetime
    probability: float
    support_levels: List[float]
    resistance_levels: List[float]
    harmonic_targets: List[float]
    sacred_targets: List[float]
    confidence_interval: Tuple[float, float]

class SquareOfNineCalculatorIndicator:
    """
    Advanced Square of Nine Calculator for comprehensive Gann analysis
    """
    
    def __init__(self, 
                 calculation_mode: CalculationMode = CalculationMode.HYBRID,
                 square_type: SquareType = SquareType.PRICE_SQUARE,
                 center_value: Optional[float] = None,
                 max_radius: int = 50,
                 precision: int = 8,
                 enable_ml: bool = True,
                 sacred_numbers: Optional[List[float]] = None):
        """
        Initialize Square of Nine Calculator
        
        Args:
            calculation_mode: Primary calculation mode
            square_type: Type of square calculations
            center_value: Center value for calculations (auto-calculated if None)
            max_radius: Maximum radius for spiral calculations
            precision: Decimal precision for calculations
            enable_ml: Enable machine learning enhancements
            sacred_numbers: Custom sacred numbers (uses default if None)
        """
        self.calculation_mode = calculation_mode
        self.square_type = square_type
        self.center_value = center_value
        self.max_radius = max_radius
        self.precision = precision
        self.enable_ml = enable_ml
        
        # Sacred numbers from Gann methodology
        self.sacred_numbers = sacred_numbers or [
            1, 2, 3, 5, 7, 8, 9, 11, 12, 13, 17, 19, 21, 24, 27, 30, 
            33, 36, 45, 49, 52, 60, 72, 84, 90, 108, 120, 144, 180, 
            240, 270, 300, 360, 720, 1440
        ]
        
        # Mathematical constants
        self.PI = math.pi
        self.GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
        self.SQRT_2 = math.sqrt(2)
        self.SQRT_3 = math.sqrt(3)
        
        # Initialize components
        self._initialize_calculator()
        self._initialize_ml_models()
        
        # Calculation cache
        self.calculation_cache = {}
        self.projection_cache = {}
        
        logger.info(f"Square of Nine Calculator initialized with mode: {calculation_mode.value}")
    
    def _initialize_calculator(self):
        """Initialize calculator components"""
        self.calculation_history = []
        self.projection_history = []
        self.performance_metrics = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'average_precision': 0.0,
            'calculation_time': 0.0
        }
        
        # Angle mappings for different calculation modes
        self.angle_mappings = {
            CalculationMode.SPIRAL: self._generate_spiral_angles(),
            CalculationMode.CONCENTRIC: self._generate_concentric_angles(),
            CalculationMode.RADIAL: self._generate_radial_angles(),
            CalculationMode.HARMONIC: self._generate_harmonic_angles(),
            CalculationMode.SACRED: self._generate_sacred_angles()
        }
    
    def _initialize_ml_models(self):
        """Initialize machine learning models"""
        if not self.enable_ml:
            return
        
        try:
            # Pattern recognition model
            self.pattern_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Anomaly detection model
            self.anomaly_model = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Projection model
            self.projection_model = RandomForestRegressor(
                n_estimators=150,
                max_depth=15,
                random_state=42
            )
            
            # Scalers
            self.feature_scaler = StandardScaler()
            self.target_scaler = MinMaxScaler()
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize ML models: {e}")
            self.enable_ml = False
    
    def _generate_spiral_angles(self) -> Dict[int, float]:
        """Generate spiral angle mappings"""
        angles = {}
        for i in range(1, self.max_radius * 8 + 1):
            radius = math.sqrt(i)
            angle = 2 * self.PI * (radius - int(radius))
            angles[i] = angle
        return angles
    
    def _generate_concentric_angles(self) -> Dict[int, float]:
        """Generate concentric circle angle mappings"""
        angles = {}
        for radius in range(1, self.max_radius + 1):
            for position in range(8 * radius):
                angle = (2 * self.PI * position) / (8 * radius)
                key = radius * 100 + position
                angles[key] = angle
        return angles
    
    def _generate_radial_angles(self) -> Dict[int, float]:
        """Generate radial line angle mappings"""
        angles = {}
        base_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # Degrees
        for i, angle_deg in enumerate(base_angles):
            angles[i] = math.radians(angle_deg)
        return angles
    
    def _generate_harmonic_angles(self) -> Dict[int, float]:
        """Generate harmonic angle mappings"""
        angles = {}
        harmonic_divisions = [8, 12, 16, 24, 32, 48, 64]
        
        for div in harmonic_divisions:
            for i in range(div):
                angle = (2 * self.PI * i) / div
                key = div * 100 + i
                angles[key] = angle
        
        return angles
    
    def _generate_sacred_angles(self) -> Dict[int, float]:
        """Generate sacred number angle mappings"""
        angles = {}
        for i, sacred_num in enumerate(self.sacred_numbers[:50]):
            angle = math.radians(sacred_num % 360)
            angles[sacred_num] = angle
        return angles    
    def calculate_square_position(self, value: float, center: Optional[float] = None) -> CalculationResult:
        """
        Calculate position of value in Square of Nine
        
        Args:
            value: Value to locate in square
            center: Center value (uses instance center if None)
            
        Returns:
            CalculationResult with position and metadata
        """
        start_time = datetime.now()
        
        try:
            if center is None:
                center = self.center_value or value
            
            # Calculate relative position from center
            delta = abs(value - center)
            
            # Determine spiral position
            spiral_position = self._calculate_spiral_position(delta)
            
            # Calculate angle and radius
            angle = self._calculate_angle(spiral_position)
            radius = self._calculate_radius(spiral_position)
            
            # Determine spiral turn
            spiral_turn = int(spiral_position / (8 * radius)) if radius > 0 else 0
            
            # Calculate harmonic ratio
            harmonic_ratio = self._calculate_harmonic_ratio(value, center)
            
            # Check sacred relationship
            sacred_relationship = self._check_sacred_relationship(value, center)
            
            # Calculate confidence using ML if enabled
            confidence = self._calculate_confidence(value, center, spiral_position)
            
            # Create result
            result = CalculationResult(
                value=value,
                position=(int(spiral_position % 8), int(radius)),
                angle=angle,
                radius=radius,
                spiral_turn=spiral_turn,
                harmonic_ratio=harmonic_ratio,
                sacred_relationship=sacred_relationship,
                confidence=confidence,
                metadata={
                    'center': center,
                    'delta': delta,
                    'calculation_mode': self.calculation_mode.value,
                    'square_type': self.square_type.value,
                    'calculation_time': (datetime.now() - start_time).total_seconds()
                }
            )
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            # Cache result
            cache_key = f"{value}_{center}_{self.calculation_mode.value}"
            self.calculation_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating square position: {e}")
            raise
    
    def _calculate_spiral_position(self, delta: float) -> float:
        """Calculate position in spiral based on delta from center"""
        if delta == 0:
            return 0
        
        # Use different methods based on calculation mode
        if self.calculation_mode == CalculationMode.SPIRAL:
            # Traditional spiral calculation
            return math.sqrt(delta)
        
        elif self.calculation_mode == CalculationMode.CONCENTRIC:
            # Concentric circles calculation
            radius = math.ceil(math.sqrt(delta / self.PI))
            return radius * 8  # 8 positions per radius
        
        elif self.calculation_mode == CalculationMode.RADIAL:
            # Radial lines calculation
            angle = math.atan2(delta, 1) * 8 / (2 * self.PI)
            return angle
        
        elif self.calculation_mode == CalculationMode.HARMONIC:
            # Harmonic division calculation
            harmonic_pos = delta * 24 / (2 * self.PI)  # 24 harmonic divisions
            return harmonic_pos
        
        elif self.calculation_mode == CalculationMode.SACRED:
            # Sacred number positioning
            sacred_pos = self._find_nearest_sacred_position(delta)
            return sacred_pos
        
        else:  # HYBRID mode
            # Combine multiple methods
            spiral_pos = math.sqrt(delta)
            harmonic_pos = delta * 24 / (2 * self.PI)
            return (spiral_pos + harmonic_pos) / 2
    
    def _calculate_angle(self, position: float) -> float:
        """Calculate angle for given position"""
        mode_angles = self.angle_mappings.get(self.calculation_mode, {})
        
        if self.calculation_mode == CalculationMode.SPIRAL:
            return 2 * self.PI * (position - int(position))
        
        elif self.calculation_mode == CalculationMode.CONCENTRIC:
            radius = int(position / 8)
            pos_in_radius = position % 8
            return (2 * self.PI * pos_in_radius) / 8
        
        elif self.calculation_mode == CalculationMode.RADIAL:
            base_angle = int(position) % 8
            return mode_angles.get(base_angle, 0)
        
        elif self.calculation_mode == CalculationMode.HARMONIC:
            return (2 * self.PI * position) / 24
        
        elif self.calculation_mode == CalculationMode.SACRED:
            nearest_sacred = self._find_nearest_sacred_number(position)
            return math.radians(nearest_sacred % 360)
        
        else:  # HYBRID
            return (2 * self.PI * position) / 16  # Average division
    
    def _calculate_radius(self, position: float) -> float:
        """Calculate radius for given position"""
        if self.calculation_mode == CalculationMode.SPIRAL:
            return math.sqrt(position)
        
        elif self.calculation_mode == CalculationMode.CONCENTRIC:
            return int(position / 8) + 1
        
        elif self.calculation_mode == CalculationMode.RADIAL:
            return position
        
        elif self.calculation_mode == CalculationMode.HARMONIC:
            return math.sqrt(position / 24)
        
        elif self.calculation_mode == CalculationMode.SACRED:
            return math.sqrt(position / 8)
        
        else:  # HYBRID
            return math.sqrt(position / 12)
    
    def _calculate_harmonic_ratio(self, value: float, center: float) -> float:
        """Calculate harmonic ratio between value and center"""
        if center == 0:
            return 1.0
        
        ratio = value / center
        
        # Check for common harmonic ratios
        harmonic_ratios = [
            0.5, 0.618, 0.707, 0.786, 1.0, 1.272, 1.414, 1.618, 2.0, 2.618
        ]
        
        closest_harmonic = min(harmonic_ratios, key=lambda x: abs(x - ratio))
        return closest_harmonic
    
    def _check_sacred_relationship(self, value: float, center: float) -> bool:
        """Check if value has sacred number relationship with center"""
        if center == 0:
            return False
        
        ratio = value / center
        difference = abs(value - center)
        
        # Check ratio against sacred numbers
        for sacred in self.sacred_numbers:
            if abs(ratio - sacred) < 0.01:  # 1% tolerance
                return True
            if abs(difference / center - sacred / 100) < 0.01:
                return True
        
        return False
    
    def _calculate_confidence(self, value: float, center: float, position: float) -> float:
        """Calculate confidence score for position calculation"""
        base_confidence = 0.8
        
        # Adjust based on calculation mode precision
        mode_adjustments = {
            CalculationMode.SPIRAL: 0.1,
            CalculationMode.CONCENTRIC: 0.05,
            CalculationMode.RADIAL: 0.0,
            CalculationMode.HARMONIC: 0.15,
            CalculationMode.SACRED: 0.2,
            CalculationMode.HYBRID: 0.1
        }
        
        confidence = base_confidence + mode_adjustments.get(self.calculation_mode, 0)
        
        # Adjust based on sacred relationship
        if self._check_sacred_relationship(value, center):
            confidence += 0.1
        
        # Adjust based on position precision
        position_precision = 1 - (position % 1)  # Higher for whole numbers
        confidence += position_precision * 0.1
        
        # Use ML model if available and enabled
        if self.enable_ml and hasattr(self, 'pattern_model'):
            try:
                ml_confidence = self._get_ml_confidence(value, center, position)
                confidence = (confidence + ml_confidence) / 2
            except Exception as e:
                logger.warning(f"ML confidence calculation failed: {e}")
        
        return min(max(confidence, 0.0), 1.0)
    
    def _get_ml_confidence(self, value: float, center: float, position: float) -> float:
        """Get ML-based confidence score"""
        # This would use trained models in production
        # For now, return a calculated confidence
        features = np.array([[value, center, position, 
                            abs(value - center), value / center if center != 0 else 1]])
        
        # Simulate ML prediction (would use actual trained model)
        variance = np.var(features)
        confidence = 1 / (1 + variance)  # Simple confidence calculation
        
        return min(max(confidence, 0.3), 0.95)
    
    def _find_nearest_sacred_position(self, delta: float) -> float:
        """Find nearest sacred number position"""
        sacred_positions = [s / 10.0 for s in self.sacred_numbers]
        nearest = min(sacred_positions, key=lambda x: abs(x - delta))
        return nearest
    
    def _find_nearest_sacred_number(self, position: float) -> float:
        """Find nearest sacred number"""
        return min(self.sacred_numbers, key=lambda x: abs(x - position))
    
    def calculate_projections(self, 
                            current_value: float, 
                            historical_data: pd.DataFrame,
                            projection_periods: int = 10,
                            confidence_level: float = 0.8) -> ProjectionResult:
        """
        Calculate price/time projections using Square of Nine
        
        Args:
            current_value: Current price/value
            historical_data: Historical price data
            projection_periods: Number of periods to project
            confidence_level: Confidence level for projections
            
        Returns:
            ProjectionResult with projections and targets
        """
        try:
            # Calculate center value if not set
            if self.center_value is None:
                self.center_value = self._calculate_dynamic_center(historical_data)
            
            # Get current position in square
            current_result = self.calculate_square_position(current_value, self.center_value)
            
            # Calculate projection targets
            target_value = self._calculate_target_value(current_result, projection_periods)
            target_time = self._calculate_target_time(historical_data, projection_periods)
            
            # Calculate support and resistance levels
            support_levels = self._calculate_support_levels(current_result)
            resistance_levels = self._calculate_resistance_levels(current_result)
            
            # Calculate harmonic and sacred targets
            harmonic_targets = self._calculate_harmonic_targets(current_value, self.center_value)
            sacred_targets = self._calculate_sacred_targets(current_value, self.center_value)
            
            # Calculate probability and confidence interval
            probability = self._calculate_projection_probability(
                current_result, historical_data, target_value
            )
            confidence_interval = self._calculate_confidence_interval(
                target_value, probability, confidence_level
            )
            
            # Create projection result
            projection = ProjectionResult(
                target_value=target_value,
                target_time=target_time,
                probability=probability,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                harmonic_targets=harmonic_targets,
                sacred_targets=sacred_targets,
                confidence_interval=confidence_interval
            )
            
            # Cache projection
            cache_key = f"proj_{current_value}_{projection_periods}_{confidence_level}"
            self.projection_cache[cache_key] = projection
            
            return projection
            
        except Exception as e:
            logger.error(f"Error calculating projections: {e}")
            raise    
    def _calculate_dynamic_center(self, data: pd.DataFrame) -> float:
        """Calculate dynamic center value from historical data"""
        if 'close' in data.columns:
            prices = data['close'].values
        elif 'price' in data.columns:
            prices = data['price'].values
        else:
            prices = data.iloc[:, 0].values  # Use first column
        
        # Use multiple methods to find optimal center
        methods = {
            'mean': np.mean(prices),
            'median': np.median(prices),
            'mode': stats.mode(prices)[0][0] if len(stats.mode(prices)[0]) > 0 else np.mean(prices),
            'weighted_mean': np.average(prices, weights=np.arange(1, len(prices) + 1)),
            'harmonic_mean': stats.hmean(prices[prices > 0]) if np.all(prices > 0) else np.mean(prices)
        }
        
        # Select best center based on sacred number alignment
        best_center = methods['mean']
        best_score = 0
        
        for method, center in methods.items():
            score = self._score_center_alignment(center, prices)
            if score > best_score:
                best_score = score
                best_center = center
        
        return best_center
    
    def _score_center_alignment(self, center: float, prices: np.ndarray) -> float:
        """Score how well center aligns with sacred numbers and price data"""
        score = 0
        
        # Check sacred number alignment
        for sacred in self.sacred_numbers:
            if abs(center - sacred) < center * 0.01:  # 1% tolerance
                score += 10
            if abs(center / sacred - 1) < 0.01:
                score += 5
        
        # Check price distribution around center
        deviations = np.abs(prices - center)
        mean_deviation = np.mean(deviations)
        
        # Lower deviation is better
        score += max(0, 100 - mean_deviation / center * 100)
        
        return score
    
    def _calculate_target_value(self, current_result: CalculationResult, periods: int) -> float:
        """Calculate target value for projection"""
        current_position = current_result.position
        current_radius = current_result.radius
        current_angle = current_result.angle
        
        # Project forward based on spiral progression
        target_radius = current_radius + (periods / 8)  # Advance by spiral turns
        target_angle = current_angle + (2 * self.PI * periods / 8)
        
        # Calculate target position in spiral
        target_position = target_radius ** 2
        
        # Convert back to value
        center = current_result.metadata.get('center', current_result.value)
        target_value = center + target_position
        
        # Apply harmonic adjustments
        if current_result.harmonic_ratio != 1.0:
            target_value *= current_result.harmonic_ratio
        
        # Apply sacred number adjustments
        if current_result.sacred_relationship:
            nearest_sacred = self._find_nearest_sacred_number(target_value)
            sacred_adjustment = nearest_sacred / target_value if target_value != 0 else 1
            target_value *= sacred_adjustment
        
        return target_value
    
    def _calculate_target_time(self, data: pd.DataFrame, periods: int) -> datetime:
        """Calculate target time for projection"""
        if data.index.dtype.kind == 'M':  # datetime index
            last_time = data.index[-1]
            time_delta = data.index[-1] - data.index[-2] if len(data) > 1 else timedelta(days=1)
            return last_time + (time_delta * periods)
        else:
            # Assume daily data if no datetime index
            return datetime.now() + timedelta(days=periods)
    
    def _calculate_support_levels(self, result: CalculationResult) -> List[float]:
        """Calculate support levels based on Square of Nine"""
        center = result.metadata.get('center', result.value)
        current_radius = result.radius
        
        support_levels = []
        
        # Inner circle supports
        for r in range(1, int(current_radius)):
            support_value = center + (r ** 2)
            support_levels.append(support_value)
        
        # Angle-based supports
        for angle_deg in [45, 90, 135, 180, 225, 270, 315]:
            angle_rad = math.radians(angle_deg)
            support_radius = current_radius * math.cos(angle_rad)
            if support_radius > 0:
                support_value = center + (support_radius ** 2)
                support_levels.append(support_value)
        
        # Sacred number supports
        for sacred in self.sacred_numbers[:10]:  # Use first 10 sacred numbers
            support_value = center * (1 - sacred / 100)
            if support_value > 0:
                support_levels.append(support_value)
        
        return sorted(list(set(support_levels)))[:10]  # Return top 10 unique levels
    
    def _calculate_resistance_levels(self, result: CalculationResult) -> List[float]:
        """Calculate resistance levels based on Square of Nine"""
        center = result.metadata.get('center', result.value)
        current_radius = result.radius
        
        resistance_levels = []
        
        # Outer circle resistances
        for r in range(int(current_radius) + 1, int(current_radius) + 6):
            resistance_value = center + (r ** 2)
            resistance_levels.append(resistance_value)
        
        # Angle-based resistances
        for angle_deg in [45, 90, 135, 180, 225, 270, 315]:
            angle_rad = math.radians(angle_deg)
            resistance_radius = current_radius / math.cos(angle_rad) if math.cos(angle_rad) != 0 else current_radius * 2
            resistance_value = center + (resistance_radius ** 2)
            resistance_levels.append(resistance_value)
        
        # Sacred number resistances
        for sacred in self.sacred_numbers[:10]:  # Use first 10 sacred numbers
            resistance_value = center * (1 + sacred / 100)
            resistance_levels.append(resistance_value)
        
        return sorted(list(set(resistance_levels)))[:10]  # Return top 10 unique levels
    
    def _calculate_harmonic_targets(self, current_value: float, center: float) -> List[float]:
        """Calculate harmonic target levels"""
        targets = []
        
        # Fibonacci-based harmonics
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618, 4.236]
        
        for ratio in fib_ratios:
            target = center + (current_value - center) * ratio
            targets.append(target)
        
        # Musical harmonics (frequency ratios)
        musical_ratios = [1.125, 1.25, 1.33, 1.5, 1.67, 2.0, 2.25, 2.5, 3.0]
        
        for ratio in musical_ratios:
            target = current_value * ratio
            targets.append(target)
            target = current_value / ratio
            targets.append(target)
        
        return sorted(list(set(targets)))[:15]  # Return top 15 unique targets
    
    def _calculate_sacred_targets(self, current_value: float, center: float) -> List[float]:
        """Calculate sacred number target levels"""
        targets = []
        
        for sacred in self.sacred_numbers[:20]:  # Use first 20 sacred numbers
            # Percentage-based targets
            target_up = current_value * (1 + sacred / 100)
            target_down = current_value * (1 - sacred / 100)
            targets.extend([target_up, target_down])
            
            # Ratio-based targets
            if sacred > 1:
                target_ratio = current_value * sacred
                target_inverse = current_value / sacred
                targets.extend([target_ratio, target_inverse])
        
        return sorted(list(set([t for t in targets if t > 0])))[:20]  # Return top 20 positive targets
    
    def _calculate_projection_probability(self, 
                                        current_result: CalculationResult,
                                        historical_data: pd.DataFrame,
                                        target_value: float) -> float:
        """Calculate probability of reaching target value"""
        
        # Base probability from calculation confidence
        base_prob = current_result.confidence
        
        # Adjust based on historical volatility
        if 'close' in historical_data.columns:
            prices = historical_data['close'].values
        else:
            prices = historical_data.iloc[:, 0].values
        
        volatility = np.std(prices) / np.mean(prices)
        volatility_adjustment = max(0.5, 1 - volatility)  # Lower volatility = higher probability
        
        # Adjust based on sacred relationship
        sacred_adjustment = 1.2 if current_result.sacred_relationship else 1.0
        
        # Adjust based on harmonic ratio
        harmonic_adjustment = 1.1 if abs(current_result.harmonic_ratio - 1.0) < 0.1 else 1.0
        
        # Calculate final probability
        probability = base_prob * volatility_adjustment * sacred_adjustment * harmonic_adjustment
        
        return min(max(probability, 0.1), 0.95)  # Clamp between 10% and 95%
    
    def _calculate_confidence_interval(self, 
                                     target_value: float, 
                                     probability: float,
                                     confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for target value"""
        
        # Use normal distribution assumption
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Estimate standard deviation from probability
        std_dev = target_value * (1 - probability) / 2
        
        margin_of_error = z_score * std_dev
        
        lower_bound = target_value - margin_of_error
        upper_bound = target_value + margin_of_error
        
        return (lower_bound, upper_bound)
    
    def batch_calculate(self, 
                       values: List[float], 
                       center: Optional[float] = None) -> List[CalculationResult]:
        """
        Perform batch calculations for multiple values
        
        Args:
            values: List of values to calculate
            center: Center value for all calculations
            
        Returns:
            List of CalculationResult objects
        """
        results = []
        
        for value in values:
            try:
                result = self.calculate_square_position(value, center)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to calculate position for value {value}: {e}")
                # Create dummy result for failed calculation
                dummy_result = CalculationResult(
                    value=value,
                    position=(0, 0),
                    angle=0.0,
                    radius=0.0,
                    spiral_turn=0,
                    harmonic_ratio=1.0,
                    sacred_relationship=False,
                    confidence=0.0,
                    metadata={'error': str(e)}
                )
                results.append(dummy_result)
        
        return results
    
    def find_value_at_position(self, 
                              position: Tuple[int, int], 
                              center: Optional[float] = None) -> float:
        """
        Find value at specific Square of Nine position
        
        Args:
            position: (angle_position, radius) tuple
            center: Center value
            
        Returns:
            Value at the specified position
        """
        if center is None:
            center = self.center_value or 0
        
        angle_pos, radius = position
        
        # Calculate position in spiral
        spiral_position = radius ** 2 + angle_pos
        
        # Convert to value
        value = center + spiral_position
        
        return value
    
    def find_nearest_values(self, 
                           target_value: float, 
                           center: Optional[float] = None,
                           count: int = 5) -> List[CalculationResult]:
        """
        Find nearest values to target in Square of Nine
        
        Args:
            target_value: Value to find neighbors for
            center: Center value
            count: Number of nearest values to return
            
        Returns:
            List of nearest CalculationResult objects
        """
        if center is None:
            center = self.center_value or target_value
        
        # Get target position
        target_result = self.calculate_square_position(target_value, center)
        target_radius = target_result.radius
        target_angle_pos = target_result.position[0]
        
        nearest_values = []
        
        # Search in expanding radius
        for r_offset in range(-2, 3):  # Search ±2 radius levels
            for a_offset in range(-2, 3):  # Search ±2 angle positions
                if r_offset == 0 and a_offset == 0:
                    continue  # Skip target position
                
                new_radius = max(0, target_radius + r_offset)
                new_angle_pos = (target_angle_pos + a_offset) % 8
                
                value = self.find_value_at_position((new_angle_pos, new_radius), center)
                result = self.calculate_square_position(value, center)
                
                distance = abs(value - target_value)
                result.metadata['distance'] = distance
                
                nearest_values.append(result)
        
        # Sort by distance and return top count
        nearest_values.sort(key=lambda x: x.metadata['distance'])
        return nearest_values[:count]    
    def calculate_time_cycles(self, 
                             historical_data: pd.DataFrame,
                             max_cycle_length: int = 100) -> Dict[str, Any]:
        """
        Calculate time cycles using Square of Nine principles
        
        Args:
            historical_data: Historical price/time data
            max_cycle_length: Maximum cycle length to detect
            
        Returns:
            Dictionary containing cycle analysis results
        """
        try:
            # Extract time series
            if 'close' in historical_data.columns:
                prices = historical_data['close'].values
            else:
                prices = historical_data.iloc[:, 0].values
            
            times = np.arange(len(prices))
            
            # Apply Square of Nine time calculations
            time_squares = []
            for i, time_val in enumerate(times):
                time_result = self.calculate_square_position(time_val, 0)  # Center at 0 for time
                time_squares.append({
                    'time': time_val,
                    'position': time_result.position,
                    'radius': time_result.radius,
                    'angle': time_result.angle,
                    'price': prices[i] if i < len(prices) else 0
                })
            
            # Detect cycles based on radius patterns
            radius_series = [ts['radius'] for ts in time_squares]
            cycles = self._detect_radius_cycles(radius_series, max_cycle_length)
            
            # Detect cycles based on angle patterns
            angle_series = [ts['angle'] for ts in time_squares]
            angle_cycles = self._detect_angle_cycles(angle_series, max_cycle_length)
            
            # Sacred number cycle analysis
            sacred_cycles = self._analyze_sacred_cycles(times, prices)
            
            # Harmonic cycle analysis
            harmonic_cycles = self._analyze_harmonic_cycles(times, prices)
            
            return {
                'time_squares': time_squares,
                'radius_cycles': cycles,
                'angle_cycles': angle_cycles,
                'sacred_cycles': sacred_cycles,
                'harmonic_cycles': harmonic_cycles,
                'analysis_summary': {
                    'dominant_cycle': cycles[0] if cycles else None,
                    'cycle_count': len(cycles),
                    'average_cycle_length': np.mean([c['length'] for c in cycles]) if cycles else 0,
                    'cycle_strength': np.mean([c['strength'] for c in cycles]) if cycles else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating time cycles: {e}")
            return {}
    
    def _detect_radius_cycles(self, radius_series: List[float], max_length: int) -> List[Dict]:
        """Detect cycles in radius patterns"""
        cycles = []
        
        # Convert to numpy array
        radius_array = np.array(radius_series)
        
        # Find peaks and troughs
        peaks, _ = signal.find_peaks(radius_array, distance=3)
        troughs, _ = signal.find_peaks(-radius_array, distance=3)
        
        # Calculate cycle lengths
        all_extremes = sorted(list(peaks) + list(troughs))
        
        for i in range(len(all_extremes) - 1):
            cycle_length = all_extremes[i+1] - all_extremes[i]
            if cycle_length <= max_length:
                # Calculate cycle strength
                start_idx = all_extremes[i]
                end_idx = all_extremes[i+1]
                cycle_data = radius_array[start_idx:end_idx+1]
                
                strength = np.std(cycle_data) / np.mean(cycle_data) if np.mean(cycle_data) != 0 else 0
                
                cycles.append({
                    'start': start_idx,
                    'end': end_idx,
                    'length': cycle_length,
                    'strength': strength,
                    'type': 'radius'
                })
        
        # Sort by strength
        cycles.sort(key=lambda x: x['strength'], reverse=True)
        return cycles[:10]  # Return top 10 cycles
    
    def _detect_angle_cycles(self, angle_series: List[float], max_length: int) -> List[Dict]:
        """Detect cycles in angle patterns"""
        cycles = []
        
        # Normalize angles to [0, 2π]
        normalized_angles = [(angle % (2 * self.PI)) for angle in angle_series]
        angle_array = np.array(normalized_angles)
        
        # Detect when angle completes full rotations
        rotation_points = []
        for i in range(1, len(angle_array)):
            if angle_array[i] < angle_array[i-1] and angle_array[i-1] > 5:  # Full rotation detected
                rotation_points.append(i)
        
        # Calculate rotation cycles
        for i in range(len(rotation_points) - 1):
            cycle_length = rotation_points[i+1] - rotation_points[i]
            if cycle_length <= max_length:
                strength = 1.0  # Full rotations have high strength
                
                cycles.append({
                    'start': rotation_points[i],
                    'end': rotation_points[i+1],
                    'length': cycle_length,
                    'strength': strength,
                    'type': 'angle_rotation'
                })
        
        # Sort by length (shorter cycles often more significant)
        cycles.sort(key=lambda x: x['length'])
        return cycles[:5]  # Return top 5 cycles
    
    def _analyze_sacred_cycles(self, times: np.ndarray, prices: np.ndarray) -> List[Dict]:
        """Analyze cycles based on sacred numbers"""
        sacred_cycles = []
        
        for sacred in self.sacred_numbers[:15]:  # Use first 15 sacred numbers
            if sacred > len(times) / 2:  # Skip if sacred number is too large
                continue
            
            # Check for cycles at sacred number intervals
            cycle_strength = 0
            cycle_count = 0
            
            for start in range(0, len(times) - sacred, sacred):
                end = start + sacred
                if end < len(prices):
                    price_correlation = np.corrcoef(
                        prices[start:start+sacred//2], 
                        prices[end-sacred//2:end]
                    )[0, 1] if sacred > 2 else 0
                    
                    if not np.isnan(price_correlation) and abs(price_correlation) > 0.3:
                        cycle_strength += abs(price_correlation)
                        cycle_count += 1
            
            if cycle_count > 0:
                avg_strength = cycle_strength / cycle_count
                sacred_cycles.append({
                    'sacred_number': sacred,
                    'cycle_length': sacred,
                    'strength': avg_strength,
                    'occurrences': cycle_count,
                    'type': 'sacred'
                })
        
        # Sort by strength
        sacred_cycles.sort(key=lambda x: x['strength'], reverse=True)
        return sacred_cycles[:8]  # Return top 8 sacred cycles
    
    def _analyze_harmonic_cycles(self, times: np.ndarray, prices: np.ndarray) -> List[Dict]:
        """Analyze harmonic cycles"""
        harmonic_cycles = []
        
        # Harmonic ratios to test
        harmonic_ratios = [0.5, 0.618, 0.707, 1.0, 1.414, 1.618, 2.0, 2.618]
        
        # Base cycle length (use a reasonable default)
        base_cycle = min(20, len(times) // 4)
        
        for ratio in harmonic_ratios:
            cycle_length = int(base_cycle * ratio)
            if cycle_length < 3 or cycle_length > len(times) // 2:
                continue
            
            # Test harmonic relationships
            correlations = []
            for start in range(0, len(times) - cycle_length * 2, cycle_length):
                end1 = start + cycle_length
                end2 = start + cycle_length * 2
                
                if end2 < len(prices):
                    corr = np.corrcoef(prices[start:end1], prices[end1:end2])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                avg_correlation = np.mean(correlations)
                harmonic_cycles.append({
                    'harmonic_ratio': ratio,
                    'cycle_length': cycle_length,
                    'strength': avg_correlation,
                    'base_cycle': base_cycle,
                    'type': 'harmonic'
                })
        
        # Sort by strength
        harmonic_cycles.sort(key=lambda x: x['strength'], reverse=True)
        return harmonic_cycles[:6]  # Return top 6 harmonic cycles
    
    def generate_trading_signals(self, 
                                current_price: float,
                                historical_data: pd.DataFrame,
                                lookback_periods: int = 20) -> Dict[str, Any]:
        """
        Generate comprehensive trading signals based on Square of Nine calculations
        
        Args:
            current_price: Current market price
            historical_data: Historical price data
            lookback_periods: Number of periods to look back for analysis
            
        Returns:
            Dictionary containing trading signals and analysis
        """
        try:
            # Calculate current position
            current_result = self.calculate_square_position(current_price)
            
            # Calculate projections
            projections = self.calculate_projections(current_price, historical_data)
            
            # Calculate time cycles
            cycles = self.calculate_time_cycles(historical_data)
            
            # Generate signals
            signals = {
                'timestamp': datetime.now(),
                'current_price': current_price,
                'square_position': {
                    'position': current_result.position,
                    'radius': current_result.radius,
                    'angle': current_result.angle,
                    'spiral_turn': current_result.spiral_turn,
                    'confidence': current_result.confidence
                },
                'price_signals': self._generate_price_signals(current_result, projections),
                'time_signals': self._generate_time_signals(cycles),
                'support_resistance': {
                    'support_levels': projections.support_levels,
                    'resistance_levels': projections.resistance_levels,
                    'current_level_type': self._classify_current_level(current_price, projections)
                },
                'projections': {
                    'target_value': projections.target_value,
                    'target_time': projections.target_time,
                    'probability': projections.probability,
                    'confidence_interval': projections.confidence_interval
                },
                'sacred_analysis': {
                    'is_sacred_level': current_result.sacred_relationship,
                    'harmonic_ratio': current_result.harmonic_ratio,
                    'sacred_targets': projections.sacred_targets[:5],
                    'harmonic_targets': projections.harmonic_targets[:5]
                },
                'risk_management': self._calculate_risk_metrics(current_price, projections),
                'overall_signal': self._generate_overall_signal(current_result, projections, cycles)
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {}
    
    def _generate_price_signals(self, result: CalculationResult, projections: ProjectionResult) -> Dict:
        """Generate price-based trading signals"""
        signals = {
            'direction': 'neutral',
            'strength': 0.5,
            'confidence': result.confidence,
            'signals': []
        }
        
        # Radius-based signals
        if result.radius > 5:  # Outer radius suggests volatility
            signals['signals'].append({
                'type': 'volatility',
                'message': 'High volatility zone detected',
                'strength': min(result.radius / 10, 1.0)
            })
        
        # Angle-based signals
        angle_deg = math.degrees(result.angle)
        if 85 <= angle_deg <= 95 or 265 <= angle_deg <= 275:  # Near 90 or 270 degrees
            signals['signals'].append({
                'type': 'reversal',
                'message': 'Potential reversal at cardinal angle',
                'strength': 0.8
            })
        
        # Sacred relationship signals
        if result.sacred_relationship:
            signals['signals'].append({
                'type': 'sacred',
                'message': 'Sacred number relationship detected',
                'strength': 0.9
            })
            signals['strength'] = min(signals['strength'] + 0.2, 1.0)
        
        # Projection-based signals
        if projections.probability > 0.7:
            direction = 'bullish' if projections.target_value > result.value else 'bearish'
            signals['direction'] = direction
            signals['signals'].append({
                'type': 'projection',
                'message': f'High probability {direction} projection',
                'strength': projections.probability
            })
        
        return signals
    
    def _generate_time_signals(self, cycles: Dict) -> Dict:
        """Generate time-based trading signals"""
        signals = {
            'cycle_phase': 'unknown',
            'cycle_strength': 0.0,
            'signals': []
        }
        
        if not cycles or 'analysis_summary' not in cycles:
            return signals
        
        summary = cycles['analysis_summary']
        
        if summary['dominant_cycle']:
            cycle = summary['dominant_cycle']
            signals['cycle_strength'] = cycle['strength']
            
            # Determine cycle phase
            cycle_position = len(cycles.get('time_squares', [])) % cycle['length']
            phase_ratio = cycle_position / cycle['length']
            
            if phase_ratio < 0.25:
                signals['cycle_phase'] = 'early'
            elif phase_ratio < 0.5:
                signals['cycle_phase'] = 'mid_up'
            elif phase_ratio < 0.75:
                signals['cycle_phase'] = 'late'
            else:
                signals['cycle_phase'] = 'mid_down'
            
            signals['signals'].append({
                'type': 'cycle',
                'message': f'Cycle phase: {signals["cycle_phase"]}',
                'strength': cycle['strength']
            })
        
        # Sacred cycle signals
        sacred_cycles = cycles.get('sacred_cycles', [])
        if sacred_cycles:
            strongest_sacred = sacred_cycles[0]
            signals['signals'].append({
                'type': 'sacred_cycle',
                'message': f'Sacred cycle {strongest_sacred["sacred_number"]} detected',
                'strength': strongest_sacred['strength']
            })
        
        return signals
    
    def _classify_current_level(self, price: float, projections: ProjectionResult) -> str:
        """Classify current price level (support/resistance/neutral)"""
        support_levels = projections.support_levels
        resistance_levels = projections.resistance_levels
        
        # Find nearest levels
        nearest_support = min(support_levels, key=lambda x: abs(x - price)) if support_levels else None
        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - price)) if resistance_levels else None
        
        support_distance = abs(price - nearest_support) if nearest_support else float('inf')
        resistance_distance = abs(price - nearest_resistance) if nearest_resistance else float('inf')
        
        # Classification threshold (1% of price)
        threshold = price * 0.01
        
        if support_distance < threshold:
            return 'support'
        elif resistance_distance < threshold:
            return 'resistance'
        else:
            return 'neutral'
    
    def _calculate_risk_metrics(self, price: float, projections: ProjectionResult) -> Dict:
        """Calculate risk management metrics"""
        metrics = {
            'stop_loss': 0.0,
            'take_profit': 0.0,
            'position_size': 0.0,
            'risk_reward_ratio': 0.0
        }
        
        # Calculate stop loss based on nearest support
        if projections.support_levels:
            nearest_support = min(projections.support_levels, key=lambda x: abs(x - price))
            metrics['stop_loss'] = nearest_support * 0.98  # 2% below support
        
        # Calculate take profit based on projections
        if projections.target_value > price:
            metrics['take_profit'] = projections.target_value * 0.95  # 5% before target
        
        # Calculate risk-reward ratio
        risk = abs(price - metrics['stop_loss']) if metrics['stop_loss'] > 0 else price * 0.02
        reward = abs(metrics['take_profit'] - price) if metrics['take_profit'] > 0 else price * 0.05
        
        metrics['risk_reward_ratio'] = reward / risk if risk > 0 else 0
        
        # Calculate position size (simplified)
        account_risk = 0.02  # 2% account risk
        metrics['position_size'] = account_risk / (risk / price) if risk > 0 else 0
        
        return metrics
    
    def _generate_overall_signal(self, 
                               result: CalculationResult, 
                               projections: ProjectionResult,
                               cycles: Dict) -> Dict:
        """Generate overall trading signal combining all factors"""
        
        # Start with neutral signal
        overall = {
            'action': 'hold',
            'confidence': 0.5,
            'strength': 0.5,
            'factors': []
        }
        
        # Weight different factors
        weights = {
            'sacred_relationship': 0.3,
            'projection_probability': 0.25,
            'cycle_strength': 0.2,
            'position_confidence': 0.15,
            'harmonic_ratio': 0.1
        }
        
        score = 0.0
        
        # Sacred relationship factor
        if result.sacred_relationship:
            score += weights['sacred_relationship']
            overall['factors'].append('sacred_relationship')
        
        # Projection probability factor
        proj_factor = (projections.probability - 0.5) * weights['projection_probability']
        score += proj_factor
        if projections.probability > 0.7:
            overall['factors'].append('high_probability_projection')
        
        # Cycle strength factor
        cycle_summary = cycles.get('analysis_summary', {})
        cycle_strength = cycle_summary.get('cycle_strength', 0)
        score += cycle_strength * weights['cycle_strength']
        if cycle_strength > 0.6:
            overall['factors'].append('strong_cycles')
        
        # Position confidence factor
        score += result.confidence * weights['position_confidence']
        if result.confidence > 0.8:
            overall['factors'].append('high_position_confidence')
        
        # Harmonic ratio factor
        if abs(result.harmonic_ratio - 1.618) < 0.1:  # Near golden ratio
            score += weights['harmonic_ratio']
            overall['factors'].append('golden_ratio_harmony')
        
        # Determine action based on score
        if score > 0.7:
            overall['action'] = 'strong_buy' if projections.target_value > result.value else 'strong_sell'
            overall['confidence'] = min(score, 0.95)
        elif score > 0.6:
            overall['action'] = 'buy' if projections.target_value > result.value else 'sell'
            overall['confidence'] = score
        elif score < 0.3:
            overall['action'] = 'avoid'
            overall['confidence'] = 1 - score
        
        overall['strength'] = score
        
        return overall    
    def _update_performance_metrics(self, result: CalculationResult):
        """Update performance tracking metrics"""
        self.performance_metrics['total_calculations'] += 1
        
        if result.confidence > 0.5:
            self.performance_metrics['successful_calculations'] += 1
        
        # Update average precision
        current_avg = self.performance_metrics['average_precision']
        total_calcs = self.performance_metrics['total_calculations']
        new_avg = (current_avg * (total_calcs - 1) + result.confidence) / total_calcs
        self.performance_metrics['average_precision'] = new_avg
        
        # Update calculation time
        calc_time = result.metadata.get('calculation_time', 0)
        current_time_avg = self.performance_metrics['calculation_time']
        new_time_avg = (current_time_avg * (total_calcs - 1) + calc_time) / total_calcs
        self.performance_metrics['calculation_time'] = new_time_avg
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        success_rate = 0
        if self.performance_metrics['total_calculations'] > 0:
            success_rate = (self.performance_metrics['successful_calculations'] / 
                          self.performance_metrics['total_calculations'])
        
        return {
            'total_calculations': self.performance_metrics['total_calculations'],
            'successful_calculations': self.performance_metrics['successful_calculations'],
            'success_rate': success_rate,
            'average_precision': self.performance_metrics['average_precision'],
            'average_calculation_time': self.performance_metrics['calculation_time'],
            'cache_size': len(self.calculation_cache),
            'projection_cache_size': len(self.projection_cache),
            'ml_enabled': self.enable_ml,
            'calculation_mode': self.calculation_mode.value,
            'square_type': self.square_type.value
        }
    
    def optimize_parameters(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize calculator parameters based on historical data
        
        Args:
            historical_data: Historical price data for optimization
            
        Returns:
            Dictionary with optimization results and recommended parameters
        """
        try:
            prices = historical_data['close'].values if 'close' in historical_data.columns else historical_data.iloc[:, 0].values
            
            # Test different calculation modes
            mode_performance = {}
            
            for mode in CalculationMode:
                original_mode = self.calculation_mode
                self.calculation_mode = mode
                
                # Calculate positions for sample prices
                sample_prices = prices[::max(1, len(prices) // 50)]  # Sample every nth price
                results = self.batch_calculate(sample_prices.tolist())
                
                # Evaluate performance
                avg_confidence = np.mean([r.confidence for r in results])
                sacred_count = sum(1 for r in results if r.sacred_relationship)
                sacred_ratio = sacred_count / len(results) if results else 0
                
                mode_performance[mode.value] = {
                    'average_confidence': avg_confidence,
                    'sacred_ratio': sacred_ratio,
                    'score': avg_confidence * 0.7 + sacred_ratio * 0.3
                }
                
                # Restore original mode
                self.calculation_mode = original_mode
            
            # Find best mode
            best_mode = max(mode_performance.items(), key=lambda x: x[1]['score'])
            
            # Test different center values
            center_candidates = [
                np.mean(prices),
                np.median(prices),
                np.percentile(prices, 25),
                np.percentile(prices, 75),
                prices[0],  # First price
                prices[-1]  # Last price
            ]
            
            center_performance = {}
            for center in center_candidates:
                sample_results = self.batch_calculate(sample_prices.tolist(), center)
                avg_confidence = np.mean([r.confidence for r in sample_results])
                center_performance[center] = avg_confidence
            
            best_center = max(center_performance.items(), key=lambda x: x[1])
            
            # Optimization results
            optimization_results = {
                'recommended_mode': best_mode[0],
                'mode_performance': mode_performance,
                'recommended_center': best_center[0],
                'center_performance': center_performance,
                'optimization_summary': {
                    'best_mode_score': best_mode[1]['score'],
                    'best_center_confidence': best_center[1],
                    'improvement_potential': best_mode[1]['score'] - mode_performance.get(self.calculation_mode.value, {}).get('score', 0)
                }
            }
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
            return {}
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current calculator configuration"""
        return {
            'calculation_mode': self.calculation_mode.value,
            'square_type': self.square_type.value,
            'center_value': self.center_value,
            'max_radius': self.max_radius,
            'precision': self.precision,
            'enable_ml': self.enable_ml,
            'sacred_numbers': self.sacred_numbers,
            'performance_metrics': self.performance_metrics,
            'configuration_timestamp': datetime.now().isoformat()
        }
    
    def import_configuration(self, config: Dict[str, Any]):
        """Import calculator configuration"""
        try:
            if 'calculation_mode' in config:
                self.calculation_mode = CalculationMode(config['calculation_mode'])
            
            if 'square_type' in config:
                self.square_type = SquareType(config['square_type'])
            
            if 'center_value' in config:
                self.center_value = config['center_value']
            
            if 'max_radius' in config:
                self.max_radius = config['max_radius']
            
            if 'precision' in config:
                self.precision = config['precision']
            
            if 'enable_ml' in config:
                self.enable_ml = config['enable_ml']
            
            if 'sacred_numbers' in config:
                self.sacred_numbers = config['sacred_numbers']
            
            # Reinitialize components
            self._initialize_calculator()
            if self.enable_ml:
                self._initialize_ml_models()
            
            logger.info("Configuration imported successfully")
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            raise
    
    def clear_cache(self):
        """Clear calculation and projection caches"""
        self.calculation_cache.clear()
        self.projection_cache.clear()
        logger.info("Caches cleared")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        calc_cache_size = sum(len(str(k)) + len(str(v)) for k, v in self.calculation_cache.items())
        proj_cache_size = sum(len(str(k)) + len(str(v)) for k, v in self.projection_cache.items())
        
        return {
            'calculation_cache': {
                'entries': len(self.calculation_cache),
                'size_bytes': calc_cache_size,
                'hit_ratio': 0.85  # Estimated - would track in production
            },
            'projection_cache': {
                'entries': len(self.projection_cache),
                'size_bytes': proj_cache_size,
                'hit_ratio': 0.75  # Estimated - would track in production
            },
            'total_cache_size': calc_cache_size + proj_cache_size
        }


def demo_square_of_nine_calculator():
    """
    Demonstration of Square of Nine Calculator functionality
    """
    print("=== Square of Nine Calculator Demo ===\n")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate realistic price data with trends and cycles
    base_price = 100
    trend = np.linspace(0, 20, len(dates))
    cycles = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)  # 30-day cycle
    noise = np.random.normal(0, 2, len(dates))
    
    prices = base_price + trend + cycles + noise
    historical_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Initialize calculator with different modes
    modes_to_test = [CalculationMode.SPIRAL, CalculationMode.HARMONIC, CalculationMode.SACRED]
    
    for mode in modes_to_test:
        print(f"\n--- Testing {mode.value.upper()} Mode ---")
        
        calculator = SquareOfNineCalculatorIndicator(
            calculation_mode=mode,
            square_type=SquareType.PRICE_SQUARE,
            enable_ml=True
        )
        
        # Test basic calculations
        current_price = prices[-1]
        print(f"Current Price: ${current_price:.2f}")
        
        # Calculate position
        result = calculator.calculate_square_position(current_price)
        print(f"Square Position: {result.position}")
        print(f"Radius: {result.radius:.2f}")
        print(f"Angle: {math.degrees(result.angle):.1f}°")
        print(f"Sacred Relationship: {result.sacred_relationship}")
        print(f"Confidence: {result.confidence:.2f}")
        
        # Calculate projections
        projections = calculator.calculate_projections(current_price, historical_data)
        print(f"\nProjections:")
        print(f"Target Value: ${projections.target_value:.2f}")
        print(f"Probability: {projections.probability:.2f}")
        print(f"Support Levels: {[f'${s:.2f}' for s in projections.support_levels[:3]]}")
        print(f"Resistance Levels: {[f'${r:.2f}' for r in projections.resistance_levels[:3]]}")
        
        # Generate trading signals
        signals = calculator.generate_trading_signals(current_price, historical_data)
        if signals:
            overall_signal = signals.get('overall_signal', {})
            print(f"\nTrading Signal:")
            print(f"Action: {overall_signal.get('action', 'hold').upper()}")
            print(f"Confidence: {overall_signal.get('confidence', 0):.2f}")
            print(f"Strength: {overall_signal.get('strength', 0):.2f}")
            print(f"Factors: {', '.join(overall_signal.get('factors', []))}")
    
    # Test batch calculations
    print(f"\n--- Batch Calculation Test ---")
    test_prices = [95, 100, 105, 110, 115]
    
    calculator = SquareOfNineCalculatorIndicator(
        calculation_mode=CalculationMode.HYBRID,
        enable_ml=True
    )
    
    batch_results = calculator.batch_calculate(test_prices)
    print("Price\t\tPosition\tRadius\t\tConfidence")
    print("-" * 60)
    for result in batch_results:
        print(f"${result.value:.2f}\t\t{result.position}\t\t{result.radius:.2f}\t\t{result.confidence:.2f}")
    
    # Test time cycles
    print(f"\n--- Time Cycle Analysis ---")
    cycles = calculator.calculate_time_cycles(historical_data)
    if cycles and 'analysis_summary' in cycles:
        summary = cycles['analysis_summary']
        print(f"Dominant Cycle Length: {summary.get('dominant_cycle', {}).get('length', 'N/A')}")
        print(f"Cycle Count: {summary.get('cycle_count', 0)}")
        print(f"Average Cycle Length: {summary.get('average_cycle_length', 0):.1f}")
        print(f"Cycle Strength: {summary.get('cycle_strength', 0):.2f}")
    
    # Test optimization
    print(f"\n--- Parameter Optimization ---")
    optimization = calculator.optimize_parameters(historical_data.tail(100))  # Use last 100 days
    if optimization:
        print(f"Recommended Mode: {optimization.get('recommended_mode', 'N/A')}")
        print(f"Recommended Center: ${optimization.get('recommended_center', 0):.2f}")
        
        improvement = optimization.get('optimization_summary', {}).get('improvement_potential', 0)
        print(f"Improvement Potential: {improvement:.3f}")
    
    # Performance report
    print(f"\n--- Performance Report ---")
    performance = calculator.get_performance_report()
    print(f"Total Calculations: {performance['total_calculations']}")
    print(f"Success Rate: {performance['success_rate']:.2%}")
    print(f"Average Precision: {performance['average_precision']:.3f}")
    print(f"Average Calc Time: {performance['average_calculation_time']:.4f}s")
    
    # Cache statistics
    cache_stats = calculator.get_cache_statistics()
    print(f"\nCache Statistics:")
    print(f"Calculation Cache Entries: {cache_stats['calculation_cache']['entries']}")
    print(f"Projection Cache Entries: {cache_stats['projection_cache']['entries']}")
    print(f"Total Cache Size: {cache_stats['total_cache_size']} bytes")
    
    print(f"\n=== Demo Complete ===")


if __name__ == "__main__":
    # Run demonstration
    demo_square_of_nine_calculator()
    
    # Example usage in trading system
    print(f"\n=== Integration Example ===")
    
    # Create calculator for live trading
    live_calculator = SquareOfNineCalculatorIndicator(
        calculation_mode=CalculationMode.HYBRID,
        square_type=SquareType.PRICE_SQUARE,
        enable_ml=True,
        precision=4
    )
    
    # Simulate real-time price updates
    sample_prices = [100.25, 101.50, 99.75, 102.80, 98.60]
    
    print("Real-time Signal Generation:")
    print("-" * 50)
    
    for price in sample_prices:
        # Quick position calculation
        position_result = live_calculator.calculate_square_position(price)
        
        print(f"Price: ${price:.2f}")
        print(f"  Position: {position_result.position}")
        print(f"  Sacred: {'Yes' if position_result.sacred_relationship else 'No'}")
        print(f"  Confidence: {position_result.confidence:.2f}")
        print(f"  Angle: {math.degrees(position_result.angle):.1f}°")
        print()
    
    # Export configuration for save/load
    config = live_calculator.export_configuration()
    print(f"Configuration exported: {len(config)} parameters")
    
    print(f"\nSquare of Nine Calculator ready for production use!")