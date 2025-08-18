"""
Gann Square Indicator

Advanced implementation of W.D. Gann's Square calculations for the humanitarian trading platform.
This indicator performs comprehensive square analysis including price squares, time squares,
geometric progressions, and advanced mathematical relationships based on Gann's principles.

Features:
- Price square calculations (square roots, perfect squares)
- Time square analysis and relationships
- Geometric progression detection
- Square of price and time analysis
- Mathematical square relationships
- Support and resistance from square levels
- ML-enhanced square pattern recognition
- Multi-timeframe square analysis
- Advanced trading signals from square levels

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
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import scipy.stats as stats
    from scipy.optimize import minimize_scalar
    SKLEARN_AVAILABLE = True
    SCIPY_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    SCIPY_AVAILABLE = False


class SquareType(Enum):
    """Types of squares in Gann analysis"""
    PRICE_SQUARE = "price_square"
    TIME_SQUARE = "time_square"
    GEOMETRIC_SQUARE = "geometric_square"
    HARMONIC_SQUARE = "harmonic_square"
    FIBONACCI_SQUARE = "fibonacci_square"
    CARDINAL_SQUARE = "cardinal_square"


class SquareLevel(Enum):
    """Square level significance"""
    MINOR = "minor"
    INTERMEDIATE = "intermediate"
    MAJOR = "major"
    CARDINAL = "cardinal"


@dataclass
class GannSquareConfig:
    """Configuration for Gann Square Indicator"""
    
    # Analysis parameters
    lookback_period: int = 252  # Trading days (1 year)
    square_precision: float = 0.01  # Precision for square calculations (1%)
    
    # Square calculation parameters
    price_square_range: Tuple[int, int] = (1, 100)  # Range for price square roots
    time_square_range: Tuple[int, int] = (1, 50)    # Range for time square roots
    geometric_step: float = 1.414                   # √2 for geometric progression
    
    # Level significance thresholds
    minor_threshold: float = 0.3
    intermediate_threshold: float = 0.5
    major_threshold: float = 0.7
    cardinal_threshold: float = 0.9
    
    # Square relationships
    fibonacci_numbers: List[int] = field(default_factory=lambda: [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144])
    cardinal_numbers: List[int] = field(default_factory=lambda: [4, 9, 16, 25, 36, 49, 64, 81, 100, 144])
    harmonic_ratios: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0])
    
    # Machine learning parameters
    enable_ml: bool = True
    ml_lookback: int = 500
    cluster_count: int = 6
    feature_scaling: bool = True
    
    # Trading parameters
    signal_strength_threshold: float = 0.6
    breakout_confirmation: float = 0.02  # 2% breakout confirmation
    max_levels_per_type: int = 20
    
    # Performance parameters
    enable_caching: bool = True
    max_cache_size: int = 1000
    parallel_processing: bool = False


@dataclass
class SquarePoint:
    """Represents a square-based level or point"""
    value: float
    square_type: SquareType
    square_level: SquareLevel
    square_root: float
    significance: float
    timestamp: Optional[float] = None
    support_count: int = 0
    resistance_count: int = 0
    last_touch_time: Optional[float] = None
    
    def __post_init__(self):
        self.significance = max(0.0, min(1.0, self.significance))


@dataclass
class SquareRelationship:
    """Represents a relationship between square levels"""
    point1: SquarePoint
    point2: SquarePoint
    relationship_type: str
    ratio: float
    strength: float
    geometric_progression: Optional[float] = None
    harmonic_level: Optional[int] = None
    
    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))


@dataclass
class GeometricProgression:
    """Represents a geometric progression in squares"""
    starting_value: float
    progression_ratio: float
    levels: List[float]
    strength: float
    square_type: SquareType
    time_discovered: float
    
    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))


@dataclass
class SquareProjection:
    """Represents a projection based on square analysis"""
    target_level: float
    projection_type: str
    confidence: float
    time_target: Optional[float] = None
    source_square: Optional[SquarePoint] = None
    supporting_progression: Optional[GeometricProgression] = None
    
    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))


class GannSquareSignal(NamedTuple):
    """Trading signal from square analysis"""
    signal_type: str
    direction: str  # bullish, bearish, neutral
    strength: float
    entry_price: float
    target_price: float
    stop_loss: float
    square_level: float
    square_type: str
    confidence: float
    geometric_support: bool
    harmonic_support: bool


class GannSquareIndicator:
    """
    Advanced Gann Square Indicator
    
    Implements comprehensive square analysis based on W.D. Gann's principles.
    Provides price squares, time squares, geometric progressions, and mathematical relationships.
    """
    
    def __init__(self, config: Optional[GannSquareConfig] = None):
        """Initialize the Gann Square Indicator"""
        
        self.config = config or GannSquareConfig()
        self.logger = logging.getLogger(__name__)
        
        # Analysis state
        self.price_data: Optional[np.ndarray] = None
        self.price_squares: List[SquarePoint] = []
        self.time_squares: List[SquarePoint] = []
        self.geometric_squares: List[SquarePoint] = []
        self.harmonic_squares: List[SquarePoint] = []
        self.fibonacci_squares: List[SquarePoint] = []
        self.cardinal_squares: List[SquarePoint] = []
        
        # Relationships and progressions
        self.square_relationships: List[SquareRelationship] = []
        self.geometric_progressions: List[GeometricProgression] = []
        self.projections: List[SquareProjection] = []
        
        # Analysis results
        self.current_square_zone: Optional[str] = None
        self.next_major_level: Optional[float] = None
        self.square_momentum: float = 0.0
        
        # ML components
        if self.config.enable_ml and SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            self.square_clusterer = None
            self.level_predictor = None
            
        # Caching
        if self.config.enable_caching:
            self._cache: Dict = {}
            self._cache_timestamps: deque = deque(maxlen=self.config.max_cache_size)
            
        self.logger.info("Gann Square Indicator initialized")
        
    def analyze(self, price_data: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive square analysis
        
        Args:
            price_data: OHLCV data array [timestamp, open, high, low, close, volume]
            
        Returns:
            Dictionary containing analysis results
        """
        
        if price_data is None or len(price_data) < 10:
            raise ValueError("Insufficient price data for square analysis")
            
        self.price_data = price_data.copy()
        
        try:
            # Clear previous analysis
            self._reset_analysis()
            
            self.logger.info(f"Starting square analysis on {len(price_data)} data points")
            
            # Step 1: Calculate price squares
            self._calculate_price_squares()
            
            # Step 2: Calculate time squares
            self._calculate_time_squares()
            
            # Step 3: Calculate geometric squares
            self._calculate_geometric_squares()
            
            # Step 4: Calculate harmonic squares
            self._calculate_harmonic_squares()
            
            # Step 5: Calculate Fibonacci squares
            self._calculate_fibonacci_squares()
            
            # Step 6: Calculate cardinal squares
            self._calculate_cardinal_squares()
            
            # Step 7: Analyze square relationships
            self._analyze_square_relationships()
            
            # Step 8: Detect geometric progressions
            self._detect_geometric_progressions()
            
            # Step 9: Apply ML analysis if enabled
            if self.config.enable_ml:
                self._ml_enhanced_analysis()
                
            # Step 10: Update square significance based on price action
            self._update_square_significance()
            
            # Step 11: Generate projections
            self._generate_projections()
            
            # Step 12: Assess current market position
            self._assess_current_position()
            
            # Compile results
            results = self._compile_results()
            
            self.logger.info("Square analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Square analysis failed: {e}")
            return self._get_default_results()
            
    def _reset_analysis(self):
        """Reset analysis state"""
        
        self.price_squares.clear()
        self.time_squares.clear()
        self.geometric_squares.clear()
        self.harmonic_squares.clear()
        self.fibonacci_squares.clear()
        self.cardinal_squares.clear()
        
        self.square_relationships.clear()
        self.geometric_progressions.clear()
        self.projections.clear()
        
        self.current_square_zone = None
        self.next_major_level = None
        self.square_momentum = 0.0
        
    def _calculate_price_squares(self):
        """Calculate price square levels"""
        
        if self.price_data is None:
            return
            
        prices = self.price_data[:, 4]  # Close prices
        price_range = (np.min(prices), np.max(prices))
        
        # Find square roots that create levels within price range
        min_root = int(math.sqrt(price_range[0]) - 1)
        max_root = int(math.sqrt(price_range[1]) + 1)
        
        min_root = max(min_root, self.config.price_square_range[0])
        max_root = min(max_root, self.config.price_square_range[1])
        
        for root in range(min_root, max_root + 1):
            square_value = root * root
            
            if price_range[0] * 0.8 <= square_value <= price_range[1] * 1.2:
                
                # Calculate significance based on price action around this level
                significance = self._calculate_level_significance(square_value, prices)
                
                if significance >= self.config.minor_threshold:
                    
                    square_level = self._determine_square_level(significance)
                    
                    square_point = SquarePoint(
                        value=square_value,
                        square_type=SquareType.PRICE_SQUARE,
                        square_level=square_level,
                        square_root=root,
                        significance=significance
                    )
                    
                    self.price_squares.append(square_point)
                    
        # Sort by value
        self.price_squares.sort(key=lambda s: s.value)
        
        self.logger.info(f"Calculated {len(self.price_squares)} price squares")
        
    def _calculate_time_squares(self):
        """Calculate time square levels"""
        
        if self.price_data is None:
            return
            
        timestamps = self.price_data[:, 0]
        time_span = timestamps[-1] - timestamps[0]
        time_units = time_span / (24 * 3600)  # Convert to days
        
        # Calculate time squares based on trading days
        for root in range(self.config.time_square_range[0], self.config.time_square_range[1] + 1):
            
            square_days = root * root
            
            if square_days <= time_units * 1.5:  # Allow some extension beyond current timeframe
                
                # Convert back to timestamp
                square_time = timestamps[0] + square_days * 24 * 3600
                
                # Calculate significance based on market activity around time squares
                significance = self._calculate_time_significance(square_time, timestamps)
                
                if significance >= self.config.minor_threshold:
                    
                    square_level = self._determine_square_level(significance)
                    
                    square_point = SquarePoint(
                        value=square_days,  # Store as days for consistency
                        square_type=SquareType.TIME_SQUARE,
                        square_level=square_level,
                        square_root=root,
                        significance=significance,
                        timestamp=square_time
                    )
                    
                    self.time_squares.append(square_point)
                    
        self.logger.info(f"Calculated {len(self.time_squares)} time squares")
        
    def _calculate_geometric_squares(self):
        """Calculate geometric progression squares"""
        
        if self.price_data is None:
            return
            
        prices = self.price_data[:, 4]
        price_range = (np.min(prices), np.max(prices))
        
        # Use different starting points and geometric ratios
        geometric_ratios = [1.414, 1.618, 2.0, 2.236]  # √2, φ, 2, √5
        
        for ratio in geometric_ratios:
            
            # Try different starting points
            start_values = [
                price_range[0],
                price_range[0] * 1.1,
                (price_range[0] + price_range[1]) / 2,
                price_range[1] * 0.9
            ]
            
            for start_value in start_values:
                
                current_value = start_value
                level_count = 0
                
                while current_value <= price_range[1] * 1.5 and level_count < 20:
                    
                    if current_value >= price_range[0] * 0.8:
                        
                        significance = self._calculate_level_significance(current_value, prices)
                        
                        if significance >= self.config.minor_threshold:
                            
                            square_level = self._determine_square_level(significance)
                            
                            square_point = SquarePoint(
                                value=current_value,
                                square_type=SquareType.GEOMETRIC_SQUARE,
                                square_level=square_level,
                                square_root=ratio,  # Store ratio instead of square root
                                significance=significance
                            )
                            
                            self.geometric_squares.append(square_point)
                            
                    current_value *= ratio
                    level_count += 1
                    
        # Remove duplicates and sort
        self.geometric_squares = self._remove_duplicate_squares(self.geometric_squares)
        self.geometric_squares.sort(key=lambda s: s.value)
        
        self.logger.info(f"Calculated {len(self.geometric_squares)} geometric squares")
        
    def _calculate_harmonic_squares(self):
        """Calculate harmonic squares"""
        
        if self.price_data is None:
            return
            
        prices = self.price_data[:, 4]
        price_range = (np.min(prices), np.max(prices))
        
        # Use harmonic ratios to create square levels
        base_price = price_range[0]
        
        for ratio in self.config.harmonic_ratios:
            
            harmonic_value = base_price * ratio
            
            if price_range[0] * 0.8 <= harmonic_value <= price_range[1] * 1.2:
                
                significance = self._calculate_level_significance(harmonic_value, prices)
                
                if significance >= self.config.minor_threshold:
                    
                    square_level = self._determine_square_level(significance)
                    
                    square_point = SquarePoint(
                        value=harmonic_value,
                        square_type=SquareType.HARMONIC_SQUARE,
                        square_level=square_level,
                        square_root=ratio,
                        significance=significance
                    )
                    
                    self.harmonic_squares.append(square_point)
                    
        # Also calculate harmonic squares from high and low
        for base_price in [price_range[1], (price_range[0] + price_range[1]) / 2]:
            
            for ratio in self.config.harmonic_ratios:
                
                harmonic_value = base_price * ratio
                
                if price_range[0] * 0.8 <= harmonic_value <= price_range[1] * 1.2:
                    
                    significance = self._calculate_level_significance(harmonic_value, prices)
                    
                    if significance >= self.config.minor_threshold:
                        
                        square_level = self._determine_square_level(significance)
                        
                        square_point = SquarePoint(
                            value=harmonic_value,
                            square_type=SquareType.HARMONIC_SQUARE,
                            square_level=square_level,
                            square_root=ratio,
                            significance=significance
                        )
                        
                        self.harmonic_squares.append(square_point)
                        
        # Remove duplicates and sort
        self.harmonic_squares = self._remove_duplicate_squares(self.harmonic_squares)
        self.harmonic_squares.sort(key=lambda s: s.value)
        
        self.logger.info(f"Calculated {len(self.harmonic_squares)} harmonic squares")
        
    def _calculate_fibonacci_squares(self):
        """Calculate Fibonacci-based squares"""
        
        if self.price_data is None:
            return
            
        prices = self.price_data[:, 4]
        price_range = (np.min(prices), np.max(prices))
        
        # Use Fibonacci numbers to create squares
        for fib_num in self.config.fibonacci_numbers:
            
            # Square of Fibonacci number
            fib_square = fib_num * fib_num
            
            # Scale to price range
            scaled_values = [
                price_range[0] * fib_square / 100,  # Percentage scaling
                price_range[0] + (price_range[1] - price_range[0]) * fib_num / 144,  # Ratio scaling
                fib_square  # Direct value (if in range)
            ]
            
            for scaled_value in scaled_values:
                
                if price_range[0] * 0.8 <= scaled_value <= price_range[1] * 1.2:
                    
                    significance = self._calculate_level_significance(scaled_value, prices)
                    
                    if significance >= self.config.minor_threshold:
                        
                        square_level = self._determine_square_level(significance)
                        
                        square_point = SquarePoint(
                            value=scaled_value,
                            square_type=SquareType.FIBONACCI_SQUARE,
                            square_level=square_level,
                            square_root=fib_num,
                            significance=significance
                        )
                        
                        self.fibonacci_squares.append(square_point)
                        
        # Remove duplicates and sort
        self.fibonacci_squares = self._remove_duplicate_squares(self.fibonacci_squares)
        self.fibonacci_squares.sort(key=lambda s: s.value)
        
        self.logger.info(f"Calculated {len(self.fibonacci_squares)} Fibonacci squares")
        
    def _calculate_cardinal_squares(self):
        """Calculate cardinal squares (special significant squares)"""
        
        if self.price_data is None:
            return
            
        prices = self.price_data[:, 4]
        price_range = (np.min(prices), np.max(prices))
        
        # Cardinal squares are special numbers in Gann theory
        for cardinal_num in self.config.cardinal_numbers:
            
            # Use cardinal number directly and as scaled values
            cardinal_values = [
                cardinal_num,
                price_range[0] * cardinal_num / 100,
                price_range[0] + (price_range[1] - price_range[0]) * cardinal_num / 144
            ]
            
            for cardinal_value in cardinal_values:
                
                if price_range[0] * 0.8 <= cardinal_value <= price_range[1] * 1.2:
                    
                    significance = self._calculate_level_significance(cardinal_value, prices)
                    
                    # Cardinal squares get bonus significance
                    significance = min(1.0, significance * 1.2)
                    
                    if significance >= self.config.minor_threshold:
                        
                        square_level = self._determine_square_level(significance)
                        
                        square_point = SquarePoint(
                            value=cardinal_value,
                            square_type=SquareType.CARDINAL_SQUARE,
                            square_level=square_level,
                            square_root=math.sqrt(cardinal_num),
                            significance=significance
                        )
                        
                        self.cardinal_squares.append(square_point)
                        
        # Remove duplicates and sort
        self.cardinal_squares = self._remove_duplicate_squares(self.cardinal_squares)
        self.cardinal_squares.sort(key=lambda s: s.value)
        
        self.logger.info(f"Calculated {len(self.cardinal_squares)} cardinal squares")
        
    def _calculate_level_significance(self, level: float, prices: np.ndarray) -> float:
        """Calculate significance of a square level based on price action"""
        
        if len(prices) == 0:
            return 0.0
            
        tolerance = level * self.config.square_precision
        
        # Count touches, bounces, and time spent near level
        touches = 0
        bounces = 0
        time_near_level = 0
        
        for i, price in enumerate(prices):
            
            if abs(price - level) <= tolerance:
                touches += 1
                time_near_level += 1
                
                # Check for bounce (reversal after touching level)
                if i < len(prices) - 2:
                    prev_distance = abs(prices[i-1] - level) if i > 0 else tolerance * 2
                    next_distance = abs(prices[i+1] - level)
                    
                    if prev_distance > tolerance and next_distance > tolerance:
                        bounces += 1
                        
        # Calculate significance factors
        touch_factor = min(1.0, touches / 10)  # Normalize to 10 touches
        bounce_factor = min(1.0, bounces / 5)  # Normalize to 5 bounces
        time_factor = min(1.0, time_near_level / len(prices))
        
        # Combine factors
        significance = (touch_factor * 0.4 + bounce_factor * 0.4 + time_factor * 0.2)
        
        return significance
        
    def _calculate_time_significance(self, target_time: float, timestamps: np.ndarray) -> float:
        """Calculate significance of a time square"""
        
        if len(timestamps) == 0:
            return 0.0
            
        # Check if target time is near significant market events
        # For now, use a simple proximity measure
        
        time_tolerance = 24 * 3600  # 1 day tolerance
        
        near_events = 0
        for timestamp in timestamps:
            if abs(timestamp - target_time) <= time_tolerance:
                near_events += 1
                
        # Simple significance based on data density around time
        significance = min(1.0, near_events / 10)
        
        return max(0.3, significance)  # Minimum significance for time squares
        
    def _determine_square_level(self, significance: float) -> SquareLevel:
        """Determine square level based on significance"""
        
        if significance >= self.config.cardinal_threshold:
            return SquareLevel.CARDINAL
        elif significance >= self.config.major_threshold:
            return SquareLevel.MAJOR
        elif significance >= self.config.intermediate_threshold:
            return SquareLevel.INTERMEDIATE
        else:
            return SquareLevel.MINOR
            
    def _remove_duplicate_squares(self, squares: List[SquarePoint]) -> List[SquarePoint]:
        """Remove duplicate square levels"""
        
        if not squares:
            return squares
            
        unique_squares = []
        tolerance = squares[0].value * 0.01 if squares else 1.0  # 1% tolerance
        
        for square in squares:
            is_duplicate = False
            
            for existing in unique_squares:
                if abs(square.value - existing.value) <= tolerance:
                    # Keep the more significant square
                    if square.significance > existing.significance:
                        unique_squares.remove(existing)
                        unique_squares.append(square)
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_squares.append(square)
                
        return unique_squares        
    def _analyze_square_relationships(self):
        """Analyze relationships between different square levels"""
        
        # Collect all squares for relationship analysis
        all_squares = (
            self.price_squares + self.time_squares + self.geometric_squares +
            self.harmonic_squares + self.fibonacci_squares + self.cardinal_squares
        )
        
        if len(all_squares) < 2:
            return
            
        # Analyze relationships between squares
        for i in range(len(all_squares)):
            for j in range(i + 1, min(i + 10, len(all_squares))):  # Limit combinations
                
                square1, square2 = all_squares[i], all_squares[j]
                relationship = self._analyze_square_pair(square1, square2)
                
                if relationship:
                    self.square_relationships.append(relationship)
                    
        # Sort relationships by strength
        self.square_relationships.sort(key=lambda r: r.strength, reverse=True)
        
        # Keep only top relationships
        max_relationships = 50
        if len(self.square_relationships) > max_relationships:
            self.square_relationships = self.square_relationships[:max_relationships]
            
        self.logger.info(f"Analyzed {len(self.square_relationships)} square relationships")
        
    def _analyze_square_pair(self, square1: SquarePoint, square2: SquarePoint) -> Optional[SquareRelationship]:
        """Analyze relationship between two square points"""
        
        if square1.value == square2.value:
            return None
            
        # Calculate ratio
        higher_square = square2 if square2.value > square1.value else square1
        lower_square = square1 if square1.value < square2.value else square2
        
        ratio = higher_square.value / lower_square.value if lower_square.value > 0 else 0
        
        if ratio < 1.01:  # Too close
            return None
            
        # Determine relationship type
        relationship_type = self._determine_relationship_type(ratio, square1, square2)
        
        if not relationship_type:
            return None
            
        # Calculate relationship strength
        strength = self._calculate_relationship_strength(square1, square2, ratio)
        
        if strength < 0.3:
            return None
            
        # Check for geometric progression
        geometric_progression = None
        if ratio in [1.414, 1.618, 2.0, 2.236]:  # Common geometric ratios
            geometric_progression = ratio
            
        # Check for harmonic level
        harmonic_level = None
        if ratio in self.config.harmonic_ratios:
            harmonic_level = self.config.harmonic_ratios.index(ratio) + 1
            
        relationship = SquareRelationship(
            point1=lower_square,
            point2=higher_square,
            relationship_type=relationship_type,
            ratio=ratio,
            strength=strength,
            geometric_progression=geometric_progression,
            harmonic_level=harmonic_level
        )
        
        return relationship
        
    def _determine_relationship_type(self, ratio: float, square1: SquarePoint, square2: SquarePoint) -> Optional[str]:
        """Determine the type of relationship between squares"""
        
        # Perfect square relationships
        if abs(ratio - round(ratio)**0.5) < 0.05:  # Close to perfect square root
            return "perfect_square"
            
        # Fibonacci relationships
        if any(abs(ratio - fib) < 0.05 for fib in [1.618, 2.618, 0.618]):
            return "fibonacci"
            
        # Geometric relationships
        if any(abs(ratio - geom) < 0.05 for geom in [1.414, 2.0, 2.236, 1.732]):
            return "geometric"
            
        # Harmonic relationships
        if any(abs(ratio - harm) < 0.05 for harm in self.config.harmonic_ratios):
            return "harmonic"
            
        # Cardinal relationships (special Gann numbers)
        if any(abs(ratio - card/10) < 0.05 for card in self.config.cardinal_numbers):
            return "cardinal"
            
        # Simple ratio relationships
        if abs(ratio - round(ratio)) < 0.1:  # Close to whole number
            return "simple_ratio"
            
        return None
        
    def _calculate_relationship_strength(self, square1: SquarePoint, square2: SquarePoint, ratio: float) -> float:
        """Calculate strength of relationship between squares"""
        
        # Base strength from square significance
        base_strength = (square1.significance + square2.significance) / 2
        
        # Type compatibility bonus
        type_bonus = 0.0
        if square1.square_type == square2.square_type:
            type_bonus = 0.2  # Same type squares have stronger relationships
        elif (square1.square_type in [SquareType.FIBONACCI_SQUARE, SquareType.HARMONIC_SQUARE] and
              square2.square_type in [SquareType.FIBONACCI_SQUARE, SquareType.HARMONIC_SQUARE]):
            type_bonus = 0.1  # Fibonacci and harmonic are related
            
        # Ratio significance bonus
        ratio_bonus = 0.0
        special_ratios = [1.414, 1.618, 2.0, 2.236, 2.618]
        if any(abs(ratio - special) < 0.05 for special in special_ratios):
            ratio_bonus = 0.15
            
        # Level significance bonus
        level_bonus = 0.0
        if square1.square_level in [SquareLevel.MAJOR, SquareLevel.CARDINAL] or \
           square2.square_level in [SquareLevel.MAJOR, SquareLevel.CARDINAL]:
            level_bonus = 0.1
            
        # Combine factors
        strength = base_strength + type_bonus + ratio_bonus + level_bonus
        
        return max(0.0, min(1.0, strength))
        
    def _detect_geometric_progressions(self):
        """Detect geometric progressions in square levels"""
        
        # Analyze each square type for geometric progressions
        square_types = [
            self.price_squares,
            self.geometric_squares,
            self.harmonic_squares,
            self.fibonacci_squares,
            self.cardinal_squares
        ]
        
        for squares in square_types:
            if len(squares) >= 3:
                progressions = self._find_progressions_in_squares(squares)
                self.geometric_progressions.extend(progressions)
                
        # Sort progressions by strength
        self.geometric_progressions.sort(key=lambda p: p.strength, reverse=True)
        
        self.logger.info(f"Detected {len(self.geometric_progressions)} geometric progressions")
        
    def _find_progressions_in_squares(self, squares: List[SquarePoint]) -> List[GeometricProgression]:
        """Find geometric progressions within a set of squares"""
        
        progressions = []
        
        if len(squares) < 3:
            return progressions
            
        # Sort squares by value
        sorted_squares = sorted(squares, key=lambda s: s.value)
        
        # Look for geometric progressions
        for i in range(len(sorted_squares) - 2):
            for j in range(i + 1, len(sorted_squares) - 1):
                for k in range(j + 1, len(sorted_squares)):
                    
                    s1, s2, s3 = sorted_squares[i], sorted_squares[j], sorted_squares[k]
                    
                    # Check if they form a geometric progression
                    if s1.value > 0 and s2.value > 0:
                        ratio1 = s2.value / s1.value
                        ratio2 = s3.value / s2.value
                        
                        # Allow some tolerance in ratios
                        if abs(ratio1 - ratio2) / max(ratio1, ratio2) < 0.1:
                            
                            # Create progression
                            levels = [s1.value, s2.value, s3.value]
                            avg_ratio = (ratio1 + ratio2) / 2
                            
                            # Calculate strength based on square significance
                            strength = (s1.significance + s2.significance + s3.significance) / 3
                            
                            # Bonus for special ratios
                            if any(abs(avg_ratio - special) < 0.05 for special in [1.414, 1.618, 2.0]):
                                strength = min(1.0, strength * 1.2)
                                
                            if strength >= 0.4:
                                
                                progression = GeometricProgression(
                                    starting_value=s1.value,
                                    progression_ratio=avg_ratio,
                                    levels=levels,
                                    strength=strength,
                                    square_type=s1.square_type,
                                    time_discovered=time.time()
                                )
                                
                                progressions.append(progression)
                                
        return progressions
        
    def _ml_enhanced_analysis(self):
        """Apply ML enhancement to square analysis"""
        
        if not SKLEARN_AVAILABLE:
            return
            
        try:
            # ML-enhanced square clustering
            self._ml_square_clustering()
            
            # ML-enhanced level prediction
            self._ml_level_prediction()
            
            # ML-enhanced significance scoring
            self._ml_significance_enhancement()
            
            self.logger.info("ML enhancement completed")
            
        except Exception as e:
            self.logger.debug(f"ML enhancement failed: {e}")
            
    def _ml_square_clustering(self):
        """Use ML to cluster similar square levels"""
        
        all_squares = (
            self.price_squares + self.geometric_squares + self.harmonic_squares +
            self.fibonacci_squares + self.cardinal_squares
        )
        
        if len(all_squares) < 5:
            return
            
        try:
            # Extract features for clustering
            features = []
            for square in all_squares:
                feature_vector = [
                    square.value,
                    square.significance,
                    square.square_root,
                    1.0 if square.square_type == SquareType.PRICE_SQUARE else 0.0,
                    1.0 if square.square_type == SquareType.GEOMETRIC_SQUARE else 0.0,
                    1.0 if square.square_type == SquareType.HARMONIC_SQUARE else 0.0,
                    1.0 if square.square_type == SquareType.FIBONACCI_SQUARE else 0.0,
                    1.0 if square.square_type == SquareType.CARDINAL_SQUARE else 0.0,
                    1.0 if square.square_level == SquareLevel.MAJOR else 0.0,
                    1.0 if square.square_level == SquareLevel.CARDINAL else 0.0
                ]
                features.append(feature_vector)
                
            X = np.array(features)
            
            # Normalize features
            if self.config.feature_scaling:
                X = self.scaler.fit_transform(X)
                
            # Cluster squares
            n_clusters = min(self.config.cluster_count, len(all_squares) // 2)
            if n_clusters >= 2:
                self.square_clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = self.square_clusterer.fit_predict(X)
                
                # Enhance significance based on cluster membership
                for i, square in enumerate(all_squares):
                    cluster_size = np.sum(cluster_labels == cluster_labels[i])
                    cluster_bonus = min(0.2, cluster_size / len(all_squares))
                    square.significance = min(1.0, square.significance + cluster_bonus)
                    
        except Exception as e:
            self.logger.debug(f"ML square clustering failed: {e}")
            
    def _ml_level_prediction(self):
        """Use ML to predict future square levels"""
        
        if not self.price_data or len(self.price_data) < 50:
            return
            
        try:
            # Prepare training data
            features, targets = self._prepare_level_prediction_data()
            
            if len(features) < 20:
                return
                
            X = np.array(features)
            y = np.array(targets)
            
            # Train prediction model
            self.level_predictor = LinearRegression()
            self.level_predictor.fit(X, y)
            
            # Evaluate model
            predictions = self.level_predictor.predict(X)
            r2 = r2_score(y, predictions)
            
            if r2 > 0.1:  # Minimum acceptable R²
                self.logger.info(f"ML level predictor trained with R² = {r2:.3f}")
            else:
                self.level_predictor = None
                
        except Exception as e:
            self.logger.debug(f"ML level prediction failed: {e}")
            
    def _prepare_level_prediction_data(self) -> Tuple[List[List[float]], List[float]]:
        """Prepare data for ML level prediction"""
        
        features = []
        targets = []
        
        if not self.price_data:
            return features, targets
            
        prices = self.price_data[:, 4]
        
        # Create features based on price patterns and square levels
        for i in range(10, len(prices) - 5):
            
            # Price features
            recent_prices = prices[i-10:i]
            feature_vector = [
                np.mean(recent_prices),
                np.std(recent_prices),
                recent_prices[-1] - recent_prices[0],
                (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2] if recent_prices[-2] != 0 else 0
            ]
            
            # Square level features
            current_price = prices[i]
            
            # Distance to nearest squares
            all_squares = self.price_squares + self.geometric_squares
            if all_squares:
                distances = [abs(s.value - current_price) / current_price for s in all_squares]
                min_distance = min(distances) if distances else 1.0
                feature_vector.append(min_distance)
            else:
                feature_vector.append(1.0)
                
            # Square momentum (simplified)
            square_momentum = self._calculate_square_momentum_at_point(i)
            feature_vector.append(square_momentum)
            
            # Target: future price relative to current
            future_price = prices[i + 5]
            target = (future_price - current_price) / current_price if current_price != 0 else 0
            
            features.append(feature_vector)
            targets.append(target)
            
        return features, targets
        
    def _calculate_square_momentum_at_point(self, index: int) -> float:
        """Calculate square momentum at a specific price point"""
        
        if not self.price_data or index >= len(self.price_data):
            return 0.0
            
        current_price = self.price_data[index, 4]
        
        # Find nearest squares
        all_squares = self.price_squares + self.geometric_squares
        
        if not all_squares:
            return 0.0
            
        # Calculate momentum based on position relative to squares
        above_squares = [s for s in all_squares if s.value < current_price]
        below_squares = [s for s in all_squares if s.value > current_price]
        
        above_strength = sum(s.significance for s in above_squares[-3:])  # Top 3 below
        below_strength = sum(s.significance for s in below_squares[:3])   # Top 3 above
        
        momentum = (above_strength - below_strength) / max(above_strength + below_strength, 1.0)
        
        return momentum
        
    def _ml_significance_enhancement(self):
        """Use ML to enhance square significance scoring"""
        
        all_squares = (
            self.price_squares + self.geometric_squares + self.harmonic_squares +
            self.fibonacci_squares + self.cardinal_squares
        )
        
        if len(all_squares) < 10 or not self.price_data:
            return
            
        try:
            # Extract enhanced features for each square
            enhanced_features = []
            
            for square in all_squares:
                
                # Basic features
                features = [
                    square.value,
                    square.square_root,
                    square.significance
                ]
                
                # Price action features around this square
                prices = self.price_data[:, 4]
                price_features = self._extract_price_action_features(square.value, prices)
                features.extend(price_features)
                
                # Relationship features
                rel_features = self._extract_relationship_features(square)
                features.extend(rel_features)
                
                enhanced_features.append(features)
                
            if enhanced_features:
                X = np.array(enhanced_features)
                
                # Use clustering to identify high-quality squares
                clusterer = KMeans(n_clusters=3, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(X)
                
                # Identify the "high quality" cluster
                cluster_qualities = []
                for cluster_id in range(3):
                    cluster_squares = [all_squares[i] for i in range(len(all_squares)) if cluster_labels[i] == cluster_id]
                    avg_significance = np.mean([s.significance for s in cluster_squares])
                    cluster_qualities.append(avg_significance)
                    
                best_cluster = np.argmax(cluster_qualities)
                
                # Enhance significance for squares in best cluster
                for i, square in enumerate(all_squares):
                    if cluster_labels[i] == best_cluster:
                        square.significance = min(1.0, square.significance * 1.1)
                        
        except Exception as e:
            self.logger.debug(f"ML significance enhancement failed: {e}")
            
    def _extract_price_action_features(self, level: float, prices: np.ndarray) -> List[float]:
        """Extract price action features around a square level"""
        
        if len(prices) == 0:
            return [0.0] * 5
            
        tolerance = level * 0.01  # 1% tolerance
        
        # Count different types of price action
        touches = sum(1 for p in prices if abs(p - level) <= tolerance)
        above_level = sum(1 for p in prices if p > level + tolerance)
        below_level = sum(1 for p in prices if p < level - tolerance)
        
        # Calculate ratios
        total_points = len(prices)
        touch_ratio = touches / total_points
        above_ratio = above_level / total_points
        below_ratio = below_level / total_points
        
        # Calculate volatility around level
        level_prices = [p for p in prices if abs(p - level) <= tolerance * 3]
        volatility = np.std(level_prices) / level if level_prices and level > 0 else 0
        
        # Recent price position relative to level
        recent_position = (prices[-1] - level) / level if len(prices) > 0 and level > 0 else 0
        
        return [touch_ratio, above_ratio, below_ratio, volatility, recent_position]
        
    def _extract_relationship_features(self, square: SquarePoint) -> List[float]:
        """Extract relationship features for a square"""
        
        # Count relationships involving this square
        relationships_count = sum(1 for r in self.square_relationships 
                                if r.point1 == square or r.point2 == square)
        
        # Average relationship strength
        related_relationships = [r for r in self.square_relationships 
                               if r.point1 == square or r.point2 == square]
        avg_rel_strength = np.mean([r.strength for r in related_relationships]) if related_relationships else 0.0
        
        # Count progressions involving this square
        progression_count = 0
        for prog in self.geometric_progressions:
            if square.value in prog.levels:
                progression_count += 1
                
        # Square type encoding
        type_features = [
            1.0 if square.square_type == SquareType.PRICE_SQUARE else 0.0,
            1.0 if square.square_type == SquareType.GEOMETRIC_SQUARE else 0.0,
            1.0 if square.square_type == SquareType.HARMONIC_SQUARE else 0.0,
            1.0 if square.square_type == SquareType.FIBONACCI_SQUARE else 0.0,
            1.0 if square.square_type == SquareType.CARDINAL_SQUARE else 0.0
        ]
        
        return [relationships_count, avg_rel_strength, progression_count] + type_features
        
    def _update_square_significance(self):
        """Update square significance based on recent price action"""
        
        if not self.price_data:
            return
            
        current_time = self.price_data[-1, 0]
        current_price = self.price_data[-1, 4]
        
        # Update significance for all squares
        all_squares = [
            self.price_squares, self.time_squares, self.geometric_squares,
            self.harmonic_squares, self.fibonacci_squares, self.cardinal_squares
        ]
        
        for square_list in all_squares:
            for square in square_list:
                
                # Update support/resistance counts
                self._update_support_resistance_counts(square)
                
                # Update last touch time
                tolerance = square.value * 0.01
                if abs(current_price - square.value) <= tolerance:
                    square.last_touch_time = current_time
                    
                # Adjust significance based on recency and activity
                self._adjust_significance_for_activity(square, current_time)
                
    def _update_support_resistance_counts(self, square: SquarePoint):
        """Update support and resistance counts for a square"""
        
        if not self.price_data:
            return
            
        prices = self.price_data[:, 4]
        tolerance = square.value * 0.01
        
        support_count = 0
        resistance_count = 0
        
        for i in range(1, len(prices)):
            
            # Check if price bounced off the level
            if (abs(prices[i] - square.value) <= tolerance and
                abs(prices[i-1] - square.value) > tolerance):
                
                # Determine if it was support or resistance
                if prices[i-1] > square.value and prices[i] > square.value:
                    support_count += 1
                elif prices[i-1] < square.value and prices[i] < square.value:
                    resistance_count += 1
                    
        square.support_count = support_count
        square.resistance_count = resistance_count
        
    def _adjust_significance_for_activity(self, square: SquarePoint, current_time: float):
        """Adjust square significance based on recent activity"""
        
        base_significance = square.significance
        
        # Recent activity bonus
        if square.last_touch_time:
            time_since_touch = current_time - square.last_touch_time
            days_since_touch = time_since_touch / (24 * 3600)
            
            if days_since_touch <= 7:  # Recent touch
                recency_bonus = 0.1 * (1 - days_since_touch / 7)
                square.significance = min(1.0, base_significance + recency_bonus)
            elif days_since_touch > 30:  # Old touch
                recency_penalty = 0.05
                square.significance = max(0.0, base_significance - recency_penalty)
                
        # Support/resistance activity bonus
        total_activity = square.support_count + square.resistance_count
        if total_activity > 0:
            activity_bonus = min(0.15, total_activity * 0.02)
            square.significance = min(1.0, square.significance + activity_bonus)
            
    def _generate_projections(self):
        """Generate projections based on square analysis"""
        
        current_price = self.price_data[-1, 4] if self.price_data is not None else 0
        
        # Generate projections from geometric progressions
        self._generate_progression_projections(current_price)
        
        # Generate projections from square relationships
        self._generate_relationship_projections(current_price)
        
        # Generate ML-based projections if available
        if self.level_predictor:
            self._generate_ml_projections(current_price)
            
        # Sort projections by confidence
        self.projections.sort(key=lambda p: p.confidence, reverse=True)
        
        # Limit number of projections
        max_projections = 15
        if len(self.projections) > max_projections:
            self.projections = self.projections[:max_projections]
            
        self.logger.info(f"Generated {len(self.projections)} square projections")
        
    def _generate_progression_projections(self, current_price: float):
        """Generate projections from geometric progressions"""
        
        for progression in self.geometric_progressions:
            
            if progression.strength < 0.5:
                continue
                
            # Project next level in progression
            last_level = progression.levels[-1]
            next_level = last_level * progression.progression_ratio
            
            # Only project if it's a reasonable distance from current price
            distance_ratio = abs(next_level - current_price) / current_price
            
            if 0.02 <= distance_ratio <= 0.5:  # 2% to 50% move
                
                confidence = progression.strength * 0.8  # Reduce confidence for projections
                
                projection = SquareProjection(
                    target_level=next_level,
                    projection_type=f"geometric_progression_{progression.square_type.value}",
                    confidence=confidence,
                    supporting_progression=progression
                )
                
                self.projections.append(projection)
                
    def _generate_relationship_projections(self, current_price: float):
        """Generate projections from square relationships"""
        
        for relationship in self.square_relationships[:10]:  # Top 10 relationships
            
            if relationship.strength < 0.6:
                continue
                
            # Project based on relationship pattern
            higher_square = relationship.point2
            lower_square = relationship.point1
            
            # Project beyond higher square
            next_level = higher_square.value * relationship.ratio
            
            # Check if projection is reasonable
            distance_ratio = abs(next_level - current_price) / current_price
            
            if 0.02 <= distance_ratio <= 0.4:  # 2% to 40% move
                
                confidence = relationship.strength * 0.7
                
                projection = SquareProjection(
                    target_level=next_level,
                    projection_type=f"relationship_{relationship.relationship_type}",
                    confidence=confidence,
                    source_square=higher_square
                )
                
                self.projections.append(projection)
                
    def _generate_ml_projections(self, current_price: float):
        """Generate ML-based projections"""
        
        try:
            if not self.price_data or len(self.price_data) < 15:
                return
                
            # Prepare current features
            current_features = self._extract_current_features_for_ml()
            
            if not current_features:
                return
                
            # Predict price changes for different horizons
            feature_vector = np.array(current_features).reshape(1, -1)
            predicted_change = self.level_predictor.predict(feature_vector)[0]
            
            # Convert to target price
            target_price = current_price * (1 + predicted_change)
            
            # Find nearest square to target
            all_squares = (
                self.price_squares + self.geometric_squares + 
                self.harmonic_squares + self.fibonacci_squares
            )
            
            if all_squares:
                distances = [abs(s.value - target_price) for s in all_squares]
                nearest_square_idx = np.argmin(distances)
                nearest_square = all_squares[nearest_square_idx]
                
                # Adjust target to nearest significant square
                if distances[nearest_square_idx] / target_price < 0.05:  # Within 5%
                    target_price = nearest_square.value
                    
            # Calculate confidence
            confidence = 0.5  # Base ML confidence
            
            # Check distance reasonableness
            distance_ratio = abs(target_price - current_price) / current_price
            
            if 0.01 <= distance_ratio <= 0.3:  # 1% to 30% move
                
                projection = SquareProjection(
                    target_level=target_price,
                    projection_type="ml_prediction",
                    confidence=confidence
                )
                
                self.projections.append(projection)
                
        except Exception as e:
            self.logger.debug(f"ML projection generation failed: {e}")
            
    def _extract_current_features_for_ml(self) -> Optional[List[float]]:
        """Extract current features for ML prediction"""
        
        try:
            if not self.price_data or len(self.price_data) < 10:
                return None
                
            prices = self.price_data[:, 4]
            recent_prices = prices[-10:]
            
            # Same features as training
            features = [
                np.mean(recent_prices),
                np.std(recent_prices),
                recent_prices[-1] - recent_prices[0],
                (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2] if recent_prices[-2] != 0 else 0
            ]
            
            # Current square features
            current_price = prices[-1]
            all_squares = self.price_squares + self.geometric_squares
            
            if all_squares:
                distances = [abs(s.value - current_price) / current_price for s in all_squares]
                min_distance = min(distances)
                features.append(min_distance)
            else:
                features.append(1.0)
                
            # Current square momentum
            square_momentum = self._calculate_square_momentum_at_point(len(prices) - 1)
            features.append(square_momentum)
            
            return features
            
        except Exception:
            return None        
    def _generate_trading_signals(self):
        """Generate trading signals based on square analysis"""
        
        self.trading_signals.clear()
        
        if not self.price_data:
            return
            
        current_price = self.price_data[-1, 4]
        current_time = self.price_data[-1, 0]
        
        # Generate signals from different sources
        self._generate_square_level_signals(current_price, current_time)
        self._generate_progression_signals(current_price, current_time)
        self._generate_relationship_signals(current_price, current_time)
        self._generate_projection_signals(current_price, current_time)
        
        # Sort signals by strength
        self.trading_signals.sort(key=lambda s: s.strength, reverse=True)
        
        # Limit signals
        max_signals = 5
        if len(self.trading_signals) > max_signals:
            self.trading_signals = self.trading_signals[:max_signals]
            
        self.logger.info(f"Generated {len(self.trading_signals)} trading signals")
        
    def _generate_square_level_signals(self, current_price: float, current_time: float):
        """Generate signals based on square level interactions"""
        
        # Combine all significant squares
        all_squares = []
        for square_list in [self.price_squares, self.geometric_squares, 
                           self.harmonic_squares, self.fibonacci_squares, self.cardinal_squares]:
            all_squares.extend([s for s in square_list if s.significance >= 0.6])
            
        for square in all_squares:
            
            # Check proximity to square level
            distance_pct = abs(current_price - square.value) / square.value
            
            if distance_pct <= 0.02:  # Within 2% of square level
                
                # Determine signal direction
                signal_direction = self._determine_square_signal_direction(square, current_price)
                
                if signal_direction:
                    
                    # Calculate signal strength
                    strength = self._calculate_square_signal_strength(square, distance_pct)
                    
                    signal = TradingSignal(
                        signal_type=signal_direction,
                        strength=strength,
                        entry_price=current_price,
                        target_price=self._calculate_square_target(square, signal_direction),
                        stop_loss=self._calculate_square_stop_loss(square, signal_direction, current_price),
                        timestamp=current_time,
                        source_type="square_level",
                        source_data={"square": square}
                    )
                    
                    self.trading_signals.append(signal)
                    
    def _determine_square_signal_direction(self, square: SquarePoint, current_price: float) -> Optional[SignalType]:
        """Determine signal direction at a square level"""
        
        # Consider historical support/resistance behavior
        total_interactions = square.support_count + square.resistance_count
        
        if total_interactions == 0:
            return None
            
        support_ratio = square.support_count / total_interactions
        resistance_ratio = square.resistance_count / total_interactions
        
        # Strong support level
        if support_ratio >= 0.7 and current_price <= square.value:
            return SignalType.BULLISH
            
        # Strong resistance level
        if resistance_ratio >= 0.7 and current_price >= square.value:
            return SignalType.BEARISH
            
        # Breakout scenarios
        if current_price > square.value and resistance_ratio >= 0.6:
            # Resistance breakout
            return SignalType.BULLISH
            
        if current_price < square.value and support_ratio >= 0.6:
            # Support breakdown
            return SignalType.BEARISH
            
        return None
        
    def _calculate_square_signal_strength(self, square: SquarePoint, distance_pct: float) -> float:
        """Calculate signal strength for square level"""
        
        # Base strength from square significance
        base_strength = square.significance
        
        # Proximity bonus (closer = stronger)
        proximity_bonus = (0.02 - distance_pct) / 0.02 * 0.2
        
        # Historical activity bonus
        total_interactions = square.support_count + square.resistance_count
        activity_bonus = min(0.15, total_interactions * 0.03)
        
        # Square type bonus
        type_bonus = 0.0
        if square.square_type in [SquareType.CARDINAL_SQUARE, SquareType.FIBONACCI_SQUARE]:
            type_bonus = 0.1
            
        strength = base_strength + proximity_bonus + activity_bonus + type_bonus
        
        return max(0.0, min(1.0, strength))
        
    def _calculate_square_target(self, square: SquarePoint, signal_direction: SignalType) -> float:
        """Calculate target price for square-based signal"""
        
        # Find next significant square in signal direction
        all_squares = (
            self.price_squares + self.geometric_squares + 
            self.harmonic_squares + self.fibonacci_squares + self.cardinal_squares
        )
        
        significant_squares = [s for s in all_squares if s.significance >= 0.5]
        
        if signal_direction == SignalType.BULLISH:
            # Find next higher square
            higher_squares = [s for s in significant_squares if s.value > square.value]
            if higher_squares:
                return min(s.value for s in higher_squares)
            else:
                return square.value * 1.1  # 10% target
        else:
            # Find next lower square
            lower_squares = [s for s in significant_squares if s.value < square.value]
            if lower_squares:
                return max(s.value for s in lower_squares)
            else:
                return square.value * 0.9  # 10% target
                
        return square.value
        
    def _calculate_square_stop_loss(self, square: SquarePoint, signal_direction: SignalType, current_price: float) -> float:
        """Calculate stop loss for square-based signal"""
        
        # Conservative stop loss based on square level
        stop_distance_pct = 0.03  # 3% default
        
        # Adjust based on square significance
        if square.significance >= 0.8:
            stop_distance_pct = 0.02  # Tighter stop for strong levels
        elif square.significance <= 0.6:
            stop_distance_pct = 0.04  # Wider stop for weak levels
            
        if signal_direction == SignalType.BULLISH:
            return current_price * (1 - stop_distance_pct)
        else:
            return current_price * (1 + stop_distance_pct)
            
    def _generate_progression_signals(self, current_price: float, current_time: float):
        """Generate signals based on geometric progressions"""
        
        for progression in self.geometric_progressions:
            
            if progression.strength < 0.6:
                continue
                
            # Check if price is near a progression level
            for level in progression.levels:
                distance_pct = abs(current_price - level) / level
                
                if distance_pct <= 0.025:  # Within 2.5%
                    
                    # Determine direction based on progression trend
                    signal_direction = self._determine_progression_direction(progression, level, current_price)
                    
                    if signal_direction:
                        
                        strength = progression.strength * 0.8  # Reduce for progression signals
                        
                        signal = TradingSignal(
                            signal_type=signal_direction,
                            strength=strength,
                            entry_price=current_price,
                            target_price=self._calculate_progression_target(progression, level, signal_direction),
                            stop_loss=self._calculate_progression_stop_loss(level, signal_direction, current_price),
                            timestamp=current_time,
                            source_type="geometric_progression",
                            source_data={"progression": progression, "level": level}
                        )
                        
                        self.trading_signals.append(signal)
                        
    def _determine_progression_direction(self, progression: GeometricProgression, level: float, current_price: float) -> Optional[SignalType]:
        """Determine signal direction for progression level"""
        
        # Find position in progression
        level_index = None
        for i, prog_level in enumerate(progression.levels):
            if abs(prog_level - level) / level < 0.001:
                level_index = i
                break
                
        if level_index is None:
            return None
            
        # If at intermediate level, expect continuation
        if 0 < level_index < len(progression.levels) - 1:
            if progression.progression_ratio > 1.0:
                return SignalType.BULLISH  # Upward progression
            else:
                return SignalType.BEARISH  # Downward progression
                
        # If at boundary level, expect reversal or continuation based on strength
        if level_index == 0:  # At start of progression
            if progression.strength >= 0.8:
                return SignalType.BULLISH  # Expect progression to continue
        elif level_index == len(progression.levels) - 1:  # At end of progression
            if progression.strength >= 0.8:
                return SignalType.BEARISH  # Expect reversal or pause
                
        return None
        
    def _calculate_progression_target(self, progression: GeometricProgression, current_level: float, signal_direction: SignalType) -> float:
        """Calculate target for progression-based signal"""
        
        if signal_direction == SignalType.BULLISH:
            return current_level * progression.progression_ratio
        else:
            return current_level / progression.progression_ratio
            
    def _calculate_progression_stop_loss(self, level: float, signal_direction: SignalType, current_price: float) -> float:
        """Calculate stop loss for progression-based signal"""
        
        stop_distance_pct = 0.025  # 2.5% for progression signals
        
        if signal_direction == SignalType.BULLISH:
            return current_price * (1 - stop_distance_pct)
        else:
            return current_price * (1 + stop_distance_pct)
            
    def _generate_relationship_signals(self, current_price: float, current_time: float):
        """Generate signals based on square relationships"""
        
        for relationship in self.square_relationships[:5]:  # Top 5 relationships
            
            if relationship.strength < 0.7:
                continue
                
            # Check if price is near either point in relationship
            point1_distance = abs(current_price - relationship.point1.value) / relationship.point1.value
            point2_distance = abs(current_price - relationship.point2.value) / relationship.point2.value
            
            near_point1 = point1_distance <= 0.02
            near_point2 = point2_distance <= 0.02
            
            if near_point1 or near_point2:
                
                active_point = relationship.point1 if near_point1 else relationship.point2
                target_point = relationship.point2 if near_point1 else relationship.point1
                
                # Determine signal direction
                if current_price <= active_point.value:
                    signal_direction = SignalType.BULLISH
                    target_price = target_point.value
                else:
                    signal_direction = SignalType.BEARISH
                    target_price = target_point.value
                    
                strength = relationship.strength * 0.75  # Reduce for relationship signals
                
                signal = TradingSignal(
                    signal_type=signal_direction,
                    strength=strength,
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=self._calculate_relationship_stop_loss(active_point.value, signal_direction, current_price),
                    timestamp=current_time,
                    source_type="square_relationship",
                    source_data={"relationship": relationship, "active_point": active_point}
                )
                
                self.trading_signals.append(signal)
                
    def _calculate_relationship_stop_loss(self, level: float, signal_direction: SignalType, current_price: float) -> float:
        """Calculate stop loss for relationship-based signal"""
        
        stop_distance_pct = 0.03  # 3% for relationship signals
        
        if signal_direction == SignalType.BULLISH:
            return current_price * (1 - stop_distance_pct)
        else:
            return current_price * (1 + stop_distance_pct)
            
    def _generate_projection_signals(self, current_price: float, current_time: float):
        """Generate signals based on square projections"""
        
        for projection in self.projections[:3]:  # Top 3 projections
            
            if projection.confidence < 0.6:
                continue
                
            # Only generate signal if projection is reasonable distance
            distance_pct = abs(projection.target_level - current_price) / current_price
            
            if 0.02 <= distance_pct <= 0.15:  # 2% to 15% move
                
                signal_direction = SignalType.BULLISH if projection.target_level > current_price else SignalType.BEARISH
                
                strength = projection.confidence * 0.7  # Reduce for projection signals
                
                signal = TradingSignal(
                    signal_type=signal_direction,
                    strength=strength,
                    entry_price=current_price,
                    target_price=projection.target_level,
                    stop_loss=self._calculate_projection_stop_loss(signal_direction, current_price, distance_pct),
                    timestamp=current_time,
                    source_type="square_projection",
                    source_data={"projection": projection}
                )
                
                self.trading_signals.append(signal)
                
    def _calculate_projection_stop_loss(self, signal_direction: SignalType, current_price: float, distance_pct: float) -> float:
        """Calculate stop loss for projection-based signal"""
        
        # Stop loss proportional to projection distance
        stop_distance_pct = min(0.04, distance_pct * 0.3)  # Max 4%, proportional to target
        
        if signal_direction == SignalType.BULLISH:
            return current_price * (1 - stop_distance_pct)
        else:
            return current_price * (1 + stop_distance_pct)
            
    # Analysis and utility methods
    def get_current_square_analysis(self) -> Dict[str, Any]:
        """Get current square analysis summary"""
        
        if not self.price_data:
            return {}
            
        current_price = self.price_data[-1, 4]
        
        analysis = {
            'current_price': current_price,
            'square_counts': {
                'price_squares': len(self.price_squares),
                'time_squares': len(self.time_squares),
                'geometric_squares': len(self.geometric_squares),
                'harmonic_squares': len(self.harmonic_squares),
                'fibonacci_squares': len(self.fibonacci_squares),
                'cardinal_squares': len(self.cardinal_squares)
            },
            'analysis_counts': {
                'relationships': len(self.square_relationships),
                'progressions': len(self.geometric_progressions),
                'projections': len(self.projections),
                'trading_signals': len(self.trading_signals)
            },
            'nearest_squares': self._find_nearest_squares(current_price),
            'key_levels': self._identify_key_levels(),
            'market_structure': self._assess_market_structure()
        }
        
        return analysis
        
    def _find_nearest_squares(self, current_price: float) -> Dict[str, Any]:
        """Find nearest squares above and below current price"""
        
        all_squares = (
            self.price_squares + self.geometric_squares + 
            self.harmonic_squares + self.fibonacci_squares + self.cardinal_squares
        )
        
        if not all_squares:
            return {}
            
        # Separate squares above and below current price
        above_squares = [s for s in all_squares if s.value > current_price]
        below_squares = [s for s in all_squares if s.value < current_price]
        
        # Sort and get nearest
        above_squares.sort(key=lambda s: s.value)
        below_squares.sort(key=lambda s: s.value, reverse=True)
        
        result = {}
        
        if above_squares:
            nearest_above = above_squares[0]
            result['nearest_resistance'] = {
                'value': nearest_above.value,
                'distance_pct': (nearest_above.value - current_price) / current_price * 100,
                'significance': nearest_above.significance,
                'type': nearest_above.square_type.value
            }
            
        if below_squares:
            nearest_below = below_squares[0]
            result['nearest_support'] = {
                'value': nearest_below.value,
                'distance_pct': (current_price - nearest_below.value) / current_price * 100,
                'significance': nearest_below.significance,
                'type': nearest_below.square_type.value
            }
            
        return result
        
    def _identify_key_levels(self) -> List[Dict[str, Any]]:
        """Identify key square levels"""
        
        all_squares = (
            self.price_squares + self.geometric_squares + 
            self.harmonic_squares + self.fibonacci_squares + self.cardinal_squares
        )
        
        # Filter for highly significant squares
        key_squares = [s for s in all_squares if s.significance >= 0.7]
        
        # Sort by significance
        key_squares.sort(key=lambda s: s.significance, reverse=True)
        
        # Return top key levels
        key_levels = []
        for square in key_squares[:10]:
            level_info = {
                'value': square.value,
                'significance': square.significance,
                'type': square.square_type.value,
                'level': square.square_level.value,
                'support_count': square.support_count,
                'resistance_count': square.resistance_count
            }
            key_levels.append(level_info)
            
        return key_levels
        
    def _assess_market_structure(self) -> Dict[str, Any]:
        """Assess overall market structure based on squares"""
        
        if not self.price_data:
            return {}
            
        current_price = self.price_data[-1, 4]
        
        all_squares = (
            self.price_squares + self.geometric_squares + 
            self.harmonic_squares + self.fibonacci_squares + self.cardinal_squares
        )
        
        if not all_squares:
            return {}
            
        # Calculate square distribution
        above_squares = [s for s in all_squares if s.value > current_price]
        below_squares = [s for s in all_squares if s.value < current_price]
        
        above_significance = sum(s.significance for s in above_squares)
        below_significance = sum(s.significance for s in below_squares)
        total_significance = above_significance + below_significance
        
        structure = {}
        
        if total_significance > 0:
            structure['resistance_density'] = above_significance / total_significance
            structure['support_density'] = below_significance / total_significance
            
            # Assess bias
            if structure['resistance_density'] > 0.6:
                structure['bias'] = 'bearish'
            elif structure['support_density'] > 0.6:
                structure['bias'] = 'bullish'
            else:
                structure['bias'] = 'neutral'
                
        # Assess progression trends
        progression_bias = 0.0
        for progression in self.geometric_progressions:
            if progression.progression_ratio > 1.0:
                progression_bias += progression.strength
            else:
                progression_bias -= progression.strength
                
        if len(self.geometric_progressions) > 0:
            structure['progression_bias'] = progression_bias / len(self.geometric_progressions)
        else:
            structure['progression_bias'] = 0.0
            
        return structure
        
    def get_trading_recommendation(self) -> Dict[str, Any]:
        """Get current trading recommendation based on square analysis"""
        
        if not self.trading_signals:
            return {'recommendation': 'NEUTRAL', 'confidence': 0.0, 'reason': 'No signals generated'}
            
        # Get strongest signal
        strongest_signal = self.trading_signals[0]
        
        # Analyze signal consensus
        bullish_signals = [s for s in self.trading_signals if s.signal_type == SignalType.BULLISH]
        bearish_signals = [s for s in self.trading_signals if s.signal_type == SignalType.BEARISH]
        
        bullish_strength = sum(s.strength for s in bullish_signals)
        bearish_strength = sum(s.strength for s in bearish_signals)
        
        # Determine recommendation
        if bullish_strength > bearish_strength * 1.5:
            recommendation = 'BUY'
            confidence = min(1.0, bullish_strength / len(self.trading_signals))
        elif bearish_strength > bullish_strength * 1.5:
            recommendation = 'SELL'
            confidence = min(1.0, bearish_strength / len(self.trading_signals))
        else:
            recommendation = 'NEUTRAL'
            confidence = 0.5
            
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'primary_signal': {
                'type': strongest_signal.signal_type.value,
                'strength': strongest_signal.strength,
                'entry': strongest_signal.entry_price,
                'target': strongest_signal.target_price,
                'stop_loss': strongest_signal.stop_loss,
                'source': strongest_signal.source_type
            },
            'signal_summary': {
                'total_signals': len(self.trading_signals),
                'bullish_signals': len(bullish_signals),
                'bearish_signals': len(bearish_signals),
                'bullish_strength': bullish_strength,
                'bearish_strength': bearish_strength
            }
        }


def run_gann_square_demo():
    """Demonstrate the Gann Square indicator"""
    
    print("=== Gann Square Indicator Demo ===")
    
    # Create sample OHLCV data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Generate realistic price data with trends
    base_price = 100.0
    price_data = []
    
    for i, date in enumerate(dates):
        # Add trend and noise
        trend = 0.02 * np.sin(i * 0.01) + 0.001 * i
        noise = np.random.normal(0, 0.02)
        
        price = base_price * (1 + trend + noise)
        
        # OHLCV format
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.randint(1000, 10000)
        
        price_data.append([
            date.timestamp(),
            price * 0.99,  # Open
            high,           # High
            low,            # Low
            price,          # Close
            volume          # Volume
        ])
        
        base_price = price
        
    ohlcv_data = np.array(price_data)
    
    # Initialize indicator
    indicator = GannSquareIndicator()
    
    # Update with data
    print("Updating indicator with sample data...")
    result = indicator.update(ohlcv_data)
    
    print(f"\nIndicator updated successfully: {result}")
    
    # Get analysis
    analysis = indicator.get_current_square_analysis()
    
    print(f"\n=== Square Analysis ===")
    print(f"Current Price: ${analysis.get('current_price', 0):.2f}")
    
    square_counts = analysis.get('square_counts', {})
    print(f"\n=== Square Counts ===")
    for square_type, count in square_counts.items():
        print(f"{square_type}: {count}")
        
    analysis_counts = analysis.get('analysis_counts', {})
    print(f"\n=== Analysis Counts ===")
    for analysis_type, count in analysis_counts.items():
        print(f"{analysis_type}: {count}")
        
    # Show nearest squares
    nearest = analysis.get('nearest_squares', {})
    if nearest:
        print(f"\n=== Nearest Squares ===")
        if 'nearest_support' in nearest:
            support = nearest['nearest_support']
            print(f"Support: ${support['value']:.2f} ({support['distance_pct']:.2f}% below, significance: {support['significance']:.2f})")
            
        if 'nearest_resistance' in nearest:
            resistance = nearest['nearest_resistance']
            print(f"Resistance: ${resistance['value']:.2f} ({resistance['distance_pct']:.2f}% above, significance: {resistance['significance']:.2f})")
            
    # Show key levels
    key_levels = analysis.get('key_levels', [])
    if key_levels:
        print(f"\n=== Key Square Levels ===")
        for i, level in enumerate(key_levels[:5]):
            print(f"{i+1}. ${level['value']:.2f} - {level['type']} (sig: {level['significance']:.2f})")
            
    # Show market structure
    structure = analysis.get('market_structure', {})
    if structure:
        print(f"\n=== Market Structure ===")
        print(f"Bias: {structure.get('bias', 'unknown')}")
        print(f"Support Density: {structure.get('support_density', 0):.2f}")
        print(f"Resistance Density: {structure.get('resistance_density', 0):.2f}")
        print(f"Progression Bias: {structure.get('progression_bias', 0):.2f}")
        
    # Get trading recommendation
    recommendation = indicator.get_trading_recommendation()
    
    print(f"\n=== Trading Recommendation ===")
    print(f"Recommendation: {recommendation['recommendation']}")
    print(f"Confidence: {recommendation['confidence']:.2f}")
    print(f"Reason: {recommendation.get('reason', 'Based on signal analysis')}")
    
    if 'primary_signal' in recommendation:
        signal = recommendation['primary_signal']
        print(f"\n=== Primary Signal ===")
        print(f"Type: {signal['type']}")
        print(f"Strength: {signal['strength']:.2f}")
        print(f"Entry: ${signal['entry']:.2f}")
        print(f"Target: ${signal['target']:.2f}")
        print(f"Stop Loss: ${signal['stop_loss']:.2f}")
        print(f"Source: {signal['source']}")
        
    signal_summary = recommendation.get('signal_summary', {})
    if signal_summary:
        print(f"\n=== Signal Summary ===")
        print(f"Total Signals: {signal_summary['total_signals']}")
        print(f"Bullish: {signal_summary['bullish_signals']} (strength: {signal_summary['bullish_strength']:.2f})")
        print(f"Bearish: {signal_summary['bearish_signals']} (strength: {signal_summary['bearish_strength']:.2f})")
        
    print(f"\n=== Demo Complete ===")


if __name__ == "__main__":
    run_gann_square_demo()