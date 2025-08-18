"""
Advanced Gann Square of Nine Indicator

This module implements a sophisticated Gann Square of Nine analysis system with advanced
mathematical calculations, spiral analysis, numerological relationships, and ML-enhanced
pattern recognition for humanitarian trading applications.

The Square of Nine is one of W.D. Gann's most famous tools, based on the mathematical
concept of squaring numbers arranged in a spiral pattern. This implementation features:

Key Features:
- Comprehensive Square of Nine calculations (clockwise/counter-clockwise spirals)
- Advanced spiral analysis and pattern recognition
- Numerological relationship detection and analysis
- Dynamic support/resistance level identification
- ML-enhanced pattern classification and prediction
- Harmonic analysis and geometric progression detection
- Multi-timeframe spiral analysis capabilities
- Advanced price projection and forecasting
- Sophisticated trading signal generation
- Real-time spiral position tracking and analysis

Author: Trading Team
Date: 2024-12-28
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import math
from collections import defaultdict, deque

# Optional ML imports
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SpiralDirection(Enum):
    """Square of Nine spiral directions"""
    CLOCKWISE = "clockwise"
    COUNTER_CLOCKWISE = "counter_clockwise"


class SpiralLevel(Enum):
    """Spiral level significance"""
    MAJOR = "major"          # Major cardinal points
    INTERMEDIATE = "intermediate"  # Quarter points
    MINOR = "minor"          # Regular spiral positions


class PatternType(Enum):
    """Square of Nine pattern types"""
    CARDINAL_CROSS = "cardinal_cross"
    DIAGONAL_CROSS = "diagonal_cross"
    SPIRAL_ARM = "spiral_arm"
    CONCENTRIC_RING = "concentric_ring"
    GEOMETRIC_PROGRESSION = "geometric_progression"
    HARMONIC_RESONANCE = "harmonic_resonance"


class SignalType(Enum):
    """Trading signal types"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class SpiralPoint:
    """Represents a point in the Square of Nine spiral"""
    value: float
    position: int              # Position in spiral (0 = center)
    ring: int                  # Ring number (0 = center)
    angle: float               # Angle in degrees
    cardinal_direction: Optional[str]  # N, S, E, W, NE, NW, SE, SW
    significance: float = 0.0  # Significance score 0-1
    level: SpiralLevel = SpiralLevel.MINOR
    support_count: int = 0     # Historical support touches
    resistance_count: int = 0  # Historical resistance touches
    last_touch_time: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived properties"""
        self._calculate_cardinal_direction()
        self._calculate_significance()
        
    def _calculate_cardinal_direction(self):
        """Calculate cardinal direction based on angle"""
        angle = self.angle % 360
        
        if 337.5 <= angle or angle < 22.5:
            self.cardinal_direction = "E"
        elif 22.5 <= angle < 67.5:
            self.cardinal_direction = "NE"
        elif 67.5 <= angle < 112.5:
            self.cardinal_direction = "N"
        elif 112.5 <= angle < 157.5:
            self.cardinal_direction = "NW"
        elif 157.5 <= angle < 202.5:
            self.cardinal_direction = "W"
        elif 202.5 <= angle < 247.5:
            self.cardinal_direction = "SW"
        elif 247.5 <= angle < 292.5:
            self.cardinal_direction = "S"
        elif 292.5 <= angle < 337.5:
            self.cardinal_direction = "SE"
            
    def _calculate_significance(self):
        """Calculate base significance score"""
        # Cardinal directions are most significant
        if self.cardinal_direction in ["N", "S", "E", "W"]:
            self.significance = 0.9
            self.level = SpiralLevel.MAJOR
        elif self.cardinal_direction in ["NE", "NW", "SE", "SW"]:
            self.significance = 0.7
            self.level = SpiralLevel.INTERMEDIATE
        else:
            self.significance = 0.5
            self.level = SpiralLevel.MINOR


@dataclass
class SpiralPattern:
    """Represents a detected pattern in the Square of Nine"""
    pattern_type: PatternType
    points: List[SpiralPoint]
    strength: float
    confidence: float
    geometric_ratio: Optional[float] = None
    harmonic_level: Optional[int] = None
    time_discovered: float = field(default_factory=time.time)


@dataclass
class SpiralRelationship:
    """Represents a relationship between spiral points"""
    point1: SpiralPoint
    point2: SpiralPoint
    relationship_type: str
    angle_difference: float
    ring_difference: int
    ratio: float
    strength: float
    harmonic_resonance: Optional[float] = None


@dataclass
class SpiralProjection:
    """Represents a projected level from Square of Nine analysis"""
    target_level: float
    projection_type: str
    confidence: float
    source_point: Optional[SpiralPoint] = None
    supporting_pattern: Optional[SpiralPattern] = None
    time_projection: Optional[float] = None


@dataclass
class TradingSignal:
    """Represents a trading signal from Square of Nine analysis"""
    signal_type: SignalType
    strength: float
    entry_price: float
    target_price: float
    stop_loss: float
    timestamp: float
    source_type: str
    source_data: Dict[str, Any] = field(default_factory=dict)


class SquareOfNineConfig:
    """Configuration for Square of Nine indicator"""
    
    def __init__(self):
        # Basic configuration
        self.max_rings = 12              # Maximum spiral rings to calculate
        self.spiral_direction = SpiralDirection.CLOCKWISE
        self.center_value_mode = "auto"  # "auto", "manual", "significant_level"
        self.manual_center_value = 1.0
        
        # Pattern detection
        self.min_pattern_strength = 0.6
        self.pattern_lookback = 100
        self.enable_pattern_prediction = True
        
        # Numerological settings
        self.sacred_numbers = [1, 3, 7, 9, 12, 21, 49, 81, 144]
        self.harmonic_divisions = [2, 3, 4, 6, 8, 9, 12, 16, 24]
        self.cardinal_angles = [0, 90, 180, 270]  # Major cardinal points
        self.intermediate_angles = [45, 135, 225, 315]  # Diagonal points
        
        # ML configuration
        self.use_ml_enhancement = SKLEARN_AVAILABLE
        self.feature_scaling = True
        self.cluster_patterns = True
        self.ml_prediction_horizon = 20
        
        # Signal generation
        self.min_signal_strength = 0.6
        self.signal_confirmation_period = 3
        self.risk_reward_ratio = 2.0
        
        # Performance optimization
        self.cache_calculations = True
        self.max_cached_spirals = 10
        self.calculation_precision = 6


class GannSquareOfNineIndicator:
    """
    Advanced Gann Square of Nine Indicator
    
    Implements sophisticated Square of Nine analysis with spiral calculations,
    pattern recognition, and ML-enhanced prediction capabilities.
    """
    
    def __init__(self, config: Optional[SquareOfNineConfig] = None):
        """Initialize the Square of Nine indicator"""
        
        self.config = config or SquareOfNineConfig()
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.price_data: Optional[np.ndarray] = None
        self.current_center_value: float = 1.0
        
        # Spiral analysis components
        self.spiral_points: List[SpiralPoint] = []
        self.spiral_rings: Dict[int, List[SpiralPoint]] = {}
        self.cardinal_points: List[SpiralPoint] = []
        self.significant_levels: List[SpiralPoint] = []
        
        # Pattern analysis
        self.detected_patterns: List[SpiralPattern] = []
        self.spiral_relationships: List[SpiralRelationship] = []
        self.geometric_progressions: List[SpiralPattern] = []
        
        # Projections and signals
        self.projections: List[SpiralProjection] = []
        self.trading_signals: List[TradingSignal] = []
        
        # ML components
        if self.config.use_ml_enhancement:
            self.scaler = StandardScaler()
            self.pattern_classifier = None
            self.level_predictor = None
            
        # Caching
        self.spiral_cache: Dict[float, List[SpiralPoint]] = {}
        
        self.logger.info("Gann Square of Nine indicator initialized")
        
    def update(self, data: np.ndarray) -> bool:
        """
        Update the indicator with new OHLCV data
        
        Args:
            data: OHLCV data array with columns [timestamp, open, high, low, close, volume]
            
        Returns:
            bool: True if update was successful
        """
        
        try:
            if data is None or len(data) == 0:
                return False
                
            self.price_data = data.copy()
            
            # Determine center value
            self._determine_center_value()
            
            # Calculate spiral
            self._calculate_square_of_nine_spiral()
            
            # Analyze patterns
            self._analyze_spiral_patterns()
            
            # Detect relationships
            self._analyze_spiral_relationships()
            
            # Generate projections
            self._generate_projections()
            
            # Apply ML enhancement
            if self.config.use_ml_enhancement:
                self._ml_enhanced_analysis()
                
            # Update significance based on price action
            self._update_level_significance()
            
            # Generate trading signals
            self._generate_trading_signals()
            
            self.logger.debug(f"Square of Nine updated with {len(data)} data points")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating Square of Nine: {e}")
            return False
            
    def _determine_center_value(self):
        """Determine the center value for the Square of Nine"""
        
        if not self.price_data:
            self.current_center_value = 1.0
            return
            
        if self.config.center_value_mode == "manual":
            self.current_center_value = self.config.manual_center_value
            
        elif self.config.center_value_mode == "significant_level":
            # Use a significant price level as center
            prices = self.price_data[:, 4]  # Close prices
            self.current_center_value = self._find_significant_level(prices)
            
        else:  # auto mode
            # Use recent price action to determine center
            recent_prices = self.price_data[-20:, 4] if len(self.price_data) >= 20 else self.price_data[:, 4]
            
            # Find a good center value around recent price action
            current_price = self.price_data[-1, 4]
            price_range = np.max(recent_prices) - np.min(recent_prices)
            
            # Round to a "clean" number for better spiral calculations
            if current_price >= 1000:
                step = 100
            elif current_price >= 100:
                step = 10
            elif current_price >= 10:
                step = 1
            else:
                step = 0.1
                
            self.current_center_value = round(current_price / step) * step
            
        self.logger.debug(f"Center value determined: {self.current_center_value}")
        
    def _find_significant_level(self, prices: np.ndarray) -> float:
        """Find a significant price level to use as center"""
        
        # Look for levels with multiple touches
        level_counts = defaultdict(int)
        tolerance_pct = 0.01  # 1% tolerance
        
        for price in prices:
            # Round to significant levels
            for precision in [2, 1, 0]:
                level = round(price, precision)
                
                # Count touches within tolerance
                touches = sum(1 for p in prices if abs(p - level) / level <= tolerance_pct)
                if touches >= 3:  # At least 3 touches
                    level_counts[level] = touches
                    
        if level_counts:
            # Return most touched level
            return max(level_counts.items(), key=lambda x: x[1])[0]
        else:
            # Fallback to median
            return float(np.median(prices))
            
    def _calculate_square_of_nine_spiral(self):
        """Calculate the complete Square of Nine spiral"""
        
        # Check cache first
        cache_key = round(self.current_center_value, self.config.calculation_precision)
        if self.config.cache_calculations and cache_key in self.spiral_cache:
            self.spiral_points = self.spiral_cache[cache_key].copy()
            self._organize_spiral_data()
            return
            
        self.spiral_points.clear()
        
        # Start with center point
        center_point = SpiralPoint(
            value=self.current_center_value,
            position=0,
            ring=0,
            angle=0.0
        )
        self.spiral_points.append(center_point)
        
        # Calculate spiral rings
        for ring in range(1, self.config.max_rings + 1):
            ring_points = self._calculate_spiral_ring(ring)
            self.spiral_points.extend(ring_points)
            
        # Organize data structures
        self._organize_spiral_data()
        
        # Cache results
        if self.config.cache_calculations:
            if len(self.spiral_cache) >= self.config.max_cached_spirals:
                # Remove oldest entry
                oldest_key = min(self.spiral_cache.keys())
                del self.spiral_cache[oldest_key]
            self.spiral_cache[cache_key] = self.spiral_points.copy()
            
        self.logger.debug(f"Calculated spiral with {len(self.spiral_points)} points")
        
    def _calculate_spiral_ring(self, ring: int) -> List[SpiralPoint]:
        """Calculate points for a specific spiral ring"""
        
        ring_points = []
        
        # Calculate starting position for this ring
        # Each ring has 8 * ring points
        points_in_ring = 8 * ring
        
        # Starting value for ring (square of ring boundary)
        ring_start_position = sum(8 * r for r in range(1, ring)) + 1
        ring_start_value = self.current_center_value + ((ring_start_position - 1) ** 0.5) ** 2
        
        # Calculate angle increment for this ring
        angle_increment = 360.0 / points_in_ring
        
        for i in range(points_in_ring):
            
            # Calculate position in overall spiral
            position = ring_start_position + i
            
            # Calculate value using square root progression
            if self.config.spiral_direction == SpiralDirection.CLOCKWISE:
                angle = i * angle_increment
            else:
                angle = 360 - (i * angle_increment)
                
            # Calculate value based on position in spiral
            # Using Gann's square root progression
            sqrt_position = position ** 0.5
            value = self.current_center_value + (sqrt_position ** 2 - 1)
            
            # Adjust value for spiral direction and sacred number resonance
            value = self._apply_sacred_number_adjustment(value, position, ring)
            
            point = SpiralPoint(
                value=value,
                position=position,
                ring=ring,
                angle=angle
            )
            
            ring_points.append(point)
            
        return ring_points
        
    def _apply_sacred_number_adjustment(self, value: float, position: int, ring: int) -> float:
        """Apply sacred number adjustments to spiral values"""
        
        # Check if position or ring aligns with sacred numbers
        for sacred_num in self.config.sacred_numbers:
            
            # Position-based adjustment
            if position % sacred_num == 0:
                value *= (1 + 0.001 * sacred_num)  # Small adjustment
                
            # Ring-based adjustment
            if ring == sacred_num:
                value *= (1 + 0.002 * sacred_num)  # Slightly larger adjustment
                
        return value
        
    def _organize_spiral_data(self):
        """Organize spiral points into data structures"""
        
        # Clear existing data
        self.spiral_rings.clear()
        self.cardinal_points.clear()
        self.significant_levels.clear()
        
        # Organize by rings
        for point in self.spiral_points:
            if point.ring not in self.spiral_rings:
                self.spiral_rings[point.ring] = []
            self.spiral_rings[point.ring].append(point)
            
        # Identify cardinal points
        for point in self.spiral_points:
            if point.cardinal_direction in ["N", "S", "E", "W"]:
                self.cardinal_points.append(point)
                
        # Identify significant levels
        for point in self.spiral_points:
            if (point.significance >= 0.7 or 
                point.level in [SpiralLevel.MAJOR, SpiralLevel.INTERMEDIATE]):
                self.significant_levels.append(point)
                
        # Sort significant levels by value
        self.significant_levels.sort(key=lambda p: p.value)
        
        self.logger.debug(f"Organized spiral: {len(self.spiral_rings)} rings, "
                         f"{len(self.cardinal_points)} cardinal points, "
                         f"{len(self.significant_levels)} significant levels")        
    def _analyze_spiral_patterns(self):
        """Analyze patterns in the Square of Nine spiral"""
        
        self.detected_patterns.clear()
        
        # Detect different types of patterns
        self._detect_cardinal_cross_patterns()
        self._detect_diagonal_cross_patterns()
        self._detect_spiral_arm_patterns()
        self._detect_concentric_ring_patterns()
        self._detect_geometric_progressions()
        self._detect_harmonic_resonance_patterns()
        
        # Sort patterns by strength
        self.detected_patterns.sort(key=lambda p: p.strength, reverse=True)
        
        # Limit patterns for performance
        max_patterns = 20
        if len(self.detected_patterns) > max_patterns:
            self.detected_patterns = self.detected_patterns[:max_patterns]
            
        self.logger.debug(f"Detected {len(self.detected_patterns)} spiral patterns")
        
    def _detect_cardinal_cross_patterns(self):
        """Detect cardinal cross patterns (N-S, E-W alignments)"""
        
        # Group cardinal points by direction
        cardinal_groups = {"N": [], "S": [], "E": [], "W": []}
        
        for point in self.cardinal_points:
            if point.cardinal_direction in cardinal_groups:
                cardinal_groups[point.cardinal_direction].append(point)
                
        # Detect N-S axis patterns
        if cardinal_groups["N"] and cardinal_groups["S"]:
            self._analyze_axis_pattern(cardinal_groups["N"], cardinal_groups["S"], "N-S")
            
        # Detect E-W axis patterns
        if cardinal_groups["E"] and cardinal_groups["W"]:
            self._analyze_axis_pattern(cardinal_groups["E"], cardinal_groups["W"], "E-W")
            
    def _analyze_axis_pattern(self, group1: List[SpiralPoint], group2: List[SpiralPoint], axis_name: str):
        """Analyze patterns along a cardinal axis"""
        
        # Combine and sort points by value
        all_points = group1 + group2
        all_points.sort(key=lambda p: p.value)
        
        if len(all_points) < 3:
            return
            
        # Look for geometric progressions along axis
        for i in range(len(all_points) - 2):
            for j in range(i + 1, len(all_points) - 1):
                for k in range(j + 1, len(all_points)):
                    
                    p1, p2, p3 = all_points[i], all_points[j], all_points[k]
                    
                    # Check for geometric progression
                    if p1.value > 0 and p2.value > 0:
                        ratio1 = p2.value / p1.value
                        ratio2 = p3.value / p2.value
                        
                        # Allow some tolerance
                        if abs(ratio1 - ratio2) / max(ratio1, ratio2) < 0.1:
                            
                            # Calculate pattern strength
                            strength = self._calculate_pattern_strength([p1, p2, p3], axis_name)
                            
                            if strength >= self.config.min_pattern_strength:
                                
                                pattern = SpiralPattern(
                                    pattern_type=PatternType.CARDINAL_CROSS,
                                    points=[p1, p2, p3],
                                    strength=strength,
                                    confidence=0.8,  # High confidence for cardinal patterns
                                    geometric_ratio=(ratio1 + ratio2) / 2
                                )
                                
                                self.detected_patterns.append(pattern)
                                
    def _detect_diagonal_cross_patterns(self):
        """Detect diagonal cross patterns (NE-SW, NW-SE alignments)"""
        
        # Group diagonal points by direction
        diagonal_groups = {"NE": [], "SW": [], "NW": [], "SE": []}
        
        for point in self.spiral_points:
            if point.cardinal_direction in diagonal_groups:
                diagonal_groups[point.cardinal_direction].append(point)
                
        # Detect NE-SW axis patterns
        if diagonal_groups["NE"] and diagonal_groups["SW"]:
            self._analyze_axis_pattern(diagonal_groups["NE"], diagonal_groups["SW"], "NE-SW")
            
        # Detect NW-SE axis patterns
        if diagonal_groups["NW"] and diagonal_groups["SE"]:
            self._analyze_axis_pattern(diagonal_groups["NW"], diagonal_groups["SE"], "NW-SE")
            
    def _detect_spiral_arm_patterns(self):
        """Detect patterns along spiral arms"""
        
        if not self.price_data:
            return
            
        current_price = self.price_data[-1, 4]
        
        # Group points by ring for spiral arm analysis
        for ring_num, ring_points in self.spiral_rings.items():
            
            if len(ring_points) < 4:
                continue
                
            # Sort points by angle to trace spiral arm
            ring_points.sort(key=lambda p: p.angle)
            
            # Look for patterns along spiral arms
            for start_idx in range(len(ring_points)):
                arm_points = []
                
                # Trace spiral arm across multiple rings
                for r in range(ring_num, min(ring_num + 4, self.config.max_rings + 1)):
                    if r in self.spiral_rings:
                        # Find point in this ring closest to same angle
                        target_angle = ring_points[start_idx].angle
                        closest_point = min(self.spiral_rings[r], 
                                          key=lambda p: abs(p.angle - target_angle))
                        arm_points.append(closest_point)
                        
                # Analyze arm pattern
                if len(arm_points) >= 3:
                    pattern_strength = self._analyze_arm_progression(arm_points, current_price)
                    
                    if pattern_strength >= self.config.min_pattern_strength:
                        
                        pattern = SpiralPattern(
                            pattern_type=PatternType.SPIRAL_ARM,
                            points=arm_points,
                            strength=pattern_strength,
                            confidence=0.7
                        )
                        
                        self.detected_patterns.append(pattern)
                        
    def _analyze_arm_progression(self, arm_points: List[SpiralPoint], current_price: float) -> float:
        """Analyze progression along a spiral arm"""
        
        if len(arm_points) < 3:
            return 0.0
            
        # Check value progression
        values = [p.value for p in arm_points]
        
        # Look for geometric progression
        ratios = []
        for i in range(len(values) - 1):
            if values[i] > 0:
                ratios.append(values[i + 1] / values[i])
                
        if len(ratios) < 2:
            return 0.0
            
        # Check ratio consistency
        avg_ratio = np.mean(ratios)
        ratio_std = np.std(ratios)
        consistency = 1.0 - min(1.0, ratio_std / avg_ratio) if avg_ratio > 0 else 0.0
        
        # Check proximity to current price
        distances = [abs(v - current_price) / current_price for v in values]
        min_distance = min(distances)
        proximity_score = max(0.0, 1.0 - min_distance)
        
        # Combine factors
        strength = (consistency * 0.6 + proximity_score * 0.4)
        
        return strength
        
    def _detect_concentric_ring_patterns(self):
        """Detect patterns in concentric rings"""
        
        if not self.price_data:
            return
            
        current_price = self.price_data[-1, 4]
        
        # Analyze patterns between consecutive rings
        for ring_num in range(1, len(self.spiral_rings)):
            
            if ring_num not in self.spiral_rings or (ring_num + 1) not in self.spiral_rings:
                continue
                
            inner_ring = self.spiral_rings[ring_num]
            outer_ring = self.spiral_rings[ring_num + 1]
            
            # Look for corresponding points between rings
            pattern_points = []
            
            for inner_point in inner_ring:
                # Find corresponding point in outer ring (similar angle)
                angle_tolerance = 10.0  # degrees
                
                for outer_point in outer_ring:
                    angle_diff = abs(inner_point.angle - outer_point.angle)
                    if angle_diff <= angle_tolerance or angle_diff >= (360 - angle_tolerance):
                        
                        # Check if ratio is significant
                        if inner_point.value > 0:
                            ratio = outer_point.value / inner_point.value
                            
                            # Look for sacred number ratios
                            if self._is_sacred_ratio(ratio):
                                pattern_points.extend([inner_point, outer_point])
                                
            if len(pattern_points) >= 4:  # At least 2 pairs
                
                strength = self._calculate_concentric_pattern_strength(pattern_points, current_price)
                
                if strength >= self.config.min_pattern_strength:
                    
                    pattern = SpiralPattern(
                        pattern_type=PatternType.CONCENTRIC_RING,
                        points=pattern_points,
                        strength=strength,
                        confidence=0.6
                    )
                    
                    self.detected_patterns.append(pattern)
                    
    def _detect_geometric_progressions(self):
        """Detect geometric progressions in spiral values"""
        
        # Analyze progressions within each direction
        directions = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
        
        for direction in directions:
            direction_points = [p for p in self.spiral_points 
                              if p.cardinal_direction == direction]
            
            if len(direction_points) < 3:
                continue
                
            # Sort by value
            direction_points.sort(key=lambda p: p.value)
            
            # Look for geometric progressions
            progressions = self._find_geometric_progressions(direction_points)
            
            for progression in progressions:
                self.detected_patterns.append(progression)
                
    def _find_geometric_progressions(self, points: List[SpiralPoint]) -> List[SpiralPattern]:
        """Find geometric progressions in a set of points"""
        
        progressions = []
        
        if len(points) < 3:
            return progressions
            
        for i in range(len(points) - 2):
            for j in range(i + 1, len(points) - 1):
                for k in range(j + 1, len(points)):
                    
                    p1, p2, p3 = points[i], points[j], points[k]
                    
                    if p1.value > 0 and p2.value > 0:
                        ratio1 = p2.value / p1.value
                        ratio2 = p3.value / p2.value
                        
                        # Check for geometric progression
                        if abs(ratio1 - ratio2) / max(ratio1, ratio2) < 0.05:  # 5% tolerance
                            
                            # Calculate strength
                            strength = self._calculate_progression_strength([p1, p2, p3], ratio1)
                            
                            if strength >= self.config.min_pattern_strength:
                                
                                pattern = SpiralPattern(
                                    pattern_type=PatternType.GEOMETRIC_PROGRESSION,
                                    points=[p1, p2, p3],
                                    strength=strength,
                                    confidence=0.8,
                                    geometric_ratio=(ratio1 + ratio2) / 2
                                )
                                
                                progressions.append(pattern)
                                
        return progressions
        
    def _detect_harmonic_resonance_patterns(self):
        """Detect harmonic resonance patterns"""
        
        if not self.price_data:
            return
            
        current_price = self.price_data[-1, 4]
        
        # Look for harmonic relationships between spiral points
        for i, point1 in enumerate(self.spiral_points):
            for j, point2 in enumerate(self.spiral_points[i + 1:], i + 1):
                
                if point1.value <= 0 or point2.value <= 0:
                    continue
                    
                # Calculate harmonic relationship
                ratio = max(point1.value, point2.value) / min(point1.value, point2.value)
                
                # Check for harmonic divisions
                for division in self.config.harmonic_divisions:
                    if abs(ratio - division) / division < 0.05:  # 5% tolerance
                        
                        # Found harmonic relationship
                        harmonic_points = [point1, point2]
                        
                        # Look for additional harmonic points
                        for point3 in self.spiral_points[j + 1:]:
                            if point3.value > 0:
                                ratio3 = point3.value / min(point1.value, point2.value)
                                
                                # Check if it fits harmonic series
                                if any(abs(ratio3 - div) / div < 0.05 for div in self.config.harmonic_divisions):
                                    harmonic_points.append(point3)
                                    
                        if len(harmonic_points) >= 2:
                            
                            strength = self._calculate_harmonic_strength(harmonic_points, current_price)
                            
                            if strength >= self.config.min_pattern_strength:
                                
                                pattern = SpiralPattern(
                                    pattern_type=PatternType.HARMONIC_RESONANCE,
                                    points=harmonic_points,
                                    strength=strength,
                                    confidence=0.7,
                                    harmonic_level=division
                                )
                                
                                self.detected_patterns.append(pattern)
                                break
                                
    def _calculate_pattern_strength(self, points: List[SpiralPoint], pattern_context: str) -> float:
        """Calculate strength of a detected pattern"""
        
        if not points or not self.price_data:
            return 0.0
            
        current_price = self.price_data[-1, 4]
        
        # Base strength from point significance
        avg_significance = np.mean([p.significance for p in points])
        
        # Proximity to current price
        distances = [abs(p.value - current_price) / current_price for p in points if p.value > 0]
        if distances:
            min_distance = min(distances)
            proximity_score = max(0.0, 1.0 - min_distance)
        else:
            proximity_score = 0.0
            
        # Pattern completeness (more points = stronger)
        completeness_score = min(1.0, len(points) / 5.0)
        
        # Context bonus
        context_bonus = 0.0
        if "cardinal" in pattern_context.lower():
            context_bonus = 0.1
        elif "diagonal" in pattern_context.lower():
            context_bonus = 0.05
            
        # Combine factors
        strength = (avg_significance * 0.4 + proximity_score * 0.3 + 
                   completeness_score * 0.2 + context_bonus * 0.1)
        
        return max(0.0, min(1.0, strength))
        
    def _calculate_concentric_pattern_strength(self, points: List[SpiralPoint], current_price: float) -> float:
        """Calculate strength of concentric ring pattern"""
        
        if not points:
            return 0.0
            
        # Group points by ring
        ring_groups = defaultdict(list)
        for point in points:
            ring_groups[point.ring].append(point)
            
        # Strength based on ring consistency
        ring_consistency = len(ring_groups) / max(1, len(points) // 2)
        
        # Proximity to current price
        distances = [abs(p.value - current_price) / current_price for p in points if p.value > 0]
        proximity_score = max(0.0, 1.0 - min(distances)) if distances else 0.0
        
        # Sacred ratio bonus
        ratio_bonus = 0.0
        for i in range(0, len(points) - 1, 2):  # Check pairs
            if i + 1 < len(points) and points[i].value > 0:
                ratio = points[i + 1].value / points[i].value
                if self._is_sacred_ratio(ratio):
                    ratio_bonus += 0.1
                    
        strength = (ring_consistency * 0.4 + proximity_score * 0.4 + 
                   min(0.2, ratio_bonus))
        
        return max(0.0, min(1.0, strength))
        
    def _calculate_progression_strength(self, points: List[SpiralPoint], ratio: float) -> float:
        """Calculate strength of geometric progression"""
        
        if not points or not self.price_data:
            return 0.0
            
        current_price = self.price_data[-1, 4]
        
        # Base strength from point significance
        avg_significance = np.mean([p.significance for p in points])
        
        # Ratio significance (sacred numbers are stronger)
        ratio_score = 0.5  # Base score
        if self._is_sacred_ratio(ratio):
            ratio_score = 0.9
        elif any(abs(ratio - r) < 0.05 for r in [1.414, 1.618, 2.0, 2.236]):
            ratio_score = 0.8
            
        # Proximity to current price
        distances = [abs(p.value - current_price) / current_price for p in points if p.value > 0]
        proximity_score = max(0.0, 1.0 - min(distances)) if distances else 0.0
        
        # Combine factors
        strength = (avg_significance * 0.4 + ratio_score * 0.3 + proximity_score * 0.3)
        
        return max(0.0, min(1.0, strength))
        
    def _calculate_harmonic_strength(self, points: List[SpiralPoint], current_price: float) -> float:
        """Calculate strength of harmonic resonance pattern"""
        
        if not points:
            return 0.0
            
        # Base strength from point significance
        avg_significance = np.mean([p.significance for p in points])
        
        # Harmonic completeness (more harmonic points = stronger)
        harmonic_completeness = min(1.0, len(points) / 4.0)
        
        # Proximity to current price
        distances = [abs(p.value - current_price) / current_price for p in points if p.value > 0]
        proximity_score = max(0.0, 1.0 - min(distances)) if distances else 0.0
        
        # Sacred harmonic bonus
        harmonic_bonus = 0.0
        values = [p.value for p in points if p.value > 0]
        if len(values) >= 2:
            for i in range(len(values) - 1):
                for j in range(i + 1, len(values)):
                    ratio = max(values[i], values[j]) / min(values[i], values[j])
                    if ratio in self.config.harmonic_divisions:
                        harmonic_bonus += 0.05
                        
        strength = (avg_significance * 0.4 + harmonic_completeness * 0.3 + 
                   proximity_score * 0.2 + min(0.1, harmonic_bonus))
        
        return max(0.0, min(1.0, strength))
        
    def _is_sacred_ratio(self, ratio: float) -> bool:
        """Check if ratio is a sacred number relationship"""
        
        sacred_ratios = [1.618, 2.618, 0.618, 1.414, 2.0, 2.236, 1.732]  # φ, √2, 2, √5, √3
        sacred_ratios.extend([n / 10 for n in self.config.sacred_numbers])  # Sacred numbers as ratios
        
        tolerance = 0.05
        return any(abs(ratio - sacred) / sacred < tolerance for sacred in sacred_ratios)
        
    def _analyze_spiral_relationships(self):
        """Analyze relationships between spiral points"""
        
        self.spiral_relationships.clear()
        
        if len(self.spiral_points) < 2:
            return
            
        # Analyze relationships between significant points
        significant_points = [p for p in self.spiral_points if p.significance >= 0.6]
        
        for i, point1 in enumerate(significant_points):
            for point2 in significant_points[i + 1:]:
                
                relationship = self._analyze_point_relationship(point1, point2)
                
                if relationship and relationship.strength >= 0.5:
                    self.spiral_relationships.append(relationship)
                    
        # Sort by strength
        self.spiral_relationships.sort(key=lambda r: r.strength, reverse=True)
        
        # Limit relationships for performance
        max_relationships = 30
        if len(self.spiral_relationships) > max_relationships:
            self.spiral_relationships = self.spiral_relationships[:max_relationships]
            
        self.logger.debug(f"Analyzed {len(self.spiral_relationships)} spiral relationships")
        
    def _analyze_point_relationship(self, point1: SpiralPoint, point2: SpiralPoint) -> Optional[SpiralRelationship]:
        """Analyze relationship between two spiral points"""
        
        if point1.value <= 0 or point2.value <= 0:
            return None
            
        # Calculate basic relationship properties
        angle_diff = abs(point1.angle - point2.angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
            
        ring_diff = abs(point1.ring - point2.ring)
        ratio = max(point1.value, point2.value) / min(point1.value, point2.value)
        
        # Determine relationship type
        relationship_type = self._determine_relationship_type(point1, point2, angle_diff, ring_diff, ratio)
        
        if not relationship_type:
            return None
            
        # Calculate relationship strength
        strength = self._calculate_relationship_strength(point1, point2, angle_diff, ring_diff, ratio)
        
        if strength < 0.5:
            return None
            
        # Check for harmonic resonance
        harmonic_resonance = None
        if any(abs(ratio - div) / div < 0.05 for div in self.config.harmonic_divisions):
            harmonic_resonance = ratio
            
        relationship = SpiralRelationship(
            point1=point1,
            point2=point2,
            relationship_type=relationship_type,
            angle_difference=angle_diff,
            ring_difference=ring_diff,
            ratio=ratio,
            strength=strength,
            harmonic_resonance=harmonic_resonance
        )
        
        return relationship
        
    def _determine_relationship_type(self, point1: SpiralPoint, point2: SpiralPoint, 
                                   angle_diff: float, ring_diff: int, ratio: float) -> Optional[str]:
        """Determine the type of relationship between points"""
        
        # Cardinal alignment
        if angle_diff < 10 and point1.cardinal_direction == point2.cardinal_direction:
            return "cardinal_alignment"
            
        # Opposite alignment
        if 170 <= angle_diff <= 190:
            return "opposite_alignment"
            
        # Right angle relationship
        if 80 <= angle_diff <= 100:
            return "right_angle"
            
        # Same ring relationship
        if ring_diff == 0:
            return "same_ring"
            
        # Adjacent ring relationship
        if ring_diff == 1:
            return "adjacent_ring"
            
        # Harmonic ratio relationship
        if self._is_sacred_ratio(ratio):
            return "harmonic_ratio"
            
        # Geometric progression
        if any(abs(ratio - r) < 0.05 for r in [1.414, 1.618, 2.0, 2.236]):
            return "geometric_progression"
            
        return None
        
    def _calculate_relationship_strength(self, point1: SpiralPoint, point2: SpiralPoint,
                                       angle_diff: float, ring_diff: int, ratio: float) -> float:
        """Calculate strength of relationship between points"""
        
        # Base strength from point significance
        base_strength = (point1.significance + point2.significance) / 2
        
        # Angle relationship bonus
        angle_bonus = 0.0
        if angle_diff < 10 or 170 <= angle_diff <= 190 or 80 <= angle_diff <= 100:
            angle_bonus = 0.15
        elif angle_diff < 30:
            angle_bonus = 0.1
            
        # Ring relationship bonus
        ring_bonus = 0.0
        if ring_diff == 0:
            ring_bonus = 0.1
        elif ring_diff == 1:
            ring_bonus = 0.05
            
        # Ratio significance bonus
        ratio_bonus = 0.0
        if self._is_sacred_ratio(ratio):
            ratio_bonus = 0.15
        elif any(abs(ratio - r) < 0.05 for r in [1.414, 1.618, 2.0]):
            ratio_bonus = 0.1
            
        # Cardinal direction bonus
        cardinal_bonus = 0.0
        if (point1.cardinal_direction in ["N", "S", "E", "W"] and
            point2.cardinal_direction in ["N", "S", "E", "W"]):
            cardinal_bonus = 0.05
            
        # Combine factors
        strength = base_strength + angle_bonus + ring_bonus + ratio_bonus + cardinal_bonus
        
        return max(0.0, min(1.0, strength))        
    def _generate_projections(self):
        """Generate projections based on Square of Nine analysis"""
        
        self.projections.clear()
        
        if not self.price_data:
            return
            
        current_price = self.price_data[-1, 4]
        
        # Generate projections from different sources
        self._generate_pattern_projections(current_price)
        self._generate_relationship_projections(current_price)
        self._generate_spiral_progression_projections(current_price)
        self._generate_harmonic_projections(current_price)
        
        # Sort projections by confidence
        self.projections.sort(key=lambda p: p.confidence, reverse=True)
        
        # Limit projections
        max_projections = 15
        if len(self.projections) > max_projections:
            self.projections = self.projections[:max_projections]
            
        self.logger.debug(f"Generated {len(self.projections)} spiral projections")
        
    def _generate_pattern_projections(self, current_price: float):
        """Generate projections from detected patterns"""
        
        for pattern in self.detected_patterns:
            
            if pattern.strength < 0.7:
                continue
                
            # Generate projections based on pattern type
            if pattern.pattern_type == PatternType.GEOMETRIC_PROGRESSION:
                self._project_from_geometric_progression(pattern, current_price)
                
            elif pattern.pattern_type == PatternType.CARDINAL_CROSS:
                self._project_from_cardinal_pattern(pattern, current_price)
                
            elif pattern.pattern_type == PatternType.SPIRAL_ARM:
                self._project_from_spiral_arm(pattern, current_price)
                
            elif pattern.pattern_type == PatternType.HARMONIC_RESONANCE:
                self._project_from_harmonic_pattern(pattern, current_price)
                
    def _project_from_geometric_progression(self, pattern: SpiralPattern, current_price: float):
        """Generate projections from geometric progression pattern"""
        
        if not pattern.geometric_ratio or len(pattern.points) < 3:
            return
            
        # Sort points by value
        sorted_points = sorted(pattern.points, key=lambda p: p.value)
        
        # Project next level in progression
        last_point = sorted_points[-1]
        next_level = last_point.value * pattern.geometric_ratio
        
        # Only project if it's a reasonable distance from current price
        distance_ratio = abs(next_level - current_price) / current_price
        
        if 0.02 <= distance_ratio <= 0.3:  # 2% to 30% move
            
            confidence = pattern.strength * 0.8  # Reduce confidence for projections
            
            projection = SpiralProjection(
                target_level=next_level,
                projection_type=f"geometric_progression_{pattern.geometric_ratio:.3f}",
                confidence=confidence,
                source_point=last_point,
                supporting_pattern=pattern
            )
            
            self.projections.append(projection)
            
        # Also project previous level (reversal scenario)
        if len(sorted_points) >= 2:
            prev_level = sorted_points[-2].value / pattern.geometric_ratio
            distance_ratio_prev = abs(prev_level - current_price) / current_price
            
            if 0.02 <= distance_ratio_prev <= 0.3:
                
                confidence_prev = pattern.strength * 0.6  # Lower confidence for reversal
                
                projection_prev = SpiralProjection(
                    target_level=prev_level,
                    projection_type=f"geometric_reversal_{pattern.geometric_ratio:.3f}",
                    confidence=confidence_prev,
                    source_point=sorted_points[-2],
                    supporting_pattern=pattern
                )
                
                self.projections.append(projection_prev)
                
    def _project_from_cardinal_pattern(self, pattern: SpiralPattern, current_price: float):
        """Generate projections from cardinal cross pattern"""
        
        # Find the strongest cardinal points in the pattern
        cardinal_points = [p for p in pattern.points if p.cardinal_direction in ["N", "S", "E", "W"]]
        
        if len(cardinal_points) < 2:
            return
            
        # Sort by distance from current price
        cardinal_points.sort(key=lambda p: abs(p.value - current_price))
        
        # Project to next cardinal level
        for i, point in enumerate(cardinal_points[:3]):  # Top 3 cardinal points
            
            # Find next cardinal point in same direction
            same_direction_points = [p for p in self.cardinal_points 
                                   if p.cardinal_direction == point.cardinal_direction 
                                   and p.value != point.value]
            
            if same_direction_points:
                
                if point.value < current_price:
                    # Find next higher cardinal point
                    higher_points = [p for p in same_direction_points if p.value > point.value]
                    if higher_points:
                        target_point = min(higher_points, key=lambda p: p.value)
                        target_level = target_point.value
                    else:
                        continue
                else:
                    # Find next lower cardinal point
                    lower_points = [p for p in same_direction_points if p.value < point.value]
                    if lower_points:
                        target_point = max(lower_points, key=lambda p: p.value)
                        target_level = target_point.value
                    else:
                        continue
                        
                distance_ratio = abs(target_level - current_price) / current_price
                
                if 0.02 <= distance_ratio <= 0.25:  # 2% to 25% move
                    
                    confidence = pattern.strength * 0.85 * (1 - i * 0.1)  # Reduce for each subsequent point
                    
                    projection = SpiralProjection(
                        target_level=target_level,
                        projection_type=f"cardinal_{point.cardinal_direction}",
                        confidence=confidence,
                        source_point=point,
                        supporting_pattern=pattern
                    )
                    
                    self.projections.append(projection)
                    
    def _project_from_spiral_arm(self, pattern: SpiralPattern, current_price: float):
        """Generate projections from spiral arm pattern"""
        
        if len(pattern.points) < 3:
            return
            
        # Sort points by ring (outward progression)
        arm_points = sorted(pattern.points, key=lambda p: p.ring)
        
        # Calculate progression ratio
        ratios = []
        for i in range(len(arm_points) - 1):
            if arm_points[i].value > 0:
                ratio = arm_points[i + 1].value / arm_points[i].value
                ratios.append(ratio)
                
        if not ratios:
            return
            
        avg_ratio = np.mean(ratios)
        
        # Project next point in spiral arm
        last_point = arm_points[-1]
        next_level = last_point.value * avg_ratio
        
        distance_ratio = abs(next_level - current_price) / current_price
        
        if 0.02 <= distance_ratio <= 0.35:  # 2% to 35% move
            
            confidence = pattern.strength * 0.75
            
            projection = SpiralProjection(
                target_level=next_level,
                projection_type=f"spiral_arm_{avg_ratio:.3f}",
                confidence=confidence,
                source_point=last_point,
                supporting_pattern=pattern
            )
            
            self.projections.append(projection)
            
    def _project_from_harmonic_pattern(self, pattern: SpiralPattern, current_price: float):
        """Generate projections from harmonic resonance pattern"""
        
        if not pattern.harmonic_level or len(pattern.points) < 2:
            return
            
        # Sort points by value
        harmonic_points = sorted(pattern.points, key=lambda p: p.value)
        
        # Generate harmonic projections
        base_value = harmonic_points[0].value
        
        for division in self.config.harmonic_divisions:
            if division != pattern.harmonic_level:
                
                # Project harmonic level
                harmonic_level = base_value * division
                
                distance_ratio = abs(harmonic_level - current_price) / current_price
                
                if 0.02 <= distance_ratio <= 0.4:  # 2% to 40% move
                    
                    confidence = pattern.strength * 0.7 * (0.8 if division in [2, 3, 4] else 0.6)
                    
                    projection = SpiralProjection(
                        target_level=harmonic_level,
                        projection_type=f"harmonic_{division}",
                        confidence=confidence,
                        source_point=harmonic_points[0],
                        supporting_pattern=pattern
                    )
                    
                    self.projections.append(projection)
                    
    def _generate_relationship_projections(self, current_price: float):
        """Generate projections from spiral relationships"""
        
        for relationship in self.spiral_relationships[:8]:  # Top 8 relationships
            
            if relationship.strength < 0.7:
                continue
                
            # Project based on relationship type
            if relationship.relationship_type == "geometric_progression":
                
                # Project continuation of geometric progression
                higher_point = relationship.point2 if relationship.point2.value > relationship.point1.value else relationship.point1
                target_level = higher_point.value * relationship.ratio
                
                distance_ratio = abs(target_level - current_price) / current_price
                
                if 0.02 <= distance_ratio <= 0.3:
                    
                    confidence = relationship.strength * 0.75
                    
                    projection = SpiralProjection(
                        target_level=target_level,
                        projection_type=f"relationship_{relationship.relationship_type}",
                        confidence=confidence,
                        source_point=higher_point
                    )
                    
                    self.projections.append(projection)
                    
            elif relationship.relationship_type == "harmonic_ratio" and relationship.harmonic_resonance:
                
                # Project harmonic extensions
                base_value = min(relationship.point1.value, relationship.point2.value)
                
                for multiplier in [2, 3, 4]:  # Harmonic extensions
                    harmonic_target = base_value * relationship.harmonic_resonance * multiplier
                    
                    distance_ratio = abs(harmonic_target - current_price) / current_price
                    
                    if 0.02 <= distance_ratio <= 0.4:
                        
                        confidence = relationship.strength * 0.65 / multiplier  # Reduce confidence for higher extensions
                        
                        projection = SpiralProjection(
                            target_level=harmonic_target,
                            projection_type=f"harmonic_extension_{multiplier}",
                            confidence=confidence,
                            source_point=relationship.point1
                        )
                        
                        self.projections.append(projection)
                        
    def _generate_spiral_progression_projections(self, current_price: float):
        """Generate projections from spiral progression analysis"""
        
        # Find current position in spiral
        current_spiral_point = self._find_nearest_spiral_point(current_price)
        
        if not current_spiral_point:
            return
            
        # Project along spiral progression
        current_ring = current_spiral_point.ring
        
        # Project to next rings
        for ring_offset in [1, 2, 3]:
            target_ring = current_ring + ring_offset
            
            if target_ring <= self.config.max_rings and target_ring in self.spiral_rings:
                
                # Find corresponding point in target ring
                target_points = self.spiral_rings[target_ring]
                
                # Find point with similar angle
                target_point = min(target_points, 
                                 key=lambda p: abs(p.angle - current_spiral_point.angle))
                
                distance_ratio = abs(target_point.value - current_price) / current_price
                
                if 0.02 <= distance_ratio <= 0.4:
                    
                    confidence = 0.7 / ring_offset  # Reduce confidence for distant rings
                    
                    projection = SpiralProjection(
                        target_level=target_point.value,
                        projection_type=f"spiral_ring_{target_ring}",
                        confidence=confidence,
                        source_point=current_spiral_point
                    )
                    
                    self.projections.append(projection)
                    
    def _generate_harmonic_projections(self, current_price: float):
        """Generate projections based on harmonic analysis"""
        
        # Find harmonic projections from current price
        for division in self.config.harmonic_divisions:
            
            # Harmonic multiples
            for multiplier in [0.5, 0.618, 0.786, 1.272, 1.618, 2.0, 2.618]:
                
                harmonic_level = current_price * division * multiplier
                
                # Check if this aligns with any spiral point
                nearest_spiral = self._find_nearest_spiral_point(harmonic_level)
                
                if nearest_spiral:
                    distance_to_spiral = abs(nearest_spiral.value - harmonic_level) / harmonic_level
                    
                    if distance_to_spiral < 0.02:  # Within 2% of spiral point
                        
                        distance_from_current = abs(harmonic_level - current_price) / current_price
                        
                        if 0.02 <= distance_from_current <= 0.3:
                            
                            confidence = 0.6 * nearest_spiral.significance
                            
                            projection = SpiralProjection(
                                target_level=harmonic_level,
                                projection_type=f"harmonic_{division}_{multiplier}",
                                confidence=confidence,
                                source_point=nearest_spiral
                            )
                            
                            self.projections.append(projection)
                            
    def _find_nearest_spiral_point(self, price: float) -> Optional[SpiralPoint]:
        """Find the nearest spiral point to a given price"""
        
        if not self.spiral_points:
            return None
            
        return min(self.spiral_points, key=lambda p: abs(p.value - price))
        
    def _ml_enhanced_analysis(self):
        """Apply ML enhancement to Square of Nine analysis"""
        
        if not SKLEARN_AVAILABLE:
            return
            
        try:
            # ML-enhanced pattern classification
            self._ml_pattern_classification()
            
            # ML-enhanced level prediction
            self._ml_level_prediction()
            
            # ML-enhanced significance scoring
            self._ml_significance_enhancement()
            
            self.logger.debug("ML enhancement completed")
            
        except Exception as e:
            self.logger.debug(f"ML enhancement failed: {e}")
            
    def _ml_pattern_classification(self):
        """Use ML to classify and validate patterns"""
        
        if len(self.detected_patterns) < 5:
            return
            
        try:
            # Extract features for pattern classification
            features = []
            
            for pattern in self.detected_patterns:
                feature_vector = self._extract_pattern_features(pattern)
                features.append(feature_vector)
                
            if features:
                X = np.array(features)
                
                # Normalize features
                if self.config.feature_scaling:
                    X = self.scaler.fit_transform(X)
                    
                # Cluster patterns to identify high-quality ones
                n_clusters = min(3, len(self.detected_patterns) // 2)
                if n_clusters >= 2:
                    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = clusterer.fit_predict(X)
                    
                    # Identify the "high quality" cluster
                    cluster_qualities = []
                    for cluster_id in range(n_clusters):
                        cluster_patterns = [self.detected_patterns[i] for i in range(len(self.detected_patterns)) 
                                          if cluster_labels[i] == cluster_id]
                        avg_strength = np.mean([p.strength for p in cluster_patterns])
                        cluster_qualities.append(avg_strength)
                        
                    best_cluster = np.argmax(cluster_qualities)
                    
                    # Enhance confidence for patterns in best cluster
                    for i, pattern in enumerate(self.detected_patterns):
                        if cluster_labels[i] == best_cluster:
                            pattern.confidence = min(1.0, pattern.confidence * 1.15)
                        else:
                            pattern.confidence = max(0.0, pattern.confidence * 0.9)
                            
        except Exception as e:
            self.logger.debug(f"ML pattern classification failed: {e}")
            
    def _extract_pattern_features(self, pattern: SpiralPattern) -> List[float]:
        """Extract features for ML pattern analysis"""
        
        features = []
        
        # Basic pattern features
        features.append(pattern.strength)
        features.append(pattern.confidence)
        features.append(len(pattern.points))
        
        # Pattern type encoding
        type_features = [0.0] * len(PatternType)
        for i, pattern_type in enumerate(PatternType):
            if pattern.pattern_type == pattern_type:
                type_features[i] = 1.0
        features.extend(type_features)
        
        # Geometric features
        if pattern.geometric_ratio:
            features.append(pattern.geometric_ratio)
        else:
            features.append(0.0)
            
        if pattern.harmonic_level:
            features.append(float(pattern.harmonic_level))
        else:
            features.append(0.0)
            
        # Point distribution features
        if pattern.points:
            values = [p.value for p in pattern.points if p.value > 0]
            if values:
                features.extend([
                    np.mean(values),
                    np.std(values),
                    max(values) / min(values) if min(values) > 0 else 1.0
                ])
            else:
                features.extend([0.0, 0.0, 1.0])
                
            # Ring distribution
            rings = [p.ring for p in pattern.points]
            features.extend([
                np.mean(rings),
                np.std(rings) if len(rings) > 1 else 0.0
            ])
            
            # Angle distribution
            angles = [p.angle for p in pattern.points]
            features.extend([
                np.mean(angles),
                np.std(angles) if len(angles) > 1 else 0.0
            ])
        else:
            features.extend([0.0] * 7)
            
        return features
        
    def _ml_level_prediction(self):
        """Use ML to predict future price levels"""
        
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
                self.logger.debug(f"ML level predictor trained with R² = {r2:.3f}")
                
                # Generate ML-based projections
                self._generate_ml_projections()
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
            
        prices = self.price_data[:, 4]  # Close prices
        
        # Create features based on spiral analysis and price patterns
        for i in range(20, len(prices) - 10):
            
            # Price features
            recent_prices = prices[i-20:i]
            feature_vector = [
                np.mean(recent_prices),
                np.std(recent_prices),
                (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] != 0 else 0,
                (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if recent_prices[-5] != 0 else 0
            ]
            
            # Spiral position features
            current_price = prices[i]
            nearest_spiral = self._find_nearest_spiral_point(current_price)
            
            if nearest_spiral:
                feature_vector.extend([
                    (current_price - nearest_spiral.value) / nearest_spiral.value if nearest_spiral.value != 0 else 0,
                    nearest_spiral.significance,
                    float(nearest_spiral.ring),
                    nearest_spiral.angle / 360.0
                ])
            else:
                feature_vector.extend([0.0, 0.0, 0.0, 0.0])
                
            # Pattern proximity features
            pattern_proximity = self._calculate_pattern_proximity_at_point(current_price)
            feature_vector.append(pattern_proximity)
            
            # Target: future price relative to current
            future_price = prices[i + 10]
            target = (future_price - current_price) / current_price if current_price != 0 else 0
            
            features.append(feature_vector)
            targets.append(target)
            
        return features, targets
        
    def _calculate_pattern_proximity_at_point(self, price: float) -> float:
        """Calculate proximity to detected patterns at a specific price"""
        
        if not self.detected_patterns:
            return 0.0
            
        proximities = []
        
        for pattern in self.detected_patterns:
            for point in pattern.points:
                if point.value > 0:
                    distance = abs(point.value - price) / price
                    proximity = max(0.0, 1.0 - distance) * pattern.strength
                    proximities.append(proximity)
                    
        return max(proximities) if proximities else 0.0
        
    def _generate_ml_projections(self):
        """Generate ML-based projections"""
        
        if not self.level_predictor or not self.price_data:
            return
            
        try:
            current_price = self.price_data[-1, 4]
            
            # Prepare current features
            current_features = self._extract_current_features_for_ml()
            
            if not current_features:
                return
                
            # Predict price change
            feature_vector = np.array(current_features).reshape(1, -1)
            predicted_change = self.level_predictor.predict(feature_vector)[0]
            
            # Convert to target price
            target_price = current_price * (1 + predicted_change)
            
            # Find nearest spiral point to target
            nearest_spiral = self._find_nearest_spiral_point(target_price)
            
            if nearest_spiral:
                distance_to_spiral = abs(nearest_spiral.value - target_price) / target_price
                
                # Adjust target to spiral point if close
                if distance_to_spiral < 0.03:  # Within 3%
                    target_price = nearest_spiral.value
                    
            # Check distance reasonableness
            distance_ratio = abs(target_price - current_price) / current_price
            
            if 0.01 <= distance_ratio <= 0.3:  # 1% to 30% move
                
                confidence = 0.6  # Base ML confidence
                
                projection = SpiralProjection(
                    target_level=target_price,
                    projection_type="ml_prediction",
                    confidence=confidence,
                    source_point=nearest_spiral
                )
                
                self.projections.append(projection)
                
        except Exception as e:
            self.logger.debug(f"ML projection generation failed: {e}")
            
    def _extract_current_features_for_ml(self) -> Optional[List[float]]:
        """Extract current features for ML prediction"""
        
        try:
            if not self.price_data or len(self.price_data) < 20:
                return None
                
            prices = self.price_data[:, 4]
            recent_prices = prices[-20:]
            current_price = prices[-1]
            
            # Same features as training
            features = [
                np.mean(recent_prices),
                np.std(recent_prices),
                (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] != 0 else 0,
                (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if recent_prices[-5] != 0 else 0
            ]
            
            # Current spiral features
            nearest_spiral = self._find_nearest_spiral_point(current_price)
            
            if nearest_spiral:
                features.extend([
                    (current_price - nearest_spiral.value) / nearest_spiral.value if nearest_spiral.value != 0 else 0,
                    nearest_spiral.significance,
                    float(nearest_spiral.ring),
                    nearest_spiral.angle / 360.0
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
                
            # Current pattern proximity
            pattern_proximity = self._calculate_pattern_proximity_at_point(current_price)
            features.append(pattern_proximity)
            
            return features
            
        except Exception:
            return None
            
    def _ml_significance_enhancement(self):
        """Use ML to enhance significance scoring"""
        
        if len(self.spiral_points) < 10 or not self.price_data:
            return
            
        try:
            # Extract features for each spiral point
            enhanced_features = []
            
            for point in self.spiral_points:
                features = self._extract_point_features_for_ml(point)
                enhanced_features.append(features)
                
            if enhanced_features:
                X = np.array(enhanced_features)
                
                # Use clustering to identify high-quality points
                clusterer = KMeans(n_clusters=3, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(X)
                
                # Identify the "high significance" cluster
                cluster_significances = []
                for cluster_id in range(3):
                    cluster_points = [self.spiral_points[i] for i in range(len(self.spiral_points)) 
                                    if cluster_labels[i] == cluster_id]
                    avg_significance = np.mean([p.significance for p in cluster_points])
                    cluster_significances.append(avg_significance)
                    
                best_cluster = np.argmax(cluster_significances)
                
                # Enhance significance for points in best cluster
                for i, point in enumerate(self.spiral_points):
                    if cluster_labels[i] == best_cluster:
                        point.significance = min(1.0, point.significance * 1.1)
                        
        except Exception as e:
            self.logger.debug(f"ML significance enhancement failed: {e}")
            
    def _extract_point_features_for_ml(self, point: SpiralPoint) -> List[float]:
        """Extract features for ML analysis of a spiral point"""
        
        features = [
            point.value,
            point.significance,
            float(point.ring),
            point.angle,
            float(point.support_count),
            float(point.resistance_count)
        ]
        
        # Cardinal direction encoding
        cardinal_features = [0.0] * 8  # N, S, E, W, NE, NW, SE, SW
        cardinal_directions = ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
        
        if point.cardinal_direction in cardinal_directions:
            idx = cardinal_directions.index(point.cardinal_direction)
            cardinal_features[idx] = 1.0
            
        features.extend(cardinal_features)
        
        # Level encoding
        level_features = [0.0] * len(SpiralLevel)
        for i, level in enumerate(SpiralLevel):
            if point.level == level:
                level_features[i] = 1.0
                
        features.extend(level_features)
        
        # Price action features around this point
        if self.price_data is not None:
            prices = self.price_data[:, 4]
            price_features = self._extract_price_action_features_for_point(point.value, prices)
            features.extend(price_features)
        else:
            features.extend([0.0] * 5)
            
        return features
        
    def _extract_price_action_features_for_point(self, level: float, prices: np.ndarray) -> List[float]:
        """Extract price action features around a spiral point level"""
        
        if len(prices) == 0:
            return [0.0] * 5
            
        tolerance = level * 0.015  # 1.5% tolerance
        
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
        level_prices = [p for p in prices if abs(p - level) <= tolerance * 2]
        volatility = np.std(level_prices) / level if level_prices and level > 0 else 0
        
        # Recent price position relative to level
        recent_position = (prices[-1] - level) / level if len(prices) > 0 and level > 0 else 0
        
        return [touch_ratio, above_ratio, below_ratio, volatility, recent_position]        
    def _update_level_significance(self):
        """Update spiral level significance based on recent price action"""
        
        if not self.price_data:
            return
            
        current_time = self.price_data[-1, 0]
        current_price = self.price_data[-1, 4]
        
        # Update significance for all spiral points
        for point in self.spiral_points:
            
            # Update support/resistance counts
            self._update_support_resistance_counts(point)
            
            # Update last touch time
            tolerance = point.value * 0.015  # 1.5% tolerance
            if abs(current_price - point.value) <= tolerance:
                point.last_touch_time = current_time
                
            # Adjust significance based on recency and activity
            self._adjust_significance_for_activity(point, current_time)
            
        # Update significant levels list
        self.significant_levels = [p for p in self.spiral_points if p.significance >= 0.7]
        self.significant_levels.sort(key=lambda p: p.value)
        
    def _update_support_resistance_counts(self, point: SpiralPoint):
        """Update support and resistance counts for a spiral point"""
        
        if not self.price_data:
            return
            
        prices = self.price_data[:, 4]
        tolerance = point.value * 0.015  # 1.5% tolerance
        
        support_count = 0
        resistance_count = 0
        
        for i in range(1, len(prices)):
            
            # Check if price bounced off the level
            if (abs(prices[i] - point.value) <= tolerance and
                abs(prices[i-1] - point.value) > tolerance):
                
                # Determine if it was support or resistance
                if prices[i-1] > point.value and prices[i] <= point.value:
                    support_count += 1
                elif prices[i-1] < point.value and prices[i] >= point.value:
                    resistance_count += 1
                    
        point.support_count = support_count
        point.resistance_count = resistance_count
        
    def _adjust_significance_for_activity(self, point: SpiralPoint, current_time: float):
        """Adjust spiral point significance based on recent activity"""
        
        base_significance = point.significance
        
        # Recent activity bonus
        if point.last_touch_time:
            time_since_touch = current_time - point.last_touch_time
            days_since_touch = time_since_touch / (24 * 3600)
            
            if days_since_touch <= 5:  # Recent touch
                recency_bonus = 0.15 * (1 - days_since_touch / 5)
                point.significance = min(1.0, base_significance + recency_bonus)
            elif days_since_touch > 30:  # Old touch
                recency_penalty = 0.05
                point.significance = max(0.0, base_significance - recency_penalty)
                
        # Support/resistance activity bonus
        total_activity = point.support_count + point.resistance_count
        if total_activity > 0:
            activity_bonus = min(0.2, total_activity * 0.025)
            point.significance = min(1.0, point.significance + activity_bonus)
            
        # Cardinal direction bonus
        if point.cardinal_direction in ["N", "S", "E", "W"]:
            cardinal_bonus = 0.05
            point.significance = min(1.0, point.significance + cardinal_bonus)
            
    def _generate_trading_signals(self):
        """Generate trading signals based on Square of Nine analysis"""
        
        self.trading_signals.clear()
        
        if not self.price_data:
            return
            
        current_price = self.price_data[-1, 4]
        current_time = self.price_data[-1, 0]
        
        # Generate signals from different sources
        self._generate_spiral_level_signals(current_price, current_time)
        self._generate_pattern_signals(current_price, current_time)
        self._generate_projection_signals(current_price, current_time)
        self._generate_harmonic_signals(current_price, current_time)
        
        # Sort signals by strength
        self.trading_signals.sort(key=lambda s: s.strength, reverse=True)
        
        # Limit signals
        max_signals = 6
        if len(self.trading_signals) > max_signals:
            self.trading_signals = self.trading_signals[:max_signals]
            
        self.logger.debug(f"Generated {len(self.trading_signals)} trading signals")
        
    def _generate_spiral_level_signals(self, current_price: float, current_time: float):
        """Generate signals based on spiral level interactions"""
        
        # Check significant spiral levels near current price
        nearby_levels = [p for p in self.significant_levels 
                        if abs(p.value - current_price) / current_price <= 0.03]  # Within 3%
        
        for level in nearby_levels:
            
            if level.significance < 0.7:
                continue
                
            # Determine signal direction
            signal_direction = self._determine_spiral_signal_direction(level, current_price)
            
            if signal_direction:
                
                # Calculate signal strength
                strength = self._calculate_spiral_signal_strength(level, current_price)
                
                if strength >= self.config.min_signal_strength:
                    
                    signal = TradingSignal(
                        signal_type=signal_direction,
                        strength=strength,
                        entry_price=current_price,
                        target_price=self._calculate_spiral_target(level, signal_direction),
                        stop_loss=self._calculate_spiral_stop_loss(level, signal_direction, current_price),
                        timestamp=current_time,
                        source_type="spiral_level",
                        source_data={"spiral_point": level}
                    )
                    
                    self.trading_signals.append(signal)
                    
    def _determine_spiral_signal_direction(self, level: SpiralPoint, current_price: float) -> Optional[SignalType]:
        """Determine signal direction at a spiral level"""
        
        # Consider historical support/resistance behavior
        total_interactions = level.support_count + level.resistance_count
        
        if total_interactions == 0:
            # Use cardinal direction bias
            if level.cardinal_direction in ["N", "E"]:
                return SignalType.BULLISH if current_price <= level.value else None
            elif level.cardinal_direction in ["S", "W"]:
                return SignalType.BEARISH if current_price >= level.value else None
            return None
            
        support_ratio = level.support_count / total_interactions
        resistance_ratio = level.resistance_count / total_interactions
        
        # Strong support level
        if support_ratio >= 0.7 and current_price <= level.value * 1.005:
            return SignalType.BULLISH
            
        # Strong resistance level
        if resistance_ratio >= 0.7 and current_price >= level.value * 0.995:
            return SignalType.BEARISH
            
        # Breakout scenarios
        if current_price > level.value * 1.01 and resistance_ratio >= 0.6:
            # Resistance breakout
            return SignalType.BULLISH
            
        if current_price < level.value * 0.99 and support_ratio >= 0.6:
            # Support breakdown
            return SignalType.BEARISH
            
        return None
        
    def _calculate_spiral_signal_strength(self, level: SpiralPoint, current_price: float) -> float:
        """Calculate signal strength for spiral level"""
        
        # Base strength from level significance
        base_strength = level.significance
        
        # Proximity bonus (closer = stronger)
        distance_pct = abs(current_price - level.value) / level.value
        proximity_bonus = (0.03 - distance_pct) / 0.03 * 0.2 if distance_pct <= 0.03 else 0.0
        
        # Historical activity bonus
        total_interactions = level.support_count + level.resistance_count
        activity_bonus = min(0.15, total_interactions * 0.025)
        
        # Cardinal direction bonus
        cardinal_bonus = 0.0
        if level.cardinal_direction in ["N", "S", "E", "W"]:
            cardinal_bonus = 0.1
        elif level.cardinal_direction in ["NE", "NW", "SE", "SW"]:
            cardinal_bonus = 0.05
            
        # Ring significance bonus (inner rings are more significant)
        ring_bonus = max(0.0, 0.1 - level.ring * 0.01)
        
        strength = base_strength + proximity_bonus + activity_bonus + cardinal_bonus + ring_bonus
        
        return max(0.0, min(1.0, strength))
        
    def _calculate_spiral_target(self, level: SpiralPoint, signal_direction: SignalType) -> float:
        """Calculate target price for spiral-based signal"""
        
        # Find next significant spiral level in signal direction
        if signal_direction == SignalType.BULLISH:
            # Find next higher significant level
            higher_levels = [p for p in self.significant_levels 
                           if p.value > level.value and p.significance >= 0.6]
            if higher_levels:
                return min(p.value for p in higher_levels)
            else:
                return level.value * 1.08  # 8% target
        else:
            # Find next lower significant level
            lower_levels = [p for p in self.significant_levels 
                          if p.value < level.value and p.significance >= 0.6]
            if lower_levels:
                return max(p.value for p in lower_levels)
            else:
                return level.value * 0.92  # 8% target
                
        return level.value
        
    def _calculate_spiral_stop_loss(self, level: SpiralPoint, signal_direction: SignalType, current_price: float) -> float:
        """Calculate stop loss for spiral-based signal"""
        
        # Conservative stop loss based on spiral level significance
        stop_distance_pct = 0.025  # 2.5% default
        
        # Adjust based on level significance
        if level.significance >= 0.9:
            stop_distance_pct = 0.02  # Tighter stop for very strong levels
        elif level.significance <= 0.7:
            stop_distance_pct = 0.03  # Wider stop for weaker levels
            
        if signal_direction == SignalType.BULLISH:
            return current_price * (1 - stop_distance_pct)
        else:
            return current_price * (1 + stop_distance_pct)
            
    def _generate_pattern_signals(self, current_price: float, current_time: float):
        """Generate signals based on detected patterns"""
        
        for pattern in self.detected_patterns[:5]:  # Top 5 patterns
            
            if pattern.strength < 0.7:
                continue
                
            # Check if current price is near pattern points
            pattern_proximity = min(abs(p.value - current_price) / current_price 
                                  for p in pattern.points if p.value > 0)
            
            if pattern_proximity <= 0.025:  # Within 2.5%
                
                signal_direction = self._determine_pattern_signal_direction(pattern, current_price)
                
                if signal_direction:
                    
                    strength = pattern.strength * 0.85  # Reduce for pattern signals
                    
                    signal = TradingSignal(
                        signal_type=signal_direction,
                        strength=strength,
                        entry_price=current_price,
                        target_price=self._calculate_pattern_target(pattern, signal_direction),
                        stop_loss=self._calculate_pattern_stop_loss(pattern, signal_direction, current_price),
                        timestamp=current_time,
                        source_type=f"pattern_{pattern.pattern_type.value}",
                        source_data={"pattern": pattern}
                    )
                    
                    self.trading_signals.append(signal)
                    
    def _determine_pattern_signal_direction(self, pattern: SpiralPattern, current_price: float) -> Optional[SignalType]:
        """Determine signal direction for pattern"""
        
        if pattern.pattern_type == PatternType.GEOMETRIC_PROGRESSION:
            
            if pattern.geometric_ratio and pattern.geometric_ratio > 1.0:
                # Upward progression
                sorted_points = sorted(pattern.points, key=lambda p: p.value)
                if current_price >= sorted_points[-1].value * 0.99:
                    return SignalType.BULLISH
                    
        elif pattern.pattern_type == PatternType.CARDINAL_CROSS:
            
            # Determine based on cardinal directions in pattern
            north_south_points = [p for p in pattern.points if p.cardinal_direction in ["N", "S"]]
            if north_south_points:
                return SignalType.BULLISH  # Cardinal cross typically bullish
                
        elif pattern.pattern_type == PatternType.HARMONIC_RESONANCE:
            
            # Harmonic patterns suggest continuation
            pattern_values = [p.value for p in pattern.points if p.value > 0]
            if pattern_values:
                if current_price >= np.median(pattern_values):
                    return SignalType.BULLISH
                else:
                    return SignalType.BEARISH
                    
        return None
        
    def _calculate_pattern_target(self, pattern: SpiralPattern, signal_direction: SignalType) -> float:
        """Calculate target for pattern-based signal"""
        
        pattern_values = [p.value for p in pattern.points if p.value > 0]
        
        if not pattern_values:
            return 0.0
            
        if signal_direction == SignalType.BULLISH:
            if pattern.geometric_ratio and pattern.geometric_ratio > 1.0:
                return max(pattern_values) * pattern.geometric_ratio
            else:
                return max(pattern_values) * 1.05  # 5% above highest point
        else:
            if pattern.geometric_ratio and pattern.geometric_ratio > 1.0:
                return min(pattern_values) / pattern.geometric_ratio
            else:
                return min(pattern_values) * 0.95  # 5% below lowest point
                
    def _calculate_pattern_stop_loss(self, pattern: SpiralPattern, signal_direction: SignalType, current_price: float) -> float:
        """Calculate stop loss for pattern-based signal"""
        
        stop_distance_pct = 0.03  # 3% for pattern signals
        
        if signal_direction == SignalType.BULLISH:
            return current_price * (1 - stop_distance_pct)
        else:
            return current_price * (1 + stop_distance_pct)
            
    def _generate_projection_signals(self, current_price: float, current_time: float):
        """Generate signals based on projections"""
        
        for projection in self.projections[:4]:  # Top 4 projections
            
            if projection.confidence < 0.7:
                continue
                
            # Only generate signal if projection is reasonable distance
            distance_pct = abs(projection.target_level - current_price) / current_price
            
            if 0.02 <= distance_pct <= 0.12:  # 2% to 12% move
                
                signal_direction = SignalType.BULLISH if projection.target_level > current_price else SignalType.BEARISH
                
                strength = projection.confidence * 0.8  # Reduce for projection signals
                
                signal = TradingSignal(
                    signal_type=signal_direction,
                    strength=strength,
                    entry_price=current_price,
                    target_price=projection.target_level,
                    stop_loss=self._calculate_projection_stop_loss(signal_direction, current_price, distance_pct),
                    timestamp=current_time,
                    source_type=f"projection_{projection.projection_type}",
                    source_data={"projection": projection}
                )
                
                self.trading_signals.append(signal)
                
    def _calculate_projection_stop_loss(self, signal_direction: SignalType, current_price: float, distance_pct: float) -> float:
        """Calculate stop loss for projection-based signal"""
        
        # Stop loss proportional to projection distance
        stop_distance_pct = min(0.035, distance_pct * 0.4)  # Max 3.5%, proportional to target
        
        if signal_direction == SignalType.BULLISH:
            return current_price * (1 - stop_distance_pct)
        else:
            return current_price * (1 + stop_distance_pct)
            
    def _generate_harmonic_signals(self, current_price: float, current_time: float):
        """Generate signals based on harmonic analysis"""
        
        # Check for harmonic resonance with sacred numbers
        for sacred_num in self.config.sacred_numbers:
            
            # Calculate harmonic levels
            for multiplier in [0.618, 0.786, 1.272, 1.618, 2.0]:
                
                harmonic_level = current_price * sacred_num * multiplier / 10
                
                # Find nearest spiral point to harmonic level
                nearest_spiral = self._find_nearest_spiral_point(harmonic_level)
                
                if nearest_spiral:
                    distance_to_spiral = abs(nearest_spiral.value - harmonic_level) / harmonic_level
                    distance_from_current = abs(harmonic_level - current_price) / current_price
                    
                    if distance_to_spiral < 0.02 and 0.02 <= distance_from_current <= 0.15:
                        
                        signal_direction = SignalType.BULLISH if harmonic_level > current_price else SignalType.BEARISH
                        
                        # Calculate strength based on sacred number and spiral significance
                        strength = (0.6 + sacred_num * 0.01) * nearest_spiral.significance
                        
                        if strength >= self.config.min_signal_strength:
                            
                            signal = TradingSignal(
                                signal_type=signal_direction,
                                strength=strength,
                                entry_price=current_price,
                                target_price=harmonic_level,
                                stop_loss=self._calculate_harmonic_stop_loss(signal_direction, current_price),
                                timestamp=current_time,
                                source_type=f"harmonic_{sacred_num}",
                                source_data={"sacred_number": sacred_num, "multiplier": multiplier}
                            )
                            
                            self.trading_signals.append(signal)
                            
    def _calculate_harmonic_stop_loss(self, signal_direction: SignalType, current_price: float) -> float:
        """Calculate stop loss for harmonic-based signal"""
        
        stop_distance_pct = 0.025  # 2.5% for harmonic signals
        
        if signal_direction == SignalType.BULLISH:
            return current_price * (1 - stop_distance_pct)
        else:
            return current_price * (1 + stop_distance_pct)
            
    # Analysis and utility methods
    def get_current_spiral_analysis(self) -> Dict[str, Any]:
        """Get current Square of Nine analysis summary"""
        
        if not self.price_data:
            return {}
            
        current_price = self.price_data[-1, 4]
        
        analysis = {
            'current_price': current_price,
            'center_value': self.current_center_value,
            'spiral_stats': {
                'total_points': len(self.spiral_points),
                'rings_calculated': len(self.spiral_rings),
                'cardinal_points': len(self.cardinal_points),
                'significant_levels': len(self.significant_levels)
            },
            'pattern_analysis': {
                'patterns_detected': len(self.detected_patterns),
                'relationships_found': len(self.spiral_relationships),
                'projections_generated': len(self.projections),
                'trading_signals': len(self.trading_signals)
            },
            'current_position': self._analyze_current_position(current_price),
            'nearest_levels': self._find_nearest_spiral_levels(current_price),
            'spiral_strength': self._assess_spiral_strength()
        }
        
        return analysis
        
    def _analyze_current_position(self, current_price: float) -> Dict[str, Any]:
        """Analyze current price position in the spiral"""
        
        nearest_point = self._find_nearest_spiral_point(current_price)
        
        if not nearest_point:
            return {}
            
        position_analysis = {
            'nearest_spiral_point': {
                'value': nearest_point.value,
                'ring': nearest_point.ring,
                'angle': nearest_point.angle,
                'cardinal_direction': nearest_point.cardinal_direction,
                'significance': nearest_point.significance,
                'distance_pct': abs(current_price - nearest_point.value) / nearest_point.value * 100
            }
        }
        
        # Analyze position relative to cardinal points
        cardinal_distances = []
        for cardinal_point in self.cardinal_points:
            distance = abs(current_price - cardinal_point.value) / current_price
            cardinal_distances.append({
                'direction': cardinal_point.cardinal_direction,
                'value': cardinal_point.value,
                'distance_pct': distance * 100,
                'significance': cardinal_point.significance
            })
            
        # Sort by distance and get closest 3
        cardinal_distances.sort(key=lambda x: x['distance_pct'])
        position_analysis['nearest_cardinals'] = cardinal_distances[:3]
        
        return position_analysis
        
    def _find_nearest_spiral_levels(self, current_price: float) -> Dict[str, Any]:
        """Find nearest spiral levels above and below current price"""
        
        # Separate levels above and below current price
        above_levels = [p for p in self.significant_levels if p.value > current_price]
        below_levels = [p for p in self.significant_levels if p.value < current_price]
        
        # Sort and get nearest
        above_levels.sort(key=lambda p: p.value)
        below_levels.sort(key=lambda p: p.value, reverse=True)
        
        result = {}
        
        if above_levels:
            nearest_above = above_levels[0]
            result['nearest_resistance'] = {
                'value': nearest_above.value,
                'distance_pct': (nearest_above.value - current_price) / current_price * 100,
                'significance': nearest_above.significance,
                'cardinal_direction': nearest_above.cardinal_direction,
                'ring': nearest_above.ring
            }
            
        if below_levels:
            nearest_below = below_levels[0]
            result['nearest_support'] = {
                'value': nearest_below.value,
                'distance_pct': (current_price - nearest_below.value) / current_price * 100,
                'significance': nearest_below.significance,
                'cardinal_direction': nearest_below.cardinal_direction,
                'ring': nearest_below.ring
            }
            
        return result
        
    def _assess_spiral_strength(self) -> Dict[str, Any]:
        """Assess overall strength of spiral analysis"""
        
        # Calculate various strength metrics
        total_patterns = len(self.detected_patterns)
        strong_patterns = len([p for p in self.detected_patterns if p.strength >= 0.8])
        
        total_relationships = len(self.spiral_relationships)
        strong_relationships = len([r for r in self.spiral_relationships if r.strength >= 0.8])
        
        high_significance_levels = len([p for p in self.spiral_points if p.significance >= 0.8])
        
        strength_assessment = {
            'pattern_strength': strong_patterns / max(1, total_patterns),
            'relationship_strength': strong_relationships / max(1, total_relationships),
            'level_quality': high_significance_levels / max(1, len(self.spiral_points)),
            'overall_strength': 0.0
        }
        
        # Calculate overall strength
        strength_assessment['overall_strength'] = (
            strength_assessment['pattern_strength'] * 0.4 +
            strength_assessment['relationship_strength'] * 0.3 +
            strength_assessment['level_quality'] * 0.3
        )
        
        return strength_assessment
        
    def get_trading_recommendation(self) -> Dict[str, Any]:
        """Get current trading recommendation based on Square of Nine analysis"""
        
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
        if bullish_strength > bearish_strength * 1.4:
            recommendation = 'BUY'
            confidence = min(1.0, bullish_strength / len(self.trading_signals))
        elif bearish_strength > bullish_strength * 1.4:
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
            },
            'spiral_context': {
                'center_value': self.current_center_value,
                'patterns_detected': len(self.detected_patterns),
                'strong_levels_nearby': len([p for p in self.significant_levels 
                                            if abs(p.value - self.price_data[-1, 4]) / self.price_data[-1, 4] <= 0.05])
            }
        }

def run_square_of_nine_demo():
    """Demonstrate the Gann Square of Nine indicator"""
    
    print("=== Gann Square of Nine Indicator Demo ===")
    
    # Create sample OHLCV data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Generate realistic price data with Gann-like movements
    base_price = 144.0  # Start with a sacred number
    price_data = []
    
    for i, date in enumerate(dates):
        # Add Gann-like movements with sacred number influences
        sacred_influence = sum(math.sin(i * 2 * math.pi / sacred) * 0.001 
                             for sacred in [9, 21, 49, 81])
        
        # Add trend and noise
        trend = 0.0002 * i + sacred_influence
        noise = np.random.normal(0, 0.015)
        
        price = base_price * (1 + trend + noise)
        
        # Occasional jumps to sacred number levels
        if i % 30 == 0:  # Every 30 days
            nearest_sacred = min([81, 100, 121, 144, 169, 196, 225], 
                                key=lambda x: abs(x - price))
            if abs(price - nearest_sacred) / price < 0.1:
                price = nearest_sacred + np.random.normal(0, 1)
        
        # OHLCV format
        high = price * (1 + abs(np.random.normal(0, 0.008)))
        low = price * (1 - abs(np.random.normal(0, 0.008)))
        volume = np.random.randint(5000, 15000)
        
        price_data.append([
            date.timestamp(),
            price * 0.998,  # Open
            high,            # High
            low,             # Low
            price,           # Close
            volume           # Volume
        ])
        
        base_price = price
        
    ohlcv_data = np.array(price_data)
    
    # Initialize indicator with custom config
    config = SquareOfNineConfig()
    config.max_rings = 8
    config.center_value_mode = "auto"
    
    indicator = GannSquareOfNineIndicator(config)
    
    # Update with data
    print("Updating Square of Nine indicator with sample data...")
    result = indicator.update(ohlcv_data)
    
    print(f"Indicator updated successfully: {result}")
    
    # Get analysis
    analysis = indicator.get_current_spiral_analysis()
    
    print(f"\n=== Square of Nine Analysis ===")
    print(f"Current Price: ${analysis.get('current_price', 0):.2f}")
    print(f"Center Value: ${analysis.get('center_value', 0):.2f}")
    
    spiral_stats = analysis.get('spiral_stats', {})
    print(f"\n=== Spiral Statistics ===")
    print(f"Total Points: {spiral_stats.get('total_points', 0)}")
    print(f"Rings Calculated: {spiral_stats.get('rings_calculated', 0)}")
    print(f"Cardinal Points: {spiral_stats.get('cardinal_points', 0)}")
    print(f"Significant Levels: {spiral_stats.get('significant_levels', 0)}")
    
    pattern_analysis = analysis.get('pattern_analysis', {})
    print(f"\n=== Pattern Analysis ===")
    print(f"Patterns Detected: {pattern_analysis.get('patterns_detected', 0)}")
    print(f"Relationships Found: {pattern_analysis.get('relationships_found', 0)}")
    print(f"Projections Generated: {pattern_analysis.get('projections_generated', 0)}")
    print(f"Trading Signals: {pattern_analysis.get('trading_signals', 0)}")
    
    # Show current position analysis
    current_position = analysis.get('current_position', {})
    if current_position:
        print(f"\n=== Current Position in Spiral ===")
        nearest_point = current_position.get('nearest_spiral_point', {})
        if nearest_point:
            print(f"Nearest Point: ${nearest_point.get('value', 0):.2f}")
            print(f"Ring: {nearest_point.get('ring', 0)}")
            print(f"Angle: {nearest_point.get('angle', 0):.1f}°")
            print(f"Cardinal Direction: {nearest_point.get('cardinal_direction', 'Unknown')}")
            print(f"Significance: {nearest_point.get('significance', 0):.2f}")
            print(f"Distance: {nearest_point.get('distance_pct', 0):.2f}%")
            
        nearest_cardinals = current_position.get('nearest_cardinals', [])
        if nearest_cardinals:
            print(f"\n=== Nearest Cardinal Points ===")
            for i, cardinal in enumerate(nearest_cardinals[:3]):
                print(f"{i+1}. {cardinal['direction']}: ${cardinal['value']:.2f} "
                      f"({cardinal['distance_pct']:.2f}% away, sig: {cardinal['significance']:.2f})")
    
    # Show nearest levels
    nearest_levels = analysis.get('nearest_levels', {})
    if nearest_levels:
        print(f"\n=== Nearest Spiral Levels ===")
        if 'nearest_support' in nearest_levels:
            support = nearest_levels['nearest_support']
            print(f"Support: ${support['value']:.2f} ({support['distance_pct']:.2f}% below)")
            print(f"  Direction: {support['cardinal_direction']}, Ring: {support['ring']}, Sig: {support['significance']:.2f}")
            
        if 'nearest_resistance' in nearest_levels:
            resistance = nearest_levels['nearest_resistance']
            print(f"Resistance: ${resistance['value']:.2f} ({resistance['distance_pct']:.2f}% above)")
            print(f"  Direction: {resistance['cardinal_direction']}, Ring: {resistance['ring']}, Sig: {resistance['significance']:.2f}")
    
    # Show spiral strength assessment
    spiral_strength = analysis.get('spiral_strength', {})
    if spiral_strength:
        print(f"\n=== Spiral Strength Assessment ===")
        print(f"Pattern Strength: {spiral_strength.get('pattern_strength', 0):.2f}")
        print(f"Relationship Strength: {spiral_strength.get('relationship_strength', 0):.2f}")
        print(f"Level Quality: {spiral_strength.get('level_quality', 0):.2f}")
        print(f"Overall Strength: {spiral_strength.get('overall_strength', 0):.2f}")
    
    # Show detected patterns (top 3)
    if hasattr(indicator, 'detected_patterns') and indicator.detected_patterns:
        print(f"\n=== Top Detected Patterns ===")
        for i, pattern in enumerate(indicator.detected_patterns[:3]):
            print(f"{i+1}. {pattern.pattern_type.value.replace('_', ' ').title()}")
            print(f"   Strength: {pattern.strength:.2f}, Confidence: {pattern.confidence:.2f}")
            if pattern.geometric_ratio:
                print(f"   Geometric Ratio: {pattern.geometric_ratio:.3f}")
            if pattern.harmonic_level:
                print(f"   Harmonic Level: {pattern.harmonic_level}")
            print(f"   Points: {len(pattern.points)}")
    
    # Show relationships (top 3)
    if hasattr(indicator, 'spiral_relationships') and indicator.spiral_relationships:
        print(f"\n=== Top Spiral Relationships ===")
        for i, rel in enumerate(indicator.spiral_relationships[:3]):
            print(f"{i+1}. {rel.relationship_type.replace('_', ' ').title()}")
            print(f"   Values: ${rel.point1.value:.2f} ↔ ${rel.point2.value:.2f}")
            print(f"   Ratio: {rel.ratio:.3f}, Strength: {rel.strength:.2f}")
            print(f"   Angle Difference: {rel.angle_difference:.1f}°")
            if rel.harmonic_resonance:
                print(f"   Harmonic Resonance: {rel.harmonic_resonance:.3f}")
    
    # Show projections (top 3)
    if hasattr(indicator, 'projections') and indicator.projections:
        print(f"\n=== Top Projections ===")
        for i, proj in enumerate(indicator.projections[:3]):
            print(f"{i+1}. {proj.projection_type.replace('_', ' ').title()}: ${proj.target_level:.2f}")
            print(f"   Confidence: {proj.confidence:.2f}")
            current_price = analysis.get('current_price', 0)
            if current_price > 0:
                move_pct = (proj.target_level - current_price) / current_price * 100
                print(f"   Move: {move_pct:+.2f}%")
    
    # Get trading recommendation
    recommendation = indicator.get_trading_recommendation()
    
    print(f"\n=== Trading Recommendation ===")
    print(f"Recommendation: {recommendation['recommendation']}")
    print(f"Confidence: {recommendation['confidence']:.2f}")
    print(f"Reason: {recommendation.get('reason', 'Based on Square of Nine analysis')}")
    
    if 'primary_signal' in recommendation:
        signal = recommendation['primary_signal']
        print(f"\n=== Primary Signal ===")
        print(f"Type: {signal['type'].upper()}")
        print(f"Strength: {signal['strength']:.2f}")
        print(f"Entry: ${signal['entry']:.2f}")
        print(f"Target: ${signal['target']:.2f}")
        print(f"Stop Loss: ${signal['stop_loss']:.2f}")
        print(f"Source: {signal['source'].replace('_', ' ').title()}")
        
        # Calculate risk/reward
        if signal['type'].lower() == 'bullish':
            risk = signal['entry'] - signal['stop_loss']
            reward = signal['target'] - signal['entry']
        else:
            risk = signal['stop_loss'] - signal['entry']
            reward = signal['entry'] - signal['target']
            
        if risk > 0:
            rr_ratio = reward / risk
            print(f"Risk/Reward Ratio: {rr_ratio:.2f}")
    
    signal_summary = recommendation.get('signal_summary', {})
    if signal_summary:
        print(f"\n=== Signal Summary ===")
        print(f"Total Signals: {signal_summary['total_signals']}")
        print(f"Bullish: {signal_summary['bullish_signals']} (strength: {signal_summary['bullish_strength']:.2f})")
        print(f"Bearish: {signal_summary['bearish_signals']} (strength: {signal_summary['bearish_strength']:.2f})")
    
    spiral_context = recommendation.get('spiral_context', {})
    if spiral_context:
        print(f"\n=== Spiral Context ===")
        print(f"Center Value: ${spiral_context['center_value']:.2f}")
        print(f"Patterns Detected: {spiral_context['patterns_detected']}")
        print(f"Strong Levels Nearby: {spiral_context['strong_levels_nearby']}")
    
    # Show some spiral calculation examples
    print(f"\n=== Spiral Calculation Examples ===")
    center = indicator.current_center_value
    print(f"Center Value: {center}")
    
    # Show some calculated spiral values
    for ring in range(1, 4):
        if ring in indicator.spiral_rings:
            ring_points = indicator.spiral_rings[ring][:8]  # First 8 points of each ring
            print(f"\nRing {ring} (first 8 points):")
            for i, point in enumerate(ring_points):
                print(f"  {i+1}: ${point.value:.2f} at {point.angle:.0f}° ({point.cardinal_direction})")
    
    print(f"\n=== Demo Complete ===")
    print("\nThe Square of Nine indicator provides sophisticated spiral-based analysis")
    print("using W.D. Gann's mathematical principles, sacred number relationships,")
    print("and advanced pattern recognition for trading signal generation.")


if __name__ == "__main__":
    run_square_of_nine_demo()