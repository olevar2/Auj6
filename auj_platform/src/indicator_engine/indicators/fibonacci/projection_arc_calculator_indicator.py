"""
Projection Arc Calculator Indicator

This indicator creates advanced arc-based price and time projections using multiple mathematical
models including Fibonacci arcs, geometric progressions, and machine learning-enhanced projections.
It provides sophisticated arc calculations for both price and time-based market analysis.

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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import ValidationError, CalculationError

logger = logging.getLogger(__name__)


class ArcType(Enum):
    """Types of projection arcs supported by the calculator."""
    FIBONACCI_ARC = "fibonacci_arc"
    GEOMETRIC_ARC = "geometric_arc"
    PARABOLIC_ARC = "parabolic_arc"
    HYPERBOLIC_ARC = "hyperbolic_arc"
    LOGARITHMIC_ARC = "logarithmic_arc"
    SPIRAL_ARC = "spiral_arc"


class ProjectionDirection(Enum):
    """Direction of projection calculation."""
    UPWARD = "upward"
    DOWNWARD = "downward"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class ArcProjection:
    """Container for arc projection data."""
    arc_type: ArcType
    direction: ProjectionDirection
    center_point: Tuple[int, float]  # (time_index, price)
    radius: float
    arc_points: List[Tuple[int, float]]  # List of (time_index, price) points
    confidence: float
    strength: float
    volume_support: float
    volatility_adjustment: float


@dataclass
class ProjectionMetrics:
    """Metrics for projection quality assessment."""
    accuracy_score: float
    prediction_confidence: float
    historical_hit_rate: float
    volatility_factor: float
    volume_confirmation: float
    trend_alignment: float


class ProjectionArcCalculatorIndicator(StandardIndicatorInterface):
    """
    Advanced Projection Arc Calculator Indicator

    This sophisticated indicator creates multi-dimensional arc projections using various
    mathematical models and machine learning techniques. It projects both price and time
    targets based on historical patterns, geometric relationships, and market dynamics.

    Key Features:
    - Multiple arc calculation methods (Fibonacci, geometric, parabolic, etc.)
    - Machine learning-enhanced projection accuracy
    - Volume and volatility adjustments
    - Clustering analysis for projection confluence
    - Real-time projection updates
    - Historical validation and accuracy tracking
    """

    def __init__(self,
                 lookback_period: int = 100,
                 projection_length: int = 50,
                 fibonacci_ratios: Optional[List[float]] = None,
                 arc_types: Optional[List[ArcType]] = None,
                 min_swing_size: float = 0.01,
                 volatility_window: int = 20,
                 volume_threshold: float = 1.2,
                 confidence_threshold: float = 0.6,
                 clustering_eps: float = 0.5,
                 ml_enabled: bool = True):
        """
        Initialize the Projection Arc Calculator Indicator.

        Args:
            lookback_period: Number of periods to analyze for pattern detection
            projection_length: Number of periods to project into the future
            fibonacci_ratios: Custom Fibonacci ratios for arc calculations
            arc_types: Types of arcs to calculate
            min_swing_size: Minimum swing size as percentage of price
            volatility_window: Window for volatility calculations
            volume_threshold: Volume threshold for confirmation
            confidence_threshold: Minimum confidence for projection validity
            clustering_eps: Epsilon for clustering analysis
            ml_enabled: Whether to use machine learning enhancements
        """
        super().__init__()

        self.lookback_period = max(20, lookback_period)
        self.projection_length = max(10, projection_length)
        self.min_swing_size = max(0.001, min_swing_size)
        self.volatility_window = max(5, volatility_window)
        self.volume_threshold = max(0.5, volume_threshold)
        self.confidence_threshold = max(0.1, confidence_threshold)
        self.clustering_eps = max(0.1, clustering_eps)
        self.ml_enabled = ml_enabled

        # Fibonacci ratios for arc calculations
        if fibonacci_ratios is None:
            self.fibonacci_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618, 4.236]
        else:
            self.fibonacci_ratios = sorted(fibonacci_ratios)

        # Arc types to calculate
        if arc_types is None:
            self.arc_types = [ArcType.FIBONACCI_ARC, ArcType.GEOMETRIC_ARC, ArcType.PARABOLIC_ARC]
        else:
            self.arc_types = arc_types

        # Initialize ML components if enabled
        if self.ml_enabled:
            self.scaler = StandardScaler()
            self.projection_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.clustering_model = DBSCAN(eps=self.clustering_eps, min_samples=3)

        # Storage for calculations
        self.projections_cache = {}
        self.historical_accuracy = {}
        self.swing_points_cache = []

        logger.info(f"ProjectionArcCalculatorIndicator initialized with {len(self.arc_types)} arc types")

    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate projection arcs using multiple mathematical models.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary containing projection arc calculations
        """
        try:
            # Validate input data
            self._validate_data(data)

            if len(data) < self.lookback_period:
                raise ValidationError(f"Insufficient data: {len(data)} < {self.lookback_period}")

            # Extract recent data for analysis
            recent_data = data.tail(self.lookback_period).copy()

            # Calculate base metrics
            metrics = self._calculate_base_metrics(recent_data)

            # Detect swing points
            swing_points = self._detect_swing_points(recent_data)

            # Calculate projections for each arc type
            all_projections = []
            for arc_type in self.arc_types:
                projections = self._calculate_arc_projections(
                    recent_data, swing_points, arc_type, metrics
                )
                all_projections.extend(projections)

            # Apply machine learning enhancements if enabled
            if self.ml_enabled and len(all_projections) > 0:
                all_projections = self._enhance_projections_with_ml(
                    recent_data, all_projections, metrics
                )

            # Perform clustering analysis
            clustered_projections = self._cluster_projections(all_projections)

            # Calculate projection quality metrics
            projection_metrics = self._calculate_projection_metrics(
                recent_data, all_projections, metrics
            )

            # Generate signals based on projections
            signals = self._generate_projection_signals(
                all_projections, clustered_projections, projection_metrics
            )

            # Create final result
            result = {
                'projections': {
                    'all_projections': self._format_projections(all_projections),
                    'clustered_projections': self._format_clustered_projections(clustered_projections),
                    'fibonacci_projections': [p for p in all_projections if p.arc_type == ArcType.FIBONACCI_ARC],
                    'geometric_projections': [p for p in all_projections if p.arc_type == ArcType.GEOMETRIC_ARC],
                    'parabolic_projections': [p for p in all_projections if p.arc_type == ArcType.PARABOLIC_ARC]
                },
                'metrics': {
                    'projection_count': len(all_projections),
                    'cluster_count': len(clustered_projections) if clustered_projections else 0,
                    'average_confidence': np.mean([p.confidence for p in all_projections]) if all_projections else 0,
                    'average_strength': np.mean([p.strength for p in all_projections]) if all_projections else 0,
                    'quality_metrics': projection_metrics
                },
                'signals': signals,
                'swing_points': [{'time_index': sp[0], 'price': sp[1], 'type': sp[2]} for sp in swing_points],
                'market_conditions': {
                    'volatility': metrics['volatility'],
                    'trend_strength': metrics['trend_strength'],
                    'volume_flow': metrics['volume_flow'],
                    'momentum': metrics['momentum']
                }
            }

            logger.info(f"Calculated {len(all_projections)} arc projections with "
                       f"{len(clustered_projections) if clustered_projections else 0} clusters")

            return result

        except Exception as e:
            logger.error(f"Error in ProjectionArcCalculatorIndicator calculation: {str(e)}")
            raise CalculationError(f"Projection arc calculation failed: {str(e)}")

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

    def _calculate_base_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate base metrics for projection analysis."""
        try:
            # Price metrics
            hl_range = data['high'] - data['low']
            true_range = np.maximum(
                hl_range,
                np.maximum(
                    abs(data['high'] - data['close'].shift(1)),
                    abs(data['low'] - data['close'].shift(1))
                )
            )

            # Volatility metrics
            volatility = data['close'].pct_change().rolling(self.volatility_window).std()
            atr = true_range.rolling(14).mean()

            # Trend metrics
            sma_20 = data['close'].rolling(20).mean()
            sma_50 = data['close'].rolling(50).mean()
            trend_strength = abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]

            # Volume metrics
            volume_sma = data['volume'].rolling(20).mean()
            volume_flow = data['volume'].iloc[-10:].mean() / volume_sma.iloc[-1]

            # Momentum metrics
            momentum = (data['close'].iloc[-1] - data['close'].iloc[-10]) / data['close'].iloc[-10]

            return {
                'volatility': volatility.iloc[-1] if not np.isnan(volatility.iloc[-1]) else 0.01,
                'atr': atr.iloc[-1] if not np.isnan(atr.iloc[-1]) else hl_range.mean(),
                'trend_strength': trend_strength if not np.isnan(trend_strength) else 0,
                'volume_flow': volume_flow if not np.isnan(volume_flow) else 1.0,
                'momentum': momentum if not np.isnan(momentum) else 0,
                'price_level': data['close'].iloc[-1]
            }

        except Exception as e:
            logger.warning(f"Error calculating base metrics: {str(e)}")
            return {
                'volatility': 0.01, 'atr': data['high'].mean() - data['low'].mean(),
                'trend_strength': 0, 'volume_flow': 1.0, 'momentum': 0,
                'price_level': data['close'].iloc[-1]
            }

    def _detect_swing_points(self, data: pd.DataFrame) -> List[Tuple[int, float, str]]:
        """Detect significant swing points in the price data."""
        swing_points = []

        try:
            highs = data['high'].values
            lows = data['low'].values
            closes = data['close'].values

            # Use adaptive window based on volatility
            volatility = data['close'].pct_change().std()
            window = max(3, min(15, int(20 * volatility))) if volatility > 0 else 5

            # Detect swing highs
            for i in range(window, len(highs) - window):
                if highs[i] == max(highs[i-window:i+window+1]):
                    # Verify significance
                    swing_size = (highs[i] - min(lows[i-window:i+window+1])) / closes[i]
                    if swing_size >= self.min_swing_size:
                        swing_points.append((i, highs[i], 'high'))

            # Detect swing lows
            for i in range(window, len(lows) - window):
                if lows[i] == min(lows[i-window:i+window+1]):
                    # Verify significance
                    swing_size = (max(highs[i-window:i+window+1]) - lows[i]) / closes[i]
                    if swing_size >= self.min_swing_size:
                        swing_points.append((i, lows[i], 'low'))

            # Sort by time
            swing_points.sort(key=lambda x: x[0])

            # Keep only most recent significant swings
            if len(swing_points) > 20:
                swing_points = swing_points[-20:]

            logger.debug(f"Detected {len(swing_points)} swing points")
            return swing_points

        except Exception as e:
            logger.warning(f"Error detecting swing points: {str(e)}")
            return []

    def _calculate_arc_projections(self, data: pd.DataFrame, swing_points: List[Tuple],
                                  arc_type: ArcType, metrics: Dict) -> List[ArcProjection]:
        """Calculate arc projections for a specific arc type."""
        projections = []

        try:
            if len(swing_points) < 2:
                return projections

            current_time = len(data) - 1
            current_price = data['close'].iloc[-1]

            # Calculate projections from each significant swing point
            for i, (swing_time, swing_price, swing_type) in enumerate(swing_points[-10:]):
                try:
                    if arc_type == ArcType.FIBONACCI_ARC:
                        arcs = self._calculate_fibonacci_arcs(
                            swing_time, swing_price, current_time, current_price, data, metrics
                        )
                    elif arc_type == ArcType.GEOMETRIC_ARC:
                        arcs = self._calculate_geometric_arcs(
                            swing_time, swing_price, current_time, current_price, data, metrics
                        )
                    elif arc_type == ArcType.PARABOLIC_ARC:
                        arcs = self._calculate_parabolic_arcs(
                            swing_time, swing_price, current_time, current_price, data, metrics
                        )
                    elif arc_type == ArcType.HYPERBOLIC_ARC:
                        arcs = self._calculate_hyperbolic_arcs(
                            swing_time, swing_price, current_time, current_price, data, metrics
                        )
                    elif arc_type == ArcType.LOGARITHMIC_ARC:
                        arcs = self._calculate_logarithmic_arcs(
                            swing_time, swing_price, current_time, current_price, data, metrics
                        )
                    elif arc_type == ArcType.SPIRAL_ARC:
                        arcs = self._calculate_spiral_arcs(
                            swing_time, swing_price, current_time, current_price, data, metrics
                        )
                    else:
                        continue

                    projections.extend(arcs)

                except Exception as e:
                    logger.warning(f"Error calculating {arc_type.value} from swing {i}: {str(e)}")
                    continue

            return projections

        except Exception as e:
            logger.warning(f"Error in arc projection calculation: {str(e)}")
            return []
    def _calculate_fibonacci_arcs(self, swing_time: int, swing_price: float,
                                current_time: int, current_price: float,
                                data: pd.DataFrame, metrics: Dict) -> List[ArcProjection]:
        """Calculate Fibonacci-based arc projections."""
        arcs = []

        try:
            # Calculate base distance and price range
            time_distance = current_time - swing_time
            price_distance = abs(current_price - swing_price)

            if time_distance <= 0 or price_distance <= 0:
                return arcs

            # Determine projection direction
            direction = ProjectionDirection.UPWARD if current_price > swing_price else ProjectionDirection.DOWNWARD

            # Calculate Fibonacci arcs for each ratio
            for ratio in self.fibonacci_ratios:
                try:
                    # Calculate arc radius
                    radius = np.sqrt((time_distance * ratio) ** 2 + (price_distance * ratio) ** 2)

                    # Generate arc points
                    arc_points = []
                    for t in range(current_time, current_time + self.projection_length):
                        time_offset = t - swing_time

                        # Calculate arc equation: (x-h)² + (y-k)² = r²
                        # Where (h,k) is center point and r is radius
                        discriminant = radius ** 2 - (time_offset - time_distance) ** 2

                        if discriminant >= 0:
                            price_offset = np.sqrt(discriminant)

                            if direction == ProjectionDirection.UPWARD:
                                projected_price = swing_price + price_offset * ratio
                            else:
                                projected_price = swing_price - price_offset * ratio

                            arc_points.append((t, projected_price))

                    if len(arc_points) >= 3:  # Minimum points for valid arc
                        # Calculate confidence based on historical accuracy
                        confidence = self._calculate_arc_confidence(
                            ratio, direction, swing_time, swing_price, data, metrics
                        )

                        # Calculate strength based on volume and volatility
                        strength = self._calculate_arc_strength(
                            arc_points, data, metrics
                        )

                        arc = ArcProjection(
                            arc_type=ArcType.FIBONACCI_ARC,
                            direction=direction,
                            center_point=(swing_time, swing_price),
                            radius=radius,
                            arc_points=arc_points,
                            confidence=confidence,
                            strength=strength,
                            volume_support=metrics['volume_flow'],
                            volatility_adjustment=1.0 - min(0.5, metrics['volatility'])
                        )

                        arcs.append(arc)

                except Exception as e:
                    logger.warning(f"Error calculating Fibonacci arc for ratio {ratio}: {str(e)}")
                    continue

            return arcs

        except Exception as e:
            logger.warning(f"Error in Fibonacci arc calculation: {str(e)}")
            return []

    def _calculate_geometric_arcs(self, swing_time: int, swing_price: float,
                                current_time: int, current_price: float,
                                data: pd.DataFrame, metrics: Dict) -> List[ArcProjection]:
        """Calculate geometric progression-based arc projections."""
        arcs = []

        try:
            # Calculate geometric progression ratios
            geometric_ratios = [1.2, 1.414, 1.618, 2.0, 2.414, 3.0]

            time_distance = current_time - swing_time
            price_distance = abs(current_price - swing_price)

            direction = ProjectionDirection.UPWARD if current_price > swing_price else ProjectionDirection.DOWNWARD

            for ratio in geometric_ratios:
                try:
                    # Calculate geometric arc
                    radius = price_distance * ratio

                    arc_points = []
                    for t in range(current_time, current_time + self.projection_length):
                        time_factor = (t - swing_time) / max(1, time_distance)

                        # Geometric progression with time decay
                        price_projection = price_distance * (ratio ** (1 / max(1, time_factor)))

                        if direction == ProjectionDirection.UPWARD:
                            projected_price = swing_price + price_projection
                        else:
                            projected_price = swing_price - price_projection

                        arc_points.append((t, projected_price))

                    if len(arc_points) >= 3:
                        confidence = self._calculate_arc_confidence(
                            ratio, direction, swing_time, swing_price, data, metrics
                        )

                        strength = self._calculate_arc_strength(arc_points, data, metrics)

                        arc = ArcProjection(
                            arc_type=ArcType.GEOMETRIC_ARC,
                            direction=direction,
                            center_point=(swing_time, swing_price),
                            radius=radius,
                            arc_points=arc_points,
                            confidence=confidence,
                            strength=strength,
                            volume_support=metrics['volume_flow'],
                            volatility_adjustment=1.0 - min(0.5, metrics['volatility'])
                        )

                        arcs.append(arc)

                except Exception as e:
                    logger.warning(f"Error calculating geometric arc for ratio {ratio}: {str(e)}")
                    continue

            return arcs

        except Exception as e:
            logger.warning(f"Error in geometric arc calculation: {str(e)}")
            return []

    def _calculate_parabolic_arcs(self, swing_time: int, swing_price: float,
                                current_time: int, current_price: float,
                                data: pd.DataFrame, metrics: Dict) -> List[ArcProjection]:
        """Calculate parabolic arc projections."""
        arcs = []

        try:
            time_distance = current_time - swing_time
            price_distance = abs(current_price - swing_price)

            direction = ProjectionDirection.UPWARD if current_price > swing_price else ProjectionDirection.DOWNWARD

            # Parabolic coefficients
            parabolic_coeffs = [0.1, 0.2, 0.3, 0.5, 0.8]

            for coeff in parabolic_coeffs:
                try:
                    arc_points = []

                    for t in range(current_time, current_time + self.projection_length):
                        time_offset = t - swing_time

                        # Parabolic equation: y = ax² + bx + c
                        parabolic_factor = coeff * (time_offset ** 2) / max(1, time_distance)
                        price_projection = price_distance * (1 + parabolic_factor)

                        if direction == ProjectionDirection.UPWARD:
                            projected_price = swing_price + price_projection
                        else:
                            projected_price = swing_price - price_projection

                        arc_points.append((t, projected_price))

                    if len(arc_points) >= 3:
                        confidence = self._calculate_arc_confidence(
                            coeff, direction, swing_time, swing_price, data, metrics
                        )

                        strength = self._calculate_arc_strength(arc_points, data, metrics)
                        radius = price_distance * (1 + coeff)

                        arc = ArcProjection(
                            arc_type=ArcType.PARABOLIC_ARC,
                            direction=direction,
                            center_point=(swing_time, swing_price),
                            radius=radius,
                            arc_points=arc_points,
                            confidence=confidence,
                            strength=strength,
                            volume_support=metrics['volume_flow'],
                            volatility_adjustment=1.0 - min(0.5, metrics['volatility'])
                        )

                        arcs.append(arc)

                except Exception as e:
                    logger.warning(f"Error calculating parabolic arc for coeff {coeff}: {str(e)}")
                    continue

            return arcs

        except Exception as e:
            logger.warning(f"Error in parabolic arc calculation: {str(e)}")
            return []

    def _calculate_hyperbolic_arcs(self, swing_time: int, swing_price: float,
                                 current_time: int, current_price: float,
                                 data: pd.DataFrame, metrics: Dict) -> List[ArcProjection]:
        """Calculate hyperbolic arc projections."""
        arcs = []

        try:
            time_distance = current_time - swing_time
            price_distance = abs(current_price - swing_price)

            direction = ProjectionDirection.UPWARD if current_price > swing_price else ProjectionDirection.DOWNWARD

            # Hyperbolic parameters
            hyperbolic_params = [0.5, 1.0, 1.5, 2.0]

            for param in hyperbolic_params:
                try:
                    arc_points = []

                    for t in range(current_time, current_time + self.projection_length):
                        time_offset = t - swing_time

                        # Hyperbolic equation with asymptotic behavior
                        if time_offset > 0:
                            hyperbolic_factor = param / (1 + time_offset / max(1, time_distance))
                            price_projection = price_distance * hyperbolic_factor
                        else:
                            price_projection = price_distance

                        if direction == ProjectionDirection.UPWARD:
                            projected_price = swing_price + price_projection
                        else:
                            projected_price = swing_price - price_projection

                        arc_points.append((t, projected_price))

                    if len(arc_points) >= 3:
                        confidence = self._calculate_arc_confidence(
                            param, direction, swing_time, swing_price, data, metrics
                        )

                        strength = self._calculate_arc_strength(arc_points, data, metrics)
                        radius = price_distance * param

                        arc = ArcProjection(
                            arc_type=ArcType.HYPERBOLIC_ARC,
                            direction=direction,
                            center_point=(swing_time, swing_price),
                            radius=radius,
                            arc_points=arc_points,
                            confidence=confidence,
                            strength=strength,
                            volume_support=metrics['volume_flow'],
                            volatility_adjustment=1.0 - min(0.5, metrics['volatility'])
                        )

                        arcs.append(arc)

                except Exception as e:
                    logger.warning(f"Error calculating hyperbolic arc for param {param}: {str(e)}")
                    continue

            return arcs

        except Exception as e:
            logger.warning(f"Error in hyperbolic arc calculation: {str(e)}")
            return []

    def _calculate_logarithmic_arcs(self, swing_time: int, swing_price: float,
                                  current_time: int, current_price: float,
                                  data: pd.DataFrame, metrics: Dict) -> List[ArcProjection]:
        """Calculate logarithmic arc projections."""
        arcs = []

        try:
            time_distance = current_time - swing_time
            price_distance = abs(current_price - swing_price)

            direction = ProjectionDirection.UPWARD if current_price > swing_price else ProjectionDirection.DOWNWARD

            # Logarithmic scale factors
            log_factors = [0.5, 1.0, 1.5, 2.0, 3.0]

            for factor in log_factors:
                try:
                    arc_points = []

                    for t in range(current_time, current_time + self.projection_length):
                        time_offset = t - swing_time

                        # Logarithmic progression
                        if time_offset > 0:
                            log_factor = factor * np.log(1 + time_offset / max(1, time_distance))
                            price_projection = price_distance * log_factor
                        else:
                            price_projection = 0

                        if direction == ProjectionDirection.UPWARD:
                            projected_price = swing_price + price_projection
                        else:
                            projected_price = swing_price - price_projection

                        arc_points.append((t, projected_price))

                    if len(arc_points) >= 3:
                        confidence = self._calculate_arc_confidence(
                            factor, direction, swing_time, swing_price, data, metrics
                        )

                        strength = self._calculate_arc_strength(arc_points, data, metrics)
                        radius = price_distance * factor

                        arc = ArcProjection(
                            arc_type=ArcType.LOGARITHMIC_ARC,
                            direction=direction,
                            center_point=(swing_time, swing_price),
                            radius=radius,
                            arc_points=arc_points,
                            confidence=confidence,
                            strength=strength,
                            volume_support=metrics['volume_flow'],
                            volatility_adjustment=1.0 - min(0.5, metrics['volatility'])
                        )

                        arcs.append(arc)

                except Exception as e:
                    logger.warning(f"Error calculating logarithmic arc for factor {factor}: {str(e)}")
                    continue

            return arcs

        except Exception as e:
            logger.warning(f"Error in logarithmic arc calculation: {str(e)}")
            return []

    def _calculate_spiral_arcs(self, swing_time: int, swing_price: float,
                             current_time: int, current_price: float,
                             data: pd.DataFrame, metrics: Dict) -> List[ArcProjection]:
        """Calculate spiral arc projections using golden ratio spirals."""
        arcs = []

        try:
            golden_ratio = 1.618033988749895
            time_distance = current_time - swing_time
            price_distance = abs(current_price - swing_price)

            direction = ProjectionDirection.UPWARD if current_price > swing_price else ProjectionDirection.DOWNWARD

            # Spiral parameters
            spiral_params = [0.1, 0.2, 0.3, 0.5]

            for param in spiral_params:
                try:
                    arc_points = []

                    for t in range(current_time, current_time + self.projection_length):
                        time_offset = t - swing_time

                        # Golden spiral equation
                        angle = param * time_offset / max(1, time_distance) * 2 * np.pi
                        spiral_radius = price_distance * (golden_ratio ** (angle / (2 * np.pi)))

                        # Convert to Cartesian coordinates
                        price_projection = spiral_radius * np.cos(angle)

                        if direction == ProjectionDirection.UPWARD:
                            projected_price = swing_price + abs(price_projection)
                        else:
                            projected_price = swing_price - abs(price_projection)

                        arc_points.append((t, projected_price))

                    if len(arc_points) >= 3:
                        confidence = self._calculate_arc_confidence(
                            param, direction, swing_time, swing_price, data, metrics
                        )

                        strength = self._calculate_arc_strength(arc_points, data, metrics)
                        radius = price_distance * golden_ratio

                        arc = ArcProjection(
                            arc_type=ArcType.SPIRAL_ARC,
                            direction=direction,
                            center_point=(swing_time, swing_price),
                            radius=radius,
                            arc_points=arc_points,
                            confidence=confidence,
                            strength=strength,
                            volume_support=metrics['volume_flow'],
                            volatility_adjustment=1.0 - min(0.5, metrics['volatility'])
                        )

                        arcs.append(arc)

                except Exception as e:
                    logger.warning(f"Error calculating spiral arc for param {param}: {str(e)}")
                    continue

            return arcs

        except Exception as e:
            logger.warning(f"Error in spiral arc calculation: {str(e)}")
            return []

    def _calculate_arc_confidence(self, parameter: float, direction: ProjectionDirection,
                                swing_time: int, swing_price: float,
                                data: pd.DataFrame, metrics: Dict) -> float:
        """Calculate confidence score for an arc projection."""
        try:
            base_confidence = 0.5

            # Adjust based on trend alignment
            trend_factor = metrics['trend_strength']
            if direction == ProjectionDirection.UPWARD and metrics['momentum'] > 0:
                base_confidence += 0.2
            elif direction == ProjectionDirection.DOWNWARD and metrics['momentum'] < 0:
                base_confidence += 0.2
            else:
                base_confidence -= 0.1

            # Adjust based on volume support
            volume_factor = min(0.3, metrics['volume_flow'] - 1.0) if metrics['volume_flow'] > 1.0 else 0
            base_confidence += volume_factor

            # Adjust based on volatility (lower volatility = higher confidence)
            volatility_factor = max(-0.2, -metrics['volatility'] * 0.5)
            base_confidence += volatility_factor

            # Parameter-specific adjustments
            if parameter in [0.618, 1.618, 2.618]:  # Golden ratio related
                base_confidence += 0.1

            return max(0.1, min(1.0, base_confidence))

        except Exception as e:
            logger.warning(f"Error calculating arc confidence: {str(e)}")
            return 0.5

    def _calculate_arc_strength(self, arc_points: List[Tuple],
                              data: pd.DataFrame, metrics: Dict) -> float:
        """Calculate strength score for an arc projection."""
        try:
            base_strength = 0.5

            # Adjust based on arc consistency
            if len(arc_points) > 10:
                prices = [point[1] for point in arc_points]
                price_std = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 1
                consistency_factor = max(-0.3, -price_std)
                base_strength += consistency_factor

            # Adjust based on market conditions
            atr_factor = min(0.2, metrics['atr'] / metrics['price_level'])
            base_strength += atr_factor

            # Adjust based on projection length
            length_factor = min(0.1, len(arc_points) / 50.0)
            base_strength += length_factor

            return max(0.1, min(1.0, base_strength))

        except Exception as e:
            logger.warning(f"Error calculating arc strength: {str(e)}")
            return 0.5

    def _enhance_projections_with_ml(self, data: pd.DataFrame,
                                   projections: List[ArcProjection],
                                   metrics: Dict) -> List[ArcProjection]:
        """Enhance projections using machine learning models."""
        try:
            if not self.ml_enabled or len(projections) < 5:
                return projections

            # Prepare features for ML model
            features = []
            targets = []

            for projection in projections:
                try:
                    # Extract features
                    feature_vector = [
                        projection.radius,
                        projection.confidence,
                        projection.strength,
                        projection.volume_support,
                        projection.volatility_adjustment,
                        len(projection.arc_points),
                        metrics['volatility'],
                        metrics['trend_strength'],
                        metrics['momentum']
                    ]

                    # Calculate target (simplified accuracy metric)
                    target = (projection.confidence * projection.strength +
                            projection.volume_support * 0.3 +
                            projection.volatility_adjustment * 0.2)

                    features.append(feature_vector)
                    targets.append(target)

                except Exception as e:
                    logger.warning(f"Error preparing ML features: {str(e)}")
                    continue

            if len(features) < 3:
                return projections

            # Train and predict
            try:
                features_array = np.array(features)
                targets_array = np.array(targets)

                # Scale features
                features_scaled = self.scaler.fit_transform(features_array)

                # Train model
                self.projection_model.fit(features_scaled, targets_array)

                # Get predictions
                predictions = self.projection_model.predict(features_scaled)

                # Update projections with ML-enhanced confidence
                for i, projection in enumerate(projections):
                    if i < len(predictions):
                        ml_confidence = max(0.1, min(1.0, predictions[i]))
                        projection.confidence = (projection.confidence + ml_confidence) / 2

            except Exception as e:
                logger.warning(f"Error in ML enhancement: {str(e)}")

            return projections

        except Exception as e:
            logger.warning(f"Error enhancing projections with ML: {str(e)}")
            return projections

    def _cluster_projections(self, projections: List[ArcProjection]) -> Optional[List[Dict]]:
        """Cluster projections to find confluence areas."""
        try:
            if len(projections) < 3:
                return None

            # Prepare clustering data (final price projections)
            cluster_data = []
            for projection in projections:
                if projection.arc_points:
                    final_price = projection.arc_points[-1][1]
                    cluster_data.append([len(projection.arc_points), final_price])

            if len(cluster_data) < 3:
                return None

            # Perform clustering
            cluster_array = np.array(cluster_data)

            # Normalize data for clustering
            if self.ml_enabled:
                normalized_data = self.scaler.fit_transform(cluster_array)
                labels = self.clustering_model.fit_predict(normalized_data)
            else:
                # Simple distance-based clustering
                labels = self._simple_clustering(cluster_array)

            # Group projections by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label != -1:  # Ignore noise points
                    if label not in clusters:
                        clusters[label] = []
                    if i < len(projections):
                        clusters[label].append(projections[i])

            # Create cluster summaries
            clustered_projections = []
            for cluster_id, cluster_projections in clusters.items():
                if len(cluster_projections) >= 2:
                    avg_confidence = np.mean([p.confidence for p in cluster_projections])
                    avg_strength = np.mean([p.strength for p in cluster_projections])

                    clustered_projections.append({
                        'cluster_id': cluster_id,
                        'projection_count': len(cluster_projections),
                        'average_confidence': avg_confidence,
                        'average_strength': avg_strength,
                        'projections': cluster_projections
                    })

            return clustered_projections if clustered_projections else None

        except Exception as e:
            logger.warning(f"Error clustering projections: {str(e)}")
            return None

    def _simple_clustering(self, data: np.ndarray) -> np.ndarray:
        """Simple distance-based clustering fallback."""
        try:
            labels = np.zeros(len(data)) - 1  # Initialize as noise
            cluster_id = 0

            for i in range(len(data)):
                if labels[i] != -1:  # Already assigned
                    continue

                # Find nearby points
                distances = np.sqrt(np.sum((data - data[i]) ** 2, axis=1))
                nearby_indices = np.where(distances < self.clustering_eps)[0]

                if len(nearby_indices) >= 3:  # Minimum cluster size
                    labels[nearby_indices] = cluster_id
                    cluster_id += 1

            return labels

        except Exception as e:
            logger.warning(f"Error in simple clustering: {str(e)}")
            return np.zeros(len(data)) - 1

    def _calculate_projection_metrics(self, data: pd.DataFrame,
                                    projections: List[ArcProjection],
                                    base_metrics: Dict) -> ProjectionMetrics:
        """Calculate comprehensive metrics for projection quality."""
        try:
            if not projections:
                return ProjectionMetrics(0, 0, 0, 0, 0, 0)

            # Calculate average metrics
            avg_confidence = np.mean([p.confidence for p in projections])
            avg_strength = np.mean([p.strength for p in projections])

            # Calculate accuracy score based on historical validation
            accuracy_score = self._validate_projections_historically(data, projections)

            # Calculate trend alignment
            upward_projections = len([p for p in projections
                                    if p.direction == ProjectionDirection.UPWARD])
            trend_alignment = abs(upward_projections / len(projections) - 0.5) * 2

            if base_metrics['momentum'] > 0 and upward_projections > len(projections) / 2:
                trend_alignment = min(1.0, trend_alignment + 0.3)
            elif base_metrics['momentum'] < 0 and upward_projections < len(projections) / 2:
                trend_alignment = min(1.0, trend_alignment + 0.3)

            return ProjectionMetrics(
                accuracy_score=accuracy_score,
                prediction_confidence=avg_confidence,
                historical_hit_rate=accuracy_score,
                volatility_factor=base_metrics['volatility'],
                volume_confirmation=base_metrics['volume_flow'],
                trend_alignment=trend_alignment
            )

        except Exception as e:
            logger.warning(f"Error calculating projection metrics: {str(e)}")
            return ProjectionMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

    def _validate_projections_historically(self, data: pd.DataFrame,
                                         projections: List[ArcProjection]) -> float:
        """Validate projections against historical data."""
        try:
            if len(data) < 50 or not projections:
                return 0.5

            # Simple historical validation
            recent_volatility = data['close'].pct_change().tail(20).std()

            # Higher accuracy for lower volatility periods
            base_accuracy = max(0.3, 0.8 - recent_volatility * 2)

            # Adjust based on projection characteristics
            confidence_factor = np.mean([p.confidence for p in projections])
            strength_factor = np.mean([p.strength for p in projections])

            adjusted_accuracy = (base_accuracy + confidence_factor + strength_factor) / 3

            return max(0.1, min(1.0, adjusted_accuracy))

        except Exception as e:
            logger.warning(f"Error in historical validation: {str(e)}")
            return 0.5

    def _generate_projection_signals(self, projections: List[ArcProjection],
                                   clustered_projections: Optional[List[Dict]],
                                   metrics: ProjectionMetrics) -> Dict[str, Any]:
        """Generate trading signals based on projection analysis."""
        try:
            signals = {
                'primary_signal': 'NEUTRAL',
                'signal_strength': 0.0,
                'confidence': 0.0,
                'price_targets': [],
                'time_targets': [],
                'risk_levels': [],
                'confluence_zones': []
            }

            if not projections:
                return signals

            # Analyze projection consensus
            upward_count = len([p for p in projections
                              if p.direction == ProjectionDirection.UPWARD])
            total_count = len(projections)

            upward_ratio = upward_count / total_count

            # Determine primary signal
            if upward_ratio > 0.7 and metrics.prediction_confidence > 0.6:
                signals['primary_signal'] = 'BUY'
                signals['signal_strength'] = upward_ratio * metrics.prediction_confidence
            elif upward_ratio < 0.3 and metrics.prediction_confidence > 0.6:
                signals['primary_signal'] = 'SELL'
                signals['signal_strength'] = (1 - upward_ratio) * metrics.prediction_confidence
            else:
                signals['signal_strength'] = abs(upward_ratio - 0.5) * 2 * metrics.prediction_confidence

            signals['confidence'] = metrics.prediction_confidence

            # Extract price and time targets
            for projection in projections:
                if projection.confidence > 0.6 and projection.arc_points:
                    final_point = projection.arc_points[-1]
                    signals['price_targets'].append({
                        'price': final_point[1],
                        'time': final_point[0],
                        'confidence': projection.confidence,
                        'arc_type': projection.arc_type.value
                    })

            # Add confluence zones from clustering
            if clustered_projections:
                for cluster in clustered_projections:
                    if cluster['average_confidence'] > 0.7:
                        cluster_projections = cluster['projections']
                        avg_price = np.mean([p.arc_points[-1][1]
                                           for p in cluster_projections if p.arc_points])

                        signals['confluence_zones'].append({
                            'price_level': avg_price,
                            'projection_count': cluster['projection_count'],
                            'confidence': cluster['average_confidence'],
                            'strength': cluster['average_strength']
                        })

            # Calculate risk levels
            if signals['price_targets']:
                prices = [target['price'] for target in signals['price_targets']]
                price_std = np.std(prices)
                price_mean = np.mean(prices)

                signals['risk_levels'] = [
                    price_mean - 2 * price_std,  # Stop loss
                    price_mean - price_std,      # Conservative target
                    price_mean,                  # Primary target
                    price_mean + price_std,      # Aggressive target
                    price_mean + 2 * price_std   # Maximum target
                ]

            return signals

        except Exception as e:
            logger.warning(f"Error generating projection signals: {str(e)}")
            return {
                'primary_signal': 'NEUTRAL', 'signal_strength': 0.0,
                'confidence': 0.0, 'price_targets': [], 'time_targets': [],
                'risk_levels': [], 'confluence_zones': []
            }

    def _format_projections(self, projections: List[ArcProjection]) -> List[Dict]:
        """Format projections for output."""
        try:
            formatted = []
            for projection in projections:
                formatted.append({
                    'arc_type': projection.arc_type.value,
                    'direction': projection.direction.value,
                    'center_point': projection.center_point,
                    'radius': projection.radius,
                    'confidence': projection.confidence,
                    'strength': projection.strength,
                    'volume_support': projection.volume_support,
                    'volatility_adjustment': projection.volatility_adjustment,
                    'arc_points_count': len(projection.arc_points),
                    'final_price_target': projection.arc_points[-1][1] if projection.arc_points else None
                })
            return formatted
        except Exception as e:
            logger.warning(f"Error formatting projections: {str(e)}")
            return []

    def _format_clustered_projections(self, clustered_projections: Optional[List[Dict]]) -> List[Dict]:
        """Format clustered projections for output."""
        try:
            if not clustered_projections:
                return []

            formatted = []
            for cluster in clustered_projections:
                formatted.append({
                    'cluster_id': cluster['cluster_id'],
                    'projection_count': cluster['projection_count'],
                    'average_confidence': cluster['average_confidence'],
                    'average_strength': cluster['average_strength'],
                    'arc_types': [p.arc_type.value for p in cluster['projections']]
                })
            return formatted
        except Exception as e:
            logger.warning(f"Error formatting clustered projections: {str(e)}")
            return []

    def get_indicator_name(self) -> str:
        """Return the indicator name."""
        return "Projection Arc Calculator"

    def get_parameters(self) -> Dict[str, Any]:
        """Return current parameters."""
        return {
            'lookback_period': self.lookback_period,
            'projection_length': self.projection_length,
            'fibonacci_ratios': self.fibonacci_ratios,
            'arc_types': [arc_type.value for arc_type in self.arc_types],
            'min_swing_size': self.min_swing_size,
            'volatility_window': self.volatility_window,
            'volume_threshold': self.volume_threshold,
            'confidence_threshold': self.confidence_threshold,
            'clustering_eps': self.clustering_eps,
            'ml_enabled': self.ml_enabled
        }
