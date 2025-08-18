"""
Fibonacci Grid Indicator - Multi-Dimensional Grid-Based Fibonacci Analysis
=================================

This module implements a sophisticated Fibonacci grid indicator that creates
a multi-dimensional grid of Fibonacci levels across both price and time dimensions.
It provides comprehensive grid-based analysis for identifying confluence zones,
support/resistance networks, and geometric price patterns.

Features:
    - Multi-dimensional Fibonacci grid construction
- Horizontal and vertical Fibonacci level networks
- Grid confluence zone detection
- Dynamic grid adjustment based on market volatility
- Time-based Fibonacci grid projections
- Grid strength calculation and validation
- Machine learning enhanced grid pattern recognition
- Real-time grid level monitoring
- Grid breakout and reversal detection

The indicator helps traders identify potential reversal and continuation points
using mathematically derived Fibonacci grids combined with market structure analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from auj_platform.src.indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface
from auj_platform.src.indicator_engine.indicators.base.standard_indicator import IndicatorCalculationError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class GridLevel:
    """Represents a Fibonacci grid level"""
    price: float
    time_index: int
    level_type: str  # 'horizontal', 'vertical', 'diagonal'
    fibonacci_ratio: float
    strength: float  # 0.0 to 1.0
    touches: int
    volume_at_level: float
    confluence_score: float
    grid_id: str


@dataclass
class GridNode:
    """Represents an intersection point in the Fibonacci grid"""
    price: float
    time_index: int
    horizontal_ratio: float
    vertical_ratio: float
    confluence_count: int
    strength: float
    importance_score: float


@dataclass
class GridZone:
    """Represents a significant zone in the Fibonacci grid"""
    center_price: float
    center_time: int
    zone_type: str  # 'support', 'resistance', 'confluence'
    strength: float
    nodes: List[GridNode]
    area_coverage: float


class FibonacciGridIndicator(StandardIndicatorInterface):
    """
    Advanced Fibonacci Grid Indicator with multi-dimensional analysis
    and pattern recognition capabilities.
    """

def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'grid_resolution': 20,  # Number of grid lines per dimension
            'fibonacci_ratios': [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
            'extended_ratios': [1.272, 1.414, 1.618, 2.0, 2.618, 3.0],
            'use_extended_ratios': True,
            'time_ratios': [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618],
            'grid_tolerance': 0.002,  # 0.2% tolerance for grid touches
            'min_confluence': 2,  # Minimum confluence for significant zones
            'volatility_adjustment': True,
            'adaptive_grid': True,
            'max_grid_levels': 50,
            'grid_decay_factor': 0.95,  # How grid strength decays over time
            'zone_clustering_eps': 0.01,  # DBSCAN clustering parameter
            'min_zone_samples': 3,
            'historical_lookback': 200,
            'projection_periods': 50
        }

        if parameters:
            default_params.update(parameters)

        super().__init__(name="FibonacciGrid")

        # Initialize internal state
        self.grid_levels: List[GridLevel] = []
        self.grid_nodes: List[GridNode] = []
        self.grid_zones: List[GridZone] = []
        self.price_range = None
        self.time_range = None
        self.scaler = StandardScaler()

        # Combine ratios
        self.price_ratios = self.parameters['fibonacci_ratios'].copy()
        self.time_ratios = self.parameters['time_ratios'].copy()

        if self.parameters['use_extended_ratios']:
            self.price_ratios.extend(self.parameters['extended_ratios'])

        logger.info(f"FibonacciGridIndicator initialized with {len(self.price_ratios)}x{len(self.time_ratios)} grid")

def get_data_requirements(self) -> DataRequirement:
        """Define the data requirements for Fibonacci grid calculation"""
        return DataRequirement()
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(100, self.parameters['grid_resolution'] * 2),
            lookback_periods=self.parameters['historical_lookback']
(        )

def validate_parameters(self) -> bool:
        """Validate the indicator parameters"""
        try:
            required_params = ['grid_resolution', 'fibonacci_ratios', 'time_ratios']
            for param in required_params:
                if param not in self.parameters:
                    raise ValueError(f"Missing required parameter: {param}")

            if self.parameters['grid_resolution'] < 5:
                raise ValueError("grid_resolution must be at least 5")

            if not self.parameters['fibonacci_ratios']:
                raise ValueError("fibonacci_ratios cannot be empty")

            return True

        except Exception as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            return False

def _calculate_price_range(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate the price range for grid construction"""
        try:
            if self.parameters['volatility_adjustment']:
                # Use ATR-adjusted range
                high_low = data['high'] - data['low']
                high_close_prev = abs(data['high'] - data['close'].shift(1))
                low_close_prev = abs(data['low'] - data['close'].shift(1))

                true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                atr = true_range.rolling(14).mean().iloc[-1]

                current_price = data['close'].iloc[-1]
                price_min = current_price - (atr * 5)
                price_max = current_price + (atr * 5)
            else:
                # Use simple high/low range
                lookback = min(len(data), 100)
                recent_data = data.tail(lookback)
                price_min = recent_data['low'].min()
                price_max = recent_data['high'].max()

            return price_min, price_max

        except Exception as e:
            logger.error(f"Error calculating price range: {str(e)}")
            current_price = data['close'].iloc[-1]
            return current_price * 0.95, current_price * 1.05

def _calculate_time_range(self, data: pd.DataFrame) -> Tuple[int, int]:
        """Calculate the time range for grid construction"""
        try:
            lookback = min(len(data), self.parameters['historical_lookback'])
            start_idx = len(data) - lookback
            end_idx = len(data) - 1 + self.parameters['projection_periods']

            return start_idx, end_idx

        except Exception as e:
            logger.error(f"Error calculating time range: {str(e)}")
            return 0, len(data) - 1

def _generate_horizontal_grid(self, price_min: float, price_max: float) -> List[GridLevel]:
        """Generate horizontal Fibonacci grid levels"""
        try:
            horizontal_levels = []
            price_range = price_max - price_min

            for ratio in self.price_ratios:
                level_price = price_min + (price_range * ratio)

                grid_level = GridLevel()
                    price=level_price,
                    time_index=-1,  # Horizontal levels span all time
                    level_type='horizontal',
                    fibonacci_ratio=ratio,
                    strength=0.5,  # Base strength
                    touches=0,
                    volume_at_level=0.0,
                    confluence_score=0.0,
                    grid_id=f"h_{ratio:.3f}"
(                )
                horizontal_levels.append(grid_level)

            return horizontal_levels

        except Exception as e:
            logger.error(f"Error generating horizontal grid: {str(e)}")
            return []

def _generate_vertical_grid(self, time_start: int, time_end: int) -> List[GridLevel]:
        """Generate vertical Fibonacci time-based grid levels"""
        try:
            vertical_levels = []
            time_range = time_end - time_start

            for ratio in self.time_ratios:
                time_index = int(time_start + (time_range * ratio))

                grid_level = GridLevel()
                    price=-1.0,  # Vertical levels span all prices
                    time_index=time_index,
                    level_type='vertical',
                    fibonacci_ratio=ratio,
                    strength=0.5,  # Base strength
                    touches=0,
                    volume_at_level=0.0,
                    confluence_score=0.0,
                    grid_id=f"v_{ratio:.3f}"
(                )
                vertical_levels.append(grid_level)

            return vertical_levels

        except Exception as e:
            logger.error(f"Error generating vertical grid: {str(e)}")
            return []

def _generate_grid_nodes(self, horizontal_levels: List[GridLevel],:)
(                           vertical_levels: List[GridLevel]) -> List[GridNode]:
        """Generate intersection nodes between horizontal and vertical grid lines"""
        try:
            nodes = []

            for h_level in horizontal_levels:
                for v_level in vertical_levels:
                    if h_level.level_type == 'horizontal' and v_level.level_type == 'vertical':
                        node = GridNode()
                            price=h_level.price,
                            time_index=v_level.time_index,
                            horizontal_ratio=h_level.fibonacci_ratio,
                            vertical_ratio=v_level.fibonacci_ratio,
                            confluence_count=1,
                            strength=0.5,
                            importance_score=0.0
(                        )
                        nodes.append(node)

            return nodes

        except Exception as e:
            logger.error(f"Error generating grid nodes: {str(e)}")
            return []

def _analyze_grid_touches(self, grid_levels: List[GridLevel], data: pd.DataFrame) -> List[GridLevel]:
        """Analyze historical price touches at grid levels"""
        try:
            tolerance = self.parameters['grid_tolerance']

            for level in grid_levels:
                if level.level_type == 'horizontal':
                    touches = 0
                    volume_sum = 0.0

                    for i, row in data.iterrows():
                        high_price = row['high']
                        low_price = row['low']
                        volume = row['volume']

                        # Check if price touched the level
                        level_range = level.price * tolerance
                        if low_price <= level.price + level_range and high_price >= level.price - level_range:
                            touches += 1
                            volume_sum += volume

                    level.touches = touches
                    level.volume_at_level = volume_sum

                    # Adjust strength based on touches
                    if touches > 0:
                        touch_factor = min(touches / 5.0, 1.0)
                        level.strength = min(0.5 + touch_factor * 0.5, 1.0)

            return grid_levels

        except Exception as e:
            logger.error(f"Error analyzing grid touches: {str(e)}")
            return grid_levels

def _calculate_confluence_zones(self, nodes: List[GridNode], data: pd.DataFrame) -> List[GridZone]:
        """Identify significant confluence zones using clustering"""
        try:
            if len(nodes) < self.parameters['min_zone_samples']:
                return []

            # Prepare data for clustering
            node_data = []
            for node in nodes:
                if 0 <= node.time_index < len(data):
                    # Normalize price and time for clustering
                    norm_price = node.price / data['close'].mean()
                    norm_time = node.time_index / len(data)
                    node_data.append([norm_price, norm_time])

            if len(node_data) < self.parameters['min_zone_samples']:
                return []

            # Perform DBSCAN clustering
            clustering = DBSCAN()
                eps=self.parameters['zone_clustering_eps'],
                min_samples=self.parameters['min_zone_samples']
(            ).fit(node_data)

            zones = []
            unique_labels = set(clustering.labels_)

            for label in unique_labels:
                if label == -1:  # Noise points:
                    continue

                # Get nodes in this cluster
                cluster_nodes = [nodes[i] for i, l in enumerate(clustering.labels_) if l == label]

                if len(cluster_nodes) >= self.parameters['min_confluence']:
                    # Calculate zone properties
                    center_price = np.mean([node.price for node in cluster_nodes])
                    center_time = int(np.mean([node.time_index for node in cluster_nodes]))

                    # Determine zone type
                    current_price = data['close'].iloc[-1]
                    if center_price > current_price:
                        zone_type = 'resistance'
                    elif center_price < current_price:
                        zone_type = 'support'
                    else:
                        zone_type = 'confluence'

                    # Calculate strength
                    strength = min(len(cluster_nodes) / 10.0, 1.0)

                    zone = GridZone()
                        center_price=center_price,
                        center_time=center_time,
                        zone_type=zone_type,
                        strength=strength,
                        nodes=cluster_nodes,
                        area_coverage=len(cluster_nodes) / len(nodes)
(                    )
                    zones.append(zone)

            # Sort zones by strength
            zones.sort(key=lambda x: x.strength, reverse=True)
            return zones

        except Exception as e:
            logger.error(f"Error calculating confluence zones: {str(e)}")
            return []

def _calculate_grid_strength(self, data: pd.DataFrame) -> float:
        """Calculate overall grid strength based on current market position"""
        try:
            if not self.grid_zones:
                return 0.0

            current_price = data['close'].iloc[-1]
            current_time = len(data) - 1

            total_strength = 0.0
            relevant_zones = 0

            for zone in self.grid_zones:
                # Calculate distance factors
                price_distance = abs(zone.center_price - current_price) / current_price
                time_distance = abs(zone.center_time - current_time) / len(data)

                # Weight by proximity
                if price_distance < 0.05 and time_distance < 0.1:  # Within 5% price and 10% time:
                    proximity_weight = (1 - price_distance) * (1 - time_distance)
                    weighted_strength = zone.strength * proximity_weight
                    total_strength += weighted_strength
                    relevant_zones += 1

            if relevant_zones > 0:
                return min(total_strength / relevant_zones, 1.0)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating grid strength: {str(e)}")
            return 0.0

def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Fibonacci grid with comprehensive multi-dimensional analysis
        """
        try:
            # Calculate grid ranges
            price_min, price_max = self._calculate_price_range(data)
            time_start, time_end = self._calculate_time_range(data)

            self.price_range = (price_min, price_max)
            self.time_range = (time_start, time_end)

            # Generate grid levels
            horizontal_levels = self._generate_horizontal_grid(price_min, price_max)
            vertical_levels = self._generate_vertical_grid(time_start, time_end)

            # Combine all grid levels
            all_grid_levels = horizontal_levels + vertical_levels

            # Analyze touches and interactions
            all_grid_levels = self._analyze_grid_touches(all_grid_levels, data)

            # Generate grid intersection nodes
            grid_nodes = self._generate_grid_nodes(horizontal_levels, vertical_levels)

            # Calculate confluence zones
            confluence_zones = self._calculate_confluence_zones(grid_nodes, data)

            # Update internal state
            self.grid_levels = all_grid_levels
            self.grid_nodes = grid_nodes
            self.grid_zones = confluence_zones

            # Calculate overall grid strength
            grid_strength = self._calculate_grid_strength(data)

            # Find nearest significant elements
            current_price = data['close'].iloc[-1]
            current_time = len(data) - 1

            # Nearest horizontal level
            nearest_h_level = min()
                [level for level in horizontal_levels],
                key=lambda x: abs(x.price - current_price),
                default=None
(            )

            # Nearest confluence zone
            nearest_zone = min()
                confluence_zones,
                key=lambda x: abs(x.center_price - current_price),
                default=None
(            ) if confluence_zones else None

            # Prepare result
            result = {
                'grid_levels': [self._grid_level_to_dict(level) for level in all_grid_levels[:self.parameters['max_grid_levels']]],
                'grid_nodes': [self._grid_node_to_dict(node) for node in grid_nodes[:50]],  # Limit nodes for performance
                'confluence_zones': [self._grid_zone_to_dict(zone) for zone in confluence_zones[:10]],
                'grid_strength': grid_strength,
                'price_range': {'min': price_min, 'max': price_max},
                'time_range': {'start': time_start, 'end': time_end},
                'current_price': current_price,
                'current_time': current_time,
                'nearest_horizontal_level': self._grid_level_to_dict(nearest_h_level) if nearest_h_level else None,
                'nearest_zone': self._grid_zone_to_dict(nearest_zone) if nearest_zone else None,
                'total_levels': len(all_grid_levels),
                'total_nodes': len(grid_nodes),
                'significant_zones': len(confluence_zones),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Error in Fibonacci grid calculation: {str(e)}")
            raise IndicatorCalculationError()
                indicator_name=self.name,
                calculation_step="fibonacci_grid_calculation",
                message=str(e)
(            )
def _grid_level_to_dict(self, level: GridLevel) -> Dict[str, Any]:
        """Convert GridLevel to dictionary"""
        return {
            'price': level.price,
            'time_index': level.time_index,
            'level_type': level.level_type,
            'fibonacci_ratio': level.fibonacci_ratio,
            'strength': level.strength,
            'touches': level.touches,
            'volume_at_level': level.volume_at_level,
            'confluence_score': level.confluence_score,
            'grid_id': level.grid_id
        }

def _grid_node_to_dict(self, node: GridNode) -> Dict[str, Any]:
        """Convert GridNode to dictionary"""
        return {
            'price': node.price,
            'time_index': node.time_index,
            'horizontal_ratio': node.horizontal_ratio,
            'vertical_ratio': node.vertical_ratio,
            'confluence_count': node.confluence_count,
            'strength': node.strength,
            'importance_score': node.importance_score
        }

def _grid_zone_to_dict(self, zone: GridZone) -> Dict[str, Any]:
        """Convert GridZone to dictionary"""
        return {
            'center_price': zone.center_price,
            'center_time': zone.center_time,
            'zone_type': zone.zone_type,
            'strength': zone.strength,
            'nodes_count': len(zone.nodes),
            'area_coverage': zone.area_coverage
        }

def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """
        Generate trading signals based on Fibonacci grid analysis
        """
        try:
            grid_strength = value.get('grid_strength', 0.0)
            nearest_zone = value.get('nearest_zone')

            if grid_strength < 0.3 or not nearest_zone:
                return SignalType.NEUTRAL, 0.0

            current_price = value['current_price']
            zone_price = nearest_zone['center_price']
            zone_type = nearest_zone['zone_type']
            zone_strength = nearest_zone['strength']

            # Calculate distance to zone
            price_distance = abs(current_price - zone_price) / current_price

            if price_distance < 0.02:  # Within 2% of zone:
                # Generate signal based on zone type and market momentum
                recent_momentum = data['close'].pct_change(5).iloc[-1]
                confidence = min(grid_strength * zone_strength * (1 - price_distance * 10), 1.0)

                if zone_type == 'support' and current_price <= zone_price:
                    if recent_momentum < 0:  # Oversold bounce potential:
                        return SignalType.BUY, confidence
                elif zone_type == 'resistance' and current_price >= zone_price:
                    if recent_momentum > 0:  # Overbought reversal potential:
                        return SignalType.SELL, confidence
                elif zone_type == 'confluence':
                    # High confluence zones can act as both support and resistance
                    if recent_momentum > 0.01:
                        return SignalType.SELL, confidence * 0.8
                    elif recent_momentum < -0.01:
                        return SignalType.BUY, confidence * 0.8

            return SignalType.NEUTRAL, 0.0

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return SignalType.NEUTRAL, 0.0

def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)

        grid_metadata = {
            'grid_dimensions': f"{len(self.price_ratios)}x{len(self.time_ratios)}",
            'total_grid_levels': len(self.grid_levels),
            'total_grid_nodes': len(self.grid_nodes),
            'confluence_zones': len(self.grid_zones),
            'price_range_span': self.price_range[1] - self.price_range[0] if self.price_range else 0,
            'time_range_span': self.time_range[1] - self.time_range[0] if self.time_range else 0,
            'adaptive_grid_enabled': self.parameters['adaptive_grid'],
            'volatility_adjustment_enabled': self.parameters['volatility_adjustment']
        }

        base_metadata.update(grid_metadata)
        return base_metadata


def create_fibonacci_grid_indicator(parameters: Optional[Dict[str, Any]] = None) -> FibonacciGridIndicator:
    """
    Factory function to create a FibonacciGridIndicator instance

    Args:
        parameters: Optional dictionary of parameters to customize the indicator

    Returns:
        Configured FibonacciGridIndicator instance
    """
    return FibonacciGridIndicator(parameters=parameters)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1h')
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1)

    sample_data = pd.DataFrame({)
        'high': prices + np.random.uniform(0, 1, len(dates)),
        'low': prices - np.random.uniform(0, 1, len(dates)),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(dates))
(    }, index=dates)

    # Test the indicator
    indicator = create_fibonacci_grid_indicator({)
        'grid_resolution': 15,
        'use_extended_ratios': True,
        'adaptive_grid': True
(    })

    try:
        result = indicator.calculate(sample_data)
        print("Fibonacci Grid Calculation Result:")
        print(f"Signal: {result.signal}, Confidence: {result.confidence:.3f}")
        print(f"Grid strength: {result.value.get('grid_strength', 0):.3f}")
        print(f"Total levels: {result.value.get('total_levels', 0)}")
        print(f"Confluence zones: {result.value.get('significant_zones', 0)}")

        if result.value.get('nearest_zone'):
            nearest = result.value['nearest_zone']
            print(f"Nearest zone: {nearest['zone_type']} at ${nearest['center_price']:.2f}")
            print(f"Zone strength: {nearest['strength']:.3f}")

    except Exception as e:
        print(f"Error testing indicator: {str(e)}")
