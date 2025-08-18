"""
Support Resistance Indicator - Advanced Level Identification and Analysis
========================================================================

This module implements a sophisticated support and resistance indicator that
identifies key price levels, analyzes their strength and reliability, and
provides dynamic level monitoring. It combines multiple detection methods
including pivot points, volume analysis, and machine learning techniques.

Features:
- Multi-method support/resistance level detection
- Dynamic level strength calculation and validation
- Volume-weighted level significance analysis
- Breakout and bounce detection at key levels
- Machine learning enhanced level prediction
- Real-time level adjustment and monitoring
- Level confluence and clustering analysis
- Time-decay modeling for level relevance
- Multi-timeframe level analysis

The indicator helps traders identify high-probability areas where price
is likely to react, providing crucial information for entry, exit,
and risk management decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface, IndicatorResult, DataRequirement, DataType, SignalType
from src.core.exceptions import IndicatorCalculationError

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SupportResistanceLevel:
    """Represents a support or resistance level"""
    price: float
    level_type: str  # 'support', 'resistance', 'pivot'
    strength: float  # 0.0 to 1.0
    touches: int
    volume_at_level: float
    first_touch: datetime
    last_touch: datetime
    time_relevance: float  # Decreases over time
    distance_from_current: float
    breakout_probability: float
    bounce_probability: float
    confidence: float
    level_id: str


@dataclass
class LevelInteraction:
    """Represents an interaction between price and a level"""
    level_id: str
    interaction_type: str  # 'touch', 'bounce', 'breakout', 'false_breakout'
    timestamp: datetime
    price: float
    volume: float
    strength: float
    success_probability: float


@dataclass
class LevelCluster:
    """Represents a cluster of nearby support/resistance levels"""
    cluster_id: int
    center_price: float
    price_range: Tuple[float, float]
    levels: List[SupportResistanceLevel]
    combined_strength: float
    cluster_type: str  # 'support', 'resistance', 'mixed'
    significance: float


class SupportResistanceIndicator(StandardIndicatorInterface):
    """
    Advanced Support Resistance Indicator with multi-method detection
    and machine learning enhanced analysis capabilities.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'pivot_window': 10,  # Window for pivot point detection
            'volume_threshold': 1.5,  # Volume spike threshold for level validation
            'touch_tolerance': 0.002,  # 0.2% tolerance for level touches
            'min_level_strength': 0.3,  # Minimum strength for valid levels
            'max_levels': 20,  # Maximum number of levels to track
            'time_decay_factor': 0.98,  # Daily decay factor for level relevance
            'clustering_distance': 0.01,  # 1% distance for level clustering
            'breakout_confirmation': 0.005,  # 0.5% move required for breakout confirmation
            'volume_weight': 0.4,  # Weight of volume in strength calculation
            'recency_weight': 0.3,  # Weight of recency in strength calculation
            'touch_weight': 0.3,  # Weight of touch count in strength calculation
            'ml_lookback': 50,  # Lookback for ML features
            'prediction_horizon': 5,  # Days to predict level interactions
            'min_touches': 2,  # Minimum touches for valid level
            'level_expiry_days': 30,  # Days after which levels expire
            'significance_threshold': 0.6,  # Minimum significance for level reporting
            'fractal_periods': [5, 10, 20],  # Periods for fractal analysis
            'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786]  # Fibonacci retracement levels
        }

        if parameters:
            default_params.update(parameters)

        super().__init__(name="SupportResistance")

        # Initialize internal state
        self.levels: List[SupportResistanceLevel] = []
        self.level_interactions: List[LevelInteraction] = []
        self.level_clusters: List[LevelCluster] = []
        self.ml_model = None
        self.scaler = StandardScaler()
        self.level_counter = 0

        logger.info(f"SupportResistanceIndicator initialized")

    def get_data_requirements(self) -> DataRequirement:
        """Define the data requirements for support/resistance calculation"""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['high', 'low', 'close', 'volume'],
            min_periods=max(50, self.parameters['pivot_window'] * 5),
            lookback_periods=200
        )

    def validate_parameters(self) -> bool:
        """Validate the indicator parameters"""
        try:
            required_params = ['pivot_window', 'touch_tolerance', 'min_level_strength']
            for param in required_params:
                if param not in self.parameters:
                    raise ValueError(f"Missing required parameter: {param}")

            if self.parameters['pivot_window'] < 3:
                raise ValueError("pivot_window must be at least 3")

            if self.parameters['touch_tolerance'] <= 0:
                raise ValueError("touch_tolerance must be positive")

            return True

        except Exception as e:
            logger.error(f"Parameter validation failed: {str(e)}")
            return False

    def _detect_pivot_levels(self, data: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Detect support and resistance levels using pivot point analysis"""
        try:
            levels = []
            window = self.parameters['pivot_window']

            # Detect pivot highs (resistance levels)
            high_peaks, high_properties = find_peaks(
                data['high'].values,
                distance=window,
                prominence=data['high'].std() * 0.5
            )

            # Detect pivot lows (support levels)
            low_peaks, low_properties = find_peaks(
                -data['low'].values,
                distance=window,
                prominence=data['low'].std() * 0.5
            )

            current_time = data.index[-1]
            current_price = data['close'].iloc[-1]

            # Process pivot highs (resistance)
            for i, peak_idx in enumerate(high_peaks):
                if peak_idx >= window and peak_idx < len(data) - 1:
                    price = data['high'].iloc[peak_idx]
                    timestamp = data.index[peak_idx]
                    volume = data['volume'].iloc[peak_idx]

                    # Calculate basic strength based on prominence
                    strength = min(high_properties['prominences'][i] / data['high'].std(), 1.0)

                    level = SupportResistanceLevel(
                        price=price,
                        level_type='resistance',
                        strength=strength,
                        touches=1,
                        volume_at_level=volume,
                        first_touch=timestamp,
                        last_touch=timestamp,
                        time_relevance=1.0,
                        distance_from_current=abs(price - current_price) / current_price,
                        breakout_probability=0.0,
                        bounce_probability=0.0,
                        confidence=strength,
                        level_id=f"R_{self.level_counter}"
                    )
                    levels.append(level)
                    self.level_counter += 1

            # Process pivot lows (support)
            for i, peak_idx in enumerate(low_peaks):
                if peak_idx >= window and peak_idx < len(data) - 1:
                    price = data['low'].iloc[peak_idx]
                    timestamp = data.index[peak_idx]
                    volume = data['volume'].iloc[peak_idx]

                    # Calculate basic strength based on prominence
                    strength = min(low_properties['prominences'][i] / data['low'].std(), 1.0)

                    level = SupportResistanceLevel(
                        price=price,
                        level_type='support',
                        strength=strength,
                        touches=1,
                        volume_at_level=volume,
                        first_touch=timestamp,
                        last_touch=timestamp,
                        time_relevance=1.0,
                        distance_from_current=abs(price - current_price) / current_price,
                        breakout_probability=0.0,
                        bounce_probability=0.0,
                        confidence=strength,
                        level_id=f"S_{self.level_counter}"
                    )
                    levels.append(level)
                    self.level_counter += 1

            return levels

        except Exception as e:
            logger.error(f"Error detecting pivot levels: {str(e)}")
            return []

    def _detect_volume_levels(self, data: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Detect levels based on volume spikes and price action"""
        try:
            levels = []
            volume_threshold = self.parameters['volume_threshold']

            # Calculate volume moving average
            volume_ma = data['volume'].rolling(20).mean()

            # Find volume spikes
            volume_spikes = data['volume'] > (volume_ma * volume_threshold)
            spike_indices = data[volume_spikes].index

            current_price = data['close'].iloc[-1]

            for spike_time in spike_indices:
                if spike_time in data.index:
                    spike_idx = data.index.get_loc(spike_time)

                    if spike_idx > 0 and spike_idx < len(data) - 1:
                        # Check price action around volume spike
                        high_price = data['high'].iloc[spike_idx]
                        low_price = data['low'].iloc[spike_idx]
                        volume = data['volume'].iloc[spike_idx]

                        # Create resistance level at high
                        resistance_strength = min(volume / volume_ma.iloc[spike_idx], 3.0) / 3.0

                        resistance_level = SupportResistanceLevel(
                            price=high_price,
                            level_type='resistance',
                            strength=resistance_strength,
                            touches=1,
                            volume_at_level=volume,
                            first_touch=spike_time,
                            last_touch=spike_time,
                            time_relevance=1.0,
                            distance_from_current=abs(high_price - current_price) / current_price,
                            breakout_probability=0.0,
                            bounce_probability=0.0,
                            confidence=resistance_strength,
                            level_id=f"VR_{self.level_counter}"
                        )
                        levels.append(resistance_level)
                        self.level_counter += 1

                        # Create support level at low
                        support_strength = min(volume / volume_ma.iloc[spike_idx], 3.0) / 3.0

                        support_level = SupportResistanceLevel(
                            price=low_price,
                            level_type='support',
                            strength=support_strength,
                            touches=1,
                            volume_at_level=volume,
                            first_touch=spike_time,
                            last_touch=spike_time,
                            time_relevance=1.0,
                            distance_from_current=abs(low_price - current_price) / current_price,
                            breakout_probability=0.0,
                            bounce_probability=0.0,
                            confidence=support_strength,
                            level_id=f"VS_{self.level_counter}"
                        )
                        levels.append(support_level)
                        self.level_counter += 1

            return levels

        except Exception as e:
            logger.error(f"Error detecting volume levels: {str(e)}")
            return []

    def _detect_fibonacci_levels(self, data: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Detect Fibonacci retracement levels as support/resistance"""
        try:
            levels = []

            if len(data) < 50:
                return levels

            # Find significant swing high and low
            lookback = min(50, len(data))
            recent_data = data.tail(lookback)

            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            price_range = swing_high - swing_low

            if price_range <= 0:
                return levels

            current_price = data['close'].iloc[-1]
            current_time = data.index[-1]

            # Calculate Fibonacci levels
            for ratio in self.parameters['fibonacci_levels']:
                fib_price = swing_high - (price_range * ratio)

                # Determine if it's support or resistance based on current price
                if fib_price > current_price:
                    level_type = 'resistance'
                else:
                    level_type = 'support'

                # Strength based on how close to standard Fibonacci ratios
                standard_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
                strength = 0.7 if ratio in standard_ratios else 0.5

                level = SupportResistanceLevel(
                    price=fib_price,
                    level_type=level_type,
                    strength=strength,
                    touches=0,  # Will be calculated later
                    volume_at_level=0.0,
                    first_touch=current_time,
                    last_touch=current_time,
                    time_relevance=1.0,
                    distance_from_current=abs(fib_price - current_price) / current_price,
                    breakout_probability=0.0,
                    bounce_probability=0.0,
                    confidence=strength,
                    level_id=f"F_{ratio}_{self.level_counter}"
                )
                levels.append(level)
                self.level_counter += 1

            return levels

        except Exception as e:
            logger.error(f"Error detecting Fibonacci levels: {str(e)}")
            return []

    def _analyze_level_touches(self, levels: List[SupportResistanceLevel],
                             data: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Analyze historical touches for each level and update strength"""
        try:
            tolerance = self.parameters['touch_tolerance']

            for level in levels:
                touches = 0
                total_volume = 0.0
                touch_times = []

                for i, row in data.iterrows():
                    high_price = row['high']
                    low_price = row['low']
                    volume = row['volume']

                    # Check if price touched the level
                    level_range = level.price * tolerance

                    if (low_price <= level.price + level_range and
                        high_price >= level.price - level_range):
                        touches += 1
                        total_volume += volume
                        touch_times.append(i)

                # Update level properties
                level.touches = touches
                level.volume_at_level = total_volume

                if touch_times:
                    level.first_touch = touch_times[0]
                    level.last_touch = touch_times[-1]

                # Recalculate strength based on touches, volume, and recency
                if touches >= self.parameters['min_touches']:
                    touch_factor = min(touches / 5.0, 1.0) * self.parameters['touch_weight']

                    # Volume factor
                    avg_volume = data['volume'].mean()
                    volume_factor = min(total_volume / (avg_volume * touches), 2.0) * self.parameters['volume_weight']

                    # Recency factor
                    days_since_last_touch = (data.index[-1] - level.last_touch).days if isinstance(level.last_touch, datetime) else 0
                    recency_factor = (self.parameters['time_decay_factor'] ** days_since_last_touch) * self.parameters['recency_weight']

                    # Combined strength
                    level.strength = min(touch_factor + volume_factor + recency_factor, 1.0)
                    level.confidence = level.strength
                else:
                    level.strength = max(level.strength * 0.5, 0.1)  # Reduce strength for untested levels
                    level.confidence = level.strength

            return levels

        except Exception as e:
            logger.error(f"Error analyzing level touches: {str(e)}")
            return levels

    def _cluster_levels(self, levels: List[SupportResistanceLevel]) -> List[LevelCluster]:
        """Cluster nearby levels to identify zones of support/resistance"""
        try:
            if len(levels) < 2:
                return []

            # Prepare data for clustering
            prices = np.array([level.price for level in levels]).reshape(-1, 1)

            # Use DBSCAN clustering
            clustering_eps = np.mean(prices) * self.parameters['clustering_distance']
            clustering = DBSCAN(eps=clustering_eps, min_samples=2).fit(prices)

            clusters = []
            unique_labels = set(clustering.labels_)

            for label in unique_labels:
                if label == -1:  # Noise points
                    continue

                # Get levels in this cluster
                cluster_levels = [levels[i] for i, l in enumerate(clustering.labels_) if l == label]

                if len(cluster_levels) >= 2:
                    # Calculate cluster properties
                    cluster_prices = [level.price for level in cluster_levels]
                    center_price = np.mean(cluster_prices)
                    price_range = (min(cluster_prices), max(cluster_prices))

                    # Combined strength
                    combined_strength = np.mean([level.strength for level in cluster_levels])

                    # Determine cluster type
                    support_count = sum(1 for level in cluster_levels if level.level_type == 'support')
                    resistance_count = sum(1 for level in cluster_levels if level.level_type == 'resistance')

                    if support_count > resistance_count:
                        cluster_type = 'support'
                    elif resistance_count > support_count:
                        cluster_type = 'resistance'
                    else:
                        cluster_type = 'mixed'

                    # Calculate significance
                    significance = min(len(cluster_levels) * combined_strength / 3.0, 1.0)

                    cluster = LevelCluster(
                        cluster_id=len(clusters),
                        center_price=center_price,
                        price_range=price_range,
                        levels=cluster_levels,
                        combined_strength=combined_strength,
                        cluster_type=cluster_type,
                        significance=significance
                    )
                    clusters.append(cluster)

            # Sort clusters by significance
            clusters.sort(key=lambda x: x.significance, reverse=True)
            return clusters

        except Exception as e:
            logger.error(f"Error clustering levels: {str(e)}")
            return []
    def _predict_level_interactions(self, levels: List[SupportResistanceLevel],
                                  data: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Predict breakout and bounce probabilities using ML"""
        try:
            current_price = data['close'].iloc[-1]

            for level in levels:
                # Simple probability calculation based on level characteristics
                distance_factor = 1.0 - min(level.distance_from_current / 0.1, 1.0)  # Closer = higher probability
                strength_factor = level.strength
                touch_factor = min(level.touches / 5.0, 1.0)

                # Base probabilities
                base_probability = (distance_factor + strength_factor + touch_factor) / 3.0

                if level.level_type == 'resistance' and current_price < level.price:
                    # Below resistance - calculate bounce vs breakout probability
                    level.bounce_probability = base_probability * 0.7
                    level.breakout_probability = (1.0 - base_probability) * 0.6
                elif level.level_type == 'support' and current_price > level.price:
                    # Above support - calculate bounce vs breakdown probability
                    level.bounce_probability = base_probability * 0.7
                    level.breakout_probability = (1.0 - base_probability) * 0.6
                else:
                    # Price already past level
                    level.bounce_probability = 0.2
                    level.breakout_probability = 0.8

            return levels

        except Exception as e:
            logger.error(f"Error predicting level interactions: {str(e)}")
            return levels

    def _update_time_relevance(self, levels: List[SupportResistanceLevel],
                             current_time: datetime) -> List[SupportResistanceLevel]:
        """Update time relevance of levels based on age"""
        try:
            decay_factor = self.parameters['time_decay_factor']
            expiry_days = self.parameters['level_expiry_days']

            for level in levels:
                # Calculate days since last touch
                if isinstance(level.last_touch, datetime):
                    days_old = (current_time - level.last_touch).days
                else:
                    days_old = 0

                # Apply time decay
                if days_old <= expiry_days:
                    level.time_relevance = decay_factor ** days_old
                else:
                    level.time_relevance = 0.0  # Expired

                # Update overall confidence with time relevance
                level.confidence = level.strength * level.time_relevance

            return levels

        except Exception as e:
            logger.error(f"Error updating time relevance: {str(e)}")
            return levels

    def _filter_significant_levels(self, levels: List[SupportResistanceLevel]) -> List[SupportResistanceLevel]:
        """Filter levels based on significance and relevance"""
        try:
            significance_threshold = self.parameters['significance_threshold']
            max_levels = self.parameters['max_levels']

            # Filter by significance
            significant_levels = [
                level for level in levels
                if (level.confidence >= significance_threshold and
                    level.time_relevance > 0.1 and
                    level.touches >= self.parameters['min_touches'])
            ]

            # Sort by confidence and take top levels
            significant_levels.sort(key=lambda x: x.confidence, reverse=True)

            return significant_levels[:max_levels]

        except Exception as e:
            logger.error(f"Error filtering significant levels: {str(e)}")
            return levels

    def _detect_level_interactions(self, levels: List[SupportResistanceLevel],
                                 data: pd.DataFrame) -> List[LevelInteraction]:
        """Detect recent interactions between price and levels"""
        try:
            interactions = []
            tolerance = self.parameters['touch_tolerance']
            breakout_confirmation = self.parameters['breakout_confirmation']

            # Look at recent price action (last 5 periods)
            recent_data = data.tail(5)

            for level in levels:
                for i, row in recent_data.iterrows():
                    high_price = row['high']
                    low_price = row['low']
                    close_price = row['close']
                    volume = row['volume']

                    level_range = level.price * tolerance

                    # Check for touch
                    if (low_price <= level.price + level_range and
                        high_price >= level.price - level_range):

                        # Determine interaction type
                        if level.level_type == 'resistance':
                            if close_price > level.price + (level.price * breakout_confirmation):
                                interaction_type = 'breakout'
                                success_prob = 0.7
                            elif high_price > level.price and close_price < level.price:
                                interaction_type = 'bounce'
                                success_prob = 0.8
                            else:
                                interaction_type = 'touch'
                                success_prob = 0.5
                        else:  # support
                            if close_price < level.price - (level.price * breakout_confirmation):
                                interaction_type = 'breakout'
                                success_prob = 0.7
                            elif low_price < level.price and close_price > level.price:
                                interaction_type = 'bounce'
                                success_prob = 0.8
                            else:
                                interaction_type = 'touch'
                                success_prob = 0.5

                        interaction = LevelInteraction(
                            level_id=level.level_id,
                            interaction_type=interaction_type,
                            timestamp=i,
                            price=close_price,
                            volume=volume,
                            strength=level.strength,
                            success_probability=success_prob
                        )
                        interactions.append(interaction)

            return interactions

        except Exception as e:
            logger.error(f"Error detecting level interactions: {str(e)}")
            return []

    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive support and resistance analysis
        """
        try:
            current_time = data.index[-1] if not data.empty else datetime.utcnow()
            current_price = data['close'].iloc[-1] if not data.empty else 0.0

            # Detect levels using multiple methods
            pivot_levels = self._detect_pivot_levels(data)
            volume_levels = self._detect_volume_levels(data)
            fibonacci_levels = self._detect_fibonacci_levels(data)

            # Combine all detected levels
            all_levels = pivot_levels + volume_levels + fibonacci_levels

            if not all_levels:
                return {
                    'support_levels': [],
                    'resistance_levels': [],
                    'level_clusters': [],
                    'level_interactions': [],
                    'nearest_support': None,
                    'nearest_resistance': None,
                    'level_strength_score': 0.0,
                    'breakout_risk': 0.0,
                    'total_levels': 0
                }

            # Analyze touches and update strength
            all_levels = self._analyze_level_touches(all_levels, data)

            # Update time relevance
            all_levels = self._update_time_relevance(all_levels, current_time)

            # Predict level interactions
            all_levels = self._predict_level_interactions(all_levels, data)

            # Filter significant levels
            significant_levels = self._filter_significant_levels(all_levels)

            # Update internal state
            self.levels = significant_levels

            # Cluster levels
            level_clusters = self._cluster_levels(significant_levels)
            self.level_clusters = level_clusters

            # Detect recent interactions
            level_interactions = self._detect_level_interactions(significant_levels, data)
            self.level_interactions = level_interactions

            # Separate support and resistance levels
            support_levels = [level for level in significant_levels if level.level_type == 'support']
            resistance_levels = [level for level in significant_levels if level.level_type == 'resistance']

            # Find nearest levels
            nearest_support = None
            nearest_resistance = None

            if support_levels:
                nearest_support = max(
                    [level for level in support_levels if level.price <= current_price],
                    key=lambda x: x.price,
                    default=None
                )
                if not nearest_support:
                    nearest_support = min(support_levels, key=lambda x: abs(x.price - current_price))

            if resistance_levels:
                nearest_resistance = min(
                    [level for level in resistance_levels if level.price >= current_price],
                    key=lambda x: x.price,
                    default=None
                )
                if not nearest_resistance:
                    nearest_resistance = min(resistance_levels, key=lambda x: abs(x.price - current_price))

            # Calculate overall metrics
            level_strength_score = np.mean([level.strength for level in significant_levels]) if significant_levels else 0.0

            # Calculate breakout risk based on price position relative to levels
            breakout_risk = 0.0
            if nearest_support and nearest_resistance:
                support_distance = abs(current_price - nearest_support.price) / current_price
                resistance_distance = abs(current_price - nearest_resistance.price) / current_price

                # Higher risk when close to levels
                if support_distance < 0.02:  # Within 2% of support
                    breakout_risk += (1 - support_distance / 0.02) * 0.5
                if resistance_distance < 0.02:  # Within 2% of resistance
                    breakout_risk += (1 - resistance_distance / 0.02) * 0.5

            breakout_risk = min(breakout_risk, 1.0)

            # Prepare result
            result = {
                'support_levels': [self._level_to_dict(level) for level in support_levels],
                'resistance_levels': [self._level_to_dict(level) for level in resistance_levels],
                'level_clusters': [self._cluster_to_dict(cluster) for cluster in level_clusters],
                'level_interactions': [self._interaction_to_dict(interaction) for interaction in level_interactions],
                'nearest_support': self._level_to_dict(nearest_support) if nearest_support else None,
                'nearest_resistance': self._level_to_dict(nearest_resistance) if nearest_resistance else None,
                'level_strength_score': level_strength_score,
                'breakout_risk': breakout_risk,
                'current_price': current_price,
                'total_levels': len(significant_levels),
                'total_clusters': len(level_clusters),
                'recent_interactions': len(level_interactions),
                'analysis_timestamp': datetime.utcnow().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Error in support resistance calculation: {str(e)}")
            raise IndicatorCalculationError(
                indicator_name=self.name,
                calculation_step="support_resistance_calculation",
                message=str(e)
            )

    def _level_to_dict(self, level: SupportResistanceLevel) -> Dict[str, Any]:
        """Convert SupportResistanceLevel to dictionary"""
        return {
            'price': level.price,
            'level_type': level.level_type,
            'strength': level.strength,
            'touches': level.touches,
            'volume_at_level': level.volume_at_level,
            'first_touch': level.first_touch.isoformat() if isinstance(level.first_touch, datetime) else str(level.first_touch),
            'last_touch': level.last_touch.isoformat() if isinstance(level.last_touch, datetime) else str(level.last_touch),
            'time_relevance': level.time_relevance,
            'distance_from_current': level.distance_from_current,
            'breakout_probability': level.breakout_probability,
            'bounce_probability': level.bounce_probability,
            'confidence': level.confidence,
            'level_id': level.level_id
        }

    def _cluster_to_dict(self, cluster: LevelCluster) -> Dict[str, Any]:
        """Convert LevelCluster to dictionary"""
        return {
            'cluster_id': cluster.cluster_id,
            'center_price': cluster.center_price,
            'price_range': cluster.price_range,
            'level_count': len(cluster.levels),
            'combined_strength': cluster.combined_strength,
            'cluster_type': cluster.cluster_type,
            'significance': cluster.significance
        }

    def _interaction_to_dict(self, interaction: LevelInteraction) -> Dict[str, Any]:
        """Convert LevelInteraction to dictionary"""
        return {
            'level_id': interaction.level_id,
            'interaction_type': interaction.interaction_type,
            'timestamp': interaction.timestamp.isoformat() if isinstance(interaction.timestamp, datetime) else str(interaction.timestamp),
            'price': interaction.price,
            'volume': interaction.volume,
            'strength': interaction.strength,
            'success_probability': interaction.success_probability
        }

    def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """
        Generate trading signals based on support/resistance analysis
        """
        try:
            breakout_risk = value.get('breakout_risk', 0.0)
            level_strength_score = value.get('level_strength_score', 0.0)
            nearest_support = value.get('nearest_support')
            nearest_resistance = value.get('nearest_resistance')
            recent_interactions = value.get('level_interactions', [])

            if not nearest_support and not nearest_resistance:
                return SignalType.NEUTRAL, 0.0

            current_price = value.get('current_price', 0.0)

            # Check for recent breakout interactions
            breakout_interactions = [i for i in recent_interactions
                                   if i.get('interaction_type') == 'breakout']

            if breakout_interactions:
                latest_breakout = breakout_interactions[-1]
                success_prob = latest_breakout.get('success_probability', 0.0)

                if success_prob > 0.6:
                    # Determine direction based on level type
                    breakout_level_id = latest_breakout.get('level_id', '')
                    if 'R_' in breakout_level_id or 'VR_' in breakout_level_id:  # Resistance breakout
                        return SignalType.BUY, success_prob
                    elif 'S_' in breakout_level_id or 'VS_' in breakout_level_id:  # Support breakdown
                        return SignalType.SELL, success_prob

            # Check for bounce opportunities
            bounce_interactions = [i for i in recent_interactions
                                 if i.get('interaction_type') == 'bounce']

            if bounce_interactions:
                latest_bounce = bounce_interactions[-1]
                success_prob = latest_bounce.get('success_probability', 0.0)

                if success_prob > 0.7:
                    bounce_level_id = latest_bounce.get('level_id', '')
                    if 'S_' in bounce_level_id or 'VS_' in bounce_level_id:  # Support bounce
                        return SignalType.BUY, success_prob * 0.8
                    elif 'R_' in bounce_level_id or 'VR_' in bounce_level_id:  # Resistance bounce
                        return SignalType.SELL, success_prob * 0.8

            # Check proximity to strong levels
            if nearest_support and nearest_support['distance_from_current'] < 0.01:  # Within 1%
                if nearest_support['strength'] > 0.7:
                    return SignalType.BUY, nearest_support['strength'] * 0.6

            if nearest_resistance and nearest_resistance['distance_from_current'] < 0.01:  # Within 1%
                if nearest_resistance['strength'] > 0.7:
                    return SignalType.SELL, nearest_resistance['strength'] * 0.6

            return SignalType.NEUTRAL, 0.0

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return SignalType.NEUTRAL, 0.0

    def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)

        sr_metadata = {
            'total_levels_tracked': len(self.levels),
            'total_clusters': len(self.level_clusters),
            'pivot_window': self.parameters['pivot_window'],
            'touch_tolerance': self.parameters['touch_tolerance'],
            'min_level_strength': self.parameters['min_level_strength'],
            'recent_interactions': len(self.level_interactions),
            'time_decay_enabled': True,
            'fibonacci_levels_enabled': bool(self.parameters['fibonacci_levels'])
        }

        base_metadata.update(sr_metadata)
        return base_metadata


def create_support_resistance_indicator(parameters: Optional[Dict[str, Any]] = None) -> SupportResistanceIndicator:
    """
    Factory function to create a SupportResistanceIndicator instance

    Args:
        parameters: Optional dictionary of parameters to customize the indicator

    Returns:
        Configured SupportResistanceIndicator instance
    """
    return SupportResistanceIndicator(parameters=parameters)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    # Create price data with clear support/resistance levels
    base_price = 100
    trend = np.linspace(0, 20, len(dates))
    noise = np.random.randn(len(dates)) * 2

    # Add some horizontal levels
    resistance_level = 115
    support_level = 105

    prices = base_price + trend + noise

    # Modify prices to respect support/resistance
    for i in range(len(prices)):
        if prices[i] > resistance_level:
            prices[i] = resistance_level + np.random.uniform(-1, 1)
        elif prices[i] < support_level:
            prices[i] = support_level + np.random.uniform(-1, 1)

    sample_data = pd.DataFrame({
        'high': prices + np.random.uniform(0.5, 2.0, len(dates)),
        'low': prices - np.random.uniform(0.5, 2.0, len(dates)),
        'close': prices,
        'volume': np.random.uniform(500000, 2000000, len(dates))
    }, index=dates)

    # Test the indicator
    indicator = create_support_resistance_indicator({
        'pivot_window': 10,
        'min_level_strength': 0.3,
        'touch_tolerance': 0.015
    })

    try:
        result = indicator.calculate(sample_data)
        print("Support Resistance Calculation Result:")
        print(f"Signal: {result.signal}, Confidence: {result.confidence:.3f}")
        print(f"Level strength score: {result.value.get('level_strength_score', 0):.3f}")
        print(f"Breakout risk: {result.value.get('breakout_risk', 0):.3f}")
        print(f"Total levels: {result.value.get('total_levels', 0)}")
        print(f"Total clusters: {result.value.get('total_clusters', 0)}")

        nearest_support = result.value.get('nearest_support')
        if nearest_support:
            print(f"Nearest support: ${nearest_support['price']:.2f} (strength: {nearest_support['strength']:.3f})")

        nearest_resistance = result.value.get('nearest_resistance')
        if nearest_resistance:
            print(f"Nearest resistance: ${nearest_resistance['price']:.2f} (strength: {nearest_resistance['strength']:.3f})")

        print(f"Recent interactions: {result.value.get('recent_interactions', 0)}")

    except Exception as e:
        print(f"Error testing indicator: {str(e)}")
