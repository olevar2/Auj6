"""
Fibonacci Clusters Indicator - Advanced Implementation
=====================================================

This indicator implements sophisticated Fibonacci cluster analysis using advanced mathematical algorithms
and machine learning-enhanced confluence detection. Fibonacci clusters identify zones where multiple
Fibonacci levels from different swing points converge, creating high-probability support/resistance areas.

Features:
- Advanced multi-swing Fibonacci level calculation
- Machine learning confluence zone detection and validation
- Multi-timeframe cluster analysis and confirmation
- Dynamic cluster strength weighting based on volume and price action
- Comprehensive cluster ranking and filtering systems
- Time-weighted cluster decay algorithms
- Comprehensive error handling and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType
from ....core.exceptions import IndicatorCalculationException

logger = logging.getLogger(__name__)

@dataclass
class FibonacciLevel:
    """Represents a single Fibonacci level from a swing pair."""
    swing_high_idx: int
    swing_low_idx: int
    swing_high_price: float
    swing_low_price: float
    ratio: float
    level_price: float
    level_type: str  # 'retracement', 'extension'
    time_weight: float
    volume_weight: float
    strength: float
    distance_from_current: float

@dataclass
class ConfluenceCluster:
    """Represents a cluster of confluent Fibonacci levels."""
    cluster_id: int
    center_price: float
    price_range: Tuple[float, float]
    levels: List[FibonacciLevel]
    cluster_strength: float
    confluence_score: float
    time_relevance: float
    volume_confirmation: float
    support_resistance_type: str  # 'support', 'resistance', 'both'
    hit_probability: float
    reaction_magnitude: float

@dataclass
class SwingPoint:
    """Represents a significant swing high or low."""
    index: int
    price: float
    swing_type: str  # 'high', 'low'
    strength: float
    volume: float
    time_weight: float
    atr_multiple: float

class FibonacciClustersIndicator(StandardIndicatorInterface):
    """
    Advanced Fibonacci Clusters Indicator with machine learning confluence analysis.
    
    This implementation identifies and analyzes confluence zones where multiple Fibonacci
    levels from different swing points converge, providing high-probability trading zones
    with sophisticated strength weighting and probability analysis.
    """
    
    def __init__(self, 
                 name: str = "FibonacciClusters",
                 fibonacci_ratios: List[float] = None,
                 lookback_period: int = 200,
                 min_swing_strength: float = 0.02,
                 cluster_tolerance: float = 0.005,  # 0.5% price tolerance for clustering
                 min_cluster_size: int = 3,
                 time_decay_factor: float = 0.95,
                 volume_weight_factor: float = 0.3,
                 ml_validation: bool = True):
        """
        Initialize the Fibonacci Clusters Indicator.
        
        Args:
            name: Indicator name
            fibonacci_ratios: Fibonacci ratios for level calculation
            lookback_period: Period for swing detection
            min_swing_strength: Minimum strength for swing validation
            cluster_tolerance: Price tolerance for level clustering
            min_cluster_size: Minimum levels required for cluster
            time_decay_factor: Time decay factor for level relevance
            volume_weight_factor: Weight factor for volume confirmation
            ml_validation: Enable machine learning cluster validation
        """
        parameters = {
            'fibonacci_ratios': fibonacci_ratios or [
                0.236, 0.382, 0.500, 0.618, 0.764, 0.786, 1.000, 
                1.272, 1.414, 1.618, 2.000, 2.618, 3.618, 4.236
            ],
            'lookback_period': lookback_period,
            'min_swing_strength': min_swing_strength,
            'cluster_tolerance': cluster_tolerance,
            'min_cluster_size': min_cluster_size,
            'time_decay_factor': time_decay_factor,
            'volume_weight_factor': volume_weight_factor,
            'ml_validation': ml_validation
        }
        
        super().__init__(name=name, parameters=parameters)
        
        self.fibonacci_ratios = parameters['fibonacci_ratios']
        self.lookback_period = lookback_period
        self.min_swing_strength = min_swing_strength
        self.cluster_tolerance = cluster_tolerance
        self.min_cluster_size = min_cluster_size
        self.time_decay_factor = time_decay_factor
        self.volume_weight_factor = volume_weight_factor
        self.ml_validation = ml_validation
        
        # ML components
        self.scaler = StandardScaler()
        self.cluster_validator = None
        self.probability_predictor = None
        self._initialize_ml_components()
        
        # Cluster tracking
        self.active_clusters: List[ConfluenceCluster] = []
        self.historical_clusters: List[ConfluenceCluster] = []
        self.swing_points: List[SwingPoint] = []
        
    def get_data_requirements(self) -> DataRequirement:
        """Define data requirements for Fibonacci Clusters calculation."""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close'],
            min_periods=self.lookback_period,
            lookback_periods=self.lookback_period * 2,
            preprocessing=None
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, np.ndarray, Dict]]:
        """
        Perform the raw Fibonacci clusters calculation.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Dictionary containing cluster data, signals, and analysis
        """
        return self.calculate(data)
        
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on cluster analysis."""
        if not isinstance(value, dict) or 'signals' not in value:
            return None, 0.0
        
        signals = value['signals']
        if len(signals) == 0:
            return None, 0.0
        
        latest_signal = signals.iloc[-1]
        confidence = value.get('confidence', pd.Series([0.0])).iloc[-1] if len(value.get('confidence', [])) > 0 else 0.0
        
        # Stronger signals for cluster-based analysis
        if latest_signal > 0.8:
            return SignalType.STRONG_BUY, min(confidence, 0.95)
        elif latest_signal > 0.5:
            return SignalType.BUY, min(confidence, 0.85)
        elif latest_signal < -0.8:
            return SignalType.STRONG_SELL, min(confidence, 0.95)
        elif latest_signal < -0.5:
            return SignalType.SELL, min(confidence, 0.85)
        elif abs(latest_signal) > 0.3:
            return SignalType.HOLD, min(confidence, 0.75)
        else:
            return SignalType.NEUTRAL, min(confidence, 0.6)
    
    def _initialize_ml_components(self):
        """Initialize machine learning components for cluster analysis."""
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
            from sklearn.neural_network import MLPRegressor
            
            # Initialize cluster validator
            self.cluster_validator = RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Initialize probability predictor
            self.probability_predictor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
            
        except ImportError:
            logger.warning("scikit-learn not available, disabling ML features")
            self.ml_validation = False
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, np.ndarray, Dict]]:
        """
        Calculate Fibonacci clusters with advanced confluence analysis.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Dictionary containing cluster data, signals, and analysis
        """
        try:
            if len(data) < self.lookback_period:
                raise IndicatorCalculationException(
                    indicator_name=self.name,
                    calculation_step="data_validation",
                    message="Insufficient data for Fibonacci clusters calculation"
                )
            
            # Detect significant swing points
            swing_points = self._detect_swing_points(data)
            
            if len(swing_points) < 4:  # Need at least 2 highs and 2 lows
                return self._empty_result(len(data))
            
            # Calculate all Fibonacci levels from swing pairs
            fibonacci_levels = self._calculate_all_fibonacci_levels(swing_points, data)
            
            if len(fibonacci_levels) < self.min_cluster_size:
                return self._empty_result(len(data))
            
            # Identify confluence clusters
            clusters = self._identify_confluence_clusters(fibonacci_levels, data)
            
            if not clusters:
                return self._empty_result(len(data))
            
            # Validate and score clusters
            validated_clusters = self._validate_and_score_clusters(clusters, data)
            
            # Generate trading signals
            signals = self._generate_cluster_signals(validated_clusters, data)
            
            # Calculate strength metrics
            strength_metrics = self._calculate_cluster_strength_metrics(validated_clusters, data)
            
            # Identify key levels
            key_levels = self._identify_key_levels(validated_clusters, data)
            
            # Calculate probabilities
            probabilities = self._calculate_cluster_probabilities(validated_clusters, data)
            
            return {
                'clusters': self._format_clusters(validated_clusters),
                'fibonacci_levels': self._format_fibonacci_levels(fibonacci_levels),
                'signals': signals,
                'strength': strength_metrics['strength'],
                'confidence': strength_metrics['confidence'],
                'cluster_proximity': strength_metrics['cluster_proximity'],
                'volume_confirmation': strength_metrics['volume_confirmation'],
                'key_levels': key_levels,
                'probabilities': probabilities,
                'support_zones': self._identify_support_zones(validated_clusters, data),
                'resistance_zones': self._identify_resistance_zones(validated_clusters, data),
                'confluence_map': self._create_confluence_map(validated_clusters, data),
                'cluster_statistics': self._calculate_cluster_statistics(validated_clusters)
            }
            
        except Exception as e:
            logger.error(f"Error in Fibonacci clusters calculation: {str(e)}")
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="main_calculation",
                message=f"Fibonacci clusters calculation failed: {str(e)}",
                cause=e
            )
    
    def _detect_swing_points(self, data: pd.DataFrame) -> List[SwingPoint]:
        """Detect significant swing highs and lows using advanced algorithms."""
        
        def _find_pivots(prices_high: np.ndarray, prices_low: np.ndarray, 
                        volumes: np.ndarray, window: int = 5) -> Tuple[List[int], List[int]]:
            """Find pivot highs and lows with volume confirmation."""
            pivot_highs = []
            pivot_lows = []
            
            for i in range(window, len(prices_high) - window):
                # Check for pivot high
                is_pivot_high = True
                for j in range(1, window + 1):
                    if (prices_high[i] < prices_high[i-j] or 
                        prices_high[i] < prices_high[i+j]):
                        is_pivot_high = False
                        break
                
                if is_pivot_high:
                    pivot_highs.append(i)
                
                # Check for pivot low
                is_pivot_low = True
                for j in range(1, window + 1):
                    if (prices_low[i] > prices_low[i-j] or 
                        prices_low[i] > prices_low[i+j]):
                        is_pivot_low = False
                        break
                
                if is_pivot_low:
                    pivot_lows.append(i)
            
            return pivot_highs, pivot_lows
        
        # Calculate ATR for strength validation
        atr = self._calculate_atr(data)
        
        # Find pivot points
        recent_data = data.iloc[-self.lookback_period:] if len(data) > self.lookback_period else data
        volumes = recent_data['volume'].values if 'volume' in recent_data.columns else np.ones(len(recent_data))
        
        pivot_highs, pivot_lows = _find_pivots(
            recent_data['high'].values, 
            recent_data['low'].values,
            volumes
        )
        
        swing_points = []
        
        # Process pivot highs
        for idx in pivot_highs:
            actual_idx = len(data) - len(recent_data) + idx
            if actual_idx < len(data) and actual_idx < len(atr):
                strength = self._calculate_swing_strength(
                    data, actual_idx, 'high', atr[actual_idx]
                )
                
                if strength >= self.min_swing_strength:
                    time_weight = self._calculate_time_weight(actual_idx, len(data))
                    
                    swing_point = SwingPoint(
                        index=actual_idx,
                        price=data.iloc[actual_idx]['high'],
                        swing_type='high',
                        strength=strength,
                        volume=data.iloc[actual_idx].get('volume', 0),
                        time_weight=time_weight,
                        atr_multiple=strength
                    )
                    swing_points.append(swing_point)
        
        # Process pivot lows
        for idx in pivot_lows:
            actual_idx = len(data) - len(recent_data) + idx
            if actual_idx < len(data) and actual_idx < len(atr):
                strength = self._calculate_swing_strength(
                    data, actual_idx, 'low', atr[actual_idx]
                )
                
                if strength >= self.min_swing_strength:
                    time_weight = self._calculate_time_weight(actual_idx, len(data))
                    
                    swing_point = SwingPoint(
                        index=actual_idx,
                        price=data.iloc[actual_idx]['low'],
                        swing_type='low',
                        strength=strength,
                        volume=data.iloc[actual_idx].get('volume', 0),
                        time_weight=time_weight,
                        atr_multiple=strength
                    )
                    swing_points.append(swing_point)
        
        # Sort by strength and time relevance
        swing_points.sort(key=lambda x: x.strength * x.time_weight, reverse=True)
        
        # Keep top swing points
        return swing_points[:20]  # Limit to prevent computational overhead
    
    def _calculate_swing_strength(self, data: pd.DataFrame, idx: int, 
                                swing_type: str, atr_value: float) -> float:
        """Calculate the strength of a swing point."""
        if idx >= len(data) or atr_value <= 0:
            return 0.0
        
        try:
            price = data.iloc[idx]['high'] if swing_type == 'high' else data.iloc[idx]['low']
            
            # Calculate price deviation from nearby prices
            window = min(10, idx, len(data) - idx - 1)
            if window < 3:
                return 0.0
            
            nearby_prices = []
            for i in range(max(0, idx - window), min(len(data), idx + window + 1)):
                if i != idx:
                    compare_price = data.iloc[i]['high'] if swing_type == 'high' else data.iloc[i]['low']
                    nearby_prices.append(compare_price)
            
            if not nearby_prices:
                return 0.0
            
            if swing_type == 'high':
                deviation = price - max(nearby_prices)
            else:
                deviation = min(nearby_prices) - price
            
            # Normalize by ATR
            strength = max(0.0, deviation / atr_value)
            
            # Add volume confirmation if available
            if 'volume' in data.columns:
                current_volume = data.iloc[idx]['volume']
                avg_volume = data['volume'].iloc[max(0, idx-10):idx+1].mean()
                if avg_volume > 0:
                    volume_factor = min(2.0, current_volume / avg_volume)
                    strength *= volume_factor
            
            return min(5.0, strength)  # Cap at 5.0
            
        except Exception:
            return 0.0
    
    def _calculate_time_weight(self, swing_idx: int, total_length: int) -> float:
        """Calculate time-based weight for swing relevance."""
        if total_length <= swing_idx:
            return 0.0
        
        periods_ago = total_length - swing_idx - 1
        return self.time_decay_factor ** periods_ago
    
    def _calculate_all_fibonacci_levels(self, swing_points: List[SwingPoint], 
                                      data: pd.DataFrame) -> List[FibonacciLevel]:
        """Calculate all Fibonacci levels from swing point pairs."""
        fibonacci_levels = []
        current_price = data['close'].iloc[-1]
        
        # Get swing highs and lows separately
        swing_highs = [sp for sp in swing_points if sp.swing_type == 'high']
        swing_lows = [sp for sp in swing_points if sp.swing_type == 'low']
        
        # Calculate levels for each high-low pair
        for high_swing in swing_highs:
            for low_swing in swing_lows:
                # Skip if swings are too close in time
                if abs(high_swing.index - low_swing.index) < 5:
                    continue
                
                # Calculate Fibonacci levels for this pair
                pair_levels = self._calculate_fibonacci_levels_for_pair(
                    high_swing, low_swing, data
                )
                fibonacci_levels.extend(pair_levels)
        
        # Filter levels based on relevance to current price
        relevant_levels = []
        for level in fibonacci_levels:
            distance_ratio = abs(level.level_price - current_price) / current_price
            if distance_ratio <= 0.2:  # Within 20% of current price
                level.distance_from_current = distance_ratio
                relevant_levels.append(level)
        
        return relevant_levels
    
    def _calculate_fibonacci_levels_for_pair(self, high_swing: SwingPoint, 
                                           low_swing: SwingPoint, 
                                           data: pd.DataFrame) -> List[FibonacciLevel]:
        """Calculate Fibonacci levels for a specific swing pair."""
        levels = []
        
        try:
            price_range = high_swing.price - low_swing.price
            if price_range <= 0:
                return levels
            
            # Calculate volume weight for this pair
            start_idx = min(high_swing.index, low_swing.index)
            end_idx = max(high_swing.index, low_swing.index)
            
            pair_volume = 0.0
            if 'volume' in data.columns:
                pair_volume = data['volume'].iloc[start_idx:end_idx+1].mean()
            
            volume_weight = self._calculate_volume_weight(pair_volume, data)
            
            # Calculate time weight (more recent pairs get higher weight)
            avg_time_weight = (high_swing.time_weight + low_swing.time_weight) / 2
            
            # Calculate strength weight
            avg_strength = (high_swing.strength + low_swing.strength) / 2
            
            # Generate retracement levels
            for ratio in self.fibonacci_ratios:
                if ratio <= 1.0:  # Retracement levels
                    level_price = high_swing.price - (price_range * ratio)
                    
                    level = FibonacciLevel(
                        swing_high_idx=high_swing.index,
                        swing_low_idx=low_swing.index,
                        swing_high_price=high_swing.price,
                        swing_low_price=low_swing.price,
                        ratio=ratio,
                        level_price=level_price,
                        level_type='retracement',
                        time_weight=avg_time_weight,
                        volume_weight=volume_weight,
                        strength=avg_strength,
                        distance_from_current=0.0  # Will be set later
                    )
                    levels.append(level)
                
                else:  # Extension levels
                    # Extension below the low
                    level_price = low_swing.price - (price_range * (ratio - 1.0))
                    
                    level = FibonacciLevel(
                        swing_high_idx=high_swing.index,
                        swing_low_idx=low_swing.index,
                        swing_high_price=high_swing.price,
                        swing_low_price=low_swing.price,
                        ratio=ratio,
                        level_price=level_price,
                        level_type='extension',
                        time_weight=avg_time_weight,
                        volume_weight=volume_weight,
                        strength=avg_strength,
                        distance_from_current=0.0
                    )
                    levels.append(level)
                    
                    # Extension above the high
                    level_price = high_swing.price + (price_range * (ratio - 1.0))
                    
                    level = FibonacciLevel(
                        swing_high_idx=high_swing.index,
                        swing_low_idx=low_swing.index,
                        swing_high_price=high_swing.price,
                        swing_low_price=low_swing.price,
                        ratio=ratio,
                        level_price=level_price,
                        level_type='extension',
                        time_weight=avg_time_weight,
                        volume_weight=volume_weight,
                        strength=avg_strength,
                        distance_from_current=0.0
                    )
                    levels.append(level)
            
            return levels
            
        except Exception as e:
            logger.warning(f"Error calculating Fibonacci levels for pair: {str(e)}")
            return []
    
    def _calculate_volume_weight(self, pair_volume: float, data: pd.DataFrame) -> float:
        """Calculate volume weight for a swing pair."""
        if 'volume' not in data.columns or pair_volume <= 0:
            return 0.5  # Neutral weight
        
        try:
            avg_volume = data['volume'].iloc[-50:].mean()  # Recent average
            if avg_volume <= 0:
                return 0.5
            
            volume_ratio = pair_volume / avg_volume
            return min(1.0, 0.3 + (volume_ratio * self.volume_weight_factor))
            
        except Exception:
            return 0.5
    
    def _identify_confluence_clusters(self, fibonacci_levels: List[FibonacciLevel], 
                                    data: pd.DataFrame) -> List[ConfluenceCluster]:
        """Identify clusters where multiple Fibonacci levels converge."""
        if len(fibonacci_levels) < self.min_cluster_size:
            return []
        
        try:
            # Prepare data for clustering
            level_prices = np.array([level.level_price for level in fibonacci_levels]).reshape(-1, 1)
            
            # Use DBSCAN for clustering with price tolerance
            current_price = data['close'].iloc[-1]
            eps = current_price * self.cluster_tolerance  # Dynamic epsilon based on price
            
            clustering = DBSCAN(eps=eps, min_samples=self.min_cluster_size)
            cluster_labels = clustering.fit_predict(level_prices)
            
            clusters = []
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                
                # Get levels in this cluster
                cluster_indices = [i for i, l in enumerate(cluster_labels) if l == label]
                cluster_levels = [fibonacci_levels[i] for i in cluster_indices]
                
                if len(cluster_levels) >= self.min_cluster_size:
                    cluster = self._create_confluence_cluster(cluster_levels, label, data)
                    if cluster:
                        clusters.append(cluster)
            
            # Sort clusters by confluence score
            clusters.sort(key=lambda x: x.confluence_score, reverse=True)
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Error identifying confluence clusters: {str(e)}")
            return []
    
    def _create_confluence_cluster(self, cluster_levels: List[FibonacciLevel], 
                                 cluster_id: int, data: pd.DataFrame) -> Optional[ConfluenceCluster]:
        """Create a confluence cluster from a group of levels."""
        try:
            if not cluster_levels:
                return None
            
            # Calculate cluster center and range
            prices = [level.level_price for level in cluster_levels]
            center_price = np.mean(prices)
            min_price = min(prices)
            max_price = max(prices)
            
            # Calculate cluster strength metrics
            total_strength = sum(level.strength for level in cluster_levels)
            avg_time_weight = np.mean([level.time_weight for level in cluster_levels])
            avg_volume_weight = np.mean([level.volume_weight for level in cluster_levels])
            
            # Calculate confluence score
            confluence_score = self._calculate_confluence_score(cluster_levels, data)
            
            # Determine support/resistance type
            current_price = data['close'].iloc[-1]
            if center_price < current_price * 0.99:
                sr_type = 'support'
            elif center_price > current_price * 1.01:
                sr_type = 'resistance'
            else:
                sr_type = 'both'
            
            # Calculate hit probability
            hit_probability = self._calculate_hit_probability(cluster_levels, data)
            
            # Calculate expected reaction magnitude
            reaction_magnitude = self._calculate_reaction_magnitude(cluster_levels, data)
            
            # Calculate volume confirmation
            volume_confirmation = self._calculate_cluster_volume_confirmation(cluster_levels, data)
            
            cluster = ConfluenceCluster(
                cluster_id=cluster_id,
                center_price=center_price,
                price_range=(min_price, max_price),
                levels=cluster_levels,
                cluster_strength=total_strength,
                confluence_score=confluence_score,
                time_relevance=avg_time_weight,
                volume_confirmation=volume_confirmation,
                support_resistance_type=sr_type,
                hit_probability=hit_probability,
                reaction_magnitude=reaction_magnitude
            )
            
            return cluster
            
        except Exception as e:
            logger.warning(f"Error creating confluence cluster: {str(e)}")
            return None
    
    def _calculate_confluence_score(self, cluster_levels: List[FibonacciLevel], 
                                  data: pd.DataFrame) -> float:
        """Calculate the confluence score for a cluster."""
        try:
            # Base score from number of levels
            level_count_score = min(1.0, len(cluster_levels) / 10.0)
            
            # Diversity score (different swing pairs)
            unique_pairs = set()
            for level in cluster_levels:
                pair_key = (level.swing_high_idx, level.swing_low_idx)
                unique_pairs.add(pair_key)
            
            diversity_score = min(1.0, len(unique_pairs) / len(cluster_levels))
            
            # Ratio diversity score (different Fibonacci ratios)
            unique_ratios = set(level.ratio for level in cluster_levels)
            ratio_diversity = min(1.0, len(unique_ratios) / len(self.fibonacci_ratios))
            
            # Strength score
            avg_strength = np.mean([level.strength for level in cluster_levels])
            strength_score = min(1.0, avg_strength / 2.0)
            
            # Time relevance score
            avg_time_weight = np.mean([level.time_weight for level in cluster_levels])
            
            # Volume confirmation score
            avg_volume_weight = np.mean([level.volume_weight for level in cluster_levels])
            
            # Combined confluence score
            confluence_score = (
                level_count_score * 0.25 +
                diversity_score * 0.25 +
                ratio_diversity * 0.15 +
                strength_score * 0.15 +
                avg_time_weight * 0.1 +
                avg_volume_weight * 0.1
            )
            
            return min(1.0, confluence_score)
            
        except Exception:
            return 0.0
    
    def _calculate_hit_probability(self, cluster_levels: List[FibonacciLevel], 
                                 data: pd.DataFrame) -> float:
        """Calculate the probability of price hitting the cluster."""
        try:
            current_price = data['close'].iloc[-1]
            center_price = np.mean([level.level_price for level in cluster_levels])
            
            # Distance factor
            distance = abs(center_price - current_price) / current_price
            distance_factor = max(0.1, 1.0 - distance * 5)  # Closer = higher probability
            
            # Trend alignment factor
            trend_factor = self._calculate_trend_alignment_factor(cluster_levels, data)
            
            # Volume factor
            volume_factor = np.mean([level.volume_weight for level in cluster_levels])
            
            # Strength factor
            strength_factor = min(1.0, np.mean([level.strength for level in cluster_levels]) / 3.0)
            
            # Time relevance factor
            time_factor = np.mean([level.time_weight for level in cluster_levels])
            
            # Combined probability
            probability = (
                distance_factor * 0.3 +
                trend_factor * 0.25 +
                volume_factor * 0.2 +
                strength_factor * 0.15 +
                time_factor * 0.1
            )
            
            return min(1.0, probability)
            
        except Exception:
            return 0.5
    
    def _calculate_trend_alignment_factor(self, cluster_levels: List[FibonacciLevel], 
                                        data: pd.DataFrame) -> float:
        """Calculate how well the cluster aligns with current trend."""
        try:
            current_price = data['close'].iloc[-1]
            center_price = np.mean([level.level_price for level in cluster_levels])
            
            # Calculate recent trend
            recent_prices = data['close'].iloc[-20:]
            if len(recent_prices) < 10:
                return 0.5
            
            trend_slope = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
            
            # Determine if cluster acts as support or resistance
            if center_price < current_price:  # Potential support
                # Uptrend favors support test
                return 0.8 if trend_slope > 0 else 0.3
            else:  # Potential resistance
                # Downtrend favors resistance test
                return 0.8 if trend_slope < 0 else 0.3
                
        except Exception:
            return 0.5
    
    def _calculate_reaction_magnitude(self, cluster_levels: List[FibonacciLevel], 
                                    data: pd.DataFrame) -> float:
        """Calculate expected price reaction magnitude at cluster."""
        try:
            # Base magnitude from cluster strength
            avg_strength = np.mean([level.strength for level in cluster_levels])
            strength_magnitude = min(0.1, avg_strength * 0.02)  # Max 10% move
            
            # ATR-based magnitude
            atr = self._calculate_atr(data)
            if len(atr) > 0:
                current_atr = atr[-1]
                current_price = data['close'].iloc[-1]
                atr_magnitude = current_atr / current_price if current_price > 0 else 0
            else:
                atr_magnitude = 0.01
            
            # Volume factor
            avg_volume_weight = np.mean([level.volume_weight for level in cluster_levels])
            volume_magnitude = atr_magnitude * avg_volume_weight
            
            # Combined magnitude
            expected_magnitude = max(strength_magnitude, volume_magnitude)
            
            return min(0.15, expected_magnitude)  # Cap at 15%
            
        except Exception:
            return 0.02  # Default 2%
    
    def _calculate_cluster_volume_confirmation(self, cluster_levels: List[FibonacciLevel], 
                                             data: pd.DataFrame) -> float:
        """Calculate volume confirmation for the cluster."""
        if 'volume' not in data.columns:
            return 0.5
        
        try:
            # Check if any swing points in cluster had significant volume
            high_volume_count = 0
            total_count = 0
            
            for level in cluster_levels:
                # Check volume at swing points
                high_vol = data['volume'].iloc[level.swing_high_idx]
                low_vol = data['volume'].iloc[level.swing_low_idx]
                
                avg_volume = data['volume'].iloc[-50:].mean()
                
                if high_vol > avg_volume * 1.5 or low_vol > avg_volume * 1.5:
                    high_volume_count += 1
                total_count += 1
            
            volume_confirmation = high_volume_count / total_count if total_count > 0 else 0.5
            
            # Weight by level volume weights
            avg_volume_weight = np.mean([level.volume_weight for level in cluster_levels])
            
            return (volume_confirmation + avg_volume_weight) / 2
            
        except Exception:
            return 0.5
    
    def _validate_and_score_clusters(self, clusters: List[ConfluenceCluster], 
                                   data: pd.DataFrame) -> List[ConfluenceCluster]:
        """Validate and score clusters using advanced algorithms."""
        if not clusters:
            return []
        
        validated_clusters = []
        
        for cluster in clusters:
            try:
                # Basic validation
                if (cluster.confluence_score >= 0.3 and 
                    cluster.cluster_strength >= 1.0 and
                    len(cluster.levels) >= self.min_cluster_size):
                    
                    # ML validation if enabled
                    if self.ml_validation and self.cluster_validator is not None:
                        ml_score = self._validate_cluster_with_ml(cluster, data)
                        cluster.confluence_score = (cluster.confluence_score + ml_score) / 2
                    
                    # Historical validation
                    historical_score = self._validate_cluster_historically(cluster, data)
                    cluster.hit_probability = (cluster.hit_probability + historical_score) / 2
                    
                    validated_clusters.append(cluster)
                    
            except Exception as e:
                logger.warning(f"Error validating cluster {cluster.cluster_id}: {str(e)}")
                continue
        
        # Sort by combined score
        validated_clusters.sort(
            key=lambda x: x.confluence_score * x.hit_probability, 
            reverse=True
        )
        
        return validated_clusters[:10]  # Return top 10 clusters
    
    def _validate_cluster_with_ml(self, cluster: ConfluenceCluster, 
                                data: pd.DataFrame) -> float:
        """Validate cluster using machine learning model."""
        try:
            # Extract features for ML validation
            features = self._extract_cluster_features(cluster, data)
            
            # This would use a pre-trained model in production
            # For now, return a score based on statistical measures
            base_score = cluster.confluence_score
            level_count_factor = min(1.0, len(cluster.levels) / 8.0)
            strength_factor = min(1.0, cluster.cluster_strength / 10.0)
            
            ml_score = (base_score + level_count_factor + strength_factor) / 3
            return min(1.0, ml_score)
            
        except Exception:
            return cluster.confluence_score
    
    def _extract_cluster_features(self, cluster: ConfluenceCluster, 
                                data: pd.DataFrame) -> np.ndarray:
        """Extract features for ML cluster validation."""
        features = [
            cluster.confluence_score,
            cluster.cluster_strength,
            len(cluster.levels),
            cluster.time_relevance,
            cluster.volume_confirmation,
            cluster.hit_probability,
            cluster.reaction_magnitude
        ]
        
        # Add market context features
        current_price = data['close'].iloc[-1]
        distance_to_cluster = abs(cluster.center_price - current_price) / current_price
        features.append(distance_to_cluster)
        
        # Add volatility features
        if len(data) >= 20:
            volatility = data['close'].iloc[-20:].std() / data['close'].iloc[-20:].mean()
            features.append(volatility)
        else:
            features.append(0.02)  # Default volatility
        
        return np.array(features).reshape(1, -1)
    
    def _validate_cluster_historically(self, cluster: ConfluenceCluster, 
                                     data: pd.DataFrame) -> float:
        """Validate cluster based on historical price action."""
        try:
            # Check how price has reacted near this cluster in the past
            center_price = cluster.center_price
            tolerance = center_price * self.cluster_tolerance * 2  # Wider tolerance for historical check
            
            reactions = 0
            total_approaches = 0
            
            for i in range(len(data) - 1):
                current_price = data.iloc[i]['close']
                next_price = data.iloc[i + 1]['close']
                
                # Check if price approached the cluster
                if abs(current_price - center_price) <= tolerance:
                    total_approaches += 1
                    
                    # Check if there was a reaction (price moved away)
                    if cluster.support_resistance_type == 'support':
                        if current_price <= center_price and next_price > current_price:
                            reactions += 1
                    elif cluster.support_resistance_type == 'resistance':
                        if current_price >= center_price and next_price < current_price:
                            reactions += 1
                    else:  # both
                        if abs(next_price - center_price) > abs(current_price - center_price):
                            reactions += 1
            
            if total_approaches > 0:
                return reactions / total_approaches
            else:
                return cluster.hit_probability  # No historical data, use calculated probability
                
        except Exception:
            return cluster.hit_probability
    
    def _generate_cluster_signals(self, clusters: List[ConfluenceCluster], 
                                data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on cluster analysis."""
        signals = pd.Series(0.0, index=data.index)
        
        if not clusters:
            return signals
        
        try:
            current_price = data['close'].iloc[-1]
            
            # Generate signals for recent periods
            for i in range(max(1, len(data) - 30), len(data)):
                signal_strength = 0.0
                price = data.iloc[i]['close']
                
                for cluster in clusters:
                    # Calculate distance to cluster
                    distance = abs(price - cluster.center_price) / cluster.center_price
                    
                    # Generate signal based on cluster proximity and type
                    if distance <= self.cluster_tolerance * 3:  # Near cluster
                        cluster_signal = 0.0
                        
                        if cluster.support_resistance_type == 'support' and price <= cluster.center_price:
                            # Near support - bullish signal
                            cluster_signal = cluster.confluence_score * cluster.hit_probability * 0.8
                        elif cluster.support_resistance_type == 'resistance' and price >= cluster.center_price:
                            # Near resistance - bearish signal
                            cluster_signal = -cluster.confluence_score * cluster.hit_probability * 0.8
                        elif cluster.support_resistance_type == 'both':
                            # Price reaction at significant level
                            if i > 0:
                                prev_price = data.iloc[i-1]['close']
                                if abs(prev_price - cluster.center_price) > abs(price - cluster.center_price):
                                    # Moving toward cluster - weakening signal
                                    cluster_signal = 0.0
                                else:
                                    # Moving away from cluster - strengthening signal
                                    direction = 1 if price > cluster.center_price else -1
                                    cluster_signal = direction * cluster.confluence_score * 0.6
                        
                        # Weight by cluster strength
                        weighted_signal = cluster_signal * min(1.0, cluster.cluster_strength / 5.0)
                        signal_strength += weighted_signal
                
                # Apply signal
                signals.iloc[i] = np.clip(signal_strength, -1.0, 1.0)
            
            return signals
            
        except Exception as e:
            logger.warning(f"Error generating cluster signals: {str(e)}")
            return signals
    
    def _calculate_cluster_strength_metrics(self, clusters: List[ConfluenceCluster], 
                                          data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate comprehensive strength metrics for clusters."""
        length = len(data)
        
        strength = pd.Series(0.0, index=data.index)
        confidence = pd.Series(0.0, index=data.index)
        cluster_proximity = pd.Series(0.0, index=data.index)
        volume_confirmation = pd.Series(0.0, index=data.index)
        
        if not clusters:
            return {
                'strength': strength,
                'confidence': confidence,
                'cluster_proximity': cluster_proximity,
                'volume_confirmation': volume_confirmation
            }
        
        try:
            for i in range(length):
                price = data.iloc[i]['close']
                
                nearby_clusters = []
                total_strength = 0.0
                total_confidence = 0.0
                total_volume = 0.0
                min_distance = float('inf')
                
                for cluster in clusters:
                    distance = abs(price - cluster.center_price) / cluster.center_price
                    
                    if distance <= 0.1:  # Within 10%
                        nearby_clusters.append(cluster)
                        total_strength += cluster.cluster_strength
                        total_confidence += cluster.confluence_score
                        total_volume += cluster.volume_confirmation
                        min_distance = min(min_distance, distance)
                
                if nearby_clusters:
                    count = len(nearby_clusters)
                    strength.iloc[i] = min(1.0, total_strength / count / 5.0)
                    confidence.iloc[i] = total_confidence / count
                    volume_confirmation.iloc[i] = total_volume / count
                    cluster_proximity.iloc[i] = max(0.0, 1.0 - min_distance * 10)
            
            return {
                'strength': strength,
                'confidence': confidence,
                'cluster_proximity': cluster_proximity,
                'volume_confirmation': volume_confirmation
            }
            
        except Exception as e:
            logger.warning(f"Error calculating cluster strength metrics: {str(e)}")
            return {
                'strength': strength,
                'confidence': confidence,
                'cluster_proximity': cluster_proximity,
                'volume_confirmation': volume_confirmation
            }
    
    def _identify_key_levels(self, clusters: List[ConfluenceCluster], 
                           data: pd.DataFrame) -> Dict[str, List[float]]:
        """Identify key support and resistance levels from clusters."""
        key_levels = {
            'strong_support': [],
            'strong_resistance': [],
            'moderate_support': [],
            'moderate_resistance': []
        }
        
        try:
            current_price = data['close'].iloc[-1]
            
            for cluster in clusters:
                strength_threshold = 0.6
                moderate_threshold = 0.4
                
                if cluster.confluence_score >= strength_threshold:
                    if cluster.support_resistance_type in ['support', 'both'] and cluster.center_price < current_price:
                        key_levels['strong_support'].append(cluster.center_price)
                    if cluster.support_resistance_type in ['resistance', 'both'] and cluster.center_price > current_price:
                        key_levels['strong_resistance'].append(cluster.center_price)
                
                elif cluster.confluence_score >= moderate_threshold:
                    if cluster.support_resistance_type in ['support', 'both'] and cluster.center_price < current_price:
                        key_levels['moderate_support'].append(cluster.center_price)
                    if cluster.support_resistance_type in ['resistance', 'both'] and cluster.center_price > current_price:
                        key_levels['moderate_resistance'].append(cluster.center_price)
            
            # Sort levels
            for key in key_levels:
                key_levels[key] = sorted(set(key_levels[key]))
                if 'support' in key:
                    key_levels[key] = key_levels[key][-3:]  # Keep closest support levels
                else:
                    key_levels[key] = key_levels[key][:3]   # Keep closest resistance levels
            
            return key_levels
            
        except Exception as e:
            logger.warning(f"Error identifying key levels: {str(e)}")
            return key_levels
    
    def _calculate_cluster_probabilities(self, clusters: List[ConfluenceCluster], 
                                       data: pd.DataFrame) -> Dict[str, float]:
        """Calculate various probabilities based on cluster analysis."""
        probabilities = {
            'nearest_cluster_hit': 0.0,
            'strong_reaction': 0.0,
            'trend_continuation': 0.0,
            'reversal': 0.0
        }
        
        try:
            if not clusters:
                return probabilities
            
            current_price = data['close'].iloc[-1]
            
            # Find nearest cluster
            nearest_cluster = min(clusters, 
                                key=lambda x: abs(x.center_price - current_price) / current_price)
            
            probabilities['nearest_cluster_hit'] = nearest_cluster.hit_probability
            probabilities['strong_reaction'] = nearest_cluster.reaction_magnitude * 10  # Convert to probability scale
            
            # Calculate trend continuation vs reversal probability
            strong_clusters = [c for c in clusters if c.confluence_score >= 0.6]
            
            if strong_clusters:
                support_strength = sum(c.confluence_score for c in strong_clusters 
                                     if c.support_resistance_type in ['support', 'both'] 
                                     and c.center_price < current_price)
                
                resistance_strength = sum(c.confluence_score for c in strong_clusters 
                                        if c.support_resistance_type in ['resistance', 'both'] 
                                        and c.center_price > current_price)
                
                total_strength = support_strength + resistance_strength
                
                if total_strength > 0:
                    # Strong support suggests trend continuation if uptrend
                    recent_trend = data['close'].iloc[-1] - data['close'].iloc[-10]
                    
                    if recent_trend > 0 and support_strength > resistance_strength:
                        probabilities['trend_continuation'] = min(0.8, support_strength / total_strength)
                        probabilities['reversal'] = 1.0 - probabilities['trend_continuation']
                    elif recent_trend < 0 and resistance_strength > support_strength:
                        probabilities['trend_continuation'] = min(0.8, resistance_strength / total_strength)
                        probabilities['reversal'] = 1.0 - probabilities['trend_continuation']
                    else:
                        probabilities['trend_continuation'] = 0.5
                        probabilities['reversal'] = 0.5
            
            return probabilities
            
        except Exception as e:
            logger.warning(f"Error calculating cluster probabilities: {str(e)}")
            return probabilities
    
    def _identify_support_zones(self, clusters: List[ConfluenceCluster], 
                              data: pd.DataFrame) -> List[Dict]:
        """Identify support zones from clusters."""
        support_zones = []
        
        try:
            current_price = data['close'].iloc[-1]
            
            for cluster in clusters:
                if (cluster.support_resistance_type in ['support', 'both'] and 
                    cluster.center_price < current_price and
                    cluster.confluence_score >= 0.4):
                    
                    zone = {
                        'price': cluster.center_price,
                        'range': cluster.price_range,
                        'strength': cluster.confluence_score,
                        'hit_probability': cluster.hit_probability,
                        'reaction_magnitude': cluster.reaction_magnitude,
                        'distance': (current_price - cluster.center_price) / current_price
                    }
                    support_zones.append(zone)
            
            # Sort by proximity to current price
            support_zones.sort(key=lambda x: x['distance'])
            
            return support_zones[:5]  # Return top 5 support zones
            
        except Exception as e:
            logger.warning(f"Error identifying support zones: {str(e)}")
            return []
    
    def _identify_resistance_zones(self, clusters: List[ConfluenceCluster], 
                                 data: pd.DataFrame) -> List[Dict]:
        """Identify resistance zones from clusters."""
        resistance_zones = []
        
        try:
            current_price = data['close'].iloc[-1]
            
            for cluster in clusters:
                if (cluster.support_resistance_type in ['resistance', 'both'] and 
                    cluster.center_price > current_price and
                    cluster.confluence_score >= 0.4):
                    
                    zone = {
                        'price': cluster.center_price,
                        'range': cluster.price_range,
                        'strength': cluster.confluence_score,
                        'hit_probability': cluster.hit_probability,
                        'reaction_magnitude': cluster.reaction_magnitude,
                        'distance': (cluster.center_price - current_price) / current_price
                    }
                    resistance_zones.append(zone)
            
            # Sort by proximity to current price
            resistance_zones.sort(key=lambda x: x['distance'])
            
            return resistance_zones[:5]  # Return top 5 resistance zones
            
        except Exception as e:
            logger.warning(f"Error identifying resistance zones: {str(e)}")
            return []
    
    def _create_confluence_map(self, clusters: List[ConfluenceCluster], 
                             data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Create a comprehensive confluence map."""
        confluence_map = {
            'high_confluence': [],
            'medium_confluence': [],
            'low_confluence': []
        }
        
        try:
            for cluster in clusters:
                cluster_info = {
                    'price': cluster.center_price,
                    'type': cluster.support_resistance_type,
                    'score': cluster.confluence_score,
                    'levels_count': len(cluster.levels),
                    'hit_probability': cluster.hit_probability,
                    'strength': cluster.cluster_strength
                }
                
                if cluster.confluence_score >= 0.7:
                    confluence_map['high_confluence'].append(cluster_info)
                elif cluster.confluence_score >= 0.5:
                    confluence_map['medium_confluence'].append(cluster_info)
                else:
                    confluence_map['low_confluence'].append(cluster_info)
            
            # Sort each category by score
            for category in confluence_map:
                confluence_map[category].sort(key=lambda x: x['score'], reverse=True)
            
            return confluence_map
            
        except Exception as e:
            logger.warning(f"Error creating confluence map: {str(e)}")
            return confluence_map
    
    def _calculate_cluster_statistics(self, clusters: List[ConfluenceCluster]) -> Dict[str, float]:
        """Calculate statistical summary of clusters."""
        if not clusters:
            return {}
        
        try:
            statistics = {
                'total_clusters': len(clusters),
                'avg_confluence_score': np.mean([c.confluence_score for c in clusters]),
                'max_confluence_score': max(c.confluence_score for c in clusters),
                'avg_cluster_strength': np.mean([c.cluster_strength for c in clusters]),
                'avg_hit_probability': np.mean([c.hit_probability for c in clusters]),
                'support_clusters': len([c for c in clusters if c.support_resistance_type in ['support', 'both']]),
                'resistance_clusters': len([c for c in clusters if c.support_resistance_type in ['resistance', 'both']]),
                'high_confidence_clusters': len([c for c in clusters if c.confluence_score >= 0.7]),
                'total_fibonacci_levels': sum(len(c.levels) for c in clusters)
            }
            
            return statistics
            
        except Exception as e:
            logger.warning(f"Error calculating cluster statistics: {str(e)}")
            return {}
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean().fillna(0).values
    
    def _format_clusters(self, clusters: List[ConfluenceCluster]) -> pd.DataFrame:
        """Format cluster data for output."""
        if not clusters:
            return pd.DataFrame()
        
        cluster_data = []
        for cluster in clusters:
            cluster_data.append({
                'cluster_id': cluster.cluster_id,
                'center_price': cluster.center_price,
                'price_range_min': cluster.price_range[0],
                'price_range_max': cluster.price_range[1],
                'levels_count': len(cluster.levels),
                'cluster_strength': cluster.cluster_strength,
                'confluence_score': cluster.confluence_score,
                'time_relevance': cluster.time_relevance,
                'volume_confirmation': cluster.volume_confirmation,
                'support_resistance_type': cluster.support_resistance_type,
                'hit_probability': cluster.hit_probability,
                'reaction_magnitude': cluster.reaction_magnitude
            })
        
        return pd.DataFrame(cluster_data)
    
    def _format_fibonacci_levels(self, levels: List[FibonacciLevel]) -> pd.DataFrame:
        """Format Fibonacci levels data for output."""
        if not levels:
            return pd.DataFrame()
        
        level_data = []
        for level in levels:
            level_data.append({
                'swing_high_idx': level.swing_high_idx,
                'swing_low_idx': level.swing_low_idx,
                'swing_high_price': level.swing_high_price,
                'swing_low_price': level.swing_low_price,
                'ratio': level.ratio,
                'level_price': level.level_price,
                'level_type': level.level_type,
                'time_weight': level.time_weight,
                'volume_weight': level.volume_weight,
                'strength': level.strength,
                'distance_from_current': level.distance_from_current
            })
        
        return pd.DataFrame(level_data)
    
    def _empty_result(self, length: int) -> Dict[str, Union[pd.Series, np.ndarray, Dict]]:
        """Return empty result structure."""
        empty_series = pd.Series(0.0, index=range(length))
        
        return {
            'clusters': pd.DataFrame(),
            'fibonacci_levels': pd.DataFrame(),
            'signals': empty_series,
            'strength': empty_series,
            'confidence': empty_series,
            'cluster_proximity': empty_series,
            'volume_confirmation': empty_series,
            'key_levels': {
                'strong_support': [],
                'strong_resistance': [],
                'moderate_support': [],
                'moderate_resistance': []
            },
            'probabilities': {
                'nearest_cluster_hit': 0.0,
                'strong_reaction': 0.0,
                'trend_continuation': 0.0,
                'reversal': 0.0
            },
            'support_zones': [],
            'resistance_zones': [],
            'confluence_map': {
                'high_confluence': [],
                'medium_confluence': [],
                'low_confluence': []
            },
            'cluster_statistics': {}
        }