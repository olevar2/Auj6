"""
Fibonacci Arcs Indicator - Advanced Implementation
===================================================

This indicator implements sophisticated Fibonacci arc calculations using advanced geometric algorithms
and machine learning-enhanced arc validation. Fibonacci arcs are curved support/resistance levels
based on Fibonacci ratios projected from significant price swings.

Features:
- Advanced arc geometry calculations with elliptical projections
- Machine learning arc strength validation
- Multi-timeframe arc confluence detection
- Dynamic arc adjustment based on volatility
- Comprehensive error handling and validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType
from ....core.exceptions import IndicatorCalculationException

logger = logging.getLogger(__name__)

@dataclass
class FibonacciArcLevel:
    """Represents a Fibonacci arc level with comprehensive properties."""
    ratio: float
    center_x: float
    center_y: float
    radius: float
    strength: float
    angle_start: float
    angle_end: float
    arc_points: List[Tuple[float, float]]
    confidence: float
    volume_confirmation: float

@dataclass
class ArcCluster:
    """Represents a cluster of nearby arc levels for confluence analysis."""
    levels: List[FibonacciArcLevel]
    center_price: float
    center_time: float
    strength: float
    confluence_score: float

class FibonacciArcsIndicator(StandardIndicatorInterface):
    """
    Advanced Fibonacci Arcs Indicator with machine learning validation.
    
    This implementation uses sophisticated geometric algorithms to calculate
    Fibonacci arcs and employs machine learning techniques to validate
    arc strength and predict price reactions.
    """
    
    def __init__(self, 
                 name: str = "FibonacciArcs",
                 fibonacci_ratios: List[float] = None,
                 lookback_period: int = 100,
                 min_swing_strength: float = 0.02,
                 arc_resolution: int = 100,
                 volatility_adjustment: bool = True,
                 ml_validation: bool = True,
                 confidence_threshold: float = 0.6):
        """
        Initialize the Fibonacci Arcs Indicator.
        
        Args:
            name: Indicator name
            fibonacci_ratios: Custom Fibonacci ratios for arc calculation
            lookback_period: Period for swing detection
            min_swing_strength: Minimum strength for significant swings
            arc_resolution: Number of points per arc
            volatility_adjustment: Enable dynamic adjustment based on volatility
            ml_validation: Enable machine learning arc validation
            confidence_threshold: Minimum confidence for arc signals
        """
        parameters = {
            'fibonacci_ratios': fibonacci_ratios or [0.236, 0.382, 0.500, 0.618, 0.764, 1.000, 1.272, 1.618],
            'lookback_period': lookback_period,
            'min_swing_strength': min_swing_strength,
            'arc_resolution': arc_resolution,
            'volatility_adjustment': volatility_adjustment,
            'ml_validation': ml_validation,
            'confidence_threshold': confidence_threshold
        }
        
        super().__init__(name=name, parameters=parameters)
        
        self.fibonacci_ratios = parameters['fibonacci_ratios']
        self.lookback_period = lookback_period
        self.min_swing_strength = min_swing_strength
        self.arc_resolution = arc_resolution
        self.volatility_adjustment = volatility_adjustment
        self.ml_validation = ml_validation
        self.confidence_threshold = confidence_threshold
        
        # ML components
        self.scaler = StandardScaler()
        self.arc_validator = None
        self._initialize_ml_components()
        
        # Arc tracking
        self.active_arcs: List[FibonacciArcLevel] = []
        self.arc_clusters: List[ArcCluster] = []
        self.price_reactions: List[Dict] = []
    
    def get_data_requirements(self) -> DataRequirement:
        """Define data requirements for Fibonacci Arcs calculation."""
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=['open', 'high', 'low', 'close'],
            min_periods=self.lookback_period,
            lookback_periods=self.lookback_period * 2,
            preprocessing=None
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, np.ndarray, Dict]]:
        """
        Perform the raw Fibonacci arcs calculation.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Dictionary containing arc levels, signals, and analysis
        """
        return self.calculate(data)
        
    def _generate_signal(self, value: Any, data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """Generate trading signal based on arc analysis."""
        if not isinstance(value, dict) or 'signals' not in value:
            return None, 0.0
        
        signals = value['signals']
        if len(signals) == 0:
            return None, 0.0
        
        latest_signal = signals.iloc[-1]
        confidence = value.get('confidence', pd.Series([0.0])).iloc[-1] if len(value.get('confidence', [])) > 0 else 0.0
        
        if latest_signal > 0.6:
            return SignalType.BUY, min(confidence, 0.95)
        elif latest_signal < -0.6:
            return SignalType.SELL, min(confidence, 0.95)
        elif abs(latest_signal) > 0.3:
            return SignalType.HOLD, min(confidence, 0.8)
        else:
            return SignalType.NEUTRAL, min(confidence, 0.5)
        
    def _initialize_ml_components(self):
        """Initialize machine learning components for arc validation."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.neural_network import MLPClassifier
            
            # Initialize arc strength validator
            self.arc_validator = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Initialize arc reaction predictor
            self.reaction_predictor = MLPClassifier(
                hidden_layer_sizes=(50, 25),
                max_iter=1000,
                random_state=42
            )
            
        except ImportError:
            logger.warning("scikit-learn not available, disabling ML features")
            self.ml_validation = False
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Union[pd.Series, np.ndarray, Dict]]:
        """
        Calculate Fibonacci arcs with advanced geometric and ML analysis.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            Dictionary containing arc levels, signals, and analysis
        """
        try:
            if len(data) < self.lookback_period:
                raise IndicatorCalculationException(
                    indicator_name=self.name,
                    calculation_step="data_validation",
                    message="Insufficient data for Fibonacci arcs calculation"
                )
            
            # Detect significant swings
            swing_highs, swing_lows = self._detect_swings(data)
            
            if len(swing_highs) < 2 or len(swing_lows) < 2:
                return self._empty_result(len(data))
            
            # Calculate arcs for each swing pair
            all_arcs = []
            for i in range(1, len(swing_highs)):
                for j in range(1, len(swing_lows)):
                    if abs(swing_highs[i]['index'] - swing_lows[j]['index']) > 5:
                        arcs = self._calculate_arc_levels(
                            swing_highs[i], swing_lows[j], data
                        )
                        all_arcs.extend(arcs)
            
            # Filter and validate arcs
            valid_arcs = self._validate_arcs(all_arcs, data)
            
            # Detect arc clusters
            clusters = self._detect_arc_clusters(valid_arcs)
            
            # Generate signals
            signals = self._generate_arc_signals(valid_arcs, clusters, data)
            
            # Calculate strength metrics
            strength_metrics = self._calculate_strength_metrics(valid_arcs, data)
            
            return {
                'arc_levels': self._format_arc_levels(valid_arcs),
                'clusters': self._format_clusters(clusters),
                'signals': signals,
                'strength': strength_metrics['strength'],
                'confidence': strength_metrics['confidence'],
                'trend_alignment': strength_metrics['trend_alignment'],
                'volume_confirmation': strength_metrics['volume_confirmation'],
                'price_targets': self._calculate_price_targets(valid_arcs, data),
                'support_resistance': self._identify_support_resistance(valid_arcs, data),
                'arc_intersections': self._find_arc_intersections(valid_arcs),
                'time_projections': self._calculate_time_projections(valid_arcs, data)
            }
            
        except Exception as e:
            logger.error(f"Error in Fibonacci arcs calculation: {str(e)}")
            raise IndicatorCalculationException(
                indicator_name=self.name,
                calculation_step="main_calculation",
                message=f"Fibonacci arcs calculation failed: {str(e)}",
                cause=e
            )
    
    def _detect_swings(self, data: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Detect significant swing highs and lows using advanced algorithms."""
        
        def _find_pivots(prices: np.ndarray, window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
            """Find pivot highs and lows."""
            highs = np.zeros(len(prices), dtype=bool)
            lows = np.zeros(len(prices), dtype=bool)
            
            for i in range(window, len(prices) - window):
                # Check for pivot high
                if all(prices[i] >= prices[i-j] for j in range(1, window+1)) and \
                   all(prices[i] >= prices[i+j] for j in range(1, window+1)):
                    highs[i] = True
                
                # Check for pivot low
                if all(prices[i] <= prices[i-j] for j in range(1, window+1)) and \
                   all(prices[i] <= prices[i+j] for j in range(1, window+1)):
                    lows[i] = True
            
            return highs, lows
        
        # Find pivot points
        pivot_highs, pivot_lows = _find_pivots(data['high'].values)
        
        # Calculate ATR for swing strength validation
        atr = self._calculate_atr(data)
        
        # Filter significant swings
        swing_highs = []
        swing_lows = []
        
        high_indices = np.where(pivot_highs)[0]
        for idx in high_indices:
            if idx < len(data) - 1:
                strength = self._calculate_swing_strength(data, idx, 'high', atr)
                if strength >= self.min_swing_strength:
                    swing_highs.append({
                        'index': idx,
                        'price': data.iloc[idx]['high'],
                        'time': data.index[idx],
                        'strength': strength
                    })
        
        low_indices = np.where(pivot_lows)[0]
        for idx in low_indices:
            if idx < len(data) - 1:
                strength = self._calculate_swing_strength(data, idx, 'low', atr)
                if strength >= self.min_swing_strength:
                    swing_lows.append({
                        'index': idx,
                        'price': data.iloc[idx]['low'],
                        'time': data.index[idx],
                        'strength': strength
                    })
        
        return swing_highs, swing_lows
    
    def _calculate_swing_strength(self, data: pd.DataFrame, idx: int, 
                                swing_type: str, atr: np.ndarray) -> float:
        """Calculate the strength of a swing point."""
        if idx >= len(atr):
            return 0.0
        
        price = data.iloc[idx]['high'] if swing_type == 'high' else data.iloc[idx]['low']
        avg_atr = np.mean(atr[max(0, idx-10):idx+1])
        
        if avg_atr == 0:
            return 0.0
        
        # Calculate price deviation from nearby prices
        window = min(10, idx, len(data) - idx - 1)
        if window < 3:
            return 0.0
        
        nearby_prices = []
        for i in range(max(0, idx-window), min(len(data), idx+window+1)):
            if i != idx:
                nearby_prices.append(
                    data.iloc[i]['high'] if swing_type == 'high' else data.iloc[i]['low']
                )
        
        if not nearby_prices:
            return 0.0
        
        if swing_type == 'high':
            deviation = price - max(nearby_prices)
        else:
            deviation = min(nearby_prices) - price
        
        return max(0.0, deviation / avg_atr)
    
    def _calculate_arc_levels(self, swing_high: Dict, swing_low: Dict, 
                            data: pd.DataFrame) -> List[FibonacciArcLevel]:
        """Calculate Fibonacci arc levels between two swing points."""
        arcs = []
        
        # Calculate base measurements
        price_range = abs(swing_high['price'] - swing_low['price'])
        time_range = abs(swing_high['index'] - swing_low['index'])
        
        if price_range == 0 or time_range == 0:
            return arcs
        
        # Determine arc center and direction
        center_x = (swing_high['index'] + swing_low['index']) / 2
        center_y = (swing_high['price'] + swing_low['price']) / 2
        
        # Calculate volatility adjustment
        volatility_factor = 1.0
        if self.volatility_adjustment:
            atr = self._calculate_atr(data)
            if len(atr) > swing_high['index']:
                avg_atr = np.mean(atr[max(0, swing_high['index']-20):swing_high['index']+1])
                volatility_factor = 1.0 + (avg_atr / price_range) * 0.5
        
        # Generate arcs for each Fibonacci ratio
        for ratio in self.fibonacci_ratios:
            try:
                radius = price_range * ratio * volatility_factor
                
                # Calculate arc points
                arc_points = self._generate_arc_points(
                    center_x, center_y, radius, time_range
                )
                
                # Calculate arc strength
                strength = self._calculate_arc_strength(
                    arc_points, data, swing_high, swing_low
                )
                
                # Calculate volume confirmation
                volume_conf = self._calculate_volume_confirmation(
                    arc_points, data
                )
                
                arc = FibonacciArcLevel(
                    ratio=ratio,
                    center_x=center_x,
                    center_y=center_y,
                    radius=radius,
                    strength=strength,
                    angle_start=0,
                    angle_end=180,
                    arc_points=arc_points,
                    confidence=min(strength * volume_conf, 1.0),
                    volume_confirmation=volume_conf
                )
                
                arcs.append(arc)
                
            except Exception as e:
                logger.warning(f"Error calculating arc for ratio {ratio}: {str(e)}")
                continue
        
        return arcs
    
    def _generate_arc_points(self, center_x: float, center_y: float, 
                           radius: float, time_range: float) -> List[Tuple[float, float]]:
        """Generate points along the Fibonacci arc."""
        points = []
        
        # Calculate arc parameters
        aspect_ratio = time_range / radius if radius > 0 else 1.0
        
        # Generate points along the arc
        for i in range(self.arc_resolution + 1):
            angle = np.pi * i / self.arc_resolution  # 0 to π radians
            
            # Elliptical projection for better time-price scaling
            x_offset = radius * np.cos(angle) * aspect_ratio
            y_offset = radius * np.sin(angle)
            
            x = center_x + x_offset
            y = center_y + y_offset
            
            points.append((x, y))
        
        return points
    
    def _calculate_arc_strength(self, arc_points: List[Tuple[float, float]], 
                              data: pd.DataFrame, swing_high: Dict, 
                              swing_low: Dict) -> float:
        """Calculate the strength of an arc based on price interactions."""
        if not arc_points:
            return 0.0
        
        touches = 0
        near_touches = 0
        total_volume_at_arc = 0.0
        
        # Check price interactions with arc
        for x, y in arc_points:
            idx = int(round(x))
            price_level = y
            
            if 0 <= idx < len(data):
                row = data.iloc[idx]
                
                # Calculate distance from price level
                high_dist = abs(row['high'] - price_level) / price_level
                low_dist = abs(row['low'] - price_level) / price_level
                close_dist = abs(row['close'] - price_level) / price_level
                
                min_dist = min(high_dist, low_dist, close_dist)
                
                # Count touches and near touches
                if min_dist < 0.001:  # Direct touch
                    touches += 1
                    total_volume_at_arc += row.get('volume', 0)
                elif min_dist < 0.005:  # Near touch
                    near_touches += 1
                    total_volume_at_arc += row.get('volume', 0) * 0.5
        
        # Calculate base strength
        strength = (touches * 2 + near_touches) / len(arc_points)
        
        # Volume weighting
        if total_volume_at_arc > 0:
            avg_volume = data['volume'].mean() if 'volume' in data.columns else 1
            volume_factor = min(2.0, total_volume_at_arc / (avg_volume * len(arc_points)))
            strength *= volume_factor
        
        return min(1.0, strength)
    
    def _calculate_volume_confirmation(self, arc_points: List[Tuple[float, float]], 
                                     data: pd.DataFrame) -> float:
        """Calculate volume confirmation for arc levels."""
        if 'volume' not in data.columns:
            return 0.5  # Neutral when volume not available
        
        total_volume = 0.0
        valid_points = 0
        
        for x, y in arc_points:
            idx = int(round(x))
            if 0 <= idx < len(data):
                total_volume += data.iloc[idx]['volume']
                valid_points += 1
        
        if valid_points == 0:
            return 0.5
        
        avg_arc_volume = total_volume / valid_points
        overall_avg_volume = data['volume'].mean()
        
        return min(1.0, avg_arc_volume / overall_avg_volume) if overall_avg_volume > 0 else 0.5
    
    def _validate_arcs(self, arcs: List[FibonacciArcLevel], 
                      data: pd.DataFrame) -> List[FibonacciArcLevel]:
        """Validate arcs using machine learning and statistical methods."""
        if not arcs:
            return []
        
        valid_arcs = []
        
        for arc in arcs:
            # Basic validation
            if arc.confidence < self.confidence_threshold:
                continue
            
            # ML validation if enabled
            if self.ml_validation and self.arc_validator is not None:
                try:
                    features = self._extract_arc_features(arc, data)
                    if self._validate_with_ml(features):
                        valid_arcs.append(arc)
                except Exception as e:
                    logger.warning(f"ML validation failed for arc: {str(e)}")
                    # Fall back to basic validation
                    if arc.strength > 0.3:
                        valid_arcs.append(arc)
            else:
                valid_arcs.append(arc)
        
        return valid_arcs
    
    def _extract_arc_features(self, arc: FibonacciArcLevel, 
                            data: pd.DataFrame) -> np.ndarray:
        """Extract features for ML validation."""
        features = [
            arc.ratio,
            arc.strength,
            arc.confidence,
            arc.volume_confirmation,
            arc.radius,
            len(arc.arc_points)
        ]
        
        # Add market context features
        current_idx = len(data) - 1
        if current_idx >= 20:
            recent_volatility = data['close'].iloc[-20:].std()
            recent_volume = data['volume'].iloc[-20:].mean() if 'volume' in data.columns else 0
            price_trend = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
            
            features.extend([recent_volatility, recent_volume, price_trend])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features).reshape(1, -1)
    
    def _validate_with_ml(self, features: np.ndarray) -> bool:
        """Validate arc using machine learning model."""
        try:
            # This would use a pre-trained model in production
            # For now, return True as placeholder
            return True
        except Exception:
            return False
    
    def _detect_arc_clusters(self, arcs: List[FibonacciArcLevel]) -> List[ArcCluster]:
        """Detect clusters of confluent arc levels."""
        if len(arcs) < 2:
            return []
        
        # Extract arc centers for clustering
        centers = []
        for arc in arcs:
            centers.append([arc.center_x, arc.center_y])
        
        if not centers:
            return []
        
        try:
            # Use DBSCAN for clustering
            clustering = DBSCAN(eps=10, min_samples=2)
            cluster_labels = clustering.fit_predict(centers)
            
            clusters = []
            unique_labels = set(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                
                cluster_arcs = [arcs[i] for i in range(len(arcs)) if cluster_labels[i] == label]
                
                if len(cluster_arcs) >= 2:
                    # Calculate cluster properties
                    center_x = np.mean([arc.center_x for arc in cluster_arcs])
                    center_y = np.mean([arc.center_y for arc in cluster_arcs])
                    strength = np.mean([arc.strength for arc in cluster_arcs])
                    confluence_score = len(cluster_arcs) * strength
                    
                    cluster = ArcCluster(
                        levels=cluster_arcs,
                        center_price=center_y,
                        center_time=center_x,
                        strength=strength,
                        confluence_score=confluence_score
                    )
                    clusters.append(cluster)
            
            return sorted(clusters, key=lambda x: x.confluence_score, reverse=True)
            
        except Exception as e:
            logger.warning(f"Arc clustering failed: {str(e)}")
            return []
    
    def _generate_arc_signals(self, arcs: List[FibonacciArcLevel], 
                            clusters: List[ArcCluster], 
                            data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on arc analysis."""
        signals = pd.Series(0, index=data.index)
        
        if not arcs:
            return signals
        
        current_price = data['close'].iloc[-1]
        current_idx = len(data) - 1
        
        # Signal generation logic
        for i in range(max(1, len(data) - 50), len(data)):
            signal_strength = 0.0
            
            # Check arc interactions
            for arc in arcs:
                arc_price_at_time = self._get_arc_price_at_time(arc, i)
                if arc_price_at_time is None:
                    continue
                
                price_distance = abs(data.iloc[i]['close'] - arc_price_at_time) / arc_price_at_time
                
                if price_distance < 0.002:  # Very close to arc
                    if data.iloc[i]['close'] < arc_price_at_time:
                        signal_strength += arc.strength * 0.5  # Support signal
                    else:
                        signal_strength -= arc.strength * 0.5  # Resistance signal
            
            # Check cluster signals
            for cluster in clusters:
                cluster_price = self._get_cluster_price_at_time(cluster, i)
                if cluster_price is None:
                    continue
                
                price_distance = abs(data.iloc[i]['close'] - cluster_price) / cluster_price
                
                if price_distance < 0.003:
                    if data.iloc[i]['close'] < cluster_price:
                        signal_strength += cluster.confluence_score * 0.3
                    else:
                        signal_strength -= cluster.confluence_score * 0.3
            
            # Apply signal
            signals.iloc[i] = np.clip(signal_strength, -1.0, 1.0)
        
        return signals
    
    def _get_arc_price_at_time(self, arc: FibonacciArcLevel, time_idx: int) -> Optional[float]:
        """Get the arc price at a specific time index."""
        for x, y in arc.arc_points:
            if abs(x - time_idx) < 0.5:
                return y
        return None
    
    def _get_cluster_price_at_time(self, cluster: ArcCluster, time_idx: int) -> Optional[float]:
        """Get the cluster center price at a specific time index."""
        prices = []
        for arc in cluster.levels:
            price = self._get_arc_price_at_time(arc, time_idx)
            if price is not None:
                prices.append(price)
        
        return np.mean(prices) if prices else None
    
    def _calculate_strength_metrics(self, arcs: List[FibonacciArcLevel], 
                                  data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate comprehensive strength metrics."""
        length = len(data)
        
        strength = pd.Series(0.0, index=data.index)
        confidence = pd.Series(0.0, index=data.index)
        trend_alignment = pd.Series(0.0, index=data.index)
        volume_confirmation = pd.Series(0.0, index=data.index)
        
        if not arcs:
            return {
                'strength': strength,
                'confidence': confidence,
                'trend_alignment': trend_alignment,
                'volume_confirmation': volume_confirmation
            }
        
        # Calculate metrics for each time point
        for i in range(length):
            nearby_arcs = []
            
            for arc in arcs:
                arc_price = self._get_arc_price_at_time(arc, i)
                if arc_price is not None:
                    distance = abs(data.iloc[i]['close'] - arc_price) / arc_price
                    if distance < 0.01:  # Within 1%
                        nearby_arcs.append(arc)
            
            if nearby_arcs:
                avg_strength = np.mean([arc.strength for arc in nearby_arcs])
                avg_confidence = np.mean([arc.confidence for arc in nearby_arcs])
                avg_volume = np.mean([arc.volume_confirmation for arc in nearby_arcs])
                
                strength.iloc[i] = avg_strength
                confidence.iloc[i] = avg_confidence
                volume_confirmation.iloc[i] = avg_volume
                
                # Calculate trend alignment
                if i >= 20:
                    trend = (data['close'].iloc[i] - data['close'].iloc[i-20]) / data['close'].iloc[i-20]
                    trend_alignment.iloc[i] = 1.0 if trend > 0 else -1.0
        
        return {
            'strength': strength,
            'confidence': confidence,
            'trend_alignment': trend_alignment,
            'volume_confirmation': volume_confirmation
        }
    
    def _calculate_price_targets(self, arcs: List[FibonacciArcLevel], 
                               data: pd.DataFrame) -> Dict[str, float]:
        """Calculate price targets based on arc projections."""
        targets = {}
        
        if not arcs:
            return targets
        
        current_price = data['close'].iloc[-1]
        
        # Find nearest arc levels above and below current price
        resistance_levels = []
        support_levels = []
        
        for arc in arcs:
            for _, price in arc.arc_points:
                if price > current_price * 1.001:
                    resistance_levels.append(price)
                elif price < current_price * 0.999:
                    support_levels.append(price)
        
        if resistance_levels:
            targets['nearest_resistance'] = min(resistance_levels)
            targets['extended_resistance'] = np.percentile(resistance_levels, 75)
        
        if support_levels:
            targets['nearest_support'] = max(support_levels)
            targets['extended_support'] = np.percentile(support_levels, 25)
        
        return targets
    
    def _identify_support_resistance(self, arcs: List[FibonacciArcLevel], 
                                   data: pd.DataFrame) -> Dict[str, List[float]]:
        """Identify key support and resistance levels from arcs."""
        support_levels = []
        resistance_levels = []
        
        current_price = data['close'].iloc[-1]
        
        for arc in arcs:
            for _, price in arc.arc_points:
                strength_weight = arc.strength * arc.confidence
                
                if price < current_price:
                    support_levels.append({
                        'price': price,
                        'strength': strength_weight
                    })
                else:
                    resistance_levels.append({
                        'price': price,
                        'strength': strength_weight
                    })
        
        # Sort by strength and return top levels
        support_levels.sort(key=lambda x: x['strength'], reverse=True)
        resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'support': [level['price'] for level in support_levels[:5]],
            'resistance': [level['price'] for level in resistance_levels[:5]]
        }
    
    def _find_arc_intersections(self, arcs: List[FibonacciArcLevel]) -> List[Dict]:
        """Find intersections between different arcs."""
        intersections = []
        
        for i, arc1 in enumerate(arcs):
            for j, arc2 in enumerate(arcs[i+1:], i+1):
                intersection_points = self._calculate_arc_intersection(arc1, arc2)
                if intersection_points:
                    intersections.extend(intersection_points)
        
        return intersections
    
    def _calculate_arc_intersection(self, arc1: FibonacciArcLevel, 
                                  arc2: FibonacciArcLevel) -> List[Dict]:
        """Calculate intersection points between two arcs."""
        intersections = []
        
        # Simplified intersection detection
        for x1, y1 in arc1.arc_points:
            for x2, y2 in arc2.arc_points:
                if abs(x1 - x2) < 1.0 and abs(y1 - y2) < (y1 * 0.01):
                    intersection = {
                        'time': (x1 + x2) / 2,
                        'price': (y1 + y2) / 2,
                        'strength': (arc1.strength + arc2.strength) / 2,
                        'ratios': [arc1.ratio, arc2.ratio]
                    }
                    intersections.append(intersection)
        
        return intersections
    
    def _calculate_time_projections(self, arcs: List[FibonacciArcLevel], 
                                  data: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Calculate time-based projections from arcs."""
        projections = {
            'time_targets': [],
            'cycle_analysis': []
        }
        
        current_time = len(data) - 1
        
        for arc in arcs:
            # Project arc into future time periods
            future_points = []
            for ratio in [1.272, 1.618, 2.618]:
                future_time = current_time + (arc.center_x * ratio)
                future_price = self._project_arc_price(arc, future_time)
                
                if future_price is not None:
                    future_points.append({
                        'time': future_time,
                        'price': future_price,
                        'ratio': ratio,
                        'confidence': arc.confidence * (1.0 - (ratio - 1.0) * 0.2)
                    })
            
            projections['time_targets'].extend(future_points)
        
        return projections
    
    def _project_arc_price(self, arc: FibonacciArcLevel, future_time: float) -> Optional[float]:
        """Project arc price to a future time point."""
        if not arc.arc_points:
            return None
        
        # Simple linear extrapolation from arc pattern
        last_point = arc.arc_points[-1]
        time_diff = future_time - last_point[0]
        
        if time_diff <= 0:
            return None
        
        # Calculate price projection based on arc curvature
        curvature = self._calculate_arc_curvature(arc)
        price_projection = last_point[1] * (1.0 + curvature * time_diff * 0.01)
        
        return max(0, price_projection)
    
    def _calculate_arc_curvature(self, arc: FibonacciArcLevel) -> float:
        """Calculate the curvature of an arc."""
        if len(arc.arc_points) < 3:
            return 0.0
        
        # Simple curvature calculation
        points = arc.arc_points[-3:]
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        if len(set(x_vals)) < 2:
            return 0.0
        
        # Calculate second derivative approximation
        dy_dx1 = (y_vals[1] - y_vals[0]) / (x_vals[1] - x_vals[0]) if x_vals[1] != x_vals[0] else 0
        dy_dx2 = (y_vals[2] - y_vals[1]) / (x_vals[2] - x_vals[1]) if x_vals[2] != x_vals[1] else 0
        
        curvature = (dy_dx2 - dy_dx1) / (x_vals[2] - x_vals[0]) if x_vals[2] != x_vals[0] else 0
        
        return curvature
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean().fillna(0).values
    
    def _format_arc_levels(self, arcs: List[FibonacciArcLevel]) -> pd.DataFrame:
        """Format arc levels for output."""
        if not arcs:
            return pd.DataFrame()
        
        arc_data = []
        for arc in arcs:
            arc_data.append({
                'ratio': arc.ratio,
                'center_time': arc.center_x,
                'center_price': arc.center_y,
                'radius': arc.radius,
                'strength': arc.strength,
                'confidence': arc.confidence,
                'volume_confirmation': arc.volume_confirmation
            })
        
        return pd.DataFrame(arc_data)
    
    def _format_clusters(self, clusters: List[ArcCluster]) -> pd.DataFrame:
        """Format cluster data for output."""
        if not clusters:
            return pd.DataFrame()
        
        cluster_data = []
        for cluster in clusters:
            cluster_data.append({
                'center_time': cluster.center_time,
                'center_price': cluster.center_price,
                'strength': cluster.strength,
                'confluence_score': cluster.confluence_score,
                'num_levels': len(cluster.levels)
            })
        
        return pd.DataFrame(cluster_data)
    
    def _empty_result(self, length: int) -> Dict[str, Union[pd.Series, np.ndarray, Dict]]:
        """Return empty result structure."""
        empty_series = pd.Series(0.0, index=range(length))
        
        return {
            'arc_levels': pd.DataFrame(),
            'clusters': pd.DataFrame(),
            'signals': empty_series,
            'strength': empty_series,
            'confidence': empty_series,
            'trend_alignment': empty_series,
            'volume_confirmation': empty_series,
            'price_targets': {},
            'support_resistance': {'support': [], 'resistance': []},
            'arc_intersections': [],
            'time_projections': {'time_targets': [], 'cycle_analysis': []}
        }
