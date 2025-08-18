"""
Grid Line Indicator - AI Enhanced Category
==========================================

Advanced AI-enhanced Grid Line trading system with dynamic level calculation,
machine learning optimization, and multi-timeframe grid analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple, List
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import signal, stats, optimize
from scipy.spatial.distance import pdist, linkage
from scipy.cluster.hierarchy import fcluster
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class GridLineIndicator(StandardIndicatorInterface):
    """
    AI-Enhanced Grid Line system with advanced features.
    
    Features:
    - Dynamic grid level calculation using ML clustering
    - Support/resistance strength analysis
    - Volume-weighted grid importance
    - Fractal-based grid level detection
    - Time-decay adjusted grid relevance
    - Breakout probability prediction
    - Multi-timeframe grid confluence
    - Adaptive grid spacing optimization
    - Price action pattern recognition at grid levels
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'lookback_period': 100,          # Period for grid calculation
            'min_grid_levels': 5,            # Minimum number of grid levels
            'max_grid_levels': 20,           # Maximum number of grid levels
            'grid_spacing_method': 'adaptive',  # 'fixed', 'adaptive', 'ml_optimized'
            'support_resistance_strength': 3,   # Minimum touches for strong level
            'volume_weight_factor': 0.3,     # Volume weighting importance
            'time_decay_factor': 0.95,       # Time decay for older levels
            'fractal_periods': [5, 8, 13, 21], # Fractal detection periods
            'confluence_threshold': 0.02,    # Price proximity for confluence (%)
            'breakout_threshold': 1.5,       # ATR multiplier for breakout detection
            'ml_optimization': True,         # Enable ML optimization
            'multi_timeframe': True,         # Enable multi-timeframe analysis
            'pattern_recognition': True,     # Enable pattern recognition
            'dynamic_adjustment': True,      # Enable dynamic grid adjustment
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("GridLineIndicator", default_params)
        
        # Initialize ML models
        self.level_clusterer = KMeans(n_clusters=10, random_state=42, n_init=10)
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        self.strength_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Grid state tracking
        self.grid_levels = []
        self.level_strengths = {}
        self.level_timestamps = {}
        self.breakout_history = []
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=["high", "low", "close", "volume"],
            min_periods=self.parameters['lookback_period']
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced grid line system with AI enhancements."""
        try:
            if len(data) < self.get_data_requirements().min_periods:
                return self._get_default_output()
            
            # Extract price and volume data
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            # Calculate fractal levels
            fractal_levels = self._calculate_fractal_levels(high, low, close)
            
            # Calculate pivot points
            pivot_levels = self._calculate_pivot_levels(high, low, close)
            
            # Calculate volume-weighted levels
            volume_levels = self._calculate_volume_levels(high, low, close, volume)
            
            # Combine all potential levels
            all_levels = fractal_levels + pivot_levels + volume_levels
            
            # ML-based level clustering and optimization
            if self.parameters['ml_optimization'] and len(all_levels) > 0:
                optimized_levels = self._ml_optimize_levels(all_levels, close)
            else:
                optimized_levels = self._basic_level_selection(all_levels, close)
            
            # Calculate level strengths and properties
            level_analysis = self._analyze_level_strengths(
                optimized_levels, high, low, close, volume
            )
            
            # Update grid state
            self._update_grid_state(optimized_levels, level_analysis)
            
            # Calculate current price position in grid
            grid_position = self._calculate_grid_position(close[-1], optimized_levels)
            
            # Detect breakouts and key events
            breakout_analysis = self._analyze_breakouts(close, optimized_levels, volume)
            
            # Pattern recognition at grid levels
            patterns = {}
            if self.parameters['pattern_recognition']:
                patterns = self._detect_grid_patterns(
                    high, low, close, optimized_levels, level_analysis
                )
            
            # Multi-timeframe confluence
            confluence_analysis = {}
            if self.parameters['multi_timeframe']:
                confluence_analysis = self._analyze_multi_timeframe_confluence(
                    data, optimized_levels
                )
            
            # Generate trading signals
            signals = self._generate_grid_signals(
                close[-1], optimized_levels, level_analysis, 
                breakout_analysis, patterns
            )
            
            # Calculate support and resistance zones
            support_resistance = self._identify_support_resistance_zones(
                optimized_levels, level_analysis, close[-1]
            )
            
            # Risk management recommendations
            risk_management = self._calculate_risk_management(
                close[-1], optimized_levels, level_analysis, data
            )
            
            return {
                'grid_levels': optimized_levels,
                'level_analysis': level_analysis,
                'grid_position': grid_position,
                'breakout_analysis': breakout_analysis,
                'patterns': patterns,
                'confluence_analysis': confluence_analysis,
                'support_resistance': support_resistance,
                'signals': signals,
                'risk_management': risk_management,
                'current_price': close[-1],
                'nearest_support': support_resistance.get('nearest_support', close[-1]),
                'nearest_resistance': support_resistance.get('nearest_resistance', close[-1]),
                'grid_strength': level_analysis.get('overall_strength', 0.0),
                'breakout_probability': breakout_analysis.get('breakout_probability', 0.5)
            }
            
        except Exception as e:
            return self._handle_calculation_error(e)
    
    def _calculate_fractal_levels(self, high: np.ndarray, low: np.ndarray, 
                                close: np.ndarray) -> List[float]:
        """Calculate fractal-based grid levels."""
        levels = []
        
        for period in self.parameters['fractal_periods']:
            # High fractals
            for i in range(period, len(high) - period):
                is_fractal_high = True
                for j in range(i - period, i + period + 1):
                    if j != i and high[j] >= high[i]:
                        is_fractal_high = False
                        break
                
                if is_fractal_high:
                    levels.append(high[i])
            
            # Low fractals
            for i in range(period, len(low) - period):
                is_fractal_low = True
                for j in range(i - period, i + period + 1):
                    if j != i and low[j] <= low[i]:
                        is_fractal_low = False
                        break
                
                if is_fractal_low:
                    levels.append(low[i])
        
        return sorted(list(set(levels)))
    
    def _calculate_pivot_levels(self, high: np.ndarray, low: np.ndarray, 
                              close: np.ndarray) -> List[float]:
        """Calculate traditional and enhanced pivot levels."""
        levels = []
        
        # Calculate for different periods
        periods = [5, 10, 20, 50]
        
        for period in periods:
            if len(high) >= period:
                # Traditional pivot points
                period_high = np.max(high[-period:])
                period_low = np.min(low[-period:])
                period_close = close[-period]
                
                pivot = (period_high + period_low + period_close) / 3
                
                # Support and resistance levels
                r1 = 2 * pivot - period_low
                s1 = 2 * pivot - period_high
                r2 = pivot + (period_high - period_low)
                s2 = pivot - (period_high - period_low)
                r3 = period_high + 2 * (pivot - period_low)
                s3 = period_low - 2 * (period_high - pivot)
                
                levels.extend([pivot, r1, s1, r2, s2, r3, s3])
                
                # Fibonacci retracement levels
                price_range = period_high - period_low
                fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                
                for fib in fib_levels:
                    levels.append(period_low + price_range * fib)
                    levels.append(period_high - price_range * fib)
        
        return sorted(list(set(levels)))
    
    def _calculate_volume_levels(self, high: np.ndarray, low: np.ndarray, 
                               close: np.ndarray, volume: np.ndarray) -> List[float]:
        """Calculate volume-weighted important levels."""
        levels = []
        
        # Volume-weighted average price levels
        periods = [20, 50, 100]
        
        for period in periods:
            if len(close) >= period:
                # VWAP calculation
                typical_price = (high[-period:] + low[-period:] + close[-period:]) / 3
                volume_period = volume[-period:]
                
                if np.sum(volume_period) > 0:
                    vwap = np.sum(typical_price * volume_period) / np.sum(volume_period)
                    levels.append(vwap)
                
                # Volume profile levels (simplified)
                price_volume_pairs = list(zip(typical_price, volume_period))
                price_volume_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Top volume prices
                top_volume_levels = [pair[0] for pair in price_volume_pairs[:5]]
                levels.extend(top_volume_levels)
        
        return sorted(list(set(levels)))
    
    def _ml_optimize_levels(self, all_levels: List[float], close: np.ndarray) -> List[float]:
        """Use machine learning to optimize grid level selection."""
        if len(all_levels) < self.parameters['min_grid_levels']:
            return all_levels
        
        try:
            # Prepare data for clustering
            levels_array = np.array(all_levels).reshape(-1, 1)
            
            # Remove outliers
            outlier_mask = self.outlier_detector.fit_predict(levels_array) == 1
            filtered_levels = np.array(all_levels)[outlier_mask]
            
            if len(filtered_levels) < self.parameters['min_grid_levels']:
                filtered_levels = all_levels
            
            # Cluster levels to find optimal grid
            n_clusters = min(
                self.parameters['max_grid_levels'],
                max(self.parameters['min_grid_levels'], len(filtered_levels) // 3)
            )
            
            self.level_clusterer.n_clusters = n_clusters
            clusters = self.level_clusterer.fit_predict(
                np.array(filtered_levels).reshape(-1, 1)
            )
            
            # Get representative level from each cluster
            optimized_levels = []
            for cluster_id in range(n_clusters):
                cluster_levels = np.array(filtered_levels)[clusters == cluster_id]
                if len(cluster_levels) > 0:
                    # Use cluster center or most significant level
                    representative_level = np.median(cluster_levels)
                    optimized_levels.append(representative_level)
            
            return sorted(optimized_levels)
            
        except Exception:
            return self._basic_level_selection(all_levels, close)
    
    def _basic_level_selection(self, all_levels: List[float], close: np.ndarray) -> List[float]:
        """Basic level selection fallback method."""
        if len(all_levels) <= self.parameters['max_grid_levels']:
            return sorted(all_levels)
        
        # Select levels based on proximity to current price and distribution
        current_price = close[-1]
        price_range = np.max(close) - np.min(close)
        
        # Score levels based on multiple factors
        level_scores = []
        for level in all_levels:
            distance_score = 1.0 / (1.0 + abs(level - current_price) / price_range)
            level_scores.append((level, distance_score))
        
        # Sort by score and select top levels
        level_scores.sort(key=lambda x: x[1], reverse=True)
        selected_levels = [item[0] for item in level_scores[:self.parameters['max_grid_levels']]]
        
        return sorted(selected_levels)
    
    def _analyze_level_strengths(self, levels: List[float], high: np.ndarray, 
                               low: np.ndarray, close: np.ndarray, 
                               volume: np.ndarray) -> Dict[str, Any]:
        """Analyze the strength and properties of grid levels."""
        level_strengths = {}
        overall_strength = 0.0
        
        for level in levels:
            strength_metrics = self._calculate_single_level_strength(
                level, high, low, close, volume
            )
            level_strengths[level] = strength_metrics
            overall_strength += strength_metrics['combined_strength']
        
        if len(levels) > 0:
            overall_strength /= len(levels)
        
        return {
            'level_strengths': level_strengths,
            'overall_strength': overall_strength,
            'strongest_level': max(level_strengths.keys(), 
                                 key=lambda x: level_strengths[x]['combined_strength']) if level_strengths else None,
            'weakest_level': min(level_strengths.keys(), 
                               key=lambda x: level_strengths[x]['combined_strength']) if level_strengths else None
        }
    
    def _calculate_single_level_strength(self, level: float, high: np.ndarray, 
                                       low: np.ndarray, close: np.ndarray, 
                                       volume: np.ndarray) -> Dict[str, float]:
        """Calculate strength metrics for a single level."""
        tolerance = (np.max(close) - np.min(close)) * 0.001  # 0.1% tolerance
        
        # Count touches and reactions
        touches = 0
        reactions = 0
        volume_at_level = 0.0
        
        for i in range(len(close)):
            # Check if price touched the level
            if (low[i] <= level + tolerance and high[i] >= level - tolerance):
                touches += 1
                volume_at_level += volume[i]
                
                # Check for reaction (bounce or break)
                if i < len(close) - 1:
                    if abs(close[i+1] - level) > abs(close[i] - level):
                        reactions += 1
        
        # Time-weighted touches (recent touches are more important)
        time_weighted_touches = 0.0
        for i in range(len(close)):
            if (low[i] <= level + tolerance and high[i] >= level - tolerance):
                weight = self.parameters['time_decay_factor'] ** (len(close) - i - 1)
                time_weighted_touches += weight
        
        # Volume strength
        avg_volume = np.mean(volume) if len(volume) > 0 else 1.0
        volume_strength = volume_at_level / (avg_volume * max(touches, 1))
        
        # Reaction ratio
        reaction_ratio = reactions / max(touches, 1)
        
        # Combined strength score
        touch_strength = min(touches / 10.0, 1.0)  # Normalize to 0-1
        time_strength = min(time_weighted_touches / 5.0, 1.0)
        vol_strength = min(volume_strength, 1.0)
        react_strength = reaction_ratio
        
        combined_strength = (
            touch_strength * 0.3 +
            time_strength * 0.25 +
            vol_strength * self.parameters['volume_weight_factor'] +
            react_strength * 0.25
        )
        
        return {
            'touches': touches,
            'reactions': reactions,
            'volume_at_level': volume_at_level,
            'time_weighted_touches': time_weighted_touches,
            'volume_strength': vol_strength,
            'reaction_ratio': reaction_ratio,
            'combined_strength': combined_strength
        }
    
    def _update_grid_state(self, levels: List[float], analysis: Dict[str, Any]):
        """Update internal grid state tracking."""
        self.grid_levels = levels
        self.level_strengths = analysis['level_strengths']
        
        # Update timestamps for new levels
        current_time = len(self.grid_levels)  # Simplified timestamp
        for level in levels:
            if level not in self.level_timestamps:
                self.level_timestamps[level] = current_time
    
    def _calculate_grid_position(self, current_price: float, levels: List[float]) -> Dict[str, Any]:
        """Calculate current price position within the grid."""
        if not levels:
            return {'position': 'undefined', 'nearest_levels': []}
        
        # Find nearest levels
        distances = [(abs(level - current_price), level) for level in levels]
        distances.sort()
        
        nearest_levels = [item[1] for item in distances[:3]]
        nearest_distance = distances[0][0] if distances else float('inf')
        
        # Determine position
        levels_above = [l for l in levels if l > current_price]
        levels_below = [l for l in levels if l < current_price]
        
        if levels_above and levels_below:
            position = 'between_levels'
        elif levels_above:
            position = 'below_grid'
        elif levels_below:
            position = 'above_grid'
        else:
            position = 'at_level'
        
        # Calculate grid percentage position
        if levels:
            min_level = min(levels)
            max_level = max(levels)
            if max_level > min_level:
                grid_percentage = (current_price - min_level) / (max_level - min_level)
            else:
                grid_percentage = 0.5
        else:
            grid_percentage = 0.5
        
        return {
            'position': position,
            'nearest_levels': nearest_levels,
            'nearest_distance': nearest_distance,
            'grid_percentage': grid_percentage,
            'levels_above': sorted(levels_above)[:3],
            'levels_below': sorted(levels_below, reverse=True)[:3]
        }    
    def _analyze_breakouts(self, close: np.ndarray, levels: List[float], 
                          volume: np.ndarray) -> Dict[str, Any]:
        """Analyze breakout patterns and probability."""
        if not levels or len(close) < 10:
            return {'breakout_probability': 0.5, 'recent_breakouts': []}
        
        # Calculate ATR for breakout threshold
        atr = self._calculate_atr(close, window=14)
        breakout_threshold = atr * self.parameters['breakout_threshold']
        
        # Detect recent breakouts
        recent_breakouts = []
        lookback = min(20, len(close))
        
        for i in range(len(close) - lookback, len(close)):
            current_price = close[i]
            
            for level in levels:
                if abs(current_price - level) < breakout_threshold:
                    # Check if this was a breakout
                    prev_prices = close[max(0, i-5):i]
                    if len(prev_prices) > 0:
                        if level > np.max(prev_prices) and current_price > level:
                            recent_breakouts.append({
                                'type': 'upward',
                                'level': level,
                                'price': current_price,
                                'index': i
                            })
                        elif level < np.min(prev_prices) and current_price < level:
                            recent_breakouts.append({
                                'type': 'downward',
                                'level': level,
                                'price': current_price,
                                'index': i
                            })
        
        # Calculate breakout probability based on current position
        current_price = close[-1]
        nearest_level = min(levels, key=lambda x: abs(x - current_price)) if levels else current_price
        distance_to_level = abs(current_price - nearest_level)
        
        # Probability increases as price approaches level
        if distance_to_level < breakout_threshold:
            proximity_factor = 1.0 - (distance_to_level / breakout_threshold)
        else:
            proximity_factor = 0.0
        
        # Volume factor
        recent_volume = np.mean(volume[-5:]) if len(volume) >= 5 else 0
        avg_volume = np.mean(volume) if len(volume) > 0 else 1
        volume_factor = min(recent_volume / avg_volume, 2.0) / 2.0
        
        breakout_probability = (proximity_factor * 0.6 + volume_factor * 0.4)
        
        return {
            'breakout_probability': breakout_probability,
            'recent_breakouts': recent_breakouts[-5:],  # Last 5 breakouts
            'nearest_level': nearest_level,
            'distance_to_level': distance_to_level,
            'breakout_threshold': breakout_threshold
        }
    
    def _detect_grid_patterns(self, high: np.ndarray, low: np.ndarray, 
                            close: np.ndarray, levels: List[float], 
                            analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect patterns around grid levels."""
        patterns = {
            'false_breakouts': [],
            'strong_bounces': [],
            'level_clusters': [],
            'trend_patterns': {}
        }
        
        if not levels or len(close) < 20:
            return patterns
        
        # Detect false breakouts
        for level in levels:
            false_breakouts = self._detect_false_breakouts(high, low, close, level)
            if false_breakouts:
                patterns['false_breakouts'].extend(false_breakouts)
        
        # Detect strong bounces
        for level in levels:
            bounces = self._detect_strong_bounces(high, low, close, level)
            if bounces:
                patterns['strong_bounces'].extend(bounces)
        
        # Detect level clusters
        patterns['level_clusters'] = self._detect_level_clusters(levels)
        
        # Trend patterns
        patterns['trend_patterns'] = self._analyze_trend_patterns(close, levels)
        
        return patterns
    
    def _detect_false_breakouts(self, high: np.ndarray, low: np.ndarray, 
                              close: np.ndarray, level: float) -> List[Dict[str, Any]]:
        """Detect false breakout patterns at a specific level."""
        false_breakouts = []
        tolerance = (np.max(close) - np.min(close)) * 0.002  # 0.2% tolerance
        
        for i in range(10, len(close) - 5):  # Need lookback and lookahead
            # Check for breakout
            if high[i] > level + tolerance:
                # Check if it was false (price returned below level)
                future_closes = close[i+1:i+6]
                if len(future_closes) > 0 and all(c < level for c in future_closes):
                    false_breakouts.append({
                        'direction': 'upward',
                        'level': level,
                        'breakout_price': high[i],
                        'return_price': np.mean(future_closes),
                        'index': i
                    })
            
            elif low[i] < level - tolerance:
                # Check if it was false (price returned above level)
                future_closes = close[i+1:i+6]
                if len(future_closes) > 0 and all(c > level for c in future_closes):
                    false_breakouts.append({
                        'direction': 'downward',
                        'level': level,
                        'breakout_price': low[i],
                        'return_price': np.mean(future_closes),
                        'index': i
                    })
        
        return false_breakouts
    
    def _detect_strong_bounces(self, high: np.ndarray, low: np.ndarray, 
                             close: np.ndarray, level: float) -> List[Dict[str, Any]]:
        """Detect strong bounce patterns at a specific level."""
        bounces = []
        tolerance = (np.max(close) - np.min(close)) * 0.001  # 0.1% tolerance
        
        for i in range(5, len(close) - 5):
            # Check for touch and bounce
            if abs(low[i] - level) < tolerance:
                # Check for upward bounce
                future_highs = high[i+1:i+6]
                if len(future_highs) > 0:
                    bounce_strength = (np.max(future_highs) - level) / level
                    if bounce_strength > 0.01:  # 1% bounce
                        bounces.append({
                            'direction': 'upward',
                            'level': level,
                            'touch_price': low[i],
                            'bounce_high': np.max(future_highs),
                            'strength': bounce_strength,
                            'index': i
                        })
            
            elif abs(high[i] - level) < tolerance:
                # Check for downward bounce
                future_lows = low[i+1:i+6]
                if len(future_lows) > 0:
                    bounce_strength = (level - np.min(future_lows)) / level
                    if bounce_strength > 0.01:  # 1% bounce
                        bounces.append({
                            'direction': 'downward',
                            'level': level,
                            'touch_price': high[i],
                            'bounce_low': np.min(future_lows),
                            'strength': bounce_strength,
                            'index': i
                        })
        
        return bounces
    
    def _detect_level_clusters(self, levels: List[float]) -> List[Dict[str, Any]]:
        """Detect clusters of closely spaced levels."""
        if len(levels) < 3:
            return []
        
        clusters = []
        price_range = max(levels) - min(levels)
        cluster_threshold = price_range * self.parameters['confluence_threshold']
        
        # Use hierarchical clustering
        try:
            levels_array = np.array(levels).reshape(-1, 1)
            distances = pdist(levels_array)
            linkage_matrix = linkage(distances, method='ward')
            cluster_labels = fcluster(linkage_matrix, cluster_threshold, criterion='distance')
            
            # Group levels by cluster
            cluster_dict = {}
            for i, label in enumerate(cluster_labels):
                if label not in cluster_dict:
                    cluster_dict[label] = []
                cluster_dict[label].append(levels[i])
            
            # Create cluster information
            for cluster_id, cluster_levels in cluster_dict.items():
                if len(cluster_levels) >= 2:
                    clusters.append({
                        'center': np.mean(cluster_levels),
                        'levels': cluster_levels,
                        'strength': len(cluster_levels),
                        'range': max(cluster_levels) - min(cluster_levels)
                    })
        
        except Exception:
            pass
        
        return clusters
    
    def _analyze_trend_patterns(self, close: np.ndarray, levels: List[float]) -> Dict[str, Any]:
        """Analyze trend patterns in relation to grid levels."""
        if not levels or len(close) < 20:
            return {}
        
        current_price = close[-1]
        price_change = close[-1] - close[-20] if len(close) >= 20 else 0
        
        # Analyze which levels are acting as support/resistance
        support_levels = [l for l in levels if l < current_price]
        resistance_levels = [l for l in levels if l > current_price]
        
        # Trend strength relative to grid
        if price_change > 0:
            trend = 'bullish'
            broken_resistance = len([l for l in levels if close[-20] < l < close[-1]]) if len(close) >= 20 else 0
        elif price_change < 0:
            trend = 'bearish'
            broken_support = len([l for l in levels if close[-1] < l < close[-20]]) if len(close) >= 20 else 0
        else:
            trend = 'sideways'
            broken_resistance = broken_support = 0
        
        return {
            'trend_direction': trend,
            'price_change': price_change,
            'support_levels_count': len(support_levels),
            'resistance_levels_count': len(resistance_levels),
            'levels_broken': broken_resistance if trend == 'bullish' else (broken_support if trend == 'bearish' else 0)
        }
    
    def _analyze_multi_timeframe_confluence(self, data: pd.DataFrame, 
                                          levels: List[float]) -> Dict[str, Any]:
        """Analyze confluence across multiple timeframes."""
        confluence_analysis = {
            'confluent_levels': [],
            'timeframe_agreement': 0.0
        }
        
        if not self.parameters['multi_timeframe'] or len(data) < 100:
            return confluence_analysis
        
        # Analyze shorter timeframes (simplified)
        timeframes = [20, 50, 100]  # Different period lookbacks
        all_timeframe_levels = []
        
        for tf in timeframes:
            if len(data) >= tf:
                tf_data = data.tail(tf)
                tf_high = tf_data['high'].values
                tf_low = tf_data['low'].values
                tf_close = tf_data['close'].values
                
                # Calculate levels for this timeframe
                tf_levels = self._calculate_fractal_levels(tf_high, tf_low, tf_close)
                all_timeframe_levels.append(tf_levels)
        
        # Find confluent levels
        confluence_threshold = (max(levels) - min(levels)) * self.parameters['confluence_threshold']
        
        for level in levels:
            confluence_count = 0
            for tf_levels in all_timeframe_levels:
                if any(abs(tf_level - level) < confluence_threshold for tf_level in tf_levels):
                    confluence_count += 1
            
            if confluence_count >= 2:  # At least 2 timeframes agree
                confluence_analysis['confluent_levels'].append({
                    'level': level,
                    'confluence_count': confluence_count,
                    'strength': confluence_count / len(all_timeframe_levels)
                })
        
        # Calculate overall timeframe agreement
        if all_timeframe_levels:
            confluence_analysis['timeframe_agreement'] = len(confluence_analysis['confluent_levels']) / len(levels) if levels else 0.0
        
        return confluence_analysis
    
    def _generate_grid_signals(self, current_price: float, levels: List[float], 
                             analysis: Dict[str, Any], breakout_analysis: Dict[str, Any],
                             patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on grid analysis."""
        signals = {
            'direction': 'neutral',
            'strength': 0.0,
            'confidence': 0.0,
            'action': 'hold',
            'entry_levels': [],
            'exit_levels': []
        }
        
        if not levels:
            return signals
        
        # Find nearest support and resistance
        support_levels = [l for l in levels if l < current_price]
        resistance_levels = [l for l in levels if l > current_price]
        
        nearest_support = max(support_levels) if support_levels else None
        nearest_resistance = min(resistance_levels) if resistance_levels else None
        
        # Generate signals based on position and patterns
        breakout_prob = breakout_analysis.get('breakout_probability', 0.5)
        
        if nearest_resistance and breakout_prob > 0.7:
            signals['direction'] = 'bullish'
            signals['action'] = 'buy'
            signals['strength'] = breakout_prob
            signals['entry_levels'] = [current_price]
            signals['exit_levels'] = resistance_levels[:2]
        
        elif nearest_support and breakout_prob > 0.7:
            signals['direction'] = 'bearish'
            signals['action'] = 'sell'
            signals['strength'] = breakout_prob
            signals['entry_levels'] = [current_price]
            signals['exit_levels'] = support_levels[-2:] if len(support_levels) >= 2 else support_levels
        
        elif nearest_support and nearest_resistance:
            # Range trading signals
            support_distance = abs(current_price - nearest_support)
            resistance_distance = abs(current_price - nearest_resistance)
            
            if support_distance < resistance_distance:
                signals['direction'] = 'bullish'
                signals['action'] = 'buy'
                signals['strength'] = 0.6
                signals['entry_levels'] = [nearest_support]
                signals['exit_levels'] = [nearest_resistance]
            else:
                signals['direction'] = 'bearish'
                signals['action'] = 'sell'
                signals['strength'] = 0.6
                signals['entry_levels'] = [nearest_resistance]
                signals['exit_levels'] = [nearest_support]
        
        # Adjust confidence based on level strength
        level_strengths = analysis.get('level_strengths', {})
        relevant_levels = [nearest_support, nearest_resistance]
        relevant_levels = [l for l in relevant_levels if l is not None]
        
        if relevant_levels:
            avg_strength = np.mean([level_strengths.get(l, {}).get('combined_strength', 0.5) 
                                   for l in relevant_levels])
            signals['confidence'] = avg_strength
        else:
            signals['confidence'] = 0.5
        
        return signals
    
    def _identify_support_resistance_zones(self, levels: List[float], 
                                         analysis: Dict[str, Any], 
                                         current_price: float) -> Dict[str, Any]:
        """Identify key support and resistance zones."""
        support_resistance = {
            'nearest_support': None,
            'nearest_resistance': None,
            'support_zone': [],
            'resistance_zone': [],
            'key_levels': []
        }
        
        if not levels:
            return support_resistance
        
        # Separate support and resistance levels
        support_levels = sorted([l for l in levels if l < current_price], reverse=True)
        resistance_levels = sorted([l for l in levels if l > current_price])
        
        # Find nearest levels
        support_resistance['nearest_support'] = support_levels[0] if support_levels else None
        support_resistance['nearest_resistance'] = resistance_levels[0] if resistance_levels else None
        
        # Create zones (groups of nearby levels)
        level_strengths = analysis.get('level_strengths', {})
        
        # Support zones
        support_resistance['support_zone'] = support_levels[:3]  # Top 3 support levels
        
        # Resistance zones
        support_resistance['resistance_zone'] = resistance_levels[:3]  # Top 3 resistance levels
        
        # Key levels (strongest levels)
        key_levels = []
        for level in levels:
            strength = level_strengths.get(level, {}).get('combined_strength', 0.0)
            if strength > 0.7:  # Strong levels only
                key_levels.append({
                    'level': level,
                    'strength': strength,
                    'type': 'support' if level < current_price else 'resistance'
                })
        
        key_levels.sort(key=lambda x: x['strength'], reverse=True)
        support_resistance['key_levels'] = key_levels[:5]  # Top 5 key levels
        
        return support_resistance
    
    def _calculate_risk_management(self, current_price: float, levels: List[float], 
                                 analysis: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk management recommendations."""
        risk_mgmt = {
            'stop_loss_levels': [],
            'take_profit_levels': [],
            'position_sizing': 'medium',
            'risk_reward_ratio': 1.0
        }
        
        if not levels or len(data) < 20:
            return risk_mgmt
        
        # Calculate ATR for stop loss
        close = data['close'].values
        atr = self._calculate_atr(close, window=14)
        
        # Find support and resistance for risk management
        support_levels = [l for l in levels if l < current_price]
        resistance_levels = [l for l in levels if l > current_price]
        
        nearest_support = max(support_levels) if support_levels else current_price - atr * 2
        nearest_resistance = min(resistance_levels) if resistance_levels else current_price + atr * 2
        
        # Stop loss recommendations
        risk_mgmt['stop_loss_levels'] = [
            nearest_support - atr * 0.5,  # Conservative
            nearest_support - atr * 1.0,  # Normal
            nearest_support - atr * 1.5   # Aggressive
        ]
        
        # Take profit recommendations
        risk_mgmt['take_profit_levels'] = [
            nearest_resistance - atr * 0.5,  # Conservative
            nearest_resistance,               # Normal
            nearest_resistance + atr * 0.5   # Aggressive
        ]
        
        # Position sizing based on level strength
        level_strengths = analysis.get('level_strengths', {})
        avg_strength = 0.5
        
        if support_levels:
            support_strength = level_strengths.get(nearest_support, {}).get('combined_strength', 0.5)
            avg_strength = support_strength
        
        if avg_strength > 0.8:
            risk_mgmt['position_sizing'] = 'large'
        elif avg_strength > 0.6:
            risk_mgmt['position_sizing'] = 'medium'
        else:
            risk_mgmt['position_sizing'] = 'small'
        
        # Risk-reward ratio
        stop_distance = abs(current_price - nearest_support)
        profit_distance = abs(nearest_resistance - current_price)
        
        if stop_distance > 0:
            risk_mgmt['risk_reward_ratio'] = profit_distance / stop_distance
        
        return risk_mgmt
    
    def _calculate_atr(self, close: np.ndarray, window: int = 14) -> float:
        """Calculate Average True Range."""
        if len(close) < window + 1:
            return np.std(close) if len(close) > 1 else 0.0
        
        # Simplified ATR calculation using close prices
        returns = np.abs(np.diff(close))
        atr = np.mean(returns[-window:]) if len(returns) >= window else np.mean(returns)
        
        return atr
    
    def _get_default_output(self) -> Dict[str, Any]:
        """Return default output when insufficient data."""
        return {
            'grid_levels': [],
            'level_analysis': {'overall_strength': 0.0},
            'grid_position': {'position': 'undefined'},
            'breakout_analysis': {'breakout_probability': 0.5},
            'patterns': {},
            'confluence_analysis': {},
            'support_resistance': {},
            'signals': {'direction': 'neutral', 'strength': 0.0, 'confidence': 0.0},
            'risk_management': {},
            'current_price': 0.0,
            'nearest_support': 0.0,
            'nearest_resistance': 0.0,
            'grid_strength': 0.0,
            'breakout_probability': 0.5
        }
    
    def _handle_calculation_error(self, error: Exception) -> Dict[str, Any]:
        """Handle calculation errors gracefully."""
        return self._get_default_output()