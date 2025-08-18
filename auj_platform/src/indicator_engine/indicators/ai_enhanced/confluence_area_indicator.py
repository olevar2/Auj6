"""
Confluence Area Detector - AI Enhanced Category
===============================================

Advanced confluence area detection using multi-indicator analysis, geometric
clustering, machine learning validation, and sophisticated signal strength scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, Tuple, List
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats, spatial
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

from ..base.standard_indicator import StandardIndicatorInterface, DataRequirement, DataType, SignalType


class ConfluenceAreaDetector(StandardIndicatorInterface):
    """
    Advanced Confluence Area Detector using multi-indicator convergence analysis.
    
    Features:
    - Multi-indicator level detection and clustering
    - Geometric confluence area identification
    - Machine learning-based confluence validation
    - Dynamic strength scoring and weighting
    - Time-based confluence persistence analysis
    - Breakout prediction from confluence zones
    - Support/resistance strength quantification
    - Multi-timeframe confluence mapping
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'lookback_period': 100,          # Period for confluence analysis
            'clustering_tolerance': 0.01,    # Price clustering tolerance (1%)
            'min_confluence_points': 3,      # Minimum points for confluence
            'strength_threshold': 0.6,       # Minimum strength for significant confluence
            'timeframe_windows': [20, 50, 100], # Multiple timeframe windows
            'indicator_types': [             # Types of levels to consider
                'pivot_points',
                'moving_averages', 
                'fibonacci_levels',
                'support_resistance',
                'volume_levels'
            ],
            'geometric_analysis': True,      # Enable geometric confluence analysis
            'machine_learning': True,        # Enable ML validation
            'persistence_analysis': True,    # Enable time persistence analysis
            'breakout_prediction': True,     # Enable breakout prediction
            'multi_timeframe': True,         # Enable multi-timeframe analysis
            'dynamic_weighting': True,       # Enable dynamic indicator weighting
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("ConfluenceAreaDetector", default_params)
        
        # Initialize ML models
        self.confluence_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.strength_predictor = RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Initialize clustering
        self.price_clusterer = DBSCAN(
            eps=default_params['clustering_tolerance'],
            min_samples=default_params['min_confluence_points']
        )
        
        # Cache for calculations
        self._confluence_cache = {}
        self._historical_confluences = []
    
    def get_data_requirements(self) -> DataRequirement:
        return DataRequirement(
            data_type=DataType.OHLCV,
            required_columns=["open", "high", "low", "close", "volume"],
            min_periods=max(self.parameters['timeframe_windows']) + 50
        )
    
    def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate confluence areas with advanced multi-indicator analysis."""
        try:
            if len(data) < self.get_data_requirements().min_periods:
                return self._get_default_output()
            
            # Extract indicator levels from multiple sources
            indicator_levels = self._extract_indicator_levels(data)
            
            # Detect confluence areas using clustering
            confluence_areas = self._detect_confluence_areas(indicator_levels, data)
            
            # Perform geometric analysis
            geometric_analysis = {}
            if self.parameters['geometric_analysis']:
                geometric_analysis = self._perform_geometric_analysis(confluence_areas, data)
            
            # Calculate confluence strength scores
            strength_analysis = self._calculate_confluence_strengths(
                confluence_areas, indicator_levels, data
            )
            
            # Multi-timeframe analysis
            mtf_analysis = {}
            if self.parameters['multi_timeframe']:
                mtf_analysis = self._analyze_multi_timeframe_confluence(data)
            
            # Persistence analysis
            persistence_analysis = {}
            if self.parameters['persistence_analysis']:
                persistence_analysis = self._analyze_confluence_persistence(
                    confluence_areas, strength_analysis
                )
            
            # Machine learning validation
            ml_validation = {}
            if self.parameters['machine_learning']:
                ml_validation = self._validate_with_machine_learning(
                    confluence_areas, strength_analysis, data
                )
            
            # Breakout prediction
            breakout_predictions = {}
            if self.parameters['breakout_prediction']:
                breakout_predictions = self._predict_breakouts_from_confluence(
                    confluence_areas, strength_analysis, data
                )
            
            # Generate trading signals
            signals = self._generate_trading_signals(
                confluence_areas, strength_analysis, ml_validation, 
                breakout_predictions, data
            )
            
            # Calculate overall confidence
            confidence = self._calculate_confidence_score(
                confluence_areas, strength_analysis, ml_validation, persistence_analysis
            )
            
            return {
                'confluence_areas': confluence_areas,
                'indicator_levels': indicator_levels,
                'geometric_analysis': geometric_analysis,
                'strength_analysis': strength_analysis,
                'multi_timeframe_analysis': mtf_analysis,
                'persistence_analysis': persistence_analysis,
                'ml_validation': ml_validation,
                'breakout_predictions': breakout_predictions,
                'signals': signals,
                'confidence': confidence,
                'strongest_confluence': self._get_strongest_confluence(confluence_areas, strength_analysis),
                'confluence_count': len(confluence_areas),
                'max_strength': max([s.get('total_strength', 0.0) for s in strength_analysis], default=0.0)
            }
            
        except Exception as e:
            return self._handle_calculation_error(e)
    
    def _extract_indicator_levels(self, data: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Extract levels from multiple technical indicators."""
        levels = {
            'pivot_points': [],
            'moving_averages': [],
            'fibonacci_levels': [],
            'support_resistance': [],
            'volume_levels': []
        }
        
        try:
            current_price = data['close'].iloc[-1]
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            # Pivot Points
            if 'pivot_points' in self.parameters['indicator_types']:
                levels['pivot_points'] = self._extract_pivot_points(high, low, close)
            
            # Moving Averages
            if 'moving_averages' in self.parameters['indicator_types']:
                levels['moving_averages'] = self._extract_moving_average_levels(data)
            
            # Fibonacci Levels
            if 'fibonacci_levels' in self.parameters['indicator_types']:
                levels['fibonacci_levels'] = self._extract_fibonacci_levels(data)
            
            # Support/Resistance
            if 'support_resistance' in self.parameters['indicator_types']:
                levels['support_resistance'] = self._extract_support_resistance_levels(data)
            
            # Volume Levels
            if 'volume_levels' in self.parameters['indicator_types']:
                levels['volume_levels'] = self._extract_volume_levels(data)
            
        except Exception:
            pass
        
        return levels
    
    def _extract_pivot_points(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> List[Dict[str, Any]]:
        """Extract pivot point levels."""
        pivots = []
        
        try:
            window = 5  # Pivot detection window
            
            for i in range(window, len(high) - window):
                # Pivot high
                if all(high[i] >= high[i-j] for j in range(1, window+1)) and \
                   all(high[i] >= high[i+j] for j in range(1, window+1)):
                    pivots.append({
                        'price': high[i],
                        'type': 'resistance',
                        'strength': self._calculate_pivot_strength(high, i, window),
                        'age': len(high) - i,
                        'source': 'pivot_high'
                    })
                
                # Pivot low
                if all(low[i] <= low[i-j] for j in range(1, window+1)) and \
                   all(low[i] <= low[i+j] for j in range(1, window+1)):
                    pivots.append({
                        'price': low[i],
                        'type': 'support',
                        'strength': self._calculate_pivot_strength(low, i, window),
                        'age': len(low) - i,
                        'source': 'pivot_low'
                    })
            
        except Exception:
            pass
        
        return pivots
    
    def _calculate_pivot_strength(self, prices: np.ndarray, index: int, window: int) -> float:
        """Calculate strength of a pivot point."""
        try:
            # Strength based on how much it stands out from surrounding prices
            pivot_price = prices[index]
            surrounding_prices = np.concatenate([
                prices[max(0, index-window):index],
                prices[index+1:min(len(prices), index+window+1)]
            ])
            
            if len(surrounding_prices) > 0:
                price_range = np.max(surrounding_prices) - np.min(surrounding_prices)
                if price_range > 0:
                    strength = abs(pivot_price - np.mean(surrounding_prices)) / price_range
                    return min(strength, 1.0)
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _extract_moving_average_levels(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract moving average levels."""
        ma_levels = []
        
        try:
            periods = [10, 20, 50, 100, 200]
            current_price = data['close'].iloc[-1]
            
            for period in periods:
                if len(data) >= period:
                    ma_value = data['close'].rolling(window=period).mean().iloc[-1]
                    
                    # Determine if MA is acting as support or resistance
                    recent_closes = data['close'].tail(5).values
                    above_ma = np.mean(recent_closes > ma_value)
                    
                    level_type = 'support' if above_ma > 0.6 else 'resistance' if above_ma < 0.4 else 'neutral'
                    
                    # Calculate strength based on recent touches
                    strength = self._calculate_ma_strength(data, ma_value, period)
                    
                    ma_levels.append({
                        'price': ma_value,
                        'type': level_type,
                        'strength': strength,
                        'age': 0,  # Current MA
                        'source': f'ma_{period}'
                    })
            
        except Exception:
            pass
        
        return ma_levels
    
    def _calculate_ma_strength(self, data: pd.DataFrame, ma_value: float, period: int) -> float:
        """Calculate strength of moving average as support/resistance."""
        try:
            # Count recent touches and bounces
            recent_data = data.tail(period).copy()
            touches = 0
            bounces = 0
            
            for i in range(1, len(recent_data)):
                current_price = recent_data['close'].iloc[i]
                prev_price = recent_data['close'].iloc[i-1]
                
                # Check for touch (within 1% of MA)
                if abs(current_price - ma_value) / ma_value < 0.01:
                    touches += 1
                    
                    # Check for bounce
                    if i < len(recent_data) - 1:
                        next_price = recent_data['close'].iloc[i+1]
                        if (prev_price < ma_value and next_price > ma_value) or \
                           (prev_price > ma_value and next_price < ma_value):
                            bounces += 1
            
            # Strength based on touches and bounce ratio
            if touches > 0:
                bounce_ratio = bounces / touches
                strength = min((touches / 10.0) * bounce_ratio, 1.0)
            else:
                strength = 0.3  # Default strength
            
            return strength
            
        except Exception:
            return 0.3
    
    def _extract_fibonacci_levels(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract Fibonacci retracement levels."""
        fib_levels = []
        
        try:
            # Find recent swing high and low
            lookback = min(100, len(data))
            recent_data = data.tail(lookback)
            
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            
            # Standard Fibonacci ratios
            fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            for ratio in fib_ratios:
                # Retracement level
                fib_price = swing_high - (swing_high - swing_low) * ratio
                
                # Determine type based on current price position
                current_price = data['close'].iloc[-1]
                if current_price > fib_price:
                    level_type = 'support'
                else:
                    level_type = 'resistance'
                
                # Calculate strength based on historical respect
                strength = self._calculate_fib_strength(data, fib_price)
                
                fib_levels.append({
                    'price': fib_price,
                    'type': level_type,
                    'strength': strength,
                    'age': 0,
                    'source': f'fib_{ratio}'
                })
            
        except Exception:
            pass
        
        return fib_levels
    
    def _calculate_fib_strength(self, data: pd.DataFrame, fib_price: float) -> float:
        """Calculate strength of Fibonacci level."""
        try:
            # Count how often price respected this level
            respect_count = 0
            total_approaches = 0
            
            for i in range(1, min(50, len(data))):
                current_price = data['close'].iloc[-i]
                prev_price = data['close'].iloc[-i-1] if i < len(data) - 1 else current_price
                
                # Check if price approached the level
                if abs(current_price - fib_price) / fib_price < 0.02:  # Within 2%
                    total_approaches += 1
                    
                    # Check if it bounced (respected the level)
                    if i > 1:
                        next_price = data['close'].iloc[-i+1]
                        if (prev_price > fib_price and current_price < fib_price and next_price > fib_price) or \
                           (prev_price < fib_price and current_price > fib_price and next_price < fib_price):
                            respect_count += 1
            
            if total_approaches > 0:
                respect_ratio = respect_count / total_approaches
                strength = min(respect_ratio + (total_approaches / 20.0), 1.0)
            else:
                strength = 0.4  # Default strength
            
            return strength
            
        except Exception:
            return 0.4
    
    def _extract_support_resistance_levels(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract horizontal support and resistance levels."""
        sr_levels = []
        
        try:
            # Use price clustering to find significant levels
            prices = np.concatenate([data['high'].values, data['low'].values, data['close'].values])
            
            # Remove outliers
            q1, q3 = np.percentile(prices, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_prices = prices[(prices >= lower_bound) & (prices <= upper_bound)]
            
            if len(filtered_prices) > 10:
                # Cluster prices
                price_points = filtered_prices.reshape(-1, 1)
                normalized_prices = self.scaler.fit_transform(price_points)
                
                clusters = self.price_clusterer.fit_predict(normalized_prices)
                unique_clusters = np.unique(clusters[clusters != -1])
                
                for cluster_id in unique_clusters:
                    cluster_prices = filtered_prices[clusters == cluster_id]
                    
                    if len(cluster_prices) >= self.parameters['min_confluence_points']:
                        level_price = np.median(cluster_prices)
                        
                        # Determine type based on recent price action
                        current_price = data['close'].iloc[-1]
                        level_type = 'support' if current_price > level_price else 'resistance'
                        
                        # Calculate strength based on cluster size and consistency
                        strength = min(len(cluster_prices) / 20.0, 1.0)
                        
                        sr_levels.append({
                            'price': level_price,
                            'type': level_type,
                            'strength': strength,
                            'age': 0,
                            'source': 'clustering'
                        })
            
        except Exception:
            pass
        
        return sr_levels
    
    def _extract_volume_levels(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract volume-based price levels."""
        volume_levels = []
        
        try:
            # Find high volume areas (Volume Profile approximation)
            price_bins = 20
            recent_data = data.tail(100)  # Recent data for volume analysis
            
            # Create price bins
            price_min = recent_data['low'].min()
            price_max = recent_data['high'].max()
            price_step = (price_max - price_min) / price_bins
            
            volume_at_price = {}
            
            for _, row in recent_data.iterrows():
                # Distribute volume across price range for this candle
                candle_low = row['low']
                candle_high = row['high']
                candle_volume = row['volume']
                
                # Find which bins this candle covers
                low_bin = int((candle_low - price_min) / price_step)
                high_bin = int((candle_high - price_min) / price_step)
                
                # Distribute volume across bins
                bins_covered = max(1, high_bin - low_bin + 1)
                volume_per_bin = candle_volume / bins_covered
                
                for bin_idx in range(max(0, low_bin), min(price_bins, high_bin + 1)):
                    bin_price = price_min + (bin_idx + 0.5) * price_step
                    if bin_price not in volume_at_price:
                        volume_at_price[bin_price] = 0
                    volume_at_price[bin_price] += volume_per_bin
            
            # Find high volume levels
            if volume_at_price:
                sorted_volumes = sorted(volume_at_price.items(), key=lambda x: x[1], reverse=True)
                top_volume_levels = sorted_volumes[:5]  # Top 5 volume levels
                
                for price, volume in top_volume_levels:
                    current_price = data['close'].iloc[-1]
                    level_type = 'support' if current_price > price else 'resistance'
                    
                    # Strength based on relative volume
                    max_volume = max(volume_at_price.values())
                    strength = volume / max_volume
                    
                    volume_levels.append({
                        'price': price,
                        'type': level_type,
                        'strength': strength,
                        'age': 0,
                        'source': 'volume_profile'
                    })
            
        except Exception:
            pass
        
        return volume_levels
    
    def _detect_confluence_areas(self, indicator_levels: Dict[str, List], data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect confluence areas by clustering nearby levels."""
        confluence_areas = []
        
        try:
            # Collect all levels
            all_levels = []
            for indicator_type, levels in indicator_levels.items():
                for level in levels:
                    level['indicator_type'] = indicator_type
                    all_levels.append(level)
            
            if len(all_levels) < self.parameters['min_confluence_points']:
                return confluence_areas
            
            # Extract prices for clustering
            prices = np.array([level['price'] for level in all_levels])
            
            # Cluster prices to find confluence areas
            price_points = prices.reshape(-1, 1)
            current_price = data['close'].iloc[-1]
            
            # Normalize prices relative to current price for clustering
            normalized_prices = (prices / current_price).reshape(-1, 1)
            
            try:
                clusters = self.price_clusterer.fit_predict(normalized_prices)
                unique_clusters = np.unique(clusters[clusters != -1])
                
                for cluster_id in unique_clusters:
                    cluster_indices = np.where(clusters == cluster_id)[0]
                    cluster_levels = [all_levels[i] for i in cluster_indices]
                    
                    if len(cluster_levels) >= self.parameters['min_confluence_points']:
                        # Calculate confluence area properties
                        cluster_prices = [level['price'] for level in cluster_levels]
                        
                        confluence_area = {
                            'center_price': np.median(cluster_prices),
                            'price_range': [np.min(cluster_prices), np.max(cluster_prices)],
                            'level_count': len(cluster_levels),
                            'contributing_levels': cluster_levels,
                            'indicator_diversity': len(set(level['indicator_type'] for level in cluster_levels)),
                            'dominant_type': self._determine_dominant_type(cluster_levels),
                            'age_score': self._calculate_age_score(cluster_levels)
                        }
                        
                        confluence_areas.append(confluence_area)
                
            except Exception:
                pass
            
        except Exception:
            pass
        
        return sorted(confluence_areas, key=lambda x: x['level_count'], reverse=True)
    
    def _determine_dominant_type(self, levels: List[Dict[str, Any]]) -> str:
        """Determine dominant type (support/resistance) for confluence area."""
        type_counts = {'support': 0, 'resistance': 0, 'neutral': 0}
        
        for level in levels:
            level_type = level.get('type', 'neutral')
            type_counts[level_type] += 1
        
        return max(type_counts.items(), key=lambda x: x[1])[0]
    
    def _calculate_age_score(self, levels: List[Dict[str, Any]]) -> float:
        """Calculate age score for confluence (older = more established)."""
        ages = [level.get('age', 0) for level in levels]
        if ages:
            avg_age = np.mean(ages)
            # Convert age to score (higher age = higher score, but cap it)
            age_score = min(avg_age / 50.0, 1.0)  # Normalize by 50 periods
            return age_score
        return 0.0
    
    def _perform_geometric_analysis(self, confluence_areas: List[Dict[str, Any]], 
                                  data: pd.DataFrame) -> Dict[str, Any]:
        """Perform geometric analysis of confluence areas."""
        geometric_analysis = {
            'area_density': 0.0,
            'price_distribution': 'normal',
            'symmetry_score': 0.0,
            'clustering_quality': 0.0
        }
        
        try:
            if not confluence_areas:
                return geometric_analysis
            
            current_price = data['close'].iloc[-1]
            
            # Calculate area density (confluences per price unit)
            price_range = data['high'].max() - data['low'].min()
            geometric_analysis['area_density'] = len(confluence_areas) / max(price_range, 1)
            
            # Analyze price distribution of confluence centers
            confluence_prices = [area['center_price'] for area in confluence_areas]
            
            if len(confluence_prices) > 2:
                # Test for normal distribution
                _, p_value = stats.normaltest(confluence_prices)
                geometric_analysis['price_distribution'] = 'normal' if p_value > 0.05 else 'non_normal'
                
                # Calculate symmetry around current price
                distances_to_current = [abs(price - current_price) for price in confluence_prices]
                above_current = [price for price in confluence_prices if price > current_price]
                below_current = [price for price in confluence_prices if price < current_price]
                
                if len(above_current) > 0 and len(below_current) > 0:
                    avg_distance_above = np.mean([price - current_price for price in above_current])
                    avg_distance_below = np.mean([current_price - price for price in below_current])
                    
                    symmetry = 1.0 - abs(avg_distance_above - avg_distance_below) / max(avg_distance_above, avg_distance_below)
                    geometric_analysis['symmetry_score'] = max(0.0, symmetry)
            
            # Clustering quality (silhouette score approximation)
            if len(confluence_prices) > 2:
                price_array = np.array(confluence_prices).reshape(-1, 1)
                try:
                    from sklearn.metrics import silhouette_score
                    from sklearn.cluster import KMeans
                    
                    if len(confluence_prices) >= 4:
                        kmeans = KMeans(n_clusters=min(3, len(confluence_prices)//2), random_state=42)
                        cluster_labels = kmeans.fit_predict(price_array)
                        silhouette_avg = silhouette_score(price_array, cluster_labels)
                        geometric_analysis['clustering_quality'] = max(0.0, silhouette_avg)
                    
                except Exception:
                    geometric_analysis['clustering_quality'] = 0.5
            
        except Exception:
            pass
        
        return geometric_analysis
    
    def _calculate_confluence_strengths(self, confluence_areas: List[Dict[str, Any]],
                                      indicator_levels: Dict[str, List],
                                      data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate strength scores for each confluence area."""
        strength_analysis = []
        
        for area in confluence_areas:
            try:
                # Component strengths
                level_count_strength = min(area['level_count'] / 10.0, 1.0)
                
                diversity_strength = min(area['indicator_diversity'] / len(self.parameters['indicator_types']), 1.0)
                
                age_strength = area.get('age_score', 0.0)
                
                # Individual level strengths
                individual_strengths = [level.get('strength', 0.5) for level in area['contributing_levels']]
                avg_individual_strength = np.mean(individual_strengths) if individual_strengths else 0.5
                
                # Price proximity strength (closer levels = stronger confluence)
                price_range = area['price_range'][1] - area['price_range'][0]
                current_price = data['close'].iloc[-1]
                proximity_strength = max(0.0, 1.0 - (price_range / current_price) * 20)  # Normalize
                
                # Dynamic weighting based on parameters
                if self.parameters['dynamic_weighting']:
                    weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Count, diversity, age, individual, proximity
                else:
                    weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal weights
                
                # Calculate total strength
                strength_components = [
                    level_count_strength,
                    diversity_strength,
                    age_strength,
                    avg_individual_strength,
                    proximity_strength
                ]
                
                total_strength = sum(w * s for w, s in zip(weights, strength_components))
                
                strength_analysis.append({
                    'area_index': len(strength_analysis),
                    'total_strength': total_strength,
                    'component_strengths': {
                        'level_count': level_count_strength,
                        'diversity': diversity_strength,
                        'age': age_strength,
                        'individual_avg': avg_individual_strength,
                        'proximity': proximity_strength
                    },
                    'strength_category': self._categorize_strength(total_strength),
                    'reliability_score': self._calculate_reliability_score(area, data)
                })
                
            except Exception:
                # Default strength analysis
                strength_analysis.append({
                    'area_index': len(strength_analysis),
                    'total_strength': 0.5,
                    'component_strengths': {},
                    'strength_category': 'moderate',
                    'reliability_score': 0.5
                })
        
        return strength_analysis
    
    def _categorize_strength(self, strength: float) -> str:
        """Categorize confluence strength."""
        if strength > 0.8:
            return 'very_strong'
        elif strength > 0.6:
            return 'strong'
        elif strength > 0.4:
            return 'moderate'
        elif strength > 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    def _calculate_reliability_score(self, area: Dict[str, Any], data: pd.DataFrame) -> float:
        """Calculate reliability score based on historical performance."""
        try:
            # Check how well this confluence area has held in recent past
            center_price = area['center_price']
            current_price = data['close'].iloc[-1]
            
            # Look at recent price action around this level
            test_count = 0
            hold_count = 0
            
            for i in range(1, min(50, len(data))):
                past_price = data['close'].iloc[-i]
                
                # Check if price tested this level (within 2%)
                if abs(past_price - center_price) / center_price < 0.02:
                    test_count += 1
                    
                    # Check if level held (price bounced within next few periods)
                    if i > 3:
                        future_prices = data['close'].iloc[-i+1:-i+4].values if i >= 4 else [current_price]
                        
                        # Determine if it's supposed to be support or resistance
                        level_type = area.get('dominant_type', 'neutral')
                        
                        if level_type == 'support':
                            # Support should bounce price up
                            if any(fp > center_price for fp in future_prices):
                                hold_count += 1
                        elif level_type == 'resistance':
                            # Resistance should bounce price down
                            if any(fp < center_price for fp in future_prices):
                                hold_count += 1
            
            if test_count > 0:
                reliability = hold_count / test_count
            else:
                reliability = 0.6  # Default for untested levels
            
            return min(max(reliability, 0.0), 1.0)
            
        except Exception:
            return 0.5
    
    def _analyze_multi_timeframe_confluence(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze confluence across multiple timeframes."""
        mtf_analysis = {
            'timeframe_confluence': {},
            'strongest_timeframe': None,
            'confluence_consistency': 0.0
        }
        
        try:
            current_price = data['close'].iloc[-1]
            
            for window in self.parameters['timeframe_windows']:
                if len(data) >= window:
                    # Analyze confluence for this timeframe
                    timeframe_data = data.tail(window)
                    timeframe_levels = self._extract_indicator_levels(timeframe_data)
                    timeframe_confluences = self._detect_confluence_areas(timeframe_levels, timeframe_data)
                    
                    # Calculate average strength for this timeframe
                    if timeframe_confluences:
                        timeframe_strengths = self._calculate_confluence_strengths(
                            timeframe_confluences, timeframe_levels, timeframe_data
                        )
                        avg_strength = np.mean([s['total_strength'] for s in timeframe_strengths])
                        confluence_count = len(timeframe_confluences)
                    else:
                        avg_strength = 0.0
                        confluence_count = 0
                    
                    mtf_analysis['timeframe_confluence'][f'window_{window}'] = {
                        'avg_strength': avg_strength,
                        'confluence_count': confluence_count,
                        'confluences': timeframe_confluences
                    }
            
            # Find strongest timeframe
            if mtf_analysis['timeframe_confluence']:
                strongest_tf = max(
                    mtf_analysis['timeframe_confluence'].items(),
                    key=lambda x: x[1]['avg_strength']
                )
                mtf_analysis['strongest_timeframe'] = strongest_tf[0]
                
                # Calculate consistency across timeframes
                strengths = [tf_data['avg_strength'] for tf_data in mtf_analysis['timeframe_confluence'].values()]
                if len(strengths) > 1:
                    consistency = 1.0 - (np.std(strengths) / max(np.mean(strengths), 0.1))
                    mtf_analysis['confluence_consistency'] = max(0.0, consistency)
            
        except Exception:
            pass
        
        return mtf_analysis
    
    def _analyze_confluence_persistence(self, confluence_areas: List[Dict[str, Any]],
                                      strength_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze persistence of confluence areas over time."""
        persistence_analysis = {
            'avg_persistence': 0.0,
            'stability_score': 0.0,
            'evolution_trend': 'stable'
        }
        
        # Store current confluence in history
        current_confluence = {
            'timestamp': len(self._historical_confluences),
            'areas': confluence_areas,
            'strengths': strength_analysis,
            'total_strength': sum(s.get('total_strength', 0) for s in strength_analysis)
        }
        
        self._historical_confluences.append(current_confluence)
        
        # Keep only recent history
        if len(self._historical_confluences) > 20:
            self._historical_confluences = self._historical_confluences[-20:]
        
        try:
            if len(self._historical_confluences) > 3:
                # Analyze persistence
                recent_histories = self._historical_confluences[-5:]
                
                # Calculate average number of confluences over time
                confluence_counts = [len(h['areas']) for h in recent_histories]
                persistence_analysis['avg_persistence'] = np.mean(confluence_counts)
                
                # Calculate stability (low variance = high stability)
                if len(confluence_counts) > 1:
                    cv = np.std(confluence_counts) / max(np.mean(confluence_counts), 1)
                    persistence_analysis['stability_score'] = max(0.0, 1.0 - cv)
                
                # Analyze evolution trend
                total_strengths = [h['total_strength'] for h in recent_histories]
                if len(total_strengths) > 2:
                    # Linear regression to detect trend
                    x = np.arange(len(total_strengths))
                    slope, _, _, _, _ = stats.linregress(x, total_strengths)
                    
                    if slope > 0.1:
                        persistence_analysis['evolution_trend'] = 'strengthening'
                    elif slope < -0.1:
                        persistence_analysis['evolution_trend'] = 'weakening'
                    else:
                        persistence_analysis['evolution_trend'] = 'stable'
        
        except Exception:
            pass
        
        return persistence_analysis
    
    def _validate_with_machine_learning(self, confluence_areas: List[Dict[str, Any]],
                                      strength_analysis: List[Dict[str, Any]],
                                      data: pd.DataFrame) -> Dict[str, Any]:
        """Validate confluence areas using machine learning."""
        ml_validation = {
            'validation_score': 0.5,
            'predicted_effectiveness': 0.5,
            'confidence_intervals': {},
            'model_trained': self.is_trained
        }
        
        try:
            # Prepare features for ML validation
            features = self._prepare_confluence_features(confluence_areas, strength_analysis, data)
            
            if len(features) > 0 and not self.is_trained:
                # Train models if enough data available
                self._train_confluence_models(data)
            
            if self.is_trained and len(features) > 0:
                # Make predictions
                latest_features = features[-1:].reshape(1, -1)
                
                try:
                    # Scale features
                    latest_features_scaled = self.scaler.transform(latest_features)
                    
                    # Predict confluence effectiveness
                    effectiveness_prob = self.confluence_classifier.predict_proba(latest_features_scaled)
                    ml_validation['predicted_effectiveness'] = effectiveness_prob[0][1] if len(effectiveness_prob[0]) > 1 else 0.5
                    
                    # Predict strength reliability
                    strength_prediction = self.strength_predictor.predict_proba(latest_features_scaled)
                    ml_validation['validation_score'] = strength_prediction[0][1] if len(strength_prediction[0]) > 1 else 0.5
                    
                    # Calculate confidence intervals (simplified)
                    ml_validation['confidence_intervals'] = {
                        'effectiveness_lower': max(0.0, ml_validation['predicted_effectiveness'] - 0.2),
                        'effectiveness_upper': min(1.0, ml_validation['predicted_effectiveness'] + 0.2),
                        'validation_lower': max(0.0, ml_validation['validation_score'] - 0.2),
                        'validation_upper': min(1.0, ml_validation['validation_score'] + 0.2)
                    }
                    
                except Exception:
                    pass
        
        except Exception:
            pass
        
        return ml_validation
    
    def _prepare_confluence_features(self, confluence_areas: List[Dict[str, Any]],
                                   strength_analysis: List[Dict[str, Any]],
                                   data: pd.DataFrame) -> np.ndarray:
        """Prepare features for machine learning models."""
        if not confluence_areas:
            return np.array([])
        
        try:
            # Current market features
            current_price = data['close'].iloc[-1]
            volatility = data['close'].pct_change().tail(20).std()
            volume_ratio = data['volume'].iloc[-1] / data['volume'].tail(20).mean()
            
            # Confluence features
            total_confluences = len(confluence_areas)
            avg_strength = np.mean([s.get('total_strength', 0) for s in strength_analysis])
            max_strength = max([s.get('total_strength', 0) for s in strength_analysis]) if strength_analysis else 0
            
            # Price position relative to confluences
            resistance_levels = [area['center_price'] for area in confluence_areas 
                               if area.get('dominant_type') == 'resistance']
            support_levels = [area['center_price'] for area in confluence_areas 
                            if area.get('dominant_type') == 'support']
            
            nearest_resistance = min([abs(r - current_price) / current_price for r in resistance_levels]) if resistance_levels else 1.0
            nearest_support = min([abs(s - current_price) / current_price for s in support_levels]) if support_levels else 1.0
            
            feature_vector = [
                total_confluences,
                avg_strength,
                max_strength,
                volatility,
                volume_ratio,
                nearest_resistance,
                nearest_support,
                len(resistance_levels),
                len(support_levels)
            ]
            
            return np.array([feature_vector])
            
        except Exception:
            return np.array([])
    
    def _train_confluence_models(self, data: pd.DataFrame):
        """Train machine learning models for confluence validation."""
        try:
            # This is a simplified training approach
            # In practice, you would need historical confluence data with outcomes
            
            # Generate synthetic training data based on historical patterns
            training_features = []
            training_labels_effectiveness = []
            training_labels_strength = []
            
            # Use historical data to create training samples
            for i in range(50, len(data) - 10):
                window_data = data.iloc[i-50:i]
                
                # Extract features for this historical window
                historical_levels = self._extract_indicator_levels(window_data)
                historical_confluences = self._detect_confluence_areas(historical_levels, window_data)
                
                if historical_confluences:
                    historical_strengths = self._calculate_confluence_strengths(
                        historical_confluences, historical_levels, window_data
                    )
                    
                    features = self._prepare_confluence_features(
                        historical_confluences, historical_strengths, window_data
                    )
                    
                    if len(features) > 0:
                        training_features.append(features[0])
                        
                        # Create labels based on future price action
                        future_prices = data['close'].iloc[i:i+10].values
                        current_price = window_data['close'].iloc[-1]
                        
                        # Effectiveness: did confluences hold?
                        max_future_move = max(abs(fp - current_price) / current_price for fp in future_prices)
                        effectiveness_label = 1 if max_future_move < 0.05 else 0  # Held within 5%
                        
                        # Strength reliability: did strong confluences perform better?
                        avg_strength = np.mean([s.get('total_strength', 0) for s in historical_strengths])
                        strength_label = 1 if avg_strength > 0.6 else 0
                        
                        training_labels_effectiveness.append(effectiveness_label)
                        training_labels_strength.append(strength_label)
            
            if len(training_features) > 10:
                # Train models
                X = np.array(training_features)
                y_effectiveness = np.array(training_labels_effectiveness)
                y_strength = np.array(training_labels_strength)
                
                # Scale features
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                
                # Train classifiers
                self.confluence_classifier.fit(X_scaled, y_effectiveness)
                self.strength_predictor.fit(X_scaled, y_strength)
                
                self.is_trained = True
        
        except Exception:
            pass
    
    def _predict_breakouts_from_confluence(self, confluence_areas: List[Dict[str, Any]],
                                         strength_analysis: List[Dict[str, Any]],
                                         data: pd.DataFrame) -> Dict[str, Any]:
        """Predict potential breakouts from confluence areas."""
        breakout_predictions = {
            'breakout_probability': 0.0,
            'breakout_direction': 'neutral',
            'target_levels': [],
            'breakout_strength': 0.0,
            'time_to_breakout': None
        }
        
        try:
            if not confluence_areas:
                return breakout_predictions
            
            current_price = data['close'].iloc[-1]
            current_volatility = data['close'].pct_change().tail(20).std()
            
            # Find nearest significant confluence areas
            nearest_resistance = None
            nearest_support = None
            
            for i, area in enumerate(confluence_areas):
                area_strength = strength_analysis[i].get('total_strength', 0) if i < len(strength_analysis) else 0
                
                # Only consider strong confluences
                if area_strength > self.parameters['strength_threshold']:
                    center_price = area['center_price']
                    dominant_type = area.get('dominant_type', 'neutral')
                    
                    if dominant_type == 'resistance' and center_price > current_price:
                        if nearest_resistance is None or center_price < nearest_resistance['price']:
                            nearest_resistance = {
                                'price': center_price,
                                'strength': area_strength,
                                'area': area
                            }
                    
                    elif dominant_type == 'support' and center_price < current_price:
                        if nearest_support is None or center_price > nearest_support['price']:
                            nearest_support = {
                                'price': center_price,
                                'strength': area_strength,
                                'area': area
                            }
            
            # Calculate breakout probability
            if nearest_resistance or nearest_support:
                # Distance to nearest strong level
                if nearest_resistance:
                    resistance_distance = (nearest_resistance['price'] - current_price) / current_price
                else:
                    resistance_distance = 1.0
                
                if nearest_support:
                    support_distance = (current_price - nearest_support['price']) / current_price
                else:
                    support_distance = 1.0
                
                # Closer to level = higher breakout probability
                min_distance = min(resistance_distance, support_distance)
                proximity_factor = max(0.0, 1.0 - min_distance * 20)  # Scale by 20
                
                # Volatility factor (higher volatility = higher breakout probability)
                volatility_factor = min(current_volatility * 10, 1.0)
                
                # Combined breakout probability
                breakout_probability = (proximity_factor + volatility_factor) / 2.0
                
                # Determine direction
                if resistance_distance < support_distance:
                    breakout_direction = 'upward'
                    breakout_strength = nearest_resistance['strength'] if nearest_resistance else 0
                    target_price = nearest_resistance['price'] * 1.02 if nearest_resistance else current_price
                else:
                    breakout_direction = 'downward'
                    breakout_strength = nearest_support['strength'] if nearest_support else 0
                    target_price = nearest_support['price'] * 0.98 if nearest_support else current_price
                
                # Estimate time to breakout (inverse of volatility and proximity)
                if breakout_probability > 0.5:
                    time_factor = (1.0 - breakout_probability) * (1.0 - volatility_factor)
                    estimated_periods = max(1, int(20 * time_factor))
                    breakout_predictions['time_to_breakout'] = estimated_periods
                
                breakout_predictions.update({
                    'breakout_probability': breakout_probability,
                    'breakout_direction': breakout_direction,
                    'breakout_strength': breakout_strength,
                    'target_levels': [target_price]
                })
        
        except Exception:
            pass
        
        return breakout_predictions
    
    def _generate_trading_signals(self, confluence_areas: List[Dict[str, Any]],
                                strength_analysis: List[Dict[str, Any]],
                                ml_validation: Dict[str, Any],
                                breakout_predictions: Dict[str, Any],
                                data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals based on confluence analysis."""
        signals = {
            'signal_type': 'neutral',
            'signal_strength': 0.0,
            'confidence': 0.0,
            'entry_levels': [],
            'stop_loss_levels': [],
            'take_profit_levels': [],
            'position_size': 'normal'
        }
        
        try:
            if not confluence_areas:
                return signals
            
            current_price = data['close'].iloc[-1]
            
            # Find actionable confluence areas
            strong_confluences = []
            for i, area in enumerate(confluence_areas):
                if i < len(strength_analysis):
                    strength = strength_analysis[i].get('total_strength', 0)
                    if strength > self.parameters['strength_threshold']:
                        strong_confluences.append({
                            'area': area,
                            'strength': strength,
                            'analysis': strength_analysis[i]
                        })
            
            if strong_confluences:
                # Sort by strength
                strong_confluences.sort(key=lambda x: x['strength'], reverse=True)
                strongest_confluence = strong_confluences[0]
                
                area = strongest_confluence['area']
                strength = strongest_confluence['strength']
                center_price = area['center_price']
                dominant_type = area.get('dominant_type', 'neutral')
                
                # Determine signal type based on price position and confluence type
                price_to_confluence = (center_price - current_price) / current_price
                
                if abs(price_to_confluence) < 0.05:  # Within 5% of confluence
                    if dominant_type == 'support' and price_to_confluence < 0:  # Near support from above
                        signals['signal_type'] = 'bounce_long'
                        signals['entry_levels'] = [center_price * 1.01]
                        signals['stop_loss_levels'] = [center_price * 0.98]
                        signals['take_profit_levels'] = [center_price * 1.05, center_price * 1.08]
                    
                    elif dominant_type == 'resistance' and price_to_confluence > 0:  # Near resistance from below
                        signals['signal_type'] = 'bounce_short'
                        signals['entry_levels'] = [center_price * 0.99]
                        signals['stop_loss_levels'] = [center_price * 1.02]
                        signals['take_profit_levels'] = [center_price * 0.95, center_price * 0.92]
                
                # Breakout signals
                breakout_prob = breakout_predictions.get('breakout_probability', 0)
                if breakout_prob > 0.7:
                    direction = breakout_predictions.get('breakout_direction', 'neutral')
                    
                    if direction == 'upward':
                        signals['signal_type'] = 'breakout_long'
                        signals['entry_levels'] = [center_price * 1.002]
                        signals['stop_loss_levels'] = [center_price * 0.99]
                        signals['take_profit_levels'] = breakout_predictions.get('target_levels', [])
                    
                    elif direction == 'downward':
                        signals['signal_type'] = 'breakout_short'
                        signals['entry_levels'] = [center_price * 0.998]
                        signals['stop_loss_levels'] = [center_price * 1.01]
                        signals['take_profit_levels'] = breakout_predictions.get('target_levels', [])
                
                # Signal strength
                ml_effectiveness = ml_validation.get('predicted_effectiveness', 0.5)
                signals['signal_strength'] = (strength + ml_effectiveness + breakout_prob) / 3.0
                
                # Position sizing based on strength and volatility
                volatility = data['close'].pct_change().tail(20).std()
                
                if signals['signal_strength'] > 0.8 and volatility < 0.03:
                    signals['position_size'] = 'large'
                elif signals['signal_strength'] > 0.6:
                    signals['position_size'] = 'normal'
                else:
                    signals['position_size'] = 'small'
                
                # Overall confidence
                reliability = strongest_confluence['analysis'].get('reliability_score', 0.5)
                ml_validation_score = ml_validation.get('validation_score', 0.5)
                
                signals['confidence'] = (signals['signal_strength'] + reliability + ml_validation_score) / 3.0
        
        except Exception:
            pass
        
        return signals
    
    def _calculate_confidence_score(self, confluence_areas: List[Dict[str, Any]],
                                  strength_analysis: List[Dict[str, Any]],
                                  ml_validation: Dict[str, Any],
                                  persistence_analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence score for confluence analysis."""
        try:
            # Component confidences
            if confluence_areas and strength_analysis:
                # Confluence quality
                avg_strength = np.mean([s.get('total_strength', 0) for s in strength_analysis])
                confluence_confidence = avg_strength
                
                # Count quality
                count_confidence = min(len(confluence_areas) / 5.0, 1.0)  # Normalize by 5 areas
            else:
                confluence_confidence = 0.0
                count_confidence = 0.0
            
            # ML validation confidence
            ml_confidence = ml_validation.get('validation_score', 0.5)
            
            # Persistence confidence
            persistence_confidence = persistence_analysis.get('stability_score', 0.5)
            
            # Combined confidence
            weights = [0.3, 0.2, 0.3, 0.2]  # Confluence, count, ML, persistence
            confidence_components = [confluence_confidence, count_confidence, ml_confidence, persistence_confidence]
            
            overall_confidence = sum(w * c for w, c in zip(weights, confidence_components))
            
            return min(max(overall_confidence, 0.0), 1.0)
            
        except Exception:
            return 0.5
    
    def _get_strongest_confluence(self, confluence_areas: List[Dict[str, Any]],
                                strength_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get the strongest confluence area."""
        if not confluence_areas or not strength_analysis:
            return {}
        
        try:
            max_strength_idx = max(range(len(strength_analysis)),
                                 key=lambda i: strength_analysis[i].get('total_strength', 0))
            
            return {
                'area': confluence_areas[max_strength_idx],
                'strength': strength_analysis[max_strength_idx].get('total_strength', 0),
                'analysis': strength_analysis[max_strength_idx]
            }
            
        except Exception:
            return {}
    
    def _get_default_output(self) -> Dict[str, Any]:
        """Return default output when insufficient data."""
        return {
            'confluence_areas': [],
            'indicator_levels': {},
            'signals': {'signal_type': 'neutral', 'signal_strength': 0.0, 'confidence': 0.0},
            'confidence': 0.0,
            'strongest_confluence': {},
            'confluence_count': 0,
            'max_strength': 0.0
        }
    
    def _handle_calculation_error(self, error: Exception) -> Dict[str, Any]:
        """Handle calculation errors gracefully."""
        return self._get_default_output()
