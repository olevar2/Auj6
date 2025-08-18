"""
Advanced Volume Profile Indicator
================================

A sophisticated implementation of Volume Profile with:
- Statistical distribution modeling
- Value area analysis with confidence intervals
- Volume node identification and clustering
- Market profile dynamics
- Institutional level detection

Mathematical Foundation:
Volume Profile analyzes the volume traded at each price level over a specified period,
creating a horizontal histogram that shows:
1. Point of Control (POC) - Price level with highest volume
2. Value Area (VA) - Price range containing 70% of total volume
3. High Volume Nodes (HVN) - Significant support/resistance levels
4. Low Volume Nodes (LVN) - Areas of price discovery/breakout potential

Advanced Features:
1. Multi-timeframe volume profile analysis
2. Statistical distribution modeling using kernel density estimation
3. Dynamic value area calculation with confidence intervals
4. Volume node clustering for institutional level identification
5. Market profile rotation analysis
6. Volume imbalance detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Import base class
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from ..indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface
from ..core.signal_type import SignalType

@dataclass
class VolumeProfileConfig:
    """Configuration for Volume Profile calculation"""
    profile_period: int = 100
    value_area_percentage: float = 0.70
    min_price_bins: int = 50
    max_price_bins: int = 200
    hvn_threshold_percentile: float = 80.0
    lvn_threshold_percentile: float = 20.0
    institutional_volume_threshold: float = 2.0
    rotation_detection_period: int = 20
    imbalance_threshold: float = 0.3
    kde_bandwidth: Optional[float] = None  # Auto-calculate if None
    
@dataclass
class VolumeNode:
    """Individual volume node structure"""
    price_level: float
    volume: float
    volume_percentage: float
    node_type: str  # 'hvn', 'lvn', 'poc', 'va_high', 'va_low'
    strength: float  # Relative strength score
    
@dataclass
class ValueArea:
    """Value Area analysis result"""
    va_high: float
    va_low: float
    va_volume: float
    va_percentage: float
    poc_price: float
    poc_volume: float
    
@dataclass
class MarketRotation:
    """Market rotation analysis"""
    rotation_type: str  # 'normal', 'trending', 'balancing'
    rotation_strength: float
    dominant_timeframe: str
    imbalance_direction: str  # 'bullish', 'bearish', 'neutral'

@dataclass
class VolumeProfileResult:
    """Result structure for Volume Profile analysis"""
    value_area: ValueArea
    volume_nodes: List[VolumeNode]
    market_rotation: MarketRotation
    institutional_levels: List[float]
    profile_shape: str  # 'normal', 'b_shape', 'p_shape', 'double_distribution'
    volume_imbalance: float
    support_resistance_levels: List[float]
    signal_confidence: float

class VolumeProfileIndicator(StandardIndicatorInterface):
    """
    Advanced Volume Profile Indicator
    
    This indicator provides sophisticated volume distribution analysis through:
    1. Multi-bin volume profile construction with statistical modeling
    2. Value area identification using percentile-based analysis
    3. Volume node clustering for support/resistance detection
    4. Market rotation analysis for trend identification
    5. Institutional level detection through volume clustering
    """
    
    def __init__(self, config: Optional[VolumeProfileConfig] = None):
        """Initialize the Volume Profile Indicator"""
        super().__init__()
        self.config = config or VolumeProfileConfig()
        self.logger = logging.getLogger(__name__)
        
        # Analysis components
        self._scaler = StandardScaler()
        self._clustering_model = KMeans(n_clusters=5, random_state=42, n_init=10)
        
        # Historical data storage
        self._profile_history: List[Dict[str, Any]] = []
        self._value_area_history: List[ValueArea] = []
        self._rotation_history: List[MarketRotation] = []
        
        # Volume distribution data
        self._price_levels: np.ndarray = np.array([])
        self._volume_at_price: np.ndarray = np.array([])
        self._cumulative_volume: np.ndarray = np.array([])
        
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Volume Profile with comprehensive analysis
        
        Args:
            data: Dictionary containing OHLCV data
            
        Returns:
            Dictionary with Volume Profile analysis results
        """
        try:
            # Validate and extract data
            if not self._validate_data(data):
                raise ValueError("Invalid or insufficient data provided")
            
            df = pd.DataFrame(data)
            
            # Ensure minimum data requirements
            if len(df) < self.config.profile_period:
                return self._create_default_result()
            
            # Extract profile period data
            profile_data = df.tail(self.config.profile_period).copy()
            
            # Build volume profile
            profile_results = self._build_volume_profile(profile_data)
            
            # Calculate value area
            value_area_results = self._calculate_value_area(profile_results)
            
            # Identify volume nodes
            node_results = self._identify_volume_nodes(profile_results, value_area_results)
            
            # Analyze market rotation
            rotation_results = self._analyze_market_rotation(profile_data, profile_results)
            
            # Detect institutional levels
            institutional_results = self._detect_institutional_levels(profile_results, node_results)
            
            # Analyze profile shape and characteristics
            shape_results = self._analyze_profile_shape(profile_results)
            
            # Calculate volume imbalances
            imbalance_results = self._calculate_volume_imbalances(profile_results, value_area_results)
            
            # Identify support/resistance levels
            sr_results = self._identify_support_resistance(node_results, value_area_results)
            
            # Calculate signal confidence
            confidence_results = self._calculate_signal_confidence(
                value_area_results, rotation_results, shape_results
            )
            
            # Compile final result
            result = self._compile_profile_result(
                value_area_results, node_results, rotation_results,
                institutional_results, shape_results, imbalance_results,
                sr_results, confidence_results
            )
            
            # Update historical data
            self._update_history(profile_data, result)
            
            # Generate trading signal
            signal = self._generate_signal(result, profile_data)
            
            return {
                'signal': signal,
                'confidence': result.signal_confidence,
                'poc_price': result.value_area.poc_price,
                'value_area_high': result.value_area.va_high,
                'value_area_low': result.value_area.va_low,
                'profile_shape': result.profile_shape,
                'rotation_type': result.market_rotation.rotation_type,
                'volume_imbalance': result.volume_imbalance,
                'institutional_levels': result.institutional_levels,
                'support_resistance_levels': result.support_resistance_levels,
                'high_volume_nodes': [node.price_level for node in result.volume_nodes if node.node_type == 'hvn'],
                'low_volume_nodes': [node.price_level for node in result.volume_nodes if node.node_type == 'lvn'],
                'metadata': {
                    'indicator_name': 'VolumeProfile',
                    'calculation_method': 'statistical_distribution_analysis',
                    'parameters': {
                        'profile_period': self.config.profile_period,
                        'value_area_percentage': self.config.value_area_percentage,
                        'price_bins': len(self._price_levels)
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Volume Profile: {str(e)}")
            return self._create_error_result(str(e))
    
    def _build_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build volume profile with statistical distribution modeling"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Calculate price range and determine optimal bin count
        price_min = np.min(low)
        price_max = np.max(high)
        price_range = price_max - price_min
        
        if price_range == 0:
            price_range = price_max * 0.01  # 1% range if no variation
        
        # Dynamic bin calculation based on price precision and data size
        tick_size = price_range / 1000  # Estimate tick size
        optimal_bins = max(self.config.min_price_bins, 
                          min(self.config.max_price_bins, 
                              int(price_range / tick_size)))
        
        # Create price bins
        price_bins = np.linspace(price_min, price_max, optimal_bins + 1)
        price_centers = (price_bins[:-1] + price_bins[1:]) / 2
        
        # Initialize volume array
        volume_at_price = np.zeros(len(price_centers))
        
        # Distribute volume across price levels using OHLC data
        for i in range(len(df)):
            bar_high = high[i]
            bar_low = low[i]
            bar_close = close[i]
            bar_volume = volume[i]
            
            # Find bins that intersect with this price bar
            intersecting_bins = np.where(
                (price_centers >= bar_low) & (price_centers <= bar_high)
            )[0]
            
            if len(intersecting_bins) > 0:
                # Distribute volume based on price position within the bar
                # Give more weight to close price
                close_weight = 0.5
                uniform_weight = 0.5
                
                for bin_idx in intersecting_bins:
                    # Calculate distance-based weight
                    distance_to_close = abs(price_centers[bin_idx] - bar_close)
                    max_distance = bar_high - bar_low if bar_high != bar_low else 1
                    
                    close_factor = max(0, 1 - (distance_to_close / max_distance))
                    uniform_factor = 1.0 / len(intersecting_bins)
                    
                    weight = close_weight * close_factor + uniform_weight * uniform_factor
                    volume_at_price[bin_idx] += bar_volume * weight
        
        # Smooth volume distribution using kernel density estimation
        if self.config.kde_bandwidth is None:
            # Auto-calculate bandwidth using Silverman's rule
            kde_bandwidth = 1.06 * np.std(price_centers) * (len(price_centers) ** (-1/5))
        else:
            kde_bandwidth = self.config.kde_bandwidth
        
        # Apply KDE smoothing
        if np.sum(volume_at_price) > 0:
            # Create weighted data points for KDE
            weighted_prices = []
            for i, vol in enumerate(volume_at_price):
                if vol > 0:
                    # Add multiple instances based on volume
                    instances = max(1, int(vol / np.max(volume_at_price) * 100))
                    weighted_prices.extend([price_centers[i]] * instances)
            
            if len(weighted_prices) > 1:
                kde = gaussian_kde(weighted_prices, bw_method=kde_bandwidth)
                smoothed_volume = kde(price_centers)
                # Normalize to preserve total volume
                volume_at_price = smoothed_volume * (np.sum(volume_at_price) / np.sum(smoothed_volume))
        
        # Calculate cumulative volume
        cumulative_volume = np.cumsum(volume_at_price)
        total_volume = np.sum(volume_at_price)
        
        # Store for future use
        self._price_levels = price_centers
        self._volume_at_price = volume_at_price
        self._cumulative_volume = cumulative_volume
        
        return {
            'price_levels': price_centers,
            'volume_at_price': volume_at_price,
            'cumulative_volume': cumulative_volume,
            'total_volume': total_volume,
            'price_range': price_range,
            'bin_size': price_bins[1] - price_bins[0]
        }
    
    def _calculate_value_area(self, profile_results: Dict[str, Any]) -> ValueArea:
        """Calculate value area using percentile-based analysis"""
        volume_at_price = profile_results['volume_at_price']
        price_levels = profile_results['price_levels']
        total_volume = profile_results['total_volume']
        
        # Find Point of Control (POC)
        poc_idx = np.argmax(volume_at_price)
        poc_price = price_levels[poc_idx]
        poc_volume = volume_at_price[poc_idx]
        
        # Calculate value area (70% of volume)
        target_volume = total_volume * self.config.value_area_percentage
        
        # Start from POC and expand outward
        va_volume = poc_volume
        low_idx = poc_idx
        high_idx = poc_idx
        
        # Expand value area by adding highest volume bins
        remaining_indices = list(range(len(volume_at_price)))
        remaining_indices.remove(poc_idx)
        
        # Sort remaining indices by volume (descending)
        remaining_indices.sort(key=lambda x: volume_at_price[x], reverse=True)
        
        for idx in remaining_indices:
            if va_volume >= target_volume:
                break
            
            va_volume += volume_at_price[idx]
            low_idx = min(low_idx, idx)
            high_idx = max(high_idx, idx)
        
        va_high = price_levels[high_idx]
        va_low = price_levels[low_idx]
        va_percentage = va_volume / total_volume
        
        return ValueArea(
            va_high=va_high,
            va_low=va_low,
            va_volume=va_volume,
            va_percentage=va_percentage,
            poc_price=poc_price,
            poc_volume=poc_volume
        )
    
    def _identify_volume_nodes(self, profile_results: Dict[str, Any], 
                              value_area: ValueArea) -> List[VolumeNode]:
        """Identify high and low volume nodes"""
        volume_at_price = profile_results['volume_at_price']
        price_levels = profile_results['price_levels']
        total_volume = profile_results['total_volume']
        
        nodes = []
        
        # Calculate volume thresholds
        hvn_threshold = np.percentile(volume_at_price, self.config.hvn_threshold_percentile)
        lvn_threshold = np.percentile(volume_at_price, self.config.lvn_threshold_percentile)
        
        # Add POC as a special node
        poc_idx = np.argmax(volume_at_price)
        nodes.append(VolumeNode(
            price_level=value_area.poc_price,
            volume=value_area.poc_volume,
            volume_percentage=value_area.poc_volume / total_volume,
            node_type='poc',
            strength=1.0
        ))
        
        # Add value area boundaries
        va_high_idx = np.argmin(np.abs(price_levels - value_area.va_high))
        va_low_idx = np.argmin(np.abs(price_levels - value_area.va_low))
        
        nodes.append(VolumeNode(
            price_level=value_area.va_high,
            volume=volume_at_price[va_high_idx],
            volume_percentage=volume_at_price[va_high_idx] / total_volume,
            node_type='va_high',
            strength=0.8
        ))
        
        nodes.append(VolumeNode(
            price_level=value_area.va_low,
            volume=volume_at_price[va_low_idx],
            volume_percentage=volume_at_price[va_low_idx] / total_volume,
            node_type='va_low',
            strength=0.8
        ))
        
        # Find peaks for HVN identification
        hvn_peaks, _ = find_peaks(volume_at_price, height=hvn_threshold, distance=3)
        
        for peak_idx in hvn_peaks:
            if peak_idx != poc_idx:  # Don't duplicate POC
                strength = volume_at_price[peak_idx] / value_area.poc_volume
                nodes.append(VolumeNode(
                    price_level=price_levels[peak_idx],
                    volume=volume_at_price[peak_idx],
                    volume_percentage=volume_at_price[peak_idx] / total_volume,
                    node_type='hvn',
                    strength=strength
                ))
        
        # Find valleys for LVN identification
        inverted_volume = np.max(volume_at_price) - volume_at_price
        lvn_peaks, _ = find_peaks(inverted_volume, height=np.max(inverted_volume) - lvn_threshold, distance=3)
        
        for valley_idx in lvn_peaks:
            strength = 1.0 - (volume_at_price[valley_idx] / value_area.poc_volume)
            nodes.append(VolumeNode(
                price_level=price_levels[valley_idx],
                volume=volume_at_price[valley_idx],
                volume_percentage=volume_at_price[valley_idx] / total_volume,
                node_type='lvn',
                strength=strength
            ))
        
        # Sort nodes by price level
        nodes.sort(key=lambda x: x.price_level)
        
        return nodes    
    def _analyze_market_rotation(self, df: pd.DataFrame, 
                               profile_results: Dict[str, Any]) -> MarketRotation:
        """Analyze market rotation patterns"""
        close = df['close'].values
        volume = df['volume'].values
        price_levels = profile_results['price_levels']
        volume_at_price = profile_results['volume_at_price']
        
        # Calculate price momentum and volume distribution
        if len(close) < self.config.rotation_detection_period:
            return MarketRotation(
                rotation_type='normal',
                rotation_strength=0.0,
                dominant_timeframe='unknown',
                imbalance_direction='neutral'
            )
        
        # Analyze price movement relative to volume profile
        recent_period = self.config.rotation_detection_period
        recent_closes = close[-recent_period:]
        recent_volumes = volume[-recent_period:]
        
        # Calculate price trend
        price_trend = np.polyfit(range(len(recent_closes)), recent_closes, 1)[0]
        
        # Find current price position in volume profile
        current_price = close[-1]
        
        # Calculate volume above and below current price
        above_price_mask = price_levels > current_price
        below_price_mask = price_levels <= current_price
        
        volume_above = np.sum(volume_at_price[above_price_mask])
        volume_below = np.sum(volume_at_price[below_price_mask])
        total_volume = volume_above + volume_below
        
        if total_volume > 0:
            volume_imbalance = (volume_above - volume_below) / total_volume
        else:
            volume_imbalance = 0.0
        
        # Determine rotation type
        price_volatility = np.std(recent_closes) / np.mean(recent_closes)
        volume_consistency = 1.0 - (np.std(recent_volumes) / np.mean(recent_volumes))
        
        if abs(price_trend) > price_volatility * 2:
            rotation_type = 'trending'
            rotation_strength = abs(price_trend) / np.mean(recent_closes)
        elif volume_consistency > 0.7 and price_volatility < 0.02:
            rotation_type = 'balancing'
            rotation_strength = volume_consistency
        else:
            rotation_type = 'normal'
            rotation_strength = 0.5
        
        # Determine imbalance direction
        if volume_imbalance > self.config.imbalance_threshold:
            imbalance_direction = 'bearish'  # More volume above price
        elif volume_imbalance < -self.config.imbalance_threshold:
            imbalance_direction = 'bullish'  # More volume below price
        else:
            imbalance_direction = 'neutral'
        
        # Determine dominant timeframe based on volume distribution
        poc_idx = np.argmax(volume_at_price)
        poc_distance_from_current = abs(price_levels[poc_idx] - current_price)
        price_range = np.max(price_levels) - np.min(price_levels)
        
        if price_range > 0:
            poc_distance_ratio = poc_distance_from_current / price_range
            if poc_distance_ratio < 0.1:
                dominant_timeframe = 'short'
            elif poc_distance_ratio < 0.3:
                dominant_timeframe = 'medium'
            else:
                dominant_timeframe = 'long'
        else:
            dominant_timeframe = 'unknown'
        
        return MarketRotation(
            rotation_type=rotation_type,
            rotation_strength=min(1.0, rotation_strength),
            dominant_timeframe=dominant_timeframe,
            imbalance_direction=imbalance_direction
        )
    
    def _detect_institutional_levels(self, profile_results: Dict[str, Any],
                                   volume_nodes: List[VolumeNode]) -> List[float]:
        """Detect institutional trading levels through volume clustering"""
        volume_at_price = profile_results['volume_at_price']
        price_levels = profile_results['price_levels']
        
        # Calculate institutional volume threshold
        avg_volume = np.mean(volume_at_price)
        institutional_threshold = avg_volume * self.config.institutional_volume_threshold
        
        # Find high-volume areas that could be institutional
        institutional_mask = volume_at_price > institutional_threshold
        institutional_prices = price_levels[institutional_mask]
        institutional_volumes = volume_at_price[institutional_mask]
        
        if len(institutional_prices) == 0:
            return []
        
        # Use clustering to identify institutional level groups
        try:
            # Prepare data for clustering
            price_volume_data = np.column_stack([
                institutional_prices,
                institutional_volumes / np.max(institutional_volumes)  # Normalize
            ])
            
            # Use DBSCAN for density-based clustering
            clustering = DBSCAN(eps=0.1, min_samples=2)
            clusters = clustering.fit_predict(price_volume_data)
            
            institutional_levels = []
            
            # Extract representative levels from each cluster
            unique_clusters = set(clusters)
            unique_clusters.discard(-1)  # Remove noise points
            
            for cluster_id in unique_clusters:
                cluster_mask = clusters == cluster_id
                cluster_prices = institutional_prices[cluster_mask]
                cluster_volumes = institutional_volumes[cluster_mask]
                
                # Volume-weighted average price for the cluster
                if np.sum(cluster_volumes) > 0:
                    weighted_price = np.sum(cluster_prices * cluster_volumes) / np.sum(cluster_volumes)
                    institutional_levels.append(weighted_price)
            
            return sorted(institutional_levels)
            
        except Exception as e:
            self.logger.warning(f"Institutional level clustering failed: {e}")
            # Fallback: use high volume nodes
            hvn_levels = [node.price_level for node in volume_nodes if node.node_type == 'hvn']
            return sorted(hvn_levels)
    
    def _analyze_profile_shape(self, profile_results: Dict[str, Any]) -> Dict[str, str]:
        """Analyze volume profile shape characteristics"""
        volume_at_price = profile_results['volume_at_price']
        price_levels = profile_results['price_levels']
        
        if len(volume_at_price) < 10:
            return {'profile_shape': 'insufficient_data'}
        
        # Find peaks in the volume distribution
        peaks, peak_properties = find_peaks(volume_at_price, height=np.max(volume_at_price) * 0.3, distance=5)
        
        # Calculate distribution characteristics
        volume_skewness = stats.skew(volume_at_price)
        volume_kurtosis = stats.kurtosis(volume_at_price)
        
        # Determine profile shape
        if len(peaks) == 1:
            # Single peak distribution
            peak_position = peaks[0] / len(volume_at_price)
            if peak_position < 0.3:
                shape = 'p_shape'  # Peak at bottom (P-shaped)
            elif peak_position > 0.7:
                shape = 'b_shape'  # Peak at top (b-shaped)
            else:
                shape = 'normal'   # Normal distribution
        elif len(peaks) >= 2:
            shape = 'double_distribution'  # Multiple peaks
        else:
            # No clear peaks, analyze based on statistical measures
            if abs(volume_skewness) < 0.5 and volume_kurtosis > -1:
                shape = 'normal'
            elif volume_skewness > 0.5:
                shape = 'p_shape'
            elif volume_skewness < -0.5:
                shape = 'b_shape'
            else:
                shape = 'irregular'
        
        return {'profile_shape': shape}
    
    def _calculate_volume_imbalances(self, profile_results: Dict[str, Any],
                                   value_area: ValueArea) -> Dict[str, float]:
        """Calculate volume imbalances within and outside value area"""
        volume_at_price = profile_results['volume_at_price']
        price_levels = profile_results['price_levels']
        total_volume = profile_results['total_volume']
        
        # Find indices for value area boundaries
        va_high_idx = np.argmin(np.abs(price_levels - value_area.va_high))
        va_low_idx = np.argmin(np.abs(price_levels - value_area.va_low))
        
        # Ensure proper ordering
        if va_low_idx > va_high_idx:
            va_low_idx, va_high_idx = va_high_idx, va_low_idx
        
        # Calculate volume distribution
        volume_below_va = np.sum(volume_at_price[:va_low_idx])
        volume_within_va = np.sum(volume_at_price[va_low_idx:va_high_idx + 1])
        volume_above_va = np.sum(volume_at_price[va_high_idx + 1:])
        
        # Calculate imbalance metrics
        if total_volume > 0:
            below_percentage = volume_below_va / total_volume
            within_percentage = volume_within_va / total_volume
            above_percentage = volume_above_va / total_volume
            
            # Overall imbalance (positive = more volume above VA)
            overall_imbalance = (above_percentage - below_percentage)
            
        else:
            overall_imbalance = 0.0
        
        return {'volume_imbalance': overall_imbalance}
    
    def _identify_support_resistance(self, volume_nodes: List[VolumeNode],
                                   value_area: ValueArea) -> List[float]:
        """Identify key support and resistance levels"""
        sr_levels = []
        
        # Add value area boundaries
        sr_levels.extend([value_area.va_high, value_area.va_low, value_area.poc_price])
        
        # Add high volume nodes (support/resistance)
        hvn_levels = [node.price_level for node in volume_nodes 
                     if node.node_type in ['hvn', 'poc'] and node.strength > 0.5]
        sr_levels.extend(hvn_levels)
        
        # Remove duplicates and sort
        sr_levels = sorted(list(set(sr_levels)))
        
        return sr_levels
    
    def _calculate_signal_confidence(self, value_area: ValueArea,
                                   rotation: MarketRotation,
                                   shape: Dict[str, str]) -> Dict[str, float]:
        """Calculate overall signal confidence"""
        confidence_factors = []
        
        # Value area confidence (higher volume percentage = higher confidence)
        va_confidence = min(1.0, value_area.va_percentage / 0.7)
        confidence_factors.append(va_confidence)
        
        # Rotation confidence
        rotation_confidence = rotation.rotation_strength
        confidence_factors.append(rotation_confidence)
        
        # Shape confidence (normal shapes are more reliable)
        shape_confidence = 0.8 if shape['profile_shape'] == 'normal' else 0.6
        confidence_factors.append(shape_confidence)
        
        # POC volume confidence
        poc_confidence = min(1.0, value_area.poc_volume / (value_area.va_volume * 0.3))
        confidence_factors.append(poc_confidence)
        
        # Calculate weighted average
        overall_confidence = np.mean(confidence_factors)
        
        return {'signal_confidence': overall_confidence}
    
    def _compile_profile_result(self, value_area: ValueArea,
                              volume_nodes: List[VolumeNode],
                              rotation: MarketRotation,
                              institutional_levels: List[float],
                              shape: Dict[str, str],
                              imbalance: Dict[str, float],
                              sr_levels: List[float],
                              confidence: Dict[str, float]) -> VolumeProfileResult:
        """Compile all volume profile analysis results"""
        return VolumeProfileResult(
            value_area=value_area,
            volume_nodes=volume_nodes,
            market_rotation=rotation,
            institutional_levels=institutional_levels,
            profile_shape=shape['profile_shape'],
            volume_imbalance=imbalance['volume_imbalance'],
            support_resistance_levels=sr_levels,
            signal_confidence=confidence['signal_confidence']
        )
    
    def _generate_signal(self, result: VolumeProfileResult, df: pd.DataFrame) -> SignalType:
        """Generate trading signal based on volume profile analysis"""
        current_price = df['close'].iloc[-1]
        
        # Minimum confidence threshold
        if result.signal_confidence < 0.4:
            return SignalType.HOLD
        
        # Signal scoring system
        bullish_score = 0
        bearish_score = 0
        
        # Current price relative to value area
        if current_price < result.value_area.va_low:
            bullish_score += 2  # Below value area - potential bounce
        elif current_price > result.value_area.va_high:
            bearish_score += 2  # Above value area - potential rejection
        elif current_price < result.value_area.poc_price:
            bullish_score += 1  # Below POC within VA
        else:
            bearish_score += 1  # Above POC within VA
        
        # Volume imbalance analysis
        if result.volume_imbalance < -0.2:
            bullish_score += 1  # More volume below (support)
        elif result.volume_imbalance > 0.2:
            bearish_score += 1  # More volume above (resistance)
        
        # Market rotation analysis
        if result.market_rotation.rotation_type == 'trending':
            if result.market_rotation.imbalance_direction == 'bullish':
                bullish_score += 2
            elif result.market_rotation.imbalance_direction == 'bearish':
                bearish_score += 2
        elif result.market_rotation.rotation_type == 'balancing':
            # In balancing market, fade extremes
            if current_price < result.value_area.va_low:
                bullish_score += 1
            elif current_price > result.value_area.va_high:
                bearish_score += 1
        
        # Profile shape analysis
        if result.profile_shape == 'p_shape':
            # P-shape suggests accumulation at lower levels
            bullish_score += 1
        elif result.profile_shape == 'b_shape':
            # b-shape suggests distribution at higher levels
            bearish_score += 1
        
        # Proximity to high volume nodes (support/resistance)
        closest_hvn_distance = float('inf')
        for node in result.volume_nodes:
            if node.node_type == 'hvn':
                distance = abs(current_price - node.price_level) / current_price
                if distance < closest_hvn_distance:
                    closest_hvn_distance = distance
        
        # If near HVN (within 1%), expect bounce or rejection
        if closest_hvn_distance < 0.01:
            if current_price < result.value_area.poc_price:
                bullish_score += 1  # Near support
            else:
                bearish_score += 1  # Near resistance
        
        # Institutional level proximity
        if result.institutional_levels:
            closest_inst_distance = min([abs(current_price - level) / current_price 
                                       for level in result.institutional_levels])
            if closest_inst_distance < 0.005:  # Within 0.5%
                # Direction depends on approach angle and volume
                if result.market_rotation.imbalance_direction == 'bullish':
                    bullish_score += 1
                elif result.market_rotation.imbalance_direction == 'bearish':
                    bearish_score += 1
        
        # Generate final signal
        score_difference = abs(bullish_score - bearish_score)
        min_score_threshold = 3
        
        if (score_difference >= 2 and max(bullish_score, bearish_score) >= min_score_threshold 
            and result.signal_confidence > 0.6):
            if bullish_score > bearish_score:
                return SignalType.BUY
            else:
                return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _update_history(self, df: pd.DataFrame, result: VolumeProfileResult):
        """Update historical data for trend analysis"""
        max_history = 50
        
        # Store current profile data
        profile_data = {
            'timestamp': len(df),
            'poc_price': result.value_area.poc_price,
            'va_high': result.value_area.va_high,
            'va_low': result.value_area.va_low,
            'profile_shape': result.profile_shape,
            'volume_imbalance': result.volume_imbalance
        }
        
        self._profile_history.append(profile_data)
        self._value_area_history.append(result.value_area)
        self._rotation_history.append(result.market_rotation)
        
        # Trim histories
        if len(self._profile_history) > max_history:
            self._profile_history = self._profile_history[-max_history:]
            self._value_area_history = self._value_area_history[-max_history:]
            self._rotation_history = self._rotation_history[-max_history:]
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate input data structure and completeness"""
        required_fields = ['high', 'low', 'close', 'volume']
        
        if not isinstance(data, dict):
            return False
        
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Missing required field: {field}")
                return False
            
            if not isinstance(data[field], (list, np.ndarray)) or len(data[field]) == 0:
                self.logger.error(f"Invalid data for field: {field}")
                return False
        
        # Check data consistency
        lengths = [len(data[field]) for field in required_fields]
        if len(set(lengths)) > 1:
            self.logger.error("Inconsistent data lengths across fields")
            return False
        
        return True
    
    def _create_default_result(self) -> Dict[str, Any]:
        """Create default result for insufficient data"""
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.0,
            'poc_price': 0.0,
            'value_area_high': 0.0,
            'value_area_low': 0.0,
            'profile_shape': 'insufficient_data',
            'rotation_type': 'unknown',
            'volume_imbalance': 0.0,
            'institutional_levels': [],
            'support_resistance_levels': [],
            'high_volume_nodes': [],
            'low_volume_nodes': [],
            'metadata': {
                'indicator_name': 'VolumeProfile',
                'error': 'Insufficient data for calculation'
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure"""
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.0,
            'poc_price': 0.0,
            'value_area_high': 0.0,
            'value_area_low': 0.0,
            'profile_shape': 'error',
            'rotation_type': 'unknown',
            'volume_imbalance': 0.0,
            'institutional_levels': [],
            'support_resistance_levels': [],
            'high_volume_nodes': [],
            'low_volume_nodes': [],
            'metadata': {
                'indicator_name': 'VolumeProfile',
                'error': error_message
            }
        }