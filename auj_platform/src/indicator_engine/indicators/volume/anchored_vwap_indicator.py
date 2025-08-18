"""
Advanced Anchored VWAP Indicator
==============================

A sophisticated implementation of Anchored Volume Weighted Average Price with:
- Multiple anchor point strategies
- Statistical bands and confidence intervals
- Adaptive volume profiling
- Institutional level detection
- Multi-timeframe analysis

Mathematical Foundation:
VWAP = Σ(Price × Volume) / Σ(Volume)

Where Price can be:
- Typical Price: (High + Low + Close) / 3
- OHLC4: (Open + High + Low + Close) / 4
- Close price for simplicity

Advanced Features:
1. Multiple anchor point strategies (time-based, volume-based, volatility-based)
2. Statistical bands using standard deviation
3. Volume profile integration
4. Institutional level detection
5. Adaptive period adjustment based on market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import base class
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from ..indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface
from ..core.signal_type import SignalType

@dataclass
class AnchoredVWAPConfig:
    """Configuration for Anchored VWAP calculation"""
    default_anchor_period: int = 20
    max_anchor_periods: int = 5
    volume_profile_bins: int = 50
    statistical_bands_multiplier: float = 2.0
    institutional_threshold_percentile: float = 95.0
    adaptive_period_range: Tuple[int, int] = (10, 100)
    volume_weight_decay: float = 0.95
    price_type: str = 'typical'  # 'typical', 'ohlc4', 'close'
    
@dataclass
class AnchorPoint:
    """Individual anchor point configuration"""
    timestamp: int
    anchor_type: str  # 'time', 'volume', 'volatility', 'event'
    weight: float
    volume_threshold: float
    price_level: float

@dataclass
class VWAPResult:
    """Result structure for Anchored VWAP analysis"""
    anchored_vwap: float
    vwap_bands_upper: float
    vwap_bands_lower: float
    volume_profile: Dict[str, float]
    institutional_levels: List[float]
    anchor_strength: float
    price_deviation: float
    volume_concentration: float
    signal_confidence: float

class AnchoredVWAPIndicator(StandardIndicatorInterface):
    """
    Advanced Anchored VWAP Indicator
    
    This indicator provides sophisticated VWAP analysis through:
    1. Multiple anchor point strategies for different market conditions
    2. Statistical bands for overbought/oversold identification
    3. Volume profile integration for support/resistance levels
    4. Institutional level detection through volume clustering
    5. Adaptive period adjustment based on market volatility
    """
    
    def __init__(self, config: Optional[AnchoredVWAPConfig] = None):
        """Initialize the Anchored VWAP Indicator"""
        super().__init__()
        self.config = config or AnchoredVWAPConfig()
        self.logger = logging.getLogger(__name__)
        
        # Anchor points management
        self.anchor_points: List[AnchorPoint] = []
        self.current_anchors: List[Dict[str, Any]] = []
        
        # Historical data storage
        self._price_history: List[float] = []
        self._volume_history: List[float] = []
        self._vwap_history: List[float] = []
        self._timestamp_history: List[int] = []
        
        # Volume profile data
        self._volume_profile: Dict[str, List[float]] = {
            'price_levels': [],
            'volume_at_price': [],
            'cumulative_volume': []
        }
        
        # Institutional analysis
        self._institutional_levels: List[float] = []
        self._volume_clusters: List[Dict[str, float]] = []
        
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate Anchored VWAP with comprehensive analysis
        
        Args:
            data: Dictionary containing OHLCV data and optional timestamp
            
        Returns:
            Dictionary with Anchored VWAP analysis results
        """
        try:
            # Validate and extract data
            if not self._validate_data(data):
                raise ValueError("Invalid or insufficient data provided")
            
            df = pd.DataFrame(data)
            
            # Ensure minimum data requirements
            if len(df) < self.config.default_anchor_period:
                return self._create_default_result()
            
            # Update anchor points based on market conditions
            self._update_anchor_points(df)
            
            # Calculate anchored VWAP for each anchor point
            vwap_results = self._calculate_anchored_vwap(df)
            
            # Calculate statistical bands
            bands_results = self._calculate_statistical_bands(df, vwap_results)
            
            # Build volume profile
            profile_results = self._build_volume_profile(df)
            
            # Detect institutional levels
            institutional_results = self._detect_institutional_levels(df, profile_results)
            
            # Calculate signal metrics
            signal_results = self._calculate_signal_metrics(
                df, vwap_results, bands_results, institutional_results
            )
            
            # Compile final result
            result = self._compile_vwap_result(
                vwap_results, bands_results, profile_results,
                institutional_results, signal_results
            )
            
            # Update historical data
            self._update_history(df, result)
            
            # Generate trading signal
            signal = self._generate_signal(result, df)
            
            return {
                'signal': signal,
                'confidence': result.signal_confidence,
                'anchored_vwap': result.anchored_vwap,
                'vwap_upper_band': result.vwap_bands_upper,
                'vwap_lower_band': result.vwap_bands_lower,
                'price_deviation': result.price_deviation,
                'volume_concentration': result.volume_concentration,
                'anchor_strength': result.anchor_strength,
                'institutional_levels': result.institutional_levels,
                'volume_profile': result.volume_profile,
                'metadata': {
                    'indicator_name': 'AnchoredVWAP',
                    'calculation_method': 'multi_anchor_analysis',
                    'active_anchors': len(self.current_anchors),
                    'parameters': {
                        'anchor_period': self.config.default_anchor_period,
                        'price_type': self.config.price_type,
                        'volume_bins': self.config.volume_profile_bins
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Anchored VWAP: {str(e)}")
            return self._create_error_result(str(e))
    
    def _update_anchor_points(self, df: pd.DataFrame):
        """Update anchor points based on market conditions"""
        current_time = len(df) - 1  # Use index as timestamp proxy
        
        # Clear old anchors beyond max periods
        max_lookback = max(self.config.adaptive_period_range)
        self.current_anchors = [
            anchor for anchor in self.current_anchors
            if current_time - anchor['start_index'] <= max_lookback
        ]
        
        # Add time-based anchor
        if (len(self.current_anchors) == 0 or 
            current_time - self.current_anchors[-1]['start_index'] >= self.config.default_anchor_period):
            self._add_time_anchor(df, current_time)
        
        # Add volume-based anchors
        self._add_volume_anchors(df, current_time)
        
        # Add volatility-based anchors
        self._add_volatility_anchors(df, current_time)
        
        # Limit total number of anchors
        if len(self.current_anchors) > self.config.max_anchor_periods:
            # Keep most recent and highest weighted anchors
            self.current_anchors.sort(key=lambda x: (x['timestamp'], x['weight']), reverse=True)
            self.current_anchors = self.current_anchors[:self.config.max_anchor_periods]
    
    def _add_time_anchor(self, df: pd.DataFrame, current_time: int):
        """Add time-based anchor point"""
        price = self._get_price(df, current_time)
        volume = df['volume'].iloc[current_time] if current_time < len(df) else 0
        
        anchor = {
            'start_index': current_time,
            'timestamp': current_time,
            'type': 'time',
            'weight': 1.0,
            'price_level': price,
            'volume_threshold': volume * 1.5
        }
        
        self.current_anchors.append(anchor)
    
    def _add_volume_anchors(self, df: pd.DataFrame, current_time: int):
        """Add volume-based anchor points for high volume periods"""
        if len(df) < 20:
            return
        
        # Look for volume spikes in recent periods
        volume = df['volume'].values
        recent_period = min(20, len(volume))
        recent_volume = volume[-recent_period:]
        
        # Calculate volume threshold (95th percentile)
        volume_threshold = np.percentile(volume, self.config.institutional_threshold_percentile)
        
        # Check if current volume is significantly high
        if volume[-1] > volume_threshold:
            price = self._get_price(df, current_time)
            
            # Calculate weight based on volume relative to threshold
            weight = min(2.0, volume[-1] / volume_threshold)
            
            anchor = {
                'start_index': current_time,
                'timestamp': current_time,
                'type': 'volume',
                'weight': weight,
                'price_level': price,
                'volume_threshold': volume[-1]
            }
            
            self.current_anchors.append(anchor)
    
    def _add_volatility_anchors(self, df: pd.DataFrame, current_time: int):
        """Add volatility-based anchor points for significant price movements"""
        if len(df) < 10:
            return
        
        # Calculate recent volatility
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        if len(close) < 10:
            return
        
        # True Range calculation
        tr1 = high[-10:] - low[-10:]
        tr2 = np.abs(high[-10:] - close[-11:-1])
        tr3 = np.abs(low[-10:] - close[-11:-1])
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        avg_true_range = np.mean(true_range)
        
        # Check for volatility spike
        current_tr = max(
            high[-1] - low[-1],
            abs(high[-1] - close[-2]) if len(close) > 1 else 0,
            abs(low[-1] - close[-2]) if len(close) > 1 else 0
        )
        
        volatility_threshold = avg_true_range * 2.0
        
        if current_tr > volatility_threshold:
            price = self._get_price(df, current_time)
            weight = min(1.5, current_tr / volatility_threshold)
            
            anchor = {
                'start_index': current_time,
                'timestamp': current_time,
                'type': 'volatility',
                'weight': weight,
                'price_level': price,
                'volume_threshold': df['volume'].iloc[current_time] if current_time < len(df) else 0
            }
            
            self.current_anchors.append(anchor)
    
    def _calculate_anchored_vwap(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate VWAP from multiple anchor points"""
        if not self.current_anchors:
            return {'vwap': self._get_price(df, -1), 'weighted_vwap': self._get_price(df, -1)}
        
        vwap_values = []
        weights = []
        
        for anchor in self.current_anchors:
            # Calculate VWAP from anchor point to current
            start_idx = anchor['start_index']
            
            if start_idx >= len(df):
                continue
            
            # Extract data from anchor point onwards
            anchor_data = df.iloc[start_idx:].copy()
            
            if len(anchor_data) == 0:
                continue
            
            # Calculate VWAP for this anchor
            prices = self._get_price_series(anchor_data)
            volumes = anchor_data['volume'].values
            
            # Apply volume weight decay for older periods
            decay_factors = np.power(self.config.volume_weight_decay, 
                                   np.arange(len(volumes)))
            adjusted_volumes = volumes * decay_factors
            
            # Calculate VWAP
            if np.sum(adjusted_volumes) > 0:
                anchor_vwap = np.sum(prices * adjusted_volumes) / np.sum(adjusted_volumes)
                vwap_values.append(anchor_vwap)
                weights.append(anchor['weight'])
        
        if not vwap_values:
            current_price = self._get_price(df, -1)
            return {'vwap': current_price, 'weighted_vwap': current_price}
        
        # Calculate weighted average of all anchor VWAPs
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        weighted_vwap = np.sum(np.array(vwap_values) * weights)
        simple_vwap = np.mean(vwap_values)
        
        return {
            'vwap': simple_vwap,
            'weighted_vwap': weighted_vwap
        }
    
    def _calculate_statistical_bands(self, df: pd.DataFrame, 
                                   vwap_results: Dict[str, float]) -> Dict[str, float]:
        """Calculate statistical bands around VWAP"""
        vwap = vwap_results['weighted_vwap']
        
        # Calculate price deviations from VWAP using recent data
        recent_period = min(50, len(df))
        recent_data = df.tail(recent_period)
        
        prices = self._get_price_series(recent_data)
        volumes = recent_data['volume'].values
        
        # Volume-weighted standard deviation
        price_deviations = prices - vwap
        squared_deviations = price_deviations ** 2
        
        # Weight by volume
        volume_weights = volumes / np.sum(volumes)
        weighted_variance = np.sum(squared_deviations * volume_weights)
        weighted_std = np.sqrt(weighted_variance)
        
        # Calculate bands
        multiplier = self.config.statistical_bands_multiplier
        upper_band = vwap + (multiplier * weighted_std)
        lower_band = vwap - (multiplier * weighted_std)
        
        return {
            'upper_band': upper_band,
            'lower_band': lower_band,
            'std_dev': weighted_std
        }
    
    def _build_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build volume profile for current period"""
        recent_period = min(100, len(df))
        recent_data = df.tail(recent_period)
        
        prices = self._get_price_series(recent_data)
        volumes = recent_data['volume'].values
        
        # Create price bins
        price_min, price_max = np.min(prices), np.max(prices)
        if price_max == price_min:
            price_max = price_min + 0.01  # Avoid division by zero
        
        price_bins = np.linspace(price_min, price_max, self.config.volume_profile_bins)
        
        # Assign volumes to price bins
        volume_at_price = np.zeros(len(price_bins) - 1)
        
        for price, volume in zip(prices, volumes):
            bin_idx = np.searchsorted(price_bins[:-1], price, side='right') - 1
            bin_idx = max(0, min(bin_idx, len(volume_at_price) - 1))
            volume_at_price[bin_idx] += volume
        
        # Find high volume nodes (POC - Point of Control)
        poc_idx = np.argmax(volume_at_price)
        poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
        
        # Calculate value area (70% of volume)
        total_volume = np.sum(volume_at_price)
        value_area_volume = total_volume * 0.7
        
        # Find value area high and low
        sorted_indices = np.argsort(volume_at_price)[::-1]
        cumulative_volume = 0
        value_area_indices = []
        
        for idx in sorted_indices:
            cumulative_volume += volume_at_price[idx]
            value_area_indices.append(idx)
            if cumulative_volume >= value_area_volume:
                break
        
        value_area_high = np.max([(price_bins[i] + price_bins[i + 1]) / 2 
                                 for i in value_area_indices])
        value_area_low = np.min([(price_bins[i] + price_bins[i + 1]) / 2 
                                for i in value_area_indices])
        
        return {
            'poc_price': poc_price,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'price_bins': price_bins,
            'volume_at_price': volume_at_price,
            'total_volume': total_volume
        }
    
    def _detect_institutional_levels(self, df: pd.DataFrame, 
                                   profile_results: Dict[str, Any]) -> Dict[str, List[float]]:
        """Detect institutional trading levels using volume clustering"""
        volume_at_price = profile_results['volume_at_price']
        price_bins = profile_results['price_bins']
        
        # Find volume clusters using K-means
        if len(volume_at_price) < 3:
            return {'institutional_levels': []}
        
        # Prepare data for clustering
        price_centers = [(price_bins[i] + price_bins[i + 1]) / 2 
                        for i in range(len(volume_at_price))]
        
        # Create features: [price, volume, volume_density]
        features = np.column_stack([
            price_centers,
            volume_at_price,
            volume_at_price / np.max(volume_at_price)  # Normalized volume
        ])
        
        # Apply clustering
        n_clusters = min(5, len(features) // 3)
        if n_clusters < 2:
            return {'institutional_levels': []}
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features)
            
            # Identify high-volume clusters as institutional levels
            institutional_levels = []
            
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                cluster_volumes = volume_at_price[cluster_mask]
                
                # Check if cluster has high volume concentration
                avg_cluster_volume = np.mean(cluster_volumes)
                volume_threshold = np.percentile(volume_at_price, 
                                               self.config.institutional_threshold_percentile)
                
                if avg_cluster_volume > volume_threshold:
                    cluster_prices = np.array(price_centers)[cluster_mask]
                    # Use volume-weighted average price for the level
                    weights = cluster_volumes / np.sum(cluster_volumes)
                    institutional_level = np.sum(cluster_prices * weights)
                    institutional_levels.append(institutional_level)
            
            return {'institutional_levels': sorted(institutional_levels)}
            
        except Exception as e:
            self.logger.warning(f"Clustering failed: {e}")
            return {'institutional_levels': []}
    
    def _calculate_signal_metrics(self, df: pd.DataFrame, vwap_results: Dict[str, float],
                                bands_results: Dict[str, float], 
                                institutional_results: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate various signal strength metrics"""
        current_price = self._get_price(df, -1)
        vwap = vwap_results['weighted_vwap']
        
        # Price deviation from VWAP
        price_deviation = (current_price - vwap) / vwap if vwap != 0 else 0
        
        # Position relative to bands
        upper_band = bands_results['upper_band']
        lower_band = bands_results['lower_band']
        
        band_position = 0.0
        if upper_band != lower_band:
            band_position = (current_price - lower_band) / (upper_band - lower_band)
        
        # Volume concentration (how much volume is near current price)
        recent_volume = df['volume'].tail(10).mean() if len(df) >= 10 else df['volume'].iloc[-1]
        avg_volume = df['volume'].mean()
        volume_concentration = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Anchor strength (average weight of active anchors)
        anchor_strength = np.mean([anchor['weight'] for anchor in self.current_anchors]) if self.current_anchors else 0.0
        
        # Proximity to institutional levels
        institutional_proximity = 0.0
        if institutional_results['institutional_levels']:
            distances = [abs(current_price - level) / current_price 
                        for level in institutional_results['institutional_levels']]
            min_distance = min(distances)
            institutional_proximity = max(0, 1 - (min_distance * 100))  # Closer = higher score
        
        # Combined signal confidence
        signal_confidence = np.mean([
            min(1.0, volume_concentration),
            min(1.0, anchor_strength),
            min(1.0, institutional_proximity),
            min(1.0, abs(price_deviation) * 2)  # Higher deviation = higher confidence
        ])
        
        return {
            'price_deviation': price_deviation,
            'band_position': band_position,
            'volume_concentration': volume_concentration,
            'anchor_strength': anchor_strength,
            'institutional_proximity': institutional_proximity,
            'signal_confidence': signal_confidence
        }
    
    def _get_price(self, df: pd.DataFrame, index: int) -> float:
        """Get price based on configured price type"""
        if index >= len(df) or index < -len(df):
            return 0.0
        
        row = df.iloc[index]
        
        if self.config.price_type == 'typical':
            return (row['high'] + row['low'] + row['close']) / 3
        elif self.config.price_type == 'ohlc4':
            return (row['open'] + row['high'] + row['low'] + row['close']) / 4
        else:  # 'close'
            return row['close']
    
    def _get_price_series(self, df: pd.DataFrame) -> np.ndarray:
        """Get price series based on configured price type"""
        if self.config.price_type == 'typical':
            return (df['high'] + df['low'] + df['close']).values / 3
        elif self.config.price_type == 'ohlc4':
            return (df['open'] + df['high'] + df['low'] + df['close']).values / 4
        else:  # 'close'
            return df['close'].values    
    def _compile_vwap_result(self, vwap_results: Dict[str, float], 
                           bands_results: Dict[str, float],
                           profile_results: Dict[str, Any],
                           institutional_results: Dict[str, List[float]],
                           signal_results: Dict[str, float]) -> VWAPResult:
        """Compile all VWAP analysis results into final structure"""
        return VWAPResult(
            anchored_vwap=vwap_results['weighted_vwap'],
            vwap_bands_upper=bands_results['upper_band'],
            vwap_bands_lower=bands_results['lower_band'],
            volume_profile={
                'poc_price': profile_results['poc_price'],
                'value_area_high': profile_results['value_area_high'],
                'value_area_low': profile_results['value_area_low']
            },
            institutional_levels=institutional_results['institutional_levels'],
            anchor_strength=signal_results['anchor_strength'],
            price_deviation=signal_results['price_deviation'],
            volume_concentration=signal_results['volume_concentration'],
            signal_confidence=signal_results['signal_confidence']
        )
    
    def _generate_signal(self, result: VWAPResult, df: pd.DataFrame) -> SignalType:
        """Generate trading signal based on VWAP analysis"""
        current_price = self._get_price(df, -1)
        
        # Minimum confidence threshold
        if result.signal_confidence < 0.4:
            return SignalType.HOLD
        
        # Signal scoring system
        bullish_score = 0
        bearish_score = 0
        
        # VWAP position analysis
        if current_price > result.anchored_vwap:
            bullish_score += 1
        else:
            bearish_score += 1
        
        # Band position analysis
        band_range = result.vwap_bands_upper - result.vwap_bands_lower
        if band_range > 0:
            distance_from_lower = current_price - result.vwap_bands_lower
            distance_from_upper = result.vwap_bands_upper - current_price
            
            # Near lower band (potential support)
            if distance_from_lower < band_range * 0.2:
                bullish_score += 1
            # Near upper band (potential resistance)
            elif distance_from_upper < band_range * 0.2:
                bearish_score += 1
        
        # Volume profile analysis
        poc_price = result.volume_profile['poc_price']
        value_area_high = result.volume_profile['value_area_high']
        value_area_low = result.volume_profile['value_area_low']
        
        # Price relative to value area
        if current_price < value_area_low:
            bullish_score += 1  # Below value area, potential bounce
        elif current_price > value_area_high:
            bearish_score += 1  # Above value area, potential rejection
        
        # Institutional level proximity
        if result.institutional_levels:
            for level in result.institutional_levels:
                distance_pct = abs(current_price - level) / current_price
                if distance_pct < 0.005:  # Within 0.5%
                    # Near institutional level, direction depends on momentum
                    if result.price_deviation > 0:
                        bearish_score += 1  # At resistance
                    else:
                        bullish_score += 1  # At support
        
        # Volume concentration factor
        if result.volume_concentration > 1.5:
            # High volume confirms the direction
            if result.price_deviation > 0:
                bullish_score += 1
            else:
                bearish_score += 1
        
        # Anchor strength factor
        if result.anchor_strength > 1.0:
            # Strong anchors provide reliable signals
            if bullish_score > bearish_score:
                bullish_score += 1
            elif bearish_score > bullish_score:
                bearish_score += 1
        
        # Generate final signal
        score_difference = abs(bullish_score - bearish_score)
        if score_difference >= 2 and result.signal_confidence > 0.6:
            if bullish_score > bearish_score:
                return SignalType.BUY
            else:
                return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _update_history(self, df: pd.DataFrame, result: VWAPResult):
        """Update historical data for future analysis"""
        max_history = 500  # Keep reasonable history length
        
        # Update histories
        current_price = self._get_price(df, -1)
        self._price_history.append(current_price)
        self._volume_history.append(df['volume'].iloc[-1])
        self._vwap_history.append(result.anchored_vwap)
        self._timestamp_history.append(len(df) - 1)
        
        # Update volume profile
        self._volume_profile['price_levels'].append(current_price)
        self._volume_profile['volume_at_price'].append(df['volume'].iloc[-1])
        
        # Update institutional levels
        self._institutional_levels = result.institutional_levels
        
        # Trim histories to maximum length
        if len(self._price_history) > max_history:
            self._price_history = self._price_history[-max_history:]
            self._volume_history = self._volume_history[-max_history:]
            self._vwap_history = self._vwap_history[-max_history:]
            self._timestamp_history = self._timestamp_history[-max_history:]
            
            self._volume_profile['price_levels'] = self._volume_profile['price_levels'][-max_history:]
            self._volume_profile['volume_at_price'] = self._volume_profile['volume_at_price'][-max_history:]
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate input data structure and completeness"""
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        
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
            'anchored_vwap': 0.0,
            'vwap_upper_band': 0.0,
            'vwap_lower_band': 0.0,
            'price_deviation': 0.0,
            'volume_concentration': 0.0,
            'anchor_strength': 0.0,
            'institutional_levels': [],
            'volume_profile': {
                'poc_price': 0.0,
                'value_area_high': 0.0,
                'value_area_low': 0.0
            },
            'metadata': {
                'indicator_name': 'AnchoredVWAP',
                'error': 'Insufficient data for calculation'
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure"""
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.0,
            'anchored_vwap': 0.0,
            'vwap_upper_band': 0.0,
            'vwap_lower_band': 0.0,
            'price_deviation': 0.0,
            'volume_concentration': 0.0,
            'anchor_strength': 0.0,
            'institutional_levels': [],
            'volume_profile': {
                'poc_price': 0.0,
                'value_area_high': 0.0,
                'value_area_low': 0.0
            },
            'metadata': {
                'indicator_name': 'AnchoredVWAP',
                'error': error_message
            }
        }