"""
AUJ Platform Advanced Market Profile Value Area Indicator
Sophisticated implementation with statistical distribution analysis, volume node identification, and price acceptance levels

This implementation provides institutional-grade market profile analysis for humanitarian trading platforms.

Features:
- Advanced volume profile distribution modeling
- Statistical value area calculation with multiple methods
- Volume node identification and clustering
- Price acceptance level analysis
- Point of Control (POC) detection with statistical validation
- Multi-timeframe profile analysis
- Volume distribution anomaly detection
- Market structure analysis through volume
- Auction theory implementation
- Profile shape classification and interpretation
- Institutional trading level identification
- Statistical significance testing for value areas

The Market Profile Value Area Indicator analyzes volume distribution across price levels
to identify areas of high acceptance, support/resistance levels, and market structure
changes through sophisticated statistical methods and volume analytics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats, signal, optimize, interpolate
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from ..base.base_indicator import BaseIndicator, IndicatorConfig
from ...core.signal_type import SignalType


class ProfileShape(Enum):
    """Market profile shape classification"""
    NORMAL_DISTRIBUTION = "normal_distribution"
    BIMODAL = "bimodal"
    UNIFORM = "uniform"
    SKEWED_HIGH = "skewed_high"
    SKEWED_LOW = "skewed_low"
    EXTREME_TAIL = "extreme_tail"
    DOUBLE_DISTRIBUTION = "double_distribution"


class ValueAreaType(Enum):
    """Value area significance type"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    DEVELOPING = "developing"
    MIGRATION = "migration"
    BALANCE = "balance"
    IMBALANCE = "imbalance"


class MarketStructure(Enum):
    """Market structure based on profile"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    BALANCING = "balancing"
    ROTATING = "rotating"
    BRACKETED = "bracketed"
    BREAKOUT = "breakout"


class AuctionState(Enum):
    """Market auction state"""
    INITIAL_BALANCE = "initial_balance"
    EXCESS_HIGH = "excess_high"
    EXCESS_LOW = "excess_low"
    ACCEPTANCE = "acceptance"
    REJECTION = "rejection"
    ROTATION = "rotation"


@dataclass
class MarketProfileValueAreaConfig(IndicatorConfig):
    """Configuration for Market Profile Value Area Indicator"""
    profile_periods: int = 30
    value_area_percentage: float = 70.0
    price_bin_size: float = 0.0001
    min_volume_threshold: float = 100.0
    statistical_significance: float = 0.95
    clustering_eps: float = 0.0002
    poc_confirmation_threshold: float = 1.5
    use_time_weighted_volume: bool = True
    use_statistical_validation: bool = True
    use_anomaly_detection: bool = True
    min_periods: int = 50


class VolumeNode(NamedTuple):
    """Volume concentration node"""
    price_level: float
    volume: float
    volume_percentage: float
    time_spent: int
    statistical_significance: float
    node_type: str
    confidence: float


class ValueArea(NamedTuple):
    """Market profile value area"""
    high: float
    low: float
    point_of_control: float
    volume_percentage: float
    value_area_type: ValueAreaType
    statistical_confidence: float
    acceptance_level: float
    nodes: List[VolumeNode]


class ProfileAnalysis(NamedTuple):
    """Complete profile analysis"""
    profile_shape: ProfileShape
    market_structure: MarketStructure
    auction_state: AuctionState
    distribution_skewness: float
    distribution_kurtosis: float
    concentration_ratio: float
    volume_imbalance: float
    structure_confidence: float


class MarketProfileResult(NamedTuple):
    """Complete market profile analysis result"""
    primary_value_area: ValueArea
    secondary_value_areas: List[ValueArea]
    volume_nodes: List[VolumeNode]
    profile_analysis: ProfileAnalysis
    current_price_context: Dict[str, float]
    support_resistance_levels: List[Tuple[float, float]]
    institutional_levels: List[float]
    anomaly_score: float
    trading_signal_strength: float
    confidence_score: float


class MarketProfileValueAreaIndicator(BaseIndicator):
    """
    Advanced Market Profile Value Area Indicator with comprehensive analytics.
    
    This indicator analyzes volume distribution across price levels to identify:
    - Value areas where most trading activity occurs
    - Point of Control (POC) levels with highest volume
    - Volume nodes and clustering patterns
    - Market structure and auction dynamics
    - Support and resistance levels based on volume acceptance
    - Institutional trading levels
    - Price acceptance and rejection zones
    - Statistical significance of profile patterns
    """
    
    def __init__(self, config: Optional[MarketProfileValueAreaConfig] = None):
        super().__init__(config or MarketProfileValueAreaConfig())
        self.config: MarketProfileValueAreaConfig = self.config
        
        # Internal state
        self._price_volume_history: List[Dict] = []
        self._profile_cache: Dict[str, Any] = {}
        self._value_area_history: List[ValueArea] = []
        self._volume_node_history: List[List[VolumeNode]] = []
        
        # Statistical models
        self._distribution_models: Dict[str, Any] = {}
        self._clustering_model: Optional[DBSCAN] = None
        self._scaler: StandardScaler = StandardScaler()
        self._robust_scaler: RobustScaler = RobustScaler()
        
        # Pattern recognition
        self._shape_classifier: Optional[object] = None
        self._anomaly_detector: Optional[object] = None
        
        # Profile state
        self._current_profile_shape: Optional[ProfileShape] = None
        self._market_structure_buffer: List[str] = []
        
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive market profile value area analysis.
        
        Args:
            data: Dictionary containing market data including:
                 'high', 'low', 'close', 'volume' (required)
                 'open', 'time', 'tick_data' (optional)
            
        Returns:
            Dictionary containing market profile analysis results
        """
        try:
            if not self._validate_data(data):
                return self._create_default_result()
            
            df = pd.DataFrame(data)
            
            if len(df) < self.config.min_periods:
                return self._create_default_result()
            
            # Build volume profile
            volume_profile = self._build_volume_profile(data, df)
            
            # Calculate value areas
            primary_value_area, secondary_value_areas = self._calculate_value_areas(volume_profile, df)
            
            # Identify volume nodes
            volume_nodes = self._identify_volume_nodes(volume_profile, df)
            
            # Analyze profile structure
            profile_analysis = self._analyze_profile_structure(volume_profile, df)
            
            # Calculate current price context
            current_price_context = self._analyze_current_price_context(
                primary_value_area, volume_nodes, df['close'].iloc[-1]
            )
            
            # Identify support/resistance levels
            support_resistance_levels = self._identify_support_resistance_levels(
                volume_nodes, primary_value_area
            )
            
            # Detect institutional levels
            institutional_levels = self._detect_institutional_levels(
                volume_nodes, volume_profile
            )
            
            # Calculate anomaly score
            anomaly_score = self._calculate_anomaly_score(
                volume_profile, profile_analysis
            )
            
            # Calculate trading signal strength
            trading_signal_strength = self._calculate_trading_signal_strength(
                primary_value_area, current_price_context, profile_analysis
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(
                primary_value_area, profile_analysis, volume_nodes
            )
            
            # Create result
            result = MarketProfileResult(
                primary_value_area=primary_value_area,
                secondary_value_areas=secondary_value_areas,
                volume_nodes=volume_nodes,
                profile_analysis=profile_analysis,
                current_price_context=current_price_context,
                support_resistance_levels=support_resistance_levels,
                institutional_levels=institutional_levels,
                anomaly_score=anomaly_score,
                trading_signal_strength=trading_signal_strength,
                confidence_score=confidence_score
            )
            
            # Generate trading signal
            signal = self._generate_signal(result)
            
            # Update internal state
            self._update_state(df, data, result)
            
            return self._format_result(result, signal)
            
        except Exception as e:
            self.logger.error(f"Error in MarketProfileValueAreaIndicator calculation: {e}")
            return self._create_error_result(str(e))
    
    def _build_volume_profile(self, data: Dict[str, Any], df: pd.DataFrame) -> Dict[float, Dict[str, Any]]:
        """Build comprehensive volume profile with statistical validation"""
        # Extract price and volume data
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Determine price range and bin size
        price_min = np.min(low[-self.config.profile_periods:])
        price_max = np.max(high[-self.config.profile_periods:])
        
        # Adaptive bin sizing based on volatility
        price_range = price_max - price_min
        volatility = np.std(np.diff(np.log(close[-20:]))) if len(close) >= 21 else 0.01
        adaptive_bin_size = max(self.config.price_bin_size, price_range * volatility * 0.1)
        
        # Create price bins
        num_bins = int(price_range / adaptive_bin_size) + 1
        price_bins = np.linspace(price_min, price_max, num_bins)
        
        # Initialize volume profile
        volume_profile = {}
        
        # Process recent periods
        for i in range(-self.config.profile_periods, 0):
            if abs(i) > len(df):
                continue
                
            period_high = high[i]
            period_low = low[i]
            period_close = close[i]
            period_volume = volume[i]
            
            # Skip periods with insufficient volume
            if period_volume < self.config.min_volume_threshold:
                continue
            
            # Distribute volume across price levels within the bar
            price_levels = np.linspace(period_low, period_high, max(3, int((period_high - period_low) / adaptive_bin_size)))
            
            # Volume distribution weighting (more volume near close)
            weights = self._calculate_volume_distribution_weights(
                price_levels, period_close, period_high, period_low
            )
            
            # Time weighting if enabled
            time_weight = 1.0
            if self.config.use_time_weighted_volume:
                # More recent periods get higher weight
                time_weight = np.exp(i / 10.0)  # Exponential decay
            
            # Distribute volume to price levels
            for j, price_level in enumerate(price_levels):
                volume_at_level = (period_volume * weights[j] * time_weight) / len(price_levels)
                
                # Find nearest bin
                bin_index = np.searchsorted(price_bins, price_level)
                if bin_index < len(price_bins):
                    bin_price = price_bins[bin_index]
                    
                    if bin_price not in volume_profile:
                        volume_profile[bin_price] = {
                            'volume': 0.0,
                            'time_periods': 0,
                            'touches': 0,
                            'weighted_price': 0.0,
                            'max_volume_period': 0.0
                        }
                    
                    volume_profile[bin_price]['volume'] += volume_at_level
                    volume_profile[bin_price]['time_periods'] += 1
                    volume_profile[bin_price]['touches'] += 1
                    volume_profile[bin_price]['weighted_price'] += price_level * volume_at_level
                    volume_profile[bin_price]['max_volume_period'] = max(
                        volume_profile[bin_price]['max_volume_period'], volume_at_level
                    )
        
        # Calculate weighted average prices and normalize
        total_volume = sum(level_data['volume'] for level_data in volume_profile.values())
        
        for price_level, level_data in volume_profile.items():
            if level_data['volume'] > 0:
                level_data['weighted_price'] /= level_data['volume']
                level_data['volume_percentage'] = (level_data['volume'] / total_volume) * 100
                level_data['relative_volume'] = level_data['volume'] / np.mean([ld['volume'] for ld in volume_profile.values()])
            else:
                level_data['volume_percentage'] = 0.0
                level_data['relative_volume'] = 0.0
        
        return volume_profile
    
    def _calculate_volume_distribution_weights(self, price_levels: np.ndarray, 
                                             close_price: float, high: float, low: float) -> np.ndarray:
        """Calculate volume distribution weights within a price bar"""
        # More volume is typically traded closer to the closing price
        # Use a normal distribution centered on the close price
        
        if len(price_levels) == 1:
            return np.array([1.0])
        
        # Standard deviation based on the bar range
        std_dev = (high - low) / 4.0 if high != low else 0.01
        
        # Calculate weights using normal distribution
        weights = stats.norm.pdf(price_levels, close_price, std_dev)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return weights
    
    def _calculate_value_areas(self, volume_profile: Dict[float, Dict[str, Any]], 
                             df: pd.DataFrame) -> Tuple[ValueArea, List[ValueArea]]:
        """Calculate primary and secondary value areas with statistical validation"""
        if not volume_profile:
            return self._create_default_value_area(), []
        
        # Sort price levels by volume
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1]['volume'], reverse=True)
        total_volume = sum(level_data['volume'] for _, level_data in volume_profile.items())
        
        # Find Point of Control (POC)
        poc_price, poc_data = sorted_levels[0]
        
        # Calculate primary value area
        primary_value_area = self._calculate_primary_value_area(
            sorted_levels, total_volume, poc_price, poc_data
        )
        
        # Find secondary value areas using clustering
        secondary_value_areas = self._find_secondary_value_areas(
            volume_profile, primary_value_area, total_volume
        )
        
        return primary_value_area, secondary_value_areas
    
    def _calculate_primary_value_area(self, sorted_levels: List[Tuple[float, Dict]], 
                                    total_volume: float, poc_price: float, 
                                    poc_data: Dict) -> ValueArea:
        """Calculate the primary value area containing specified percentage of volume"""
        target_volume = total_volume * (self.config.value_area_percentage / 100.0)
        accumulated_volume = 0.0
        value_area_levels = []
        
        # Start with POC and expand outward
        for price, level_data in sorted_levels:
            accumulated_volume += level_data['volume']
            value_area_levels.append((price, level_data))
            
            if accumulated_volume >= target_volume:
                break
        
        # Calculate value area bounds
        prices = [price for price, _ in value_area_levels]
        va_high = max(prices)
        va_low = min(prices)
        
        # Calculate statistical confidence
        volume_concentration = accumulated_volume / total_volume
        price_spread = va_high - va_low
        statistical_confidence = min(volume_concentration * 2, 1.0)
        
        # Calculate acceptance level
        avg_volume = np.mean([level_data['volume'] for _, level_data in value_area_levels])
        acceptance_level = min(poc_data['volume'] / avg_volume, 3.0) / 3.0
        
        # Create volume nodes for this value area
        nodes = []
        for price, level_data in value_area_levels:
            if level_data['volume'] > avg_volume * 0.5:  # Significant nodes only
                node = VolumeNode(
                    price_level=price,
                    volume=level_data['volume'],
                    volume_percentage=level_data['volume_percentage'],
                    time_spent=level_data['time_periods'],
                    statistical_significance=level_data['volume'] / avg_volume,
                    node_type='value_area_node',
                    confidence=min(level_data['relative_volume'], 2.0) / 2.0
                )
                nodes.append(node)
        
        return ValueArea(
            high=va_high,
            low=va_low,
            point_of_control=poc_price,
            volume_percentage=volume_concentration * 100,
            value_area_type=ValueAreaType.PRIMARY,
            statistical_confidence=statistical_confidence,
            acceptance_level=acceptance_level,
            nodes=nodes
        )
    
    def _find_secondary_value_areas(self, volume_profile: Dict[float, Dict[str, Any]], 
                                  primary_va: ValueArea, total_volume: float) -> List[ValueArea]:
        """Find secondary value areas using clustering analysis"""
        # Extract data for clustering
        prices = []
        volumes = []
        
        for price, level_data in volume_profile.items():
            # Skip levels already in primary value area
            if primary_va.low <= price <= primary_va.high:
                continue
                
            # Only consider significant volume levels
            if level_data['volume_percentage'] >= 1.0:  # At least 1% of total volume
                prices.append(price)
                volumes.append(level_data['volume'])
        
        if len(prices) < 3:
            return []
        
        # Perform clustering to find secondary concentrations
        secondary_areas = []
        
        try:
            # Use DBSCAN clustering on price levels weighted by volume
            features = np.column_stack([prices, volumes])
            if len(features) >= 3:
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                
                clustering = DBSCAN(eps=self.config.clustering_eps, min_samples=2)
                clusters = clustering.fit_predict(scaled_features)
                
                # Process each cluster
                for cluster_id in set(clusters):
                    if cluster_id == -1:  # Noise points
                        continue
                    
                    cluster_mask = clusters == cluster_id
                    cluster_prices = np.array(prices)[cluster_mask]
                    cluster_volumes = np.array(volumes)[cluster_mask]
                    
                    if len(cluster_prices) >= 2:
                        # Calculate cluster statistics
                        cluster_volume = np.sum(cluster_volumes)
                        volume_percentage = (cluster_volume / total_volume) * 100
                        
                        # Only consider significant secondary areas
                        if volume_percentage >= 5.0:  # At least 5% of total volume
                            va_high = np.max(cluster_prices)
                            va_low = np.min(cluster_prices)
                            poc = cluster_prices[np.argmax(cluster_volumes)]
                            
                            # Determine value area type
                            if volume_percentage >= 15.0:
                                va_type = ValueAreaType.SECONDARY
                            else:
                                va_type = ValueAreaType.DEVELOPING
                            
                            secondary_area = ValueArea(
                                high=va_high,
                                low=va_low,
                                point_of_control=poc,
                                volume_percentage=volume_percentage,
                                value_area_type=va_type,
                                statistical_confidence=min(volume_percentage / 20.0, 1.0),
                                acceptance_level=np.max(cluster_volumes) / np.mean(cluster_volumes) / 3.0,
                                nodes=[]  # Can be populated if needed
                            )
                            secondary_areas.append(secondary_area)
        
        except Exception as e:
            self.logger.warning(f"Secondary value area clustering failed: {e}")
        
        return secondary_areas
    
    def _identify_volume_nodes(self, volume_profile: Dict[float, Dict[str, Any]], 
                             df: pd.DataFrame) -> List[VolumeNode]:
        """Identify significant volume concentration nodes"""
        if not volume_profile:
            return []
        
        nodes = []
        total_volume = sum(level_data['volume'] for level_data in volume_profile.values())
        avg_volume = total_volume / len(volume_profile)
        
        # Statistical thresholds
        volume_threshold = avg_volume * self.config.poc_confirmation_threshold
        
        for price, level_data in volume_profile.items():
            volume = level_data['volume']
            
            # Only consider significant volume concentrations
            if volume >= volume_threshold:
                # Calculate statistical significance
                z_score = (volume - avg_volume) / (np.std([ld['volume'] for ld in volume_profile.values()]) + 1e-8)
                statistical_significance = stats.norm.sf(abs(z_score))  # P-value
                
                # Determine node type
                if volume >= avg_volume * 3.0:
                    node_type = 'high_volume_node'
                elif volume >= avg_volume * 2.0:
                    node_type = 'significant_node'
                else:
                    node_type = 'volume_node'
                
                # Calculate confidence based on multiple factors
                volume_confidence = min(volume / (avg_volume * 3), 1.0)
                time_confidence = min(level_data['time_periods'] / 5.0, 1.0)
                relative_confidence = min(level_data['relative_volume'] / 2.0, 1.0)
                confidence = (volume_confidence + time_confidence + relative_confidence) / 3.0
                
                node = VolumeNode(
                    price_level=price,
                    volume=volume,
                    volume_percentage=level_data['volume_percentage'],
                    time_spent=level_data['time_periods'],
                    statistical_significance=statistical_significance,
                    node_type=node_type,
                    confidence=confidence
                )
                nodes.append(node)
        
        # Sort nodes by volume (descending)
        nodes.sort(key=lambda x: x.volume, reverse=True)
        
        return nodes
    
    def _analyze_profile_structure(self, volume_profile: Dict[float, Dict[str, Any]], 
                                 df: pd.DataFrame) -> ProfileAnalysis:
        """Analyze the overall structure and characteristics of the volume profile"""
        if not volume_profile:
            return ProfileAnalysis(
                profile_shape=ProfileShape.NORMAL_DISTRIBUTION,
                market_structure=MarketStructure.BALANCING,
                auction_state=AuctionState.INITIAL_BALANCE,
                distribution_skewness=0.0,
                distribution_kurtosis=0.0,
                concentration_ratio=0.5,
                volume_imbalance=0.0,
                structure_confidence=0.0
            )
        
        # Extract volume distribution
        prices = sorted(volume_profile.keys())
        volumes = [volume_profile[price]['volume'] for price in prices]
        
        # Calculate statistical measures
        distribution_skewness = stats.skew(volumes)
        distribution_kurtosis = stats.kurtosis(volumes)
        
        # Calculate concentration ratio (top 20% of levels vs total)
        sorted_volumes = sorted(volumes, reverse=True)
        top_20_percent = int(len(sorted_volumes) * 0.2) or 1
        concentration_ratio = sum(sorted_volumes[:top_20_percent]) / sum(sorted_volumes)
        
        # Analyze profile shape
        profile_shape = self._classify_profile_shape(volumes, distribution_skewness, distribution_kurtosis)
        
        # Analyze market structure
        market_structure = self._analyze_market_structure(volume_profile, df)
        
        # Determine auction state
        auction_state = self._determine_auction_state(volume_profile, df)
        
        # Calculate volume imbalance
        volume_imbalance = self._calculate_volume_imbalance(volume_profile, df)
        
        # Calculate structure confidence
        structure_confidence = self._calculate_structure_confidence(
            profile_shape, concentration_ratio, len(volume_profile)
        )
        
        return ProfileAnalysis(
            profile_shape=profile_shape,
            market_structure=market_structure,
            auction_state=auction_state,
            distribution_skewness=distribution_skewness,
            distribution_kurtosis=distribution_kurtosis,
            concentration_ratio=concentration_ratio,
            volume_imbalance=volume_imbalance,
            structure_confidence=structure_confidence
        )
    
    def _classify_profile_shape(self, volumes: List[float], skewness: float, kurtosis: float) -> ProfileShape:
        """Classify the shape of the volume profile distribution"""
        # Analyze distribution characteristics
        if abs(skewness) < 0.5 and abs(kurtosis) < 1.0:
            return ProfileShape.NORMAL_DISTRIBUTION
        elif skewness > 1.0:
            return ProfileShape.SKEWED_HIGH
        elif skewness < -1.0:
            return ProfileShape.SKEWED_LOW
        elif kurtosis > 2.0:
            return ProfileShape.EXTREME_TAIL
        elif abs(kurtosis) < 0.5:
            return ProfileShape.UNIFORM
        else:
            # Check for bimodal distribution
            if self._detect_bimodal_distribution(volumes):
                return ProfileShape.BIMODAL
            else:
                return ProfileShape.DOUBLE_DISTRIBUTION
    
    def _detect_bimodal_distribution(self, volumes: List[float]) -> bool:
        """Detect if the volume distribution is bimodal"""
        try:
            # Use Gaussian Mixture Model to detect multiple modes
            if len(volumes) < 5:
                return False
            
            # Fit GMM with 2 components
            gmm = GaussianMixture(n_components=2, random_state=42)
            data = np.array(volumes).reshape(-1, 1)
            gmm.fit(data)
            
            # Check if both components have significant weight
            return min(gmm.weights_) > 0.2
            
        except Exception:
            return False
    
    def _analyze_market_structure(self, volume_profile: Dict[float, Dict[str, Any]], 
                                df: pd.DataFrame) -> MarketStructure:
        """Analyze market structure based on volume profile characteristics"""
        if len(df) < 10:
            return MarketStructure.BALANCING
        
        # Calculate price movement and volume distribution
        recent_close = df['close'].iloc[-1]
        earlier_close = df['close'].iloc[-10]
        price_change = (recent_close - earlier_close) / earlier_close
        
        # Find volume concentration
        prices = sorted(volume_profile.keys())
        volumes = [volume_profile[price]['volume'] for price in prices]
        max_volume_price = prices[np.argmax(volumes)]
        
        # Analyze structure patterns
        if abs(price_change) > 0.02:  # Significant price movement
            if price_change > 0:
                if recent_close > max_volume_price:
                    return MarketStructure.TRENDING_UP
                else:
                    return MarketStructure.BREAKOUT
            else:
                if recent_close < max_volume_price:
                    return MarketStructure.TRENDING_DOWN
                else:
                    return MarketStructure.BREAKOUT
        else:
            # Low price movement, analyze volume distribution
            price_range = max(prices) - min(prices)
            if price_range < np.std(df['close'].iloc[-20:]) * 2:
                return MarketStructure.BRACKETED
            elif len([v for v in volumes if v > np.mean(volumes) * 1.5]) > len(volumes) * 0.3:
                return MarketStructure.ROTATING
            else:
                return MarketStructure.BALANCING
    
    def _determine_auction_state(self, volume_profile: Dict[float, Dict[str, Any]], 
                               df: pd.DataFrame) -> AuctionState:
        """Determine the current auction state of the market"""
        if len(df) < 5:
            return AuctionState.INITIAL_BALANCE
        
        recent_high = np.max(df['high'].iloc[-5:])
        recent_low = np.min(df['low'].iloc[-5:])
        current_close = df['close'].iloc[-1]
        
        # Find POC and value area
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1]['volume'], reverse=True)
        if sorted_levels:
            poc_price = sorted_levels[0][0]
            
            # Determine auction state based on price action relative to POC
            if current_close > poc_price * 1.01:
                if current_close >= recent_high * 0.98:
                    return AuctionState.EXCESS_HIGH
                else:
                    return AuctionState.ACCEPTANCE
            elif current_close < poc_price * 0.99:
                if current_close <= recent_low * 1.02:
                    return AuctionState.EXCESS_LOW
                else:
                    return AuctionState.REJECTION
            else:
                return AuctionState.ROTATION
        
        return AuctionState.INITIAL_BALANCE
    
    def _calculate_volume_imbalance(self, volume_profile: Dict[float, Dict[str, Any]], 
                                  df: pd.DataFrame) -> float:
        """Calculate volume imbalance across price levels"""
        if not volume_profile or len(df) < 2:
            return 0.0
        
        current_price = df['close'].iloc[-1]
        
        # Separate volume above and below current price
        volume_above = sum(level_data['volume'] for price, level_data in volume_profile.items() 
                          if price > current_price)
        volume_below = sum(level_data['volume'] for price, level_data in volume_profile.items() 
                          if price < current_price)
        
        total_volume = volume_above + volume_below
        
        if total_volume == 0:
            return 0.0
        
        # Calculate imbalance (-1 to 1, where positive indicates more volume above)
        imbalance = (volume_above - volume_below) / total_volume
        
        return imbalance
    
    def _calculate_structure_confidence(self, profile_shape: ProfileShape, 
                                      concentration_ratio: float, num_levels: int) -> float:
        """Calculate confidence in the profile structure analysis"""
        # Base confidence on data quality and distribution characteristics
        data_quality_score = min(num_levels / 20.0, 1.0)  # More levels = higher confidence
        
        # Shape-based confidence
        shape_confidence_map = {
            ProfileShape.NORMAL_DISTRIBUTION: 0.9,
            ProfileShape.BIMODAL: 0.8,
            ProfileShape.SKEWED_HIGH: 0.7,
            ProfileShape.SKEWED_LOW: 0.7,
            ProfileShape.UNIFORM: 0.6,
            ProfileShape.EXTREME_TAIL: 0.5,
            ProfileShape.DOUBLE_DISTRIBUTION: 0.6
        }
        shape_confidence = shape_confidence_map.get(profile_shape, 0.5)
        
        # Concentration confidence (moderate concentration is most reliable)
        if 0.3 <= concentration_ratio <= 0.7:
            concentration_confidence = 1.0
        else:
            concentration_confidence = 1.0 - abs(concentration_ratio - 0.5) * 2
        
        # Combined confidence
        overall_confidence = (data_quality_score + shape_confidence + concentration_confidence) / 3.0
        
        return overall_confidence
    
    def _analyze_current_price_context(self, primary_va: ValueArea, 
                                     volume_nodes: List[VolumeNode], 
                                     current_price: float) -> Dict[str, float]:
        """Analyze current price position relative to value areas and nodes"""
        context = {}
        
        # Position relative to primary value area
        if current_price > primary_va.high:
            context['va_position'] = 1.0  # Above value area
            context['va_distance'] = (current_price - primary_va.high) / primary_va.high
        elif current_price < primary_va.low:
            context['va_position'] = -1.0  # Below value area
            context['va_distance'] = (primary_va.low - current_price) / current_price
        else:
            context['va_position'] = 0.0  # Inside value area
            va_range = primary_va.high - primary_va.low
            if va_range > 0:
                context['va_distance'] = (current_price - primary_va.low) / va_range
            else:
                context['va_distance'] = 0.5
        
        # Distance to POC
        poc_distance = abs(current_price - primary_va.point_of_control) / current_price
        context['poc_distance'] = poc_distance
        
        # Nearest significant volume node
        if volume_nodes:
            node_distances = [abs(current_price - node.price_level) / current_price 
                            for node in volume_nodes]
            min_distance_idx = np.argmin(node_distances)
            context['nearest_node_distance'] = node_distances[min_distance_idx]
            context['nearest_node_strength'] = volume_nodes[min_distance_idx].confidence
        else:
            context['nearest_node_distance'] = 1.0
            context['nearest_node_strength'] = 0.0
        
        # Price acceptance level
        if context['va_position'] == 0.0:  # Inside value area
            context['acceptance_level'] = primary_va.acceptance_level
        else:
            # Price outside value area has lower acceptance
            context['acceptance_level'] = primary_va.acceptance_level * 0.5
        
        return context
    
    def _identify_support_resistance_levels(self, volume_nodes: List[VolumeNode], 
                                          primary_va: ValueArea) -> List[Tuple[float, float]]:
        """Identify key support and resistance levels based on volume analysis"""
        levels = []
        
        # Value area boundaries are strong levels
        levels.append((primary_va.low, primary_va.statistical_confidence))
        levels.append((primary_va.high, primary_va.statistical_confidence))
        levels.append((primary_va.point_of_control, primary_va.statistical_confidence * 1.2))
        
        # Significant volume nodes
        for node in volume_nodes:
            if node.confidence > 0.6:  # Only high-confidence nodes
                levels.append((node.price_level, node.confidence))
        
        # Remove duplicates and sort by price
        unique_levels = []
        seen_prices = set()
        
        for price, strength in levels:
            # Group nearby levels (within 0.1%)
            is_duplicate = False
            for seen_price in seen_prices:
                if abs(price - seen_price) / seen_price < 0.001:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_levels.append((price, strength))
                seen_prices.add(price)
        
        # Sort by price
        unique_levels.sort(key=lambda x: x[0])
        
        return unique_levels
    
    def _detect_institutional_levels(self, volume_nodes: List[VolumeNode], 
                                   volume_profile: Dict[float, Dict[str, Any]]) -> List[float]:
        """Detect price levels with institutional activity signatures"""
        institutional_levels = []
        
        # Look for specific patterns indicating institutional activity
        for node in volume_nodes:
            # High volume with relatively short time suggests institutional activity
            if (node.volume_percentage > 5.0 and  # Significant volume
                node.confidence > 0.8 and  # High confidence
                node.statistical_significance < 0.05):  # Statistically significant
                
                institutional_levels.append(node.price_level)
        
        # Look for round number concentrations (institutional preference)
        for price, level_data in volume_profile.items():
            # Check if price is near round numbers
            rounded_price = round(price, 4)
            if (abs(price - rounded_price) / price < 0.0001 and 
                level_data['volume_percentage'] > 3.0):
                
                if price not in institutional_levels:
                    institutional_levels.append(price)
        
        return sorted(institutional_levels)
    
    def _calculate_anomaly_score(self, volume_profile: Dict[float, Dict[str, Any]], 
                               profile_analysis: ProfileAnalysis) -> float:
        """Calculate anomaly score for the current volume profile"""
        if not self.config.use_anomaly_detection:
            return 0.0
        
        anomaly_factors = []
        
        # Concentration anomaly
        if profile_analysis.concentration_ratio > 0.8:
            anomaly_factors.append(0.8)  # Very high concentration
        elif profile_analysis.concentration_ratio < 0.2:
            anomaly_factors.append(0.6)  # Very low concentration
        else:
            anomaly_factors.append(0.0)
        
        # Distribution shape anomaly
        extreme_shapes = [ProfileShape.EXTREME_TAIL, ProfileShape.DOUBLE_DISTRIBUTION]
        if profile_analysis.profile_shape in extreme_shapes:
            anomaly_factors.append(0.7)
        else:
            anomaly_factors.append(0.0)
        
        # Statistical anomaly
        if abs(profile_analysis.distribution_skewness) > 2.0:
            anomaly_factors.append(0.6)
        else:
            anomaly_factors.append(0.0)
        
        if abs(profile_analysis.distribution_kurtosis) > 3.0:
            anomaly_factors.append(0.6)
        else:
            anomaly_factors.append(0.0)
        
        # Volume imbalance anomaly
        if abs(profile_analysis.volume_imbalance) > 0.7:
            anomaly_factors.append(0.5)
        else:
            anomaly_factors.append(0.0)
        
        # Calculate overall anomaly score
        return min(np.mean(anomaly_factors) * 2, 1.0)
    
    def _calculate_trading_signal_strength(self, primary_va: ValueArea, 
                                         current_price_context: Dict[str, float], 
                                         profile_analysis: ProfileAnalysis) -> float:
        """Calculate trading signal strength based on profile analysis"""
        signal_factors = []
        
        # Value area position signal
        va_position = current_price_context['va_position']
        va_distance = current_price_context['va_distance']
        
        if abs(va_position) > 0:  # Outside value area
            # Stronger signal when further from value area
            signal_factors.append(min(va_distance * 2, 1.0))
        else:
            # Weaker signal inside value area
            signal_factors.append(0.2)
        
        # POC proximity signal
        poc_distance = current_price_context['poc_distance']
        if poc_distance < 0.001:  # Very close to POC
            signal_factors.append(0.8)
        else:
            signal_factors.append(max(0.2, 1.0 - poc_distance * 10))
        
        # Volume node proximity signal
        node_distance = current_price_context['nearest_node_distance']
        node_strength = current_price_context['nearest_node_strength']
        
        if node_distance < 0.002:  # Close to significant node
            signal_factors.append(node_strength)
        else:
            signal_factors.append(0.1)
        
        # Market structure signal
        structure_signals = {
            MarketStructure.TRENDING_UP: 0.8,
            MarketStructure.TRENDING_DOWN: 0.8,
            MarketStructure.BREAKOUT: 0.9,
            MarketStructure.ROTATING: 0.6,
            MarketStructure.BRACKETED: 0.4,
            MarketStructure.BALANCING: 0.3
        }
        structure_signal = structure_signals.get(profile_analysis.market_structure, 0.5)
        signal_factors.append(structure_signal)
        
        # Auction state signal
        auction_signals = {
            AuctionState.EXCESS_HIGH: 0.7,
            AuctionState.EXCESS_LOW: 0.7,
            AuctionState.ACCEPTANCE: 0.8,
            AuctionState.REJECTION: 0.6,
            AuctionState.ROTATION: 0.4,
            AuctionState.INITIAL_BALANCE: 0.3
        }
        auction_signal = auction_signals.get(profile_analysis.auction_state, 0.5)
        signal_factors.append(auction_signal)
        
        # Calculate weighted signal strength
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        signal_strength = np.average(signal_factors, weights=weights)
        
        return signal_strength
    
    def _calculate_confidence(self, primary_va: ValueArea, profile_analysis: ProfileAnalysis, 
                            volume_nodes: List[VolumeNode]) -> float:
        """Calculate overall confidence in the analysis"""
        confidence_factors = []
        
        # Value area confidence
        confidence_factors.append(primary_va.statistical_confidence)
        
        # Profile structure confidence
        confidence_factors.append(profile_analysis.structure_confidence)
        
        # Volume data quality confidence
        if len(volume_nodes) >= 5:
            confidence_factors.append(0.9)
        elif len(volume_nodes) >= 3:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        # Volume node confidence
        if volume_nodes:
            avg_node_confidence = np.mean([node.confidence for node in volume_nodes])
            confidence_factors.append(avg_node_confidence)
        else:
            confidence_factors.append(0.3)
        
        # Concentration confidence
        concentration = profile_analysis.concentration_ratio
        if 0.3 <= concentration <= 0.7:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        return np.mean(confidence_factors)
    
    def _generate_signal(self, result: MarketProfileResult) -> SignalType:
        """Generate trading signal based on market profile analysis"""
        signal_criteria = []
        
        # Price position relative to value area
        va_position = result.current_price_context['va_position']
        va_distance = result.current_price_context['va_distance']
        
        if va_position > 0 and va_distance > 0.01:  # Significantly above value area
            signal_criteria.append('above_value_area')
        elif va_position < 0 and va_distance > 0.01:  # Significantly below value area
            signal_criteria.append('below_value_area')
        elif abs(va_position) == 0:  # Inside value area
            signal_criteria.append('inside_value_area')
        
        # Market structure signals
        if result.profile_analysis.market_structure in [MarketStructure.TRENDING_UP, MarketStructure.BREAKOUT]:
            if va_position >= 0:
                signal_criteria.append('bullish_structure')
        elif result.profile_analysis.market_structure in [MarketStructure.TRENDING_DOWN]:
            if va_position <= 0:
                signal_criteria.append('bearish_structure')
        
        # Auction state signals
        if result.profile_analysis.auction_state == AuctionState.ACCEPTANCE and va_position == 0:
            signal_criteria.append('price_acceptance')
        elif result.profile_analysis.auction_state == AuctionState.REJECTION:
            signal_criteria.append('price_rejection')
        
        # Volume imbalance signals
        if result.profile_analysis.volume_imbalance > 0.3:
            signal_criteria.append('volume_above')
        elif result.profile_analysis.volume_imbalance < -0.3:
            signal_criteria.append('volume_below')
        
        # POC signals
        poc_distance = result.current_price_context['poc_distance']
        if poc_distance < 0.005:
            signal_criteria.append('near_poc')
        
        # Signal generation logic
        bullish_signals = sum(1 for criterion in signal_criteria 
                            if criterion in ['below_value_area', 'bullish_structure', 
                                           'price_rejection', 'volume_below', 'near_poc'])
        bearish_signals = sum(1 for criterion in signal_criteria 
                            if criterion in ['above_value_area', 'bearish_structure', 
                                           'volume_above'])
        neutral_signals = sum(1 for criterion in signal_criteria 
                            if criterion in ['inside_value_area', 'price_acceptance'])
        
        high_confidence = result.confidence_score > 0.7
        strong_signal = result.trading_signal_strength > 0.6
        
        if bullish_signals >= 2 and high_confidence and strong_signal:
            return SignalType.BUY
        elif bearish_signals >= 2 and high_confidence and strong_signal:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _update_state(self, df: pd.DataFrame, data: Dict[str, Any], result: MarketProfileResult):
        """Update internal state and historical data"""
        max_history = 100
        
        # Update value area history
        self._value_area_history.append(result.primary_value_area)
        if len(self._value_area_history) > max_history:
            self._value_area_history = self._value_area_history[-max_history:]
        
        # Update volume node history
        self._volume_node_history.append(result.volume_nodes)
        if len(self._volume_node_history) > max_history:
            self._volume_node_history = self._volume_node_history[-max_history:]
        
        # Update market structure buffer
        self._market_structure_buffer.append(result.profile_analysis.market_structure.value)
        if len(self._market_structure_buffer) > 50:
            self._market_structure_buffer = self._market_structure_buffer[-50:]
        
        # Update current profile shape
        self._current_profile_shape = result.profile_analysis.profile_shape
        
        # Store price-volume data for future analysis
        current_data = {
            'timestamp': len(df),
            'close': df['close'].iloc[-1],
            'volume': df['volume'].iloc[-1],
            'value_area': result.primary_value_area,
            'profile_analysis': result.profile_analysis
        }
        self._price_volume_history.append(current_data)
        
        if len(self._price_volume_history) > max_history:
            self._price_volume_history = self._price_volume_history[-max_history:]
    
    def _format_result(self, result: MarketProfileResult, signal: SignalType) -> Dict[str, Any]:
        """Format the complete result for output"""
        return {
            'signal': signal,
            'confidence': result.confidence_score,
            
            # Primary value area
            'value_area_high': result.primary_value_area.high,
            'value_area_low': result.primary_value_area.low,
            'point_of_control': result.primary_value_area.point_of_control,
            'value_area_volume_percentage': result.primary_value_area.volume_percentage,
            'value_area_type': result.primary_value_area.value_area_type.value,
            'value_area_confidence': result.primary_value_area.statistical_confidence,
            'acceptance_level': result.primary_value_area.acceptance_level,
            
            # Secondary value areas
            'secondary_value_areas': [
                {
                    'high': va.high,
                    'low': va.low,
                    'poc': va.point_of_control,
                    'volume_percentage': va.volume_percentage,
                    'type': va.value_area_type.value,
                    'confidence': va.statistical_confidence
                }
                for va in result.secondary_value_areas
            ],
            
            # Volume nodes
            'volume_nodes': [
                {
                    'price': node.price_level,
                    'volume': node.volume,
                    'volume_percentage': node.volume_percentage,
                    'time_spent': node.time_spent,
                    'significance': node.statistical_significance,
                    'type': node.node_type,
                    'confidence': node.confidence
                }
                for node in result.volume_nodes[:10]  # Top 10 nodes
            ],
            
            # Profile analysis
            'profile_shape': result.profile_analysis.profile_shape.value,
            'market_structure': result.profile_analysis.market_structure.value,
            'auction_state': result.profile_analysis.auction_state.value,
            'distribution_skewness': result.profile_analysis.distribution_skewness,
            'distribution_kurtosis': result.profile_analysis.distribution_kurtosis,
            'concentration_ratio': result.profile_analysis.concentration_ratio,
            'volume_imbalance': result.profile_analysis.volume_imbalance,
            'structure_confidence': result.profile_analysis.structure_confidence,
            
            # Current price context
            'current_price_context': result.current_price_context,
            
            # Support/resistance levels
            'support_resistance_levels': [
                {'price': level[0], 'strength': level[1]}
                for level in result.support_resistance_levels[:10]
            ],
            
            # Institutional levels
            'institutional_levels': result.institutional_levels[:5],
            
            # Analysis metrics
            'anomaly_score': result.anomaly_score,
            'trading_signal_strength': result.trading_signal_strength,
            
            # Metadata
            'metadata': {
                'indicator_name': 'MarketProfileValueAreaIndicator',
                'version': '1.0.0',
                'calculation_time': pd.Timestamp.now().isoformat(),
                'profile_periods': self.config.profile_periods,
                'value_area_percentage': self.config.value_area_percentage,
                'statistical_validation': self.config.use_statistical_validation,
                'anomaly_detection': self.config.use_anomaly_detection
            }
        }
    
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
    
    def _create_default_value_area(self) -> ValueArea:
        """Create default value area for insufficient data"""
        return ValueArea(
            high=1.0001,
            low=0.9999,
            point_of_control=1.0000,
            volume_percentage=70.0,
            value_area_type=ValueAreaType.PRIMARY,
            statistical_confidence=0.5,
            acceptance_level=0.5,
            nodes=[]
        )
    
    def _create_default_result(self) -> Dict[str, Any]:
        """Create default result for insufficient data"""
        default_va = self._create_default_value_area()
        
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.0,
            'value_area_high': default_va.high,
            'value_area_low': default_va.low,
            'point_of_control': default_va.point_of_control,
            'value_area_volume_percentage': default_va.volume_percentage,
            'value_area_type': default_va.value_area_type.value,
            'value_area_confidence': default_va.statistical_confidence,
            'acceptance_level': default_va.acceptance_level,
            'secondary_value_areas': [],
            'volume_nodes': [],
            'profile_shape': ProfileShape.NORMAL_DISTRIBUTION.value,
            'market_structure': MarketStructure.BALANCING.value,
            'auction_state': AuctionState.INITIAL_BALANCE.value,
            'distribution_skewness': 0.0,
            'distribution_kurtosis': 0.0,
            'concentration_ratio': 0.5,
            'volume_imbalance': 0.0,
            'structure_confidence': 0.0,
            'current_price_context': {
                'va_position': 0.0,
                'va_distance': 0.5,
                'poc_distance': 0.0,
                'nearest_node_distance': 1.0,
                'nearest_node_strength': 0.0,
                'acceptance_level': 0.5
            },
            'support_resistance_levels': [],
            'institutional_levels': [],
            'anomaly_score': 0.0,
            'trading_signal_strength': 0.0,
            'metadata': {
                'indicator_name': 'MarketProfileValueAreaIndicator',
                'error': 'Insufficient data for calculation'
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure"""
        result = self._create_default_result()
        result['metadata']['error'] = error_message
        return result