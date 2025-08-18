"""
Time Segmented Volume Indicator

Analyzes volume patterns across different time segments to identify institutional activity,
accumulation/distribution patterns, and volume-based trading opportunities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, time, timedelta
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

from auj_platform.src.indicator_engine.indicators.base.standard_indicator import StandardIndicatorInterface
from auj_platform.src.indicator_engine.indicators.base.standard_indicator import IndicatorCalculationError

logger = logging.getLogger(__name__)


@dataclass
class VolumeSegment:
    """Data structure for volume segment analysis"""
    segment_name: str
    start_time: time
    end_time: time
    total_volume: float
    avg_volume: float
    volume_percentage: float
    price_efficiency: float
    accumulation_score: float
    distribution_score: float
    institutional_activity: float


@dataclass
class VolumeProfile:
    """Data structure for volume profile analysis"""
    price_level: float
    volume: float
    volume_percentage: float
    value_area_high: bool
    value_area_low: bool
    point_of_control: bool
    support_resistance_strength: float


@dataclass
class InstitutionalFlow:
    """Data structure for institutional flow analysis"""
    flow_direction: str
    flow_strength: float
    large_block_ratio: float
    stealth_accumulation: float
    institutional_signature: float
    smart_money_confidence: float


class TimeSegmentedVolumeIndicator(StandardIndicatorInterface):
    """
    Advanced Time Segmented Volume Indicator
    
    This indicator provides sophisticated volume analysis across different time segments,
    identifying institutional activity patterns, accumulation/distribution zones,
    and volume-based trading opportunities.
    """
    
def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            'analysis_window': 100,
            'segment_resolution': 30,  # minutes per segment
            'volume_threshold_factor': 1.5,
            'institutional_threshold': 0.7,
            'accumulation_lookback': 20,
            'clustering_eps': 0.3,
            'clustering_min_samples': 3,
            'vwap_periods': [20, 50, 100],
            'price_efficiency_window': 10,
            'smart_money_window': 50,
            'volume_profile_bins': 20
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name="Time Segmented Volume")
        
        # State tracking
        self.volume_segments = []
        self.volume_profiles = []
        self.institutional_flows = []
        self.cumulative_vwap = None
        self.volume_clusters = None
        
def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with time segments and volume analysis"""
        try:
            df = data.copy()
            
            # Ensure datetime index
            if 'datetime' in df.columns and df.index.name != 'datetime':
                df.set_index('datetime', inplace=True)
            
            # Calculate basic volume metrics
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['volume_weighted_price'] = df['typical_price'] * df['volume']
            
            # Add time-based features
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['time_segment'] = self._calculate_time_segment(df.index)
            
            # Calculate price efficiency
            df['price_change'] = df['close'].pct_change()
            df['price_efficiency'] = self._calculate_price_efficiency(df)
            
            # Volume anomaly detection
            df['volume_zscore'] = stats.zscore(df['volume'].fillna(0))
            df['volume_anomaly'] = np.abs(df['volume_zscore']) > 2
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
def _calculate_time_segment(self, datetime_index: pd.DatetimeIndex) -> np.ndarray:
        """Calculate time segments based on resolution"""
        try:
            resolution_minutes = self.parameters['segment_resolution']
            
            # Convert to minutes since midnight
            minutes_since_midnight = datetime_index.hour * 60 + datetime_index.minute
            
            # Calculate segment number
            segments = minutes_since_midnight // resolution_minutes
            
            return segments.values
            
        except Exception as e:
            logger.error(f"Error calculating time segments: {str(e)}")
            return np.zeros(len(datetime_index))
    
def _calculate_price_efficiency(self, data: pd.DataFrame) -> pd.Series:
        """Calculate price efficiency - how much price moves relative to volume"""
        try:
            window = self.parameters['price_efficiency_window']
            
            # Price movement efficiency
            price_range = data['high'] - data['low']
            volume_normalized = data['volume'] / data['volume'].rolling(window).mean()
            
            # Efficiency = price movement / volume (higher = more efficient)
            efficiency = price_range / (volume_normalized + 1e-10)
            
            return efficiency.rolling(window).mean()
            
        except Exception as e:
            logger.error(f"Error calculating price efficiency: {str(e)}")
            return pd.Series(index=data.index, data=0.0)
    
def _analyze_volume_segments(self, data: pd.DataFrame) -> List[VolumeSegment]:
        """Analyze volume patterns across time segments"""
        try:
            segments = []
            
            # Group by time segment
            for segment_id in data['time_segment'].unique():
                if np.isnan(segment_id):
                    continue
                    
                segment_data = data[data['time_segment'] == segment_id]
                
                if len(segment_data) < 3:
                    continue
                
                # Calculate segment metrics
                total_volume = segment_data['volume'].sum()
                avg_volume = segment_data['volume'].mean()
                volume_percentage = total_volume / data['volume'].sum() * 100
                
                # Price efficiency for this segment
                price_efficiency = segment_data['price_efficiency'].mean()
                
                # Accumulation/Distribution analysis
                accumulation_score = self._calculate_accumulation_score(segment_data)
                distribution_score = self._calculate_distribution_score(segment_data)
                
                # Institutional activity estimation
                institutional_activity = self._estimate_institutional_activity(segment_data)
                
                # Calculate time range
                resolution_minutes = self.parameters['segment_resolution']
                start_hour = int(segment_id * resolution_minutes // 60)
                start_minute = int(segment_id * resolution_minutes % 60)
                end_hour = int((segment_id + 1) * resolution_minutes // 60)
                end_minute = int((segment_id + 1) * resolution_minutes % 60)
                
                start_time = time(start_hour % 24, start_minute)
                end_time = time(end_hour % 24, end_minute)
                
                segment = VolumeSegment()
                    segment_name=f"Segment_{int(segment_id)}",
                    start_time=start_time,
                    end_time=end_time,
                    total_volume=total_volume,
                    avg_volume=avg_volume,
                    volume_percentage=volume_percentage,
                    price_efficiency=price_efficiency,
                    accumulation_score=accumulation_score,
                    distribution_score=distribution_score,
                    institutional_activity=institutional_activity
(                )
                
                segments.append(segment)
            
            # Sort by volume percentage (most active first)
            segments.sort(key=lambda x: x.volume_percentage, reverse=True)
            
            return segments
            
        except Exception as e:
            logger.error(f"Error analyzing volume segments: {str(e)}")
            return []
    
def _calculate_accumulation_score(self, segment_data: pd.DataFrame) -> float:
        """Calculate accumulation score for a segment"""
        try:
            if len(segment_data) < 3:
                return 0.0
            
            # Accumulation indicators:
            # 1. Price trend vs volume
            price_trend = (segment_data['close'].iloc[-1] - segment_data['close'].iloc[0]) / segment_data['close'].iloc[0]
            volume_trend = segment_data['volume'].corr(pd.Series(range(len(segment_data))))
            
            # 2. High volume on up moves
            up_moves = segment_data['close'] > segment_data['close'].shift(1)
            up_volume_ratio = segment_data[up_moves]['volume'].mean() / segment_data['volume'].mean()
            
            # 3. Price efficiency during accumulation
            efficiency_factor = segment_data['price_efficiency'].mean()
            
            # Combine factors
            accumulation_score = ()
                max(0, price_trend) * 0.4 +
                max(0, volume_trend) * 0.3 +
                max(0, up_volume_ratio - 1) * 0.2 +
                min(1, efficiency_factor / efficiency_factor.quantile(0.8) if efficiency_factor.quantile(0.8) > 0 else 0) * 0.1
(            )
            
            return min(1.0, accumulation_score)
            
        except Exception as e:
            logger.error(f"Error calculating accumulation score: {str(e)}")
            return 0.0
    
def _calculate_distribution_score(self, segment_data: pd.DataFrame) -> float:
        """Calculate distribution score for a segment"""
        try:
            if len(segment_data) < 3:
                return 0.0
            
            # Distribution indicators:
            # 1. Price decline vs volume
            price_trend = (segment_data['close'].iloc[0] - segment_data['close'].iloc[-1]) / segment_data['close'].iloc[0]
            volume_trend = segment_data['volume'].corr(pd.Series(range(len(segment_data))))
            
            # 2. High volume on down moves
            down_moves = segment_data['close'] < segment_data['close'].shift(1)
            down_volume_ratio = segment_data[down_moves]['volume'].mean() / segment_data['volume'].mean()
            
            # 3. Volume spikes at highs
            high_prices = segment_data['high'] >= segment_data['high'].quantile(0.8)
            high_volume_at_peaks = segment_data[high_prices]['volume'].mean() / segment_data['volume'].mean()
            
            # Combine factors
            distribution_score = ()
                max(0, price_trend) * 0.3 +
                max(0, volume_trend) * 0.3 +
                max(0, down_volume_ratio - 1) * 0.2 +
                max(0, high_volume_at_peaks - 1) * 0.2
(            )
            
            return min(1.0, distribution_score)
            
        except Exception as e:
            logger.error(f"Error calculating distribution score: {str(e)}")
            return 0.0
    
def _estimate_institutional_activity(self, segment_data: pd.DataFrame) -> float:
        """Estimate institutional trading activity"""
        try:
            if len(segment_data) < 5:
                return 0.0
            
            # Institutional activity indicators:
            # 1. Large volume blocks
            volume_threshold = segment_data['volume'].quantile(0.8)
            large_volume_ratio = (segment_data['volume'] > volume_threshold).sum() / len(segment_data)
            
            # 2. Volume persistence
            volume_autocorr = segment_data['volume'].autocorr(lag=1)
            
            # 3. Price impact efficiency
            price_changes = segment_data['close'].pct_change().abs()
            volume_impact = price_changes.corr(segment_data['volume'])
            
            # 4. VWAP deviation patterns
            cumsum_volume = segment_data['volume'].cumsum()
            cumsum_vwp = (segment_data['volume_weighted_price']).cumsum()
            vwap = cumsum_vwp / cumsum_volume
            vwap_deviation = (segment_data['close'] - vwap).std()
            
            # Combine indicators
            institutional_score = ()
                large_volume_ratio * 0.3 +
                max(0, volume_autocorr) * 0.2 +
                (1 - min(1, abs(volume_impact))) * 0.3 +  # Institutions minimize price impact
                min(1, vwap_deviation / segment_data['close'].std()) * 0.2
(            )
            
            return min(1.0, institutional_score)
            
        except Exception as e:
            logger.error(f"Error estimating institutional activity: {str(e)}")
            return 0.0
    
def _create_volume_profile(self, data: pd.DataFrame) -> List[VolumeProfile]:
        """Create volume profile analysis"""
        try:
            profiles = []
            
            # Calculate price bins
            price_min = data['low'].min()
            price_max = data['high'].max()
            num_bins = self.parameters['volume_profile_bins']
            
            price_bins = np.linspace(price_min, price_max, num_bins + 1)
            
            # Calculate volume at each price level
            for i in range(len(price_bins) - 1):
                bin_low = price_bins[i]
                bin_high = price_bins[i + 1]
                bin_center = (bin_low + bin_high) / 2
                
                # Find bars that traded in this price range
                in_range = ()
                    (data['low'] <= bin_high) & 
                    (data['high'] >= bin_low)
(                )
                
                if not in_range.any():
                    continue
                
                # Calculate volume for this price level
                range_data = data[in_range]
                total_volume = range_data['volume'].sum()
                
                # Volume percentage
                volume_percentage = total_volume / data['volume'].sum() * 100
                
                # Support/Resistance strength based on volume concentration
                sr_strength = volume_percentage / 100 * (1 + len(range_data) / len(data))
                
                profile = VolumeProfile()
                    price_level=bin_center,
                    volume=total_volume,
                    volume_percentage=volume_percentage,
                    value_area_high=False,  # Will be determined later
                    value_area_low=False,   # Will be determined later
                    point_of_control=False, # Will be determined later
                    support_resistance_strength=sr_strength
(                )
                
                profiles.append(profile)
            
            # Sort by volume
            profiles.sort(key=lambda x: x.volume, reverse=True)
            
            # Identify Point of Control (highest volume)
            if profiles:
                profiles[0].point_of_control = True
                
                # Identify Value Area (70% of volume)
                total_volume = sum(p.volume for p in profiles)
                cumulative_volume = 0
                value_area_threshold = total_volume * 0.7
                
                for profile in profiles:
                    cumulative_volume += profile.volume
                    if cumulative_volume <= value_area_threshold:
                        if profile == profiles[0]:
                            continue  # POC is separate
                        elif profile.price_level > profiles[0].price_level:
                            profile.value_area_high = True
                        else:
                            profile.value_area_low = True
            
            return profiles
            
        except Exception as e:
            logger.error(f"Error creating volume profile: {str(e)}")
            return []
    
def _analyze_institutional_flow(self, data: pd.DataFrame) -> InstitutionalFlow:
        """Analyze institutional money flow patterns"""
        try:
            if len(data) < self.parameters['smart_money_window']:
                return InstitutionalFlow()
                    flow_direction='neutral',
                    flow_strength=0.0,
                    large_block_ratio=0.0,
                    stealth_accumulation=0.0,
                    institutional_signature=0.0,
                    smart_money_confidence=0.0
(                )
            
            window = self.parameters['smart_money_window']
            recent_data = data.tail(window)
            
            # 1. Large block detection
            volume_threshold = recent_data['volume'].quantile(0.8)
            large_blocks = recent_data['volume'] > volume_threshold
            large_block_ratio = large_blocks.sum() / len(recent_data)
            
            # 2. Stealth accumulation (consistent buying without price spikes)
            price_efficiency = recent_data['price_efficiency']
            volume_consistency = 1 - recent_data['volume'].std() / recent_data['volume'].mean()
            stealth_score = volume_consistency * (1 - price_efficiency.std())
            
            # 3. Institutional signature patterns
            # - Volume leading price
            # - Low volatility during accumulation
            # - VWAP adherence
            volume_price_corr = recent_data['volume'].corr(recent_data['close'].pct_change().abs())
            volatility = recent_data['close'].pct_change().std()
            avg_volatility = data['close'].pct_change().rolling(100).std().mean()
            
            volatility_ratio = volatility / avg_volatility if avg_volatility > 0 else 1
            institutional_signature = (1 - min(1, abs(volume_price_corr))) * (1 - min(1, volatility_ratio))
            
            # 4. Flow direction
            net_volume_flow = 0
            for _, row in recent_data.iterrows():
                if row['close'] > row['typical_price']:
                    net_volume_flow += row['volume']
                else:
                    net_volume_flow -= row['volume']
            
            total_volume = recent_data['volume'].sum()
            flow_ratio = net_volume_flow / total_volume if total_volume > 0 else 0
            
            if flow_ratio > 0.1:
                flow_direction = 'bullish'
            elif flow_ratio < -0.1:
                flow_direction = 'bearish'
            else:
                flow_direction = 'neutral'
            
            flow_strength = abs(flow_ratio)
            
            # 5. Smart money confidence
            confidence_factors = []
                large_block_ratio,
                min(1, stealth_score),
                institutional_signature,
                min(1, flow_strength * 2)
[            ]
            smart_money_confidence = np.mean(confidence_factors)
            
            return InstitutionalFlow()
                flow_direction=flow_direction,
                flow_strength=flow_strength,
                large_block_ratio=large_block_ratio,
                stealth_accumulation=stealth_score,
                institutional_signature=institutional_signature,
                smart_money_confidence=smart_money_confidence
(            )
            
        except Exception as e:
            logger.error(f"Error analyzing institutional flow: {str(e)}")
            return InstitutionalFlow()
                flow_direction='neutral',
                flow_strength=0.0,
                large_block_ratio=0.0,
                stealth_accumulation=0.0,
                institutional_signature=0.0,
                smart_money_confidence=0.0
(            )
    
def _perform_volume_clustering(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Perform clustering analysis on volume patterns"""
        try:
            if len(data) < 20:
                return None
            
            # Prepare features for clustering
            features = []
            
            # Volume features
            volume_features = data[['volume', 'volume_ratio', 'price_efficiency']].fillna(0)
            
            # Time features
            time_features = pd.DataFrame({)
                'hour': data['hour'],
                'minute_segment': data['time_segment']
(            })
            
            # Price action features
            price_features = pd.DataFrame({)
                'price_change': data['price_change'].fillna(0),
                'volume_zscore': data['volume_zscore'].fillna(0)
(            })
            
            # Combine all features
            all_features = pd.concat([volume_features, time_features, price_features], axis=1)
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(all_features)
            
            # Perform DBSCAN clustering
            eps = self.parameters['clustering_eps']
            min_samples = self.parameters['clustering_min_samples']
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clustering.fit_predict(scaled_features)
            
            return cluster_labels
            
        except Exception as e:
            logger.error(f"Error performing volume clustering: {str(e)}")
            return None    
def calculate_raw(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive time segmented volume analysis
        """
        try:
            if len(data) < self.parameters['analysis_window']:
                return {
                    'volume_segments': [],
                    'volume_profiles': [],
                    'institutional_flow': {},
                    'dominant_segment': None,
                    'volume_concentration': 0.0,
                    'market_phase': 'unknown'
                }
            
            # Prepare data with volume analysis
            prepared_data = self._prepare_data(data)
            
            # Analyze volume segments
            volume_segments = self._analyze_volume_segments(prepared_data)
            self.volume_segments = volume_segments
            
            # Create volume profile
            volume_profiles = self._create_volume_profile(prepared_data)
            self.volume_profiles = volume_profiles
            
            # Analyze institutional flow
            institutional_flow = self._analyze_institutional_flow(prepared_data)
            self.institutional_flows.append(institutional_flow)
            
            # Perform volume clustering
            volume_clusters = self._perform_volume_clustering(prepared_data)
            self.volume_clusters = volume_clusters
            
            # Determine dominant segment
            dominant_segment = volume_segments[0] if volume_segments else None
            
            # Calculate volume concentration (Herfindahl-Hirschman Index for volume)
            volume_concentration = self._calculate_volume_concentration(volume_segments)
            
            # Determine market phase
            market_phase = self._determine_market_phase()
                volume_segments, institutional_flow, prepared_data
(            )
            
            # Calculate additional metrics
            vwap_deviation = self._calculate_vwap_deviation(prepared_data)
            volume_momentum = self._calculate_volume_momentum(prepared_data)
            smart_money_index = self._calculate_smart_money_index(prepared_data)
            
            # Volume anomaly analysis
            volume_anomalies = self._detect_volume_anomalies(prepared_data)
            
            # Time-based volume patterns
            hourly_patterns = self._analyze_hourly_patterns(prepared_data)
            
            result = {
                'volume_segments': [self._segment_to_dict(seg) for seg in volume_segments[:10]],
                'volume_profiles': [self._profile_to_dict(prof) for prof in volume_profiles[:15]],
                'institutional_flow': self._flow_to_dict(institutional_flow),
                'dominant_segment': self._segment_to_dict(dominant_segment) if dominant_segment else None,
                'volume_concentration': volume_concentration,
                'market_phase': market_phase,
                'vwap_deviation': vwap_deviation,
                'volume_momentum': volume_momentum,
                'smart_money_index': smart_money_index,
                'volume_anomalies': volume_anomalies,
                'hourly_patterns': hourly_patterns,
                'cluster_analysis': {
                    'n_clusters': len(set(volume_clusters)) - (1 if -1 in volume_clusters else 0) if volume_clusters is not None else 0,
                    'noise_ratio': (volume_clusters == -1).sum() / len(volume_clusters) if volume_clusters is not None else 0
                },
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in time segmented volume calculation: {str(e)}")
            raise IndicatorCalculationError()
                indicator_name=self.name,
                calculation_step="time_segmented_volume_calculation",
                message=str(e)
(            )
    
def _calculate_volume_concentration(self, segments: List[VolumeSegment]) -> float:
        """Calculate volume concentration using Herfindahl-Hirschman Index"""
        try:
            if not segments:
                return 0.0
            
            # Calculate HHI for volume concentration
            hhi = sum((seg.volume_percentage / 100) ** 2 for seg in segments)
            
            # Normalize to 0-1 scale (1 = highly concentrated, 0 = evenly distributed)
            max_hhi = 1.0  # Perfect concentration
            min_hhi = 1.0 / len(segments)  # Perfect distribution
            
            if max_hhi == min_hhi:
                return 0.5
            
            normalized_hhi = (hhi - min_hhi) / (max_hhi - min_hhi)
            
            return min(1.0, max(0.0, normalized_hhi))
            
        except Exception as e:
            logger.error(f"Error calculating volume concentration: {str(e)}")
            return 0.0
    
def _determine_market_phase(self, segments: List[VolumeSegment],:)
                              institutional_flow: InstitutionalFlow,
(                              data: pd.DataFrame) -> str:
        """Determine current market phase based on volume analysis"""
        try:
            if not segments:
                return 'unknown'
            
            # Get dominant segment characteristics
            dominant = segments[0]
            
            # Analyze recent price action
            recent_data = data.tail(20)
            price_trend = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            volume_trend = recent_data['volume'].corr(pd.Series(range(len(recent_data))))
            
            # Phase determination logic
            if dominant.accumulation_score > 0.7:
                if institutional_flow.flow_direction == 'bullish':
                    return 'accumulation'
                else:
                    return 'stealth_accumulation'
            
            elif dominant.distribution_score > 0.7:
                if institutional_flow.flow_direction == 'bearish':
                    return 'distribution'
                else:
                    return 'topping'
            
            elif institutional_flow.smart_money_confidence > 0.8:
                if abs(price_trend) < 0.02:  # Sideways:
                    return 'consolidation'
                elif price_trend > 0:
                    return 'markup'
                else:
                    return 'markdown'
            
            elif volume_trend > 0.3 and price_trend > 0:
                return 'trending_up'
            
            elif volume_trend > 0.3 and price_trend < 0:
                return 'trending_down'
            
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error determining market phase: {str(e)}")
            return 'unknown'
    
def _calculate_vwap_deviation(self, data: pd.DataFrame) -> float:
        """Calculate current VWAP deviation"""
        try:
            if len(data) < 20:
                return 0.0
            
            # Calculate VWAP
            cumsum_volume = data['volume'].cumsum()
            cumsum_vwp = (data['volume_weighted_price']).cumsum()
            vwap = cumsum_vwp / cumsum_volume
            
            # Current deviation
            current_price = data['close'].iloc[-1]
            current_vwap = vwap.iloc[-1]
            
            deviation = (current_price - current_vwap) / current_vwap
            
            return deviation
            
        except Exception as e:
            logger.error(f"Error calculating VWAP deviation: {str(e)}")
            return 0.0
    
def _calculate_volume_momentum(self, data: pd.DataFrame) -> float:
        """Calculate volume momentum indicator"""
        try:
            if len(data) < 20:
                return 0.0
            
            # Volume rate of change
            volume_roc = data['volume'].pct_change(periods=5).fillna(0)
            
            # Volume momentum = correlation between volume and price movement
            price_roc = data['close'].pct_change(periods=5).fillna(0)
            
            # Recent momentum
            recent_vol_mom = volume_roc.tail(10).mean()
            vol_price_corr = volume_roc.tail(20).corr(price_roc.tail(20))
            
            # Combine factors
            momentum = recent_vol_mom * (1 + abs(vol_price_corr))
            
            return min(2.0, max(-2.0, momentum))
            
        except Exception as e:
            logger.error(f"Error calculating volume momentum: {str(e)}")
            return 0.0
    
def _calculate_smart_money_index(self, data: pd.DataFrame) -> float:
        """Calculate Smart Money Index (SMI)"""
        try:
            if len(data) < 30:
                return 0.0
            
            # Smart Money Index tracks institutional money flow
            # Based on the premise that the market action during the last hour
            # represents the "smart money" moves
            
            recent_data = data.tail(30)
            
            # Calculate intraday high and low
            intraday_high = recent_data['high'].max()
            intraday_low = recent_data['low'].min()
            
            # Current position within the range
            current_price = recent_data['close'].iloc[-1]
            range_position = (current_price - intraday_low) / (intraday_high - intraday_low) if intraday_high != intraday_low else 0.5
            
            # Volume weighted position
            total_volume = recent_data['volume'].sum()
            if total_volume > 0:
                weighted_position = (recent_data['volume'] * recent_data['close']).sum() / (total_volume * current_price)
            else:
                weighted_position = 0.5
            
            # Smart money tends to buy low and sell high
            # Higher values indicate smart money accumulation at current levels
            smi = (1 - range_position) * weighted_position * 2
            
            return min(1.0, max(0.0, smi))
            
        except Exception as e:
            logger.error(f"Error calculating smart money index: {str(e)}")
            return 0.0
    
def _detect_volume_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect volume anomalies and unusual patterns"""
        try:
            anomalies = {
                'volume_spikes': [],
                'volume_droughts': [],
                'unusual_patterns': []
            }
            
            if len(data) < 50:
                return anomalies
            
            # Volume spike detection
            volume_threshold = data['volume'].quantile(0.95)
            spikes = data[data['volume'] > volume_threshold]
            
            for idx, spike in spikes.tail(5).iterrows():
                anomalies['volume_spikes'].append({)
                    'timestamp': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                    'volume': spike['volume'],
                    'volume_ratio': spike['volume_ratio'],
                    'price_change': spike['price_change']
(                })
            
            # Volume drought detection
            drought_threshold = data['volume'].quantile(0.05)
            droughts = data[data['volume'] < drought_threshold]
            
            for idx, drought in droughts.tail(3).iterrows():
                anomalies['volume_droughts'].append({)
                    'timestamp': idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                    'volume': drought['volume'],
                    'volume_ratio': drought['volume_ratio']
(                })
            
            # Unusual pattern detection
            recent_data = data.tail(20)
            
            # Divergence between price and volume
            price_direction = 1 if recent_data['close'].iloc[-1] > recent_data['close'].iloc[0] else -1
            volume_direction = 1 if recent_data['volume'].iloc[-1] > recent_data['volume'].iloc[0] else -1
            
            if price_direction != volume_direction:
                anomalies['unusual_patterns'].append({)
                    'pattern': 'price_volume_divergence',
                    'description': 'Price and volume moving in opposite directions',
                    'severity': abs(recent_data['close'].pct_change().sum()) * abs(recent_data['volume'].pct_change().sum())
(                })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting volume anomalies: {str(e)}")
            return {'volume_spikes': [], 'volume_droughts': [], 'unusual_patterns': []}
    
def _analyze_hourly_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns by hour of day"""
        try:
            patterns = {}
            
            if len(data) < 100:
                return patterns
            
            # Group by hour
            hourly_data = data.groupby('hour').agg({)
                'volume': ['mean', 'std', 'sum'],
                'volume_ratio': 'mean',
                'price_efficiency': 'mean',
                'close': 'count'
(            }).round(4)
            
            # Flatten column names
            hourly_data.columns = ['_'.join(col).strip() for col in hourly_data.columns]
            
            # Find peak and low volume hours
            peak_hour = hourly_data['volume_mean'].idxmax()
            low_hour = hourly_data['volume_mean'].idxmin()
            
            patterns = {
                'peak_volume_hour': int(peak_hour),
                'low_volume_hour': int(low_hour),
                'hourly_volume_ratio': hourly_data['volume_ratio_mean'].to_dict(),
                'hourly_efficiency': hourly_data['price_efficiency_mean'].to_dict(),
                'volume_volatility_by_hour': (hourly_data['volume_std'] / hourly_data['volume_mean']).to_dict()
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing hourly patterns: {str(e)}")
            return {}
    
def _segment_to_dict(self, segment: VolumeSegment) -> Dict[str, Any]:
        """Convert VolumeSegment to dictionary"""
        return {
            'segment_name': segment.segment_name,
            'start_time': segment.start_time.strftime('%H:%M'),
            'end_time': segment.end_time.strftime('%H:%M'),
            'total_volume': segment.total_volume,
            'avg_volume': segment.avg_volume,
            'volume_percentage': segment.volume_percentage,
            'price_efficiency': segment.price_efficiency,
            'accumulation_score': segment.accumulation_score,
            'distribution_score': segment.distribution_score,
            'institutional_activity': segment.institutional_activity
        }
    
def _profile_to_dict(self, profile: VolumeProfile) -> Dict[str, Any]:
        """Convert VolumeProfile to dictionary"""
        return {
            'price_level': profile.price_level,
            'volume': profile.volume,
            'volume_percentage': profile.volume_percentage,
            'value_area_high': profile.value_area_high,
            'value_area_low': profile.value_area_low,
            'point_of_control': profile.point_of_control,
            'support_resistance_strength': profile.support_resistance_strength
        }
    
def _flow_to_dict(self, flow: InstitutionalFlow) -> Dict[str, Any]:
        """Convert InstitutionalFlow to dictionary"""
        return {
            'flow_direction': flow.flow_direction,
            'flow_strength': flow.flow_strength,
            'large_block_ratio': flow.large_block_ratio,
            'stealth_accumulation': flow.stealth_accumulation,
            'institutional_signature': flow.institutional_signature,
            'smart_money_confidence': flow.smart_money_confidence
        }
    
def _generate_signal(self, value: Dict[str, Any], data: pd.DataFrame) -> Tuple[Optional[SignalType], float]:
        """
        Generate trading signals based on time segmented volume analysis
        """
        try:
            institutional_flow = value.get('institutional_flow', {})
            market_phase = value.get('market_phase', 'unknown')
            volume_concentration = value.get('volume_concentration', 0.0)
            smart_money_index = value.get('smart_money_index', 0.0)
            vwap_deviation = value.get('vwap_deviation', 0.0)
            
            # Signal strength factors
            signal_strength = 0.0
            signal_type = SignalType.NEUTRAL
            
            # Institutional flow signals
            flow_direction = institutional_flow.get('flow_direction', 'neutral')
            smart_money_confidence = institutional_flow.get('smart_money_confidence', 0.0)
            
            if smart_money_confidence > 0.7:
                if flow_direction == 'bullish':
                    signal_type = SignalType.BUY
                    signal_strength += smart_money_confidence * 0.4
                elif flow_direction == 'bearish':
                    signal_type = SignalType.SELL
                    signal_strength += smart_money_confidence * 0.4
            
            # Market phase signals
            if market_phase == 'accumulation':
                if signal_type != SignalType.SELL:
                    signal_type = SignalType.BUY
                    signal_strength += 0.3
            elif market_phase == 'distribution':
                if signal_type != SignalType.BUY:
                    signal_type = SignalType.SELL
                    signal_strength += 0.3
            
            # Volume concentration signals (high concentration = strong move)
            if volume_concentration > 0.6:
                signal_strength += 0.2
            
            # Smart money index signals
            if smart_money_index > 0.7:  # Smart money accumulating:
                if signal_type != SignalType.SELL:
                    signal_type = SignalType.BUY
                    signal_strength += 0.2
            elif smart_money_index < 0.3:  # Smart money distributing:
                if signal_type != SignalType.BUY:
                    signal_type = SignalType.SELL
                    signal_strength += 0.2
            
            # VWAP deviation signals
            if abs(vwap_deviation) > 0.02:  # Significant deviation:
                if vwap_deviation < -0.02 and signal_type == SignalType.BUY:
                    signal_strength += 0.1  # Buying below VWAP
                elif vwap_deviation > 0.02 and signal_type == SignalType.SELL:
                    signal_strength += 0.1  # Selling above VWAP
            
            # Ensure signal strength is within bounds
            signal_strength = min(1.0, signal_strength)
            
            # Minimum threshold for signal generation
            if signal_strength < 0.3:
                return SignalType.NEUTRAL, 0.0
            
            return signal_type, signal_strength
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return SignalType.NEUTRAL, 0.0
    
def _get_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get additional metadata for the calculation"""
        base_metadata = super()._get_metadata(data)
        
        volume_metadata = {
            'segment_resolution_minutes': self.parameters['segment_resolution'],
            'analysis_window': self.parameters['analysis_window'],
            'total_segments_analyzed': len(self.volume_segments),
            'total_volume_profiles': len(self.volume_profiles),
            'institutional_flows_tracked': len(self.institutional_flows),
            'clustering_algorithm': 'DBSCAN',
            'volume_profile_bins': self.parameters['volume_profile_bins']
        }
        
        base_metadata.update(volume_metadata)
        return base_metadata


def create_time_segmented_volume_indicator(parameters: Optional[Dict[str, Any]] = None) -> TimeSegmentedVolumeIndicator:
    """
    Factory function to create a TimeSegmentedVolumeIndicator instance
    
    Args:
        parameters: Optional dictionary of parameters to customize the indicator
        
    Returns:
        Configured TimeSegmentedVolumeIndicator instance
    """
    return TimeSegmentedVolumeIndicator(parameters=parameters)
def get_data_requirements(self):
        """
        Get data requirements for TimeSegmentedVolumeIndicator.
        
        Returns:
            list: List of DataRequirement objects
        """
from auj_platform.src.indicator_engine.indicators.base.standard_indicator import DataRequirement, DataType
        
        # Standard OHLCV requirements for most indicators
        return []
            DataRequirement()
                data_type=DataType.OHLCV,
                required_columns=['open', 'high', 'low', 'close', 'volume'],
                min_periods=20  # Reasonable default for most indicators
(            )
[        ]



# Example usage
if __name__ == "__main__":
    # Create sample data with realistic volume patterns
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01 09:00:00', end='2023-01-01 17:00:00', freq='1min')
    
    # Create realistic price and volume data
    base_price = 100
    base_volume = 1000
    
    # Simulate different volume patterns throughout the day
    n_periods = len(dates)
    price_changes = np.random.normal(0, 0.001, n_periods)
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # Volume patterns: higher volume during market open/close
    hour_factors = []
    for date in dates:
        hour = date.hour
        if hour in [9, 10]:  # Market open:
            factor = 2.0
        elif hour in [15, 16]:  # Market close:
            factor = 1.8
        elif hour in [11, 12, 13, 14]:  # Midday:
            factor = 0.7
        else:
            factor = 1.0
        hour_factors.append(factor)
    
    volumes = base_volume * np.array(hour_factors) * np.random.lognormal(0, 0.3, n_periods)
    
    # Add some volume spikes
    spike_indices = np.random.choice(n_periods, size=10, replace=False)
    volumes[spike_indices] *= np.random.uniform(3, 8, 10)
    
    sample_data = pd.DataFrame({)
        'datetime': dates,
        'high': prices * np.random.uniform(1.001, 1.01, n_periods),
        'low': prices * np.random.uniform(0.99, 0.999, n_periods),
        'close': prices,
        'volume': volumes
(    })
    sample_data.set_index('datetime', inplace=True)
    
    # Test the indicator
    indicator = create_time_segmented_volume_indicator({)
        'analysis_window': 200,
        'segment_resolution': 60,  # 1-hour segments
        'volume_profile_bins': 15
(    })
    
    try:
        result = indicator.calculate(sample_data)
        print("Time Segmented Volume Analysis Result:")
        print(f"Signal: {result.signal}, Confidence: {result.confidence:.3f}")
        print(f"Market Phase: {result.value.get('market_phase', 'unknown')}")
        print(f"Volume Concentration: {result.value.get('volume_concentration', 0):.3f}")
        print(f"Smart Money Index: {result.value.get('smart_money_index', 0):.3f}")
        print(f"VWAP Deviation: {result.value.get('vwap_deviation', 0):.1%}")
        
        # Display top volume segments
        segments = result.value.get('volume_segments', [])
        print(f"\nTop Volume Segments ({len(segments)}):")
        for i, segment in enumerate(segments[:3]):
            print(f"{i+1}. {segment['segment_name']} ({segment['start_time']}-{segment['end_time']})")
            print(f"   Volume: {segment['volume_percentage']:.1f}%, Accumulation: {segment['accumulation_score']:.2f}")
        
        # Display institutional flow
        inst_flow = result.value.get('institutional_flow', {})
        print(f"\nInstitutional Flow:")
        print(f"Direction: {inst_flow.get('flow_direction', 'unknown')}")
        print(f"Smart Money Confidence: {inst_flow.get('smart_money_confidence', 0):.2f}")
        print(f"Large Block Ratio: {inst_flow.get('large_block_ratio', 0):.2f}")
        
    except Exception as e:
        print(f"Error testing indicator: {str(e)}")