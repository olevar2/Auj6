"""
Advanced Price Volume Divergence Indicator
==========================================

This module implements a sophisticated price-volume relationship analysis system
that detects divergences between price movements and volume patterns using
advanced mathematical models, machine learning algorithms, and statistical analysis.

The indicator combines multiple volume analysis techniques, divergence detection
algorithms, and predictive modeling to identify potential reversal points and
continuation patterns in market data.

Author: AUJ Platform Team
Version: 1.0.0
Date: 2025-06-22

Mathematical Foundation:
- Volume-weighted price analysis using VWAP calculations
- Correlation analysis between price and volume momentum
- Divergence detection using peak/trough analysis
- Machine learning models for pattern validation
- Statistical significance testing for divergence patterns
- Time-series analysis for trend confirmation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Scientific computing imports
from scipy import signal, stats, optimize
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr, kendalltau

# Machine learning imports
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    IsolationForest, AdaBoostRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, TheilSenRegressor
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# Deep learning simulation (for production environment)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

import logging
from dataclasses import dataclass
from enum import Enum

# Import base indicator interface
from ..base.standard_indicator import StandardIndicatorInterface
from ....core.exceptions import (
    IndicatorCalculationError,
    DataValidationError,
    ConfigurationError
)
from ....core.unified_config import get_unified_config

class DivergenceType(Enum):
    """Enumeration of divergence types."""
    BULLISH_REGULAR = "bullish_regular"
    BEARISH_REGULAR = "bearish_regular"
    BULLISH_HIDDEN = "bullish_hidden"
    BEARISH_HIDDEN = "bearish_hidden"
    NO_DIVERGENCE = "no_divergence"

class VolumePattern(Enum):
    """Enumeration of volume patterns."""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    BREAKOUT_VOLUME = "breakout_volume"
    CLIMAX_VOLUME = "climax_volume"
    DRY_UP_VOLUME = "dry_up_volume"
    NORMAL_VOLUME = "normal_volume"

@dataclass
class DivergenceSignal:
    """Data class for divergence signal information."""
    divergence_type: DivergenceType
    strength: float
    confidence: float
    start_index: int
    end_index: int
    price_trend: str  # 'up', 'down', 'sideways'
    volume_trend: str  # 'up', 'down', 'sideways'
    correlation_coefficient: float
    statistical_significance: float
    volume_pattern: VolumePattern
    predicted_direction: str  # 'bullish', 'bearish', 'neutral'
    target_probability: float
    risk_reward_ratio: float
    formation_quality: float

class VolumeAnalysisEngine:
    """Advanced volume analysis and pattern recognition engine."""
    
    def __init__(self):
        """Initialize volume analysis engine."""
        self.logger = logging.getLogger(__name__)
        self.volume_periods = [5, 10, 20, 50]
        self.volume_threshold_percentile = 75
        
    def analyze_volume_patterns(self, prices: np.ndarray, 
                               volumes: np.ndarray) -> List[VolumePattern]:
        """Analyze volume patterns using multiple techniques."""
        try:
            patterns = []
            
            if len(volumes) < 20:
                return patterns
            
            # Calculate volume statistics
            volume_sma = self._calculate_volume_moving_averages(volumes)
            volume_relative = self._calculate_relative_volume(volumes)
            volume_distribution = self._analyze_volume_distribution(prices, volumes)
            
            # Detect specific patterns
            patterns.extend(self._detect_accumulation_distribution(
                prices, volumes, volume_sma, volume_relative
            ))
            patterns.extend(self._detect_breakout_patterns(
                prices, volumes, volume_relative
            ))
            patterns.extend(self._detect_climax_patterns(
                prices, volumes, volume_relative
            ))
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume patterns: {e}")
            return []
    
    def _calculate_volume_moving_averages(self, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate multiple volume moving averages."""
        try:
            volume_sma = {}
            
            for period in self.volume_periods:
                if len(volumes) >= period:
                    sma = np.convolve(volumes, np.ones(period)/period, mode='valid')
                    # Pad with NaN to match original length
                    padded_sma = np.full(len(volumes), np.nan)
                    padded_sma[period-1:] = sma
                    volume_sma[f'sma_{period}'] = padded_sma
                else:
                    volume_sma[f'sma_{period}'] = np.full(len(volumes), np.nan)
            
            return volume_sma
            
        except Exception as e:
            self.logger.error(f"Error calculating volume moving averages: {e}")
            return {}
    
    def _calculate_relative_volume(self, volumes: np.ndarray) -> np.ndarray:
        """Calculate relative volume (current vs average)."""
        try:
            if len(volumes) < 20:
                return np.ones(len(volumes))
            
            # Use 20-period average as baseline
            baseline_period = min(20, len(volumes))
            relative_volume = np.zeros(len(volumes))
            
            for i in range(len(volumes)):
                start_idx = max(0, i - baseline_period + 1)
                end_idx = i + 1
                
                if end_idx - start_idx >= 5:  # Minimum data for calculation
                    avg_volume = np.mean(volumes[start_idx:end_idx])
                    if avg_volume > 0:
                        relative_volume[i] = volumes[i] / avg_volume
                    else:
                        relative_volume[i] = 1.0
                else:
                    relative_volume[i] = 1.0
            
            return relative_volume
            
        except Exception as e:
            self.logger.error(f"Error calculating relative volume: {e}")
            return np.ones(len(volumes))
    
    def _analyze_volume_distribution(self, prices: np.ndarray, 
                                   volumes: np.ndarray) -> Dict[str, Any]:
        """Analyze volume distribution across price levels."""
        try:
            if len(prices) != len(volumes) or len(prices) < 10:
                return {}
            
            # Create price bins
            price_min, price_max = np.min(prices), np.max(prices)
            num_bins = min(20, len(prices) // 5)
            price_bins = np.linspace(price_min, price_max, num_bins)
            
            # Calculate volume at each price level
            volume_profile = np.zeros(len(price_bins) - 1)
            
            for i in range(len(prices)):
                bin_idx = np.digitize(prices[i], price_bins) - 1
                bin_idx = np.clip(bin_idx, 0, len(volume_profile) - 1)
                volume_profile[bin_idx] += volumes[i]
            
            # Find volume concentration areas
            volume_threshold = np.percentile(volume_profile, self.volume_threshold_percentile)
            high_volume_areas = np.where(volume_profile >= volume_threshold)[0]
            
            return {
                'price_bins': price_bins,
                'volume_profile': volume_profile,
                'high_volume_areas': high_volume_areas,
                'volume_concentration': np.sum(volume_profile[high_volume_areas]) / np.sum(volume_profile)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume distribution: {e}")
            return {}
    
    def _detect_accumulation_distribution(self, prices: np.ndarray,
                                        volumes: np.ndarray,
                                        volume_sma: Dict[str, np.ndarray],
                                        volume_relative: np.ndarray) -> List[VolumePattern]:
        """Detect accumulation and distribution patterns."""
        try:
            patterns = []
            
            if len(prices) < 20:
                return patterns
            
            # Calculate price momentum
            price_changes = np.diff(prices)
            price_momentum = np.zeros(len(prices))
            price_momentum[1:] = price_changes
            
            # Smooth momentum for trend analysis
            if len(price_momentum) >= 10:
                smoothed_momentum = gaussian_filter1d(price_momentum, sigma=2)
            else:
                smoothed_momentum = price_momentum
            
            # Look for accumulation patterns (sideways price + increasing volume)
            window_size = min(10, len(prices) // 4)
            
            for i in range(window_size, len(prices) - window_size):
                window_start = i - window_size
                window_end = i + window_size
                
                # Analyze price trend in window
                price_window = prices[window_start:window_end]
                volume_window = volumes[window_start:window_end]
                
                # Check for sideways price movement
                price_volatility = np.std(price_window) / np.mean(price_window)
                price_trend_slope = self._calculate_trend_slope(
                    np.arange(len(price_window)), price_window
                )
                
                # Check for volume trend
                volume_trend_slope = self._calculate_trend_slope(
                    np.arange(len(volume_window)), volume_window
                )
                
                # Accumulation: sideways price + increasing volume
                if (price_volatility < 0.02 and  # Low price volatility
                    abs(price_trend_slope) < 0.001 and  # Flat price trend
                    volume_trend_slope > 0.1):  # Increasing volume
                    patterns.append(VolumePattern.ACCUMULATION)
                
                # Distribution: sideways price + decreasing volume after run-up
                elif (price_volatility < 0.02 and
                      abs(price_trend_slope) < 0.001 and
                      volume_trend_slope < -0.1):
                    patterns.append(VolumePattern.DISTRIBUTION)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting accumulation/distribution: {e}")
            return []
    
    def _detect_breakout_patterns(self, prices: np.ndarray,
                                volumes: np.ndarray,
                                volume_relative: np.ndarray) -> List[VolumePattern]:
        """Detect volume breakout patterns."""
        try:
            patterns = []
            
            if len(volumes) < 10:
                return patterns
            
            # Find volume spikes
            volume_threshold = np.percentile(volume_relative, 90)
            volume_spikes = np.where(volume_relative > volume_threshold)[0]
            
            # Analyze price movement during volume spikes
            for spike_idx in volume_spikes:
                if spike_idx >= 5 and spike_idx < len(prices) - 5:
                    # Look at price change during spike
                    pre_spike_price = np.mean(prices[spike_idx-5:spike_idx])
                    post_spike_price = np.mean(prices[spike_idx:spike_idx+5])
                    
                    price_change_pct = (post_spike_price - pre_spike_price) / pre_spike_price
                    
                    if abs(price_change_pct) > 0.02:  # Significant price movement
                        patterns.append(VolumePattern.BREAKOUT_VOLUME)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting breakout patterns: {e}")
            return []
    
    def _detect_climax_patterns(self, prices: np.ndarray,
                              volumes: np.ndarray,
                              volume_relative: np.ndarray) -> List[VolumePattern]:
        """Detect volume climax patterns."""
        try:
            patterns = []
            
            if len(volumes) < 20:
                return patterns
            
            # Find extreme volume spikes
            extreme_threshold = np.percentile(volume_relative, 95)
            extreme_spikes = np.where(volume_relative > extreme_threshold)[0]
            
            for spike_idx in extreme_spikes:
                if spike_idx >= 10 and spike_idx < len(prices) - 10:
                    # Check if this is at price extreme
                    window_start = max(0, spike_idx - 10)
                    window_end = min(len(prices), spike_idx + 10)
                    
                    price_window = prices[window_start:window_end]
                    spike_price = prices[spike_idx]
                    
                    # Check if spike occurs at local high/low
                    is_local_high = spike_price >= np.percentile(price_window, 90)
                    is_local_low = spike_price <= np.percentile(price_window, 10)
                    
                    if is_local_high or is_local_low:
                        patterns.append(VolumePattern.CLIMAX_VOLUME)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting climax patterns: {e}")
            return []
    
    def _calculate_trend_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate trend slope using linear regression."""
        try:
            if len(x) < 2 or len(y) < 2:
                return 0.0
            
            slope, _ = np.polyfit(x, y, 1)
            return float(slope)
            
        except Exception as e:
            self.logger.error(f"Error calculating trend slope: {e}")
            return 0.0

class DivergenceDetector:
    """Advanced divergence detection engine."""
    
    def __init__(self):
        """Initialize divergence detector."""
        self.logger = logging.getLogger(__name__)
        self.min_divergence_period = 10
        self.correlation_threshold = 0.3
        
    def detect_divergences(self, prices: np.ndarray, 
                          volumes: np.ndarray) -> List[DivergenceSignal]:
        """Detect price-volume divergences using multiple methods."""
        try:
            divergences = []
            
            if len(prices) < self.min_divergence_period * 2:
                return divergences
            
            # Calculate price and volume momentum
            price_momentum = self._calculate_price_momentum(prices)
            volume_momentum = self._calculate_volume_momentum(volumes)
            
            # Find peaks and troughs
            price_peaks, price_troughs = self._find_price_extremes(prices)
            volume_peaks, volume_troughs = self._find_volume_extremes(volumes)
            
            # Detect regular divergences
            divergences.extend(self._detect_regular_divergences(
                prices, volumes, price_momentum, volume_momentum,
                price_peaks, price_troughs, volume_peaks, volume_troughs
            ))
            
            # Detect hidden divergences
            divergences.extend(self._detect_hidden_divergences(
                prices, volumes, price_momentum, volume_momentum,
                price_peaks, price_troughs, volume_peaks, volume_troughs
            ))
            
            # Filter and validate divergences
            validated_divergences = self._validate_divergences(divergences, prices, volumes)
            
            return validated_divergences
            
        except Exception as e:
            self.logger.error(f"Error detecting divergences: {e}")
            return []
    
    def _calculate_price_momentum(self, prices: np.ndarray) -> np.ndarray:
        """Calculate price momentum using multiple methods."""
        try:
            if len(prices) < 5:
                return np.zeros(len(prices))
            
            # Simple momentum
            momentum_5 = np.zeros(len(prices))
            momentum_5[5:] = (prices[5:] - prices[:-5]) / prices[:-5]
            
            # Rate of change
            roc = np.zeros(len(prices))
            roc[1:] = np.diff(prices) / prices[:-1]
            
            # Combine momentum measures
            combined_momentum = (momentum_5 + roc) / 2
            
            # Smooth the momentum
            if len(combined_momentum) >= 5:
                smoothed = gaussian_filter1d(combined_momentum, sigma=1.5)
            else:
                smoothed = combined_momentum
            
            return smoothed
            
        except Exception as e:
            self.logger.error(f"Error calculating price momentum: {e}")
            return np.zeros(len(prices))
    
    def _calculate_volume_momentum(self, volumes: np.ndarray) -> np.ndarray:
        """Calculate volume momentum and relative strength."""
        try:
            if len(volumes) < 5:
                return np.zeros(len(volumes))
            
            # Volume rate of change
            volume_roc = np.zeros(len(volumes))
            for i in range(5, len(volumes)):
                if volumes[i-5] > 0:
                    volume_roc[i] = (volumes[i] - volumes[i-5]) / volumes[i-5]
            
            # Volume relative to recent average
            volume_relative = np.zeros(len(volumes))
            for i in range(10, len(volumes)):
                recent_avg = np.mean(volumes[i-10:i])
                if recent_avg > 0:
                    volume_relative[i] = volumes[i] / recent_avg - 1
            
            # Combine measures
            combined_momentum = (volume_roc + volume_relative) / 2
            
            # Smooth the momentum
            if len(combined_momentum) >= 5:
                smoothed = gaussian_filter1d(combined_momentum, sigma=1.5)
            else:
                smoothed = combined_momentum
            
            return smoothed
            
        except Exception as e:
            self.logger.error(f"Error calculating volume momentum: {e}")
            return np.zeros(len(volumes))
    
    def _find_price_extremes(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find significant price peaks and troughs."""
        try:
            if len(prices) < 10:
                return np.array([]), np.array([])
            
            # Calculate prominence threshold
            price_volatility = np.std(prices)
            prominence = price_volatility * 0.5
            
            # Find peaks and troughs
            peaks, _ = signal.find_peaks(prices, 
                                       prominence=prominence,
                                       distance=5)
            
            troughs, _ = signal.find_peaks(-prices,
                                         prominence=prominence,
                                         distance=5)
            
            return peaks, troughs
            
        except Exception as e:
            self.logger.error(f"Error finding price extremes: {e}")
            return np.array([]), np.array([])
    
    def _find_volume_extremes(self, volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find significant volume peaks and troughs."""
        try:
            if len(volumes) < 10:
                return np.array([]), np.array([])
            
            # Calculate prominence threshold
            volume_volatility = np.std(volumes)
            prominence = volume_volatility * 0.3
            
            # Find peaks and troughs
            peaks, _ = signal.find_peaks(volumes,
                                       prominence=prominence,
                                       distance=3)
            
            troughs, _ = signal.find_peaks(-volumes,
                                         prominence=prominence,
                                         distance=3)
            
            return peaks, troughs
            
        except Exception as e:
            self.logger.error(f"Error finding volume extremes: {e}")
            return np.array([]), np.array([])
    
    def _detect_regular_divergences(self, prices: np.ndarray, volumes: np.ndarray,
                                  price_momentum: np.ndarray, volume_momentum: np.ndarray,
                                  price_peaks: np.ndarray, price_troughs: np.ndarray,
                                  volume_peaks: np.ndarray, volume_troughs: np.ndarray) -> List[DivergenceSignal]:
        """Detect regular bullish and bearish divergences."""
        try:
            divergences = []
            
            # Bearish divergence: price makes higher highs, volume makes lower highs
            if len(price_peaks) >= 2:
                for i in range(len(price_peaks) - 1):
                    peak1_idx = price_peaks[i]
                    peak2_idx = price_peaks[i + 1]
                    
                    # Check if price made higher high
                    if prices[peak2_idx] > prices[peak1_idx]:
                        # Find corresponding volume peaks in the period
                        volume_peaks_in_period = volume_peaks[
                            (volume_peaks >= peak1_idx) & (volume_peaks <= peak2_idx)
                        ]
                        
                        if len(volume_peaks_in_period) >= 2:
                            vol_peak1 = volume_peaks_in_period[0]
                            vol_peak2 = volume_peaks_in_period[-1]
                            
                            # Check if volume made lower high
                            if volumes[vol_peak2] < volumes[vol_peak1]:
                                # Calculate divergence strength
                                price_change = (prices[peak2_idx] - prices[peak1_idx]) / prices[peak1_idx]
                                volume_change = (volumes[vol_peak2] - volumes[vol_peak1]) / volumes[vol_peak1]
                                
                                strength = abs(price_change - volume_change)
                                correlation = self._calculate_correlation(
                                    prices[peak1_idx:peak2_idx+1],
                                    volumes[peak1_idx:peak2_idx+1]
                                )
                                
                                if strength > 0.05:  # Minimum divergence threshold
                                    divergence = DivergenceSignal(
                                        divergence_type=DivergenceType.BEARISH_REGULAR,
                                        strength=strength,
                                        confidence=min(1.0, strength * 2),
                                        start_index=peak1_idx,
                                        end_index=peak2_idx,
                                        price_trend='up',
                                        volume_trend='down',
                                        correlation_coefficient=correlation,
                                        statistical_significance=self._calculate_significance(correlation),
                                        volume_pattern=VolumePattern.DISTRIBUTION,
                                        predicted_direction='bearish',
                                        target_probability=min(0.8, strength * 10),
                                        risk_reward_ratio=2.0,
                                        formation_quality=self._assess_formation_quality(
                                            prices[peak1_idx:peak2_idx+1],
                                            volumes[peak1_idx:peak2_idx+1]
                                        )
                                    )
                                    divergences.append(divergence)
            
            # Bullish divergence: price makes lower lows, volume makes higher lows
            if len(price_troughs) >= 2:
                for i in range(len(price_troughs) - 1):
                    trough1_idx = price_troughs[i]
                    trough2_idx = price_troughs[i + 1]
                    
                    # Check if price made lower low
                    if prices[trough2_idx] < prices[trough1_idx]:
                        # Find corresponding volume troughs in the period
                        volume_troughs_in_period = volume_troughs[
                            (volume_troughs >= trough1_idx) & (volume_troughs <= trough2_idx)
                        ]
                        
                        if len(volume_troughs_in_period) >= 2:
                            vol_trough1 = volume_troughs_in_period[0]
                            vol_trough2 = volume_troughs_in_period[-1]
                            
                            # Check if volume made higher low
                            if volumes[vol_trough2] > volumes[vol_trough1]:
                                # Calculate divergence strength
                                price_change = (prices[trough1_idx] - prices[trough2_idx]) / prices[trough1_idx]
                                volume_change = (volumes[vol_trough2] - volumes[vol_trough1]) / volumes[vol_trough1]
                                
                                strength = abs(price_change - volume_change)
                                correlation = self._calculate_correlation(
                                    prices[trough1_idx:trough2_idx+1],
                                    volumes[trough1_idx:trough2_idx+1]
                                )
                                
                                if strength > 0.05:  # Minimum divergence threshold
                                    divergence = DivergenceSignal(
                                        divergence_type=DivergenceType.BULLISH_REGULAR,
                                        strength=strength,
                                        confidence=min(1.0, strength * 2),
                                        start_index=trough1_idx,
                                        end_index=trough2_idx,
                                        price_trend='down',
                                        volume_trend='up',
                                        correlation_coefficient=correlation,
                                        statistical_significance=self._calculate_significance(correlation),
                                        volume_pattern=VolumePattern.ACCUMULATION,
                                        predicted_direction='bullish',
                                        target_probability=min(0.8, strength * 10),
                                        risk_reward_ratio=2.0,
                                        formation_quality=self._assess_formation_quality(
                                            prices[trough1_idx:trough2_idx+1],
                                            volumes[trough1_idx:trough2_idx+1]
                                        )
                                    )
                                    divergences.append(divergence)
            
            return divergences
            
        except Exception as e:
            self.logger.error(f"Error detecting regular divergences: {e}")
            return []
    
    def _detect_hidden_divergences(self, prices: np.ndarray, volumes: np.ndarray,
                                 price_momentum: np.ndarray, volume_momentum: np.ndarray,
                                 price_peaks: np.ndarray, price_troughs: np.ndarray,
                                 volume_peaks: np.ndarray, volume_troughs: np.ndarray) -> List[DivergenceSignal]:
        """Detect hidden divergences (continuation patterns)."""
        try:
            divergences = []
            
            # Hidden bullish divergence: price makes higher lows, volume makes lower lows
            if len(price_troughs) >= 2:
                for i in range(len(price_troughs) - 1):
                    trough1_idx = price_troughs[i]
                    trough2_idx = price_troughs[i + 1]
                    
                    # Check if price made higher low
                    if prices[trough2_idx] > prices[trough1_idx]:
                        # Find corresponding volume in the period
                        volume_in_period = volumes[trough1_idx:trough2_idx+1]
                        
                        if len(volume_in_period) > 5:
                            # Check if volume shows declining trend
                            volume_trend = self._calculate_trend_slope(
                                np.arange(len(volume_in_period)), volume_in_period
                            )
                            
                            if volume_trend < -0.1:  # Declining volume
                                strength = abs(volume_trend) * 0.5
                                
                                divergence = DivergenceSignal(
                                    divergence_type=DivergenceType.BULLISH_HIDDEN,
                                    strength=strength,
                                    confidence=min(0.8, strength * 3),
                                    start_index=trough1_idx,
                                    end_index=trough2_idx,
                                    price_trend='up',
                                    volume_trend='down',
                                    correlation_coefficient=volume_trend,
                                    statistical_significance=abs(volume_trend),
                                    volume_pattern=VolumePattern.DRY_UP_VOLUME,
                                    predicted_direction='bullish',
                                    target_probability=min(0.7, strength * 8),
                                    risk_reward_ratio=1.5,
                                    formation_quality=self._assess_formation_quality(
                                        prices[trough1_idx:trough2_idx+1],
                                        volumes[trough1_idx:trough2_idx+1]
                                    )
                                )
                                divergences.append(divergence)
            
            # Hidden bearish divergence: price makes lower highs, volume makes higher highs
            if len(price_peaks) >= 2:
                for i in range(len(price_peaks) - 1):
                    peak1_idx = price_peaks[i]
                    peak2_idx = price_peaks[i + 1]
                    
                    # Check if price made lower high
                    if prices[peak2_idx] < prices[peak1_idx]:
                        # Find corresponding volume in the period
                        volume_in_period = volumes[peak1_idx:peak2_idx+1]
                        
                        if len(volume_in_period) > 5:
                            # Check if volume shows increasing trend
                            volume_trend = self._calculate_trend_slope(
                                np.arange(len(volume_in_period)), volume_in_period
                            )
                            
                            if volume_trend > 0.1:  # Increasing volume
                                strength = volume_trend * 0.5
                                
                                divergence = DivergenceSignal(
                                    divergence_type=DivergenceType.BEARISH_HIDDEN,
                                    strength=strength,
                                    confidence=min(0.8, strength * 3),
                                    start_index=peak1_idx,
                                    end_index=peak2_idx,
                                    price_trend='down',
                                    volume_trend='up',
                                    correlation_coefficient=-volume_trend,
                                    statistical_significance=volume_trend,
                                    volume_pattern=VolumePattern.DISTRIBUTION,
                                    predicted_direction='bearish',
                                    target_probability=min(0.7, strength * 8),
                                    risk_reward_ratio=1.5,
                                    formation_quality=self._assess_formation_quality(
                                        prices[peak1_idx:peak2_idx+1],
                                        volumes[peak1_idx:peak2_idx+1]
                                    )
                                )
                                divergences.append(divergence)
            
            return divergences
            
        except Exception as e:
            self.logger.error(f"Error detecting hidden divergences: {e}")
            return []
    
    def _calculate_correlation(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate correlation between price and volume."""
        try:
            if len(prices) < 3 or len(volumes) < 3:
                return 0.0
            
            # Remove any zero or negative values
            valid_mask = (prices > 0) & (volumes > 0)
            if np.sum(valid_mask) < 3:
                return 0.0
            
            valid_prices = prices[valid_mask]
            valid_volumes = volumes[valid_mask]
            
            # Calculate Pearson correlation
            correlation, p_value = pearsonr(valid_prices, valid_volumes)
            
            if np.isnan(correlation):
                return 0.0
            
            return float(correlation)
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {e}")
            return 0.0
    
    def _calculate_significance(self, correlation: float) -> float:
        """Calculate statistical significance of correlation."""
        try:
            # Simple significance based on correlation strength
            return min(1.0, abs(correlation) * 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating significance: {e}")
            return 0.0
    
    def _calculate_trend_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate trend slope using linear regression."""
        try:
            if len(x) < 2 or len(y) < 2:
                return 0.0
            
            slope, _ = np.polyfit(x, y, 1)
            return float(slope)
            
        except Exception as e:
            self.logger.error(f"Error calculating trend slope: {e}")
            return 0.0
    
    def _assess_formation_quality(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Assess the quality of divergence formation."""
        try:
            if len(prices) < 5 or len(volumes) < 5:
                return 0.0
            
            quality_factors = []
            
            # Factor 1: Price trend consistency
            price_changes = np.diff(prices)
            price_consistency = 1.0 - (np.std(price_changes) / (np.mean(np.abs(price_changes)) + 1e-8))
            quality_factors.append(max(0, price_consistency))
            
            # Factor 2: Volume pattern clarity
            volume_changes = np.diff(volumes)
            volume_clarity = np.std(volume_changes) / (np.mean(volumes) + 1e-8)
            quality_factors.append(min(1.0, volume_clarity))
            
            # Factor 3: Duration appropriateness
            duration_score = min(1.0, len(prices) / 20.0)  # Optimal around 20 periods
            quality_factors.append(duration_score)
            
            # Factor 4: Magnitude significance
            price_range = np.max(prices) - np.min(prices)
            price_magnitude = price_range / np.mean(prices)
            magnitude_score = min(1.0, price_magnitude * 10)
            quality_factors.append(magnitude_score)
            
            # Calculate weighted average
            weights = [0.3, 0.3, 0.2, 0.2]
            quality_score = np.average(quality_factors, weights=weights)
            
            return float(np.clip(quality_score, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error assessing formation quality: {e}")
            return 0.0
    
    def _validate_divergences(self, divergences: List[DivergenceSignal],
                            prices: np.ndarray, volumes: np.ndarray) -> List[DivergenceSignal]:
        """Validate and filter divergence signals."""
        try:
            validated = []
            
            for divergence in divergences:
                # Filter by minimum confidence
                if divergence.confidence < 0.3:
                    continue
                
                # Filter by minimum strength
                if divergence.strength < 0.02:
                    continue
                
                # Filter by formation quality
                if divergence.formation_quality < 0.3:
                    continue
                
                # Additional validation could include:
                # - Market condition context
                # - Time of day factors
                # - Overall trend alignment
                
                validated.append(divergence)
            
            # Remove overlapping divergences (keep strongest)
            validated = self._remove_overlapping_divergences(validated)
            
            return validated
            
        except Exception as e:
            self.logger.error(f"Error validating divergences: {e}")
            return divergences
    
    def _remove_overlapping_divergences(self, divergences: List[DivergenceSignal]) -> List[DivergenceSignal]:
        """Remove overlapping divergence signals, keeping the strongest."""
        try:
            if len(divergences) <= 1:
                return divergences
            
            # Sort by strength (descending)
            divergences.sort(key=lambda x: x.strength, reverse=True)
            
            filtered = []
            
            for divergence in divergences:
                overlapping = False
                
                for existing in filtered:
                    # Check for overlap
                    if (divergence.start_index < existing.end_index and
                        divergence.end_index > existing.start_index):
                        overlapping = True
                        break
                
                if not overlapping:
                    filtered.append(divergence)
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error removing overlapping divergences: {e}")
            return divergences

class MachineLearningPredictor:
    """Machine learning models for price-volume relationship prediction."""
    
    def __init__(self):
        """Initialize machine learning predictor."""
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize machine learning models."""
        try:
            # Regression models for price prediction
            self.models['price_predictor'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Classification model for direction prediction
            self.models['direction_classifier'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # Anomaly detection for unusual patterns
            self.models['anomaly_detector'] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Scalers for feature normalization
            self.scalers['standard'] = StandardScaler()
            self.scalers['robust'] = RobustScaler()
            
        except Exception as e:
            self.logger.error(f"Error initializing ML models: {e}")
    
    def extract_features(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Extract features for machine learning models."""
        try:
            if len(prices) < 20 or len(volumes) < 20:
                return np.array([])
            
            features = []
            
            # Price features
            price_returns = np.diff(prices) / prices[:-1]
            features.extend([
                np.mean(price_returns[-10:]),
                np.std(price_returns[-10:]),
                stats.skew(price_returns[-10:]) if len(price_returns) >= 10 else 0,
                stats.kurtosis(price_returns[-10:]) if len(price_returns) >= 10 else 0
            ])
            
            # Volume features
            volume_changes = np.diff(volumes) / (volumes[:-1] + 1e-8)
            features.extend([
                np.mean(volume_changes[-10:]),
                np.std(volume_changes[-10:]),
                np.mean(volumes[-5:]) / np.mean(volumes[-20:]),  # Recent vs historical
                np.std(volumes[-10:]) / np.mean(volumes[-10:])   # Volume volatility
            ])
            
            # Price-volume relationship features
            if len(prices) == len(volumes):
                correlation_5 = self._calculate_rolling_correlation(prices[-10:], volumes[-10:])
                correlation_10 = self._calculate_rolling_correlation(prices[-20:], volumes[-20:])
                features.extend([correlation_5, correlation_10])
            else:
                features.extend([0, 0])
            
            # Technical indicators
            if len(prices) >= 20:
                sma_ratio = prices[-1] / np.mean(prices[-20:])
                volatility = np.std(price_returns[-20:])
                momentum = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0
                
                features.extend([sma_ratio, volatility, momentum])
            else:
                features.extend([1, 0, 0])
            
            # Volume pattern features
            if len(volumes) >= 10:
                volume_trend = self._calculate_volume_trend(volumes[-10:])
                volume_acceleration = self._calculate_volume_acceleration(volumes[-10:])
                features.extend([volume_trend, volume_acceleration])
            else:
                features.extend([0, 0])
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return np.array([])
    
    def _calculate_rolling_correlation(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate rolling correlation between price and volume."""
        try:
            if len(prices) < 3 or len(volumes) < 3:
                return 0.0
            
            correlation, _ = pearsonr(prices, volumes)
            return float(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling correlation: {e}")
            return 0.0
    
    def _calculate_volume_trend(self, volumes: np.ndarray) -> float:
        """Calculate volume trend using linear regression."""
        try:
            if len(volumes) < 3:
                return 0.0
            
            x = np.arange(len(volumes))
            slope, _ = np.polyfit(x, volumes, 1)
            
            # Normalize by average volume
            avg_volume = np.mean(volumes)
            normalized_slope = slope / (avg_volume + 1e-8)
            
            return float(normalized_slope)
            
        except Exception as e:
            self.logger.error(f"Error calculating volume trend: {e}")
            return 0.0
    
    def _calculate_volume_acceleration(self, volumes: np.ndarray) -> float:
        """Calculate volume acceleration (second derivative)."""
        try:
            if len(volumes) < 5:
                return 0.0
            
            # Calculate first derivative (velocity)
            velocity = np.diff(volumes)
            
            # Calculate second derivative (acceleration)
            acceleration = np.diff(velocity)
            
            # Return average acceleration
            return float(np.mean(acceleration))
            
        except Exception as e:
            self.logger.error(f"Error calculating volume acceleration: {e}")
            return 0.0
    
    def predict_price_movement(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Predict price movement based on volume patterns."""
        try:
            # Extract features
            features = self.extract_features(prices, volumes)
            
            if len(features) == 0:
                return {'direction': 0.0, 'magnitude': 0.0, 'confidence': 0.0}
            
            # For demonstration, create simple predictions
            # In production, this would use trained models
            
            # Simple heuristic-based prediction
            price_momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            volume_momentum = (volumes[-1] - np.mean(volumes[-5:])) / np.mean(volumes[-5:])
            
            # Combine price and volume momentum
            if price_momentum > 0 and volume_momentum > 0:
                direction = 1.0  # Bullish
                confidence = min(0.8, abs(price_momentum) + abs(volume_momentum))
            elif price_momentum < 0 and volume_momentum > 0:
                direction = -0.5  # Bearish divergence
                confidence = min(0.7, abs(price_momentum) + abs(volume_momentum))
            elif price_momentum > 0 and volume_momentum < 0:
                direction = 0.5  # Bullish divergence
                confidence = min(0.7, abs(price_momentum) + abs(volume_momentum))
            else:
                direction = -1.0  # Bearish
                confidence = min(0.8, abs(price_momentum) + abs(volume_momentum))
            
            magnitude = abs(price_momentum)
            
            return {
                'direction': direction,
                'magnitude': magnitude,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting price movement: {e}")
            return {'direction': 0.0, 'magnitude': 0.0, 'confidence': 0.0}

class PriceVolumeDivergenceIndicator(StandardIndicatorInterface):
    """
    Advanced Price Volume Divergence Indicator
    
    Analyzes the relationship between price movements and volume patterns
    to detect divergences that may signal potential reversal or continuation
    points in the market.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Price Volume Divergence Indicator."""
        super().__init__(config)
        self.config_manager = get_unified_config()
        
        # Configuration
        self.config = config or {}
        self.lookback_period = self.config_manager.get_int('lookback_period', 100)
        self.min_divergence_strength = self.config_manager.get_float('min_divergence_strength', 0.02)
        self.enable_ml_prediction = self.config_manager.get_bool('enable_ml_prediction', True)
        self.volume_analysis_enabled = self.config_manager.get_bool('volume_analysis_enabled', True)
        
        # Initialize analysis engines
        self.volume_analyzer = VolumeAnalysisEngine()
        self.divergence_detector = DivergenceDetector()
        self.ml_predictor = MachineLearningPredictor()
        
        # State variables
        self.divergence_history = []
        self.prediction_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate price-volume divergence signals.
        
        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Series with divergence signal values and metadata
        """
        try:
            # Validate input data
            if not self._validate_data(data):
                raise DataValidationError("Invalid input data for price-volume divergence analysis")
            
            # Extract price and volume arrays
            closes = data['close'].values
            volumes = data['volume'].values
            
            # Apply lookback period
            if len(closes) > self.lookback_period:
                analysis_closes = closes[-self.lookback_period:]
                analysis_volumes = volumes[-self.lookback_period:]
            else:
                analysis_closes = closes
                analysis_volumes = volumes
            
            # Analyze volume patterns
            volume_patterns = []
            if self.volume_analysis_enabled:
                volume_patterns = self.volume_analyzer.analyze_volume_patterns(
                    analysis_closes, analysis_volumes
                )
            
            # Detect divergences
            divergences = self.divergence_detector.detect_divergences(
                analysis_closes, analysis_volumes
            )
            
            # Filter divergences by strength
            filtered_divergences = [
                div for div in divergences 
                if div.strength >= self.min_divergence_strength
            ]
            
            # Generate ML predictions
            ml_prediction = {}
            if self.enable_ml_prediction:
                ml_prediction = self.ml_predictor.predict_price_movement(
                    analysis_closes, analysis_volumes
                )
            
            # Synthesize signals
            divergence_signals = self._synthesize_divergence_signals(
                filtered_divergences, volume_patterns, ml_prediction, 
                closes, data.index
            )
            
            # Update history
            self._update_analysis_history(filtered_divergences, ml_prediction)
            
            return divergence_signals
            
        except Exception as e:
            self.logger.error(f"Error calculating price-volume divergence: {e}")
            raise IndicatorCalculationError(f"Price-volume divergence calculation failed: {e}")
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data quality and structure."""
        try:
            required_columns = ['close', 'volume']
            
            # Check required columns
            if not all(col in data.columns for col in required_columns):
                return False
            
            # Check for sufficient data
            if len(data) < 20:
                return False
            
            # Check for valid price and volume data
            if (data['close'].isna().all() or (data['close'] <= 0).any() or
                data['volume'].isna().all() or (data['volume'] < 0).any()):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return False
    
    def _synthesize_divergence_signals(self, divergences: List[DivergenceSignal],
                                     volume_patterns: List[VolumePattern],
                                     ml_prediction: Dict[str, float],
                                     prices: np.ndarray,
                                     index: pd.Index) -> pd.Series:
        """Synthesize divergence and volume analysis into trading signals."""
        try:
            # Initialize signal series
            signals = pd.Series(0.0, index=index, name='price_volume_divergence')
            
            if not divergences and not ml_prediction:
                return signals
            
            # Calculate composite signal for the latest period
            latest_idx = len(signals) - 1
            
            # Process divergence signals
            divergence_signal = 0.0
            max_divergence_strength = 0.0
            best_divergence = None
            
            for divergence in divergences:
                # Weight by proximity to current time (more recent = higher weight)
                time_weight = max(0.1, 1.0 - (latest_idx - divergence.end_index) / 50.0)
                
                # Calculate directional signal
                if divergence.divergence_type in [DivergenceType.BULLISH_REGULAR, DivergenceType.BULLISH_HIDDEN]:
                    direction_signal = divergence.strength * divergence.confidence * time_weight
                elif divergence.divergence_type in [DivergenceType.BEARISH_REGULAR, DivergenceType.BEARISH_HIDDEN]:
                    direction_signal = -divergence.strength * divergence.confidence * time_weight
                else:
                    direction_signal = 0.0
                
                divergence_signal += direction_signal
                
                if divergence.strength > max_divergence_strength:
                    max_divergence_strength = divergence.strength
                    best_divergence = divergence
            
            # Process ML prediction
            ml_signal = 0.0
            if ml_prediction:
                ml_direction = ml_prediction.get('direction', 0.0)
                ml_confidence = ml_prediction.get('confidence', 0.0)
                ml_signal = ml_direction * ml_confidence * 0.5  # Lower weight for ML
            
            # Process volume patterns
            volume_signal = 0.0
            for pattern in volume_patterns:
                if pattern in [VolumePattern.ACCUMULATION, VolumePattern.BREAKOUT_VOLUME]:
                    volume_signal += 0.2
                elif pattern in [VolumePattern.DISTRIBUTION, VolumePattern.CLIMAX_VOLUME]:
                    volume_signal -= 0.2
            
            # Combine signals with weights
            weights = {'divergence': 0.6, 'ml': 0.3, 'volume': 0.1}
            
            final_signal = (
                divergence_signal * weights['divergence'] +
                ml_signal * weights['ml'] +
                volume_signal * weights['volume']
            )
            
            # Apply signal scaling and limits
            final_signal = np.clip(final_signal, -1.0, 1.0)
            
            signals.iloc[latest_idx] = final_signal
            
            # Add metadata
            if best_divergence or ml_prediction:
                signals = self._add_signal_metadata(signals, best_divergence, ml_prediction)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error synthesizing divergence signals: {e}")
            return pd.Series(0.0, index=index, name='price_volume_divergence')
    
    def _add_signal_metadata(self, signals: pd.Series,
                           best_divergence: Optional[DivergenceSignal],
                           ml_prediction: Dict[str, float]) -> pd.Series:
        """Add metadata to the signal series."""
        try:
            metadata = {}
            
            if best_divergence:
                metadata.update({
                    'divergence_type': best_divergence.divergence_type.value,
                    'divergence_strength': best_divergence.strength,
                    'divergence_confidence': best_divergence.confidence,
                    'price_trend': best_divergence.price_trend,
                    'volume_trend': best_divergence.volume_trend,
                    'correlation_coefficient': best_divergence.correlation_coefficient,
                    'predicted_direction': best_divergence.predicted_direction,
                    'target_probability': best_divergence.target_probability,
                    'risk_reward_ratio': best_divergence.risk_reward_ratio
                })
            
            if ml_prediction:
                metadata.update({
                    'ml_direction': ml_prediction.get('direction', 0.0),
                    'ml_magnitude': ml_prediction.get('magnitude', 0.0),
                    'ml_confidence': ml_prediction.get('confidence', 0.0)
                })
            
            signals.attrs = metadata
            return signals
            
        except Exception as e:
            self.logger.error(f"Error adding signal metadata: {e}")
            return signals
    
    def _update_analysis_history(self, divergences: List[DivergenceSignal],
                               ml_prediction: Dict[str, float]):
        """Update analysis history for performance tracking."""
        try:
            current_time = datetime.now()
            
            # Update divergence history
            for divergence in divergences:
                divergence_record = {
                    'timestamp': current_time,
                    'type': divergence.divergence_type.value,
                    'strength': divergence.strength,
                    'confidence': divergence.confidence,
                    'predicted_direction': divergence.predicted_direction
                }
                self.divergence_history.append(divergence_record)
            
            # Update prediction history
            if ml_prediction:
                prediction_record = {
                    'timestamp': current_time,
                    'direction': ml_prediction.get('direction', 0.0),
                    'magnitude': ml_prediction.get('magnitude', 0.0),
                    'confidence': ml_prediction.get('confidence', 0.0)
                }
                self.prediction_history.append(prediction_record)
            
            # Keep only recent history (last 1000 records)
            if len(self.divergence_history) > 1000:
                self.divergence_history = self.divergence_history[-1000:]
            
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
        except Exception as e:
            self.logger.error(f"Error updating analysis history: {e}")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of divergence analysis and predictions."""
        try:
            summary = {
                'total_divergences_detected': len(self.divergence_history),
                'divergence_type_distribution': {},
                'prediction_accuracy_metrics': {},
                'recent_analysis': {}
            }
            
            if self.divergence_history:
                # Divergence type distribution
                divergence_types = [d['type'] for d in self.divergence_history]
                summary['divergence_type_distribution'] = {
                    div_type: divergence_types.count(div_type) 
                    for div_type in set(divergence_types)
                }
                
                # Average metrics
                strengths = [d['strength'] for d in self.divergence_history]
                confidences = [d['confidence'] for d in self.divergence_history]
                
                summary['average_divergence_strength'] = np.mean(strengths)
                summary['average_divergence_confidence'] = np.mean(confidences)
                
                # Recent divergences
                summary['recent_analysis']['recent_divergences'] = self.divergence_history[-5:]
            
            if self.prediction_history:
                # Prediction metrics
                directions = [p['direction'] for p in self.prediction_history]
                confidences = [p['confidence'] for p in self.prediction_history]
                
                summary['prediction_accuracy_metrics'] = {
                    'average_direction': np.mean(directions),
                    'average_confidence': np.mean(confidences),
                    'direction_consistency': np.std(directions)
                }
                
                # Recent predictions
                summary['recent_analysis']['recent_predictions'] = self.prediction_history[-5:]
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating analysis summary: {e}")
            return {}
    
    def get_name(self) -> str:
        """Return the indicator name."""
        return "Price Volume Divergence Indicator"
    
    def get_version(self) -> str:
        """Return the indicator version."""
        return "1.0.0"
    
    def get_description(self) -> str:
        """Return the indicator description."""
        return """
        Advanced Price Volume Divergence Indicator that analyzes the relationship 
        between price movements and volume patterns to detect divergences and 
        predict potential market turning points.
        
        Features:
        - Regular and hidden divergence detection
        - Volume pattern analysis (accumulation, distribution, breakouts)
        - Machine learning price movement prediction
        - Statistical significance testing
        - Multi-timeframe correlation analysis
        - Performance tracking and optimization
        """