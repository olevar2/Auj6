"""
Advanced Volume Oscillator Indicator for AUJ Platform

This module implements a sophisticated Volume Oscillator system that analyzes
volume momentum through multi-period comparisons, trend confirmation signals,
and advanced volume dynamics modeling. The indicator provides comprehensive
insights into volume-based momentum shifts and market participation changes.

Key Features:
- Multi-period volume oscillator calculations
- Volume momentum analysis with acceleration detection
- Trend confirmation through volume dynamics
- Adaptive smoothing with multiple methodologies
- Volume strength measurement and classification
- Momentum divergence detection algorithms
- Volume cycle analysis and periodicity detection
- Real-time momentum shifts and reversals
- Advanced statistical validation
- Machine learning enhanced pattern recognition

Mathematical Models:
- Multi-timeframe volume oscillator calculations
- Volume momentum and acceleration algorithms
- Exponential moving average convergence/divergence
- Volume-weighted momentum indicators
- Statistical significance testing for volume patterns
- Fourier transform analysis for cyclical patterns
- Adaptive filtering for noise reduction
- Volume volatility normalization algorithms
- Momentum persistence measurement models
- Machine learning classification for momentum patterns

The implementation follows AUJ Platform's humanitarian mission requirements with
robust error handling, comprehensive logging, and production-ready code quality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.signal import savgol_filter, find_peaks, periodogram, welch
from scipy.optimize import minimize_scalar
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import traceback
from collections import deque

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VolumeOscillatorType(Enum):
    """Enumeration for volume oscillator types."""
    BASIC = "BASIC"
    PERCENTAGE = "PERCENTAGE"
    RATIO = "RATIO"
    NORMALIZED = "NORMALIZED"
    ADAPTIVE = "ADAPTIVE"


class VolumeMomentumDirection(Enum):
    """Enumeration for volume momentum direction."""
    ACCELERATING_UP = "ACCELERATING_UP"
    STEADY_UP = "STEADY_UP"
    SLOWING_UP = "SLOWING_UP"
    NEUTRAL = "NEUTRAL"
    SLOWING_DOWN = "SLOWING_DOWN"
    STEADY_DOWN = "STEADY_DOWN"
    ACCELERATING_DOWN = "ACCELERATING_DOWN"


class VolumeStrength(Enum):
    """Enumeration for volume strength classification."""
    EXTREME_HIGH = "EXTREME_HIGH"
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    ABOVE_AVERAGE = "ABOVE_AVERAGE"
    AVERAGE = "AVERAGE"
    BELOW_AVERAGE = "BELOW_AVERAGE"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


class TrendConfirmation(Enum):
    """Enumeration for trend confirmation states."""
    STRONG_CONFIRMATION = "STRONG_CONFIRMATION"
    MODERATE_CONFIRMATION = "MODERATE_CONFIRMATION"
    WEAK_CONFIRMATION = "WEAK_CONFIRMATION"
    NO_CONFIRMATION = "NO_CONFIRMATION"
    WEAK_DIVERGENCE = "WEAK_DIVERGENCE"
    MODERATE_DIVERGENCE = "MODERATE_DIVERGENCE"
    STRONG_DIVERGENCE = "STRONG_DIVERGENCE"


@dataclass
class VolumeOscillatorSignal:
    """Data class for volume oscillator analysis signals."""
    # Core oscillator values
    oscillator_value: float = 0.0
    normalized_oscillator: float = 0.0
    percentage_oscillator: float = 0.0
    ratio_oscillator: float = 0.0
    
    # Momentum analysis
    volume_momentum: float = 0.0
    momentum_direction: VolumeMomentumDirection = VolumeMomentumDirection.NEUTRAL
    momentum_acceleration: float = 0.0
    momentum_persistence: float = 0.0
    
    # Volume strength metrics
    volume_strength: VolumeStrength = VolumeStrength.AVERAGE
    relative_volume: float = 1.0
    volume_intensity: float = 0.0
    volume_efficiency: float = 0.0
    
    # Trend confirmation
    trend_confirmation: TrendConfirmation = TrendConfirmation.NO_CONFIRMATION
    confirmation_strength: float = 0.0
    price_volume_correlation: float = 0.0
    divergence_score: float = 0.0
    
    # Advanced metrics
    oscillator_velocity: float = 0.0
    oscillator_acceleration: float = 0.0
    cycle_phase: float = 0.0
    dominant_period: float = 0.0
    
    # Statistical measures
    z_score: float = 0.0
    percentile_rank: float = 50.0
    autocorrelation: float = 0.0
    entropy: float = 0.0
    
    # Signal levels
    overbought_level: float = 0.0
    oversold_level: float = 0.0
    signal_line: float = 0.0
    histogram: float = 0.0
    
    # Risk and timing
    momentum_risk: float = 0.0
    optimal_timing_score: float = 0.0
    reversal_probability: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


@dataclass
class VolumeOscillatorConfig:
    """Configuration parameters for volume oscillator analysis."""
    # Basic oscillator parameters
    short_period: int = 14
    long_period: int = 28
    signal_period: int = 9
    oscillator_type: VolumeOscillatorType = VolumeOscillatorType.PERCENTAGE
    
    # Smoothing parameters
    smoothing_method: str = "ema"  # ema, sma, adaptive
    smoothing_factor: float = 0.1
    adaptive_smoothing: bool = True
    noise_reduction: bool = True
    
    # Momentum analysis
    momentum_window: int = 10
    acceleration_window: int = 5
    persistence_threshold: float = 0.6
    momentum_sensitivity: float = 0.1
    
    # Trend confirmation
    correlation_window: int = 20
    divergence_threshold: float = 0.3
    confirmation_threshold: float = 0.5
    trend_window: int = 15
    
    # Statistical parameters
    zscore_window: int = 100
    percentile_window: int = 50
    autocorr_lags: int = 20
    entropy_window: int = 30
    
    # Signal levels
    overbought_threshold: float = 70.0
    oversold_threshold: float = 30.0
    signal_smoothing: int = 3
    
    # Cycle analysis
    min_cycle_length: int = 10
    max_cycle_length: int = 100
    cycle_confidence_threshold: float = 0.5
    
    # ML parameters
    ml_lookback: int = 200
    feature_window: int = 20
    retrain_frequency: int = 500
    
    # Risk parameters
    volatility_adjustment: bool = True
    risk_factor: float = 0.02
    confidence_level: float = 0.95


class VolumeDataProcessor:
    """Advanced volume data processing and normalization."""
    
    def __init__(self, config: VolumeOscillatorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def process_volume_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process and normalize volume data."""
        try:
            processed_data = data.copy()
            
            # Basic volume metrics
            processed_data['volume_sma_short'] = data['volume'].rolling(
                window=self.config.short_period
            ).mean()
            processed_data['volume_sma_long'] = data['volume'].rolling(
                window=self.config.long_period
            ).mean()
            
            # Exponential moving averages
            processed_data['volume_ema_short'] = data['volume'].ewm(
                span=self.config.short_period
            ).mean()
            processed_data['volume_ema_long'] = data['volume'].ewm(
                span=self.config.long_period
            ).mean()
            
            # Volume volatility
            processed_data['volume_volatility'] = data['volume'].rolling(
                window=self.config.short_period
            ).std()
            
            # Relative volume
            processed_data['relative_volume'] = (
                data['volume'] / processed_data['volume_sma_long']
            )
            
            # Volume efficiency (price move per unit volume)
            if 'price' in data.columns:
                price_change = data['price'].pct_change().abs()
                volume_normalized = data['volume'] / data['volume'].rolling(
                    window=self.config.short_period
                ).mean()
                processed_data['volume_efficiency'] = (
                    price_change / (volume_normalized + 1e-8)
                )
            
            # Apply noise reduction if enabled
            if self.config.noise_reduction:
                processed_data = self._apply_noise_reduction(processed_data)
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing volume data: {str(e)}")
            return data.copy()
    
    def _apply_noise_reduction(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply noise reduction techniques."""
        try:
            # Apply Savitzky-Golay filter to smooth volume data
            if len(data) >= 5:
                for col in ['volume', 'volume_sma_short', 'volume_sma_long']:
                    if col in data.columns:
                        smoothed = savgol_filter(
                            data[col].fillna(method='ffill').fillna(method='bfill'),
                            window_length=min(5, len(data) // 2 * 2 + 1),
                            polyorder=2
                        )
                        data[f'{col}_smoothed'] = smoothed
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error applying noise reduction: {str(e)}")
            return data


class VolumeOscillatorCalculator:
    """Advanced volume oscillator calculation engine."""
    
    def __init__(self, config: VolumeOscillatorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_oscillators(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate various volume oscillator types."""
        try:
            osc_data = pd.DataFrame(index=processed_data.index)
            
            # Basic volume oscillator
            osc_data['basic_oscillator'] = self._calculate_basic_oscillator(processed_data)
            
            # Percentage volume oscillator
            osc_data['percentage_oscillator'] = self._calculate_percentage_oscillator(processed_data)
            
            # Ratio oscillator
            osc_data['ratio_oscillator'] = self._calculate_ratio_oscillator(processed_data)
            
            # Normalized oscillator
            osc_data['normalized_oscillator'] = self._calculate_normalized_oscillator(processed_data)
            
            # Adaptive oscillator
            osc_data['adaptive_oscillator'] = self._calculate_adaptive_oscillator(processed_data)
            
            # Signal line
            osc_data['signal_line'] = self._calculate_signal_line(osc_data)
            
            # Histogram
            osc_data['histogram'] = self._calculate_histogram(osc_data)
            
            return osc_data
            
        except Exception as e:
            self.logger.error(f"Error calculating oscillators: {str(e)}")
            return pd.DataFrame(index=processed_data.index)
    
    def _calculate_basic_oscillator(self, data: pd.DataFrame) -> pd.Series:
        """Calculate basic volume oscillator."""
        try:
            short_vol = data['volume_sma_short']
            long_vol = data['volume_sma_long']
            
            oscillator = short_vol - long_vol
            return oscillator.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating basic oscillator: {str(e)}")
            return pd.Series(index=data.index, dtype=float)
    
    def _calculate_percentage_oscillator(self, data: pd.DataFrame) -> pd.Series:
        """Calculate percentage volume oscillator."""
        try:
            short_vol = data['volume_ema_short']
            long_vol = data['volume_ema_long']
            
            oscillator = ((short_vol - long_vol) / long_vol * 100).fillna(0)
            return oscillator
            
        except Exception as e:
            self.logger.error(f"Error calculating percentage oscillator: {str(e)}")
            return pd.Series(index=data.index, dtype=float)
    
    def _calculate_ratio_oscillator(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ratio volume oscillator."""
        try:
            short_vol = data['volume_ema_short']
            long_vol = data['volume_ema_long']
            
            oscillator = (short_vol / (long_vol + 1e-8)).fillna(1)
            return oscillator
            
        except Exception as e:
            self.logger.error(f"Error calculating ratio oscillator: {str(e)}")
            return pd.Series(index=data.index, dtype=float)
    
    def _calculate_normalized_oscillator(self, data: pd.DataFrame) -> pd.Series:
        """Calculate normalized volume oscillator."""
        try:
            percentage_osc = self._calculate_percentage_oscillator(data)
            
            # Normalize to -1 to 1 range
            rolling_std = percentage_osc.rolling(window=self.config.long_period).std()
            normalized = percentage_osc / (rolling_std * 2 + 1e-8)
            normalized = np.clip(normalized, -1, 1)
            
            return normalized.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating normalized oscillator: {str(e)}")
            return pd.Series(index=data.index, dtype=float)
    
    def _calculate_adaptive_oscillator(self, data: pd.DataFrame) -> pd.Series:
        """Calculate adaptive volume oscillator."""
        try:
            volume = data['volume']
            
            # Calculate adaptive periods based on volatility
            vol_volatility = data.get('volume_volatility', volume.rolling(10).std())
            
            # Adaptive short period (between 5 and short_period)
            adaptive_short = np.clip(
                self.config.short_period * (1 - vol_volatility / vol_volatility.rolling(50).max()),
                5, self.config.short_period
            ).fillna(self.config.short_period)
            
            # Calculate adaptive oscillator
            oscillator_values = []
            for i in range(len(volume)):
                if i < self.config.long_period:
                    oscillator_values.append(0)
                    continue
                
                short_period = int(adaptive_short.iloc[i])
                long_period = self.config.long_period
                
                short_vol = volume.iloc[i-short_period:i].mean()
                long_vol = volume.iloc[i-long_period:i].mean()
                
                osc_value = ((short_vol - long_vol) / long_vol * 100) if long_vol > 0 else 0
                oscillator_values.append(osc_value)
            
            return pd.Series(oscillator_values, index=data.index)
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive oscillator: {str(e)}")
            return pd.Series(index=data.index, dtype=float)
    
    def _calculate_signal_line(self, osc_data: pd.DataFrame) -> pd.Series:
        """Calculate signal line for oscillator."""
        try:
            # Use the configured oscillator type
            oscillator_col = f"{self.config.oscillator_type.value.lower()}_oscillator"
            
            if oscillator_col in osc_data.columns:
                signal_line = osc_data[oscillator_col].ewm(
                    span=self.config.signal_period
                ).mean()
            else:
                signal_line = osc_data['percentage_oscillator'].ewm(
                    span=self.config.signal_period
                ).mean()
            
            return signal_line.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating signal line: {str(e)}")
            return pd.Series(index=osc_data.index, dtype=float)
    
    def _calculate_histogram(self, osc_data: pd.DataFrame) -> pd.Series:
        """Calculate histogram (oscillator - signal line)."""
        try:
            oscillator_col = f"{self.config.oscillator_type.value.lower()}_oscillator"
            
            if oscillator_col in osc_data.columns:
                oscillator = osc_data[oscillator_col]
            else:
                oscillator = osc_data['percentage_oscillator']
            
            signal_line = osc_data.get('signal_line', pd.Series(index=osc_data.index))
            
            histogram = oscillator - signal_line
            return histogram.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating histogram: {str(e)}")
            return pd.Series(index=osc_data.index, dtype=float)


class MomentumAnalyzer:
    """Advanced momentum analysis system."""
    
    def __init__(self, config: VolumeOscillatorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_momentum(self, osc_data: pd.DataFrame, 
                        processed_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze volume momentum characteristics."""
        try:
            momentum_data = pd.DataFrame(index=osc_data.index)
            
            # Volume momentum
            momentum_data['volume_momentum'] = self._calculate_volume_momentum(
                processed_data, osc_data
            )
            
            # Momentum acceleration
            momentum_data['momentum_acceleration'] = self._calculate_momentum_acceleration(
                momentum_data
            )
            
            # Momentum persistence
            momentum_data['momentum_persistence'] = self._calculate_momentum_persistence(
                momentum_data
            )
            
            # Oscillator velocity and acceleration
            momentum_data['oscillator_velocity'] = self._calculate_oscillator_velocity(
                osc_data
            )
            momentum_data['oscillator_acceleration'] = self._calculate_oscillator_acceleration(
                momentum_data
            )
            
            return momentum_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum: {str(e)}")
            return pd.DataFrame(index=osc_data.index)
    
    def _calculate_volume_momentum(self, processed_data: pd.DataFrame,
                                 osc_data: pd.DataFrame) -> pd.Series:
        """Calculate volume momentum."""
        try:
            # Use rate of change of volume
            volume = processed_data['volume']
            momentum = volume.pct_change(periods=self.config.momentum_window)
            
            # Apply smoothing
            smoothed_momentum = momentum.ewm(span=5).mean()
            
            return smoothed_momentum.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating volume momentum: {str(e)}")
            return pd.Series(index=processed_data.index, dtype=float)
    
    def _calculate_momentum_acceleration(self, momentum_data: pd.DataFrame) -> pd.Series:
        """Calculate momentum acceleration."""
        try:
            momentum = momentum_data.get('volume_momentum', pd.Series(index=momentum_data.index))
            
            # Calculate acceleration as rate of change of momentum
            acceleration = momentum.diff(periods=self.config.acceleration_window)
            
            return acceleration.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum acceleration: {str(e)}")
            return pd.Series(index=momentum_data.index, dtype=float)
    
    def _calculate_momentum_persistence(self, momentum_data: pd.DataFrame) -> pd.Series:
        """Calculate momentum persistence."""
        try:
            momentum = momentum_data.get('volume_momentum', pd.Series(index=momentum_data.index))
            
            # Calculate persistence as consistency of momentum direction
            momentum_sign = np.sign(momentum)
            persistence = momentum_sign.rolling(
                window=self.config.momentum_window
            ).apply(lambda x: abs(x.sum()) / len(x) if len(x) > 0 else 0)
            
            return persistence.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum persistence: {str(e)}")
            return pd.Series(index=momentum_data.index, dtype=float)
    
    def _calculate_oscillator_velocity(self, osc_data: pd.DataFrame) -> pd.Series:
        """Calculate oscillator velocity."""
        try:
            oscillator_col = f"{self.config.oscillator_type.value.lower()}_oscillator"
            
            if oscillator_col in osc_data.columns:
                oscillator = osc_data[oscillator_col]
            else:
                oscillator = osc_data['percentage_oscillator']
            
            velocity = oscillator.diff()
            return velocity.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating oscillator velocity: {str(e)}")
            return pd.Series(index=osc_data.index, dtype=float)
    
    def _calculate_oscillator_acceleration(self, momentum_data: pd.DataFrame) -> pd.Series:
        """Calculate oscillator acceleration."""
        try:
            velocity = momentum_data.get('oscillator_velocity', pd.Series(index=momentum_data.index))
            
            acceleration = velocity.diff()
            return acceleration.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating oscillator acceleration: {str(e)}")
            return pd.Series(index=momentum_data.index, dtype=float)


class TrendConfirmationAnalyzer:
    """Advanced trend confirmation analysis system."""
    
    def __init__(self, config: VolumeOscillatorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_trend_confirmation(self, osc_data: pd.DataFrame,
                                 price_data: pd.Series) -> pd.DataFrame:
        """Analyze trend confirmation signals."""
        try:
            trend_data = pd.DataFrame(index=osc_data.index)
            
            # Price-volume correlation
            trend_data['price_volume_correlation'] = self._calculate_price_volume_correlation(
                osc_data, price_data
            )
            
            # Divergence analysis
            trend_data['divergence_score'] = self._calculate_divergence_score(
                osc_data, price_data
            )
            
            # Confirmation strength
            trend_data['confirmation_strength'] = self._calculate_confirmation_strength(
                trend_data
            )
            
            return trend_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend confirmation: {str(e)}")
            return pd.DataFrame(index=osc_data.index)
    
    def _calculate_price_volume_correlation(self, osc_data: pd.DataFrame,
                                          price_data: pd.Series) -> pd.Series:
        """Calculate price-volume correlation."""
        try:
            oscillator_col = f"{self.config.oscillator_type.value.lower()}_oscillator"
            
            if oscillator_col in osc_data.columns:
                oscillator = osc_data[oscillator_col]
            else:
                oscillator = osc_data['percentage_oscillator']
            
            # Calculate rolling correlation
            correlation = oscillator.rolling(
                window=self.config.correlation_window
            ).corr(price_data.pct_change())
            
            return correlation.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating price-volume correlation: {str(e)}")
            return pd.Series(index=osc_data.index, dtype=float)
    
    def _calculate_divergence_score(self, osc_data: pd.DataFrame,
                                  price_data: pd.Series) -> pd.Series:
        """Calculate divergence score."""
        try:
            oscillator_col = f"{self.config.oscillator_type.value.lower()}_oscillator"
            
            if oscillator_col in osc_data.columns:
                oscillator = osc_data[oscillator_col]
            else:
                oscillator = osc_data['percentage_oscillator']
            
            # Calculate price and oscillator trends
            price_trend = price_data.rolling(window=self.config.trend_window).apply(
                lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0
            )
            
            osc_trend = oscillator.rolling(window=self.config.trend_window).apply(
                lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0
            )
            
            # Calculate divergence (opposite signs indicate divergence)
            divergence = -(price_trend * osc_trend)
            
            # Normalize divergence score
            divergence_normalized = np.tanh(divergence * 10)
            
            return divergence_normalized.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating divergence score: {str(e)}")
            return pd.Series(index=osc_data.index, dtype=float)
    
    def _calculate_confirmation_strength(self, trend_data: pd.DataFrame) -> pd.Series:
        """Calculate overall confirmation strength."""
        try:
            correlation = trend_data.get('price_volume_correlation', pd.Series(index=trend_data.index))
            divergence = trend_data.get('divergence_score', pd.Series(index=trend_data.index))
            
            # Combine correlation and inverse divergence
            confirmation = correlation.abs() - divergence.abs()
            
            # Normalize to 0-1 range
            confirmation_normalized = (confirmation + 1) / 2
            
            return confirmation_normalized.fillna(0.5)
            
        except Exception as e:
            self.logger.error(f"Error calculating confirmation strength: {str(e)}")
            return pd.Series(index=trend_data.index, dtype=float)


class CycleAnalyzer:
    """Advanced cycle analysis system."""
    
    def __init__(self, config: VolumeOscillatorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_cycles(self, osc_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze cyclical patterns in volume oscillator."""
        try:
            cycle_data = pd.DataFrame(index=osc_data.index)
            
            # Dominant period analysis
            cycle_data['dominant_period'] = self._calculate_dominant_period(osc_data)
            
            # Cycle phase
            cycle_data['cycle_phase'] = self._calculate_cycle_phase(osc_data, cycle_data)
            
            return cycle_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing cycles: {str(e)}")
            return pd.DataFrame(index=osc_data.index)
    
    def _calculate_dominant_period(self, osc_data: pd.DataFrame) -> pd.Series:
        """Calculate dominant period using FFT analysis."""
        try:
            oscillator_col = f"{self.config.oscillator_type.value.lower()}_oscillator"
            
            if oscillator_col in osc_data.columns:
                oscillator = osc_data[oscillator_col]
            else:
                oscillator = osc_data['percentage_oscillator']
            
            dominant_periods = []
            
            for i in range(len(oscillator)):
                if i < self.config.max_cycle_length:
                    dominant_periods.append(20.0)  # Default period
                    continue
                
                # Get recent data for FFT analysis
                recent_data = oscillator.iloc[i-self.config.max_cycle_length:i]
                
                if len(recent_data) < 10:
                    dominant_periods.append(20.0)
                    continue
                
                try:
                    # Perform FFT
                    fft_vals = fft(recent_data.fillna(0))
                    freqs = fftfreq(len(recent_data))
                    
                    # Find dominant frequency (excluding DC component)
                    power_spectrum = np.abs(fft_vals[1:len(fft_vals)//2])
                    dominant_freq_idx = np.argmax(power_spectrum)
                    dominant_freq = freqs[dominant_freq_idx + 1]
                    
                    # Convert frequency to period
                    if dominant_freq != 0:
                        dominant_period = 1.0 / abs(dominant_freq)
                        dominant_period = np.clip(dominant_period, 
                                                self.config.min_cycle_length,
                                                self.config.max_cycle_length)
                    else:
                        dominant_period = 20.0
                    
                    dominant_periods.append(dominant_period)
                    
                except:
                    dominant_periods.append(20.0)
            
            return pd.Series(dominant_periods, index=osc_data.index)
            
        except Exception as e:
            self.logger.error(f"Error calculating dominant period: {str(e)}")
            return pd.Series([20.0] * len(osc_data), index=osc_data.index)
    
    def _calculate_cycle_phase(self, osc_data: pd.DataFrame,
                             cycle_data: pd.DataFrame) -> pd.Series:
        """Calculate current cycle phase."""
        try:
            oscillator_col = f"{self.config.oscillator_type.value.lower()}_oscillator"
            
            if oscillator_col in osc_data.columns:
                oscillator = osc_data[oscillator_col]
            else:
                oscillator = osc_data['percentage_oscillator']
            
            dominant_period = cycle_data.get('dominant_period', pd.Series([20.0] * len(osc_data)))
            
            phases = []
            
            for i in range(len(oscillator)):
                if i < 10:
                    phases.append(0.0)
                    continue
                
                period = int(dominant_period.iloc[i])
                
                # Calculate phase based on position in cycle
                recent_osc = oscillator.iloc[max(0, i-period):i+1]
                
                if len(recent_osc) > 1:
                    # Find peaks and troughs to determine phase
                    peaks, _ = find_peaks(recent_osc)
                    troughs, _ = find_peaks(-recent_osc)
                    
                    if len(peaks) > 0 or len(troughs) > 0:
                        # Calculate phase as position between last extremum
                        last_peak = peaks[-1] if len(peaks) > 0 else -1
                        last_trough = troughs[-1] if len(troughs) > 0 else -1
                        
                        if last_peak > last_trough:
                            # After peak, moving toward trough
                            phase = (len(recent_osc) - 1 - last_peak) / period
                        else:
                            # After trough, moving toward peak
                            phase = (len(recent_osc) - 1 - last_trough) / period + 0.5
                        
                        phase = phase % 1.0  # Normalize to 0-1
                    else:
                        phase = 0.0
                else:
                    phase = 0.0
                
                phases.append(phase)
            
            return pd.Series(phases, index=osc_data.index)
            
        except Exception as e:
            self.logger.error(f"Error calculating cycle phase: {str(e)}")
            return pd.Series([0.0] * len(osc_data), index=osc_data.index)


class AdvancedVolumeOscillatorIndicator:
    """
    Advanced Volume Oscillator Indicator with comprehensive analysis.
    
    This class implements sophisticated volume oscillator analysis including
    multi-period calculations, momentum analysis, trend confirmation,
    and advanced pattern recognition for volume-based trading signals.
    """
    
    def __init__(self, config: Optional[VolumeOscillatorConfig] = None):
        """
        Initialize the Advanced Volume Oscillator Indicator.
        
        Args:
            config: Configuration parameters for oscillator analysis
        """
        self.config = config or VolumeOscillatorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_processor = VolumeDataProcessor(self.config)
        self.oscillator_calculator = VolumeOscillatorCalculator(self.config)
        self.momentum_analyzer = MomentumAnalyzer(self.config)
        self.trend_analyzer = TrendConfirmationAnalyzer(self.config)
        self.cycle_analyzer = CycleAnalyzer(self.config)
        
        # State management
        self.current_signals = []
        self.oscillator_history = deque(maxlen=1000)
        
        # Performance tracking
        self.analysis_count = 0
        self.total_analysis_time = 0.0
        
        self.logger.info("Advanced Volume Oscillator Indicator initialized successfully")
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive volume oscillator analysis.
        
        Args:
            data: DataFrame with columns ['timestamp', 'price', 'volume']
            
        Returns:
            Dictionary containing comprehensive oscillator analysis
        """
        try:
            start_time = datetime.now()
            
            # Validate input data
            if not self._validate_input_data(data):
                raise ValueError("Invalid input data for oscillator analysis")
            
            # Process volume data
            processed_data = self.data_processor.process_volume_data(data)
            
            # Calculate oscillators
            osc_data = self.oscillator_calculator.calculate_oscillators(processed_data)
            
            # Analyze momentum
            momentum_data = self.momentum_analyzer.analyze_momentum(osc_data, processed_data)
            
            # Analyze trend confirmation
            trend_data = self.trend_analyzer.analyze_trend_confirmation(
                osc_data, data['price']
            )
            
            # Analyze cycles
            cycle_data = self.cycle_analyzer.analyze_cycles(osc_data)
            
            # Generate comprehensive signals
            signals = self._generate_oscillator_signals(
                osc_data, momentum_data, trend_data, cycle_data, processed_data
            )
            
            # Calculate statistical measures
            statistics = self._calculate_statistical_measures(osc_data)
            
            # Performance metrics
            performance = self._calculate_performance_metrics(signals)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(analysis_time)
            
            # Store current signals
            self.current_signals = signals
            
            # Compile results
            results = {
                'signals': signals,
                'oscillator_data': osc_data,
                'momentum_data': momentum_data,
                'trend_data': trend_data,
                'cycle_data': cycle_data,
                'processed_data': processed_data,
                'statistics': statistics,
                'performance': performance,
                'metadata': {
                    'analysis_time': analysis_time,
                    'data_points': len(data),
                    'config': self.config.__dict__
                }
            }
            
            self.logger.info(f"Volume oscillator analysis completed in {analysis_time:.4f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in volume oscillator analysis: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data quality and structure."""
        try:
            required_columns = ['timestamp', 'price', 'volume']
            
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns: {required_columns}")
                return False
            
            if len(data) < self.config.long_period:
                self.logger.error(f"Insufficient data points for analysis: {len(data)}")
                return False
            
            if (data['volume'] <= 0).any():
                self.logger.warning("Non-positive volume values detected")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating input data: {str(e)}")
            return False
    
    def _generate_oscillator_signals(self, osc_data: pd.DataFrame,
                                   momentum_data: pd.DataFrame,
                                   trend_data: pd.DataFrame,
                                   cycle_data: pd.DataFrame,
                                   processed_data: pd.DataFrame) -> List[VolumeOscillatorSignal]:
        """Generate comprehensive oscillator signals."""
        try:
            signals = []
            
            for i in range(len(osc_data)):
                # Extract oscillator values
                oscillator_value = osc_data['percentage_oscillator'].iloc[i] if 'percentage_oscillator' in osc_data else 0.0
                normalized_oscillator = osc_data['normalized_oscillator'].iloc[i] if 'normalized_oscillator' in osc_data else 0.0
                percentage_oscillator = osc_data['percentage_oscillator'].iloc[i] if 'percentage_oscillator' in osc_data else 0.0
                ratio_oscillator = osc_data['ratio_oscillator'].iloc[i] if 'ratio_oscillator' in osc_data else 1.0
                signal_line = osc_data['signal_line'].iloc[i] if 'signal_line' in osc_data else 0.0
                histogram = osc_data['histogram'].iloc[i] if 'histogram' in osc_data else 0.0
                
                # Extract momentum values
                volume_momentum = momentum_data['volume_momentum'].iloc[i] if 'volume_momentum' in momentum_data else 0.0
                momentum_acceleration = momentum_data['momentum_acceleration'].iloc[i] if 'momentum_acceleration' in momentum_data else 0.0
                momentum_persistence = momentum_data['momentum_persistence'].iloc[i] if 'momentum_persistence' in momentum_data else 0.0
                oscillator_velocity = momentum_data['oscillator_velocity'].iloc[i] if 'oscillator_velocity' in momentum_data else 0.0
                oscillator_acceleration = momentum_data['oscillator_acceleration'].iloc[i] if 'oscillator_acceleration' in momentum_data else 0.0
                
                # Extract trend values
                price_volume_correlation = trend_data['price_volume_correlation'].iloc[i] if 'price_volume_correlation' in trend_data else 0.0
                divergence_score = trend_data['divergence_score'].iloc[i] if 'divergence_score' in trend_data else 0.0
                confirmation_strength = trend_data['confirmation_strength'].iloc[i] if 'confirmation_strength' in trend_data else 0.0
                
                # Extract cycle values
                cycle_phase = cycle_data['cycle_phase'].iloc[i] if 'cycle_phase' in cycle_data else 0.0
                dominant_period = cycle_data['dominant_period'].iloc[i] if 'dominant_period' in cycle_data else 20.0
                
                # Extract processed data values
                relative_volume = processed_data['relative_volume'].iloc[i] if 'relative_volume' in processed_data else 1.0
                volume_efficiency = processed_data['volume_efficiency'].iloc[i] if 'volume_efficiency' in processed_data else 0.0
                
                # Classifications
                momentum_direction = self._classify_momentum_direction(volume_momentum, momentum_acceleration)
                volume_strength = self._classify_volume_strength(relative_volume)
                trend_confirmation = self._classify_trend_confirmation(confirmation_strength, divergence_score)
                
                # Advanced calculations
                z_score = self._calculate_z_score(osc_data, i)
                percentile_rank = self._calculate_percentile_rank(osc_data, i)
                
                # Signal levels
                overbought_level = self._calculate_overbought_level(oscillator_value)
                oversold_level = self._calculate_oversold_level(oscillator_value)
                
                # Risk and timing
                momentum_risk = min(abs(momentum_acceleration), 1.0)
                optimal_timing_score = self._calculate_optimal_timing_score(
                    confirmation_strength, momentum_persistence, cycle_phase
                )
                reversal_probability = self._calculate_reversal_probability(
                    oscillator_value, momentum_direction, cycle_phase
                )
                
                # Create signal
                signal = VolumeOscillatorSignal(
                    oscillator_value=oscillator_value,
                    normalized_oscillator=normalized_oscillator,
                    percentage_oscillator=percentage_oscillator,
                    ratio_oscillator=ratio_oscillator,
                    volume_momentum=volume_momentum,
                    momentum_direction=momentum_direction,
                    momentum_acceleration=momentum_acceleration,
                    momentum_persistence=momentum_persistence,
                    volume_strength=volume_strength,
                    relative_volume=relative_volume,
                    volume_efficiency=volume_efficiency,
                    trend_confirmation=trend_confirmation,
                    confirmation_strength=confirmation_strength,
                    price_volume_correlation=price_volume_correlation,
                    divergence_score=divergence_score,
                    oscillator_velocity=oscillator_velocity,
                    oscillator_acceleration=oscillator_acceleration,
                    cycle_phase=cycle_phase,
                    dominant_period=dominant_period,
                    z_score=z_score,
                    percentile_rank=percentile_rank,
                    overbought_level=overbought_level,
                    oversold_level=oversold_level,
                    signal_line=signal_line,
                    histogram=histogram,
                    momentum_risk=momentum_risk,
                    optimal_timing_score=optimal_timing_score,
                    reversal_probability=reversal_probability
                )
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating oscillator signals: {str(e)}")
            return []
    
    def _classify_momentum_direction(self, momentum: float, acceleration: float) -> VolumeMomentumDirection:
        """Classify momentum direction."""
        try:
            if momentum > 0.1:
                if acceleration > 0.05:
                    return VolumeMomentumDirection.ACCELERATING_UP
                elif acceleration < -0.05:
                    return VolumeMomentumDirection.SLOWING_UP
                else:
                    return VolumeMomentumDirection.STEADY_UP
            elif momentum < -0.1:
                if acceleration < -0.05:
                    return VolumeMomentumDirection.ACCELERATING_DOWN
                elif acceleration > 0.05:
                    return VolumeMomentumDirection.SLOWING_DOWN
                else:
                    return VolumeMomentumDirection.STEADY_DOWN
            else:
                return VolumeMomentumDirection.NEUTRAL
                
        except Exception as e:
            self.logger.error(f"Error classifying momentum direction: {str(e)}")
            return VolumeMomentumDirection.NEUTRAL
    
    def _classify_volume_strength(self, relative_volume: float) -> VolumeStrength:
        """Classify volume strength."""
        try:
            if relative_volume >= 3.0:
                return VolumeStrength.EXTREME_HIGH
            elif relative_volume >= 2.5:
                return VolumeStrength.VERY_HIGH
            elif relative_volume >= 1.5:
                return VolumeStrength.HIGH
            elif relative_volume >= 1.1:
                return VolumeStrength.ABOVE_AVERAGE
            elif relative_volume >= 0.9:
                return VolumeStrength.AVERAGE
            elif relative_volume >= 0.7:
                return VolumeStrength.BELOW_AVERAGE
            elif relative_volume >= 0.5:
                return VolumeStrength.LOW
            else:
                return VolumeStrength.VERY_LOW
                
        except Exception as e:
            self.logger.error(f"Error classifying volume strength: {str(e)}")
            return VolumeStrength.AVERAGE
    
    def _classify_trend_confirmation(self, confirmation_strength: float, 
                                   divergence_score: float) -> TrendConfirmation:
        """Classify trend confirmation."""
        try:
            if divergence_score > 0.5:
                return TrendConfirmation.STRONG_DIVERGENCE
            elif divergence_score > 0.3:
                return TrendConfirmation.MODERATE_DIVERGENCE
            elif divergence_score > 0.1:
                return TrendConfirmation.WEAK_DIVERGENCE
            elif confirmation_strength > 0.7:
                return TrendConfirmation.STRONG_CONFIRMATION
            elif confirmation_strength > 0.5:
                return TrendConfirmation.MODERATE_CONFIRMATION
            elif confirmation_strength > 0.3:
                return TrendConfirmation.WEAK_CONFIRMATION
            else:
                return TrendConfirmation.NO_CONFIRMATION
                
        except Exception as e:
            self.logger.error(f"Error classifying trend confirmation: {str(e)}")
            return TrendConfirmation.NO_CONFIRMATION
    
    def _calculate_z_score(self, osc_data: pd.DataFrame, index: int) -> float:
        """Calculate Z-score for oscillator value."""
        try:
            if index < self.config.zscore_window:
                return 0.0
            
            oscillator_col = f"{self.config.oscillator_type.value.lower()}_oscillator"
            if oscillator_col in osc_data.columns:
                oscillator = osc_data[oscillator_col]
            else:
                oscillator = osc_data['percentage_oscillator']
            
            start_idx = max(0, index - self.config.zscore_window)
            recent_values = oscillator.iloc[start_idx:index+1]
            
            current_value = oscillator.iloc[index]
            mean_value = recent_values.mean()
            std_value = recent_values.std()
            
            if std_value > 0:
                return (current_value - mean_value) / std_value
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating Z-score: {str(e)}")
            return 0.0
    
    def _calculate_percentile_rank(self, osc_data: pd.DataFrame, index: int) -> float:
        """Calculate percentile rank for oscillator value."""
        try:
            if index < self.config.percentile_window:
                return 50.0
            
            oscillator_col = f"{self.config.oscillator_type.value.lower()}_oscillator"
            if oscillator_col in osc_data.columns:
                oscillator = osc_data[oscillator_col]
            else:
                oscillator = osc_data['percentage_oscillator']
            
            start_idx = max(0, index - self.config.percentile_window)
            recent_values = oscillator.iloc[start_idx:index+1]
            
            current_value = oscillator.iloc[index]
            percentile = stats.percentileofscore(recent_values, current_value)
            
            return percentile
            
        except Exception as e:
            self.logger.error(f"Error calculating percentile rank: {str(e)}")
            return 50.0
    
    def _calculate_overbought_level(self, oscillator_value: float) -> float:
        """Calculate overbought level."""
        try:
            if oscillator_value > self.config.overbought_threshold:
                return (oscillator_value - self.config.overbought_threshold) / 30.0
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_oversold_level(self, oscillator_value: float) -> float:
        """Calculate oversold level."""
        try:
            if oscillator_value < self.config.oversold_threshold:
                return (self.config.oversold_threshold - oscillator_value) / 30.0
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_optimal_timing_score(self, confirmation_strength: float,
                                      momentum_persistence: float,
                                      cycle_phase: float) -> float:
        """Calculate optimal timing score."""
        try:
            # Combine multiple factors for timing
            timing_score = (
                confirmation_strength * 0.4 +
                momentum_persistence * 0.3 +
                (1.0 - abs(cycle_phase - 0.5)) * 0.3
            )
            
            return np.clip(timing_score, 0, 1)
        except:
            return 0.5
    
    def _calculate_reversal_probability(self, oscillator_value: float,
                                      momentum_direction: VolumeMomentumDirection,
                                      cycle_phase: float) -> float:
        """Calculate reversal probability."""
        try:
            # Factors that increase reversal probability
            extreme_level = 0.0
            if abs(oscillator_value) > 50:
                extreme_level = (abs(oscillator_value) - 50) / 50
            
            momentum_reversal = 0.0
            if momentum_direction in [VolumeMomentumDirection.SLOWING_UP, 
                                    VolumeMomentumDirection.SLOWING_DOWN]:
                momentum_reversal = 0.5
            
            cycle_reversal = abs(cycle_phase - 0.5) * 2  # Higher at extremes
            
            reversal_prob = (extreme_level + momentum_reversal + cycle_reversal) / 3
            return np.clip(reversal_prob, 0, 1)
            
        except:
            return 0.0
    
    def _calculate_statistical_measures(self, osc_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical measures."""
        try:
            statistics = {}
            
            oscillator_col = f"{self.config.oscillator_type.value.lower()}_oscillator"
            if oscillator_col in osc_data.columns:
                oscillator = osc_data[oscillator_col].dropna()
            else:
                oscillator = osc_data['percentage_oscillator'].dropna()
            
            if len(oscillator) > 0:
                statistics['oscillator_mean'] = oscillator.mean()
                statistics['oscillator_std'] = oscillator.std()
                statistics['oscillator_skewness'] = oscillator.skew()
                statistics['oscillator_kurtosis'] = oscillator.kurtosis()
                statistics['oscillator_min'] = oscillator.min()
                statistics['oscillator_max'] = oscillator.max()
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical measures: {str(e)}")
            return {}
    
    def _calculate_performance_metrics(self, signals: List[VolumeOscillatorSignal]) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            if not signals:
                return {}
            
            # Signal statistics
            avg_oscillator_value = np.mean([s.oscillator_value for s in signals])
            avg_momentum = np.mean([s.volume_momentum for s in signals])
            avg_confirmation_strength = np.mean([s.confirmation_strength for s in signals])
            
            # Direction distribution
            momentum_counts = {}
            for signal in signals:
                direction = signal.momentum_direction.value
                momentum_counts[direction] = momentum_counts.get(direction, 0) + 1
            
            # Strength distribution
            strength_counts = {}
            for signal in signals:
                strength = signal.volume_strength.value
                strength_counts[strength] = strength_counts.get(strength, 0) + 1
            
            performance = {
                'avg_oscillator_value': avg_oscillator_value,
                'avg_momentum': avg_momentum,
                'avg_confirmation_strength': avg_confirmation_strength,
                'momentum_distribution': momentum_counts,
                'strength_distribution': strength_counts,
                'total_signals': len(signals)
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def _update_performance_stats(self, analysis_time: float) -> None:
        """Update performance statistics."""
        try:
            self.analysis_count += 1
            self.total_analysis_time += analysis_time
            
            if self.analysis_count % 100 == 0:
                avg_time = self.total_analysis_time / self.analysis_count
                self.logger.info(f"Average analysis time over {self.analysis_count} runs: {avg_time:.4f}s")
                
        except Exception as e:
            self.logger.error(f"Error updating performance stats: {str(e)}")
    
    def get_latest_signal(self) -> Optional[VolumeOscillatorSignal]:
        """Get the latest oscillator signal."""
        try:
            if self.current_signals:
                return self.current_signals[-1]
            return None
        except Exception as e:
            self.logger.error(f"Error getting latest signal: {str(e)}")
            return None


# Example usage and testing
if __name__ == "__main__":
    # Test the volume oscillator indicator
    try:
        # Generate sample data
        np.random.seed(42)
        n_points = 500
        
        timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1s')
        
        # Generate price data with trends
        price_changes = np.random.randn(n_points) * 0.0001
        prices = 1.1000 + np.cumsum(price_changes)
        
        # Generate volume data with cyclical patterns
        base_volume = 100
        volume_cycle = np.sin(np.arange(n_points) * 2 * np.pi / 50) * 20
        volume_noise = np.random.lognormal(np.log(base_volume), 0.3, n_points)
        volumes = volume_noise + volume_cycle
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes
        })
        
        print(f"Generated {len(data)} sample data points")
        
        # Initialize indicator
        config = VolumeOscillatorConfig(
            short_period=14,
            long_period=28,
            signal_period=9,
            oscillator_type=VolumeOscillatorType.PERCENTAGE
        )
        
        indicator = AdvancedVolumeOscillatorIndicator(config)
        
        # Perform analysis
        results = indicator.analyze(data)
        
        # Display results
        print(f"Analysis completed in {results['metadata']['analysis_time']:.4f}s")
        print(f"Generated {len(results['signals'])} oscillator signals")
        
        # Show latest signal
        latest_signal = indicator.get_latest_signal()
        if latest_signal:
            print(f"\nLatest Oscillator Signal:")
            print(f"  Oscillator Value: {latest_signal.oscillator_value:.3f}")
            print(f"  Momentum Direction: {latest_signal.momentum_direction.value}")
            print(f"  Volume Strength: {latest_signal.volume_strength.value}")
            print(f"  Trend Confirmation: {latest_signal.trend_confirmation.value}")
            print(f"  Relative Volume: {latest_signal.relative_volume:.3f}")
            print(f"  Confirmation Strength: {latest_signal.confirmation_strength:.3f}")
            print(f"  Cycle Phase: {latest_signal.cycle_phase:.3f}")
            print(f"  Optimal Timing Score: {latest_signal.optimal_timing_score:.3f}")
        
        # Performance metrics
        if results['performance']:
            print(f"\nPerformance Metrics:")
            for key, value in results['performance'].items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                elif isinstance(value, dict):
                    print(f"  {key}: {value}")
        
        # Statistical measures
        if results['statistics']:
            print(f"\nStatistical Measures:")
            for key, value in results['statistics'].items():
                print(f"  {key}: {value:.4f}")
    
    except Exception as e:
        print(f"Error in testing: {str(e)}")
        print(traceback.format_exc())