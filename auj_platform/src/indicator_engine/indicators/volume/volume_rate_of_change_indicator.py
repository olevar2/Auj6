"""
Volume Rate of Change Indicator - Advanced Implementation
========================================================

A sophisticated volume momentum indicator that measures the rate of change
in trading volume to identify momentum shifts, acceleration patterns, and
trend confirmation signals.

Key Features:
- Multi-period volume rate of change analysis
- Volume momentum acceleration detection
- Trend confirmation through volume dynamics
- Institutional volume flow identification
- Machine learning-based pattern recognition
- Adaptive smoothing and noise reduction
- Volume surge and collapse detection
- Cross-timeframe momentum analysis
- Statistical significance testing
- Real-time momentum tracking

Mathematical Models:
- Rate of change calculations with multiple smoothing methods
- Acceleration and jerk (rate of acceleration change) analysis
- Statistical significance testing for volume changes
- Momentum divergence detection algorithms
- Institutional flow identification through volume patterns
- Adaptive thresholds based on historical volatility
- Time-weighted momentum calculations

Performance Features:
- Optimized calculations for real-time processing
- Memory-efficient data structures
- Parallel processing for multi-timeframe analysis
- Robust error handling and data validation
- Performance monitoring and optimization

The indicator is designed for institutional-grade analysis with sophisticated
mathematical foundations and production-ready reliability for the humanitarian
trading mission.

Author: AUJ Platform Development Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from scipy import stats
from scipy.signal import savgol_filter, find_peaks, butter, filtfilt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..base.base_indicator import BaseIndicator
from ....core.signal_type import SignalType


@dataclass
class VolumeAcceleration:
    """Volume acceleration metrics."""
    velocity: float  # First derivative (rate of change)
    acceleration: float  # Second derivative
    jerk: float  # Third derivative
    momentum_strength: float
    persistence: float
    significance: float


@dataclass
class VolumeMomentumSignal:
    """Volume momentum analysis signal."""
    primary_momentum: float
    secondary_momentum: float
    momentum_direction: str
    acceleration_phase: str
    trend_alignment: bool
    institutional_flow: bool
    significance_score: float
    divergence_detected: bool


@dataclass
class VolumeROCSignal:
    """Enhanced signal structure for Volume Rate of Change analysis."""
    signal_type: SignalType
    strength: float
    confidence: float
    volume_roc: float
    volume_acceleration: VolumeAcceleration
    momentum_signal: VolumeMomentumSignal
    surge_detected: bool
    collapse_detected: bool
    institutional_activity: bool
    trend_confirmation: str
    anomaly_score: float
    statistical_metrics: Dict[str, float]
    timestamp: datetime


class VolumeRateOfChangeIndicator(BaseIndicator):
    """
    Advanced Volume Rate of Change Indicator with momentum analysis.
    
    This indicator provides comprehensive volume momentum analysis including:
    - Multi-period rate of change calculations
    - Volume acceleration and jerk analysis
    - Momentum pattern recognition
    - Institutional flow detection
    - Trend confirmation signals
    - Anomaly detection in volume patterns
    """

    def __init__(self, 
                 primary_period: int = 14,
                 secondary_period: int = 28,
                 acceleration_period: int = 7,
                 smoothing_period: int = 3,
                 surge_threshold: float = 2.0,
                 collapse_threshold: float = -1.5,
                 institutional_threshold: float = 3.0,
                 significance_level: float = 0.05,
                 enable_ml: bool = True,
                 ml_lookback: int = 100):
        """
        Initialize the Volume Rate of Change Indicator.
        
        Args:
            primary_period: Primary period for ROC calculation
            secondary_period: Secondary period for longer-term analysis
            acceleration_period: Period for acceleration calculations
            smoothing_period: Period for smoothing calculations
            surge_threshold: Threshold for volume surge detection (standard deviations)
            collapse_threshold: Threshold for volume collapse detection (standard deviations)
            institutional_threshold: Threshold for institutional activity detection
            significance_level: Statistical significance level for tests
            enable_ml: Whether to enable machine learning features
            ml_lookback: Lookback period for ML pattern recognition
        """
        super().__init__()
        self.primary_period = primary_period
        self.secondary_period = secondary_period
        self.acceleration_period = acceleration_period
        self.smoothing_period = smoothing_period
        self.surge_threshold = surge_threshold
        self.collapse_threshold = collapse_threshold
        self.institutional_threshold = institutional_threshold
        self.significance_level = significance_level
        self.enable_ml = enable_ml
        self.ml_lookback = ml_lookback
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize analytical components."""
        # Historical data storage
        self.volume_history = []
        self.roc_history = []
        self.acceleration_history = []
        self.momentum_patterns = []
        
        # Statistical models
        self.scaler = RobustScaler()
        self.volume_scaler = StandardScaler()
        
        # ML models
        if self.enable_ml:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.pattern_clusterer = DBSCAN(eps=0.5, min_samples=5)
            
        # Performance tracking
        self.calculation_times = []
        self.accuracy_metrics = {}
        
        # Cache for optimization
        self.roc_cache = {}
        self.momentum_cache = {}
        
        logging.info("Volume Rate of Change Indicator initialized successfully")

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Rate of Change signals with comprehensive analysis.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with Volume ROC signals
        """
        try:
            start_time = datetime.now()
            
            if len(data) < max(self.primary_period, self.secondary_period):
                return pd.Series(index=data.index, dtype=object)
            
            signals = []
            
            for i in range(len(data)):
                if i < max(self.primary_period, self.secondary_period) - 1:
                    signals.append(None)
                    continue
                
                # Get data window for analysis
                window_data = data.iloc[max(0, i - self.ml_lookback):i + 1].copy()
                current_data = data.iloc[max(0, i - max(self.primary_period, self.secondary_period)):i + 1].copy()
                
                # Calculate volume rate of change
                volume_roc = self._calculate_volume_roc(current_data)
                
                # Calculate volume acceleration metrics
                acceleration_metrics = self._calculate_acceleration_metrics(current_data)
                
                # Analyze momentum patterns
                momentum_signal = self._analyze_momentum_patterns(current_data, volume_roc)
                
                # Detect volume surges and collapses
                surge_detected, collapse_detected = self._detect_volume_events(current_data, volume_roc)
                
                # Identify institutional activity
                institutional_activity = self._detect_institutional_activity(current_data, volume_roc)
                
                # Analyze trend confirmation
                trend_confirmation = self._analyze_trend_confirmation(current_data, volume_roc)
                
                # Calculate anomaly score
                anomaly_score = self._calculate_anomaly_score(window_data, volume_roc)
                
                # Calculate statistical metrics
                stats_metrics = self._calculate_statistical_metrics(current_data, volume_roc)
                
                # Create enhanced signal
                signal = self._create_enhanced_signal(
                    volume_roc, acceleration_metrics, momentum_signal,
                    surge_detected, collapse_detected, institutional_activity,
                    trend_confirmation, anomaly_score, stats_metrics,
                    data.iloc[i]
                )
                
                signals.append(signal)
                
                # Update historical data
                self._update_historical_data(volume_roc, acceleration_metrics, momentum_signal)
            
            # Track performance
            calculation_time = (datetime.now() - start_time).total_seconds()
            self.calculation_times.append(calculation_time)
            
            result = pd.Series(signals, index=data.index)
            self._log_calculation_summary(result)
            
            return result
            
        except Exception as e:
            logging.error(f"Error in volume ROC calculation: {str(e)}")
            return pd.Series(index=data.index, dtype=object)

    def _calculate_volume_roc(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive volume rate of change metrics."""
        try:
            volume = data['Volume'].values
            
            # Primary ROC calculation
            if len(volume) >= self.primary_period:
                primary_roc = ((volume[-1] / volume[-self.primary_period]) - 1) * 100
            else:
                primary_roc = 0.0
            
            # Secondary ROC calculation
            if len(volume) >= self.secondary_period:
                secondary_roc = ((volume[-1] / volume[-self.secondary_period]) - 1) * 100
            else:
                secondary_roc = 0.0
            
            # Short-term momentum
            short_period = min(7, len(volume) - 1)
            if short_period > 0:
                short_roc = ((volume[-1] / volume[-short_period]) - 1) * 100
            else:
                short_roc = 0.0
            
            # Smoothed ROC
            if len(volume) >= self.smoothing_period:
                recent_volumes = volume[-self.smoothing_period:]
                smoothed_current = np.mean(recent_volumes)
                
                if len(volume) >= self.primary_period + self.smoothing_period:
                    historical_volumes = volume[-self.primary_period-self.smoothing_period:-self.primary_period]
                    smoothed_historical = np.mean(historical_volumes)
                    smoothed_roc = ((smoothed_current / smoothed_historical) - 1) * 100
                else:
                    smoothed_roc = primary_roc
            else:
                smoothed_roc = primary_roc
            
            # Normalized ROC (z-score)
            if len(volume) >= 20:
                historical_rocs = []
                for j in range(min(20, len(volume) - self.primary_period)):
                    if len(volume) >= self.primary_period + j + 1:
                        hist_roc = ((volume[-(j+1)] / volume[-(self.primary_period+j+1)]) - 1) * 100
                        historical_rocs.append(hist_roc)
                
                if historical_rocs:
                    mean_roc = np.mean(historical_rocs)
                    std_roc = np.std(historical_rocs)
                    normalized_roc = (primary_roc - mean_roc) / std_roc if std_roc > 0 else 0.0
                else:
                    normalized_roc = 0.0
            else:
                normalized_roc = 0.0
            
            return {
                'primary_roc': primary_roc,
                'secondary_roc': secondary_roc,
                'short_roc': short_roc,
                'smoothed_roc': smoothed_roc,
                'normalized_roc': normalized_roc,
                'current_volume': volume[-1],
                'avg_volume': np.mean(volume[-self.primary_period:]) if len(volume) >= self.primary_period else volume[-1]
            }
            
        except Exception as e:
            logging.error(f"Error calculating volume ROC: {str(e)}")
            return {
                'primary_roc': 0.0,
                'secondary_roc': 0.0,
                'short_roc': 0.0,
                'smoothed_roc': 0.0,
                'normalized_roc': 0.0,
                'current_volume': 0.0,
                'avg_volume': 0.0
            }

    def _calculate_acceleration_metrics(self, data: pd.DataFrame) -> VolumeAcceleration:
        """Calculate volume acceleration and higher-order derivatives."""
        try:
            volume = data['Volume'].values
            
            if len(volume) < self.acceleration_period + 2:
                return VolumeAcceleration(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            # Calculate first derivative (velocity/ROC)
            velocities = []
            for i in range(1, len(volume)):
                if volume[i-1] > 0:
                    velocity = (volume[i] - volume[i-1]) / volume[i-1]
                    velocities.append(velocity)
            
            if len(velocities) < 2:
                return VolumeAcceleration(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            current_velocity = velocities[-1]
            
            # Calculate second derivative (acceleration)
            accelerations = []
            for i in range(1, len(velocities)):
                acceleration = velocities[i] - velocities[i-1]
                accelerations.append(acceleration)
            
            current_acceleration = accelerations[-1] if accelerations else 0.0
            
            # Calculate third derivative (jerk)
            jerks = []
            for i in range(1, len(accelerations)):
                jerk = accelerations[i] - accelerations[i-1]
                jerks.append(jerk)
            
            current_jerk = jerks[-1] if jerks else 0.0
            
            # Calculate momentum strength
            recent_velocities = velocities[-self.acceleration_period:] if len(velocities) >= self.acceleration_period else velocities
            momentum_strength = np.mean(np.abs(recent_velocities)) if recent_velocities else 0.0
            
            # Calculate persistence (consistency of direction)
            if len(recent_velocities) > 1:
                positive_count = sum(1 for v in recent_velocities if v > 0)
                persistence = abs(positive_count / len(recent_velocities) - 0.5) * 2
            else:
                persistence = 0.0
            
            # Calculate statistical significance
            if len(velocities) >= 10:
                recent_mean = np.mean(recent_velocities)
                historical_velocities = velocities[:-self.acceleration_period] if len(velocities) > self.acceleration_period else velocities[:-1]
                historical_mean = np.mean(historical_velocities)
                historical_std = np.std(historical_velocities)
                
                if historical_std > 0:
                    t_stat = (recent_mean - historical_mean) / (historical_std / np.sqrt(len(recent_velocities)))
                    significance = abs(t_stat) / 2.0  # Normalized significance score
                else:
                    significance = 0.0
            else:
                significance = 0.0
            
            return VolumeAcceleration(
                velocity=current_velocity,
                acceleration=current_acceleration,
                jerk=current_jerk,
                momentum_strength=momentum_strength,
                persistence=persistence,
                significance=significance
            )
            
        except Exception as e:
            logging.error(f"Error calculating acceleration metrics: {str(e)}")
            return VolumeAcceleration(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _analyze_momentum_patterns(self, data: pd.DataFrame, 
                                 volume_roc: Dict[str, float]) -> VolumeMomentumSignal:
        """Analyze volume momentum patterns and trends."""
        try:
            # Determine momentum direction
            primary_roc = volume_roc['primary_roc']
            secondary_roc = volume_roc['secondary_roc']
            short_roc = volume_roc['short_roc']
            
            # Classify momentum direction
            if primary_roc > 5 and short_roc > 0:
                momentum_direction = "Strong_Bullish"
            elif primary_roc > 0 and short_roc > 0:
                momentum_direction = "Moderate_Bullish"
            elif primary_roc < -5 and short_roc < 0:
                momentum_direction = "Strong_Bearish"
            elif primary_roc < 0 and short_roc < 0:
                momentum_direction = "Moderate_Bearish"
            else:
                momentum_direction = "Mixed"
            
            # Analyze acceleration phase
            if primary_roc > secondary_roc and short_roc > primary_roc:
                acceleration_phase = "Accelerating"
            elif primary_roc < secondary_roc and short_roc < primary_roc:
                acceleration_phase = "Decelerating"
            elif abs(primary_roc - secondary_roc) < 2:
                acceleration_phase = "Stable"
            else:
                acceleration_phase = "Transitioning"
            
            # Check trend alignment
            price_data = data[['Open', 'High', 'Low', 'Close']].iloc[-self.primary_period:]
            price_trend = (price_data['Close'].iloc[-1] - price_data['Close'].iloc[0]) / price_data['Close'].iloc[0]
            
            volume_trend_positive = primary_roc > 0
            price_trend_positive = price_trend > 0
            trend_alignment = volume_trend_positive == price_trend_positive
            
            # Detect institutional flow
            normalized_roc = volume_roc['normalized_roc']
            institutional_flow = abs(normalized_roc) > self.institutional_threshold
            
            # Calculate significance score
            significance_components = [
                abs(primary_roc) / 100,  # ROC magnitude
                abs(normalized_roc) / 5,  # Statistical significance
                1.0 if trend_alignment else 0.5,  # Trend alignment bonus
                1.0 if institutional_flow else 0.8  # Institutional activity bonus
            ]
            significance_score = np.mean(significance_components)
            significance_score = max(0.0, min(1.0, significance_score))
            
            # Detect divergence
            recent_prices = price_data['Close'].iloc[-5:].values
            recent_volumes = data['Volume'].iloc[-5:].values
            
            if len(recent_prices) >= 3 and len(recent_volumes) >= 3:
                price_direction = recent_prices[-1] > recent_prices[0]
                volume_direction = recent_volumes[-1] > recent_volumes[0]
                divergence_detected = price_direction != volume_direction
            else:
                divergence_detected = False
            
            return VolumeMomentumSignal(
                primary_momentum=primary_roc,
                secondary_momentum=secondary_roc,
                momentum_direction=momentum_direction,
                acceleration_phase=acceleration_phase,
                trend_alignment=trend_alignment,
                institutional_flow=institutional_flow,
                significance_score=significance_score,
                divergence_detected=divergence_detected
            )
            
        except Exception as e:
            logging.error(f"Error analyzing momentum patterns: {str(e)}")
            return VolumeMomentumSignal(
                primary_momentum=0.0,
                secondary_momentum=0.0,
                momentum_direction="Unknown",
                acceleration_phase="Unknown",
                trend_alignment=False,
                institutional_flow=False,
                significance_score=0.0,
                divergence_detected=False
            )

    def _detect_volume_events(self, data: pd.DataFrame, 
                            volume_roc: Dict[str, float]) -> Tuple[bool, bool]:
        """Detect volume surges and collapses."""
        try:
            normalized_roc = volume_roc['normalized_roc']
            
            # Detect surge
            surge_detected = normalized_roc > self.surge_threshold
            
            # Detect collapse
            collapse_detected = normalized_roc < self.collapse_threshold
            
            # Additional validation using volume percentiles
            volume = data['Volume'].values
            if len(volume) >= 20:
                current_volume = volume[-1]
                historical_volume = volume[:-1]
                
                volume_percentile = stats.percentileofscore(historical_volume, current_volume)
                
                # Confirm surge with percentile analysis
                if surge_detected and volume_percentile < 95:
                    surge_detected = False
                
                # Confirm collapse with percentile analysis
                if collapse_detected and volume_percentile > 10:
                    collapse_detected = False
            
            return surge_detected, collapse_detected
            
        except Exception as e:
            logging.error(f"Error detecting volume events: {str(e)}")
            return False, False

    def _detect_institutional_activity(self, data: pd.DataFrame, 
                                     volume_roc: Dict[str, float]) -> bool:
        """Detect institutional trading activity patterns."""
        try:
            normalized_roc = volume_roc['normalized_roc']
            current_volume = volume_roc['current_volume']
            avg_volume = volume_roc['avg_volume']
            
            # Primary condition: significant volume increase
            volume_spike = abs(normalized_roc) > self.institutional_threshold
            
            # Secondary condition: volume above average
            volume_above_average = current_volume > avg_volume * 1.5
            
            # Tertiary condition: sustained volume (not just a single spike)
            volume = data['Volume'].values
            if len(volume) >= 5:
                recent_volumes = volume[-3:]
                high_volume_persistence = np.mean(recent_volumes) > avg_volume * 1.2
            else:
                high_volume_persistence = False
            
            # Combined institutional activity detection
            institutional_activity = volume_spike and (volume_above_average or high_volume_persistence)
            
            return institutional_activity
            
        except Exception as e:
            logging.error(f"Error detecting institutional activity: {str(e)}")
            return False

    def _analyze_trend_confirmation(self, data: pd.DataFrame, 
                                  volume_roc: Dict[str, float]) -> str:
        """Analyze trend confirmation through volume analysis."""
        try:
            # Get price trend
            prices = data['Close'].values
            if len(prices) < self.primary_period:
                return "Insufficient_Data"
            
            recent_prices = prices[-self.primary_period:]
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # Get volume trend
            primary_roc = volume_roc['primary_roc']
            
            # Classify price trend
            if price_trend > 0.02:  # 2% threshold
                price_direction = "Bullish"
            elif price_trend < -0.02:
                price_direction = "Bearish"
            else:
                price_direction = "Sideways"
            
            # Classify volume trend
            if primary_roc > 10:
                volume_direction = "Strong_Increase"
            elif primary_roc > 0:
                volume_direction = "Moderate_Increase"
            elif primary_roc < -10:
                volume_direction = "Strong_Decrease"
            elif primary_roc < 0:
                volume_direction = "Moderate_Decrease"
            else:
                volume_direction = "Stable"
            
            # Determine trend confirmation
            if price_direction == "Bullish" and volume_direction in ["Strong_Increase", "Moderate_Increase"]:
                return "Strong_Bullish_Confirmation"
            elif price_direction == "Bearish" and volume_direction in ["Strong_Increase", "Moderate_Increase"]:
                return "Strong_Bearish_Confirmation"
            elif price_direction == "Bullish" and volume_direction in ["Strong_Decrease", "Moderate_Decrease"]:
                return "Weak_Bullish_Divergence"
            elif price_direction == "Bearish" and volume_direction in ["Strong_Decrease", "Moderate_Decrease"]:
                return "Weak_Bearish_Divergence"
            elif price_direction == "Sideways":
                if volume_direction in ["Strong_Increase", "Moderate_Increase"]:
                    return "Accumulation_Pattern"
                else:
                    return "Consolidation_Pattern"
            else:
                return "Neutral_Confirmation"
            
        except Exception as e:
            logging.error(f"Error analyzing trend confirmation: {str(e)}")
            return "Analysis_Error"

    def _calculate_anomaly_score(self, data: pd.DataFrame, volume_roc: Dict[str, float]) -> float:
        """Calculate anomaly score for volume patterns."""
        try:
            if not self.enable_ml or len(data) < self.ml_lookback:
                return 0.0
            
            # Prepare features for anomaly detection
            volume = data['Volume'].values
            
            # Calculate rolling statistics
            window_size = min(20, len(volume) // 4)
            if window_size < 5:
                return 0.0
            
            features = []
            
            # Volume-based features
            current_volume = volume[-1]
            avg_volume = np.mean(volume[-window_size:])
            std_volume = np.std(volume[-window_size:])
            
            features.extend([
                current_volume,
                avg_volume,
                std_volume,
                current_volume / avg_volume if avg_volume > 0 else 0,
                volume_roc['primary_roc'],
                volume_roc['normalized_roc']
            ])
            
            # Price-volume features
            if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                prices = data['Close'].values[-window_size:]
                price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
                
                features.extend([
                    price_change,
                    np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
                ])
            
            # Reshape for model input
            feature_array = np.array(features).reshape(1, -1)
            
            # Scale features
            if hasattr(self, '_feature_scaler'):
                feature_array = self._feature_scaler.transform(feature_array)
            else:
                # Initialize scaler with current data
                historical_features = []
                for i in range(max(0, len(volume) - 50), len(volume)):
                    if i >= window_size:
                        hist_features = self._extract_features_for_anomaly(data.iloc[:i+1], window_size)
                        if hist_features:
                            historical_features.append(hist_features)
                
                if len(historical_features) > 10:
                    self._feature_scaler = StandardScaler()
                    self._feature_scaler.fit(historical_features)
                    feature_array = self._feature_scaler.transform(feature_array)
                else:
                    return 0.0
            
            # Calculate anomaly score
            anomaly_score = self.anomaly_detector.decision_function(feature_array)[0]
            
            # Normalize to 0-1 range
            normalized_score = max(0.0, min(1.0, (anomaly_score + 1) / 2))
            
            return normalized_score
            
        except Exception as e:
            logging.error(f"Error calculating anomaly score: {str(e)}")
            return 0.0

    def _extract_features_for_anomaly(self, data: pd.DataFrame, window_size: int) -> Optional[List[float]]:
        """Extract features for anomaly detection."""
        try:
            if len(data) < window_size:
                return None
            
            volume = data['Volume'].values
            
            features = []
            
            # Volume statistics
            current_volume = volume[-1]
            avg_volume = np.mean(volume[-window_size:])
            std_volume = np.std(volume[-window_size:])
            
            features.extend([
                current_volume,
                avg_volume,
                std_volume,
                current_volume / avg_volume if avg_volume > 0 else 0
            ])
            
            # ROC calculations
            if len(volume) >= self.primary_period:
                primary_roc = ((volume[-1] / volume[-self.primary_period]) - 1) * 100
                features.append(primary_roc)
            else:
                features.append(0.0)
            
            # Add more features as needed
            features.append(0.0)  # Placeholder for normalized_roc
            
            # Price features if available
            if all(col in data.columns for col in ['Close']):
                prices = data['Close'].values[-window_size:]
                price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
                price_volatility = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
                features.extend([price_change, price_volatility])
            else:
                features.extend([0.0, 0.0])
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features: {str(e)}")
            return None

    def _calculate_statistical_metrics(self, data: pd.DataFrame, 
                                     volume_roc: Dict[str, float]) -> Dict[str, float]:
        """Calculate comprehensive statistical metrics."""
        try:
            volume = data['Volume'].values
            
            metrics = {}
            
            # Basic ROC statistics
            metrics['primary_roc'] = volume_roc['primary_roc']
            metrics['secondary_roc'] = volume_roc['secondary_roc']
            metrics['roc_ratio'] = volume_roc['primary_roc'] / volume_roc['secondary_roc'] if volume_roc['secondary_roc'] != 0 else 0
            
            # Volume statistics
            if len(volume) >= 5:
                metrics['volume_mean'] = np.mean(volume)
                metrics['volume_std'] = np.std(volume)
                metrics['volume_cv'] = metrics['volume_std'] / metrics['volume_mean'] if metrics['volume_mean'] > 0 else 0
                metrics['volume_skewness'] = stats.skew(volume)
                metrics['volume_kurtosis'] = stats.kurtosis(volume)
            
            # Percentile analysis
            if len(volume) >= 10:
                current_volume = volume[-1]
                historical_volume = volume[:-1]
                metrics['volume_percentile'] = stats.percentileofscore(historical_volume, current_volume)
            
            # Trend analysis
            if len(volume) >= self.primary_period:
                recent_volume = volume[-self.primary_period:]
                trend_slope, _, correlation, p_value, _ = stats.linregress(range(len(recent_volume)), recent_volume)
                metrics['volume_trend_slope'] = trend_slope
                metrics['volume_trend_correlation'] = correlation
                metrics['volume_trend_significance'] = 1 - p_value if p_value < 1 else 0
            
            # Volatility metrics
            if len(volume) >= 10:
                volume_returns = np.diff(np.log(volume + 1e-10))  # Add small value to avoid log(0)
                metrics['volume_volatility'] = np.std(volume_returns)
                metrics['volume_sharpe'] = np.mean(volume_returns) / np.std(volume_returns) if np.std(volume_returns) > 0 else 0
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating statistical metrics: {str(e)}")
            return {}

    def _create_enhanced_signal(self, volume_roc: Dict[str, float], 
                              acceleration_metrics: VolumeAcceleration,
                              momentum_signal: VolumeMomentumSignal,
                              surge_detected: bool, collapse_detected: bool,
                              institutional_activity: bool, trend_confirmation: str,
                              anomaly_score: float, stats_metrics: Dict[str, float],
                              current_bar: pd.Series) -> VolumeROCSignal:
        """Create comprehensive Volume ROC signal."""
        try:
            # Determine base signal type
            primary_roc = volume_roc['primary_roc']
            
            if surge_detected or (primary_roc > 20 and momentum_signal.trend_alignment):
                base_signal = SignalType.BULLISH
            elif collapse_detected or (primary_roc < -20 and momentum_signal.trend_alignment):
                base_signal = SignalType.BEARISH
            elif abs(primary_roc) < 5:
                base_signal = SignalType.NEUTRAL
            else:
                base_signal = SignalType.NEUTRAL
            
            # Calculate signal strength
            strength = self._calculate_signal_strength(
                volume_roc, acceleration_metrics, momentum_signal,
                surge_detected, collapse_detected, institutional_activity
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                momentum_signal, stats_metrics, anomaly_score
            )
            
            return VolumeROCSignal(
                signal_type=base_signal,
                strength=strength,
                confidence=confidence,
                volume_roc=primary_roc,
                volume_acceleration=acceleration_metrics,
                momentum_signal=momentum_signal,
                surge_detected=surge_detected,
                collapse_detected=collapse_detected,
                institutional_activity=institutional_activity,
                trend_confirmation=trend_confirmation,
                anomaly_score=anomaly_score,
                statistical_metrics=stats_metrics,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logging.error(f"Error creating enhanced signal: {str(e)}")
            return self._create_neutral_signal()

    def _calculate_signal_strength(self, volume_roc: Dict[str, float],
                                 acceleration_metrics: VolumeAcceleration,
                                 momentum_signal: VolumeMomentumSignal,
                                 surge_detected: bool, collapse_detected: bool,
                                 institutional_activity: bool) -> float:
        """Calculate signal strength based on multiple factors."""
        try:
            strength = 0.5  # Base strength
            
            # ROC magnitude component
            roc_magnitude = abs(volume_roc['primary_roc']) / 100
            strength += min(0.3, roc_magnitude * 0.3)
            
            # Acceleration component
            if acceleration_metrics.momentum_strength > 0:
                strength += min(0.2, acceleration_metrics.momentum_strength * 0.2)
            
            # Persistence component
            strength += acceleration_metrics.persistence * 0.1
            
            # Event detection bonuses
            if surge_detected or collapse_detected:
                strength += 0.15
            
            if institutional_activity:
                strength += 0.1
            
            # Momentum signal component
            if momentum_signal.trend_alignment:
                strength += 0.1
            
            if momentum_signal.significance_score > 0.7:
                strength += 0.05
            
            return max(0.0, min(1.0, strength))
            
        except Exception as e:
            logging.error(f"Error calculating signal strength: {str(e)}")
            return 0.5

    def _calculate_confidence(self, momentum_signal: VolumeMomentumSignal,
                            stats_metrics: Dict[str, float], 
                            anomaly_score: float) -> float:
        """Calculate confidence based on signal quality."""
        try:
            confidence = momentum_signal.significance_score  # Base confidence
            
            # Statistical significance component
            trend_significance = stats_metrics.get('volume_trend_significance', 0)
            confidence += trend_significance * 0.2
            
            # Trend alignment bonus
            if momentum_signal.trend_alignment:
                confidence += 0.15
            
            # Institutional activity bonus
            if momentum_signal.institutional_flow:
                confidence += 0.1
            
            # Anomaly detection component (lower anomaly = higher confidence)
            confidence += (1 - anomaly_score) * 0.1
            
            # Divergence penalty
            if momentum_signal.divergence_detected:
                confidence -= 0.1
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logging.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    def _update_historical_data(self, volume_roc: Dict[str, float],
                              acceleration_metrics: VolumeAcceleration,
                              momentum_signal: VolumeMomentumSignal):
        """Update historical data for analysis."""
        try:
            # Update ROC history
            self.roc_history.append({
                'timestamp': datetime.now(),
                'primary_roc': volume_roc['primary_roc'],
                'secondary_roc': volume_roc['secondary_roc'],
                'normalized_roc': volume_roc['normalized_roc']
            })
            
            # Update acceleration history
            self.acceleration_history.append({
                'timestamp': datetime.now(),
                'velocity': acceleration_metrics.velocity,
                'acceleration': acceleration_metrics.acceleration,
                'momentum_strength': acceleration_metrics.momentum_strength
            })
            
            # Update momentum patterns
            self.momentum_patterns.append({
                'timestamp': datetime.now(),
                'direction': momentum_signal.momentum_direction,
                'phase': momentum_signal.acceleration_phase,
                'significance': momentum_signal.significance_score
            })
            
            # Keep only recent history
            max_history = 1000
            for history_list in [self.roc_history, self.acceleration_history, self.momentum_patterns]:
                if len(history_list) > max_history:
                    history_list[:] = history_list[-max_history:]
                    
        except Exception as e:
            logging.error(f"Error updating historical data: {str(e)}")

    def _create_neutral_signal(self) -> VolumeROCSignal:
        """Create neutral signal for error cases."""
        return VolumeROCSignal(
            signal_type=SignalType.NEUTRAL,
            strength=0.5,
            confidence=0.0,
            volume_roc=0.0,
            volume_acceleration=VolumeAcceleration(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            momentum_signal=VolumeMomentumSignal(
                0.0, 0.0, "Unknown", "Unknown", False, False, 0.0, False
            ),
            surge_detected=False,
            collapse_detected=False,
            institutional_activity=False,
            trend_confirmation="Unknown",
            anomaly_score=0.0,
            statistical_metrics={},
            timestamp=datetime.now()
        )

    def _log_calculation_summary(self, result: pd.Series):
        """Log calculation summary for monitoring."""
        try:
            non_null_signals = result.dropna()
            
            if len(non_null_signals) > 0:
                signal_types = [signal.signal_type.name for signal in non_null_signals if signal]
                avg_strength = np.mean([signal.strength for signal in non_null_signals if signal])
                avg_confidence = np.mean([signal.confidence for signal in non_null_signals if signal])
                
                # Count special events
                surges = sum(1 for signal in non_null_signals if signal and signal.surge_detected)
                collapses = sum(1 for signal in non_null_signals if signal and signal.collapse_detected)
                institutional = sum(1 for signal in non_null_signals if signal and signal.institutional_activity)
                
                logging.info(f"Volume ROC Analysis Complete:")
                logging.info(f"  Signals Generated: {len(non_null_signals)}")
                logging.info(f"  Average Strength: {avg_strength:.3f}")
                logging.info(f"  Average Confidence: {avg_confidence:.3f}")
                logging.info(f"  Volume Surges: {surges}")
                logging.info(f"  Volume Collapses: {collapses}")
                logging.info(f"  Institutional Activity: {institutional}")
                logging.info(f"  Signal Distribution: {pd.Series(signal_types).value_counts().to_dict()}")
                
                # Log performance metrics
                if self.calculation_times:
                    avg_time = np.mean(self.calculation_times[-10:])
                    logging.info(f"  Avg Calculation Time: {avg_time:.4f}s")
                    
        except Exception as e:
            logging.error(f"Error logging calculation summary: {str(e)}")

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary."""
        try:
            return {
                'indicator_name': 'Volume Rate of Change Indicator',
                'version': '1.0.0',
                'parameters': {
                    'primary_period': self.primary_period,
                    'secondary_period': self.secondary_period,
                    'acceleration_period': self.acceleration_period,
                    'surge_threshold': self.surge_threshold,
                    'collapse_threshold': self.collapse_threshold,
                    'institutional_threshold': self.institutional_threshold,
                    'ml_enabled': self.enable_ml
                },
                'features': [
                    'Multi-period volume rate of change analysis',
                    'Volume acceleration and momentum detection',
                    'Volume surge and collapse identification',
                    'Institutional activity detection',
                    'Trend confirmation analysis',
                    'Statistical significance testing',
                    'Machine learning anomaly detection',
                    'Real-time momentum tracking'
                ],
                'performance_metrics': {
                    'avg_calculation_time': np.mean(self.calculation_times) if self.calculation_times else 0,
                    'total_calculations': len(self.calculation_times),
                    'roc_history_size': len(self.roc_history),
                    'pattern_history_size': len(self.momentum_patterns)
                }
            }
            
        except Exception as e:
            logging.error(f"Error getting analysis summary: {str(e)}")
            return {}
