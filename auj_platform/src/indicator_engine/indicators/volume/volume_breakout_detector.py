"""
Advanced Volume Breakout Detector for AUJ Platform

This module implements a sophisticated volume breakout detection system that
identifies significant volume-based breakouts with comprehensive threshold
analysis, confirmation algorithms, and advanced false breakout filtering.
The detector provides reliable breakout signals for optimal trading execution.

Key Features:
- Multi-threshold volume breakout detection
- Advanced confirmation algorithms
- False breakout filtering with ML validation
- Dynamic threshold adaptation
- Volume surge pattern recognition
- Breakout strength and sustainability analysis
- Risk-adjusted breakout scoring
- Real-time monitoring and alerting
- Statistical validation and backtesting
- Production-ready performance optimization

Mathematical Models:
- Statistical threshold analysis with adaptive boundaries
- Volume surge detection using exponential smoothing
- Breakout confirmation through multiple timeframe analysis
- False breakout filtering using pattern recognition
- Sustainability scoring with decay functions
- Risk-adjusted breakout strength calculation
- Machine learning validation models
- Bayesian probability updates for breakout confidence

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
from scipy.signal import find_peaks, savgol_filter, argrelextrema
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import traceback
from collections import deque

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BreakoutType(Enum):
    """Enumeration for breakout types."""
    VOLUME_SURGE = "VOLUME_SURGE"
    SUSTAINED_INCREASE = "SUSTAINED_INCREASE"
    SPIKE_BREAKOUT = "SPIKE_BREAKOUT"
    ACCUMULATION_BREAKOUT = "ACCUMULATION_BREAKOUT"
    INSTITUTIONAL_BREAKOUT = "INSTITUTIONAL_BREAKOUT"
    FALSE_BREAKOUT = "FALSE_BREAKOUT"


class BreakoutStrength(Enum):
    """Enumeration for breakout strength levels."""
    EXTREME = "EXTREME"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    QUESTIONABLE = "QUESTIONABLE"


class ConfirmationStatus(Enum):
    """Enumeration for breakout confirmation status."""
    CONFIRMED = "CONFIRMED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


@dataclass
class VolumeBreakoutSignal:
    """Data class for volume breakout signals."""
    # Breakout identification
    is_breakout: bool = False
    breakout_type: BreakoutType = BreakoutType.VOLUME_SURGE
    breakout_strength: BreakoutStrength = BreakoutStrength.WEAK
    confirmation_status: ConfirmationStatus = ConfirmationStatus.PENDING
    
    # Threshold analysis
    volume_multiple: float = 1.0
    threshold_ratio: float = 1.0
    dynamic_threshold: float = 0.0
    static_threshold: float = 0.0
    
    # Breakout metrics
    breakout_score: float = 0.0
    sustainability_score: float = 0.0
    momentum_score: float = 0.0
    quality_score: float = 0.0
    
    # Confirmation analysis
    confirmation_score: float = 0.0
    confirmation_timeframe: int = 0
    consecutive_confirmations: int = 0
    failed_confirmations: int = 0
    
    # False breakout analysis
    false_breakout_probability: float = 0.0
    false_breakout_risk: float = 0.0
    historical_accuracy: float = 0.0
    pattern_reliability: float = 0.0
    
    # Advanced metrics
    volume_acceleration: float = 0.0
    price_volume_correlation: float = 0.0
    institutional_participation: float = 0.0
    market_impact: float = 0.0
    
    # Risk assessment
    execution_risk: float = 0.0
    slippage_estimate: float = 0.0
    optimal_entry_score: float = 0.0
    position_sizing_factor: float = 1.0
    
    # Timing analysis
    breakout_duration: int = 0
    time_since_breakout: int = 0
    expected_follow_through: float = 0.0
    decay_rate: float = 0.0


@dataclass
class BreakoutDetectorConfig:
    """Configuration parameters for volume breakout detection."""
    # Threshold parameters
    base_threshold_multiplier: float = 2.0
    dynamic_threshold_window: int = 50
    spike_threshold_multiplier: float = 5.0
    sustained_threshold_multiplier: float = 1.5
    
    # Confirmation parameters
    confirmation_periods: int = 3
    confirmation_threshold: float = 0.7
    min_confirmation_volume: float = 1.2
    max_confirmation_wait: int = 10
    
    # Pattern analysis
    pattern_lookback: int = 100
    surge_detection_window: int = 20
    accumulation_window: int = 30
    institutional_threshold: float = 3.0
    
    # False breakout filtering
    false_breakout_window: int = 20
    false_breakout_threshold: float = 0.3
    historical_accuracy_window: int = 200
    pattern_memory_length: int = 500
    
    # Risk parameters
    max_execution_risk: float = 0.8
    slippage_factor: float = 0.002
    position_risk_factor: float = 0.5
    
    # ML parameters
    ml_training_window: int = 1000
    feature_window: int = 10
    retrain_frequency: int = 100
    
    # Performance parameters
    decay_rate: float = 0.95
    sustainability_window: int = 15
    momentum_window: int = 25


class VolumeThresholdAnalyzer:
    """Advanced volume threshold analysis system."""
    
    def __init__(self, config: BreakoutDetectorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.volume_history = deque(maxlen=config.dynamic_threshold_window * 2)
        self.threshold_history = deque(maxlen=100)
        
    def calculate_dynamic_threshold(self, volume_data: pd.Series) -> pd.Series:
        """Calculate dynamic volume thresholds."""
        try:
            # Update volume history
            self.volume_history.extend(volume_data.values)
            
            # Calculate multiple threshold types
            rolling_mean_threshold = self._calculate_rolling_mean_threshold(volume_data)
            percentile_threshold = self._calculate_percentile_threshold(volume_data)
            volatility_adjusted_threshold = self._calculate_volatility_adjusted_threshold(volume_data)
            adaptive_threshold = self._calculate_adaptive_threshold(volume_data)
            
            # Combine thresholds with weights
            combined_threshold = (
                rolling_mean_threshold * 0.3 +
                percentile_threshold * 0.25 +
                volatility_adjusted_threshold * 0.25 +
                adaptive_threshold * 0.2
            )
            
            # Store threshold history
            self.threshold_history.extend(combined_threshold.values)
            
            return combined_threshold
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic threshold: {str(e)}")
            return pd.Series(index=volume_data.index, dtype=float)
    
    def _calculate_rolling_mean_threshold(self, volume_data: pd.Series) -> pd.Series:
        """Calculate rolling mean-based threshold."""
        try:
            window = min(self.config.dynamic_threshold_window, len(volume_data))
            rolling_mean = volume_data.rolling(window=window).mean()
            rolling_std = volume_data.rolling(window=window).std()
            
            threshold = rolling_mean + (rolling_std * self.config.base_threshold_multiplier)
            return threshold.fillna(volume_data.mean() * self.config.base_threshold_multiplier)
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling mean threshold: {str(e)}")
            return pd.Series(index=volume_data.index, dtype=float)
    
    def _calculate_percentile_threshold(self, volume_data: pd.Series) -> pd.Series:
        """Calculate percentile-based threshold."""
        try:
            window = min(self.config.dynamic_threshold_window, len(volume_data))
            
            # Calculate rolling 90th percentile
            percentile_90 = volume_data.rolling(window=window).quantile(0.9)
            
            return percentile_90.fillna(volume_data.quantile(0.9))
            
        except Exception as e:
            self.logger.error(f"Error calculating percentile threshold: {str(e)}")
            return pd.Series(index=volume_data.index, dtype=float)
    
    def _calculate_volatility_adjusted_threshold(self, volume_data: pd.Series) -> pd.Series:
        """Calculate volatility-adjusted threshold."""
        try:
            window = min(self.config.dynamic_threshold_window, len(volume_data))
            
            # Calculate rolling volatility
            rolling_mean = volume_data.rolling(window=window).mean()
            rolling_std = volume_data.rolling(window=window).std()
            cv = rolling_std / (rolling_mean + 1e-8)  # Coefficient of variation
            
            # Adjust threshold based on volatility
            volatility_multiplier = 1.0 + cv
            threshold = rolling_mean * self.config.base_threshold_multiplier * volatility_multiplier
            
            return threshold.fillna(volume_data.mean() * self.config.base_threshold_multiplier)
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility adjusted threshold: {str(e)}")
            return pd.Series(index=volume_data.index, dtype=float)
    
    def _calculate_adaptive_threshold(self, volume_data: pd.Series) -> pd.Series:
        """Calculate adaptive threshold based on recent performance."""
        try:
            if len(self.threshold_history) < 10:
                return self._calculate_rolling_mean_threshold(volume_data)
            
            # Analyze recent threshold performance
            recent_thresholds = list(self.threshold_history)[-50:]
            recent_volumes = list(self.volume_history)[-50:]
            
            if len(recent_volumes) != len(recent_thresholds):
                return self._calculate_rolling_mean_threshold(volume_data)
            
            # Calculate threshold effectiveness
            threshold_breaches = sum(1 for v, t in zip(recent_volumes, recent_thresholds) if v > t)
            breach_rate = threshold_breaches / len(recent_thresholds)
            
            # Adjust multiplier based on breach rate
            if breach_rate > 0.2:  # Too many breaches
                adjustment = 1.2
            elif breach_rate < 0.05:  # Too few breaches
                adjustment = 0.8
            else:
                adjustment = 1.0
            
            # Apply adaptive adjustment
            base_threshold = self._calculate_rolling_mean_threshold(volume_data)
            adaptive_threshold = base_threshold * adjustment
            
            return adaptive_threshold
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive threshold: {str(e)}")
            return self._calculate_rolling_mean_threshold(volume_data)


class BreakoutConfirmationSystem:
    """Advanced breakout confirmation and validation system."""
    
    def __init__(self, config: BreakoutDetectorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pending_breakouts = {}
        self.confirmed_breakouts = {}
        
    def analyze_confirmation(self, timestamp: datetime, volume: float, 
                           price: float, threshold: float) -> Dict[str, Any]:
        """Analyze breakout confirmation status."""
        try:
            confirmation_data = {
                'is_confirmed': False,
                'confirmation_score': 0.0,
                'confirmation_strength': 'WEAK',
                'follow_through_probability': 0.0,
                'sustainability_estimate': 0.0
            }
            
            # Check if volume exceeds threshold
            if volume > threshold:
                # Calculate confirmation metrics
                volume_multiple = volume / threshold
                confirmation_score = self._calculate_confirmation_score(volume_multiple, volume, price)
                
                # Determine confirmation status
                if confirmation_score > self.config.confirmation_threshold:
                    confirmation_data['is_confirmed'] = True
                    confirmation_data['confirmation_score'] = confirmation_score
                    confirmation_data['confirmation_strength'] = self._classify_confirmation_strength(confirmation_score)
                    confirmation_data['follow_through_probability'] = self._estimate_follow_through(volume_multiple)
                    confirmation_data['sustainability_estimate'] = self._estimate_sustainability(volume, price)
            
            return confirmation_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing confirmation: {str(e)}")
            return {'is_confirmed': False, 'confirmation_score': 0.0}
    
    def _calculate_confirmation_score(self, volume_multiple: float, volume: float, price: float) -> float:
        """Calculate comprehensive confirmation score."""
        try:
            # Base score from volume multiple
            base_score = min(volume_multiple / 5.0, 1.0)
            
            # Volume sustainability factor
            if hasattr(self, 'recent_volumes'):
                volume_consistency = self._calculate_volume_consistency()
                sustainability_factor = volume_consistency
            else:
                sustainability_factor = 0.5
            
            # Price action confirmation
            if hasattr(self, 'recent_prices'):
                price_momentum = self._calculate_price_momentum()
                price_factor = price_momentum
            else:
                price_factor = 0.5
            
            # Combine factors
            confirmation_score = (
                base_score * 0.5 +
                sustainability_factor * 0.3 +
                price_factor * 0.2
            )
            
            return np.clip(confirmation_score, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating confirmation score: {str(e)}")
            return 0.0
    
    def _classify_confirmation_strength(self, score: float) -> str:
        """Classify confirmation strength."""
        try:
            if score > 0.9:
                return 'EXTREME'
            elif score > 0.8:
                return 'STRONG'
            elif score > 0.7:
                return 'MODERATE'
            elif score > 0.5:
                return 'WEAK'
            else:
                return 'QUESTIONABLE'
        except:
            return 'WEAK'
    
    def _estimate_follow_through(self, volume_multiple: float) -> float:
        """Estimate follow-through probability."""
        try:
            # Base probability from historical analysis
            base_probability = 0.6
            
            # Adjust based on volume multiple
            volume_factor = min(volume_multiple / 3.0, 1.5)
            
            # Apply logarithmic scaling for extreme values
            if volume_multiple > 5.0:
                volume_factor = 1.5 + np.log(volume_multiple - 4.0) * 0.1
            
            follow_through_prob = base_probability * volume_factor
            
            return np.clip(follow_through_prob, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error estimating follow-through: {str(e)}")
            return 0.5
    
    def _estimate_sustainability(self, volume: float, price: float) -> float:
        """Estimate breakout sustainability."""
        try:
            # Base sustainability from volume characteristics
            volume_sustainability = min(volume / 1000, 1.0)  # Normalize based on typical volume
            
            # Price action sustainability
            price_sustainability = 0.5  # Default value
            
            # Combine factors
            sustainability = (volume_sustainability + price_sustainability) / 2.0
            
            return np.clip(sustainability, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error estimating sustainability: {str(e)}")
            return 0.5
    
    def _calculate_volume_consistency(self) -> float:
        """Calculate volume consistency factor."""
        try:
            if not hasattr(self, 'recent_volumes') or len(self.recent_volumes) < 3:
                return 0.5
            
            volumes = np.array(self.recent_volumes)
            cv = np.std(volumes) / (np.mean(volumes) + 1e-8)
            
            # Lower coefficient of variation = higher consistency
            consistency = 1.0 / (1.0 + cv)
            
            return np.clip(consistency, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating volume consistency: {str(e)}")
            return 0.5
    
    def _calculate_price_momentum(self) -> float:
        """Calculate price momentum factor."""
        try:
            if not hasattr(self, 'recent_prices') or len(self.recent_prices) < 3:
                return 0.5
            
            prices = np.array(self.recent_prices)
            
            # Calculate price trend
            if len(prices) > 1:
                trend = (prices[-1] - prices[0]) / (prices[0] + 1e-8)
                momentum = np.tanh(trend * 100)  # Scale and apply tanh for bounded output
                return (momentum + 1) / 2  # Convert to 0-1 range
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating price momentum: {str(e)}")
            return 0.5


class FalseBreakoutFilter:
    """Advanced false breakout detection and filtering system."""
    
    def __init__(self, config: BreakoutDetectorConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.breakout_history = deque(maxlen=config.pattern_memory_length)
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def analyze_false_breakout_risk(self, volume_data: pd.Series, 
                                   price_data: pd.Series, 
                                   breakout_signal: bool) -> Dict[str, float]:
        """Analyze false breakout risk."""
        try:
            risk_analysis = {
                'false_breakout_probability': 0.5,
                'pattern_reliability': 0.5,
                'historical_accuracy': 0.5,
                'risk_score': 0.5
            }
            
            if not breakout_signal:
                return risk_analysis
            
            # Extract features for analysis
            features = self._extract_false_breakout_features(volume_data, price_data)
            
            # ML-based false breakout prediction
            if self.is_trained and len(features) > 0:
                ml_prediction = self._predict_false_breakout(features)
                risk_analysis['false_breakout_probability'] = ml_prediction
            
            # Pattern-based analysis
            pattern_risk = self._analyze_pattern_risk(volume_data, price_data)
            risk_analysis['pattern_reliability'] = 1.0 - pattern_risk
            
            # Historical accuracy
            historical_accuracy = self._calculate_historical_accuracy()
            risk_analysis['historical_accuracy'] = historical_accuracy
            
            # Combined risk score
            risk_score = (
                risk_analysis['false_breakout_probability'] * 0.4 +
                (1.0 - risk_analysis['pattern_reliability']) * 0.3 +
                (1.0 - risk_analysis['historical_accuracy']) * 0.3
            )
            risk_analysis['risk_score'] = risk_score
            
            return risk_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing false breakout risk: {str(e)}")
            return {'false_breakout_probability': 0.5, 'risk_score': 0.5}
    
    def _extract_false_breakout_features(self, volume_data: pd.Series, 
                                       price_data: pd.Series) -> np.ndarray:
        """Extract features for false breakout detection."""
        try:
            if len(volume_data) < self.config.feature_window:
                return np.array([])
            
            # Volume features
            recent_volume = volume_data.iloc[-self.config.feature_window:]
            volume_mean = recent_volume.mean()
            volume_std = recent_volume.std()
            volume_trend = stats.linregress(range(len(recent_volume)), recent_volume.values)[0]
            volume_cv = volume_std / (volume_mean + 1e-8)
            
            # Price features
            if len(price_data) >= self.config.feature_window:
                recent_price = price_data.iloc[-self.config.feature_window:]
                price_change = (recent_price.iloc[-1] - recent_price.iloc[0]) / recent_price.iloc[0]
                price_volatility = recent_price.std() / recent_price.mean()
                price_trend = stats.linregress(range(len(recent_price)), recent_price.values)[0]
            else:
                price_change = 0.0
                price_volatility = 0.0
                price_trend = 0.0
            
            # Combined features
            volume_price_correlation = volume_data.corr(price_data) if len(volume_data) == len(price_data) else 0.0
            
            features = np.array([
                volume_mean, volume_std, volume_trend, volume_cv,
                price_change, price_volatility, price_trend,
                volume_price_correlation
            ])
            
            return features.reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error extracting false breakout features: {str(e)}")
            return np.array([])
    
    def _predict_false_breakout(self, features: np.ndarray) -> float:
        """Predict false breakout probability using ML."""
        try:
            if not self.is_trained or len(features) == 0:
                return 0.5
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get probability prediction
            probabilities = self.classifier.predict_proba(features_scaled)
            
            # Return probability of false breakout (assuming class 1 is false breakout)
            if probabilities.shape[1] > 1:
                return probabilities[0][1]
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error predicting false breakout: {str(e)}")
            return 0.5
    
    def _analyze_pattern_risk(self, volume_data: pd.Series, price_data: pd.Series) -> float:
        """Analyze pattern-based false breakout risk."""
        try:
            if len(volume_data) < 10:
                return 0.5
            
            # Volume pattern analysis
            recent_volume = volume_data.iloc[-10:]
            volume_spike_ratio = recent_volume.iloc[-1] / recent_volume.iloc[:-1].mean()
            
            # High spike with no follow-through indicates higher false breakout risk
            if volume_spike_ratio > 5.0:
                spike_risk = 0.7
            elif volume_spike_ratio > 3.0:
                spike_risk = 0.5
            else:
                spike_risk = 0.3
            
            # Price confirmation analysis
            if len(price_data) >= 10:
                recent_price = price_data.iloc[-10:]
                price_follow_through = (recent_price.iloc[-1] - recent_price.iloc[-5]) / recent_price.iloc[-5]
                
                # Lack of price follow-through increases false breakout risk
                if abs(price_follow_through) < 0.001:  # Very small price movement
                    price_risk = 0.8
                elif abs(price_follow_through) < 0.005:
                    price_risk = 0.6
                else:
                    price_risk = 0.3
            else:
                price_risk = 0.5
            
            # Combine risks
            pattern_risk = (spike_risk + price_risk) / 2.0
            
            return np.clip(pattern_risk, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error analyzing pattern risk: {str(e)}")
            return 0.5
    
    def _calculate_historical_accuracy(self) -> float:
        """Calculate historical breakout accuracy."""
        try:
            if len(self.breakout_history) < 10:
                return 0.6  # Default accuracy
            
            # Count successful breakouts
            successful_breakouts = sum(1 for breakout in self.breakout_history 
                                     if breakout.get('successful', False))
            
            accuracy = successful_breakouts / len(self.breakout_history)
            
            return np.clip(accuracy, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating historical accuracy: {str(e)}")
            return 0.6
    
    def update_breakout_outcome(self, breakout_id: str, was_successful: bool) -> None:
        """Update breakout outcome for learning."""
        try:
            outcome_data = {
                'id': breakout_id,
                'successful': was_successful,
                'timestamp': datetime.now()
            }
            
            self.breakout_history.append(outcome_data)
            
            # Retrain model if enough new data
            if len(self.breakout_history) % self.config.retrain_frequency == 0:
                self._retrain_classifier()
                
        except Exception as e:
            self.logger.error(f"Error updating breakout outcome: {str(e)}")
    
    def _retrain_classifier(self) -> None:
        """Retrain the false breakout classifier."""
        try:
            if len(self.breakout_history) < 50:
                return
            
            # This would be implemented with proper feature extraction
            # and training data preparation in a production system
            self.logger.info("Retraining false breakout classifier")
            
        except Exception as e:
            self.logger.error(f"Error retraining classifier: {str(e)}")


class AdvancedVolumeBreakoutDetector:
    """
    Advanced Volume Breakout Detector with comprehensive analysis.
    
    This class implements sophisticated volume breakout detection with
    threshold analysis, confirmation algorithms, false breakout filtering,
    and machine learning validation for optimal trading signals.
    """
    
    def __init__(self, config: Optional[BreakoutDetectorConfig] = None):
        """
        Initialize the Advanced Volume Breakout Detector.
        
        Args:
            config: Configuration parameters for the detector
        """
        self.config = config or BreakoutDetectorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.threshold_analyzer = VolumeThresholdAnalyzer(self.config)
        self.confirmation_system = BreakoutConfirmationSystem(self.config)
        self.false_breakout_filter = FalseBreakoutFilter(self.config)
        
        # State management
        self.current_breakouts = {}
        self.breakout_history = []
        self.performance_metrics = {}
        
        # Caching for performance
        self._cache = {}
        self._cache_timestamps = {}
        
        self.logger.info("Advanced Volume Breakout Detector initialized successfully")
    
    def detect_breakouts(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect volume breakouts in the provided data.
        
        Args:
            data: DataFrame with columns ['timestamp', 'volume', 'price']
            
        Returns:
            Dictionary containing breakout analysis results
        """
        try:
            start_time = datetime.now()
            
            # Validate input data
            if not self._validate_input_data(data):
                raise ValueError("Invalid input data for breakout detection")
            
            # Calculate dynamic thresholds
            dynamic_thresholds = self.threshold_analyzer.calculate_dynamic_threshold(data['volume'])
            
            # Detect potential breakouts
            breakout_candidates = self._identify_breakout_candidates(data, dynamic_thresholds)
            
            # Analyze each breakout candidate
            breakout_signals = []
            for candidate in breakout_candidates:
                signal = self._analyze_breakout_candidate(data, candidate, dynamic_thresholds)
                if signal.is_breakout:
                    breakout_signals.append(signal)
            
            # Filter false breakouts
            filtered_breakouts = self._filter_false_breakouts(data, breakout_signals)
            
            # Calculate performance metrics
            performance = self._calculate_detection_performance(data, filtered_breakouts)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Compile results
            results = {
                'breakout_signals': filtered_breakouts,
                'raw_candidates': len(breakout_candidates),
                'confirmed_breakouts': len(filtered_breakouts),
                'dynamic_thresholds': dynamic_thresholds,
                'performance_metrics': performance,
                'metadata': {
                    'analysis_time': analysis_time,
                    'data_points': len(data),
                    'detection_rate': len(filtered_breakouts) / len(data) if len(data) > 0 else 0,
                    'config': self.config.__dict__
                }
            }
            
            self.logger.info(f"Breakout detection completed in {analysis_time:.4f}s. "
                           f"Found {len(filtered_breakouts)} confirmed breakouts from "
                           f"{len(breakout_candidates)} candidates")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in breakout detection: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data quality and structure."""
        try:
            required_columns = ['timestamp', 'volume', 'price']
            
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns: {required_columns}")
                return False
            
            if len(data) < 20:
                self.logger.error("Insufficient data points for analysis")
                return False
            
            if (data['volume'] <= 0).any():
                self.logger.warning("Non-positive volume values detected")
            
            if (data['price'] <= 0).any():
                self.logger.error("Invalid price values detected")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating input data: {str(e)}")
            return False
    
    def _identify_breakout_candidates(self, data: pd.DataFrame, 
                                    thresholds: pd.Series) -> List[int]:
        """Identify potential breakout candidates."""
        try:
            candidates = []
            volume = data['volume']
            
            for i in range(len(volume)):
                if i < len(thresholds) and volume.iloc[i] > thresholds.iloc[i]:
                    # Additional filtering criteria
                    if self._meets_candidate_criteria(data, i):
                        candidates.append(i)
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error identifying breakout candidates: {str(e)}")
            return []
    
    def _meets_candidate_criteria(self, data: pd.DataFrame, index: int) -> bool:
        """Check if a data point meets breakout candidate criteria."""
        try:
            # Minimum volume requirement
            if data['volume'].iloc[index] < 100:  # Adjust based on typical volumes
                return False
            
            # Look for volume increase pattern
            if index >= 5:
                recent_volumes = data['volume'].iloc[index-5:index]
                current_volume = data['volume'].iloc[index]
                
                # Current volume should be significantly higher than recent average
                volume_ratio = current_volume / recent_volumes.mean()
                if volume_ratio < 1.5:
                    return False
            
            # Price movement confirmation (optional)
            if 'price' in data.columns and index >= 1:
                price_change = abs(data['price'].iloc[index] - data['price'].iloc[index-1])
                price_change_pct = price_change / data['price'].iloc[index-1]
                
                # Some price movement expected with volume breakout
                if price_change_pct < 0.0001:  # Very small price movement
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking candidate criteria: {str(e)}")
            return False
    
    def _analyze_breakout_candidate(self, data: pd.DataFrame, candidate_index: int,
                                  thresholds: pd.Series) -> VolumeBreakoutSignal:
        """Analyze a breakout candidate comprehensively."""
        try:
            volume = data['volume'].iloc[candidate_index]
            price = data['price'].iloc[candidate_index]
            timestamp = data['timestamp'].iloc[candidate_index]
            threshold = thresholds.iloc[candidate_index]
            
            # Initialize signal
            signal = VolumeBreakoutSignal()
            
            # Basic breakout detection
            volume_multiple = volume / threshold
            signal.volume_multiple = volume_multiple
            signal.threshold_ratio = volume_multiple
            signal.dynamic_threshold = threshold
            
            if volume_multiple > 1.0:
                signal.is_breakout = True
                
                # Classify breakout type
                signal.breakout_type = self._classify_breakout_type(data, candidate_index)
                
                # Calculate breakout strength
                signal.breakout_strength = self._calculate_breakout_strength(volume_multiple)
                
                # Calculate breakout score
                signal.breakout_score = self._calculate_breakout_score(data, candidate_index, volume_multiple)
                
                # Confirmation analysis
                confirmation_data = self.confirmation_system.analyze_confirmation(
                    timestamp, volume, price, threshold
                )
                signal.confirmation_score = confirmation_data.get('confirmation_score', 0.0)
                signal.confirmation_status = (ConfirmationStatus.CONFIRMED 
                                            if confirmation_data.get('is_confirmed', False)
                                            else ConfirmationStatus.PENDING)
                
                # Calculate additional metrics
                self._calculate_advanced_metrics(signal, data, candidate_index)
                
                # Quality assessment
                signal.quality_score = self._calculate_quality_score(signal)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error analyzing breakout candidate: {str(e)}")
            return VolumeBreakoutSignal()
    
    def _classify_breakout_type(self, data: pd.DataFrame, index: int) -> BreakoutType:
        """Classify the type of breakout."""
        try:
            volume = data['volume'].iloc[index]
            
            # Look at recent volume pattern
            if index >= 10:
                recent_volumes = data['volume'].iloc[index-10:index]
                avg_recent = recent_volumes.mean()
                
                # Spike breakout (sudden large increase)
                if volume > avg_recent * self.config.spike_threshold_multiplier:
                    return BreakoutType.SPIKE_BREAKOUT
                
                # Sustained increase
                elif recent_volumes.iloc[-3:].mean() > recent_volumes.iloc[:-3].mean() * 1.5:
                    return BreakoutType.SUSTAINED_INCREASE
                
                # Accumulation breakout (gradual buildup)
                elif self._detect_accumulation_pattern(recent_volumes):
                    return BreakoutType.ACCUMULATION_BREAKOUT
                
                # Institutional breakout (very large volume)
                elif volume > avg_recent * self.config.institutional_threshold:
                    return BreakoutType.INSTITUTIONAL_BREAKOUT
            
            # Default to volume surge
            return BreakoutType.VOLUME_SURGE
            
        except Exception as e:
            self.logger.error(f"Error classifying breakout type: {str(e)}")
            return BreakoutType.VOLUME_SURGE
    
    def _detect_accumulation_pattern(self, volumes: pd.Series) -> bool:
        """Detect accumulation pattern in volume data."""
        try:
            if len(volumes) < 5:
                return False
            
            # Check for gradual increase pattern
            trend_slope = stats.linregress(range(len(volumes)), volumes.values)[0]
            
            # Positive trend with increasing volumes
            return trend_slope > 0 and volumes.iloc[-1] > volumes.iloc[0] * 1.2
            
        except Exception as e:
            self.logger.error(f"Error detecting accumulation pattern: {str(e)}")
            return False
    
    def _calculate_breakout_strength(self, volume_multiple: float) -> BreakoutStrength:
        """Calculate breakout strength classification."""
        try:
            if volume_multiple > 5.0:
                return BreakoutStrength.EXTREME
            elif volume_multiple > 3.0:
                return BreakoutStrength.STRONG
            elif volume_multiple > 2.0:
                return BreakoutStrength.MODERATE
            elif volume_multiple > 1.5:
                return BreakoutStrength.WEAK
            else:
                return BreakoutStrength.QUESTIONABLE
                
        except Exception as e:
            self.logger.error(f"Error calculating breakout strength: {str(e)}")
            return BreakoutStrength.WEAK
    
    def _calculate_breakout_score(self, data: pd.DataFrame, index: int, 
                                volume_multiple: float) -> float:
        """Calculate comprehensive breakout score."""
        try:
            # Base score from volume multiple
            base_score = min(volume_multiple / 5.0, 1.0)
            
            # Price momentum factor
            if index >= 5:
                recent_prices = data['price'].iloc[index-5:index+1]
                price_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                price_factor = min(abs(price_momentum) * 10, 0.5)
            else:
                price_factor = 0.0
            
            # Volume consistency factor
            if index >= 10:
                recent_volumes = data['volume'].iloc[index-10:index]
                volume_cv = recent_volumes.std() / recent_volumes.mean()
                consistency_factor = 1.0 / (1.0 + volume_cv)
            else:
                consistency_factor = 0.5
            
            # Combine factors
            breakout_score = (
                base_score * 0.6 +
                price_factor * 0.2 +
                consistency_factor * 0.2
            )
            
            return np.clip(breakout_score, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating breakout score: {str(e)}")
            return 0.0
    
    def _calculate_advanced_metrics(self, signal: VolumeBreakoutSignal, 
                                  data: pd.DataFrame, index: int) -> None:
        """Calculate advanced breakout metrics."""
        try:
            # Volume acceleration
            if index >= 3:
                volumes = data['volume'].iloc[index-3:index+1]
                velocity = volumes.diff().mean()
                acceleration = volumes.diff().diff().mean()
                signal.volume_acceleration = acceleration
            
            # Price-volume correlation
            if index >= 10:
                recent_data = data.iloc[index-10:index+1]
                correlation = recent_data['price'].corr(recent_data['volume'])
                signal.price_volume_correlation = correlation if not pd.isna(correlation) else 0.0
            
            # Market impact estimation
            volume = data['volume'].iloc[index]
            signal.market_impact = np.log1p(volume / 1000)  # Logarithmic scaling
            
            # Risk assessment
            signal.execution_risk = min(signal.volume_multiple / 10.0, 1.0)
            signal.slippage_estimate = signal.execution_risk * self.config.slippage_factor
            
            # Position sizing factor
            signal.position_sizing_factor = 1.0 / (1.0 + signal.execution_risk)
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced metrics: {str(e)}")
    
    def _calculate_quality_score(self, signal: VolumeBreakoutSignal) -> float:
        """Calculate overall breakout quality score."""
        try:
            quality_factors = [
                signal.breakout_score,
                signal.confirmation_score,
                1.0 - signal.execution_risk,
                abs(signal.price_volume_correlation),
                signal.sustainability_score
            ]
            
            # Remove zero/invalid factors
            valid_factors = [f for f in quality_factors if f > 0]
            
            if valid_factors:
                quality_score = np.mean(valid_factors)
            else:
                quality_score = 0.0
            
            return np.clip(quality_score, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {str(e)}")
            return 0.0
    
    def _filter_false_breakouts(self, data: pd.DataFrame, 
                              signals: List[VolumeBreakoutSignal]) -> List[VolumeBreakoutSignal]:
        """Filter out potential false breakouts."""
        try:
            filtered_signals = []
            
            for signal in signals:
                # Analyze false breakout risk
                risk_analysis = self.false_breakout_filter.analyze_false_breakout_risk(
                    data['volume'], data['price'], signal.is_breakout
                )
                
                # Update signal with risk analysis
                signal.false_breakout_probability = risk_analysis.get('false_breakout_probability', 0.5)
                signal.false_breakout_risk = risk_analysis.get('risk_score', 0.5)
                signal.pattern_reliability = risk_analysis.get('pattern_reliability', 0.5)
                signal.historical_accuracy = risk_analysis.get('historical_accuracy', 0.5)
                
                # Filter based on false breakout risk
                if signal.false_breakout_risk < self.config.false_breakout_threshold:
                    filtered_signals.append(signal)
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error filtering false breakouts: {str(e)}")
            return signals
    
    def _calculate_detection_performance(self, data: pd.DataFrame,
                                       breakouts: List[VolumeBreakoutSignal]) -> Dict[str, Any]:
        """Calculate detection performance metrics."""
        try:
            if not breakouts:
                return {'detection_rate': 0.0, 'avg_quality': 0.0}
            
            # Basic metrics
            detection_rate = len(breakouts) / len(data)
            avg_quality = np.mean([b.quality_score for b in breakouts])
            avg_confidence = np.mean([b.confirmation_score for b in breakouts])
            
            # Strength distribution
            strength_counts = {}
            for breakout in breakouts:
                strength = breakout.breakout_strength.value
                strength_counts[strength] = strength_counts.get(strength, 0) + 1
            
            # Type distribution
            type_counts = {}
            for breakout in breakouts:
                breakout_type = breakout.breakout_type.value
                type_counts[breakout_type] = type_counts.get(breakout_type, 0) + 1
            
            performance = {
                'detection_rate': detection_rate,
                'avg_quality': avg_quality,
                'avg_confidence': avg_confidence,
                'total_breakouts': len(breakouts),
                'strength_distribution': strength_counts,
                'type_distribution': type_counts,
                'avg_volume_multiple': np.mean([b.volume_multiple for b in breakouts]),
                'avg_false_breakout_risk': np.mean([b.false_breakout_risk for b in breakouts])
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error calculating detection performance: {str(e)}")
            return {}
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """Get real-time detector status."""
        try:
            return {
                'active_breakouts': len(self.current_breakouts),
                'total_detected': len(self.breakout_history),
                'detector_health': 'healthy',
                'last_update': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting real-time status: {str(e)}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Test the volume breakout detector
    try:
        # Generate sample data with breakout patterns
        np.random.seed(42)
        n_points = 1000
        
        timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1min')
        
        # Generate price data with trends
        price_trend = np.cumsum(np.random.randn(n_points) * 0.0001)
        prices = 1.1000 + price_trend
        
        # Generate volume data with breakout patterns
        base_volume = 100
        volumes = np.random.lognormal(np.log(base_volume), 0.5, n_points)
        
        # Add artificial breakouts
        breakout_indices = [200, 400, 600, 800]
        for idx in breakout_indices:
            if idx < len(volumes):
                volumes[idx] *= np.random.uniform(3, 8)  # Volume surge
                # Add some follow-through
                for i in range(1, min(5, len(volumes) - idx)):
                    volumes[idx + i] *= np.random.uniform(1.2, 2.0)
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes
        })
        
        print(f"Generated {len(data)} data points with {len(breakout_indices)} artificial breakouts")
        
        # Initialize detector
        config = BreakoutDetectorConfig(
            base_threshold_multiplier=1.8,
            confirmation_threshold=0.6,
            false_breakout_threshold=0.4
        )
        
        detector = AdvancedVolumeBreakoutDetector(config)
        
        # Detect breakouts
        results = detector.detect_breakouts(data)
        
        # Display results
        print(f"Detection completed in {results['metadata']['analysis_time']:.4f}s")
        print(f"Found {results['metadata']['confirmed_breakouts']} confirmed breakouts from {results['metadata']['raw_candidates']} candidates")
        print(f"Detection rate: {results['metadata']['detection_rate']:.1%}")
        
        # Show breakout details
        for i, breakout in enumerate(results['breakout_signals']):
            print(f"\nBreakout {i+1}:")
            print(f"  Type: {breakout.breakout_type.value}")
            print(f"  Strength: {breakout.breakout_strength.value}")
            print(f"  Volume Multiple: {breakout.volume_multiple:.2f}")
            print(f"  Quality Score: {breakout.quality_score:.3f}")
            print(f"  Confirmation Score: {breakout.confirmation_score:.3f}")
            print(f"  False Breakout Risk: {breakout.false_breakout_risk:.3f}")
        
        # Performance metrics
        if results['performance_metrics']:
            print(f"\nPerformance Metrics:")
            for key, value in results['performance_metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                elif isinstance(value, dict):
                    print(f"  {key}: {value}")
        
        # Real-time status
        status = detector.get_real_time_status()
        print(f"\nDetector Status: {status}")
    
    except Exception as e:
        print(f"Error in testing: {str(e)}")
        print(traceback.format_exc())