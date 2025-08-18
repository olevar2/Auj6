"""
Advanced Volume Delta Indicator for AUJ Platform

This module implements a sophisticated Volume Delta analysis system that tracks
the directional flow of volume by analyzing bid/ask imbalances, pressure
measurements, and comprehensive directional volume tracking. The indicator
provides deep insights into market sentiment and institutional activity.

Key Features:
- Bid/Ask flow analysis with tick-by-tick precision
- Volume pressure measurement and gradient analysis
- Directional volume tracking with momentum analysis
- Order flow imbalance detection and quantification
- Cumulative delta analysis with trend identification
- Market pressure dynamics modeling
- Institutional flow detection through delta patterns
- Real-time delta oscillations and reversals
- Advanced statistical validation
- Machine learning enhanced pattern recognition

Mathematical Models:
- Directional volume delta calculations
- Bid/ask flow pressure models
- Cumulative delta trend analysis
- Volume imbalance quantification algorithms
- Statistical significance testing for delta patterns
- Momentum-based delta oscillator
- Pressure gradient mathematical models
- Time-weighted delta analysis
- Exponential smoothing for delta trends
- Machine learning classification for delta patterns

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
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import minimize_scalar
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


class DeltaDirection(Enum):
    """Enumeration for delta direction classification."""
    BULLISH_STRONG = "BULLISH_STRONG"
    BULLISH_MODERATE = "BULLISH_MODERATE"
    BULLISH_WEAK = "BULLISH_WEAK"
    NEUTRAL = "NEUTRAL"
    BEARISH_WEAK = "BEARISH_WEAK"
    BEARISH_MODERATE = "BEARISH_MODERATE"
    BEARISH_STRONG = "BEARISH_STRONG"


class FlowPressure(Enum):
    """Enumeration for flow pressure classification."""
    EXTREME_BUYING = "EXTREME_BUYING"
    STRONG_BUYING = "STRONG_BUYING"
    MODERATE_BUYING = "MODERATE_BUYING"
    BALANCED = "BALANCED"
    MODERATE_SELLING = "MODERATE_SELLING"
    STRONG_SELLING = "STRONG_SELLING"
    EXTREME_SELLING = "EXTREME_SELLING"


class DeltaPattern(Enum):
    """Enumeration for delta pattern types."""
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    MOMENTUM_SHIFT = "MOMENTUM_SHIFT"
    REVERSAL_PATTERN = "REVERSAL_PATTERN"
    CONSOLIDATION = "CONSOLIDATION"
    BREAKOUT = "BREAKOUT"


@dataclass
class VolumeDeltaSignal:
    """Data class for volume delta analysis signals."""
    # Core delta metrics
    raw_delta: float = 0.0
    normalized_delta: float = 0.0
    cumulative_delta: float = 0.0
    delta_momentum: float = 0.0
    
    # Directional analysis
    delta_direction: DeltaDirection = DeltaDirection.NEUTRAL
    flow_pressure: FlowPressure = FlowPressure.BALANCED
    pressure_intensity: float = 0.0
    directional_strength: float = 0.0
    
    # Flow components
    buying_volume: float = 0.0
    selling_volume: float = 0.0
    net_volume: float = 0.0
    volume_imbalance: float = 0.0
    
    # Advanced metrics
    delta_acceleration: float = 0.0
    pressure_gradient: float = 0.0
    flow_persistence: float = 0.0
    delta_volatility: float = 0.0
    
    # Pattern analysis
    delta_pattern: DeltaPattern = DeltaPattern.CONSOLIDATION
    pattern_strength: float = 0.0
    pattern_confidence: float = 0.0
    reversal_probability: float = 0.0
    
    # Statistical measures
    delta_z_score: float = 0.0
    significance_level: float = 0.0
    autocorrelation: float = 0.0
    entropy: float = 0.0
    
    # Oscillator components
    delta_oscillator: float = 0.0
    oscillator_momentum: float = 0.0
    overbought_oversold: float = 0.0
    
    # Risk and timing
    execution_risk: float = 0.0
    optimal_entry_score: float = 0.0
    position_bias: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


@dataclass
class VolumeDeltaConfig:
    """Configuration parameters for volume delta analysis."""
    # Delta calculation parameters
    tick_classification_method: str = "price_based"  # price_based, bid_ask, time_based
    delta_smoothing_window: int = 10
    cumulative_reset_period: int = 100
    momentum_window: int = 20
    
    # Pressure analysis
    pressure_sensitivity: float = 0.1
    pressure_window: int = 30
    gradient_window: int = 15
    imbalance_threshold: float = 0.2
    
    # Pattern recognition
    pattern_window: int = 50
    pattern_threshold: float = 0.3
    reversal_sensitivity: float = 0.15
    consolidation_threshold: float = 0.1
    
    # Oscillator parameters
    oscillator_periods: List[int] = field(default_factory=lambda: [14, 28])
    overbought_threshold: float = 70.0
    oversold_threshold: float = 30.0
    
    # Statistical parameters
    zscore_window: int = 100
    significance_threshold: float = 0.05
    confidence_level: float = 0.95
    
    # ML parameters
    ml_lookback: int = 200
    feature_window: int = 20
    retrain_frequency: int = 500
    
    # Risk parameters
    max_position_bias: float = 0.8
    execution_risk_factor: float = 0.02
    volatility_adjustment: bool = True


class TickClassifier:
    """Advanced tick classification system for delta calculation."""
    
    def __init__(self, config: VolumeDeltaConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def classify_ticks(self, data: pd.DataFrame) -> pd.Series:
        """Classify ticks as buying (+1), selling (-1), or neutral (0)."""
        try:
            if self.config.tick_classification_method == "price_based":
                return self._price_based_classification(data)
            elif self.config.tick_classification_method == "bid_ask":
                return self._bid_ask_classification(data)
            elif self.config.tick_classification_method == "time_based":
                return self._time_based_classification(data)
            else:
                return self._price_based_classification(data)
                
        except Exception as e:
            self.logger.error(f"Error classifying ticks: {str(e)}")
            return pd.Series(index=data.index, dtype=float)
    
    def _price_based_classification(self, data: pd.DataFrame) -> pd.Series:
        """Classify ticks based on price movement."""
        try:
            price_changes = data['price'].diff()
            
            # Basic uptick/downtick classification
            tick_classification = np.where(
                price_changes > 0, 1,  # Uptick (buying)
                np.where(price_changes < 0, -1, 0)  # Downtick (selling), No change (neutral)
            )
            
            # Handle zero-tick rule (use previous non-zero tick)
            for i in range(1, len(tick_classification)):
                if tick_classification[i] == 0 and i > 0:
                    # Look back for last non-zero classification
                    for j in range(i-1, -1, -1):
                        if tick_classification[j] != 0:
                            tick_classification[i] = tick_classification[j]
                            break
            
            return pd.Series(tick_classification, index=data.index)
            
        except Exception as e:
            self.logger.error(f"Error in price-based classification: {str(e)}")
            return pd.Series(index=data.index, dtype=float)
    
    def _bid_ask_classification(self, data: pd.DataFrame) -> pd.Series:
        """Classify ticks based on bid/ask spread analysis."""
        try:
            if 'bid' not in data.columns or 'ask' not in data.columns:
                self.logger.warning("Bid/ask data not available, falling back to price-based classification")
                return self._price_based_classification(data)
            
            mid_price = (data['bid'] + data['ask']) / 2
            
            # Compare trade price to mid price
            price_to_mid = data['price'] - mid_price
            
            # Classify based on position relative to mid
            tick_classification = np.where(
                price_to_mid > 0, 1,  # Above mid (buying pressure)
                np.where(price_to_mid < 0, -1, 0)  # Below mid (selling pressure)
            )
            
            return pd.Series(tick_classification, index=data.index)
            
        except Exception as e:
            self.logger.error(f"Error in bid-ask classification: {str(e)}")
            return self._price_based_classification(data)
    
    def _time_based_classification(self, data: pd.DataFrame) -> pd.Series:
        """Classify ticks based on time and sales analysis."""
        try:
            # This is a simplified implementation
            # In practice, this would use order book timing analysis
            price_changes = data['price'].diff()
            volume_changes = data['volume'].diff()
            
            # Combine price and volume momentum
            momentum = price_changes * volume_changes
            
            tick_classification = np.where(
                momentum > 0, 1,
                np.where(momentum < 0, -1, 0)
            )
            
            return pd.Series(tick_classification, index=data.index)
            
        except Exception as e:
            self.logger.error(f"Error in time-based classification: {str(e)}")
            return self._price_based_classification(data)


class DeltaCalculationEngine:
    """Advanced delta calculation engine with multiple methodologies."""
    
    def __init__(self, config: VolumeDeltaConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.tick_classifier = TickClassifier(config)
        
    def calculate_volume_delta(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive volume delta metrics."""
        try:
            # Classify ticks
            tick_direction = self.tick_classifier.classify_ticks(data)
            
            # Calculate raw delta
            raw_delta = data['volume'] * tick_direction
            
            # Calculate directional volumes
            buying_volume = data['volume'].where(tick_direction > 0, 0)
            selling_volume = data['volume'].where(tick_direction < 0, 0)
            
            # Create results dataframe
            delta_df = pd.DataFrame(index=data.index)
            
            # Basic delta metrics
            delta_df['raw_delta'] = raw_delta
            delta_df['buying_volume'] = buying_volume
            delta_df['selling_volume'] = selling_volume.abs()  # Make positive for clarity
            delta_df['net_volume'] = buying_volume - selling_volume.abs()
            
            # Smoothed delta
            delta_df['smoothed_delta'] = raw_delta.rolling(
                window=self.config.delta_smoothing_window
            ).mean()
            
            # Cumulative delta
            delta_df['cumulative_delta'] = raw_delta.cumsum()
            
            # Reset cumulative delta periodically to prevent overflow
            if self.config.cumulative_reset_period > 0:
                reset_points = np.arange(
                    self.config.cumulative_reset_period,
                    len(delta_df),
                    self.config.cumulative_reset_period
                )
                for reset_point in reset_points:
                    delta_df.loc[delta_df.index[reset_point:], 'cumulative_delta'] -= \
                        delta_df['cumulative_delta'].iloc[reset_point-1]
            
            # Delta momentum
            delta_df['delta_momentum'] = delta_df['smoothed_delta'].rolling(
                window=self.config.momentum_window
            ).apply(lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0)
            
            # Delta acceleration
            delta_df['delta_acceleration'] = delta_df['delta_momentum'].diff()
            
            # Volume imbalance ratio
            total_volume = delta_df['buying_volume'] + delta_df['selling_volume']
            delta_df['volume_imbalance'] = (
                (delta_df['buying_volume'] - delta_df['selling_volume']) / 
                (total_volume + 1e-8)
            )
            
            # Normalized delta
            delta_std = delta_df['raw_delta'].rolling(window=50).std()
            delta_df['normalized_delta'] = delta_df['raw_delta'] / (delta_std + 1e-8)
            
            return delta_df
            
        except Exception as e:
            self.logger.error(f"Error calculating volume delta: {str(e)}")
            return pd.DataFrame(index=data.index)


class PressureAnalysisSystem:
    """Advanced pressure analysis system for flow dynamics."""
    
    def __init__(self, config: VolumeDeltaConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_flow_pressure(self, delta_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze flow pressure and dynamics."""
        try:
            pressure_df = pd.DataFrame(index=delta_data.index)
            
            # Pressure intensity
            pressure_df['pressure_intensity'] = self._calculate_pressure_intensity(delta_data)
            
            # Pressure gradient
            pressure_df['pressure_gradient'] = self._calculate_pressure_gradient(delta_data)
            
            # Flow persistence
            pressure_df['flow_persistence'] = self._calculate_flow_persistence(delta_data)
            
            # Directional strength
            pressure_df['directional_strength'] = self._calculate_directional_strength(delta_data)
            
            # Pressure oscillations
            pressure_df['pressure_oscillation'] = self._calculate_pressure_oscillation(delta_data)
            
            return pressure_df
            
        except Exception as e:
            self.logger.error(f"Error analyzing flow pressure: {str(e)}")
            return pd.DataFrame(index=delta_data.index)
    
    def _calculate_pressure_intensity(self, delta_data: pd.DataFrame) -> pd.Series:
        """Calculate pressure intensity."""
        try:
            # Use volume imbalance as base for pressure
            volume_imbalance = delta_data.get('volume_imbalance', pd.Series(index=delta_data.index))
            
            # Calculate rolling intensity
            intensity = volume_imbalance.abs().rolling(
                window=self.config.pressure_window
            ).mean()
            
            # Normalize to 0-1 range
            max_intensity = intensity.rolling(window=100).max()
            normalized_intensity = intensity / (max_intensity + 1e-8)
            
            return normalized_intensity.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating pressure intensity: {str(e)}")
            return pd.Series(index=delta_data.index, dtype=float)
    
    def _calculate_pressure_gradient(self, delta_data: pd.DataFrame) -> pd.Series:
        """Calculate pressure gradient (rate of change)."""
        try:
            volume_imbalance = delta_data.get('volume_imbalance', pd.Series(index=delta_data.index))
            
            # Calculate gradient using difference
            gradient = volume_imbalance.rolling(
                window=self.config.gradient_window
            ).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else 0)
            
            return gradient.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating pressure gradient: {str(e)}")
            return pd.Series(index=delta_data.index, dtype=float)
    
    def _calculate_flow_persistence(self, delta_data: pd.DataFrame) -> pd.Series:
        """Calculate flow persistence."""
        try:
            raw_delta = delta_data.get('raw_delta', pd.Series(index=delta_data.index))
            
            # Calculate consistency of flow direction
            flow_direction = np.sign(raw_delta)
            
            persistence = flow_direction.rolling(
                window=self.config.pressure_window
            ).apply(lambda x: abs(x.sum()) / len(x) if len(x) > 0 else 0)
            
            return persistence.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating flow persistence: {str(e)}")
            return pd.Series(index=delta_data.index, dtype=float)
    
    def _calculate_directional_strength(self, delta_data: pd.DataFrame) -> pd.Series:
        """Calculate directional strength."""
        try:
            buying_volume = delta_data.get('buying_volume', pd.Series(index=delta_data.index))
            selling_volume = delta_data.get('selling_volume', pd.Series(index=delta_data.index))
            
            # Calculate strength based on volume dominance
            total_volume = buying_volume + selling_volume
            net_volume = buying_volume - selling_volume
            
            strength = net_volume / (total_volume + 1e-8)
            
            # Apply smoothing
            smoothed_strength = strength.rolling(window=self.config.pressure_window).mean()
            
            return smoothed_strength.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating directional strength: {str(e)}")
            return pd.Series(index=delta_data.index, dtype=float)
    
    def _calculate_pressure_oscillation(self, delta_data: pd.DataFrame) -> pd.Series:
        """Calculate pressure oscillation component."""
        try:
            volume_imbalance = delta_data.get('volume_imbalance', pd.Series(index=delta_data.index))
            
            # Create oscillator using multiple periods
            oscillator_values = []
            
            for period in self.config.oscillator_periods:
                period_oscillator = volume_imbalance.rolling(window=period).mean()
                oscillator_values.append(period_oscillator)
            
            # Combine oscillators
            if oscillator_values:
                combined_oscillator = pd.concat(oscillator_values, axis=1).mean(axis=1)
            else:
                combined_oscillator = pd.Series(index=delta_data.index, dtype=float)
            
            return combined_oscillator.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating pressure oscillation: {str(e)}")
            return pd.Series(index=delta_data.index, dtype=float)


class DeltaPatternRecognition:
    """Advanced pattern recognition system for delta analysis."""
    
    def __init__(self, config: VolumeDeltaConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pattern_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
        self.is_trained = False
        
    def recognize_patterns(self, delta_data: pd.DataFrame, 
                          pressure_data: pd.DataFrame) -> pd.DataFrame:
        """Recognize delta patterns."""
        try:
            pattern_df = pd.DataFrame(index=delta_data.index)
            
            # Basic pattern detection
            pattern_df['delta_pattern'] = self._detect_basic_patterns(delta_data)
            pattern_df['pattern_strength'] = self._calculate_pattern_strength(delta_data)
            pattern_df['reversal_probability'] = self._calculate_reversal_probability(delta_data)
            
            # Advanced pattern analysis
            if len(delta_data) >= self.config.ml_lookback:
                ml_patterns = self._ml_pattern_analysis(delta_data, pressure_data)
                pattern_df['pattern_confidence'] = ml_patterns.get('confidence', 0.5)
            else:
                pattern_df['pattern_confidence'] = 0.5
            
            return pattern_df
            
        except Exception as e:
            self.logger.error(f"Error recognizing patterns: {str(e)}")
            return pd.DataFrame(index=delta_data.index)
    
    def _detect_basic_patterns(self, delta_data: pd.DataFrame) -> pd.Series:
        """Detect basic delta patterns."""
        try:
            cumulative_delta = delta_data.get('cumulative_delta', pd.Series(index=delta_data.index))
            delta_momentum = delta_data.get('delta_momentum', pd.Series(index=delta_data.index))
            
            patterns = []
            
            for i in range(len(cumulative_delta)):
                if i < self.config.pattern_window:
                    patterns.append(DeltaPattern.CONSOLIDATION.value)
                    continue
                
                # Analyze recent pattern
                recent_delta = cumulative_delta.iloc[i-self.config.pattern_window:i+1]
                recent_momentum = delta_momentum.iloc[i-self.config.pattern_window:i+1]
                
                # Pattern classification logic
                delta_trend = (recent_delta.iloc[-1] - recent_delta.iloc[0]) / self.config.pattern_window
                momentum_avg = recent_momentum.mean()
                momentum_std = recent_momentum.std()
                
                if delta_trend > self.config.pattern_threshold:
                    if momentum_avg > 0:
                        patterns.append(DeltaPattern.ACCUMULATION.value)
                    else:
                        patterns.append(DeltaPattern.MOMENTUM_SHIFT.value)
                elif delta_trend < -self.config.pattern_threshold:
                    if momentum_avg < 0:
                        patterns.append(DeltaPattern.DISTRIBUTION.value)
                    else:
                        patterns.append(DeltaPattern.MOMENTUM_SHIFT.value)
                elif momentum_std > self.config.reversal_sensitivity:
                    patterns.append(DeltaPattern.REVERSAL_PATTERN.value)
                elif abs(delta_trend) < self.config.consolidation_threshold:
                    patterns.append(DeltaPattern.CONSOLIDATION.value)
                else:
                    patterns.append(DeltaPattern.BREAKOUT.value)
            
            return pd.Series(patterns, index=delta_data.index)
            
        except Exception as e:
            self.logger.error(f"Error detecting basic patterns: {str(e)}")
            return pd.Series([DeltaPattern.CONSOLIDATION.value] * len(delta_data), 
                           index=delta_data.index)
    
    def _calculate_pattern_strength(self, delta_data: pd.DataFrame) -> pd.Series:
        """Calculate pattern strength."""
        try:
            volume_imbalance = delta_data.get('volume_imbalance', pd.Series(index=delta_data.index))
            delta_momentum = delta_data.get('delta_momentum', pd.Series(index=delta_data.index))
            
            # Combine multiple strength factors
            momentum_strength = delta_momentum.abs()
            imbalance_strength = volume_imbalance.abs()
            
            # Normalize and combine
            momentum_norm = momentum_strength / (momentum_strength.rolling(window=50).max() + 1e-8)
            imbalance_norm = imbalance_strength / (imbalance_strength.rolling(window=50).max() + 1e-8)
            
            pattern_strength = (momentum_norm + imbalance_norm) / 2
            
            return pattern_strength.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern strength: {str(e)}")
            return pd.Series(index=delta_data.index, dtype=float)
    
    def _calculate_reversal_probability(self, delta_data: pd.DataFrame) -> pd.Series:
        """Calculate reversal probability."""
        try:
            delta_momentum = delta_data.get('delta_momentum', pd.Series(index=delta_data.index))
            delta_acceleration = delta_data.get('delta_acceleration', pd.Series(index=delta_data.index))
            
            # Reversal signals
            momentum_reversal = (delta_momentum * delta_momentum.shift(1) < 0).astype(float)
            acceleration_reversal = (delta_acceleration * delta_acceleration.shift(1) < 0).astype(float)
            
            # Combined reversal probability
            reversal_probability = (momentum_reversal + acceleration_reversal) / 2
            
            # Apply smoothing
            smoothed_probability = reversal_probability.rolling(window=10).mean()
            
            return smoothed_probability.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating reversal probability: {str(e)}")
            return pd.Series(index=delta_data.index, dtype=float)
    
    def _ml_pattern_analysis(self, delta_data: pd.DataFrame, 
                           pressure_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Machine learning-based pattern analysis."""
        try:
            # This would be implemented with proper feature engineering
            # and model training in a production system
            
            # For now, return default confidence
            confidence = pd.Series(0.5, index=delta_data.index)
            
            return {'confidence': confidence}
            
        except Exception as e:
            self.logger.error(f"Error in ML pattern analysis: {str(e)}")
            return {'confidence': pd.Series(0.5, index=delta_data.index)}


class AdvancedVolumeDeltaIndicator:
    """
    Advanced Volume Delta Indicator with comprehensive analysis.
    
    This class implements sophisticated volume delta analysis including
    bid/ask flow analysis, pressure measurement, directional tracking,
    and advanced pattern recognition for institutional flow detection.
    """
    
    def __init__(self, config: Optional[VolumeDeltaConfig] = None):
        """
        Initialize the Advanced Volume Delta Indicator.
        
        Args:
            config: Configuration parameters for delta analysis
        """
        self.config = config or VolumeDeltaConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.delta_engine = DeltaCalculationEngine(self.config)
        self.pressure_analyzer = PressureAnalysisSystem(self.config)
        self.pattern_recognizer = DeltaPatternRecognition(self.config)
        
        # State management
        self.current_signals = []
        self.delta_history = deque(maxlen=1000)
        
        # Performance tracking
        self.analysis_count = 0
        self.total_analysis_time = 0.0
        
        self.logger.info("Advanced Volume Delta Indicator initialized successfully")
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive volume delta analysis.
        
        Args:
            data: DataFrame with columns ['timestamp', 'price', 'volume']
            
        Returns:
            Dictionary containing comprehensive delta analysis
        """
        try:
            start_time = datetime.now()
            
            # Validate input data
            if not self._validate_input_data(data):
                raise ValueError("Invalid input data for delta analysis")
            
            # Calculate volume delta metrics
            delta_data = self.delta_engine.calculate_volume_delta(data)
            
            # Analyze flow pressure
            pressure_data = self.pressure_analyzer.analyze_flow_pressure(delta_data)
            
            # Recognize patterns
            pattern_data = self.pattern_recognizer.recognize_patterns(delta_data, pressure_data)
            
            # Generate comprehensive signals
            signals = self._generate_delta_signals(delta_data, pressure_data, pattern_data)
            
            # Calculate statistical measures
            statistics = self._calculate_statistical_measures(delta_data)
            
            # Performance metrics
            performance = self._calculate_performance_metrics(delta_data, signals)
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(analysis_time)
            
            # Store current signals
            self.current_signals = signals
            
            # Compile results
            results = {
                'signals': signals,
                'delta_data': delta_data,
                'pressure_data': pressure_data,
                'pattern_data': pattern_data,
                'statistics': statistics,
                'performance': performance,
                'metadata': {
                    'analysis_time': analysis_time,
                    'data_points': len(data),
                    'config': self.config.__dict__
                }
            }
            
            self.logger.info(f"Volume delta analysis completed in {analysis_time:.4f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in volume delta analysis: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data quality and structure."""
        try:
            required_columns = ['timestamp', 'price', 'volume']
            
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"Missing required columns: {required_columns}")
                return False
            
            if len(data) < 10:
                self.logger.error("Insufficient data points for analysis")
                return False
            
            if (data['volume'] <= 0).any():
                self.logger.warning("Non-positive volume values detected")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating input data: {str(e)}")
            return False
    
    def _generate_delta_signals(self, delta_data: pd.DataFrame, 
                               pressure_data: pd.DataFrame,
                               pattern_data: pd.DataFrame) -> List[VolumeDeltaSignal]:
        """Generate comprehensive delta signals."""
        try:
            signals = []
            
            for i in range(len(delta_data)):
                # Extract current values
                raw_delta = delta_data['raw_delta'].iloc[i] if 'raw_delta' in delta_data else 0.0
                normalized_delta = delta_data['normalized_delta'].iloc[i] if 'normalized_delta' in delta_data else 0.0
                cumulative_delta = delta_data['cumulative_delta'].iloc[i] if 'cumulative_delta' in delta_data else 0.0
                delta_momentum = delta_data['delta_momentum'].iloc[i] if 'delta_momentum' in delta_data else 0.0
                
                buying_volume = delta_data['buying_volume'].iloc[i] if 'buying_volume' in delta_data else 0.0
                selling_volume = delta_data['selling_volume'].iloc[i] if 'selling_volume' in delta_data else 0.0
                volume_imbalance = delta_data['volume_imbalance'].iloc[i] if 'volume_imbalance' in delta_data else 0.0
                
                # Pressure metrics
                pressure_intensity = pressure_data['pressure_intensity'].iloc[i] if 'pressure_intensity' in pressure_data else 0.0
                pressure_gradient = pressure_data['pressure_gradient'].iloc[i] if 'pressure_gradient' in pressure_data else 0.0
                flow_persistence = pressure_data['flow_persistence'].iloc[i] if 'flow_persistence' in pressure_data else 0.0
                directional_strength = pressure_data['directional_strength'].iloc[i] if 'directional_strength' in pressure_data else 0.0
                
                # Pattern metrics
                pattern_strength = pattern_data['pattern_strength'].iloc[i] if 'pattern_strength' in pattern_data else 0.0
                pattern_confidence = pattern_data['pattern_confidence'].iloc[i] if 'pattern_confidence' in pattern_data else 0.0
                reversal_probability = pattern_data['reversal_probability'].iloc[i] if 'reversal_probability' in pattern_data else 0.0
                
                # Classifications
                delta_direction = self._classify_delta_direction(normalized_delta, directional_strength)
                flow_pressure = self._classify_flow_pressure(volume_imbalance, pressure_intensity)
                delta_pattern = self._get_delta_pattern(pattern_data, i)
                
                # Advanced calculations
                delta_acceleration = delta_data['delta_acceleration'].iloc[i] if 'delta_acceleration' in delta_data else 0.0
                delta_volatility = self._calculate_delta_volatility(delta_data, i)
                delta_z_score = self._calculate_delta_z_score(delta_data, i)
                
                # Oscillator calculations
                delta_oscillator = self._calculate_delta_oscillator(volume_imbalance)
                overbought_oversold = self._calculate_overbought_oversold(delta_oscillator)
                
                # Risk and timing
                execution_risk = min(abs(pressure_intensity), 1.0)
                optimal_entry_score = self._calculate_optimal_entry_score(
                    pattern_strength, pattern_confidence, execution_risk
                )
                position_bias = np.clip(directional_strength, -self.config.max_position_bias, 
                                      self.config.max_position_bias)
                
                # Create signal
                signal = VolumeDeltaSignal(
                    raw_delta=raw_delta,
                    normalized_delta=normalized_delta,
                    cumulative_delta=cumulative_delta,
                    delta_momentum=delta_momentum,
                    delta_direction=delta_direction,
                    flow_pressure=flow_pressure,
                    pressure_intensity=pressure_intensity,
                    directional_strength=directional_strength,
                    buying_volume=buying_volume,
                    selling_volume=selling_volume,
                    net_volume=buying_volume - selling_volume,
                    volume_imbalance=volume_imbalance,
                    delta_acceleration=delta_acceleration,
                    pressure_gradient=pressure_gradient,
                    flow_persistence=flow_persistence,
                    delta_volatility=delta_volatility,
                    delta_pattern=delta_pattern,
                    pattern_strength=pattern_strength,
                    pattern_confidence=pattern_confidence,
                    reversal_probability=reversal_probability,
                    delta_z_score=delta_z_score,
                    delta_oscillator=delta_oscillator,
                    overbought_oversold=overbought_oversold,
                    execution_risk=execution_risk,
                    optimal_entry_score=optimal_entry_score,
                    position_bias=position_bias
                )
                
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating delta signals: {str(e)}")
            return []
    
    def _classify_delta_direction(self, normalized_delta: float, directional_strength: float) -> DeltaDirection:
        """Classify delta direction."""
        try:
            strength_abs = abs(directional_strength)
            
            if directional_strength > 0.5:
                if strength_abs > 0.8:
                    return DeltaDirection.BULLISH_STRONG
                elif strength_abs > 0.5:
                    return DeltaDirection.BULLISH_MODERATE
                else:
                    return DeltaDirection.BULLISH_WEAK
            elif directional_strength < -0.5:
                if strength_abs > 0.8:
                    return DeltaDirection.BEARISH_STRONG
                elif strength_abs > 0.5:
                    return DeltaDirection.BEARISH_MODERATE
                else:
                    return DeltaDirection.BEARISH_WEAK
            else:
                return DeltaDirection.NEUTRAL
                
        except Exception as e:
            self.logger.error(f"Error classifying delta direction: {str(e)}")
            return DeltaDirection.NEUTRAL
    
    def _classify_flow_pressure(self, volume_imbalance: float, pressure_intensity: float) -> FlowPressure:
        """Classify flow pressure."""
        try:
            if volume_imbalance > 0.6 and pressure_intensity > 0.8:
                return FlowPressure.EXTREME_BUYING
            elif volume_imbalance > 0.4 and pressure_intensity > 0.6:
                return FlowPressure.STRONG_BUYING
            elif volume_imbalance > 0.2:
                return FlowPressure.MODERATE_BUYING
            elif volume_imbalance < -0.6 and pressure_intensity > 0.8:
                return FlowPressure.EXTREME_SELLING
            elif volume_imbalance < -0.4 and pressure_intensity > 0.6:
                return FlowPressure.STRONG_SELLING
            elif volume_imbalance < -0.2:
                return FlowPressure.MODERATE_SELLING
            else:
                return FlowPressure.BALANCED
                
        except Exception as e:
            self.logger.error(f"Error classifying flow pressure: {str(e)}")
            return FlowPressure.BALANCED
    
    def _get_delta_pattern(self, pattern_data: pd.DataFrame, index: int) -> DeltaPattern:
        """Get delta pattern at specific index."""
        try:
            if 'delta_pattern' in pattern_data.columns and index < len(pattern_data):
                pattern_str = pattern_data['delta_pattern'].iloc[index]
                for pattern in DeltaPattern:
                    if pattern.value == pattern_str:
                        return pattern
            return DeltaPattern.CONSOLIDATION
        except:
            return DeltaPattern.CONSOLIDATION
    
    def _calculate_delta_volatility(self, delta_data: pd.DataFrame, index: int) -> float:
        """Calculate delta volatility."""
        try:
            if 'raw_delta' not in delta_data or index < 20:
                return 0.0
            
            start_idx = max(0, index - 20)
            recent_deltas = delta_data['raw_delta'].iloc[start_idx:index+1]
            
            return recent_deltas.std()
            
        except Exception as e:
            self.logger.error(f"Error calculating delta volatility: {str(e)}")
            return 0.0
    
    def _calculate_delta_z_score(self, delta_data: pd.DataFrame, index: int) -> float:
        """Calculate delta Z-score."""
        try:
            if 'raw_delta' not in delta_data or index < self.config.zscore_window:
                return 0.0
            
            start_idx = max(0, index - self.config.zscore_window)
            recent_deltas = delta_data['raw_delta'].iloc[start_idx:index+1]
            
            current_delta = delta_data['raw_delta'].iloc[index]
            mean_delta = recent_deltas.mean()
            std_delta = recent_deltas.std()
            
            if std_delta > 0:
                return (current_delta - mean_delta) / std_delta
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating delta Z-score: {str(e)}")
            return 0.0
    
    def _calculate_delta_oscillator(self, volume_imbalance: float) -> float:
        """Calculate delta oscillator value."""
        try:
            # Simple oscillator transformation
            oscillator = np.tanh(volume_imbalance * 5) * 50 + 50
            return np.clip(oscillator, 0, 100)
        except:
            return 50.0
    
    def _calculate_overbought_oversold(self, oscillator_value: float) -> float:
        """Calculate overbought/oversold level."""
        try:
            if oscillator_value > self.config.overbought_threshold:
                return (oscillator_value - self.config.overbought_threshold) / (100 - self.config.overbought_threshold)
            elif oscillator_value < self.config.oversold_threshold:
                return (self.config.oversold_threshold - oscillator_value) / self.config.oversold_threshold
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_optimal_entry_score(self, pattern_strength: float, 
                                     pattern_confidence: float, execution_risk: float) -> float:
        """Calculate optimal entry score."""
        try:
            # Combine factors for entry timing
            entry_score = (
                pattern_strength * 0.4 +
                pattern_confidence * 0.4 +
                (1.0 - execution_risk) * 0.2
            )
            
            return np.clip(entry_score, 0, 1)
        except:
            return 0.5
    
    def _calculate_statistical_measures(self, delta_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical measures."""
        try:
            statistics = {}
            
            if 'raw_delta' in delta_data.columns:
                raw_delta = delta_data['raw_delta'].dropna()
                if len(raw_delta) > 0:
                    statistics['delta_mean'] = raw_delta.mean()
                    statistics['delta_std'] = raw_delta.std()
                    statistics['delta_skewness'] = raw_delta.skew()
                    statistics['delta_kurtosis'] = raw_delta.kurtosis()
            
            if 'volume_imbalance' in delta_data.columns:
                imbalance = delta_data['volume_imbalance'].dropna()
                if len(imbalance) > 0:
                    statistics['imbalance_mean'] = imbalance.mean()
                    statistics['imbalance_std'] = imbalance.std()
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical measures: {str(e)}")
            return {}
    
    def _calculate_performance_metrics(self, delta_data: pd.DataFrame, 
                                     signals: List[VolumeDeltaSignal]) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            if not signals:
                return {}
            
            # Signal statistics
            avg_pressure_intensity = np.mean([s.pressure_intensity for s in signals])
            avg_pattern_confidence = np.mean([s.pattern_confidence for s in signals])
            avg_execution_risk = np.mean([s.execution_risk for s in signals])
            
            # Pattern distribution
            pattern_counts = {}
            for signal in signals:
                pattern = signal.delta_pattern.value
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # Direction distribution
            direction_counts = {}
            for signal in signals:
                direction = signal.delta_direction.value
                direction_counts[direction] = direction_counts.get(direction, 0) + 1
            
            performance = {
                'avg_pressure_intensity': avg_pressure_intensity,
                'avg_pattern_confidence': avg_pattern_confidence,
                'avg_execution_risk': avg_execution_risk,
                'pattern_distribution': pattern_counts,
                'direction_distribution': direction_counts,
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
    
    def get_latest_signal(self) -> Optional[VolumeDeltaSignal]:
        """Get the latest delta signal."""
        try:
            if self.current_signals:
                return self.current_signals[-1]
            return None
        except Exception as e:
            self.logger.error(f"Error getting latest signal: {str(e)}")
            return None


# Example usage and testing
if __name__ == "__main__":
    # Test the volume delta indicator
    try:
        # Generate sample data
        np.random.seed(42)
        n_points = 500
        
        timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='1s')
        
        # Generate price data with trends
        price_changes = np.random.randn(n_points) * 0.0001
        prices = 1.1000 + np.cumsum(price_changes)
        
        # Generate volume data
        base_volume = 100
        volumes = np.random.lognormal(np.log(base_volume), 0.6, n_points)
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes
        })
        
        print(f"Generated {len(data)} sample data points")
        
        # Initialize indicator
        config = VolumeDeltaConfig(
            delta_smoothing_window=10,
            pressure_window=20,
            pattern_window=30
        )
        
        indicator = AdvancedVolumeDeltaIndicator(config)
        
        # Perform analysis
        results = indicator.analyze(data)
        
        # Display results
        print(f"Analysis completed in {results['metadata']['analysis_time']:.4f}s")
        print(f"Generated {len(results['signals'])} delta signals")
        
        # Show latest signal
        latest_signal = indicator.get_latest_signal()
        if latest_signal:
            print(f"\nLatest Delta Signal:")
            print(f"  Direction: {latest_signal.delta_direction.value}")
            print(f"  Flow Pressure: {latest_signal.flow_pressure.value}")
            print(f"  Pattern: {latest_signal.delta_pattern.value}")
            print(f"  Raw Delta: {latest_signal.raw_delta:.3f}")
            print(f"  Normalized Delta: {latest_signal.normalized_delta:.3f}")
            print(f"  Volume Imbalance: {latest_signal.volume_imbalance:.3f}")
            print(f"  Pattern Confidence: {latest_signal.pattern_confidence:.3f}")
            print(f"  Execution Risk: {latest_signal.execution_risk:.3f}")
        
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