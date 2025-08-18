"""
Standard Deviation Channel Signal Indicator - Advanced Statistical Channel Analysis

This module implements a sophisticated standard deviation channel indicator with:
- Dynamic channel calculation with adaptive parameters
- Multiple statistical methods (Bollinger, Keltner, Linear Regression channels)
- Volatility-adjusted channel width and positioning
- Advanced breakout detection with statistical significance testing
- Channel squeeze and expansion pattern recognition
- Multi-timeframe channel analysis and consensus building
- Machine learning-enhanced channel prediction and validation
- Mean reversion vs trend continuation signal classification
- Volume-weighted channel calculations and confirmations
- Regime-based parameter adaptation and optimization
- Production-grade error handling and performance tracking

Author: AI Enhancement Team
Version: 9.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from scipy import stats, optimize
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import norm, t, chi2, jarque_bera
import warnings
warnings.filterwarnings('ignore')

class ChannelType(Enum):
    """Enumeration of channel calculation methods."""
    BOLLINGER = "bollinger"
    KELTNER = "keltner"
    LINEAR_REGRESSION = "linear_regression"
    STANDARD_DEVIATION = "standard_deviation"
    PRICE_CHANNEL = "price_channel"
    ADAPTIVE = "adaptive"

class SignalType(Enum):
    """Enumeration of signal types."""
    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"
    MEAN_REVERSION_UP = "mean_reversion_up"
    MEAN_REVERSION_DOWN = "mean_reversion_down"
    SQUEEZE_ENTRY = "squeeze_entry"
    EXPANSION_ENTRY = "expansion_entry"
    CHANNEL_REJECTION = "channel_rejection"

class VolatilityRegime(Enum):
    """Enumeration of volatility regimes."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class ChannelData:
    """Channel data structure."""
    upper_band: float
    lower_band: float
    middle_line: float
    width: float
    position: float  # Price position within channel (0-1)
    channel_type: ChannelType
    confidence: float
    volatility_adjustment: float

@dataclass
class ChannelSignal:
    """Channel signal structure."""
    timestamp: datetime
    signal_type: SignalType
    entry_price: float
    upper_band: float
    lower_band: float
    middle_line: float
    channel_width: float
    position_in_channel: float
    strength: float
    confidence: float
    probability: float
    statistical_significance: float
    volume_confirmation: bool
    target_price: Optional[float]
    stop_loss: Optional[float]
    risk_reward_ratio: float

@dataclass
class ChannelPattern:
    """Channel pattern recognition structure."""
    pattern_type: str
    start_time: datetime
    duration_periods: int
    significance: float
    breakout_probability: float
    mean_reversion_probability: float

class StandardDeviationChannelIndicator:
    """
    Advanced Standard Deviation Channel Signal Indicator with machine learning enhancement.
    
    This indicator provides comprehensive statistical channel analysis including:
    - Dynamic channel calculation with multiple methods
    - Advanced breakout detection with statistical significance testing
    - Channel squeeze and expansion pattern recognition
    - Volatility regime detection and adaptive parameter adjustment
    - Machine learning-enhanced signal classification and prediction
    - Multi-timeframe analysis with consensus building
    - Volume-weighted calculations and confirmations
    - Mean reversion vs trend continuation signal differentiation
    - Risk management with dynamic stop-loss and target calculations
    - Real-time performance tracking and optimization
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the Standard Deviation Channel Indicator.
        
        Args:
            parameters: Configuration parameters for the indicator
        """
        self.parameters = self._set_default_parameters(parameters or {})
        self.logger = self._setup_logger()
        
        # Core components
        self.channel_calculator = ChannelCalculator(self.parameters)
        self.volatility_analyzer = VolatilityAnalyzer(self.parameters)
        self.pattern_recognizer = PatternRecognizer(self.parameters)
        self.signal_classifier = SignalClassifier(self.parameters)
        self.breakout_detector = BreakoutDetector(self.parameters)
        
        # State management
        self.channel_history: List[ChannelData] = []
        self.signal_history: List[ChannelSignal] = []
        self.pattern_history: List[ChannelPattern] = []
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        
        # Machine learning models
        self.breakout_classifier = None
        self.direction_predictor = None
        self.volatility_predictor = None
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.is_trained = False
        
        # Statistical tracking
        self.statistical_metrics = {
            'breakout_accuracy': 0.0,
            'mean_reversion_accuracy': 0.0,
            'false_signal_rate': 0.0,
            'average_signal_quality': 0.0
        }
        
        # Volatility regime tracking
        self.current_regime = VolatilityRegime.NORMAL
        self.regime_history: List[VolatilityRegime] = []
        
        # Performance metrics
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'breakout_success_rate': 0.0,
            'mean_reversion_success_rate': 0.0
        }
        
        self.logger.info("Standard Deviation Channel Indicator initialized with advanced ML enhancement")
    
    def _set_default_parameters(self, user_params: Dict[str, Any]) -> Dict[str, Any]:
        """Set default parameters for the indicator."""
        defaults = {
            # Basic channel parameters
            'period': 20,
            'std_multiplier': 2.0,
            'channel_type': ChannelType.BOLLINGER,
            'adaptive_period': True,
            'min_period': 10,
            'max_period': 50,
            
            # Statistical parameters
            'confidence_level': 0.95,
            'significance_threshold': 0.05,
            'lookback_period': 100,
            'volatility_window': 14,
            'trend_window': 20,
            
            # Breakout detection
            'breakout_threshold': 0.02,  # 2% beyond channel
            'breakout_confirmation_periods': 3,
            'volume_breakout_multiplier': 1.5,
            'momentum_confirmation_required': True,
            'false_breakout_filter': True,
            
            # Mean reversion parameters
            'reversion_threshold': 0.8,  # 80% of channel width
            'reversion_confirmation_periods': 2,
            'oversold_threshold': 0.1,
            'overbought_threshold': 0.9,
            
            # Channel patterns
            'squeeze_threshold': 0.5,  # Relative to historical volatility
            'expansion_threshold': 1.5,
            'pattern_min_duration': 5,
            'pattern_max_duration': 30,
            
            # Machine learning
            'ml_enabled': True,
            'ml_training_period': 200,
            'ml_retrain_frequency': 50,
            'feature_count': 25,
            'prediction_horizon': 5,
            'cross_validation_folds': 5,
            
            # Volatility adjustment
            'volatility_adjustment': True,
            'regime_detection': True,
            'dynamic_std_multiplier': True,
            'vol_regime_window': 30,
            
            # Signal filtering
            'min_signal_strength': 0.4,
            'min_confidence': 0.3,
            'min_statistical_significance': 0.05,
            'signal_clustering_prevention': True,
            'max_signals_per_period': 1,
            
            # Risk management
            'default_risk_reward_ratio': 2.0,
            'max_risk_per_trade': 0.02,
            'adaptive_stop_loss': True,
            'dynamic_targets': True,
            'position_sizing_enabled': True,
            
            # Multi-timeframe analysis
            'timeframes': ['short', 'medium', 'long'],
            'timeframe_weights': [0.3, 0.5, 0.2],
            'consensus_required': True,
            'consensus_threshold': 0.6,
            
            # Performance optimization
            'real_time_optimization': True,
            'parameter_adaptation': True,
            'performance_tracking': True,
            'memory_management': True
        }
        
        # Update with user parameters
        defaults.update(user_params)
        return defaults
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the indicator."""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def calculate(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                 volume: np.ndarray = None, timestamp: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate standard deviation channel signals with advanced statistical analysis.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Optional volume data
            timestamp: Optional timestamp data
            
        Returns:
            Dict containing comprehensive channel analysis results
        """
        try:
            if len(close) < self.parameters['lookback_period']:
                return self._generate_default_result()
            
            # Store price and volume history
            self.price_history = close.tolist()
            if volume is not None:
                self.volume_history = volume.tolist()
            
            # Analyze volatility regime
            volatility_analysis = self._analyze_volatility_regime(high, low, close)
            self.current_regime = volatility_analysis['current_regime']
            
            # Calculate adaptive parameters based on regime
            adaptive_params = self._calculate_adaptive_parameters(volatility_analysis)
            
            # Calculate channel data for multiple methods
            channel_data = self._calculate_channel_data(high, low, close, adaptive_params)
            
            # Train or update ML models if needed
            if self.parameters['ml_enabled']:
                if not self.is_trained and len(close) >= self.parameters['ml_training_period']:
                    self._train_ml_models(high, low, close, volume)
                
                if self.is_trained:
                    # Enhance channel analysis with ML predictions
                    ml_enhanced_data = self._enhance_with_ml_predictions(
                        channel_data, high, low, close, volume
                    )
                else:
                    ml_enhanced_data = channel_data
            else:
                ml_enhanced_data = channel_data
            
            # Detect channel patterns
            patterns = self._detect_channel_patterns(ml_enhanced_data, high, low, close)
            
            # Generate signals based on channel analysis
            signals = self._generate_channel_signals(
                ml_enhanced_data, patterns, high, low, close, volume
            )
            
            # Perform statistical significance testing
            statistical_analysis = self._perform_statistical_analysis(
                signals, ml_enhanced_data, close
            )
            
            # Calculate multi-timeframe consensus if enabled
            if len(self.parameters['timeframes']) > 1:
                consensus_analysis = self._calculate_timeframe_consensus(
                    high, low, close, volume
                )
            else:
                consensus_analysis = {'consensus_strength': 1.0, 'consensus_direction': 0}
            
            # Assess current market position
            current_assessment = self._assess_current_position(
                close[-1], ml_enhanced_data, patterns
            )
            
            # Generate comprehensive results
            current_channel = ml_enhanced_data[-1] if ml_enhanced_data else None
            active_signals = [sig for sig in signals if sig.timestamp >= datetime.now() - timedelta(hours=24)]
            
            result = {
                # Current channel state
                'upper_band': current_channel.upper_band if current_channel else 0.0,
                'lower_band': current_channel.lower_band if current_channel else 0.0,
                'middle_line': current_channel.middle_line if current_channel else 0.0,
                'channel_width': current_channel.width if current_channel else 0.0,
                'position_in_channel': current_channel.position if current_channel else 0.5,
                'channel_type': current_channel.channel_type.value if current_channel else 'standard_deviation',
                
                # Current market assessment
                'current_price': close[-1],
                'price_vs_upper': (close[-1] - current_channel.upper_band) / current_channel.upper_band if current_channel else 0.0,
                'price_vs_lower': (current_channel.lower_band - close[-1]) / current_channel.lower_band if current_channel else 0.0,
                'channel_position_pct': current_assessment.get('position_pct', 50.0),
                'channel_bias': current_assessment.get('bias', 'neutral'),
                
                # Active signals
                'active_signals': [self._signal_to_dict(sig) for sig in active_signals[-5:]],
                'signal_count': len(active_signals),
                'strongest_signal': self._get_strongest_signal(active_signals),
                'latest_signal': self._signal_to_dict(active_signals[-1]) if active_signals else None,
                
                # Pattern analysis
                'active_patterns': [self._pattern_to_dict(pat) for pat in patterns[-3:]],
                'pattern_count': len(patterns),
                'dominant_pattern': self._get_dominant_pattern(patterns),
                
                # Volatility and regime analysis
                'volatility_regime': volatility_analysis['current_regime'].value,
                'volatility_percentile': volatility_analysis.get('percentile', 50.0),
                'regime_stability': volatility_analysis.get('stability', 0.5),
                'volatility_trend': volatility_analysis.get('trend', 'stable'),
                
                # Channel characteristics
                'channel_slope': self._calculate_channel_slope(ml_enhanced_data),
                'channel_stability': self._calculate_channel_stability(ml_enhanced_data),
                'squeeze_score': self._calculate_squeeze_score(ml_enhanced_data, volatility_analysis),
                'expansion_score': self._calculate_expansion_score(ml_enhanced_data, volatility_analysis),
                
                # Breakout analysis
                'breakout_probability_up': self._calculate_breakout_probability(
                    close[-1], current_channel, 'up', volatility_analysis
                ),
                'breakout_probability_down': self._calculate_breakout_probability(
                    close[-1], current_channel, 'down', volatility_analysis
                ),
                'mean_reversion_probability': self._calculate_mean_reversion_probability(
                    close[-1], current_channel, volatility_analysis
                ),
                
                # Statistical analysis
                'statistical_significance': statistical_analysis.get('overall_significance', 0.0),
                'signal_quality_score': statistical_analysis.get('signal_quality', 0.0),
                'confidence_interval_width': statistical_analysis.get('confidence_width', 0.0),
                'distribution_normality': statistical_analysis.get('normality_test', 0.5),
                
                # Multi-timeframe analysis
                'timeframe_consensus': consensus_analysis.get('consensus_strength', 1.0),
                'consensus_direction': consensus_analysis.get('consensus_direction', 0),
                'timeframe_agreement': consensus_analysis.get('agreement_score', 1.0),
                
                # Risk metrics
                'risk_level': self._calculate_risk_level(current_channel, volatility_analysis),
                'recommended_position_size': self._calculate_position_size(
                    active_signals, volatility_analysis
                ),
                'max_favorable_excursion': self._calculate_max_favorable_excursion(close, current_channel),
                'max_adverse_excursion': self._calculate_max_adverse_excursion(close, current_channel),
                
                # Performance metrics
                'breakout_accuracy': self.statistical_metrics.get('breakout_accuracy', 0.0),
                'mean_reversion_accuracy': self.statistical_metrics.get('mean_reversion_accuracy', 0.0),
                'false_signal_rate': self.statistical_metrics.get('false_signal_rate', 0.0),
                'overall_success_rate': self.performance_metrics.get('successful_signals', 0) / 
                                      max(self.performance_metrics.get('total_signals', 1), 1),
                
                # ML metrics (if enabled)
                'ml_enabled': self.parameters['ml_enabled'],
                'ml_trained': self.is_trained,
                'ml_prediction_confidence': self._get_ml_prediction_confidence() if self.is_trained else 0.0,
                'feature_importance_score': self._get_feature_importance_score() if self.is_trained else 0.0,
                
                # Advanced analytics
                'adaptive_parameters': adaptive_params,
                'volatility_analysis': volatility_analysis,
                'statistical_analysis': statistical_analysis,
                'consensus_analysis': consensus_analysis,
                'current_assessment': current_assessment,
                
                # Metadata
                'calculation_timestamp': datetime.now().isoformat(),
                'parameters_used': self._get_active_parameters(),
                'data_quality_score': self._assess_data_quality(high, low, close, volume)
            }
            
            # Store results for future analysis
            if current_channel:
                self.channel_history.append(current_channel)
            self.signal_history.extend(active_signals)
            self.pattern_history.extend(patterns)
            self.regime_history.append(self.current_regime)
            
            # Update performance metrics
            self._update_performance_metrics(active_signals)
            
            # Cleanup old data
            self._cleanup_old_data()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating SD channel signals: {e}")
            return self._generate_error_result(str(e))
    
    def _analyze_volatility_regime(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Analyze current volatility regime and characteristics."""
        try:
            # Calculate various volatility measures
            returns = np.diff(np.log(close))
            
            # Rolling volatility
            vol_window = self.parameters['volatility_window']
            if len(returns) >= vol_window:
                rolling_vol = pd.Series(returns).rolling(window=vol_window).std().fillna(0)
                current_vol = rolling_vol.iloc[-1]
            else:
                current_vol = np.std(returns)
            
            # Historical volatility percentile
            regime_window = self.parameters['vol_regime_window']
            if len(rolling_vol) >= regime_window:
                hist_vol = rolling_vol.iloc[-regime_window:]
                vol_percentile = (hist_vol < current_vol).sum() / len(hist_vol) * 100
            else:
                vol_percentile = 50.0
            
            # Classify regime
            if vol_percentile >= 90:
                regime = VolatilityRegime.EXTREME
            elif vol_percentile >= 75:
                regime = VolatilityRegime.HIGH
            elif vol_percentile <= 25:
                regime = VolatilityRegime.LOW
            else:
                regime = VolatilityRegime.NORMAL
            
            # Calculate True Range for additional volatility measure
            if len(high) >= 2:
                tr = np.maximum(
                    high[1:] - low[1:],
                    np.maximum(
                        np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:] - close[:-1])
                    )
                )
                atr = pd.Series(tr).rolling(window=vol_window).mean().fillna(0)
                current_atr = atr.iloc[-1] if len(atr) > 0 else 0
            else:
                current_atr = 0
            
            # Volatility trend
            if len(rolling_vol) >= 10:
                vol_trend_slope = np.polyfit(range(10), rolling_vol.iloc[-10:].values, 1)[0]
                if vol_trend_slope > 0.0001:
                    vol_trend = 'increasing'
                elif vol_trend_slope < -0.0001:
                    vol_trend = 'decreasing'
                else:
                    vol_trend = 'stable'
            else:
                vol_trend = 'stable'
            
            # Regime stability (how long in current regime)
            stability = self._calculate_regime_stability()
            
            return {
                'current_regime': regime,
                'current_volatility': current_vol,
                'percentile': vol_percentile,
                'atr': current_atr,
                'trend': vol_trend,
                'stability': stability,
                'regime_score': self._calculate_regime_score(regime, vol_percentile),
                'volatility_ratio': current_vol / (np.mean(rolling_vol) + 1e-10) if len(rolling_vol) > 0 else 1.0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility regime: {e}")
            return {
                'current_regime': VolatilityRegime.NORMAL,
                'current_volatility': 0.01,
                'percentile': 50.0,
                'trend': 'stable',
                'stability': 0.5
            }
    
    def _calculate_regime_stability(self) -> float:
        """Calculate how stable the current volatility regime is."""
        try:
            if len(self.regime_history) < 5:
                return 0.5
            
            recent_regimes = self.regime_history[-10:]
            current_regime = recent_regimes[-1]
            
            # Count consecutive periods in current regime
            consecutive_count = 0
            for regime in reversed(recent_regimes):
                if regime == current_regime:
                    consecutive_count += 1
                else:
                    break
            
            # Stability score (0-1)
            stability = min(consecutive_count / 10, 1.0)
            return stability
            
        except Exception as e:
            self.logger.error(f"Error calculating regime stability: {e}")
            return 0.5
    
    def _calculate_regime_score(self, regime: VolatilityRegime, percentile: float) -> float:
        """Calculate a numerical score for the volatility regime."""
        try:
            base_scores = {
                VolatilityRegime.LOW: 0.2,
                VolatilityRegime.NORMAL: 0.5,
                VolatilityRegime.HIGH: 0.8,
                VolatilityRegime.EXTREME: 1.0
            }
            
            base_score = base_scores[regime]
            
            # Adjust based on percentile within regime
            percentile_adjustment = (percentile - 50) / 100 * 0.2
            
            regime_score = np.clip(base_score + percentile_adjustment, 0, 1)
            return regime_score
            
        except Exception as e:
            self.logger.error(f"Error calculating regime score: {e}")
            return 0.5    
    def _calculate_adaptive_parameters(self, volatility_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate adaptive parameters based on current market conditions."""
        try:
            regime = volatility_analysis['current_regime']
            vol_percentile = volatility_analysis['percentile']
            vol_ratio = volatility_analysis.get('volatility_ratio', 1.0)
            
            # Base parameters
            base_period = self.parameters['period']
            base_std_mult = self.parameters['std_multiplier']
            
            # Regime-based adjustments
            regime_adjustments = {
                VolatilityRegime.LOW: {
                    'period_mult': 1.2,
                    'std_mult': 0.8,
                    'breakout_threshold_mult': 1.5,
                    'reversion_threshold_mult': 0.7
                },
                VolatilityRegime.NORMAL: {
                    'period_mult': 1.0,
                    'std_mult': 1.0,
                    'breakout_threshold_mult': 1.0,
                    'reversion_threshold_mult': 1.0
                },
                VolatilityRegime.HIGH: {
                    'period_mult': 0.8,
                    'std_mult': 1.2,
                    'breakout_threshold_mult': 0.7,
                    'reversion_threshold_mult': 1.3
                },
                VolatilityRegime.EXTREME: {
                    'period_mult': 0.6,
                    'std_mult': 1.5,
                    'breakout_threshold_mult': 0.5,
                    'reversion_threshold_mult': 1.5
                }
            }
            
            adjustments = regime_adjustments[regime]
            
            # Calculate adaptive parameters
            adaptive_period = int(np.clip(
                base_period * adjustments['period_mult'] * (1 + (vol_ratio - 1) * 0.2),
                self.parameters['min_period'],
                self.parameters['max_period']
            ))
            
            adaptive_std_mult = base_std_mult * adjustments['std_mult'] * (1 + vol_percentile/200)
            
            adaptive_breakout_threshold = (
                self.parameters['breakout_threshold'] * 
                adjustments['breakout_threshold_mult'] * 
                (1 + vol_ratio * 0.3)
            )
            
            adaptive_reversion_threshold = (
                self.parameters['reversion_threshold'] * 
                adjustments['reversion_threshold_mult']
            )
            
            return {
                'period': adaptive_period,
                'std_multiplier': adaptive_std_mult,
                'breakout_threshold': adaptive_breakout_threshold,
                'reversion_threshold': adaptive_reversion_threshold,
                'volatility_adjustment_factor': vol_ratio,
                'regime_adjustment_strength': adjustments['std_mult']
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive parameters: {e}")
            return {
                'period': self.parameters['period'],
                'std_multiplier': self.parameters['std_multiplier'],
                'breakout_threshold': self.parameters['breakout_threshold'],
                'reversion_threshold': self.parameters['reversion_threshold']
            }
    
    def _calculate_channel_data(self, high: np.ndarray, low: np.ndarray, 
                              close: np.ndarray, adaptive_params: Dict[str, Any]) -> List[ChannelData]:
        """Calculate channel data using specified method with adaptive parameters."""
        try:
            period = adaptive_params['period']
            std_mult = adaptive_params['std_multiplier']
            channel_type = self.parameters['channel_type']
            
            if len(close) < period:
                return []
            
            channels = []
            
            # Calculate typical price for volume-weighted calculations
            typical_price = (high + low + close) / 3
            
            for i in range(period, len(close)):
                window_data = {
                    'high': high[i-period:i],
                    'low': low[i-period:i],
                    'close': close[i-period:i],
                    'typical': typical_price[i-period:i]
                }
                
                # Calculate channel based on type
                if channel_type == ChannelType.BOLLINGER or channel_type == ChannelType.STANDARD_DEVIATION:
                    channel = self._calculate_bollinger_channel(window_data, std_mult, close[i])
                elif channel_type == ChannelType.KELTNER:
                    channel = self._calculate_keltner_channel(window_data, std_mult, close[i])
                elif channel_type == ChannelType.LINEAR_REGRESSION:
                    channel = self._calculate_linear_regression_channel(window_data, std_mult, close[i])
                elif channel_type == ChannelType.PRICE_CHANNEL:
                    channel = self._calculate_price_channel(window_data, close[i])
                elif channel_type == ChannelType.ADAPTIVE:
                    channel = self._calculate_adaptive_channel(window_data, std_mult, close[i], adaptive_params)
                else:
                    channel = self._calculate_bollinger_channel(window_data, std_mult, close[i])
                
                channels.append(channel)
            
            return channels
            
        except Exception as e:
            self.logger.error(f"Error calculating channel data: {e}")
            return []
    
    def _calculate_bollinger_channel(self, window_data: Dict[str, np.ndarray], 
                                   std_mult: float, current_price: float) -> ChannelData:
        """Calculate Bollinger Bands style channel."""
        try:
            close_data = window_data['close']
            
            # Calculate moving average (middle line)
            middle_line = np.mean(close_data)
            
            # Calculate standard deviation
            std_dev = np.std(close_data, ddof=1)
            
            # Calculate bands
            upper_band = middle_line + (std_dev * std_mult)
            lower_band = middle_line - (std_dev * std_mult)
            
            # Calculate channel width and position
            width = upper_band - lower_band
            position = (current_price - lower_band) / width if width > 0 else 0.5
            
            # Calculate confidence based on price distribution
            confidence = self._calculate_channel_confidence(close_data, middle_line, std_dev)
            
            return ChannelData(
                upper_band=upper_band,
                lower_band=lower_band,
                middle_line=middle_line,
                width=width,
                position=np.clip(position, 0, 1),
                channel_type=ChannelType.BOLLINGER,
                confidence=confidence,
                volatility_adjustment=std_mult
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger channel: {e}")
            return self._get_default_channel_data(current_price)
    
    def _calculate_keltner_channel(self, window_data: Dict[str, np.ndarray], 
                                 std_mult: float, current_price: float) -> ChannelData:
        """Calculate Keltner Channel using Average True Range."""
        try:
            high_data = window_data['high']
            low_data = window_data['low']
            close_data = window_data['close']
            
            # Calculate middle line (EMA of typical price)
            typical_data = window_data['typical']
            middle_line = np.mean(typical_data)
            
            # Calculate Average True Range
            if len(close_data) > 1:
                tr = np.maximum(
                    high_data[1:] - low_data[1:],
                    np.maximum(
                        np.abs(high_data[1:] - close_data[:-1]),
                        np.abs(low_data[1:] - close_data[:-1])
                    )
                )
                atr = np.mean(tr)
            else:
                atr = np.mean(high_data - low_data)
            
            # Calculate bands
            upper_band = middle_line + (atr * std_mult)
            lower_band = middle_line - (atr * std_mult)
            
            # Calculate channel metrics
            width = upper_band - lower_band
            position = (current_price - lower_band) / width if width > 0 else 0.5
            confidence = self._calculate_channel_confidence(close_data, middle_line, atr)
            
            return ChannelData(
                upper_band=upper_band,
                lower_band=lower_band,
                middle_line=middle_line,
                width=width,
                position=np.clip(position, 0, 1),
                channel_type=ChannelType.KELTNER,
                confidence=confidence,
                volatility_adjustment=std_mult
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating Keltner channel: {e}")
            return self._get_default_channel_data(current_price)
    
    def _calculate_linear_regression_channel(self, window_data: Dict[str, np.ndarray], 
                                           std_mult: float, current_price: float) -> ChannelData:
        """Calculate Linear Regression Channel."""
        try:
            close_data = window_data['close']
            x = np.arange(len(close_data))
            
            # Fit linear regression
            slope, intercept = np.polyfit(x, close_data, 1)
            
            # Calculate middle line (regression line)
            middle_line = slope * (len(close_data) - 1) + intercept
            
            # Calculate standard error
            regression_line = slope * x + intercept
            residuals = close_data - regression_line
            std_error = np.std(residuals, ddof=1)
            
            # Calculate bands
            upper_band = middle_line + (std_error * std_mult)
            lower_band = middle_line - (std_error * std_mult)
            
            # Calculate channel metrics
            width = upper_band - lower_band
            position = (current_price - lower_band) / width if width > 0 else 0.5
            
            # Confidence based on R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((close_data - np.mean(close_data)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            confidence = r_squared
            
            return ChannelData(
                upper_band=upper_band,
                lower_band=lower_band,
                middle_line=middle_line,
                width=width,
                position=np.clip(position, 0, 1),
                channel_type=ChannelType.LINEAR_REGRESSION,
                confidence=confidence,
                volatility_adjustment=std_mult
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating linear regression channel: {e}")
            return self._get_default_channel_data(current_price)
    
    def _calculate_price_channel(self, window_data: Dict[str, np.ndarray], 
                               current_price: float) -> ChannelData:
        """Calculate simple price channel (highest high, lowest low)."""
        try:
            high_data = window_data['high']
            low_data = window_data['low']
            
            # Calculate bands
            upper_band = np.max(high_data)
            lower_band = np.min(low_data)
            middle_line = (upper_band + lower_band) / 2
            
            # Calculate channel metrics
            width = upper_band - lower_band
            position = (current_price - lower_band) / width if width > 0 else 0.5
            
            # Confidence based on channel utilization
            close_data = window_data['close']
            utilization = np.sum((close_data >= middle_line * 0.9) & 
                               (close_data <= middle_line * 1.1)) / len(close_data)
            confidence = 1 - utilization  # Lower utilization = higher confidence
            
            return ChannelData(
                upper_band=upper_band,
                lower_band=lower_band,
                middle_line=middle_line,
                width=width,
                position=np.clip(position, 0, 1),
                channel_type=ChannelType.PRICE_CHANNEL,
                confidence=confidence,
                volatility_adjustment=1.0
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating price channel: {e}")
            return self._get_default_channel_data(current_price)
    
    def _calculate_adaptive_channel(self, window_data: Dict[str, np.ndarray], 
                                  std_mult: float, current_price: float,
                                  adaptive_params: Dict[str, Any]) -> ChannelData:
        """Calculate adaptive channel using multiple methods and weighted combination."""
        try:
            # Calculate multiple channel types
            bollinger = self._calculate_bollinger_channel(window_data, std_mult, current_price)
            keltner = self._calculate_keltner_channel(window_data, std_mult, current_price)
            price_channel = self._calculate_price_channel(window_data, current_price)
            
            # Weight channels based on their confidence and current regime
            vol_ratio = adaptive_params.get('volatility_adjustment_factor', 1.0)
            
            if vol_ratio > 1.5:  # High volatility - prefer ATR-based
                weights = [0.2, 0.6, 0.2]  # Bollinger, Keltner, Price
            elif vol_ratio < 0.7:  # Low volatility - prefer statistical
                weights = [0.6, 0.2, 0.2]
            else:  # Normal volatility - balanced
                weights = [0.4, 0.4, 0.2]
            
            channels = [bollinger, keltner, price_channel]
            
            # Weighted combination
            upper_band = sum(w * ch.upper_band for w, ch in zip(weights, channels))
            lower_band = sum(w * ch.lower_band for w, ch in zip(weights, channels))
            middle_line = sum(w * ch.middle_line for w, ch in zip(weights, channels))
            
            # Calculate metrics
            width = upper_band - lower_band
            position = (current_price - lower_band) / width if width > 0 else 0.5
            confidence = sum(w * ch.confidence for w, ch in zip(weights, channels))
            
            return ChannelData(
                upper_band=upper_band,
                lower_band=lower_band,
                middle_line=middle_line,
                width=width,
                position=np.clip(position, 0, 1),
                channel_type=ChannelType.ADAPTIVE,
                confidence=confidence,
                volatility_adjustment=std_mult
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptive channel: {e}")
            return self._get_default_channel_data(current_price)
    
    def _calculate_channel_confidence(self, close_data: np.ndarray, 
                                    middle_line: float, deviation: float) -> float:
        """Calculate confidence score for channel calculations."""
        try:
            if len(close_data) < 3:
                return 0.5
            
            # Test for normality of residuals
            residuals = close_data - middle_line
            _, p_value = stats.jarque_bera(residuals)
            normality_score = p_value  # Higher p-value = more normal
            
            # Calculate channel utilization
            within_channel = np.sum(np.abs(residuals) <= deviation * 2) / len(residuals)
            
            # Combine metrics
            confidence = (normality_score * 0.3 + within_channel * 0.7)
            return np.clip(confidence, 0, 1)
            
        except Exception as e:
            return 0.5
    
    def _get_default_channel_data(self, current_price: float) -> ChannelData:
        """Get default channel data when calculation fails."""
        width = current_price * 0.02  # 2% default width
        return ChannelData(
            upper_band=current_price + width/2,
            lower_band=current_price - width/2,
            middle_line=current_price,
            width=width,
            position=0.5,
            channel_type=ChannelType.STANDARD_DEVIATION,
            confidence=0.1,
            volatility_adjustment=1.0
        )
    
    def _train_ml_models(self, high: np.ndarray, low: np.ndarray, 
                        close: np.ndarray, volume: np.ndarray = None):
        """Train machine learning models for enhanced signal prediction."""
        try:
            self.logger.info("Training ML models for SD Channel analysis...")
            
            # Prepare features
            features = self._extract_features(high, low, close, volume)
            
            if len(features) < self.parameters['ml_training_period']:
                return
            
            # Prepare targets for different models
            breakout_targets = self._prepare_breakout_targets(high, low, close)
            direction_targets = self._prepare_direction_targets(close)
            volatility_targets = self._prepare_volatility_targets(high, low, close)
            
            # Ensure all arrays have same length
            min_length = min(len(features), len(breakout_targets), 
                           len(direction_targets), len(volatility_targets))
            
            features = features[:min_length]
            breakout_targets = breakout_targets[:min_length]
            direction_targets = direction_targets[:min_length]
            volatility_targets = volatility_targets[:min_length]
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train breakout classifier
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, breakout_targets, test_size=0.2, random_state=42
            )
            
            self.breakout_classifier = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
            self.breakout_classifier.fit(X_train, y_train)
            
            # Evaluate breakout classifier
            y_pred = self.breakout_classifier.predict(X_test)
            breakout_accuracy = accuracy_score(y_test, y_pred)
            
            # Train direction predictor
            self.direction_predictor = GradientBoostingRegressor(
                n_estimators=100, max_depth=6, random_state=42
            )
            self.direction_predictor.fit(X_train, direction_targets[:len(X_train)])
            
            # Train volatility predictor
            self.volatility_predictor = GradientBoostingRegressor(
                n_estimators=100, max_depth=6, random_state=42
            )
            self.volatility_predictor.fit(X_train, volatility_targets[:len(X_train)])
            
            self.is_trained = True
            
            # Update statistical metrics
            self.statistical_metrics['breakout_accuracy'] = breakout_accuracy
            
            self.logger.info(f"ML models trained successfully. Breakout accuracy: {breakout_accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error training ML models: {e}")
            self.is_trained = False    
    def _extract_features(self, high: np.ndarray, low: np.ndarray, 
                         close: np.ndarray, volume: np.ndarray = None) -> np.ndarray:
        """Extract comprehensive features for ML models."""
        try:
            features_list = []
            
            # Price-based features
            returns = np.diff(np.log(close), prepend=0)
            
            # Technical indicators as features
            periods = [5, 10, 20, 50]
            
            for period in periods:
                if len(close) >= period:
                    # Moving averages
                    ma = pd.Series(close).rolling(window=period).mean().fillna(method='bfill')
                    features_list.append((close / ma - 1).values)  # Price relative to MA
                    
                    # Volatility
                    vol = pd.Series(returns).rolling(window=period).std().fillna(0)
                    features_list.append(vol.values)
                    
                    # RSI-like momentum
                    gains = np.where(returns > 0, returns, 0)
                    losses = np.where(returns < 0, -returns, 0)
                    avg_gain = pd.Series(gains).rolling(window=period).mean().fillna(0)
                    avg_loss = pd.Series(losses).rolling(window=period).mean().fillna(0.01)
                    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
                    features_list.append(rsi.values)
            
            # Channel position features
            if len(self.channel_history) >= 10:
                recent_channels = self.channel_history[-10:]
                positions = [ch.position for ch in recent_channels]
                widths = [ch.width / close[i] for i, ch in enumerate(recent_channels[-len(positions):])]
                
                features_list.append(np.array(positions + [0] * (len(close) - len(positions))))
                features_list.append(np.array(widths + [0] * (len(close) - len(widths))))
            
            # Volume features (if available)
            if volume is not None and len(volume) > 0:
                vol_ma = pd.Series(volume).rolling(window=20).mean().fillna(method='bfill')
                vol_ratio = volume / vol_ma
                features_list.append(vol_ratio.values)
                
                # Price-volume correlation
                pv_corr = pd.Series(close).rolling(window=20).corr(pd.Series(volume)).fillna(0)
                features_list.append(pv_corr.values)
            
            # Ensure all feature arrays have the same length
            if features_list:
                min_length = min(len(arr) for arr in features_list)
                features_array = np.column_stack([arr[:min_length] for arr in features_list])
                return features_array
            else:
                # Return dummy features if nothing calculated
                return np.zeros((len(close), 5))
                
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return np.zeros((len(close), 5))
    
    def _prepare_breakout_targets(self, high: np.ndarray, low: np.ndarray, 
                                close: np.ndarray) -> np.ndarray:
        """Prepare target labels for breakout prediction."""
        try:
            targets = np.zeros(len(close))
            horizon = self.parameters['prediction_horizon']
            
            for i in range(len(close) - horizon):
                future_high = np.max(high[i+1:i+1+horizon])
                future_low = np.min(low[i+1:i+1+horizon])
                current_price = close[i]
                
                # Calculate percentage moves
                up_move = (future_high - current_price) / current_price
                down_move = (current_price - future_low) / current_price
                
                # Classify as breakout if move exceeds threshold
                breakout_threshold = self.parameters['breakout_threshold']
                
                if up_move > breakout_threshold and up_move > down_move:
                    targets[i] = 1  # Upward breakout
                elif down_move > breakout_threshold and down_move > up_move:
                    targets[i] = -1  # Downward breakout
                else:
                    targets[i] = 0  # No significant breakout
            
            return targets
            
        except Exception as e:
            self.logger.error(f"Error preparing breakout targets: {e}")
            return np.zeros(len(close))
    
    def _prepare_direction_targets(self, close: np.ndarray) -> np.ndarray:
        """Prepare target labels for direction prediction."""
        try:
            horizon = self.parameters['prediction_horizon']
            targets = np.zeros(len(close))
            
            for i in range(len(close) - horizon):
                future_return = (close[i + horizon] - close[i]) / close[i]
                targets[i] = future_return
            
            return targets
            
        except Exception as e:
            self.logger.error(f"Error preparing direction targets: {e}")
            return np.zeros(len(close))
    
    def _prepare_volatility_targets(self, high: np.ndarray, low: np.ndarray, 
                                  close: np.ndarray) -> np.ndarray:
        """Prepare target labels for volatility prediction."""
        try:
            horizon = self.parameters['prediction_horizon']
            targets = np.zeros(len(close))
            
            for i in range(len(close) - horizon):
                future_prices = close[i+1:i+1+horizon]
                if len(future_prices) > 1:
                    future_vol = np.std(np.diff(np.log(future_prices)))
                    targets[i] = future_vol
            
            return targets
            
        except Exception as e:
            self.logger.error(f"Error preparing volatility targets: {e}")
            return np.zeros(len(close))
    
    def _enhance_with_ml_predictions(self, channel_data: List[ChannelData], 
                                   high: np.ndarray, low: np.ndarray, 
                                   close: np.ndarray, volume: np.ndarray = None) -> List[ChannelData]:
        """Enhance channel data with ML predictions."""
        try:
            if not self.is_trained or not channel_data:
                return channel_data
            
            # Extract features for current market state
            features = self._extract_features(high, low, close, volume)
            
            if len(features) == 0:
                return channel_data
            
            # Get recent features for prediction
            recent_features = features[-min(len(features), 10):]
            features_scaled = self.scaler.transform(recent_features)
            
            # Make predictions
            breakout_probs = self.breakout_classifier.predict_proba(features_scaled)
            direction_pred = self.direction_predictor.predict(features_scaled)
            volatility_pred = self.volatility_predictor.predict(features_scaled)
            
            # Enhance channel data with predictions
            enhanced_channels = []
            
            for i, channel in enumerate(channel_data):
                if i < len(breakout_probs):
                    # Adjust confidence based on ML predictions
                    ml_confidence = np.max(breakout_probs[i])
                    enhanced_confidence = (channel.confidence + ml_confidence) / 2
                    
                    # Adjust volatility based on prediction
                    vol_adjustment = 1.0 + (volatility_pred[i] - np.mean(volatility_pred)) * 0.5
                    vol_adjustment = np.clip(vol_adjustment, 0.5, 2.0)
                    
                    enhanced_channel = ChannelData(
                        upper_band=channel.upper_band * vol_adjustment,
                        lower_band=channel.lower_band / vol_adjustment,
                        middle_line=channel.middle_line,
                        width=channel.width * vol_adjustment,
                        position=channel.position,
                        channel_type=channel.channel_type,
                        confidence=enhanced_confidence,
                        volatility_adjustment=vol_adjustment
                    )
                    enhanced_channels.append(enhanced_channel)
                else:
                    enhanced_channels.append(channel)
            
            return enhanced_channels
            
        except Exception as e:
            self.logger.error(f"Error enhancing with ML predictions: {e}")
            return channel_data
    
    def _detect_channel_patterns(self, channel_data: List[ChannelData], 
                               high: np.ndarray, low: np.ndarray, 
                               close: np.ndarray) -> List[ChannelPattern]:
        """Detect channel patterns (squeeze, expansion, trend)."""
        try:
            if len(channel_data) < self.parameters['pattern_min_duration']:
                return []
            
            patterns = []
            
            # Extract channel widths for pattern analysis
            widths = [ch.width for ch in channel_data]
            positions = [ch.position for ch in channel_data]
            
            # Detect squeeze patterns
            squeeze_patterns = self._detect_squeeze_patterns(widths, channel_data)
            patterns.extend(squeeze_patterns)
            
            # Detect expansion patterns
            expansion_patterns = self._detect_expansion_patterns(widths, channel_data)
            patterns.extend(expansion_patterns)
            
            # Detect trend channel patterns
            trend_patterns = self._detect_trend_patterns(positions, channel_data, close)
            patterns.extend(trend_patterns)
            
            # Detect channel break patterns
            break_patterns = self._detect_channel_breaks(positions, channel_data, high, low, close)
            patterns.extend(break_patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting channel patterns: {e}")
            return []
    
    def _detect_squeeze_patterns(self, widths: List[float], 
                               channel_data: List[ChannelData]) -> List[ChannelPattern]:
        """Detect channel squeeze patterns."""
        try:
            patterns = []
            
            if len(widths) < 10:
                return patterns
            
            # Calculate rolling statistics
            width_series = pd.Series(widths)
            rolling_mean = width_series.rolling(window=10).mean()
            rolling_std = width_series.rolling(window=10).std()
            
            # Detect squeeze (width significantly below average)
            squeeze_threshold = self.parameters['squeeze_threshold']
            
            in_squeeze = False
            squeeze_start = 0
            
            for i in range(10, len(widths)):
                current_width = widths[i]
                avg_width = rolling_mean.iloc[i]
                std_width = rolling_std.iloc[i]
                
                # Check if in squeeze
                is_squeeze = current_width < (avg_width - squeeze_threshold * std_width)
                
                if is_squeeze and not in_squeeze:
                    # Squeeze starting
                    in_squeeze = True
                    squeeze_start = i
                elif not is_squeeze and in_squeeze:
                    # Squeeze ending
                    in_squeeze = False
                    duration = i - squeeze_start
                    
                    if duration >= self.parameters['pattern_min_duration']:
                        # Calculate pattern significance
                        squeeze_widths = widths[squeeze_start:i]
                        avg_squeeze_width = np.mean(squeeze_widths)
                        normal_width = np.mean(widths[max(0, squeeze_start-20):squeeze_start])
                        
                        if normal_width > 0:
                            compression_ratio = avg_squeeze_width / normal_width
                            significance = max(0, 1 - compression_ratio)
                        else:
                            significance = 0.5
                        
                        pattern = ChannelPattern(
                            pattern_type="squeeze",
                            start_time=datetime.now() - timedelta(hours=duration),
                            duration_periods=duration,
                            significance=significance,
                            breakout_probability=significance * 0.8,  # High squeeze = high breakout prob
                            mean_reversion_probability=0.2
                        )
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting squeeze patterns: {e}")
            return []
    
    def _detect_expansion_patterns(self, widths: List[float], 
                                 channel_data: List[ChannelData]) -> List[ChannelPattern]:
        """Detect channel expansion patterns."""
        try:
            patterns = []
            
            if len(widths) < 10:
                return patterns
            
            # Calculate rolling statistics
            width_series = pd.Series(widths)
            rolling_mean = width_series.rolling(window=10).mean()
            rolling_std = width_series.rolling(window=10).std()
            
            # Detect expansion (width significantly above average)
            expansion_threshold = self.parameters['expansion_threshold']
            
            in_expansion = False
            expansion_start = 0
            
            for i in range(10, len(widths)):
                current_width = widths[i]
                avg_width = rolling_mean.iloc[i]
                std_width = rolling_std.iloc[i]
                
                # Check if in expansion
                is_expansion = current_width > (avg_width + expansion_threshold * std_width)
                
                if is_expansion and not in_expansion:
                    # Expansion starting
                    in_expansion = True
                    expansion_start = i
                elif not is_expansion and in_expansion:
                    # Expansion ending
                    in_expansion = False
                    duration = i - expansion_start
                    
                    if duration >= self.parameters['pattern_min_duration']:
                        # Calculate pattern significance
                        expansion_widths = widths[expansion_start:i]
                        avg_expansion_width = np.mean(expansion_widths)
                        normal_width = np.mean(widths[max(0, expansion_start-20):expansion_start])
                        
                        if normal_width > 0:
                            expansion_ratio = avg_expansion_width / normal_width
                            significance = min(1, expansion_ratio - 1)
                        else:
                            significance = 0.5
                        
                        pattern = ChannelPattern(
                            pattern_type="expansion",
                            start_time=datetime.now() - timedelta(hours=duration),
                            duration_periods=duration,
                            significance=significance,
                            breakout_probability=0.3,  # Expansion already happened
                            mean_reversion_probability=significance * 0.7  # Higher chance of reversion
                        )
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting expansion patterns: {e}")
            return []
    
    def _detect_trend_patterns(self, positions: List[float], 
                             channel_data: List[ChannelData], 
                             close: np.ndarray) -> List[ChannelPattern]:
        """Detect trending channel patterns."""
        try:
            patterns = []
            
            if len(positions) < 20:
                return patterns
            
            # Analyze position trend
            window_size = 15
            
            for i in range(window_size, len(positions)):
                window_positions = positions[i-window_size:i]
                
                # Calculate trend in channel position
                x = np.arange(len(window_positions))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, window_positions)
                
                # Significant trend if R-squared > 0.5 and p-value < 0.05
                if r_value**2 > 0.25 and p_value < 0.1:
                    trend_strength = abs(slope)
                    
                    if trend_strength > 0.02:  # Minimum trend strength
                        pattern_type = "uptrend_channel" if slope > 0 else "downtrend_channel"
                        
                        pattern = ChannelPattern(
                            pattern_type=pattern_type,
                            start_time=datetime.now() - timedelta(hours=window_size),
                            duration_periods=window_size,
                            significance=min(1, trend_strength * 10),
                            breakout_probability=0.6 if abs(slope) > 0.05 else 0.3,
                            mean_reversion_probability=0.4 if abs(slope) > 0.05 else 0.7
                        )
                        patterns.append(pattern)
                        break  # Only add one trend pattern per analysis
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting trend patterns: {e}")
            return []    
    def _detect_channel_breaks(self, positions: List[float], 
                             channel_data: List[ChannelData],
                             high: np.ndarray, low: np.ndarray, 
                             close: np.ndarray) -> List[ChannelPattern]:
        """Detect channel breakout patterns."""
        try:
            patterns = []
            
            if len(positions) < 5:
                return patterns
            
            # Look for recent channel breaks
            lookback = min(10, len(positions))
            recent_positions = positions[-lookback:]
            recent_channels = channel_data[-lookback:]
            
            # Check for breakouts (position > 1 or < 0)
            for i, (pos, channel) in enumerate(zip(recent_positions, recent_channels)):
                if pos > 1.0:  # Upper breakout
                    # Verify with actual price data
                    if len(close) > len(positions) - lookback + i:
                        price_idx = len(close) - lookback + i
                        if close[price_idx] > channel.upper_band:
                            pattern = ChannelPattern(
                                pattern_type="upper_breakout",
                                start_time=datetime.now() - timedelta(hours=lookback-i),
                                duration_periods=1,
                                significance=min(1, pos - 1),
                                breakout_probability=0.8,
                                mean_reversion_probability=0.2
                            )
                            patterns.append(pattern)
                
                elif pos < 0.0:  # Lower breakout
                    if len(close) > len(positions) - lookback + i:
                        price_idx = len(close) - lookback + i
                        if close[price_idx] < channel.lower_band:
                            pattern = ChannelPattern(
                                pattern_type="lower_breakout",
                                start_time=datetime.now() - timedelta(hours=lookback-i),
                                duration_periods=1,
                                significance=min(1, abs(pos)),
                                breakout_probability=0.8,
                                mean_reversion_probability=0.2
                            )
                            patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting channel breaks: {e}")
            return []
    
    def _generate_channel_signals(self, channel_data: List[ChannelData], 
                                patterns: List[ChannelPattern],
                                high: np.ndarray, low: np.ndarray, 
                                close: np.ndarray, volume: np.ndarray = None) -> List[ChannelSignal]:
        """Generate trading signals based on channel analysis."""
        try:
            signals = []
            
            if not channel_data:
                return signals
            
            current_channel = channel_data[-1]
            current_price = close[-1]
            
            # Breakout signals
            breakout_signals = self._generate_breakout_signals(
                current_channel, current_price, patterns, high, low, close, volume
            )
            signals.extend(breakout_signals)
            
            # Mean reversion signals
            reversion_signals = self._generate_mean_reversion_signals(
                current_channel, current_price, patterns, close, volume
            )
            signals.extend(reversion_signals)
            
            # Squeeze signals
            squeeze_signals = self._generate_squeeze_signals(
                current_channel, patterns, close, volume
            )
            signals.extend(squeeze_signals)
            
            # Channel rejection signals
            rejection_signals = self._generate_rejection_signals(
                current_channel, current_price, high, low, close, volume
            )
            signals.extend(rejection_signals)
            
            # Filter and rank signals
            filtered_signals = self._filter_and_rank_signals(signals)
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error generating channel signals: {e}")
            return []
    
    def _generate_breakout_signals(self, current_channel: ChannelData, 
                                 current_price: float, patterns: List[ChannelPattern],
                                 high: np.ndarray, low: np.ndarray, 
                                 close: np.ndarray, volume: np.ndarray = None) -> List[ChannelSignal]:
        """Generate breakout signals."""
        try:
            signals = []
            breakout_threshold = self.parameters['breakout_threshold']
            
            # Check for upper breakout
            upper_breakout_pct = (current_price - current_channel.upper_band) / current_channel.upper_band
            if upper_breakout_pct > breakout_threshold:
                
                # Calculate signal strength
                strength = min(1, upper_breakout_pct / (breakout_threshold * 2))
                
                # Calculate confidence based on patterns and volume
                confidence = self._calculate_breakout_confidence(
                    'up', patterns, current_channel, volume
                )
                
                # Calculate probability using ML if available
                probability = self._calculate_breakout_probability(
                    current_price, current_channel, 'up', {'current_regime': self.current_regime}
                )
                
                # Calculate targets and stops
                target_price = current_price * (1 + current_channel.width / current_price * 0.5)
                stop_loss = current_channel.upper_band * 0.99
                risk_reward = (target_price - current_price) / (current_price - stop_loss) if current_price > stop_loss else 1.0
                
                signal = ChannelSignal(
                    timestamp=datetime.now(),
                    signal_type=SignalType.BREAKOUT_UP,
                    entry_price=current_price,
                    upper_band=current_channel.upper_band,
                    lower_band=current_channel.lower_band,
                    middle_line=current_channel.middle_line,
                    channel_width=current_channel.width,
                    position_in_channel=current_channel.position,
                    strength=strength,
                    confidence=confidence,
                    probability=probability,
                    statistical_significance=self._calculate_statistical_significance(
                        upper_breakout_pct, current_channel
                    ),
                    volume_confirmation=self._check_volume_confirmation(volume, 'breakout'),
                    target_price=target_price,
                    stop_loss=stop_loss,
                    risk_reward_ratio=risk_reward
                )
                signals.append(signal)
            
            # Check for lower breakout
            lower_breakout_pct = (current_channel.lower_band - current_price) / current_channel.lower_band
            if lower_breakout_pct > breakout_threshold:
                
                strength = min(1, lower_breakout_pct / (breakout_threshold * 2))
                
                confidence = self._calculate_breakout_confidence(
                    'down', patterns, current_channel, volume
                )
                
                probability = self._calculate_breakout_probability(
                    current_price, current_channel, 'down', {'current_regime': self.current_regime}
                )
                
                target_price = current_price * (1 - current_channel.width / current_price * 0.5)
                stop_loss = current_channel.lower_band * 1.01
                risk_reward = (current_price - target_price) / (stop_loss - current_price) if stop_loss > current_price else 1.0
                
                signal = ChannelSignal(
                    timestamp=datetime.now(),
                    signal_type=SignalType.BREAKOUT_DOWN,
                    entry_price=current_price,
                    upper_band=current_channel.upper_band,
                    lower_band=current_channel.lower_band,
                    middle_line=current_channel.middle_line,
                    channel_width=current_channel.width,
                    position_in_channel=current_channel.position,
                    strength=strength,
                    confidence=confidence,
                    probability=probability,
                    statistical_significance=self._calculate_statistical_significance(
                        lower_breakout_pct, current_channel
                    ),
                    volume_confirmation=self._check_volume_confirmation(volume, 'breakout'),
                    target_price=target_price,
                    stop_loss=stop_loss,
                    risk_reward_ratio=risk_reward
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating breakout signals: {e}")
            return []
    
    def _generate_mean_reversion_signals(self, current_channel: ChannelData, 
                                       current_price: float, patterns: List[ChannelPattern],
                                       close: np.ndarray, volume: np.ndarray = None) -> List[ChannelSignal]:
        """Generate mean reversion signals."""
        try:
            signals = []
            reversion_threshold = self.parameters['reversion_threshold']
            
            # Check for mean reversion from upper extreme
            if current_channel.position > reversion_threshold:
                
                # Calculate signal strength (stronger near extremes)
                strength = (current_channel.position - reversion_threshold) / (1 - reversion_threshold)
                
                # Calculate confidence
                confidence = self._calculate_reversion_confidence(
                    'down', patterns, current_channel, volume
                )
                
                # Calculate probability
                probability = self._calculate_mean_reversion_probability(
                    current_price, current_channel, {'current_regime': self.current_regime}
                )
                
                # Calculate targets
                target_price = current_channel.middle_line
                stop_loss = current_channel.upper_band * 1.01
                risk_reward = (current_price - target_price) / (stop_loss - current_price) if stop_loss > current_price else 1.0
                
                signal = ChannelSignal(
                    timestamp=datetime.now(),
                    signal_type=SignalType.MEAN_REVERSION_DOWN,
                    entry_price=current_price,
                    upper_band=current_channel.upper_band,
                    lower_band=current_channel.lower_band,
                    middle_line=current_channel.middle_line,
                    channel_width=current_channel.width,
                    position_in_channel=current_channel.position,
                    strength=strength,
                    confidence=confidence,
                    probability=probability,
                    statistical_significance=self._calculate_statistical_significance(
                        current_channel.position - 0.5, current_channel
                    ),
                    volume_confirmation=self._check_volume_confirmation(volume, 'reversion'),
                    target_price=target_price,
                    stop_loss=stop_loss,
                    risk_reward_ratio=risk_reward
                )
                signals.append(signal)
            
            # Check for mean reversion from lower extreme
            elif current_channel.position < (1 - reversion_threshold):
                
                strength = (reversion_threshold - current_channel.position) / reversion_threshold
                
                confidence = self._calculate_reversion_confidence(
                    'up', patterns, current_channel, volume
                )
                
                probability = self._calculate_mean_reversion_probability(
                    current_price, current_channel, {'current_regime': self.current_regime}
                )
                
                target_price = current_channel.middle_line
                stop_loss = current_channel.lower_band * 0.99
                risk_reward = (target_price - current_price) / (current_price - stop_loss) if current_price > stop_loss else 1.0
                
                signal = ChannelSignal(
                    timestamp=datetime.now(),
                    signal_type=SignalType.MEAN_REVERSION_UP,
                    entry_price=current_price,
                    upper_band=current_channel.upper_band,
                    lower_band=current_channel.lower_band,
                    middle_line=current_channel.middle_line,
                    channel_width=current_channel.width,
                    position_in_channel=current_channel.position,
                    strength=strength,
                    confidence=confidence,
                    probability=probability,
                    statistical_significance=self._calculate_statistical_significance(
                        0.5 - current_channel.position, current_channel
                    ),
                    volume_confirmation=self._check_volume_confirmation(volume, 'reversion'),
                    target_price=target_price,
                    stop_loss=stop_loss,
                    risk_reward_ratio=risk_reward
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating mean reversion signals: {e}")
            return []
    
    def _generate_squeeze_signals(self, current_channel: ChannelData, 
                                patterns: List[ChannelPattern],
                                close: np.ndarray, volume: np.ndarray = None) -> List[ChannelSignal]:
        """Generate signals based on channel squeeze patterns."""
        try:
            signals = []
            
            # Check for squeeze patterns
            squeeze_patterns = [p for p in patterns if p.pattern_type == "squeeze"]
            
            if squeeze_patterns:
                latest_squeeze = squeeze_patterns[-1]
                
                # Check if squeeze is ending (potential breakout setup)
                if latest_squeeze.significance > 0.6:
                    
                    current_price = close[-1]
                    
                    # Generate squeeze entry signal
                    strength = latest_squeeze.significance
                    confidence = latest_squeeze.breakout_probability
                    probability = confidence
                    
                    # Targets based on historical volatility expansion
                    avg_expansion = self._calculate_average_post_squeeze_expansion()
                    target_up = current_price * (1 + avg_expansion)
                    target_down = current_price * (1 - avg_expansion)
                    
                    signal = ChannelSignal(
                        timestamp=datetime.now(),
                        signal_type=SignalType.SQUEEZE_ENTRY,
                        entry_price=current_price,
                        upper_band=current_channel.upper_band,
                        lower_band=current_channel.lower_band,
                        middle_line=current_channel.middle_line,
                        channel_width=current_channel.width,
                        position_in_channel=current_channel.position,
                        strength=strength,
                        confidence=confidence,
                        probability=probability,
                        statistical_significance=latest_squeeze.significance,
                        volume_confirmation=self._check_volume_confirmation(volume, 'squeeze'),
                        target_price=target_up if current_channel.position > 0.5 else target_down,
                        stop_loss=current_channel.middle_line,
                        risk_reward_ratio=2.0  # Typical for squeeze plays
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating squeeze signals: {e}")
            return []
    
    def _generate_rejection_signals(self, current_channel: ChannelData, 
                                  current_price: float, high: np.ndarray, 
                                  low: np.ndarray, close: np.ndarray, 
                                  volume: np.ndarray = None) -> List[ChannelSignal]:
        """Generate signals based on channel boundary rejections."""
        try:
            signals = []
            
            if len(close) < 3:
                return signals
            
            # Check for upper band rejection
            recent_high = np.max(high[-3:])
            if (recent_high > current_channel.upper_band and 
                current_price < current_channel.upper_band * 0.995):
                
                # Confirmed rejection
                strength = (recent_high - current_channel.upper_band) / current_channel.upper_band
                strength = min(1, strength * 10)
                
                confidence = 0.7 + (strength * 0.3)
                probability = confidence
                
                target_price = current_channel.middle_line
                stop_loss = recent_high
                risk_reward = (current_price - target_price) / (stop_loss - current_price) if stop_loss > current_price else 1.0
                
                signal = ChannelSignal(
                    timestamp=datetime.now(),
                    signal_type=SignalType.CHANNEL_REJECTION,
                    entry_price=current_price,
                    upper_band=current_channel.upper_band,
                    lower_band=current_channel.lower_band,
                    middle_line=current_channel.middle_line,
                    channel_width=current_channel.width,
                    position_in_channel=current_channel.position,
                    strength=strength,
                    confidence=confidence,
                    probability=probability,
                    statistical_significance=strength,
                    volume_confirmation=self._check_volume_confirmation(volume, 'rejection'),
                    target_price=target_price,
                    stop_loss=stop_loss,
                    risk_reward_ratio=risk_reward
                )
                signals.append(signal)
            
            # Check for lower band rejection
            recent_low = np.min(low[-3:])
            if (recent_low < current_channel.lower_band and 
                current_price > current_channel.lower_band * 1.005):
                
                strength = (current_channel.lower_band - recent_low) / current_channel.lower_band
                strength = min(1, strength * 10)
                
                confidence = 0.7 + (strength * 0.3)
                probability = confidence
                
                target_price = current_channel.middle_line
                stop_loss = recent_low
                risk_reward = (target_price - current_price) / (current_price - stop_loss) if current_price > stop_loss else 1.0
                
                signal = ChannelSignal(
                    timestamp=datetime.now(),
                    signal_type=SignalType.CHANNEL_REJECTION,
                    entry_price=current_price,
                    upper_band=current_channel.upper_band,
                    lower_band=current_channel.lower_band,
                    middle_line=current_channel.middle_line,
                    channel_width=current_channel.width,
                    position_in_channel=current_channel.position,
                    strength=strength,
                    confidence=confidence,
                    probability=probability,
                    statistical_significance=strength,
                    volume_confirmation=self._check_volume_confirmation(volume, 'rejection'),
                    target_price=target_price,
                    stop_loss=stop_loss,
                    risk_reward_ratio=risk_reward
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating rejection signals: {e}")
            return []    
    def _calculate_breakout_confidence(self, direction: str, patterns: List[ChannelPattern],
                                     current_channel: ChannelData, volume: np.ndarray = None) -> float:
        """Calculate confidence for breakout signals."""
        try:
            base_confidence = 0.5
            
            # Pattern-based confidence adjustment
            squeeze_patterns = [p for p in patterns if p.pattern_type == "squeeze"]
            if squeeze_patterns:
                latest_squeeze = squeeze_patterns[-1]
                base_confidence += latest_squeeze.significance * 0.3
            
            # Volume confirmation
            if volume is not None and self._check_volume_confirmation(volume, 'breakout'):
                base_confidence += 0.2
            
            # Channel confidence
            base_confidence += current_channel.confidence * 0.2
            
            # Volatility regime adjustment
            if self.current_regime == VolatilityRegime.HIGH:
                base_confidence += 0.1  # High vol favors breakouts
            elif self.current_regime == VolatilityRegime.LOW:
                base_confidence -= 0.1  # Low vol favors mean reversion
            
            return np.clip(base_confidence, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating breakout confidence: {e}")
            return 0.5
    
    def _calculate_reversion_confidence(self, direction: str, patterns: List[ChannelPattern],
                                      current_channel: ChannelData, volume: np.ndarray = None) -> float:
        """Calculate confidence for mean reversion signals."""
        try:
            base_confidence = 0.6  # Mean reversion has higher base probability
            
            # Pattern-based adjustment
            expansion_patterns = [p for p in patterns if p.pattern_type == "expansion"]
            if expansion_patterns:
                latest_expansion = expansion_patterns[-1]
                base_confidence += latest_expansion.significance * 0.2
            
            # Volume confirmation (lower volume often better for mean reversion)
            if volume is not None:
                if not self._check_volume_confirmation(volume, 'breakout'):
                    base_confidence += 0.1
            
            # Channel confidence
            base_confidence += current_channel.confidence * 0.15
            
            # Volatility regime adjustment
            if self.current_regime == VolatilityRegime.LOW:
                base_confidence += 0.15  # Low vol favors mean reversion
            elif self.current_regime == VolatilityRegime.EXTREME:
                base_confidence -= 0.1  # Extreme vol can break mean reversion
            
            return np.clip(base_confidence, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating reversion confidence: {e}")
            return 0.6
    
    def _calculate_breakout_probability(self, current_price: float, current_channel: ChannelData,
                                      direction: str, volatility_analysis: Dict[str, Any]) -> float:
        """Calculate probability of breakout in specified direction."""
        try:
            if not self.is_trained:
                # Use statistical approach
                return self._statistical_breakout_probability(current_price, current_channel, direction)
            
            # Use ML prediction if available
            try:
                features = self._get_current_features()
                if len(features) > 0:
                    features_scaled = self.scaler.transform([features])
                    breakout_probs = self.breakout_classifier.predict_proba(features_scaled)[0]
                    
                    if direction == 'up':
                        return breakout_probs[1] if len(breakout_probs) > 1 else 0.5
                    else:
                        return breakout_probs[0] if len(breakout_probs) > 0 else 0.5
                else:
                    return self._statistical_breakout_probability(current_price, current_channel, direction)
            except:
                return self._statistical_breakout_probability(current_price, current_channel, direction)
                
        except Exception as e:
            self.logger.error(f"Error calculating breakout probability: {e}")
            return 0.5
    
    def _statistical_breakout_probability(self, current_price: float, current_channel: ChannelData,
                                        direction: str) -> float:
        """Calculate breakout probability using statistical methods."""
        try:
            # Base probability based on channel position
            if direction == 'up':
                position_factor = current_channel.position
            else:
                position_factor = 1 - current_channel.position
            
            # Adjust for volatility regime
            regime_multiplier = {
                VolatilityRegime.LOW: 0.7,
                VolatilityRegime.NORMAL: 1.0,
                VolatilityRegime.HIGH: 1.3,
                VolatilityRegime.EXTREME: 1.5
            }
            
            base_prob = position_factor * 0.5  # Max 50% base probability
            regime_adj_prob = base_prob * regime_multiplier[self.current_regime]
            
            # Channel confidence adjustment
            confidence_adj = regime_adj_prob * (0.5 + current_channel.confidence * 0.5)
            
            return np.clip(confidence_adj, 0, 0.9)  # Cap at 90%
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical breakout probability: {e}")
            return 0.3
    
    def _calculate_mean_reversion_probability(self, current_price: float, 
                                            current_channel: ChannelData,
                                            volatility_analysis: Dict[str, Any]) -> float:
        """Calculate probability of mean reversion."""
        try:
            # Base probability inversely related to breakout probability
            breakout_prob_up = self._calculate_breakout_probability(
                current_price, current_channel, 'up', volatility_analysis
            )
            breakout_prob_down = self._calculate_breakout_probability(
                current_price, current_channel, 'down', volatility_analysis
            )
            
            max_breakout_prob = max(breakout_prob_up, breakout_prob_down)
            base_reversion_prob = 1 - max_breakout_prob
            
            # Adjust for extreme positions (higher reversion probability)
            position_factor = 1.0
            if current_channel.position > 0.8 or current_channel.position < 0.2:
                position_factor = 1.2
            
            # Adjust for volatility regime
            regime_adjustment = {
                VolatilityRegime.LOW: 1.2,
                VolatilityRegime.NORMAL: 1.0,
                VolatilityRegime.HIGH: 0.8,
                VolatilityRegime.EXTREME: 0.6
            }
            
            reversion_prob = (base_reversion_prob * position_factor * 
                            regime_adjustment[self.current_regime])
            
            return np.clip(reversion_prob, 0.1, 0.9)
            
        except Exception as e:
            self.logger.error(f"Error calculating mean reversion probability: {e}")
            return 0.6
    
    def _calculate_statistical_significance(self, move_pct: float, 
                                          current_channel: ChannelData) -> float:
        """Calculate statistical significance of a move."""
        try:
            # Use normal distribution to assess significance
            # Assume moves are normally distributed around the mean
            z_score = abs(move_pct) / (current_channel.width / current_channel.middle_line / 2)
            
            # Calculate p-value (two-tailed test)
            p_value = 2 * (1 - stats.norm.cdf(z_score))
            
            # Convert to significance (1 - p_value)
            significance = 1 - p_value
            
            return np.clip(significance, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical significance: {e}")
            return 0.5
    
    def _check_volume_confirmation(self, volume: np.ndarray = None, signal_type: str = 'breakout') -> bool:
        """Check if volume confirms the signal."""
        try:
            if volume is None or len(volume) < 10:
                return False
            
            current_volume = volume[-1]
            avg_volume = np.mean(volume[-10:])
            
            if signal_type == 'breakout':
                # Breakouts should have higher volume
                return current_volume > avg_volume * self.parameters['volume_breakout_multiplier']
            elif signal_type == 'reversion':
                # Mean reversion often happens on lower volume
                return current_volume < avg_volume * 1.2
            elif signal_type == 'squeeze':
                # Squeeze breakouts need volume confirmation
                return current_volume > avg_volume * 1.5
            elif signal_type == 'rejection':
                # Rejections can happen on any volume
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking volume confirmation: {e}")
            return False
    
    def _filter_and_rank_signals(self, signals: List[ChannelSignal]) -> List[ChannelSignal]:
        """Filter and rank signals by quality."""
        try:
            if not signals:
                return signals
            
            # Filter by minimum criteria
            filtered_signals = []
            for signal in signals:
                if (signal.strength >= self.parameters['min_signal_strength'] and
                    signal.confidence >= self.parameters['min_confidence'] and
                    signal.statistical_significance >= self.parameters['min_statistical_significance']):
                    filtered_signals.append(signal)
            
            # Prevent signal clustering if enabled
            if self.parameters['signal_clustering_prevention']:
                filtered_signals = self._prevent_signal_clustering(filtered_signals)
            
            # Rank by composite score
            for signal in filtered_signals:
                signal.composite_score = (
                    signal.strength * 0.3 +
                    signal.confidence * 0.3 +
                    signal.probability * 0.2 +
                    signal.statistical_significance * 0.2
                )
            
            # Sort by composite score
            filtered_signals.sort(key=lambda x: x.composite_score, reverse=True)
            
            # Limit number of signals
            max_signals = self.parameters['max_signals_per_period']
            return filtered_signals[:max_signals]
            
        except Exception as e:
            self.logger.error(f"Error filtering and ranking signals: {e}")
            return signals
    
    def _prevent_signal_clustering(self, signals: List[ChannelSignal]) -> List[ChannelSignal]:
        """Prevent too many signals of the same type in short timeframe."""
        try:
            if len(signals) <= 1:
                return signals
            
            # Group signals by type
            signal_groups = {}
            for signal in signals:
                signal_type = signal.signal_type
                if signal_type not in signal_groups:
                    signal_groups[signal_type] = []
                signal_groups[signal_type].append(signal)
            
            # Keep only the best signal from each type
            filtered_signals = []
            for signal_type, type_signals in signal_groups.items():
                # Sort by strength and take the best
                type_signals.sort(key=lambda x: x.strength, reverse=True)
                filtered_signals.append(type_signals[0])
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error preventing signal clustering: {e}")
            return signals
    
    def _perform_statistical_analysis(self, signals: List[ChannelSignal], 
                                    channel_data: List[ChannelData], 
                                    close: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        try:
            analysis = {}
            
            if not signals or not channel_data:
                return {
                    'overall_significance': 0.0,
                    'signal_quality': 0.0,
                    'confidence_width': 0.0,
                    'normality_test': 0.5
                }
            
            # Overall signal significance
            if signals:
                avg_significance = np.mean([s.statistical_significance for s in signals])
                analysis['overall_significance'] = avg_significance
            else:
                analysis['overall_significance'] = 0.0
            
            # Signal quality score
            if signals:
                quality_scores = []
                for signal in signals:
                    quality = (signal.strength + signal.confidence + signal.probability) / 3
                    quality_scores.append(quality)
                analysis['signal_quality'] = np.mean(quality_scores)
            else:
                analysis['signal_quality'] = 0.0
            
            # Channel confidence interval width
            if channel_data:
                widths = [ch.width / ch.middle_line for ch in channel_data if ch.middle_line > 0]
                if widths:
                    analysis['confidence_width'] = np.mean(widths)
                else:
                    analysis['confidence_width'] = 0.0
            else:
                analysis['confidence_width'] = 0.0
            
            # Test price distribution normality
            if len(close) >= 20:
                returns = np.diff(np.log(close))
                if len(returns) > 8:  # Minimum for Jarque-Bera test
                    _, p_value = stats.jarque_bera(returns)
                    analysis['normality_test'] = p_value
                else:
                    analysis['normality_test'] = 0.5
            else:
                analysis['normality_test'] = 0.5
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error performing statistical analysis: {e}")
            return {
                'overall_significance': 0.0,
                'signal_quality': 0.0,
                'confidence_width': 0.0,
                'normality_test': 0.5
            }
    
    def _calculate_timeframe_consensus(self, high: np.ndarray, low: np.ndarray, 
                                     close: np.ndarray, volume: np.ndarray = None) -> Dict[str, Any]:
        """Calculate consensus across multiple timeframes."""
        try:
            timeframes = self.parameters['timeframes']
            weights = self.parameters['timeframe_weights']
            
            if len(timeframes) == 1:
                return {'consensus_strength': 1.0, 'consensus_direction': 0, 'agreement_score': 1.0}
            
            consensus_signals = []
            
            # Analyze each timeframe
            for i, timeframe in enumerate(timeframes):
                if timeframe == 'short':
                    tf_period = max(5, self.parameters['period'] // 2)
                elif timeframe == 'medium':
                    tf_period = self.parameters['period']
                elif timeframe == 'long':
                    tf_period = min(len(close) // 2, self.parameters['period'] * 2)
                else:
                    tf_period = self.parameters['period']
                
                # Calculate channel for this timeframe
                if len(close) >= tf_period:
                    tf_params = self.parameters.copy()
                    tf_params['period'] = tf_period
                    
                    # Simplified channel calculation for timeframe analysis
                    tf_channels = self._calculate_simple_channel(high, low, close, tf_params)
                    
                    if tf_channels:
                        latest_channel = tf_channels[-1]
                        
                        # Determine signal direction
                        if latest_channel.position > 0.7:
                            signal_direction = 1  # Bullish
                        elif latest_channel.position < 0.3:
                            signal_direction = -1  # Bearish
                        else:
                            signal_direction = 0  # Neutral
                        
                        consensus_signals.append({
                            'direction': signal_direction,
                            'strength': abs(latest_channel.position - 0.5) * 2,
                            'weight': weights[i] if i < len(weights) else 1/len(timeframes)
                        })
            
            if not consensus_signals:
                return {'consensus_strength': 0.0, 'consensus_direction': 0, 'agreement_score': 0.0}
            
            # Calculate weighted consensus
            weighted_direction = sum(sig['direction'] * sig['weight'] for sig in consensus_signals)
            weighted_strength = sum(sig['strength'] * sig['weight'] for sig in consensus_signals)
            
            # Calculate agreement score
            directions = [sig['direction'] for sig in consensus_signals]
            agreement_score = 1.0 - (len(set(directions)) - 1) / 2  # Penalize disagreement
            
            return {
                'consensus_strength': weighted_strength,
                'consensus_direction': weighted_direction,
                'agreement_score': agreement_score
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating timeframe consensus: {e}")
            return {'consensus_strength': 0.0, 'consensus_direction': 0, 'agreement_score': 0.0}
    
    def _calculate_simple_channel(self, high: np.ndarray, low: np.ndarray, 
                                close: np.ndarray, params: Dict[str, Any]) -> List[ChannelData]:
        """Calculate simple channel for timeframe analysis."""
        try:
            period = params['period']
            std_mult = params['std_multiplier']
            
            if len(close) < period:
                return []
            
            channels = []
            for i in range(period, len(close)):
                window_close = close[i-period:i]
                
                middle_line = np.mean(window_close)
                std_dev = np.std(window_close, ddof=1)
                
                upper_band = middle_line + (std_dev * std_mult)
                lower_band = middle_line - (std_dev * std_mult)
                width = upper_band - lower_band
                position = (close[i] - lower_band) / width if width > 0 else 0.5
                
                channel = ChannelData(
                    upper_band=upper_band,
                    lower_band=lower_band,
                    middle_line=middle_line,
                    width=width,
                    position=np.clip(position, 0, 1),
                    channel_type=ChannelType.BOLLINGER,
                    confidence=0.5,
                    volatility_adjustment=1.0
                )
                channels.append(channel)
            
            return channels
            
        except Exception as e:
            self.logger.error(f"Error calculating simple channel: {e}")
            return []    
    def _assess_current_position(self, current_price: float, channel_data: List[ChannelData], 
                               patterns: List[ChannelPattern]) -> Dict[str, Any]:
        """Assess current market position within channels."""
        try:
            if not channel_data:
                return {
                    'position_pct': 50.0,
                    'bias': 'neutral',
                    'zone': 'middle',
                    'action_suggestion': 'wait'
                }
            
            current_channel = channel_data[-1]
            position_pct = current_channel.position * 100
            
            # Determine zone
            if position_pct >= 80:
                zone = 'overbought'
                bias = 'bearish'
                action_suggestion = 'consider_sell'
            elif position_pct >= 60:
                zone = 'upper'
                bias = 'neutral_bearish'
                action_suggestion = 'watch_for_reversal'
            elif position_pct <= 20:
                zone = 'oversold'
                bias = 'bullish'
                action_suggestion = 'consider_buy'
            elif position_pct <= 40:
                zone = 'lower'
                bias = 'neutral_bullish'
                action_suggestion = 'watch_for_reversal'
            else:
                zone = 'middle'
                bias = 'neutral'
                action_suggestion = 'wait_for_setup'
            
            # Adjust based on patterns
            squeeze_patterns = [p for p in patterns if p.pattern_type == "squeeze"]
            if squeeze_patterns and squeeze_patterns[-1].significance > 0.6:
                action_suggestion = 'prepare_for_breakout'
            
            return {
                'position_pct': position_pct,
                'bias': bias,
                'zone': zone,
                'action_suggestion': action_suggestion,
                'channel_strength': current_channel.confidence,
                'volatility_level': self.current_regime.value
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing current position: {e}")
            return {
                'position_pct': 50.0,
                'bias': 'neutral',
                'zone': 'middle',
                'action_suggestion': 'wait'
            }
    
    def _calculate_channel_slope(self, channel_data: List[ChannelData]) -> float:
        """Calculate the slope of the channel middle line."""
        try:
            if len(channel_data) < 10:
                return 0.0
            
            recent_channels = channel_data[-10:]
            middle_lines = [ch.middle_line for ch in recent_channels]
            
            x = np.arange(len(middle_lines))
            slope, _, _, _, _ = stats.linregress(x, middle_lines)
            
            # Normalize slope as percentage
            avg_price = np.mean(middle_lines)
            slope_pct = (slope / avg_price) * 100 if avg_price > 0 else 0
            
            return slope_pct
            
        except Exception as e:
            self.logger.error(f"Error calculating channel slope: {e}")
            return 0.0
    
    def _calculate_channel_stability(self, channel_data: List[ChannelData]) -> float:
        """Calculate channel stability (consistency of width and position)."""
        try:
            if len(channel_data) < 5:
                return 0.5
            
            recent_channels = channel_data[-10:]
            
            # Width stability
            widths = [ch.width for ch in recent_channels]
            width_cv = np.std(widths) / (np.mean(widths) + 1e-10)  # Coefficient of variation
            
            # Position stability
            positions = [ch.position for ch in recent_channels]
            position_std = np.std(positions)
            
            # Combine metrics (lower values = higher stability)
            stability = 1 / (1 + width_cv + position_std)
            
            return np.clip(stability, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating channel stability: {e}")
            return 0.5
    
    def _calculate_squeeze_score(self, channel_data: List[ChannelData], 
                               volatility_analysis: Dict[str, Any]) -> float:
        """Calculate current squeeze intensity score."""
        try:
            if len(channel_data) < 10:
                return 0.0
            
            current_width = channel_data[-1].width
            
            # Compare to historical average
            recent_widths = [ch.width for ch in channel_data[-20:]]
            avg_width = np.mean(recent_widths)
            
            if avg_width > 0:
                width_ratio = current_width / avg_width
                squeeze_score = max(0, 1 - width_ratio)  # Lower width = higher squeeze
            else:
                squeeze_score = 0.0
            
            # Adjust for volatility regime
            if self.current_regime == VolatilityRegime.LOW:
                squeeze_score *= 1.2  # Low vol enhances squeeze
            
            return np.clip(squeeze_score, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating squeeze score: {e}")
            return 0.0
    
    def _calculate_expansion_score(self, channel_data: List[ChannelData], 
                                 volatility_analysis: Dict[str, Any]) -> float:
        """Calculate current expansion intensity score."""
        try:
            if len(channel_data) < 10:
                return 0.0
            
            current_width = channel_data[-1].width
            
            # Compare to historical average
            recent_widths = [ch.width for ch in channel_data[-20:]]
            avg_width = np.mean(recent_widths)
            
            if avg_width > 0:
                width_ratio = current_width / avg_width
                expansion_score = max(0, width_ratio - 1)  # Higher width = higher expansion
            else:
                expansion_score = 0.0
            
            # Adjust for volatility regime
            if self.current_regime == VolatilityRegime.HIGH:
                expansion_score *= 1.2  # High vol enhances expansion
            
            return np.clip(expansion_score, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating expansion score: {e}")
            return 0.0
    
    def _calculate_risk_level(self, current_channel: ChannelData, 
                            volatility_analysis: Dict[str, Any]) -> str:
        """Calculate current risk level."""
        try:
            # Base risk on volatility regime
            regime_risk = {
                VolatilityRegime.LOW: 'low',
                VolatilityRegime.NORMAL: 'medium',
                VolatilityRegime.HIGH: 'high',
                VolatilityRegime.EXTREME: 'very_high'
            }
            
            base_risk = regime_risk[self.current_regime]
            
            # Adjust for channel position
            if current_channel.position > 0.8 or current_channel.position < 0.2:
                if base_risk == 'low':
                    base_risk = 'medium'
                elif base_risk == 'medium':
                    base_risk = 'high'
            
            # Adjust for channel confidence
            if current_channel.confidence < 0.3:
                if base_risk == 'low':
                    base_risk = 'medium'
                elif base_risk == 'medium':
                    base_risk = 'high'
            
            return base_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating risk level: {e}")
            return 'medium'
    
    def _calculate_position_size(self, signals: List[ChannelSignal], 
                               volatility_analysis: Dict[str, Any]) -> float:
        """Calculate recommended position size based on signals and risk."""
        try:
            if not signals:
                return 0.0
            
            # Base position size on signal quality
            best_signal = max(signals, key=lambda s: s.confidence)
            
            base_size = best_signal.confidence * 0.5  # Max 50% of capital
            
            # Adjust for volatility regime
            regime_adjustment = {
                VolatilityRegime.LOW: 1.2,
                VolatilityRegime.NORMAL: 1.0,
                VolatilityRegime.HIGH: 0.8,
                VolatilityRegime.EXTREME: 0.5
            }
            
            adjusted_size = base_size * regime_adjustment[self.current_regime]
            
            # Adjust for risk-reward ratio
            if best_signal.risk_reward_ratio > 2.0:
                adjusted_size *= 1.2
            elif best_signal.risk_reward_ratio < 1.0:
                adjusted_size *= 0.5
            
            return np.clip(adjusted_size, 0, self.parameters['max_risk_per_trade'])
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _calculate_max_favorable_excursion(self, close: np.ndarray, 
                                         current_channel: ChannelData) -> float:
        """Calculate maximum favorable excursion potential."""
        try:
            current_price = close[-1]
            
            # Calculate potential upward move to upper band
            upward_potential = (current_channel.upper_band - current_price) / current_price
            
            # Calculate potential downward move to lower band
            downward_potential = (current_price - current_channel.lower_band) / current_price
            
            # Return the maximum potential
            return max(upward_potential, downward_potential)
            
        except Exception as e:
            self.logger.error(f"Error calculating max favorable excursion: {e}")
            return 0.0
    
    def _calculate_max_adverse_excursion(self, close: np.ndarray, 
                                       current_channel: ChannelData) -> float:
        """Calculate maximum adverse excursion risk."""
        try:
            current_price = close[-1]
            
            # Calculate risk to middle line (typical stop level)
            risk_to_middle = abs(current_price - current_channel.middle_line) / current_price
            
            # Adjust for volatility
            volatility_multiplier = {
                VolatilityRegime.LOW: 0.8,
                VolatilityRegime.NORMAL: 1.0,
                VolatilityRegime.HIGH: 1.3,
                VolatilityRegime.EXTREME: 1.5
            }
            
            adjusted_risk = risk_to_middle * volatility_multiplier[self.current_regime]
            
            return adjusted_risk
            
        except Exception as e:
            self.logger.error(f"Error calculating max adverse excursion: {e}")
            return 0.02  # Default 2% risk
    
    def _calculate_average_post_squeeze_expansion(self) -> float:
        """Calculate average expansion after squeeze patterns."""
        try:
            # Historical average - this would be calculated from backtesting
            # For now, use regime-based estimates
            regime_expansions = {
                VolatilityRegime.LOW: 0.015,      # 1.5%
                VolatilityRegime.NORMAL: 0.025,   # 2.5%
                VolatilityRegime.HIGH: 0.035,     # 3.5%
                VolatilityRegime.EXTREME: 0.050   # 5.0%
            }
            
            return regime_expansions[self.current_regime]
            
        except Exception as e:
            self.logger.error(f"Error calculating average post-squeeze expansion: {e}")
            return 0.025
    
    def _get_current_features(self) -> np.ndarray:
        """Get current features for ML prediction."""
        try:
            if len(self.price_history) < 50:
                return np.array([])
            
            # Use recent price data to extract features
            recent_prices = np.array(self.price_history[-50:])
            
            # Simple feature extraction
            returns = np.diff(np.log(recent_prices))
            
            features = [
                np.mean(returns[-10:]),  # Recent return
                np.std(returns[-10:]),   # Recent volatility
                (recent_prices[-1] / np.mean(recent_prices[-20:]) - 1),  # Price vs MA
                np.mean(returns[-5:]) / (np.std(returns[-20:]) + 1e-10),  # Risk-adjusted return
                len([r for r in returns[-10:] if r > 0]) / 10  # Win rate
            ]
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error getting current features: {e}")
            return np.array([])
    
    def _get_ml_prediction_confidence(self) -> float:
        """Get confidence score for ML predictions."""
        try:
            if not self.is_trained:
                return 0.0
            
            # Use cross-validation score as confidence measure
            features = self._get_current_features()
            if len(features) == 0:
                return 0.0
            
            # Return stored training accuracy as confidence proxy
            return self.statistical_metrics.get('breakout_accuracy', 0.0)
            
        except Exception as e:
            self.logger.error(f"Error getting ML prediction confidence: {e}")
            return 0.0
    
    def _get_feature_importance_score(self) -> float:
        """Get feature importance score for current prediction."""
        try:
            if not self.is_trained or not hasattr(self.breakout_classifier, 'feature_importances_'):
                return 0.0
            
            # Return max feature importance as indicator of model confidence
            max_importance = np.max(self.breakout_classifier.feature_importances_)
            return max_importance
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance score: {e}")
            return 0.0
    
    def _get_active_parameters(self) -> Dict[str, Any]:
        """Get currently active parameters."""
        return {
            'period': self.parameters['period'],
            'std_multiplier': self.parameters['std_multiplier'],
            'channel_type': self.parameters['channel_type'].value,
            'volatility_regime': self.current_regime.value,
            'ml_enabled': self.parameters['ml_enabled'],
            'ml_trained': self.is_trained
        }
    
    def _assess_data_quality(self, high: np.ndarray, low: np.ndarray, 
                           close: np.ndarray, volume: np.ndarray = None) -> float:
        """Assess quality of input data."""
        try:
            quality_score = 1.0
            
            # Check for data completeness
            if len(close) < self.parameters['lookback_period']:
                quality_score *= 0.5
            
            # Check for extreme values or gaps
            if len(close) > 1:
                returns = np.diff(close) / close[:-1]
                extreme_moves = np.sum(np.abs(returns) > 0.1)  # >10% moves
                if extreme_moves > len(returns) * 0.05:  # >5% of data
                    quality_score *= 0.8
            
            # Check for zero or negative prices
            if np.any(close <= 0) or np.any(high <= 0) or np.any(low <= 0):
                quality_score *= 0.6
            
            # Check for logical consistency (high >= low)
            if np.any(high < low):
                quality_score *= 0.7
            
            # Check volume data quality if provided
            if volume is not None:
                if np.any(volume < 0):
                    quality_score *= 0.9
                if np.sum(volume == 0) > len(volume) * 0.1:  # >10% zero volume
                    quality_score *= 0.9
            
            return np.clip(quality_score, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {e}")
            return 0.5
    
    def _signal_to_dict(self, signal: ChannelSignal) -> Dict[str, Any]:
        """Convert signal to dictionary format."""
        return {
            'timestamp': signal.timestamp.isoformat(),
            'signal_type': signal.signal_type.value,
            'entry_price': signal.entry_price,
            'target_price': signal.target_price,
            'stop_loss': signal.stop_loss,
            'strength': signal.strength,
            'confidence': signal.confidence,
            'probability': signal.probability,
            'risk_reward_ratio': signal.risk_reward_ratio,
            'volume_confirmation': signal.volume_confirmation
        }
    
    def _pattern_to_dict(self, pattern: ChannelPattern) -> Dict[str, Any]:
        """Convert pattern to dictionary format."""
        return {
            'pattern_type': pattern.pattern_type,
            'start_time': pattern.start_time.isoformat(),
            'duration_periods': pattern.duration_periods,
            'significance': pattern.significance,
            'breakout_probability': pattern.breakout_probability,
            'mean_reversion_probability': pattern.mean_reversion_probability
        }
    
    def _get_strongest_signal(self, signals: List[ChannelSignal]) -> Optional[Dict[str, Any]]:
        """Get the strongest signal from the list."""
        if not signals:
            return None
        
        strongest = max(signals, key=lambda s: s.strength)
        return self._signal_to_dict(strongest)
    
    def _get_dominant_pattern(self, patterns: List[ChannelPattern]) -> Optional[Dict[str, Any]]:
        """Get the dominant pattern from the list."""
        if not patterns:
            return None
        
        dominant = max(patterns, key=lambda p: p.significance)
        return self._pattern_to_dict(dominant)
    
    def _update_performance_metrics(self, signals: List[ChannelSignal]):
        """Update performance tracking metrics."""
        try:
            self.performance_metrics['total_signals'] += len(signals)
            
            # This would be updated based on actual trade outcomes
            # For now, we'll use signal quality as a proxy
            for signal in signals:
                if signal.confidence > 0.7:
                    self.performance_metrics['successful_signals'] += 1
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data to manage memory."""
        try:
            max_history = 1000
            
            if len(self.channel_history) > max_history:
                self.channel_history = self.channel_history[-max_history:]
            
            if len(self.signal_history) > max_history:
                self.signal_history = self.signal_history[-max_history:]
            
            if len(self.pattern_history) > max_history:
                self.pattern_history = self.pattern_history[-max_history:]
            
            if len(self.regime_history) > max_history:
                self.regime_history = self.regime_history[-max_history:]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def _generate_default_result(self) -> Dict[str, Any]:
        """Generate default result when insufficient data."""
        return {
            'upper_band': 0.0,
            'lower_band': 0.0,
            'middle_line': 0.0,
            'channel_width': 0.0,
            'position_in_channel': 0.5,
            'channel_type': 'standard_deviation',
            'current_price': 0.0,
            'active_signals': [],
            'signal_count': 0,
            'volatility_regime': 'normal',
            'ml_enabled': False,
            'error': 'Insufficient data for analysis'
        }
    
    def _generate_error_result(self, error_message: str) -> Dict[str, Any]:
        """Generate error result."""
        return {
            'error': error_message,
            'upper_band': 0.0,
            'lower_band': 0.0,
            'middle_line': 0.0,
            'channel_width': 0.0,
            'position_in_channel': 0.5,
            'signal_count': 0,
            'ml_enabled': False
        }


# Helper classes for modular design
class ChannelCalculator:
    """Helper class for channel calculations."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
    
    def calculate_multiple_channels(self, high: np.ndarray, low: np.ndarray, 
                                  close: np.ndarray) -> Dict[str, List[ChannelData]]:
        """Calculate multiple channel types for comparison."""
        # Implementation would go here
        pass


class VolatilityAnalyzer:
    """Helper class for volatility analysis."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
    
    def analyze_regime_changes(self, price_data: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility regime changes."""
        # Implementation would go here
        pass


class PatternRecognizer:
    """Helper class for pattern recognition."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
    
    def recognize_complex_patterns(self, channel_data: List[ChannelData]) -> List[ChannelPattern]:
        """Recognize complex channel patterns."""
        # Implementation would go here
        pass


class SignalClassifier:
    """Helper class for signal classification."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
    
    def classify_signal_quality(self, signal: ChannelSignal) -> float:
        """Classify signal quality score."""
        # Implementation would go here
        pass


class BreakoutDetector:
    """Helper class for breakout detection."""
    
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
    
    def detect_false_breakouts(self, price_data: np.ndarray, 
                             channel_data: List[ChannelData]) -> List[bool]:
        """Detect false breakouts."""
        # Implementation would go here
        pass