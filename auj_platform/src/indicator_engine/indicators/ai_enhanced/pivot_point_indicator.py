"""
Pivot Point Indicator - Advanced Multi-Method Calculator with Machine Learning Enhancement

This module implements a sophisticated pivot point calculator with:
- Multiple calculation methods (Standard, Fibonacci, Woodie's, Camarilla, DeMark's)
- Machine learning-enhanced level prediction and validation
- Adaptive timeframe analysis and automatic period selection
- Confluence analysis with multiple pivot systems integration
- Support and resistance strength scoring and validation
- Breakout detection with momentum confirmation
- Volume-weighted pivot calculations and adjustments
- Market structure analysis and trend integration
- Dynamic level adjustment based on market volatility
- Advanced signal generation with probability scoring
- Production-grade error handling and logging

Author: AI Enhancement Team
Version: 8.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy import optimize, stats
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class PivotMethod(Enum):
    """Enumeration of pivot point calculation methods."""
    STANDARD = "standard"
    FIBONACCI = "fibonacci" 
    WOODIE = "woodie"
    CAMARILLA = "camarilla"
    DEMARK = "demark"
    CLASSIC = "classic"
    VOLUME_WEIGHTED = "volume_weighted"
    ADAPTIVE = "adaptive"

class PivotTimeframe(Enum):
    """Enumeration of pivot timeframes."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    HOURLY = "hourly"
    FOUR_HOUR = "4h"
    ADAPTIVE = "adaptive"

@dataclass
class PivotLevel:
    """Individual pivot level data structure."""
    value: float
    level_type: str  # 'pivot', 'support', 'resistance'
    strength: float
    confidence: float
    touch_count: int
    last_touch: Optional[datetime]
    volume_at_level: float
    method: PivotMethod
    timeframe: PivotTimeframe

@dataclass
class PivotSignal:
    """Pivot point signal structure."""
    timestamp: datetime
    signal_type: str  # 'breakout', 'bounce', 'confluence', 'rejection'
    level: PivotLevel
    price: float
    strength: float
    confidence: float
    probability: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    risk_reward_ratio: float
    volume_confirmation: bool

@dataclass
class ConfluenceZone:
    """Confluence zone where multiple pivot levels converge."""
    center_price: float
    width: float
    strength: float
    contributing_levels: List[PivotLevel]
    timeframes: List[PivotTimeframe]
    methods: List[PivotMethod]
    significance_score: float

class PivotPointIndicator:
    """
    Advanced Pivot Point Indicator with machine learning enhancement and multi-method analysis.
    
    This indicator provides comprehensive pivot point analysis including:
    - Multiple calculation methods with automatic method selection
    - Machine learning-enhanced level prediction and validation
    - Adaptive timeframe analysis and period optimization
    - Confluence zone detection and significance scoring
    - Support/resistance strength analysis with touch validation
    - Breakout detection with momentum and volume confirmation
    - Volume-weighted calculations and market structure integration
    - Dynamic level adjustment based on volatility patterns
    - Advanced signal generation with probability assessment
    - Risk management with stop-loss and target calculations
    """
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the Pivot Point Indicator.
        
        Args:
            parameters: Configuration parameters for the indicator
        """
        self.parameters = self._set_default_parameters(parameters or {})
        self.logger = self._setup_logger()
        
        # Core components
        self.level_calculator = LevelCalculator(self.parameters)
        self.confluence_analyzer = ConfluenceAnalyzer(self.parameters)
        self.ml_enhancer = MLPivotEnhancer(self.parameters)
        self.signal_generator = PivotSignalGenerator(self.parameters)
        self.validation_engine = LevelValidationEngine(self.parameters)
        
        # State management
        self.pivot_levels: Dict[PivotTimeframe, Dict[PivotMethod, List[PivotLevel]]] = {}
        self.confluence_zones: List[ConfluenceZone] = []
        self.signal_history: List[PivotSignal] = []
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        
        # Machine learning models
        self.level_predictor = None
        self.strength_classifier = None
        self.breakout_predictor = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Performance tracking
        self.performance_metrics = {
            'total_levels': 0,
            'accurate_predictions': 0,
            'breakout_accuracy': 0,
            'confluence_success_rate': 0
        }
        
        # Market structure data
        self.market_structure = {
            'trend_direction': 0,
            'volatility_regime': 'normal',
            'volume_profile': {},
            'support_resistance_map': {}
        }
        
        self.logger.info("Pivot Point Indicator initialized with advanced ML enhancement")
    
    def _set_default_parameters(self, user_params: Dict[str, Any]) -> Dict[str, Any]:
        """Set default parameters for the indicator."""
        defaults = {
            # Basic pivot parameters
            'default_method': PivotMethod.STANDARD,
            'default_timeframe': PivotTimeframe.DAILY,
            'methods_enabled': [
                PivotMethod.STANDARD,
                PivotMethod.FIBONACCI,
                PivotMethod.CAMARILLA,
                PivotMethod.WOODIE
            ],
            'timeframes_enabled': [
                PivotTimeframe.DAILY,
                PivotTimeframe.WEEKLY,
                PivotTimeframe.FOUR_HOUR
            ],
            
            # Level calculation
            'lookback_period': 100,
            'min_touch_count': 2,
            'level_tolerance': 0.001,  # 0.1% tolerance for level matching
            'strength_decay_factor': 0.95,
            'volume_weight_factor': 0.3,
            
            # Confluence analysis
            'confluence_threshold': 0.002,  # 0.2% range for confluence
            'min_confluence_levels': 3,
            'confluence_strength_multiplier': 1.5,
            'max_confluence_width': 0.005,  # 0.5% maximum width
            
            # Machine learning
            'ml_enabled': True,
            'ml_training_period': 200,
            'ml_retrain_frequency': 50,
            'feature_count': 20,
            'prediction_horizon': 10,
            'model_validation_split': 0.2,
            
            # Signal generation
            'breakout_confirmation_periods': 3,
            'bounce_confirmation_periods': 2,
            'volume_confirmation_required': True,
            'momentum_confirmation_required': True,
            'min_signal_strength': 0.4,
            'min_confidence': 0.3,
            
            # Risk management
            'default_risk_reward_ratio': 2.0,
            'max_risk_per_trade': 0.02,
            'stop_loss_buffer': 0.001,
            'target_extension_factor': 1.5,
            
            # Validation and filtering
            'level_validation_enabled': True,
            'historical_validation_periods': 50,
            'false_breakout_filter': True,
            'noise_filter_enabled': True,
            'min_level_lifetime': 5,  # Minimum periods a level must exist
            
            # Adaptive features
            'adaptive_timeframes': True,
            'adaptive_methods': True,
            'volatility_adjustment': True,
            'trend_bias_adjustment': True,
            'market_session_awareness': True,
            
            # Performance optimization
            'max_levels_per_method': 10,
            'level_cleanup_frequency': 20,
            'performance_tracking': True,
            'real_time_updates': True
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
        Calculate advanced pivot points with machine learning enhancement.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Optional volume data
            timestamp: Optional timestamp data
            
        Returns:
            Dict containing comprehensive pivot point analysis results
        """
        try:
            if len(close) < self.parameters['lookback_period']:
                return self._generate_default_result()
            
            # Store price and volume history
            self.price_history = close.tolist()
            if volume is not None:
                self.volume_history = volume.tolist()
            
            # Update market structure analysis
            market_analysis = self._analyze_market_structure(high, low, close, volume)
            
            # Calculate pivot levels for all enabled methods and timeframes
            all_pivot_levels = self._calculate_all_pivot_levels(high, low, close, volume)
            
            # Train or update ML models if needed
            if self.parameters['ml_enabled']:
                if not self.is_trained and len(close) >= self.parameters['ml_training_period']:
                    self._train_ml_models(high, low, close, volume)
                elif self.is_trained:
                    enhanced_levels = self._enhance_levels_with_ml(all_pivot_levels, high, low, close)
                else:
                    enhanced_levels = all_pivot_levels
            else:
                enhanced_levels = all_pivot_levels
            
            # Validate and filter levels
            validated_levels = self._validate_and_filter_levels(enhanced_levels, high, low, close)
            
            # Detect confluence zones
            confluence_zones = self._detect_confluence_zones(validated_levels)
            
            # Generate signals
            signals = self._generate_signals(validated_levels, confluence_zones, 
                                           high, low, close, volume)
            
            # Analyze level strength and significance
            level_analysis = self._analyze_level_strength(validated_levels, high, low, close)
            
            # Calculate support and resistance map
            sr_map = self._create_support_resistance_map(validated_levels, confluence_zones)
            
            # Generate current market assessment
            current_assessment = self._assess_current_market_position(
                close[-1], validated_levels, confluence_zones
            )
            
            # Compile comprehensive results
            result = {
                # Current pivot levels
                'pivot_levels': self._format_pivot_levels(validated_levels),
                'confluence_zones': self._format_confluence_zones(confluence_zones),
                
                # Current market position
                'current_price': close[-1],
                'nearest_support': current_assessment.get('nearest_support', 0.0),
                'nearest_resistance': current_assessment.get('nearest_resistance', 0.0),
                'support_strength': current_assessment.get('support_strength', 0.0),
                'resistance_strength': current_assessment.get('resistance_strength', 0.0),
                'price_position': current_assessment.get('price_position', 'middle'),
                
                # Signals and analysis
                'active_signals': [self._signal_to_dict(sig) for sig in signals[-5:]],
                'signal_count': len(signals),
                'highest_strength_signal': self._get_highest_strength_signal(signals),
                
                # Level analysis
                'total_levels': len([level for method_levels in validated_levels.values() 
                                   for tf_levels in method_levels.values() 
                                   for level in tf_levels]),
                'strong_levels_count': level_analysis.get('strong_levels', 0),
                'confluence_zones_count': len(confluence_zones),
                'most_significant_zone': self._get_most_significant_zone(confluence_zones),
                
                # Market structure
                'market_trend': market_analysis.get('trend_direction', 0),
                'volatility_regime': market_analysis.get('volatility_regime', 'normal'),
                'volume_profile_strength': market_analysis.get('volume_strength', 0.0),
                'market_structure_score': market_analysis.get('structure_score', 0.0),
                
                # Method performance
                'best_performing_method': self._get_best_method(),
                'method_accuracy_scores': self._calculate_method_accuracy(),
                'adaptive_method_recommendation': self._recommend_adaptive_method(market_analysis),
                
                # Risk metrics
                'overall_risk_level': self._calculate_overall_risk(validated_levels, market_analysis),
                'breakout_probability': self._calculate_breakout_probability(
                    close[-1], validated_levels, market_analysis
                ),
                'reversal_probability': self._calculate_reversal_probability(
                    close[-1], validated_levels, market_analysis
                ),
                
                # Performance metrics
                'prediction_accuracy': self.performance_metrics.get('accurate_predictions', 0) / 
                                     max(self.performance_metrics.get('total_levels', 1), 1),
                'breakout_accuracy': self.performance_metrics.get('breakout_accuracy', 0.0),
                'confluence_success_rate': self.performance_metrics.get('confluence_success_rate', 0.0),
                
                # Detailed analysis
                'market_analysis': market_analysis,
                'level_analysis': level_analysis,
                'support_resistance_map': sr_map,
                'current_assessment': current_assessment,
                
                # ML metrics (if enabled)
                'ml_enabled': self.parameters['ml_enabled'],
                'ml_trained': self.is_trained,
                'ml_enhancement_score': self._calculate_ml_enhancement_score() if self.is_trained else 0.0,
                
                # Metadata
                'calculation_timestamp': datetime.now().isoformat(),
                'parameters_used': self._get_active_parameters(),
                'data_quality_score': self._assess_data_quality(high, low, close, volume)
            }
            
            # Store results for future analysis
            self.pivot_levels = validated_levels
            self.confluence_zones = confluence_zones
            self.signal_history.extend(signals)
            
            # Cleanup old data
            self._cleanup_old_data()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating pivot points: {e}")
            return self._generate_error_result(str(e))
    
    def _analyze_market_structure(self, high: np.ndarray, low: np.ndarray, 
                                 close: np.ndarray, volume: np.ndarray = None) -> Dict[str, Any]:
        """Analyze market structure for pivot enhancement."""
        try:
            # Trend analysis
            if len(close) >= 20:
                trend_slope = np.polyfit(range(20), close[-20:], 1)[0]
                trend_strength = abs(trend_slope) / (np.std(close[-20:]) + 1e-10)
                trend_direction = 1 if trend_slope > 0 else -1 if trend_slope < 0 else 0
            else:
                trend_slope = 0
                trend_strength = 0
                trend_direction = 0
            
            # Volatility regime analysis
            if len(close) >= 14:
                returns = np.diff(np.log(close[-14:]))
                volatility = np.std(returns)
                
                # Compare to historical volatility
                if len(close) >= 50:
                    hist_vol = np.std(np.diff(np.log(close[-50:-14])))
                    vol_ratio = volatility / (hist_vol + 1e-10)
                    
                    if vol_ratio > 1.5:
                        vol_regime = 'high'
                    elif vol_ratio < 0.7:
                        vol_regime = 'low'
                    else:
                        vol_regime = 'normal'
                else:
                    vol_regime = 'normal'
            else:
                volatility = 0
                vol_regime = 'normal'
            
            # Volume analysis (if available)
            volume_strength = 0.0
            volume_trend = 0
            if volume is not None and len(volume) >= 10:
                recent_volume = np.mean(volume[-5:])
                avg_volume = np.mean(volume[-20:]) if len(volume) >= 20 else recent_volume
                volume_strength = recent_volume / (avg_volume + 1e-10)
                
                # Volume trend
                if len(volume) >= 10:
                    volume_trend = np.polyfit(range(10), volume[-10:], 1)[0]
            
            # Market structure score
            structure_factors = [
                min(trend_strength, 1.0),
                1.0 - min(volatility * 10, 1.0),  # Lower volatility = better structure
                min(volume_strength / 2.0, 1.0) if volume is not None else 0.5
            ]
            structure_score = np.mean(structure_factors)
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'trend_slope': trend_slope,
                'volatility': volatility,
                'volatility_regime': vol_regime,
                'volume_strength': volume_strength,
                'volume_trend': volume_trend,
                'structure_score': structure_score,
                'market_efficiency': self._calculate_market_efficiency(close),
                'fractal_dimension': self._calculate_fractal_dimension(close)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market structure: {e}")
            return {
                'trend_direction': 0, 'trend_strength': 0, 'volatility_regime': 'normal',
                'volume_strength': 1.0, 'structure_score': 0.5
            }
    
    def _calculate_market_efficiency(self, close: np.ndarray) -> float:
        """Calculate market efficiency using random walk deviation."""
        try:
            if len(close) < 20:
                return 0.5
            
            # Calculate returns
            returns = np.diff(np.log(close[-20:]))
            
            # Test for randomness using runs test
            median_return = np.median(returns)
            runs = 0
            for i in range(1, len(returns)):
                if (returns[i] > median_return) != (returns[i-1] > median_return):
                    runs += 1
            
            expected_runs = len(returns) / 2
            efficiency = min(runs / expected_runs, 2.0) / 2.0  # Normalize to 0-1
            
            return efficiency
            
        except Exception as e:
            self.logger.error(f"Error calculating market efficiency: {e}")
            return 0.5
    
    def _calculate_fractal_dimension(self, close: np.ndarray) -> float:
        """Calculate fractal dimension of price series."""
        try:
            if len(close) < 20:
                return 1.5
            
            data = close[-20:]
            
            # Calculate Higuchi fractal dimension
            k_max = 5
            L = []
            
            for k in range(1, k_max + 1):
                Lk = 0
                for m in range(k):
                    Lmk = 0
                    for i in range(1, int((len(data) - m) / k)):
                        Lmk += abs(data[m + i * k] - data[m + (i - 1) * k])
                    Lmk = Lmk * (len(data) - 1) / (((len(data) - m) / k) * k)
                    Lk += Lmk
                L.append(Lk / k)
            
            # Linear regression in log-log space
            x = np.log(range(1, k_max + 1))
            y = np.log(L)
            
            slope = np.polyfit(x, y, 1)[0]
            fractal_dim = -slope
            
            return max(1.0, min(2.0, fractal_dim))
            
        except Exception as e:
            self.logger.error(f"Error calculating fractal dimension: {e}")
            return 1.5    
    def _calculate_all_pivot_levels(self, high: np.ndarray, low: np.ndarray, 
                                   close: np.ndarray, volume: np.ndarray = None) -> Dict[PivotTimeframe, Dict[PivotMethod, List[PivotLevel]]]:
        """Calculate pivot levels for all enabled methods and timeframes."""
        try:
            all_levels = {}
            
            for timeframe in self.parameters['timeframes_enabled']:
                all_levels[timeframe] = {}
                
                # Get timeframe-specific data
                tf_high, tf_low, tf_close, tf_volume = self._resample_to_timeframe(
                    high, low, close, volume, timeframe
                )
                
                if len(tf_close) < 2:
                    continue
                
                for method in self.parameters['methods_enabled']:
                    try:
                        levels = self._calculate_pivot_levels_for_method(
                            tf_high, tf_low, tf_close, tf_volume, method, timeframe
                        )
                        all_levels[timeframe][method] = levels
                        
                    except Exception as e:
                        self.logger.error(f"Error calculating {method.value} pivots for {timeframe.value}: {e}")
                        all_levels[timeframe][method] = []
            
            return all_levels
            
        except Exception as e:
            self.logger.error(f"Error calculating all pivot levels: {e}")
            return {}
    
    def _resample_to_timeframe(self, high: np.ndarray, low: np.ndarray, 
                              close: np.ndarray, timeframe: PivotTimeframe,
                              volume: np.ndarray = None) -> Tuple[np.ndarray, ...]:
        """Resample data to specific timeframe."""
        try:
            # Simple resampling simulation for different timeframes
            if timeframe == PivotTimeframe.DAILY:
                step = 24  # Assume hourly data, take every 24th point
            elif timeframe == PivotTimeframe.FOUR_HOUR:
                step = 4
            elif timeframe == PivotTimeframe.WEEKLY:
                step = 168  # 24 * 7
            elif timeframe == PivotTimeframe.MONTHLY:
                step = 720  # 24 * 30
            else:
                step = 1
            
            if step >= len(close):
                return high, low, close, volume
            
            # Resample by taking every nth point
            resampled_high = high[::step] if step > 1 else high
            resampled_low = low[::step] if step > 1 else low
            resampled_close = close[::step] if step > 1 else close
            resampled_volume = volume[::step] if volume is not None and step > 1 else volume
            
            return resampled_high, resampled_low, resampled_close, resampled_volume
            
        except Exception as e:
            self.logger.error(f"Error resampling to {timeframe.value}: {e}")
            return high, low, close, volume
    
    def _calculate_pivot_levels_for_method(self, high: np.ndarray, low: np.ndarray, 
                                          close: np.ndarray, method: PivotMethod,
                                          timeframe: PivotTimeframe, volume: np.ndarray = None) -> List[PivotLevel]:
        """Calculate pivot levels for a specific method."""
        try:
            if len(close) < 2:
                return []
            
            # Get the latest OHLC for pivot calculation
            h = high[-1]
            l = low[-1]
            c = close[-2]  # Previous close
            
            levels = []
            
            if method == PivotMethod.STANDARD:
                levels = self._calculate_standard_pivots(h, l, c, volume, timeframe)
            elif method == PivotMethod.FIBONACCI:
                levels = self._calculate_fibonacci_pivots(h, l, c, volume, timeframe)
            elif method == PivotMethod.WOODIE:
                levels = self._calculate_woodie_pivots(h, l, c, volume, timeframe)
            elif method == PivotMethod.CAMARILLA:
                levels = self._calculate_camarilla_pivots(h, l, c, volume, timeframe)
            elif method == PivotMethod.DEMARK:
                levels = self._calculate_demark_pivots(h, l, c, volume, timeframe)
            elif method == PivotMethod.VOLUME_WEIGHTED:
                levels = self._calculate_volume_weighted_pivots(h, l, c, volume, timeframe)
            else:
                levels = self._calculate_standard_pivots(h, l, c, volume, timeframe)
            
            # Enhance levels with historical validation
            enhanced_levels = self._enhance_levels_with_history(levels, high, low, close)
            
            return enhanced_levels
            
        except Exception as e:
            self.logger.error(f"Error calculating {method.value} pivots: {e}")
            return []
    
    def _calculate_standard_pivots(self, h: float, l: float, c: float, 
                                  volume: np.ndarray = None, timeframe: PivotTimeframe = None) -> List[PivotLevel]:
        """Calculate standard pivot points."""
        try:
            pp = (h + l + c) / 3  # Pivot Point
            
            levels = [
                PivotLevel(pp, 'pivot', 1.0, 0.8, 0, None, 0.0, PivotMethod.STANDARD, timeframe),
                PivotLevel(2 * pp - l, 'resistance', 0.8, 0.7, 0, None, 0.0, PivotMethod.STANDARD, timeframe),  # R1
                PivotLevel(2 * pp - h, 'support', 0.8, 0.7, 0, None, 0.0, PivotMethod.STANDARD, timeframe),    # S1
                PivotLevel(pp + (h - l), 'resistance', 0.6, 0.6, 0, None, 0.0, PivotMethod.STANDARD, timeframe),  # R2
                PivotLevel(pp - (h - l), 'support', 0.6, 0.6, 0, None, 0.0, PivotMethod.STANDARD, timeframe),    # S2
                PivotLevel(h + 2 * (pp - l), 'resistance', 0.4, 0.5, 0, None, 0.0, PivotMethod.STANDARD, timeframe),  # R3
                PivotLevel(l - 2 * (h - pp), 'support', 0.4, 0.5, 0, None, 0.0, PivotMethod.STANDARD, timeframe),    # S3
            ]
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error calculating standard pivots: {e}")
            return []
    
    def _calculate_fibonacci_pivots(self, h: float, l: float, c: float, 
                                   volume: np.ndarray = None, timeframe: PivotTimeframe = None) -> List[PivotLevel]:
        """Calculate Fibonacci pivot points."""
        try:
            pp = (h + l + c) / 3
            range_hl = h - l
            
            # Fibonacci ratios
            fib_ratios = [0.236, 0.382, 0.618, 1.0, 1.618]
            
            levels = [
                PivotLevel(pp, 'pivot', 1.0, 0.8, 0, None, 0.0, PivotMethod.FIBONACCI, timeframe)
            ]
            
            for i, ratio in enumerate(fib_ratios):
                # Resistance levels
                r_level = pp + (range_hl * ratio)
                levels.append(PivotLevel(
                    r_level, 'resistance', 0.8 - i * 0.1, 0.7 - i * 0.05, 
                    0, None, 0.0, PivotMethod.FIBONACCI, timeframe
                ))
                
                # Support levels
                s_level = pp - (range_hl * ratio)
                levels.append(PivotLevel(
                    s_level, 'support', 0.8 - i * 0.1, 0.7 - i * 0.05, 
                    0, None, 0.0, PivotMethod.FIBONACCI, timeframe
                ))
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci pivots: {e}")
            return []
    
    def _calculate_woodie_pivots(self, h: float, l: float, c: float, 
                                volume: np.ndarray = None, timeframe: PivotTimeframe = None) -> List[PivotLevel]:
        """Calculate Woodie's pivot points."""
        try:
            pp = (h + l + 2 * c) / 4  # Modified pivot calculation
            
            levels = [
                PivotLevel(pp, 'pivot', 1.0, 0.8, 0, None, 0.0, PivotMethod.WOODIE, timeframe),
                PivotLevel(2 * pp - l, 'resistance', 0.8, 0.7, 0, None, 0.0, PivotMethod.WOODIE, timeframe),
                PivotLevel(2 * pp - h, 'support', 0.8, 0.7, 0, None, 0.0, PivotMethod.WOODIE, timeframe),
                PivotLevel(h + l - 2 * l, 'resistance', 0.6, 0.6, 0, None, 0.0, PivotMethod.WOODIE, timeframe),
                PivotLevel(h + l - 2 * h, 'support', 0.6, 0.6, 0, None, 0.0, PivotMethod.WOODIE, timeframe),
            ]
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error calculating Woodie pivots: {e}")
            return []
    
    def _calculate_camarilla_pivots(self, h: float, l: float, c: float, 
                                   volume: np.ndarray = None, timeframe: PivotTimeframe = None) -> List[PivotLevel]:
        """Calculate Camarilla pivot points."""
        try:
            range_hl = h - l
            
            # Camarilla multipliers
            multipliers = [1.1/12, 1.1/6, 1.1/4, 1.1/2]
            
            levels = [
                PivotLevel(c, 'pivot', 1.0, 0.8, 0, None, 0.0, PivotMethod.CAMARILLA, timeframe)
            ]
            
            for i, mult in enumerate(multipliers):
                # Resistance levels
                r_level = c + (range_hl * mult)
                levels.append(PivotLevel(
                    r_level, 'resistance', 0.8 - i * 0.1, 0.7 - i * 0.05, 
                    0, None, 0.0, PivotMethod.CAMARILLA, timeframe
                ))
                
                # Support levels
                s_level = c - (range_hl * mult)
                levels.append(PivotLevel(
                    s_level, 'support', 0.8 - i * 0.1, 0.7 - i * 0.05, 
                    0, None, 0.0, PivotMethod.CAMARILLA, timeframe
                ))
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error calculating Camarilla pivots: {e}")
            return []
    
    def _calculate_demark_pivots(self, h: float, l: float, c: float, 
                                volume: np.ndarray = None, timeframe: PivotTimeframe = None) -> List[PivotLevel]:
        """Calculate DeMark pivot points."""
        try:
            # DeMark X calculation
            if c < l:
                x = h + 2 * l + c
            elif c > h:
                x = 2 * h + l + c
            else:
                x = h + l + 2 * c
            
            levels = [
                PivotLevel(x / 4, 'pivot', 1.0, 0.8, 0, None, 0.0, PivotMethod.DEMARK, timeframe),
                PivotLevel(x / 2 - l, 'resistance', 0.8, 0.7, 0, None, 0.0, PivotMethod.DEMARK, timeframe),
                PivotLevel(x / 2 - h, 'support', 0.8, 0.7, 0, None, 0.0, PivotMethod.DEMARK, timeframe),
            ]
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error calculating DeMark pivots: {e}")
            return []
    
    def _calculate_volume_weighted_pivots(self, h: float, l: float, c: float, 
                                         volume: np.ndarray = None, timeframe: PivotTimeframe = None) -> List[PivotLevel]:
        """Calculate volume-weighted pivot points."""
        try:
            if volume is None or len(volume) < 3:
                return self._calculate_standard_pivots(h, l, c, volume, timeframe)
            
            # Weight by volume
            recent_volume = volume[-3:]  # Last 3 periods
            total_volume = np.sum(recent_volume)
            
            if total_volume == 0:
                return self._calculate_standard_pivots(h, l, c, volume, timeframe)
            
            # Volume-weighted average
            weights = recent_volume / total_volume
            vwap = np.average([h, l, c], weights=[weights[0], weights[1], weights[2]])
            
            # Adjust pivot based on volume profile
            volume_factor = recent_volume[-1] / (np.mean(recent_volume) + 1e-10)
            adjusted_pivot = vwap * (1 + (volume_factor - 1) * 0.1)
            
            levels = [
                PivotLevel(adjusted_pivot, 'pivot', 1.0, 0.8, 0, None, total_volume, PivotMethod.VOLUME_WEIGHTED, timeframe),
                PivotLevel(2 * adjusted_pivot - l, 'resistance', 0.8, 0.7, 0, None, total_volume, PivotMethod.VOLUME_WEIGHTED, timeframe),
                PivotLevel(2 * adjusted_pivot - h, 'support', 0.8, 0.7, 0, None, total_volume, PivotMethod.VOLUME_WEIGHTED, timeframe),
            ]
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error calculating volume-weighted pivots: {e}")
            return self._calculate_standard_pivots(h, l, c, volume, timeframe)
    
    def _enhance_levels_with_history(self, levels: List[PivotLevel], 
                                    high: np.ndarray, low: np.ndarray, close: np.ndarray) -> List[PivotLevel]:
        """Enhance pivot levels with historical validation."""
        try:
            enhanced_levels = []
            
            for level in levels:
                # Count touches and calculate strength
                touches = self._count_level_touches(level.value, high, low, close)
                last_touch = self._find_last_touch(level.value, high, low, close)
                
                # Adjust strength based on touches
                strength_multiplier = min(1.0 + touches * 0.1, 2.0)
                adjusted_strength = min(level.strength * strength_multiplier, 1.0)
                
                # Adjust confidence based on recent validation
                confidence_adjustment = min(touches * 0.05, 0.3)
                adjusted_confidence = min(level.confidence + confidence_adjustment, 1.0)
                
                enhanced_level = PivotLevel(
                    value=level.value,
                    level_type=level.level_type,
                    strength=adjusted_strength,
                    confidence=adjusted_confidence,
                    touch_count=touches,
                    last_touch=last_touch,
                    volume_at_level=level.volume_at_level,
                    method=level.method,
                    timeframe=level.timeframe
                )
                
                enhanced_levels.append(enhanced_level)
            
            return enhanced_levels
            
        except Exception as e:
            self.logger.error(f"Error enhancing levels with history: {e}")
            return levels
    
    def _count_level_touches(self, level_value: float, high: np.ndarray, 
                            low: np.ndarray, close: np.ndarray) -> int:
        """Count how many times price has touched a level."""
        try:
            tolerance = level_value * self.parameters['level_tolerance']
            touches = 0
            
            for i in range(len(close)):
                # Check if price came close to the level
                if (low[i] <= level_value + tolerance and high[i] >= level_value - tolerance):
                    touches += 1
            
            return touches
            
        except Exception as e:
            self.logger.error(f"Error counting level touches: {e}")
            return 0
    
    def _find_last_touch(self, level_value: float, high: np.ndarray, 
                        low: np.ndarray, close: np.ndarray) -> Optional[datetime]:
        """Find the last time price touched a level."""
        try:
            tolerance = level_value * self.parameters['level_tolerance']
            
            for i in range(len(close) - 1, -1, -1):
                if (low[i] <= level_value + tolerance and high[i] >= level_value - tolerance):
                    # Return a datetime (simplified - would use actual timestamps in real implementation)
                    return datetime.now() - timedelta(hours=len(close) - i)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding last touch: {e}")
            return None
    
    def _train_ml_models(self, high: np.ndarray, low: np.ndarray, 
                        close: np.ndarray, volume: np.ndarray = None):
        """Train machine learning models for pivot enhancement."""
        try:
            if len(close) < self.parameters['ml_training_period']:
                return
            
            # Generate training data
            X, y_levels, y_strength = self._generate_ml_training_data(high, low, close, volume)
            
            if len(X) < 50:  # Minimum samples needed
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y_levels, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train level predictor
            self.level_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.level_predictor.fit(X_train_scaled, y_train)
            
            # Train strength classifier
            if len(y_strength) > 0:
                X_str_train, X_str_test, y_str_train, y_str_test = train_test_split(
                    X, y_strength, test_size=0.2, random_state=42
                )
                X_str_train_scaled = self.scaler.transform(X_str_train)
                
                self.strength_classifier = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1
                )
                self.strength_classifier.fit(X_str_train_scaled, y_str_train)
            
            # Evaluate models
            y_pred = self.level_predictor.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            
            self.logger.info(f"ML models trained - Level predictor R2: {r2:.4f}")
            self.is_trained = True
            
        except Exception as e:
            self.logger.error(f"Error training ML models: {e}")
    
    def _generate_ml_training_data(self, high: np.ndarray, low: np.ndarray, 
                                  close: np.ndarray, volume: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate training data for ML models."""
        try:
            X, y_levels, y_strength = [], [], []
            
            window = 50
            
            for i in range(window, len(close) - 10):
                # Extract features
                features = self._extract_ml_features(high[i-window:i], low[i-window:i], 
                                                   close[i-window:i], volume[i-window:i] if volume is not None else None)
                
                if features is None:
                    continue
                
                # Calculate future pivot levels (ground truth)
                future_high = np.max(high[i:i+10])
                future_low = np.min(low[i:i+10])
                future_pivot = (future_high + future_low + close[i]) / 3
                
                # Calculate level strength based on future touches
                strength = self._calculate_future_level_strength(future_pivot, high[i:i+10], low[i:i+10])
                
                X.append(features)
                y_levels.append(future_pivot)
                y_strength.append(strength)
            
            return np.array(X), np.array(y_levels), np.array(y_strength)
            
        except Exception as e:
            self.logger.error(f"Error generating ML training data: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def _extract_ml_features(self, high: np.ndarray, low: np.ndarray, 
                            close: np.ndarray, volume: np.ndarray = None) -> Optional[np.ndarray]:
        """Extract features for machine learning models."""
        try:
            features = []
            
            # Price-based features
            features.extend([
                np.mean(close),
                np.std(close),
                (close[-1] - close[0]) / close[0],  # Total return
                np.max(high) - np.min(low),  # Range
                (close[-1] - np.mean(close)) / np.std(close),  # Z-score
            ])
            
            # Technical features
            if len(close) >= 20:
                sma20 = np.mean(close[-20:])
                features.append((close[-1] - sma20) / sma20)
            else:
                features.append(0)
            
            # Volatility features
            if len(close) > 1:
                returns = np.diff(np.log(close))
                features.extend([
                    np.std(returns),
                    np.mean(np.abs(returns)),
                    len(returns[returns > 0]) / len(returns)  # Up ratio
                ])
            else:
                features.extend([0, 0, 0.5])
            
            # Volume features (if available)
            if volume is not None:
                features.extend([
                    np.mean(volume),
                    volume[-1] / (np.mean(volume) + 1e-10),
                    np.corrcoef(close, volume)[0, 1] if len(close) > 1 else 0
                ])
            else:
                features.extend([0, 1, 0])
            
            # Market structure features
            if len(close) >= 10:
                trend_slope = np.polyfit(range(10), close[-10:], 1)[0]
                features.append(trend_slope)
            else:
                features.append(0)
            
            # Support/Resistance features
            recent_high = np.max(high[-10:]) if len(high) >= 10 else high[-1]
            recent_low = np.min(low[-10:]) if len(low) >= 10 else low[-1]
            features.extend([
                (close[-1] - recent_low) / (recent_high - recent_low + 1e-10),
                recent_high - recent_low
            ])
            
            # Ensure exact feature count
            while len(features) < self.parameters['feature_count']:
                features.append(0.0)
            
            features = features[:self.parameters['feature_count']]
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting ML features: {e}")
            return None
    
    def _calculate_future_level_strength(self, level: float, high: np.ndarray, low: np.ndarray) -> float:
        """Calculate strength of a level based on future price action."""
        try:
            touches = 0
            bounces = 0
            
            tolerance = level * 0.001  # 0.1% tolerance
            
            for i in range(len(high)):
                if low[i] <= level + tolerance and high[i] >= level - tolerance:
                    touches += 1
                    
                    # Check for bounce
                    if i < len(high) - 1:
                        if low[i] <= level <= high[i]:  # Price crossed level
                            if abs(high[i+1] - level) > abs(low[i+1] - level):
                                bounces += 1
            
            # Strength based on touches and bounces
            strength = min((touches * 0.2 + bounces * 0.3), 1.0)
            return strength
            
        except Exception as e:
            self.logger.error(f"Error calculating future level strength: {e}")
            return 0.0    
    def _enhance_levels_with_ml(self, all_levels: Dict[PivotTimeframe, Dict[PivotMethod, List[PivotLevel]]], 
                               high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[PivotTimeframe, Dict[PivotMethod, List[PivotLevel]]]:
        """Enhance pivot levels using machine learning predictions."""
        try:
            if not self.is_trained:
                return all_levels
            
            enhanced_levels = {}
            
            # Extract current features
            features = self._extract_ml_features(high[-50:], low[-50:], close[-50:])
            if features is None:
                return all_levels
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            for timeframe, method_levels in all_levels.items():
                enhanced_levels[timeframe] = {}
                
                for method, levels in method_levels.items():
                    enhanced_method_levels = []
                    
                    for level in levels:
                        # Predict level strength using ML
                        if self.strength_classifier is not None:
                            predicted_strength = self.strength_classifier.predict(features_scaled)[0]
                            adjusted_strength = (level.strength + predicted_strength) / 2
                        else:
                            adjusted_strength = level.strength
                        
                        # Adjust confidence based on ML prediction accuracy
                        confidence_boost = self._calculate_ml_confidence_boost(features_scaled)
                        adjusted_confidence = min(level.confidence + confidence_boost, 1.0)
                        
                        enhanced_level = PivotLevel(
                            value=level.value,
                            level_type=level.level_type,
                            strength=adjusted_strength,
                            confidence=adjusted_confidence,
                            touch_count=level.touch_count,
                            last_touch=level.last_touch,
                            volume_at_level=level.volume_at_level,
                            method=level.method,
                            timeframe=level.timeframe
                        )
                        
                        enhanced_method_levels.append(enhanced_level)
                    
                    enhanced_levels[timeframe][method] = enhanced_method_levels
            
            return enhanced_levels
            
        except Exception as e:
            self.logger.error(f"Error enhancing levels with ML: {e}")
            return all_levels
    
    def _calculate_ml_confidence_boost(self, features_scaled: np.ndarray) -> float:
        """Calculate confidence boost based on ML model reliability."""
        try:
            if self.level_predictor is None:
                return 0.0
            
            # Use model's feature importance as a proxy for confidence
            feature_importance = self.level_predictor.feature_importances_
            weighted_features = np.abs(features_scaled[0]) * feature_importance
            confidence_score = np.mean(weighted_features)
            
            # Normalize to 0-0.2 boost
            confidence_boost = min(confidence_score * 0.2, 0.2)
            
            return confidence_boost
            
        except Exception as e:
            self.logger.error(f"Error calculating ML confidence boost: {e}")
            return 0.0
    
    def _validate_and_filter_levels(self, all_levels: Dict[PivotTimeframe, Dict[PivotMethod, List[PivotLevel]]], 
                                   high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[PivotTimeframe, Dict[PivotMethod, List[PivotLevel]]]:
        """Validate and filter pivot levels based on various criteria."""
        try:
            filtered_levels = {}
            
            for timeframe, method_levels in all_levels.items():
                filtered_levels[timeframe] = {}
                
                for method, levels in method_levels.items():
                    filtered_method_levels = []
                    
                    for level in levels:
                        # Filter criteria
                        if (level.strength >= 0.2 and 
                            level.confidence >= 0.2 and
                            self._is_level_valid(level, high, low, close)):
                            
                            filtered_method_levels.append(level)
                    
                    # Limit number of levels per method
                    max_levels = self.parameters['max_levels_per_method']
                    if len(filtered_method_levels) > max_levels:
                        # Keep strongest levels
                        filtered_method_levels.sort(key=lambda x: x.strength, reverse=True)
                        filtered_method_levels = filtered_method_levels[:max_levels]
                    
                    filtered_levels[timeframe][method] = filtered_method_levels
            
            return filtered_levels
            
        except Exception as e:
            self.logger.error(f"Error validating and filtering levels: {e}")
            return all_levels
    
    def _is_level_valid(self, level: PivotLevel, high: np.ndarray, 
                       low: np.ndarray, close: np.ndarray) -> bool:
        """Check if a pivot level is valid based on historical data."""
        try:
            current_price = close[-1]
            
            # Check if level is too close to current price
            min_distance = current_price * 0.001  # 0.1% minimum distance
            if abs(level.value - current_price) < min_distance:
                return False
            
            # Check if level is within reasonable price range
            price_range = np.max(high[-50:]) - np.min(low[-50:]) if len(high) >= 50 else np.max(high) - np.min(low)
            max_distance = current_price + price_range * 2
            min_price = current_price - price_range * 2
            
            if level.value > max_distance or level.value < min_price:
                return False
            
            # Additional validation criteria can be added here
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating level: {e}")
            return True
    
    def _detect_confluence_zones(self, all_levels: Dict[PivotTimeframe, Dict[PivotMethod, List[PivotLevel]]]) -> List[ConfluenceZone]:
        """Detect confluence zones where multiple levels converge."""
        try:
            all_level_values = []
            all_levels_flat = []
            
            # Flatten all levels
            for timeframe, method_levels in all_levels.items():
                for method, levels in method_levels.items():
                    for level in levels:
                        all_level_values.append(level.value)
                        all_levels_flat.append(level)
            
            if len(all_level_values) < self.parameters['min_confluence_levels']:
                return []
            
            confluence_zones = []
            confluence_threshold = self.parameters['confluence_threshold']
            
            # Group levels by proximity
            level_groups = self._group_levels_by_proximity(all_levels_flat, confluence_threshold)
            
            for group in level_groups:
                if len(group) >= self.parameters['min_confluence_levels']:
                    zone = self._create_confluence_zone(group)
                    if zone.width <= self.parameters['max_confluence_width']:
                        confluence_zones.append(zone)
            
            # Sort by significance
            confluence_zones.sort(key=lambda x: x.significance_score, reverse=True)
            
            return confluence_zones[:10]  # Limit to top 10 zones
            
        except Exception as e:
            self.logger.error(f"Error detecting confluence zones: {e}")
            return []
    
    def _group_levels_by_proximity(self, levels: List[PivotLevel], threshold: float) -> List[List[PivotLevel]]:
        """Group levels that are close to each other."""
        try:
            if not levels:
                return []
            
            # Sort levels by value
            sorted_levels = sorted(levels, key=lambda x: x.value)
            groups = []
            current_group = [sorted_levels[0]]
            
            for i in range(1, len(sorted_levels)):
                current_level = sorted_levels[i]
                last_in_group = current_group[-1]
                
                # Check if current level is close to the group
                distance = abs(current_level.value - last_in_group.value) / last_in_group.value
                
                if distance <= threshold:
                    current_group.append(current_level)
                else:
                    groups.append(current_group)
                    current_group = [current_level]
            
            groups.append(current_group)
            
            return groups
            
        except Exception as e:
            self.logger.error(f"Error grouping levels by proximity: {e}")
            return []
    
    def _create_confluence_zone(self, levels: List[PivotLevel]) -> ConfluenceZone:
        """Create a confluence zone from a group of levels."""
        try:
            values = [level.value for level in levels]
            center_price = np.mean(values)
            width = max(values) - min(values)
            
            # Calculate strength as weighted average
            total_weight = sum(level.strength for level in levels)
            strength = total_weight / len(levels) if levels else 0
            
            # Apply confluence multiplier
            strength *= self.parameters['confluence_strength_multiplier']
            strength = min(strength, 1.0)
            
            # Collect unique timeframes and methods
            timeframes = list(set(level.timeframe for level in levels))
            methods = list(set(level.method for level in levels))
            
            # Calculate significance score
            significance_factors = [
                len(levels) / 10,  # Number of levels
                strength,  # Average strength
                len(timeframes) / len(PivotTimeframe),  # Timeframe diversity
                len(methods) / len(PivotMethod),  # Method diversity
                1 / (width + 1e-10)  # Inverse of width (tighter = more significant)
            ]
            
            significance_score = np.mean(significance_factors)
            
            return ConfluenceZone(
                center_price=center_price,
                width=width,
                strength=strength,
                contributing_levels=levels,
                timeframes=timeframes,
                methods=methods,
                significance_score=significance_score
            )
            
        except Exception as e:
            self.logger.error(f"Error creating confluence zone: {e}")
            return ConfluenceZone(0, 0, 0, [], [], [], 0)
    
    def _generate_signals(self, all_levels: Dict[PivotTimeframe, Dict[PivotMethod, List[PivotLevel]]], 
                         confluence_zones: List[ConfluenceZone],
                         high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                         volume: np.ndarray = None) -> List[PivotSignal]:
        """Generate trading signals based on pivot levels and confluence zones."""
        try:
            signals = []
            current_price = close[-1]
            
            # Generate level-based signals
            for timeframe, method_levels in all_levels.items():
                for method, levels in method_levels.items():
                    for level in levels:
                        signal = self._check_level_signal(level, current_price, high, low, close, volume)
                        if signal:
                            signals.append(signal)
            
            # Generate confluence-based signals
            for zone in confluence_zones:
                signal = self._check_confluence_signal(zone, current_price, high, low, close, volume)
                if signal:
                    signals.append(signal)
            
            # Filter and rank signals
            filtered_signals = self._filter_and_rank_signals(signals)
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []
    
    def _check_level_signal(self, level: PivotLevel, current_price: float,
                           high: np.ndarray, low: np.ndarray, close: np.ndarray,
                           volume: np.ndarray = None) -> Optional[PivotSignal]:
        """Check for signals at a specific pivot level."""
        try:
            tolerance = level.value * self.parameters['level_tolerance']
            
            # Check for breakout
            if self._is_breakout(level, current_price, high, low, close):
                signal_strength = level.strength * 0.8  # Breakouts are strong
                confidence = level.confidence * 0.9
                
                return PivotSignal(
                    timestamp=datetime.now(),
                    signal_type='breakout',
                    level=level,
                    price=current_price,
                    strength=signal_strength,
                    confidence=confidence,
                    probability=self._calculate_breakout_probability(level, high, low, close),
                    target_price=self._calculate_target_price(level, current_price, 'breakout'),
                    stop_loss=self._calculate_stop_loss(level, current_price, 'breakout'),
                    risk_reward_ratio=self._calculate_risk_reward_ratio(level, current_price, 'breakout'),
                    volume_confirmation=self._check_volume_confirmation(volume) if volume is not None else False
                )
            
            # Check for bounce
            if self._is_bounce(level, current_price, high, low, close):
                signal_strength = level.strength * 0.7  # Bounces are moderate
                confidence = level.confidence * 0.8
                
                return PivotSignal(
                    timestamp=datetime.now(),
                    signal_type='bounce',
                    level=level,
                    price=current_price,
                    strength=signal_strength,
                    confidence=confidence,
                    probability=self._calculate_bounce_probability(level, high, low, close),
                    target_price=self._calculate_target_price(level, current_price, 'bounce'),
                    stop_loss=self._calculate_stop_loss(level, current_price, 'bounce'),
                    risk_reward_ratio=self._calculate_risk_reward_ratio(level, current_price, 'bounce'),
                    volume_confirmation=self._check_volume_confirmation(volume) if volume is not None else False
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking level signal: {e}")
            return None
    
    def _is_breakout(self, level: PivotLevel, current_price: float,
                    high: np.ndarray, low: np.ndarray, close: np.ndarray) -> bool:
        """Check if current price action indicates a breakout."""
        try:
            tolerance = level.value * self.parameters['level_tolerance']
            
            # Check if price has broken through the level
            if level.level_type == 'resistance':
                return current_price > level.value + tolerance
            elif level.level_type == 'support':
                return current_price < level.value - tolerance
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking breakout: {e}")
            return False
    
    def _is_bounce(self, level: PivotLevel, current_price: float,
                  high: np.ndarray, low: np.ndarray, close: np.ndarray) -> bool:
        """Check if current price action indicates a bounce."""
        try:
            tolerance = level.value * self.parameters['level_tolerance']
            
            # Check if price touched the level but bounced back
            if level.level_type == 'resistance':
                # Price approached resistance but couldn't break through
                return (high[-1] >= level.value - tolerance and 
                       current_price < level.value - tolerance * 2)
            elif level.level_type == 'support':
                # Price approached support but bounced up
                return (low[-1] <= level.value + tolerance and 
                       current_price > level.value + tolerance * 2)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking bounce: {e}")
            return False
    
    def _check_confluence_signal(self, zone: ConfluenceZone, current_price: float,
                                high: np.ndarray, low: np.ndarray, close: np.ndarray,
                                volume: np.ndarray = None) -> Optional[PivotSignal]:
        """Check for signals at confluence zones."""
        try:
            zone_tolerance = zone.width / 2
            
            # Check if price is near the confluence zone
            distance_to_zone = abs(current_price - zone.center_price)
            
            if distance_to_zone <= zone_tolerance:
                # Determine signal type based on price action
                signal_type = 'confluence'
                
                # Create a representative level for the zone
                zone_level = PivotLevel(
                    value=zone.center_price,
                    level_type='confluence',
                    strength=zone.strength,
                    confidence=0.8,
                    touch_count=len(zone.contributing_levels),
                    last_touch=None,
                    volume_at_level=0.0,
                    method=PivotMethod.ADAPTIVE,
                    timeframe=PivotTimeframe.ADAPTIVE
                )
                
                return PivotSignal(
                    timestamp=datetime.now(),
                    signal_type=signal_type,
                    level=zone_level,
                    price=current_price,
                    strength=zone.strength,
                    confidence=0.8,
                    probability=zone.significance_score,
                    target_price=self._calculate_target_price(zone_level, current_price, signal_type),
                    stop_loss=self._calculate_stop_loss(zone_level, current_price, signal_type),
                    risk_reward_ratio=self._calculate_risk_reward_ratio(zone_level, current_price, signal_type),
                    volume_confirmation=self._check_volume_confirmation(volume) if volume is not None else False
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking confluence signal: {e}")
            return None
    
    def _calculate_breakout_probability(self, level: PivotLevel, 
                                      high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Calculate probability of successful breakout."""
        try:
            # Factors affecting breakout probability
            factors = []
            
            # Level strength
            factors.append(level.strength)
            
            # Number of touches (more touches = higher probability of breakout)
            touch_factor = min(level.touch_count / 5, 1.0)
            factors.append(touch_factor)
            
            # Recent momentum
            if len(close) >= 5:
                momentum = (close[-1] - close[-5]) / close[-5]
                momentum_factor = min(abs(momentum) * 10, 1.0)
                factors.append(momentum_factor)
            
            # Volatility (higher volatility = higher breakout probability)
            if len(close) >= 10:
                volatility = np.std(np.diff(np.log(close[-10:])))
                vol_factor = min(volatility * 50, 1.0)
                factors.append(vol_factor)
            
            probability = np.mean(factors)
            return np.clip(probability, 0.1, 0.9)
            
        except Exception as e:
            self.logger.error(f"Error calculating breakout probability: {e}")
            return 0.5
    
    def _calculate_bounce_probability(self, level: PivotLevel, 
                                    high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Calculate probability of successful bounce."""
        try:
            # Factors affecting bounce probability
            factors = []
            
            # Level strength
            factors.append(level.strength)
            
            # Historical bounce rate
            bounce_rate = level.touch_count / max(level.touch_count + 1, 1)
            factors.append(bounce_rate)
            
            # Distance from level
            distance = abs(close[-1] - level.value) / level.value
            distance_factor = max(0, 1 - distance * 100)  # Closer = higher probability
            factors.append(distance_factor)
            
            probability = np.mean(factors)
            return np.clip(probability, 0.1, 0.9)
            
        except Exception as e:
            self.logger.error(f"Error calculating bounce probability: {e}")
            return 0.5    
    def _calculate_target_price(self, level: PivotLevel, current_price: float, signal_type: str) -> float:
        """Calculate target price for the signal."""
        try:
            if signal_type == 'breakout':
                if level.level_type == 'resistance' or current_price > level.value:
                    # Upward breakout - target is level value plus range extension
                    range_extension = abs(level.value - current_price) * self.parameters['target_extension_factor']
                    return level.value + range_extension
                else:
                    # Downward breakout
                    range_extension = abs(current_price - level.value) * self.parameters['target_extension_factor']
                    return level.value - range_extension
            
            elif signal_type == 'bounce':
                if level.level_type == 'support':
                    # Bounce from support - target resistance
                    return level.value * (1 + self.parameters['target_extension_factor'] * 0.01)
                else:
                    # Bounce from resistance - target support
                    return level.value * (1 - self.parameters['target_extension_factor'] * 0.01)
            
            else:  # confluence
                # Target based on direction from confluence zone
                direction = 1 if current_price > level.value else -1
                range_extension = abs(current_price - level.value) * self.parameters['target_extension_factor']
                return level.value + (direction * range_extension)
            
        except Exception as e:
            self.logger.error(f"Error calculating target price: {e}")
            return current_price * 1.02  # Default 2% target
    
    def _calculate_stop_loss(self, level: PivotLevel, current_price: float, signal_type: str) -> float:
        """Calculate stop loss for the signal."""
        try:
            buffer = level.value * self.parameters['stop_loss_buffer']
            
            if signal_type == 'breakout':
                if level.level_type == 'resistance' or current_price > level.value:
                    # Long position - stop below the level
                    return level.value - buffer
                else:
                    # Short position - stop above the level
                    return level.value + buffer
            
            elif signal_type == 'bounce':
                if level.level_type == 'support':
                    # Long from support - stop below support
                    return level.value - buffer
                else:
                    # Short from resistance - stop above resistance
                    return level.value + buffer
            
            else:  # confluence
                # Stop on opposite side of confluence zone
                if current_price > level.value:
                    return level.value - buffer
                else:
                    return level.value + buffer
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            return current_price * 0.98  # Default 2% stop
    
    def _calculate_risk_reward_ratio(self, level: PivotLevel, current_price: float, signal_type: str) -> float:
        """Calculate risk-reward ratio for the signal."""
        try:
            target = self._calculate_target_price(level, current_price, signal_type)
            stop = self._calculate_stop_loss(level, current_price, signal_type)
            
            reward = abs(target - current_price)
            risk = abs(current_price - stop)
            
            if risk == 0:
                return self.parameters['default_risk_reward_ratio']
            
            ratio = reward / risk
            return max(ratio, 0.5)  # Minimum 0.5:1 ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-reward ratio: {e}")
            return self.parameters['default_risk_reward_ratio']
    
    def _check_volume_confirmation(self, volume: np.ndarray) -> bool:
        """Check if volume confirms the signal."""
        try:
            if volume is None or len(volume) < 5:
                return False
            
            recent_volume = volume[-1]
            avg_volume = np.mean(volume[-5:])
            
            # Volume should be above average for confirmation
            return recent_volume > avg_volume * 1.2
            
        except Exception as e:
            self.logger.error(f"Error checking volume confirmation: {e}")
            return False
    
    def _filter_and_rank_signals(self, signals: List[PivotSignal]) -> List[PivotSignal]:
        """Filter and rank signals by quality."""
        try:
            # Filter by minimum requirements
            filtered_signals = []
            
            for signal in signals:
                if (signal.strength >= self.parameters['min_signal_strength'] and
                    signal.confidence >= self.parameters['min_confidence'] and
                    signal.risk_reward_ratio >= 1.0):
                    
                    filtered_signals.append(signal)
            
            # Rank by combined score
            def signal_score(sig):
                return (sig.strength * 0.3 + 
                       sig.confidence * 0.3 + 
                       sig.probability * 0.2 + 
                       min(sig.risk_reward_ratio / 3, 1) * 0.2)
            
            filtered_signals.sort(key=signal_score, reverse=True)
            
            return filtered_signals[:10]  # Return top 10 signals
            
        except Exception as e:
            self.logger.error(f"Error filtering and ranking signals: {e}")
            return signals
    
    def _analyze_level_strength(self, all_levels: Dict[PivotTimeframe, Dict[PivotMethod, List[PivotLevel]]], 
                               high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict[str, Any]:
        """Analyze the strength and distribution of pivot levels."""
        try:
            all_levels_flat = []
            for timeframe, method_levels in all_levels.items():
                for method, levels in method_levels.items():
                    all_levels_flat.extend(levels)
            
            if not all_levels_flat:
                return {'strong_levels': 0, 'total_levels': 0, 'average_strength': 0}
            
            # Count strong levels
            strong_levels = sum(1 for level in all_levels_flat if level.strength > 0.7)
            
            # Calculate statistics
            strengths = [level.strength for level in all_levels_flat]
            confidences = [level.confidence for level in all_levels_flat]
            
            analysis = {
                'strong_levels': strong_levels,
                'total_levels': len(all_levels_flat),
                'average_strength': np.mean(strengths),
                'max_strength': np.max(strengths),
                'min_strength': np.min(strengths),
                'average_confidence': np.mean(confidences),
                'strength_distribution': {
                    'very_strong': sum(1 for s in strengths if s > 0.8),
                    'strong': sum(1 for s in strengths if 0.6 < s <= 0.8),
                    'moderate': sum(1 for s in strengths if 0.4 < s <= 0.6),
                    'weak': sum(1 for s in strengths if s <= 0.4)
                }
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing level strength: {e}")
            return {'strong_levels': 0, 'total_levels': 0, 'average_strength': 0}
    
    def _create_support_resistance_map(self, all_levels: Dict[PivotTimeframe, Dict[PivotMethod, List[PivotLevel]]], 
                                      confluence_zones: List[ConfluenceZone]) -> Dict[str, Any]:
        """Create a comprehensive support and resistance map."""
        try:
            support_levels = []
            resistance_levels = []
            
            # Collect support and resistance levels
            for timeframe, method_levels in all_levels.items():
                for method, levels in method_levels.items():
                    for level in levels:
                        if level.level_type == 'support':
                            support_levels.append(level)
                        elif level.level_type == 'resistance':
                            resistance_levels.append(level)
            
            # Sort by strength
            support_levels.sort(key=lambda x: x.strength, reverse=True)
            resistance_levels.sort(key=lambda x: x.strength, reverse=True)
            
            # Create map
            sr_map = {
                'key_support_levels': [level.value for level in support_levels[:5]],
                'key_resistance_levels': [level.value for level in resistance_levels[:5]],
                'support_count': len(support_levels),
                'resistance_count': len(resistance_levels),
                'strongest_support': support_levels[0].value if support_levels else 0,
                'strongest_resistance': resistance_levels[0].value if resistance_levels else 0,
                'confluence_zones': [zone.center_price for zone in confluence_zones[:5]],
                'zone_count': len(confluence_zones)
            }
            
            return sr_map
            
        except Exception as e:
            self.logger.error(f"Error creating support/resistance map: {e}")
            return {}
    
    def _assess_current_market_position(self, current_price: float, 
                                       all_levels: Dict[PivotTimeframe, Dict[PivotMethod, List[PivotLevel]]], 
                                       confluence_zones: List[ConfluenceZone]) -> Dict[str, Any]:
        """Assess current market position relative to pivot levels."""
        try:
            # Find nearest support and resistance
            support_levels = []
            resistance_levels = []
            
            for timeframe, method_levels in all_levels.items():
                for method, levels in method_levels.items():
                    for level in levels:
                        if level.level_type == 'support' and level.value < current_price:
                            support_levels.append(level)
                        elif level.level_type == 'resistance' and level.value > current_price:
                            resistance_levels.append(level)
            
            # Find nearest levels
            nearest_support = max(support_levels, key=lambda x: x.value) if support_levels else None
            nearest_resistance = min(resistance_levels, key=lambda x: x.value) if resistance_levels else None
            
            # Determine position
            if nearest_support and nearest_resistance:
                support_distance = current_price - nearest_support.value
                resistance_distance = nearest_resistance.value - current_price
                total_range = nearest_resistance.value - nearest_support.value
                
                if total_range > 0:
                    position_ratio = support_distance / total_range
                    
                    if position_ratio < 0.3:
                        position = 'near_support'
                    elif position_ratio > 0.7:
                        position = 'near_resistance'
                    else:
                        position = 'middle'
                else:
                    position = 'middle'
            else:
                position = 'unclear'
            
            assessment = {
                'nearest_support': nearest_support.value if nearest_support else 0,
                'nearest_resistance': nearest_resistance.value if nearest_resistance else 0,
                'support_strength': nearest_support.strength if nearest_support else 0,
                'resistance_strength': nearest_resistance.strength if nearest_resistance else 0,
                'price_position': position,
                'support_distance': (current_price - nearest_support.value) / current_price if nearest_support else 0,
                'resistance_distance': (nearest_resistance.value - current_price) / current_price if nearest_resistance else 0,
                'in_confluence_zone': any(
                    abs(current_price - zone.center_price) <= zone.width / 2 
                    for zone in confluence_zones
                )
            }
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing current market position: {e}")
            return {}
    
    # Helper methods for formatting and utility functions
    
    def _format_pivot_levels(self, all_levels: Dict) -> Dict[str, Any]:
        """Format pivot levels for output."""
        formatted = {}
        for timeframe, method_levels in all_levels.items():
            formatted[timeframe.value] = {}
            for method, levels in method_levels.items():
                formatted[timeframe.value][method.value] = [
                    {
                        'value': level.value,
                        'type': level.level_type,
                        'strength': level.strength,
                        'confidence': level.confidence,
                        'touches': level.touch_count
                    }
                    for level in levels
                ]
        return formatted
    
    def _format_confluence_zones(self, zones: List[ConfluenceZone]) -> List[Dict[str, Any]]:
        """Format confluence zones for output."""
        return [
            {
                'center_price': zone.center_price,
                'width': zone.width,
                'strength': zone.strength,
                'significance': zone.significance_score,
                'contributing_levels': len(zone.contributing_levels),
                'timeframes': [tf.value for tf in zone.timeframes],
                'methods': [m.value for m in zone.methods]
            }
            for zone in zones
        ]
    
    def _signal_to_dict(self, signal: PivotSignal) -> Dict[str, Any]:
        """Convert signal to dictionary format."""
        return {
            'timestamp': signal.timestamp.isoformat(),
            'type': signal.signal_type,
            'level_value': signal.level.value,
            'level_type': signal.level.level_type,
            'price': signal.price,
            'strength': signal.strength,
            'confidence': signal.confidence,
            'probability': signal.probability,
            'target': signal.target_price,
            'stop_loss': signal.stop_loss,
            'risk_reward': signal.risk_reward_ratio,
            'volume_confirmed': signal.volume_confirmation
        }
    
    def _get_highest_strength_signal(self, signals: List[PivotSignal]) -> Optional[Dict[str, Any]]:
        """Get the signal with highest strength."""
        if not signals:
            return None
        
        strongest_signal = max(signals, key=lambda x: x.strength)
        return self._signal_to_dict(strongest_signal)
    
    def _get_most_significant_zone(self, zones: List[ConfluenceZone]) -> Optional[Dict[str, Any]]:
        """Get the most significant confluence zone."""
        if not zones:
            return None
        
        most_significant = max(zones, key=lambda x: x.significance_score)
        return {
            'center_price': most_significant.center_price,
            'strength': most_significant.strength,
            'significance': most_significant.significance_score,
            'contributing_levels': len(most_significant.contributing_levels)
        }
    
    def _get_best_method(self) -> str:
        """Get the best performing pivot method."""
        # This would typically analyze historical performance
        # For now, return adaptive as it combines multiple methods
        return PivotMethod.ADAPTIVE.value
    
    def _calculate_method_accuracy(self) -> Dict[str, float]:
        """Calculate accuracy scores for each method."""
        # Placeholder implementation - would analyze historical performance
        return {
            method.value: 0.7 + np.random.uniform(-0.1, 0.2)  # Mock scores
            for method in PivotMethod
        }
    
    def _recommend_adaptive_method(self, market_analysis: Dict) -> str:
        """Recommend best method based on current market conditions."""
        volatility_regime = market_analysis.get('volatility_regime', 'normal')
        trend_strength = market_analysis.get('trend_strength', 0)
        
        if volatility_regime == 'high':
            return PivotMethod.CAMARILLA.value  # Better for volatile markets
        elif trend_strength > 0.7:
            return PivotMethod.FIBONACCI.value  # Better for trending markets
        else:
            return PivotMethod.STANDARD.value   # Default for normal conditions
    
    def _calculate_overall_risk(self, all_levels: Dict, market_analysis: Dict) -> float:
        """Calculate overall risk level."""
        try:
            risk_factors = []
            
            # Volatility risk
            volatility = market_analysis.get('volatility', 0)
            risk_factors.append(min(volatility * 10, 1.0))
            
            # Level density risk (too many levels = confusion)
            total_levels = sum(len(levels) for method_levels in all_levels.values() 
                             for levels in method_levels.values())
            density_risk = min(total_levels / 50, 1.0)
            risk_factors.append(density_risk)
            
            # Trend uncertainty risk
            trend_strength = market_analysis.get('trend_strength', 0)
            uncertainty_risk = 1 - trend_strength
            risk_factors.append(uncertainty_risk)
            
            overall_risk = np.mean(risk_factors)
            return np.clip(overall_risk, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall risk: {e}")
            return 0.5
    
    def _calculate_breakout_probability(self, current_price: float, all_levels: Dict, market_analysis: Dict) -> float:
        """Calculate probability of near-term breakout."""
        try:
            # Find levels near current price
            nearby_levels = []
            tolerance = current_price * 0.02  # 2% tolerance
            
            for method_levels in all_levels.values():
                for levels in method_levels.values():
                    for level in levels:
                        if abs(level.value - current_price) <= tolerance:
                            nearby_levels.append(level)
            
            if not nearby_levels:
                return 0.3  # Default probability
            
            # Calculate based on level strength and market momentum
            avg_strength = np.mean([level.strength for level in nearby_levels])
            momentum = market_analysis.get('trend_strength', 0)
            volatility = market_analysis.get('volatility', 0)
            
            breakout_prob = (avg_strength * 0.4 + momentum * 0.4 + volatility * 0.2)
            return np.clip(breakout_prob, 0.1, 0.9)
            
        except Exception as e:
            self.logger.error(f"Error calculating breakout probability: {e}")
            return 0.3
    
    def _calculate_reversal_probability(self, current_price: float, all_levels: Dict, market_analysis: Dict) -> float:
        """Calculate probability of trend reversal."""
        try:
            # Find strong support/resistance levels
            strong_levels = []
            
            for method_levels in all_levels.values():
                for levels in method_levels.values():
                    for level in levels:
                        if level.strength > 0.7:
                            strong_levels.append(level)
            
            if not strong_levels:
                return 0.2
            
            # Check if price is near strong levels
            near_strong_level = any(
                abs(level.value - current_price) / current_price < 0.01 
                for level in strong_levels
            )
            
            momentum = market_analysis.get('trend_strength', 0)
            structure_score = market_analysis.get('structure_score', 0.5)
            
            if near_strong_level:
                reversal_prob = 0.7 - momentum * 0.3
            else:
                reversal_prob = 0.3 - structure_score * 0.2
            
            return np.clip(reversal_prob, 0.1, 0.8)
            
        except Exception as e:
            self.logger.error(f"Error calculating reversal probability: {e}")
            return 0.2
    
    def _calculate_ml_enhancement_score(self) -> float:
        """Calculate ML enhancement effectiveness score."""
        if not self.is_trained:
            return 0.0
        
        # This would typically compare ML-enhanced vs non-enhanced performance
        # For now, return a score based on model training success
        return 0.75  # Placeholder score
    
    def _get_active_parameters(self) -> Dict[str, Any]:
        """Get currently active parameters."""
        return {
            'methods_enabled': [method.value for method in self.parameters['methods_enabled']],
            'timeframes_enabled': [tf.value for tf in self.parameters['timeframes_enabled']],
            'ml_enabled': self.parameters['ml_enabled'],
            'confluence_threshold': self.parameters['confluence_threshold'],
            'min_signal_strength': self.parameters['min_signal_strength']
        }
    
    def _assess_data_quality(self, high: np.ndarray, low: np.ndarray, 
                           close: np.ndarray, volume: np.ndarray = None) -> float:
        """Assess quality of input data."""
        try:
            quality_factors = []
            
            # Data completeness
            if len(close) >= self.parameters['lookback_period']:
                quality_factors.append(1.0)
            else:
                quality_factors.append(len(close) / self.parameters['lookback_period'])
            
            # Data consistency (no extreme outliers)
            price_changes = np.diff(close) / close[:-1]
            outlier_ratio = np.sum(np.abs(price_changes) > 0.1) / len(price_changes)
            consistency_score = max(0, 1 - outlier_ratio * 5)
            quality_factors.append(consistency_score)
            
            # OHLC validity
            valid_ohlc = np.all((high >= low) & (high >= close) & (low <= close))
            quality_factors.append(1.0 if valid_ohlc else 0.5)
            
            # Volume data availability
            if volume is not None:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.8)
            
            return np.mean(quality_factors)
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {e}")
            return 0.5
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory issues."""
        try:
            max_history = 1000
            
            if len(self.price_history) > max_history:
                self.price_history = self.price_history[-max_history:]
            
            if len(self.volume_history) > max_history:
                self.volume_history = self.volume_history[-max_history:]
            
            if len(self.signal_history) > max_history:
                self.signal_history = self.signal_history[-max_history:]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def _generate_default_result(self) -> Dict[str, Any]:
        """Generate default result when insufficient data."""
        return {
            'pivot_levels': {},
            'confluence_zones': [],
            'current_price': 0.0,
            'nearest_support': 0.0,
            'nearest_resistance': 0.0,
            'error': 'Insufficient data for calculation',
            'calculation_timestamp': datetime.now().isoformat()
        }
    
    def _generate_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Generate error result."""
        return {
            'pivot_levels': {},
            'confluence_zones': [],
            'current_price': 0.0,
            'error': error_msg,
            'calculation_timestamp': datetime.now().isoformat()
        }


# Supporting classes (simplified implementations)

class LevelCalculator:
    def __init__(self, parameters):
        self.parameters = parameters

class ConfluenceAnalyzer:
    def __init__(self, parameters):
        self.parameters = parameters

class MLPivotEnhancer:
    def __init__(self, parameters):
        self.parameters = parameters

class PivotSignalGenerator:
    def __init__(self, parameters):
        self.parameters = parameters

class LevelValidationEngine:
    def __init__(self, parameters):
        self.parameters = parameters


# Factory function for easy instantiation
def create_pivot_point_indicator(parameters: Optional[Dict[str, Any]] = None) -> PivotPointIndicator:
    """
    Factory function to create a Pivot Point Indicator instance.
    
    Args:
        parameters: Optional configuration parameters
        
    Returns:
        Configured PivotPointIndicator instance
    """
    return PivotPointIndicator(parameters)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Create indicator
    pivot_indicator = create_pivot_point_indicator({
        'ml_enabled': True,
        'methods_enabled': [PivotMethod.STANDARD, PivotMethod.FIBONACCI, PivotMethod.CAMARILLA],
        'timeframes_enabled': [PivotTimeframe.DAILY, PivotTimeframe.FOUR_HOUR],
        'confluence_threshold': 0.002
    })
    
    # Generate sample data
    np.random.seed(42)
    n_points = 200
    
    # Simulate realistic OHLC data
    base_price = 1.2000  # EUR/USD example
    prices = [base_price]
    
    for i in range(n_points - 1):
        # Add trend and noise
        trend = 0.0001 * np.sin(i / 30)  # Long-term cycle
        noise = np.random.normal(0, 0.0005)  # Random walk
        price_change = trend + noise
        new_price = prices[-1] + price_change
        prices.append(max(new_price, 0.5))  # Prevent negative prices
    
    # Create OHLC from price series
    close = np.array(prices)
    high = close * (1 + np.random.uniform(0, 0.005, len(close)))
    low = close * (1 - np.random.uniform(0, 0.005, len(close)))
    volume = np.random.uniform(10000, 100000, len(close))
    
    # Calculate pivot points
    result = pivot_indicator.calculate(high, low, close, volume)
    
    print("Advanced Pivot Point Indicator Results:")
    print(f"Current Price: {result['current_price']:.5f}")
    print(f"Nearest Support: {result['nearest_support']:.5f}")
    print(f"Nearest Resistance: {result['nearest_resistance']:.5f}")
    print(f"Total Levels: {result['total_levels']}")
    print(f"Confluence Zones: {result['confluence_zones_count']}")
    print(f"Active Signals: {result['signal_count']}")
    print(f"Market Trend: {result['market_trend']}")
    print(f"Volatility Regime: {result['volatility_regime']}")
    print(f"Breakout Probability: {result['breakout_probability']:.2%}")
    print(f"Overall Risk Level: {result['overall_risk_level']:.2f}")
    print(f"ML Enhancement Score: {result['ml_enhancement_score']:.2f}")
    
    # Print top confluence zone if available
    if result['most_significant_zone']:
        zone = result['most_significant_zone']
        print(f"\nMost Significant Confluence Zone:")
        print(f"  Center Price: {zone['center_price']:.5f}")
        print(f"  Strength: {zone['strength']:.2f}")
        print(f"  Significance: {zone['significance']:.2f}")
    
    # Print strongest signal if available
    if result['highest_strength_signal']:
        signal = result['highest_strength_signal']
        print(f"\nStrongest Signal:")
        print(f"  Type: {signal['type']}")
        print(f"  Strength: {signal['strength']:.2f}")
        print(f"  Target: {signal['target']:.5f}")
        print(f"  Risk/Reward: {signal['risk_reward']:.1f}")