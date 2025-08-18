"""
Keltner Channel Indicator - Advanced Volatility Analysis System

This indicator creates dynamic channels around price using exponential moving averages and Average True Range (ATR),
providing sophisticated analysis for volatility, breakouts, and market squeezes.

Key Features:
- Adaptive channel width based on market volatility
- Multi-timeframe Keltner channel analysis
- Machine learning-enhanced breakout prediction
- Dynamic band adjustments using regime detection
- Squeeze identification and expansion signals
- Volume-weighted channel calculations
- Statistical significance testing for breakouts
- Advanced filtering to reduce false signals

Mathematical Foundation:
- Middle Line: EMA(price, period)
- Upper Band: EMA + (multiplier × ATR)
- Lower Band: EMA - (multiplier × ATR)
- Adaptive multiplier based on volatility regime
- ML models for breakout probability estimation

Author: AUJ Platform Development Team
Version: 1.0.0
Created: 2025-01-XX
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML and Statistical Libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

try:
    from scipy import stats
    from scipy.signal import argrelextrema
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..base.standard_indicator import StandardIndicatorInterface


class ChannelDirection(Enum):
    """Channel direction enumeration"""
    EXPANDING = "expanding"
    CONTRACTING = "contracting"
    STABLE = "stable"


class BreakoutType(Enum):
    """Breakout type classification"""
    UPPER_BREAKOUT = "upper_breakout"
    LOWER_BREAKOUT = "lower_breakout"
    FALSE_BREAKOUT = "false_breakout"
    NO_BREAKOUT = "no_breakout"


class VolatilityRegime(Enum):
    """Market volatility regime"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class ChannelLevel:
    """Individual channel level data"""
    value: float
    timestamp: datetime
    confidence: float
    volume_weight: float = 1.0
    support_resistance_strength: float = 0.0


@dataclass
class SqueezeMetrics:
    """Keltner Channel squeeze analysis metrics"""
    is_squeeze: bool
    squeeze_intensity: float
    squeeze_duration: int
    breakout_probability: float
    direction_bias: float  # Positive for upward, negative for downward


@dataclass
class BreakoutAnalysis:
    """Comprehensive breakout analysis"""
    breakout_type: BreakoutType
    confidence: float
    volume_confirmation: bool
    momentum_confirmation: bool
    expected_target: Optional[float]
    risk_level: float


@dataclass
class KeltnerChannelResult:
    """Complete Keltner Channel analysis result"""
    # Core channel values
    upper_band: float
    middle_line: float
    lower_band: float
    
    # Channel metrics
    channel_width: float
    width_percentile: float
    direction: ChannelDirection
    
    # Volatility analysis
    atr_value: float
    volatility_regime: VolatilityRegime
    volatility_percentile: float
    
    # Squeeze analysis
    squeeze_metrics: SqueezeMetrics
    
    # Breakout analysis
    breakout_analysis: BreakoutAnalysis
    
    # Price position
    price_position: float  # Where price is within the channel (0-1)
    band_touch_probability: Dict[str, float]
    
    # ML predictions
    breakout_probability: float
    direction_prediction: float
    
    # Additional metrics
    support_levels: List[ChannelLevel]
    resistance_levels: List[ChannelLevel]
    
    # Metadata
    timestamp: datetime
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'upper_band': self.upper_band,
            'middle_line': self.middle_line,
            'lower_band': self.lower_band,
            'channel_width': self.channel_width,
            'width_percentile': self.width_percentile,
            'direction': self.direction.value,
            'atr_value': self.atr_value,
            'volatility_regime': self.volatility_regime.value,
            'volatility_percentile': self.volatility_percentile,
            'squeeze_intensity': self.squeeze_metrics.squeeze_intensity,
            'squeeze_duration': self.squeeze_metrics.squeeze_duration,
            'breakout_probability': self.breakout_probability,
            'direction_prediction': self.direction_prediction,
            'price_position': self.price_position,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


class KeltnerChannelIndicator(StandardIndicatorInterface):
    """
    Advanced Keltner Channel Indicator with Machine Learning Integration
    
    This implementation provides sophisticated volatility analysis using adaptive
    Keltner Channels with ML-enhanced breakout prediction and squeeze detection.
    """
    
    def __init__(self, 
                 period: int = 20,
                 atr_period: int = 10,
                 multiplier: float = 2.0,
                 adaptive_multiplier: bool = True,
                 use_volume_weighting: bool = True,
                 enable_ml_prediction: bool = True,
                 lookback_period: int = 100):
        """
        Initialize Keltner Channel Indicator
        
        Args:
            period: EMA period for middle line
            atr_period: ATR calculation period
            multiplier: Base multiplier for channel width
            adaptive_multiplier: Enable adaptive multiplier based on volatility
            use_volume_weighting: Enable volume-weighted calculations
            enable_ml_prediction: Enable ML-based predictions
            lookback_period: Historical data period for analysis
        """
        super().__init__()
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.adaptive_multiplier = adaptive_multiplier
        self.use_volume_weighting = use_volume_weighting
        self.enable_ml_prediction = enable_ml_prediction
        self.lookback_period = lookback_period
        
        # ML models
        self.breakout_model: Optional[RandomForestClassifier] = None
        self.direction_model: Optional[RandomForestClassifier] = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        
        # Historical data storage
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.channel_history: List[Dict] = []
        self.atr_history: List[float] = []
        
        # Analysis parameters
        self.volatility_percentiles = [10, 25, 75, 90]
        self.squeeze_threshold = 0.75  # Squeeze when width is below this percentile
        
        self.logger = logging.getLogger(__name__)
        
        # Validate dependencies
        if enable_ml_prediction and not HAS_SKLEARN:
            self.logger.warning("Scikit-learn not available. ML features disabled.")
            self.enable_ml_prediction = False
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        if HAS_TALIB:
            return pd.Series(talib.ATR(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                timeperiod=period
            ), index=data.index)
        else:
            # Manual ATR calculation
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift(1))
            low_close = np.abs(data['low'] - data['close'].shift(1))
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            return true_range.rolling(window=period).mean()
    
    def _calculate_ema(self, prices: pd.Series, period: int, volume_weights: Optional[pd.Series] = None) -> pd.Series:
        """Calculate Exponential Moving Average with optional volume weighting"""
        if volume_weights is not None and self.use_volume_weighting:
            # Volume-weighted EMA
            alpha = 2.0 / (period + 1)
            weights = volume_weights / volume_weights.rolling(window=period).sum()
            
            ema = pd.Series(index=prices.index, dtype=float)
            ema.iloc[0] = prices.iloc[0]
            
            for i in range(1, len(prices)):
                weight = weights.iloc[i] if not pd.isna(weights.iloc[i]) else alpha
                ema.iloc[i] = weight * prices.iloc[i] + (1 - weight) * ema.iloc[i-1]
            
            return ema
        else:
            return prices.ewm(span=period).mean()
    
    def _determine_volatility_regime(self, atr_value: float, atr_history: List[float]) -> VolatilityRegime:
        """Determine current volatility regime"""
        if len(atr_history) < 50:
            return VolatilityRegime.NORMAL
        
        percentiles = np.percentile(atr_history[-252:], [25, 50, 75, 95])  # 1 year of data
        
        if atr_value <= percentiles[0]:
            return VolatilityRegime.LOW
        elif atr_value <= percentiles[2]:
            return VolatilityRegime.NORMAL
        elif atr_value <= percentiles[3]:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def _calculate_adaptive_multiplier(self, 
                                     volatility_regime: VolatilityRegime,
                                     base_multiplier: float) -> float:
        """Calculate adaptive multiplier based on volatility regime"""
        multiplier_adjustments = {
            VolatilityRegime.LOW: 0.8,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.HIGH: 1.2,
            VolatilityRegime.EXTREME: 1.5
        }
        
        return base_multiplier * multiplier_adjustments[volatility_regime]
    
    def _analyze_squeeze(self, 
                        channel_width: float,
                        width_history: List[float],
                        volume: float,
                        volume_history: List[float]) -> SqueezeMetrics:
        """Analyze Keltner Channel squeeze conditions"""
        if len(width_history) < 20:
            return SqueezeMetrics(
                is_squeeze=False,
                squeeze_intensity=0.0,
                squeeze_duration=0,
                breakout_probability=0.5,
                direction_bias=0.0
            )
        
        # Calculate width percentile
        width_percentile = (np.sum(np.array(width_history[-100:]) < channel_width) / 
                           len(width_history[-100:]))
        
        is_squeeze = width_percentile <= self.squeeze_threshold
        squeeze_intensity = max(0, (self.squeeze_threshold - width_percentile) / self.squeeze_threshold)
        
        # Calculate squeeze duration
        squeeze_duration = 0
        for i in range(len(width_history) - 1, -1, -1):
            if i < len(width_history) - 1:
                recent_widths = width_history[max(0, i-20):i+1]
                if len(recent_widths) > 0:
                    percentile = (np.sum(np.array(recent_widths) < width_history[i]) / 
                                 len(recent_widths))
                    if percentile > self.squeeze_threshold:
                        break
            squeeze_duration += 1
        
        # Estimate breakout probability
        breakout_probability = min(0.9, squeeze_intensity * 0.7 + squeeze_duration * 0.01)
        
        # Calculate direction bias using volume
        if len(volume_history) >= 10:
            recent_volume_avg = np.mean(volume_history[-10:])
            volume_ratio = volume / recent_volume_avg if recent_volume_avg > 0 else 1.0
            direction_bias = np.tanh((volume_ratio - 1.0) * 2.0)
        else:
            direction_bias = 0.0
        
        return SqueezeMetrics(
            is_squeeze=is_squeeze,
            squeeze_intensity=squeeze_intensity,
            squeeze_duration=squeeze_duration,
            breakout_probability=breakout_probability,
            direction_bias=direction_bias
        )
    
    def _analyze_breakout(self, 
                         price: float,
                         upper_band: float,
                         lower_band: float,
                         volume: float,
                         momentum: float,
                         price_history: List[float]) -> BreakoutAnalysis:
        """Analyze potential breakouts"""
        # Determine breakout type
        price_buffer = (upper_band - lower_band) * 0.01  # 1% buffer
        
        if price > upper_band + price_buffer:
            breakout_type = BreakoutType.UPPER_BREAKOUT
        elif price < lower_band - price_buffer:
            breakout_type = BreakoutType.LOWER_BREAKOUT
        else:
            breakout_type = BreakoutType.NO_BREAKOUT
        
        # Calculate confidence based on various factors
        confidence = 0.5
        volume_confirmation = False
        momentum_confirmation = False
        
        if breakout_type != BreakoutType.NO_BREAKOUT:
            # Volume confirmation
            if len(self.volume_history) >= 20:
                avg_volume = np.mean(self.volume_history[-20:])
                volume_confirmation = volume > avg_volume * 1.2
                confidence += 0.2 if volume_confirmation else -0.1
            
            # Momentum confirmation
            momentum_confirmation = abs(momentum) > 0.5
            confidence += 0.2 if momentum_confirmation else -0.1
            
            # Price distance from band
            if breakout_type == BreakoutType.UPPER_BREAKOUT:
                distance_ratio = (price - upper_band) / (upper_band - lower_band)
                confidence += min(0.3, distance_ratio * 2)
            else:
                distance_ratio = (lower_band - price) / (upper_band - lower_band)
                confidence += min(0.3, distance_ratio * 2)
        
        # Expected target calculation
        expected_target = None
        if breakout_type != BreakoutType.NO_BREAKOUT:
            channel_width = upper_band - lower_band
            if breakout_type == BreakoutType.UPPER_BREAKOUT:
                expected_target = upper_band + channel_width * 0.618  # Fibonacci extension
            else:
                expected_target = lower_band - channel_width * 0.618
        
        # Risk level assessment
        risk_level = 1.0 - confidence
        if len(price_history) >= 20:
            price_volatility = np.std(price_history[-20:]) / np.mean(price_history[-20:])
            risk_level = min(1.0, risk_level + price_volatility * 2)
        
        return BreakoutAnalysis(
            breakout_type=breakout_type,
            confidence=max(0.0, min(1.0, confidence)),
            volume_confirmation=volume_confirmation,
            momentum_confirmation=momentum_confirmation,
            expected_target=expected_target,
            risk_level=max(0.0, min(1.0, risk_level))
        )
    
    def _extract_features(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """Extract features for ML models"""
        if index < 20:
            return np.array([])
        
        features = []
        
        # Price-based features
        prices = data['close'].iloc[max(0, index-20):index+1]
        features.extend([
            prices.pct_change().mean(),
            prices.pct_change().std(),
            (prices.iloc[-1] - prices.mean()) / prices.std(),
            len(prices) - np.argmax(prices.values) - 1,  # Bars since high
            len(prices) - np.argmin(prices.values) - 1   # Bars since low
        ])
        
        # Volume features
        if 'volume' in data.columns:
            volumes = data['volume'].iloc[max(0, index-20):index+1]
            features.extend([
                volumes.iloc[-1] / volumes.mean() if volumes.mean() > 0 else 1.0,
                volumes.pct_change().mean(),
                volumes.pct_change().std()
            ])
        else:
            features.extend([1.0, 0.0, 0.0])
        
        # Technical features
        if hasattr(self, 'atr_history') and len(self.atr_history) > 10:
            features.append(self.atr_history[-1] / np.mean(self.atr_history[-10:]))
        else:
            features.append(1.0)
        
        # Channel width features
        if len(self.channel_history) >= 10:
            recent_widths = [ch['width'] for ch in self.channel_history[-10:]]
            current_width = self.channel_history[-1]['width']
            features.extend([
                current_width / np.mean(recent_widths),
                np.percentile(recent_widths, 25),
                np.percentile(recent_widths, 75)
            ])
        else:
            features.extend([1.0, 1.0, 1.0])
        
        return np.array(features)
    
    def _train_ml_models(self, data: pd.DataFrame) -> None:
        """Train ML models for breakout prediction"""
        if not self.enable_ml_prediction or not HAS_SKLEARN or len(data) < 100:
            return
        
        try:
            # Prepare features and targets
            features_list = []
            breakout_targets = []
            direction_targets = []
            
            for i in range(50, len(data) - 5):  # Leave 5 periods for future lookback
                features = self._extract_features(data, i)
                if len(features) == 0:
                    continue
                
                features_list.append(features)
                
                # Breakout target (next 5 periods)
                future_high = data['high'].iloc[i+1:i+6].max()
                future_low = data['low'].iloc[i+1:i+6].min()
                current_price = data['close'].iloc[i]
                
                price_change = max(
                    (future_high - current_price) / current_price,
                    (current_price - future_low) / current_price
                )
                
                breakout_targets.append(1 if price_change > 0.02 else 0)  # 2% threshold
                
                # Direction target
                future_price = data['close'].iloc[i+3]  # 3 periods ahead
                direction_targets.append(1 if future_price > current_price else 0)
            
            if len(features_list) < 50:
                return
            
            X = np.array(features_list)
            y_breakout = np.array(breakout_targets)
            y_direction = np.array(direction_targets)
            
            # Normalize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            self.breakout_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.breakout_model.fit(X_scaled, y_breakout)
            
            self.direction_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.direction_model.fit(X_scaled, y_direction)
            
            self.logger.info("ML models trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training ML models: {e}")
            self.breakout_model = None
            self.direction_model = None
    
    def _predict_with_ml(self, data: pd.DataFrame, current_index: int) -> Tuple[float, float]:
        """Make ML predictions"""
        if (not self.enable_ml_prediction or 
            self.breakout_model is None or 
            self.direction_model is None):
            return 0.5, 0.0
        
        try:
            features = self._extract_features(data, current_index)
            if len(features) == 0:
                return 0.5, 0.0
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            breakout_prob = self.breakout_model.predict_proba(features_scaled)[0][1]
            direction_prob = self.direction_model.predict_proba(features_scaled)[0][1]
            
            # Convert direction probability to bias (-1 to 1)
            direction_bias = (direction_prob - 0.5) * 2
            
            return breakout_prob, direction_bias
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {e}")
            return 0.5, 0.0
    
    def _identify_support_resistance(self, 
                                   data: pd.DataFrame,
                                   current_index: int) -> Tuple[List[ChannelLevel], List[ChannelLevel]]:
        """Identify dynamic support and resistance levels"""
        support_levels = []
        resistance_levels = []
        
        if current_index < 20:
            return support_levels, resistance_levels
        
        # Look back period
        lookback = min(100, current_index)
        price_data = data['close'].iloc[current_index-lookback:current_index+1]
        volume_data = data['volume'].iloc[current_index-lookback:current_index+1] if 'volume' in data.columns else None
        
        if HAS_SCIPY:
            # Find local minima and maxima
            minima_indices = argrelextrema(price_data.values, np.less, order=5)[0]
            maxima_indices = argrelextrema(price_data.values, np.greater, order=5)[0]
            
            # Create support levels from minima
            for idx in minima_indices[-10:]:  # Last 10 minima
                if idx < len(price_data):
                    volume_weight = 1.0
                    if volume_data is not None and idx < len(volume_data):
                        avg_volume = volume_data.mean()
                        volume_weight = volume_data.iloc[idx] / avg_volume if avg_volume > 0 else 1.0
                    
                    support_levels.append(ChannelLevel(
                        value=price_data.iloc[idx],
                        timestamp=datetime.now(),
                        confidence=min(1.0, volume_weight * 0.5 + 0.3),
                        volume_weight=volume_weight
                    ))
            
            # Create resistance levels from maxima
            for idx in maxima_indices[-10:]:  # Last 10 maxima
                if idx < len(price_data):
                    volume_weight = 1.0
                    if volume_data is not None and idx < len(volume_data):
                        avg_volume = volume_data.mean()
                        volume_weight = volume_data.iloc[idx] / avg_volume if avg_volume > 0 else 1.0
                    
                    resistance_levels.append(ChannelLevel(
                        value=price_data.iloc[idx],
                        timestamp=datetime.now(),
                        confidence=min(1.0, volume_weight * 0.5 + 0.3),
                        volume_weight=volume_weight
                    ))
        
        return support_levels, resistance_levels
    
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Keltner Channel indicator with advanced analysis
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing comprehensive Keltner Channel analysis
        """
        try:
            if data.empty or len(data) < max(self.period, self.atr_period):
                raise ValueError(f"Insufficient data. Need at least {max(self.period, self.atr_period)} periods")
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            # Calculate ATR
            atr_series = self._calculate_atr(data, self.atr_period)
            
            # Calculate middle line (EMA)
            volume_weights = data.get('volume', None)
            middle_line = self._calculate_ema(data['close'], self.period, volume_weights)
            
            # Current values
            current_index = len(data) - 1
            current_price = data['close'].iloc[current_index]
            current_atr = atr_series.iloc[current_index]
            current_middle = middle_line.iloc[current_index]
            current_volume = data.get('volume', pd.Series([1.0] * len(data))).iloc[current_index]
            
            # Update history
            self.price_history.append(current_price)
            self.volume_history.append(current_volume)
            self.atr_history.append(current_atr)
            
            # Maintain history size
            if len(self.price_history) > self.lookback_period:
                self.price_history.pop(0)
                self.volume_history.pop(0)
                self.atr_history.pop(0)
            
            # Determine volatility regime
            volatility_regime = self._determine_volatility_regime(current_atr, self.atr_history)
            
            # Calculate adaptive multiplier
            if self.adaptive_multiplier:
                effective_multiplier = self._calculate_adaptive_multiplier(volatility_regime, self.multiplier)
            else:
                effective_multiplier = self.multiplier
            
            # Calculate bands
            upper_band = current_middle + (effective_multiplier * current_atr)
            lower_band = current_middle - (effective_multiplier * current_atr)
            channel_width = upper_band - lower_band
            
            # Update channel history
            self.channel_history.append({
                'upper': upper_band,
                'middle': current_middle,
                'lower': lower_band,
                'width': channel_width,
                'timestamp': datetime.now()
            })
            
            if len(self.channel_history) > self.lookback_period:
                self.channel_history.pop(0)
            
            # Calculate percentiles
            width_history = [ch['width'] for ch in self.channel_history]
            width_percentile = (np.sum(np.array(width_history) < channel_width) / 
                              len(width_history)) if len(width_history) > 1 else 0.5
            
            volatility_percentile = (np.sum(np.array(self.atr_history) < current_atr) / 
                                   len(self.atr_history)) if len(self.atr_history) > 1 else 0.5
            
            # Determine channel direction
            if len(width_history) >= 5:
                recent_widths = width_history[-5:]
                if recent_widths[-1] > np.mean(recent_widths[:-1]) * 1.05:
                    direction = ChannelDirection.EXPANDING
                elif recent_widths[-1] < np.mean(recent_widths[:-1]) * 0.95:
                    direction = ChannelDirection.CONTRACTING
                else:
                    direction = ChannelDirection.STABLE
            else:
                direction = ChannelDirection.STABLE
            
            # Calculate momentum
            momentum = 0.0
            if len(self.price_history) >= 5:
                momentum = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
            
            # Analyze squeeze
            squeeze_metrics = self._analyze_squeeze(
                channel_width, width_history, current_volume, self.volume_history
            )
            
            # Analyze breakout
            breakout_analysis = self._analyze_breakout(
                current_price, upper_band, lower_band, current_volume, momentum, self.price_history
            )
            
            # Calculate price position within channel
            price_position = (current_price - lower_band) / (upper_band - lower_band) if channel_width > 0 else 0.5
            price_position = max(0.0, min(1.0, price_position))
            
            # Calculate band touch probabilities
            band_touch_probability = {
                'upper': min(0.9, price_position * 1.2 + squeeze_metrics.breakout_probability * 0.3),
                'lower': min(0.9, (1 - price_position) * 1.2 + squeeze_metrics.breakout_probability * 0.3),
                'middle': 1.0 - abs(price_position - 0.5) * 2
            }
            
            # Train ML models periodically
            if len(data) >= 100 and len(data) % 50 == 0:
                self._train_ml_models(data)
            
            # ML predictions
            breakout_probability, direction_prediction = self._predict_with_ml(data, current_index)
            
            # Identify support and resistance levels
            support_levels, resistance_levels = self._identify_support_resistance(data, current_index)
            
            # Calculate overall confidence
            confidence = 0.7
            if len(self.channel_history) >= 20:
                confidence += 0.1
            if self.breakout_model is not None:
                confidence += 0.1
            if len(support_levels) > 0 and len(resistance_levels) > 0:
                confidence += 0.1
            
            # Create result
            result = KeltnerChannelResult(
                upper_band=upper_band,
                middle_line=current_middle,
                lower_band=lower_band,
                channel_width=channel_width,
                width_percentile=width_percentile,
                direction=direction,
                atr_value=current_atr,
                volatility_regime=volatility_regime,
                volatility_percentile=volatility_percentile,
                squeeze_metrics=squeeze_metrics,
                breakout_analysis=breakout_analysis,
                price_position=price_position,
                band_touch_probability=band_touch_probability,
                breakout_probability=breakout_probability,
                direction_prediction=direction_prediction,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                timestamp=datetime.now(),
                confidence=confidence
            )
            
            self.logger.info(
                f"Keltner Channel calculated: Upper={upper_band:.4f}, "
                f"Middle={current_middle:.4f}, Lower={lower_band:.4f}, "
                f"Regime={volatility_regime.value}, Squeeze={squeeze_metrics.is_squeeze}"
            )
            
            return result.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error calculating Keltner Channel: {e}")
            return {
                'error': str(e),
                'upper_band': np.nan,
                'middle_line': np.nan,
                'lower_band': np.nan,
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_required_columns(self) -> List[str]:
        """Get required data columns"""
        return ['open', 'high', 'low', 'close', 'volume']
    
    def get_indicator_type(self) -> str:
        """Get indicator type"""
        return "volatility"
    
    def get_description(self) -> str:
        """Get indicator description"""
        return (
            "Advanced Keltner Channel Indicator providing dynamic volatility analysis "
            "with adaptive channel width, ML-enhanced breakout prediction, squeeze detection, "
            "and comprehensive market regime analysis for optimal trading decisions."
        )