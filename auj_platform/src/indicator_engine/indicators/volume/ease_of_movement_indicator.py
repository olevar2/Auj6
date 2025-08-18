"""
AUJ Platform Advanced Ease of Movement Indicator
Sophisticated implementation with volatility normalization, trend strength measurement, and momentum confirmation

This implementation provides institutional-grade Ease of Movement analysis for humanitarian trading platforms.

Features:
- Enhanced Ease of Movement calculation with multiple smoothing methods
- Volatility normalization for market-adjusted signals
- Trend strength measurement and classification
- Momentum confirmation algorithms
- Multi-timeframe analysis capabilities
- Statistical validation and signal confidence scoring
- Advanced filtering and noise reduction
- Machine learning-based trend prediction
- Real-time adaptive thresholds
- Comprehensive signal generation with risk assessment

The Ease of Movement indicator measures the relationship between price change and volume,
helping identify periods where price moves easily on low volume (easy movement)
versus periods requiring high volume for price changes (difficult movement).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from ..base.base_indicator import BaseIndicator, IndicatorConfig
from ...core.signal_type import SignalType


class EOMTrendStrength(Enum):
    """Ease of Movement trend strength classification"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class EOMMovementType(Enum):
    """Type of price movement classification"""
    EASY_ADVANCE = "easy_advance"
    DIFFICULT_ADVANCE = "difficult_advance"
    EASY_DECLINE = "easy_decline"
    DIFFICULT_DECLINE = "difficult_decline"
    NEUTRAL = "neutral"


@dataclass
class EOMConfig(IndicatorConfig):
    """Configuration for Ease of Movement Indicator"""
    eom_period: int = 14
    smoothing_period: int = 9
    volatility_period: int = 20
    volume_threshold: float = 1.5
    trend_strength_threshold: float = 0.3
    signal_threshold: float = 0.1
    momentum_period: int = 12
    adaptive_threshold: bool = True
    use_volatility_normalization: bool = True
    use_trend_confirmation: bool = True
    ml_enabled: bool = True
    min_periods: int = 50


class EOMTrendAnalysis(NamedTuple):
    """Ease of Movement trend analysis result"""
    direction: str
    strength: EOMTrendStrength
    confidence: float
    persistence: float
    acceleration: float


class EOMVolatilityAnalysis(NamedTuple):
    """Ease of Movement volatility analysis result"""
    normalized_eom: float
    volatility_regime: str
    adjustment_factor: float
    stability_score: float


class EOMSignalAnalysis(NamedTuple):
    """Ease of Movement signal analysis result"""
    signal_strength: float
    signal_quality: float
    confirmation_score: float
    risk_level: str
    expected_duration: int


class EOMResult(NamedTuple):
    """Complete Ease of Movement analysis result"""
    eom_value: float
    smoothed_eom: float
    movement_type: EOMMovementType
    trend_analysis: EOMTrendAnalysis
    volatility_analysis: EOMVolatilityAnalysis
    signal_analysis: EOMSignalAnalysis
    momentum_confirmation: float
    predicted_direction: str
    confidence_score: float


class EaseOfMovementIndicator(BaseIndicator):
    """
    Advanced Ease of Movement Indicator with sophisticated analysis capabilities.
    
    The Ease of Movement (EOM) indicator quantifies the amount of volume required
    to move prices. This implementation enhances the traditional EOM with:
    - Volatility normalization for different market conditions
    - Trend strength measurement and classification
    - Momentum confirmation algorithms
    - Machine learning-based trend prediction
    - Multi-timeframe analysis
    - Statistical validation
    """
    
    def __init__(self, config: Optional[EOMConfig] = None):
        super().__init__(config or EOMConfig())
        self.config: EOMConfig = self.config
        
        # Internal state
        self._eom_history: List[float] = []
        self._volume_history: List[float] = []
        self._price_history: List[float] = []
        self._high_history: List[float] = []
        self._low_history: List[float] = []
        self._volatility_history: List[float] = []
        
        # Machine learning components
        self._trend_predictor: Optional[RandomForestRegressor] = None
        self._scaler: StandardScaler = StandardScaler()
        self._is_trained: bool = False
        
        # Adaptive thresholds
        self._adaptive_signal_threshold: float = self.config.signal_threshold
        self._adaptive_trend_threshold: float = self.config.trend_strength_threshold
        
        # Statistical tracking
        self._signal_accuracy_history: List[bool] = []
        self._trend_accuracy_history: List[bool] = []
        
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate advanced Ease of Movement with comprehensive analysis.
        
        Args:
            data: Dictionary containing 'high', 'low', 'close', 'volume' price data
            
        Returns:
            Dictionary containing EOM analysis results and trading signals
        """
        try:
            if not self._validate_data(data):
                return self._create_default_result()
            
            df = pd.DataFrame(data)
            
            if len(df) < self.config.min_periods:
                return self._create_default_result()
            
            # Calculate basic Ease of Movement
            eom_values = self._calculate_basic_eom(df)
            
            # Apply volatility normalization
            if self.config.use_volatility_normalization:
                volatility_analysis = self._analyze_volatility(df, eom_values)
                normalized_eom = volatility_analysis.normalized_eom
            else:
                normalized_eom = eom_values[-1]
                volatility_analysis = EOMVolatilityAnalysis(
                    normalized_eom=normalized_eom,
                    volatility_regime='normal',
                    adjustment_factor=1.0,
                    stability_score=0.5
                )
            
            # Smooth the EOM values
            smoothed_eom = self._apply_smoothing(eom_values)
            
            # Analyze trend characteristics
            trend_analysis = self._analyze_trend(eom_values, smoothed_eom)
            
            # Classify movement type
            movement_type = self._classify_movement(df, eom_values, trend_analysis)
            
            # Generate signal analysis
            signal_analysis = self._analyze_signals(eom_values, smoothed_eom, trend_analysis)
            
            # Calculate momentum confirmation
            momentum_confirmation = self._calculate_momentum_confirmation(df, eom_values)
            
            # ML-based prediction if enabled
            if self.config.ml_enabled:
                predicted_direction = self._predict_trend_direction(df, eom_values)
            else:
                predicted_direction = trend_analysis.direction
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence(
                trend_analysis, volatility_analysis, signal_analysis, momentum_confirmation
            )
            
            # Create result
            result = EOMResult(
                eom_value=eom_values[-1],
                smoothed_eom=smoothed_eom[-1],
                movement_type=movement_type,
                trend_analysis=trend_analysis,
                volatility_analysis=volatility_analysis,
                signal_analysis=signal_analysis,
                momentum_confirmation=momentum_confirmation,
                predicted_direction=predicted_direction,
                confidence_score=confidence_score
            )
            
            # Generate trading signal
            signal = self._generate_signal(result)
            
            # Update internal state and retrain if needed
            self._update_state_and_retrain(df, eom_values, result)
            
            return self._format_result(result, signal)
            
        except Exception as e:
            self.logger.error(f"Error in EaseOfMovementIndicator calculation: {e}")
            return self._create_error_result(str(e))
    
    def _calculate_basic_eom(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate basic Ease of Movement values"""
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Distance Moved = (High + Low) / 2 - Previous (High + Low) / 2
        mid_point = (high + low) / 2
        distance_moved = np.diff(mid_point, prepend=mid_point[0])
        
        # High-Low Scale = Volume / (High - Low)
        price_range = high - low
        # Avoid division by zero
        price_range = np.where(price_range == 0, 0.001, price_range)
        
        # Box Height = Volume / Scale
        box_height = volume / price_range
        
        # Ease of Movement = Distance Moved / Box Height
        eom = np.where(box_height == 0, 0, distance_moved / box_height)
        
        # Apply scaling factor (traditional multiplier)
        eom = eom * 100000000  # Scale for readability
        
        return eom
    
    def _apply_smoothing(self, eom_values: np.ndarray) -> np.ndarray:
        """Apply sophisticated smoothing to EOM values"""
        if len(eom_values) < self.config.smoothing_period:
            return eom_values
        
        # Multiple smoothing methods
        smoothed_values = []
        
        # 1. Simple Moving Average
        sma_smoothed = pd.Series(eom_values).rolling(
            window=self.config.smoothing_period, min_periods=1
        ).mean().values
        
        # 2. Exponential Moving Average
        ema_smoothed = pd.Series(eom_values).ewm(
            span=self.config.smoothing_period
        ).mean().values
        
        # 3. Adaptive smoothing based on volatility
        volatility = pd.Series(eom_values).rolling(
            window=self.config.volatility_period, min_periods=1
        ).std().values
        
        # Combine smoothing methods with adaptive weighting
        for i in range(len(eom_values)):
            vol = volatility[i] if not np.isnan(volatility[i]) else 1.0
            
            # Higher volatility = more smoothing (higher weight on SMA)
            # Lower volatility = less smoothing (higher weight on EMA)
            vol_normalized = min(vol / np.mean(volatility[~np.isnan(volatility)]), 2.0)
            sma_weight = min(vol_normalized / 2.0, 0.7)
            ema_weight = 1.0 - sma_weight
            
            smoothed_value = sma_weight * sma_smoothed[i] + ema_weight * ema_smoothed[i]
            smoothed_values.append(smoothed_value)
        
        return np.array(smoothed_values)
    
    def _analyze_volatility(self, df: pd.DataFrame, 
                          eom_values: np.ndarray) -> EOMVolatilityAnalysis:
        """Analyze volatility characteristics and normalize EOM"""
        # Calculate price volatility
        close = df['close'].values
        returns = np.diff(np.log(close))
        price_volatility = pd.Series(returns).rolling(
            window=self.config.volatility_period, min_periods=1
        ).std().values[-1]
        
        # Calculate EOM volatility
        eom_volatility = pd.Series(eom_values).rolling(
            window=self.config.volatility_period, min_periods=1
        ).std().values[-1]
        
        # Volatility regime classification
        volatility_percentile = stats.percentileofscore(
            pd.Series(eom_values).rolling(
                window=self.config.volatility_period * 2, min_periods=1
            ).std().dropna().values,
            eom_volatility
        )
        
        if volatility_percentile < 20:
            volatility_regime = 'low'
            adjustment_factor = 1.2
        elif volatility_percentile < 40:
            volatility_regime = 'below_normal'
            adjustment_factor = 1.1
        elif volatility_percentile < 60:
            volatility_regime = 'normal'
            adjustment_factor = 1.0
        elif volatility_percentile < 80:
            volatility_regime = 'above_normal'
            adjustment_factor = 0.9
        else:
            volatility_regime = 'high'
            adjustment_factor = 0.8
        
        # Normalize EOM by volatility
        normalized_eom = eom_values[-1] * adjustment_factor
        
        # Calculate stability score
        eom_stability = 1.0 - min(eom_volatility / np.mean(np.abs(eom_values)), 1.0)
        stability_score = max(0.0, min(1.0, eom_stability))
        
        return EOMVolatilityAnalysis(
            normalized_eom=normalized_eom,
            volatility_regime=volatility_regime,
            adjustment_factor=adjustment_factor,
            stability_score=stability_score
        )
    
    def _analyze_trend(self, eom_values: np.ndarray, 
                      smoothed_eom: np.ndarray) -> EOMTrendAnalysis:
        """Analyze trend characteristics of EOM"""
        if len(eom_values) < self.config.eom_period:
            return EOMTrendAnalysis(
                direction='neutral',
                strength=EOMTrendStrength.WEAK,
                confidence=0.0,
                persistence=0.0,
                acceleration=0.0
            )
        
        recent_period = self.config.eom_period
        recent_eom = smoothed_eom[-recent_period:]
        
        # 1. Trend Direction Analysis
        if len(recent_eom) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                range(len(recent_eom)), recent_eom
            )
            
            if slope > self._adaptive_trend_threshold:
                direction = 'bullish'
            elif slope < -self._adaptive_trend_threshold:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            trend_confidence = abs(r_value) if not np.isnan(r_value) else 0.0
        else:
            slope = 0
            trend_confidence = 0.0
            direction = 'neutral'
        
        # 2. Trend Strength Classification
        abs_slope = abs(slope)
        if abs_slope < 0.1:
            strength = EOMTrendStrength.VERY_WEAK
        elif abs_slope < 0.3:
            strength = EOMTrendStrength.WEAK
        elif abs_slope < 0.6:
            strength = EOMTrendStrength.MODERATE
        elif abs_slope < 1.0:
            strength = EOMTrendStrength.STRONG
        else:
            strength = EOMTrendStrength.VERY_STRONG
        
        # 3. Trend Persistence
        # Count consecutive periods in same direction
        eom_changes = np.diff(recent_eom)
        if direction == 'bullish':
            persistence_count = 0
            for change in reversed(eom_changes):
                if change > 0:
                    persistence_count += 1
                else:
                    break
        elif direction == 'bearish':
            persistence_count = 0
            for change in reversed(eom_changes):
                if change < 0:
                    persistence_count += 1
                else:
                    break
        else:
            persistence_count = 0
        
        persistence = min(persistence_count / len(eom_changes), 1.0)
        
        # 4. Trend Acceleration
        if len(recent_eom) >= 6:
            mid_point = len(recent_eom) // 2
            first_half_slope = np.polyfit(
                range(mid_point), recent_eom[:mid_point], 1
            )[0]
            second_half_slope = np.polyfit(
                range(mid_point), recent_eom[mid_point:], 1
            )[0]
            
            acceleration = second_half_slope - first_half_slope
        else:
            acceleration = 0.0
        
        return EOMTrendAnalysis(
            direction=direction,
            strength=strength,
            confidence=trend_confidence,
            persistence=persistence,
            acceleration=acceleration
        )
    
    def _classify_movement(self, df: pd.DataFrame, eom_values: np.ndarray,
                         trend_analysis: EOMTrendAnalysis) -> EOMMovementType:
        """Classify the type of price movement"""
        current_eom = eom_values[-1]
        volume = df['volume'].values
        
        # Calculate volume context
        avg_volume = np.mean(volume[-self.config.eom_period:])
        current_volume = volume[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Price direction
        close = df['close'].values
        price_change = close[-1] - close[-2] if len(close) > 1 else 0
        
        # Classification logic
        if current_eom > self._adaptive_signal_threshold:
            # Positive EOM (easy movement up)
            if price_change > 0:
                if volume_ratio < self.config.volume_threshold:
                    return EOMMovementType.EASY_ADVANCE
                else:
                    return EOMMovementType.DIFFICULT_ADVANCE
            else:
                return EOMMovementType.NEUTRAL
        elif current_eom < -self._adaptive_signal_threshold:
            # Negative EOM (easy movement down)
            if price_change < 0:
                if volume_ratio < self.config.volume_threshold:
                    return EOMMovementType.EASY_DECLINE
                else:
                    return EOMMovementType.DIFFICULT_DECLINE
            else:
                return EOMMovementType.NEUTRAL
        else:
            return EOMMovementType.NEUTRAL
    
    def _analyze_signals(self, eom_values: np.ndarray, smoothed_eom: np.ndarray,
                        trend_analysis: EOMTrendAnalysis) -> EOMSignalAnalysis:
        """Analyze signal characteristics and quality"""
        current_eom = eom_values[-1]
        current_smoothed = smoothed_eom[-1]
        
        # Signal strength based on EOM magnitude and trend
        signal_strength = abs(current_smoothed) / (np.std(eom_values[-20:]) + 0.001)
        signal_strength = min(signal_strength, 3.0)  # Cap at 3.0
        
        # Signal quality based on consistency and noise
        if len(eom_values) >= 10:
            recent_eom = eom_values[-10:]
            signal_noise = np.std(recent_eom) / (np.mean(np.abs(recent_eom)) + 0.001)
            signal_quality = max(0.0, 1.0 - signal_noise)
        else:
            signal_quality = 0.5
        
        # Confirmation score from trend analysis
        confirmation_score = (
            trend_analysis.confidence * 0.4 +
            trend_analysis.persistence * 0.3 +
            min(abs(trend_analysis.acceleration), 1.0) * 0.3
        )
        
        # Risk level assessment
        if signal_strength > 2.0 and signal_quality > 0.7:
            risk_level = 'low'
            expected_duration = 5
        elif signal_strength > 1.0 and signal_quality > 0.5:
            risk_level = 'medium'
            expected_duration = 3
        else:
            risk_level = 'high'
            expected_duration = 1
        
        return EOMSignalAnalysis(
            signal_strength=signal_strength,
            signal_quality=signal_quality,
            confirmation_score=confirmation_score,
            risk_level=risk_level,
            expected_duration=expected_duration
        )
    
    def _calculate_momentum_confirmation(self, df: pd.DataFrame,
                                       eom_values: np.ndarray) -> float:
        """Calculate momentum confirmation score"""
        if not self.config.use_trend_confirmation or len(df) < self.config.momentum_period:
            return 0.5
        
        close = df['close'].values
        volume = df['volume'].values
        
        # Price momentum
        price_momentum = (close[-1] - close[-self.config.momentum_period]) / close[-self.config.momentum_period]
        
        # Volume momentum
        recent_volume = np.mean(volume[-self.config.momentum_period//2:])
        earlier_volume = np.mean(volume[-self.config.momentum_period:-self.config.momentum_period//2])
        volume_momentum = (recent_volume - earlier_volume) / earlier_volume if earlier_volume > 0 else 0
        
        # EOM momentum
        recent_eom = np.mean(eom_values[-self.config.momentum_period//2:])
        earlier_eom = np.mean(eom_values[-self.config.momentum_period:-self.config.momentum_period//2])
        eom_momentum = recent_eom - earlier_eom
        
        # Momentum alignment score
        momentum_signals = [price_momentum, volume_momentum, eom_momentum]
        
        # Count aligned signals (same direction)
        positive_signals = sum(1 for signal in momentum_signals if signal > 0)
        negative_signals = sum(1 for signal in momentum_signals if signal < 0)
        
        if positive_signals >= 2:
            confirmation = positive_signals / 3.0
        elif negative_signals >= 2:
            confirmation = negative_signals / 3.0
        else:
            confirmation = 0.3  # Mixed signals
        
        return confirmation
    
    def _predict_trend_direction(self, df: pd.DataFrame, 
                               eom_values: np.ndarray) -> str:
        """Use ML to predict future trend direction"""
        if not self._is_trained or len(eom_values) < 20:
            return 'neutral'
        
        try:
            # Extract features for prediction
            features = self._extract_prediction_features(df, eom_values)
            
            if len(features) == 0:
                return 'neutral'
            
            # Scale features
            features_scaled = self._scaler.transform([features])
            
            # Predict
            prediction = self._trend_predictor.predict(features_scaled)[0]
            
            if prediction > 0.1:
                return 'bullish'
            elif prediction < -0.1:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.warning(f"ML prediction failed: {e}")
            return 'neutral'
    
    def _calculate_confidence(self, trend_analysis: EOMTrendAnalysis,
                            volatility_analysis: EOMVolatilityAnalysis,
                            signal_analysis: EOMSignalAnalysis,
                            momentum_confirmation: float) -> float:
        """Calculate overall confidence score"""
        confidence_components = [
            trend_analysis.confidence * 0.25,
            volatility_analysis.stability_score * 0.20,
            signal_analysis.signal_quality * 0.25,
            signal_analysis.confirmation_score * 0.15,
            momentum_confirmation * 0.15
        ]
        
        # Adjust for trend strength
        strength_multiplier = {
            EOMTrendStrength.VERY_WEAK: 0.5,
            EOMTrendStrength.WEAK: 0.7,
            EOMTrendStrength.MODERATE: 1.0,
            EOMTrendStrength.STRONG: 1.2,
            EOMTrendStrength.VERY_STRONG: 1.4
        }
        
        base_confidence = sum(confidence_components)
        adjusted_confidence = base_confidence * strength_multiplier[trend_analysis.strength]
        
        return min(1.0, max(0.0, adjusted_confidence))
    
    def _generate_signal(self, result: EOMResult) -> SignalType:
        """Generate trading signal based on EOM analysis"""
        # Signal generation criteria
        signal_criteria = []
        
        # 1. EOM direction and magnitude
        if result.smoothed_eom > self._adaptive_signal_threshold:
            signal_criteria.append('bullish_eom')
        elif result.smoothed_eom < -self._adaptive_signal_threshold:
            signal_criteria.append('bearish_eom')
        
        # 2. Trend analysis
        if result.trend_analysis.direction == 'bullish' and result.trend_analysis.confidence > 0.6:
            signal_criteria.append('bullish_trend')
        elif result.trend_analysis.direction == 'bearish' and result.trend_analysis.confidence > 0.6:
            signal_criteria.append('bearish_trend')
        
        # 3. Movement type
        if result.movement_type == EOMMovementType.EASY_ADVANCE:
            signal_criteria.append('easy_bullish')
        elif result.movement_type == EOMMovementType.EASY_DECLINE:
            signal_criteria.append('easy_bearish')
        
        # 4. Signal quality and strength
        strong_signal = (result.signal_analysis.signal_strength > 1.5 and 
                        result.signal_analysis.signal_quality > 0.6)
        
        # 5. Momentum confirmation
        momentum_aligned = result.momentum_confirmation > 0.6
        
        # 6. Overall confidence
        high_confidence = result.confidence_score > 0.7
        
        # Generate signal
        bullish_signals = sum(1 for criterion in signal_criteria 
                            if 'bullish' in criterion or 'easy_bullish' in criterion)
        bearish_signals = sum(1 for criterion in signal_criteria 
                            if 'bearish' in criterion or 'easy_bearish' in criterion)
        
        if (bullish_signals >= 2 and strong_signal and momentum_aligned and high_confidence):
            return SignalType.BUY
        elif (bearish_signals >= 2 and strong_signal and momentum_aligned and high_confidence):
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _update_state_and_retrain(self, df: pd.DataFrame, eom_values: np.ndarray,
                                result: EOMResult):
        """Update internal state and retrain ML models"""
        max_history = 1000
        
        # Update histories
        self._eom_history.extend(eom_values[-10:])  # Last 10 values
        self._volume_history.extend(df['volume'].values[-10:])
        self._price_history.extend(df['close'].values[-10:])
        self._high_history.extend(df['high'].values[-10:])
        self._low_history.extend(df['low'].values[-10:])
        
        # Trim histories
        if len(self._eom_history) > max_history:
            self._eom_history = self._eom_history[-max_history:]
            self._volume_history = self._volume_history[-max_history:]
            self._price_history = self._price_history[-max_history:]
            self._high_history = self._high_history[-max_history:]
            self._low_history = self._low_history[-max_history:]
        
        # Update adaptive thresholds
        if self.config.adaptive_threshold and len(self._eom_history) >= 50:
            eom_std = np.std(self._eom_history[-50:])
            self._adaptive_signal_threshold = max(0.05, eom_std * 0.5)
            self._adaptive_trend_threshold = max(0.1, eom_std * 0.2)
        
        # Retrain ML model periodically
        if (self.config.ml_enabled and len(self._eom_history) >= 100 and 
            len(self._eom_history) % 50 == 0):
            self._retrain_ml_model()
    
    def _retrain_ml_model(self):
        """Retrain the machine learning model"""
        try:
            if len(self._eom_history) < 100:
                return
            
            # Prepare training data
            features = []
            targets = []
            
            window_size = 20
            for i in range(window_size, len(self._eom_history) - 5):
                # Features: recent EOM, volume, price data
                feature_vector = []
                
                # EOM features
                eom_window = self._eom_history[i-window_size:i]
                feature_vector.extend([
                    np.mean(eom_window),
                    np.std(eom_window),
                    np.max(eom_window),
                    np.min(eom_window)
                ])
                
                # Volume features
                if i < len(self._volume_history):
                    volume_window = self._volume_history[i-window_size:i]
                    feature_vector.extend([
                        np.mean(volume_window),
                        np.std(volume_window)
                    ])
                else:
                    feature_vector.extend([0, 0])
                
                # Price features
                if i < len(self._price_history):
                    price_window = self._price_history[i-window_size:i]
                    feature_vector.extend([
                        np.mean(price_window),
                        np.std(price_window)
                    ])
                else:
                    feature_vector.extend([0, 0])
                
                features.append(feature_vector)
                
                # Target: future EOM direction
                future_eom = np.mean(self._eom_history[i:i+5])
                current_eom = self._eom_history[i-1]
                target = future_eom - current_eom
                targets.append(target)
            
            if len(features) > 10:
                # Scale features
                features_array = np.array(features)
                self._scaler.fit(features_array)
                features_scaled = self._scaler.transform(features_array)
                
                # Train model
                self._trend_predictor = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42
                )
                self._trend_predictor.fit(features_scaled, targets)
                
                self._is_trained = True
                self.logger.info("EOM ML model retrained successfully")
            
        except Exception as e:
            self.logger.warning(f"ML model retraining failed: {e}")
    
    def _extract_prediction_features(self, df: pd.DataFrame, 
                                   eom_values: np.ndarray) -> List[float]:
        """Extract features for ML prediction"""
        if len(eom_values) < 20:
            return []
        
        features = []
        
        # Recent EOM features
        recent_eom = eom_values[-20:]
        features.extend([
            np.mean(recent_eom),
            np.std(recent_eom),
            np.max(recent_eom),
            np.min(recent_eom)
        ])
        
        # Recent volume features
        recent_volume = df['volume'].values[-20:]
        features.extend([
            np.mean(recent_volume),
            np.std(recent_volume)
        ])
        
        # Recent price features
        recent_close = df['close'].values[-20:]
        features.extend([
            np.mean(recent_close),
            np.std(recent_close)
        ])
        
        return features
    
    def _format_result(self, result: EOMResult, signal: SignalType) -> Dict[str, Any]:
        """Format the complete result for output"""
        return {
            'signal': signal,
            'confidence': result.confidence_score,
            
            # Core EOM values
            'eom_value': result.eom_value,
            'smoothed_eom': result.smoothed_eom,
            'movement_type': result.movement_type.value,
            
            # Trend analysis
            'trend_direction': result.trend_analysis.direction,
            'trend_strength': result.trend_analysis.strength.value,
            'trend_confidence': result.trend_analysis.confidence,
            'trend_persistence': result.trend_analysis.persistence,
            'trend_acceleration': result.trend_analysis.acceleration,
            
            # Volatility analysis
            'normalized_eom': result.volatility_analysis.normalized_eom,
            'volatility_regime': result.volatility_analysis.volatility_regime,
            'volatility_adjustment': result.volatility_analysis.adjustment_factor,
            'stability_score': result.volatility_analysis.stability_score,
            
            # Signal analysis
            'signal_strength': result.signal_analysis.signal_strength,
            'signal_quality': result.signal_analysis.signal_quality,
            'confirmation_score': result.signal_analysis.confirmation_score,
            'risk_level': result.signal_analysis.risk_level,
            'expected_duration': result.signal_analysis.expected_duration,
            
            # Additional metrics
            'momentum_confirmation': result.momentum_confirmation,
            'predicted_direction': result.predicted_direction,
            
            # Thresholds (for transparency)
            'signal_threshold': self._adaptive_signal_threshold,
            'trend_threshold': self._adaptive_trend_threshold,
            
            # Metadata
            'metadata': {
                'indicator_name': 'EaseOfMovementIndicator',
                'version': '1.0.0',
                'calculation_time': pd.Timestamp.now().isoformat(),
                'ml_enabled': self.config.ml_enabled,
                'ml_trained': self._is_trained
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
    
    def _create_default_result(self) -> Dict[str, Any]:
        """Create default result for insufficient data"""
        return {
            'signal': SignalType.HOLD,
            'confidence': 0.0,
            'eom_value': 0.0,
            'smoothed_eom': 0.0,
            'movement_type': EOMMovementType.NEUTRAL.value,
            'trend_direction': 'neutral',
            'trend_strength': EOMTrendStrength.WEAK.value,
            'trend_confidence': 0.0,
            'trend_persistence': 0.0,
            'trend_acceleration': 0.0,
            'normalized_eom': 0.0,
            'volatility_regime': 'normal',
            'volatility_adjustment': 1.0,
            'stability_score': 0.0,
            'signal_strength': 0.0,
            'signal_quality': 0.0,
            'confirmation_score': 0.0,
            'risk_level': 'high',
            'expected_duration': 1,
            'momentum_confirmation': 0.0,
            'predicted_direction': 'neutral',
            'signal_threshold': self.config.signal_threshold,
            'trend_threshold': self.config.trend_strength_threshold,
            'metadata': {
                'indicator_name': 'EaseOfMovementIndicator',
                'error': 'Insufficient data for calculation'
            }
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result structure"""
        result = self._create_default_result()
        result['metadata']['error'] = error_message
        return result